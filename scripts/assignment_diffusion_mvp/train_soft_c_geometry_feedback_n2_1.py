#!/usr/bin/env python3
"""Train N2.1 causal edge modulator (frozen GemNet + frozen N1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
    PROVENANCE,
)
from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS
from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss
from mattergen.assignment.soft_c_geometry_feedback_n2_1.preflight import run_n21_preflight
from mattergen.assignment.soft_c_geometry_feedback_n2_1.setup_utils import (
    build_chemgraph_batch,
    load_fixed_sample_bundle,
    load_n21_stack,
)


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    p.add_argument("--n1-checkpoint", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["assignment_n21"]
    device = resolve_device(str(cfg.get("device", "auto")))
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    n1_ckpt = args.n1_checkpoint or cfg.get("n1_checkpoint")
    if not n1_ckpt or not Path(n1_ckpt).exists():
        raise FileNotFoundError(f"N1 checkpoint missing: {n1_ckpt}")

    model, pl_module, load_prov = load_n21_stack(
        cfg=cfg,
        device=device,
        n1_ckpt_path=n1_ckpt,
        mattergen_model_path=args.mattergen_model_path,
        mattergen_load_epoch=args.mattergen_load_epoch,
        mattergen_checkpoint=args.mattergen_checkpoint,
    )
    sample, _partition, backbone, o2_target, oracle_bar = load_fixed_sample_bundle(cfg, device)
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    loss_fn = pl_module.diffusion_module.loss_fn
    T = float(noise.T)

    audit = model.param_audit()
    print(json.dumps({"event": "n21_param_audit", **audit}), flush=True)
    if audit["trainable_base_gemnet_params"] != 0 or audit["trainable_n1_params"] != 0:
        raise RuntimeError("base GemNet and N1 must be frozen")
    if audit["trainable_n21_params"] <= 0:
        raise RuntimeError("N2.1 modulator must be trainable")

    n1_h0 = model.n1_parameter_hash()
    gem_h0 = model.gemnet_parameter_hash()
    n21_h0 = model.n21_parameter_hash()

    runtime = {
        "N21_MODE": "causal_edge",
        "FORMULA": "delta_e = g * q * s * F_psi(e)",
        "F_PSI_INPUTS": "edge_emb_only",
        "GROUP_CONTEXT": False,
        "MATTERGEN_EPOCH": load_prov.get("mattergen_load_epoch"),
        "MATTERGEN_CHECKPOINT_SHA": load_prov.get("mattergen_checkpoint_sha256"),
        "N1_CHECKPOINT_SHA": load_prov.get("n1_checkpoint_sha256"),
        "SOFT_C_SEMANTICS": SOFT_C_SEMANTICS,
        "NOISE_GATE_FULL_ON": model.config.noise_gate.full_on_t_fraction,
        "NOISE_GATE_FULL_OFF": model.config.noise_gate.full_off_t_fraction,
        "ASSIGNMENT_FROZEN": True,
        "BASE_GEMNET_FROZEN": True,
        "G_DIFFUSION": False,
        "NOISE_SOURCE": PROVENANCE["noise_source"],
        **load_prov,
        "param_audit": audit,
    }
    print(json.dumps({"event": "n21_runtime", **{k: runtime[k] for k in (
        "FORMULA", "GROUP_CONTEXT", "NOISE_GATE_FULL_ON", "NOISE_GATE_FULL_OFF"
    )}}), flush=True)
    (out / "runtime_provenance.json").write_text(json.dumps(runtime, indent=2, default=str))

    pre = run_n21_preflight(
        model=model,
        sample=sample,
        backbone_tree=backbone,
        o2_target=o2_target,
        oracle_bar=oracle_bar,
        noise_adapter=noise,
        seed=int(cfg.get("seed", 17)) + 999,
    )
    (out / "equality_preflight.json").write_text(json.dumps(pre, indent=2))
    print(json.dumps({"event": "n21_preflight", **pre}), flush=True)
    if not pre.get("ok"):
        raise RuntimeError(f"N2.1 preflight failed: {pre.get('failure')}")

    params = list(model.trainable_n21_parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    steps = int(args.steps or cfg.get("steps", 1000))
    clip = float(cfg.get("gradient_clip_norm", 1.0))
    log_every = int(cfg.get("log_every_steps", 20))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)))
    best = float("inf")

    def _grad_norm(mod: torch.nn.Module) -> float:
        s = 0.0
        any_g = False
        for p in mod.parameters():
            if p.grad is not None:
                any_g = True
                s += float(p.grad.detach().float().pow(2).sum())
        return float(s ** 0.5) if any_g else 0.0

    with (out / "training_trace.jsonl").open("w", buffering=1) as stream:
        for step in range(steps):
            t = noise.sample_t(1, device=device)
            noisy = noise.corrupt_fixed_sample(
                frac_coords_0=sample["pos"],
                lattice_0=sample["cell"],
                num_atoms=int(sample["N"]),
                t=t,
                generator=g,
            )
            t_frac = float(noisy.t.detach()) / T
            soft_c, meta = model.pass_a_soft_c(
                sample=sample,
                frac_t=noisy.frac_coords_t,
                cell_t=noisy.lattice_t,
                t=noisy.t,
                o2_target=o2_target,
                backbone_tree=backbone,
                oracle_bar=oracle_bar,
                mode="B2_correct_c",
            )
            clean_cg = build_chemgraph_batch(sample, sample["pos"], sample["cell"])
            noisy_cg = build_chemgraph_batch(sample, noisy.frac_coords_t, noisy.lattice_t)
            opt.zero_grad(set_to_none=True)
            score_out = model.forward_geometry(
                chemgraph=noisy_cg,
                t=noisy.t.reshape(-1),
                soft_c=soft_c,
                t_fraction=t_frac,
                mode="B2_correct_c",
                soft_c_source=str(meta.get("soft_c_source", "")),
            )
            loss, metrics = mattergen_geometry_loss(
                loss_fn=loss_fn,
                corruption=noise.corruption,
                clean_batch=clean_cg,
                noisy_batch=noisy_cg,
                score_model_output=score_out,
                t=noisy.t.reshape(-1),
            )
            loss.backward()
            for p in model.n1.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) != 0:
                    raise RuntimeError("FROZEN_N1_VIOLATION")
            for p in model.denoiser.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) != 0:
                    raise RuntimeError("FROZEN_BASE_GEMNET_VIOLATION")
            edge_gn = _grad_norm(model.edge_modulator)
            g_noise = model._last_diag.g_noise if model._last_diag else None
            if g_noise is not None and g_noise > 0 and soft_c is not None and edge_gn <= 0:
                # At p≈0.5 for all pairs coeff can be ~0 and grads vanish; only fail if mean |C-0.5| high
                # Minimal: require grad when g_noise>0.5 (low-noise region)
                if g_noise >= 0.5:
                    raise RuntimeError(
                        f"edge_modulator_grad_norm=0 at step={step} g_noise={g_noise}"
                    )
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            row = {
                "step": step,
                "t": float(noisy.t.detach()),
                "t_fraction": t_frac,
                "sigma_x": float(noisy.sigma_x),
                "sigma_l": float(noisy.sigma_l),
                "log_snr_x": float(noisy.log_snr_x),
                "geometry_loss": float(loss.detach()),
                "gradient_norm": float(gn),
                "edge_adapter_grad_norm": edge_gn,
                "g_noise": g_noise,
                **{f"loss_{k}": v for k, v in metrics.items()},
            }
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == steps:
                print(json.dumps(row), flush=True)
            if float(loss.detach()) < best:
                best = float(loss.detach())
                torch.save(
                    {
                        "edge_modulator": model.edge_modulator.state_dict(),
                        "config": runtime,
                        "step": step,
                        "geometry_loss": best,
                    },
                    out / "best_adapter_checkpoint.pt",
                )

    n1_h1 = model.n1_parameter_hash()
    gem_h1 = model.gemnet_parameter_hash()
    n21_h1 = model.n21_parameter_hash()
    if n1_h0 != n1_h1 or gem_h0 != gem_h1:
        raise RuntimeError("frozen hash changed")
    if n21_h0 == n21_h1:
        raise RuntimeError("N2.1 adapter not updated")
    ha = {
        "n1_unchanged": True,
        "gemnet_unchanged": True,
        "n21_updated": True,
        "n1_parameter_hash_before": n1_h0,
        "n1_parameter_hash_after": n1_h1,
        "gemnet_parameter_hash_before": gem_h0,
        "gemnet_parameter_hash_after": gem_h1,
        "n21_parameter_hash_before": n21_h0,
        "n21_parameter_hash_after": n21_h1,
    }
    (out / "parameter_hash_audit.json").write_text(json.dumps(ha, indent=2))
    torch.save(
        {
            "edge_modulator": model.edge_modulator.state_dict(),
            "config": runtime,
            "steps": steps,
            "parameter_hash_audit": ha,
        },
        out / "final_adapter_checkpoint.pt",
    )
    print(json.dumps({"event": "n21_training_done", "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
