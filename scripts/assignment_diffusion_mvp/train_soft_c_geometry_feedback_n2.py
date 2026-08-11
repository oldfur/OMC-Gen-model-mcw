#!/usr/bin/env python3
"""Train N2 soft-C geometry feedback adapters (frozen GemNet + frozen N1)."""
from __future__ import annotations

import argparse
import hashlib
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
from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss
from mattergen.assignment.soft_c_geometry_feedback_n2.module import SoftCGeometryFeedbackN2
from mattergen.assignment.soft_c_geometry_feedback_n2.preflight import run_equality_preflight
from mattergen.assignment.soft_c_geometry_feedback_n2.setup_utils import (
    build_chemgraph_batch,
    load_fixed_sample_bundle,
    load_n2_stack,
)
from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS


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
    p.add_argument("--n2-mode", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to train N2 without --execute")

    cfg = yaml.safe_load(args.config.read_text())["assignment_n2"]
    if args.n2_mode:
        cfg["n2_mode"] = args.n2_mode
    device = resolve_device(str(cfg.get("device", "auto")))
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    n1_ckpt = args.n1_checkpoint or cfg.get("n1_checkpoint")
    if not n1_ckpt or not Path(n1_ckpt).exists():
        raise FileNotFoundError(f"N1 checkpoint missing: {n1_ckpt}")

    model, pl_module, load_prov = load_n2_stack(
        cfg=cfg,
        device=device,
        n1_ckpt_path=n1_ckpt,
        mattergen_model_path=args.mattergen_model_path,
        mattergen_load_epoch=args.mattergen_load_epoch,
        mattergen_checkpoint=args.mattergen_checkpoint,
    )
    sample, partition, backbone, o2_target, oracle_bar = load_fixed_sample_bundle(cfg, device)
    del partition

    dm = pl_module.diffusion_module
    loss_fn = dm.loss_fn
    # Geometry-only native pos/cell corruption (same classes as CSP training).
    # Do not pass full dm.corruption (may include atom-type D3PM fields).
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))

    audit = model.param_audit()
    print(json.dumps({"event": "n2_param_audit", **audit}), flush=True)
    if audit["trainable_base_gemnet_params"] != 0:
        raise RuntimeError("base GemNet must be frozen")
    if audit["trainable_n1_params"] != 0:
        raise RuntimeError("N1 must be frozen")
    if audit["trainable_n2_params"] <= 0:
        raise RuntimeError("N2 adapters must be trainable")

    n1_hash_before = model.n1_parameter_hash()
    gem_hash_before = model.gemnet_parameter_hash()
    n2_hash_before = model.n2_parameter_hash()

    params = list(model.trainable_n2_parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    steps = int(args.steps or cfg.get("steps", 1000))
    clip = float(cfg.get("gradient_clip_norm", 1.0))
    log_every = int(cfg.get("log_every_steps", 20))
    mode = str(cfg.get("n2_mode", "B2_combined"))
    model.config.mode = mode  # type: ignore[assignment]
    T = float(noise.T)

    runtime = {
        "N2_MODE": mode,
        "MATTERGEN_EPOCH": load_prov.get("mattergen_load_epoch"),
        "MATTERGEN_CHECKPOINT_SHA": load_prov.get("mattergen_checkpoint_sha256"),
        "N1_CHECKPOINT_SHA": load_prov.get("n1_checkpoint_sha256"),
        "ASSIGNMENT_TRAJECTORY": "geometry_induced_inference",
        "G_DIFFUSION": False,
        "SOFT_C_SEMANTICS": SOFT_C_SEMANTICS,
        "EDGE_SEMANTICS": model.config.edge_semantics_enabled,
        "GROUP_CONTEXT": model.config.group_context_enabled,
        "PAIRWISE_CONFIDENCE": model.config.pairwise_confidence_enabled,
        "NOISE_GATE": model.config.noise_gate.enabled,
        "NOISE_GATE_FULL_ON": model.config.noise_gate.full_on_t_fraction,
        "NOISE_GATE_FULL_OFF": model.config.noise_gate.full_off_t_fraction,
        "ASSIGNMENT_FROZEN": True,
        "BASE_GEMNET_FROZEN": True,
        "GEOMETRY_OBJECTIVE": "mattergen_native",
        "NOISE_SOURCE": PROVENANCE["noise_source"],
        **{k: load_prov[k] for k in load_prov},
        "param_audit": audit,
        "n1_parameter_hash_before": n1_hash_before,
        "gemnet_parameter_hash_before": gem_hash_before,
        "n2_parameter_hash_before": n2_hash_before,
    }
    for k in (
        "N2_MODE",
        "MATTERGEN_EPOCH",
        "SOFT_C_SEMANTICS",
        "ASSIGNMENT_FROZEN",
        "BASE_GEMNET_FROZEN",
        "GEOMETRY_OBJECTIVE",
    ):
        print(f"{k}={runtime[k]}", flush=True)
    (out / "runtime_provenance.json").write_text(json.dumps(runtime, indent=2, default=str))

    # --- Minimal baseline-equivalence preflight (fail loud before training) ---
    pre = run_equality_preflight(
        model=model,
        sample=sample,
        backbone_tree=backbone,
        o2_target=o2_target,
        oracle_bar=oracle_bar,
        noise_adapter=noise,
        seed=int(cfg.get("seed", 17)) + 999,
    )
    (out / "equality_preflight.json").write_text(json.dumps(pre, indent=2))
    print(json.dumps({"event": "n2_equality_preflight", **pre}), flush=True)
    if not pre.get("ok"):
        raise RuntimeError(f"N2 equality preflight failed: {pre.get('failure')}")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)))
    best_loss = float("inf")

    def _module_grad_norm(module: torch.nn.Module) -> float:
        total = 0.0
        found = False
        for p in module.parameters():
            if p.grad is not None:
                found = True
                total += float(p.grad.detach().float().pow(2).sum().item())
        return float(total ** 0.5) if found else 0.0

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
            # Pass A
            soft_c, h_a, a_meta = model.pass_a_assignment(
                sample=sample,
                frac_t=noisy.frac_coords_t,
                cell_t=noisy.lattice_t,
                t=noisy.t,
                o2_target=o2_target,
                backbone_tree=backbone,
                oracle_bar=oracle_bar,
                orbit_mode="predicted_orbit",
                mode=mode,  # type: ignore[arg-type]
            )
            # Build ChemGraphs for native loss
            clean_cg = build_chemgraph_batch(sample, sample["pos"], sample["cell"])
            noisy_cg = build_chemgraph_batch(sample, noisy.frac_coords_t, noisy.lattice_t)
            opt.zero_grad(set_to_none=True)
            score_out = model.forward_geometry(
                chemgraph=noisy_cg,
                t=noisy.t.reshape(-1),
                soft_c=soft_c,
                h_a=h_a,
                t_fraction=t_frac,
                mode=mode,  # type: ignore[arg-type]
            )
            # Native pos/cell MultiCorruption + MaterialsLoss from the loaded module.
            loss, metrics = mattergen_geometry_loss(
                loss_fn=loss_fn,
                corruption=noise.corruption,
                clean_batch=clean_cg,
                noisy_batch=noisy_cg,
                score_model_output=score_out,
                t=noisy.t.reshape(-1),
            )
            loss.backward()
            # N1 / base grads must be absent or zero
            for p in model.n1.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) != 0:
                    raise RuntimeError("FROZEN_N1_VIOLATION: N1 received gradients")
            for p in model.denoiser.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) != 0:
                    raise RuntimeError("FROZEN_BASE_GEMNET_VIOLATION: base GemNet received gradients")
            edge_gn = _module_grad_norm(model.edge_adapter)
            group_gn = _module_grad_norm(model.group_adapter)
            g_noise = float(model._last_diag.g_noise) if model._last_diag else None
            # When feedback is active (g_noise>0), at least the enabled adapter path
            # should receive geometry-loss gradient (log always; fail if none).
            if g_noise is not None and g_noise > 0.0 and soft_c is not None:
                if mode == "B2_combined" and edge_gn <= 0.0 and group_gn <= 0.0:
                    raise RuntimeError(
                        f"both adapter grad norms are 0 at step={step} with g_noise={g_noise}>0"
                    )
                if mode == "B3_edge_only" and edge_gn <= 0.0:
                    raise RuntimeError(
                        f"edge_adapter_grad_norm=0 at step={step} with g_noise={g_noise}>0"
                    )
                # B4 group-only may legitimately have ~0 grad when group mass is 0;
                # still log group_adapter_grad_norm every step.
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
                "group_adapter_grad_norm": group_gn,
                "n2_mode": mode,
                "g_noise": g_noise,
                **{f"loss_{k}": v for k, v in metrics.items()},
            }
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == steps:
                print(json.dumps(row), flush=True)
            if float(loss.detach()) < best_loss:
                best_loss = float(loss.detach())
                torch.save(
                    {
                        "edge_adapter": model.edge_adapter.state_dict(),
                        "group_adapter": model.group_adapter.state_dict(),
                        "config": runtime,
                        "step": step,
                        "geometry_loss": best_loss,
                    },
                    out / "best_adapter_checkpoint.pt",
                )

    n1_hash_after = model.n1_parameter_hash()
    gem_hash_after = model.gemnet_parameter_hash()
    n2_hash_after = model.n2_parameter_hash()
    if n1_hash_before != n1_hash_after:
        raise RuntimeError("FROZEN_N1_VIOLATION: N1 parameter hash changed")
    if gem_hash_before != gem_hash_after:
        raise RuntimeError("FROZEN_BASE_GEMNET_VIOLATION: GemNet parameter hash changed")
    if n2_hash_before == n2_hash_after:
        raise RuntimeError("N2_ADAPTER_NOT_UPDATED: adapter hash unchanged")

    hash_audit = {
        "n1_parameter_hash_before": n1_hash_before,
        "n1_parameter_hash_after": n1_hash_after,
        "gemnet_parameter_hash_before": gem_hash_before,
        "gemnet_parameter_hash_after": gem_hash_after,
        "n2_parameter_hash_before": n2_hash_before,
        "n2_parameter_hash_after": n2_hash_after,
        "n1_unchanged": True,
        "gemnet_unchanged": True,
        "n2_updated": True,
    }
    (out / "parameter_hash_audit.json").write_text(json.dumps(hash_audit, indent=2))
    torch.save(
        {
            "edge_adapter": model.edge_adapter.state_dict(),
            "group_adapter": model.group_adapter.state_dict(),
            "config": runtime,
            "steps": steps,
            "parameter_hash_audit": hash_audit,
        },
        out / "final_adapter_checkpoint.pt",
    )
    print(json.dumps({"event": "n2_training_done", "output": str(out), **hash_audit}), flush=True)


if __name__ == "__main__":
    main()
