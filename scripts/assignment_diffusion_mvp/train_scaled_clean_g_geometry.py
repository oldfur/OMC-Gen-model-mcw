#!/usr/bin/env python3
"""Train Original vs gated Clean-G on the scaled OMC25 assignment subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.joint_assignment_diffusion.geometry_ablation import resolve_geometry_ablation_arm
from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
from mattergen.assignment.joint_assignment_diffusion.losses import geometry_step_loss
from mattergen.assignment.joint_assignment_diffusion.scaled_dataset import (
    clean_state_from_sample,
    crystal_to_tensors,
)
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.symmetry import (
    apply_symmetry_to_geometry,
    apply_symmetry_to_state,
    sample_symmetry_augment,
)
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    build_mol_conditioning_from_sample,
    load_molecular_csp_gemnet,
)
from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import MatterGenNativeNoiseAdapter
from mattergen.common.data.chemgraph import ChemGraph
import importlib.util

_j1 = ROOT / "scripts/assignment_diffusion_mvp/train_joint_axl_diffusion_j1.py"
_spec = importlib.util.spec_from_file_location("train_joint_axl_diffusion_j1", _j1)
_j1mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_j1mod)
build_cg = _j1mod.build_cg
resolve_device = _j1mod.resolve_device


def _tbin(t: float) -> str:
    if t < 0.2:
        return "[0.0,0.2)"
    if t < 0.4:
        return "[0.2,0.4)"
    if t < 0.6:
        return "[0.4,0.6)"
    if t < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ablation-arm", type=str, required=True, choices=["original", "clean_g"])
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["scaled_clean_g"]
    device = resolve_device(str(cfg.get("device", "auto")))
    out = Path(args.output_dir or (Path(cfg["output_dir"]) / args.ablation_arm))
    out.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.dataset_dir or cfg["dataset_dir"])
    train_samples = torch.load(ds_dir / "train.pt", map_location="cpu", weights_only=False)
    val_samples = torch.load(ds_dir / "val.pt", map_location="cpu", weights_only=False)

    ablation = resolve_geometry_ablation_arm(ablation_arm=args.ablation_arm)
    geom_cond = bool(ablation["geometry_assignment_conditioning"])
    gate = str(cfg.get("scf_time_gate") or "hard") if args.ablation_arm == "clean_g" else None
    thresh = float(cfg.get("scf_gate_threshold", 0.5))

    gem = cfg.get("gemnet") or {}
    bundle = load_molecular_csp_gemnet(
        model_path=args.mattergen_model_path or gem["model_path"],
        load_epoch=args.mattergen_load_epoch if args.mattergen_load_epoch is not None else gem.get("load_epoch", 294),
        checkpoint_path=args.mattergen_checkpoint or gem.get("checkpoint_path"),
        freeze=False,
        strict=True,
    )
    denoiser = bundle.denoiser.to(device)
    for par in denoiser.parameters():
        par.requires_grad_(True)
    pl = bundle.lightning_module.to(device)
    schedule = AsyncJumpSchedule()
    model = JointAXLModel(
        denoiser,
        num_orbits=int(cfg.get("max_orbits", 64)),
        schedule=schedule,
        geometry_assignment_conditioning=geom_cond,
        scf_time_gate=gate,
        scf_gate_threshold=thresh,
    ).to(device)

    opt_cfg = cfg.get("optim") or {}
    # GJumpHead / RJumpHead / orbit_slot_ctx are not trained in this L_geom-only run.
    skip_ids = {
        id(p)
        for p in list(model.r_head.parameters())
        + list(model.g_head.parameters())
        + list(model.orbit_slot_ctx.parameters())
    }
    new_params = [p for p in model.new_module_parameters() if id(p) not in skip_ids]
    opt = torch.optim.AdamW(
        [
            {"params": list(model.pretrained_parameters()), "lr": float(opt_cfg.get("lr_pretrained", 1e-5))},
            {"params": new_params, "lr": float(opt_cfg.get("lr_new", 1e-4))},
        ],
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    loss_fn = pl.diffusion_module.loss_fn
    steps = int(args.steps or cfg.get("steps", 8000))
    clip = float(opt_cfg.get("gradient_clip_norm", 1.0))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)))
    order = torch.randperm(len(train_samples), generator=g).tolist()

    prov = {
        "ABLATION_ARM": args.ablation_arm,
        "GEOMETRY_ASSIGNMENT_CONDITIONING": geom_cond,
        "TRAIN_ASSIGNMENT_HEADS": False,
        "CLEAN_G": args.ablation_arm == "clean_g",
        "SCF_TIME_GATE": gate,
        "SCF_GATE_THRESHOLD": thresh,
        "ASSIGNMENT_LOSS": "none_L_geom_only",
        "ORACLE": "clean_A0" if args.ablation_arm == "clean_g" else "none",
        "STEPS": steps,
        "SEED": int(cfg.get("seed", 17)),
        "N_TRAIN": len(train_samples),
        "N_VAL": len(val_samples),
        "MATTERGEN_LOAD_EPOCH": bundle.load_epoch,
        "MATTERGEN_CHECKPOINT_SHA": bundle.checkpoint_sha256,
        "DATASET": str(ds_dir),
    }
    (out / "runtime_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(json.dumps({"event": "scaled_clean_g_start", **prov}), flush=True)

    def one_geom(sample_cpu, t_f: float):
        sample = crystal_to_tensors(sample_cpu, device=device)
        st0, partition = clean_state_from_sample(sample)
        model.set_orbit_relations(partition, sample["role_edge_index"], sample["role_bond_type"])
        aug = sample_symmetry_augment(atomic_numbers=sample["z"], K=int(sample["K"]), generator=g)
        st0 = apply_symmetry_to_state(st0, aug)
        geo = apply_symmetry_to_geometry(
            frac=sample["pos"],
            atomic_numbers=sample["z"],
            role=sample["role"],
            copy=sample["copy"],
            aug=aug,
        )
        t = torch.tensor([t_f], device=device, dtype=torch.float32)
        noisy = noise.corrupt_fixed_sample(
            frac_coords_0=geo["pos"],
            lattice_0=sample["cell"],
            num_atoms=int(sample["N"]),
            t=t,
            generator=g,
        )
        extra = build_mol_conditioning_from_sample(
            {
                "z": geo["z"],
                "role": geo.get("role", sample["role"]),
                "copy": geo.get("copy", sample["copy"]),
                "role_edge_index": sample["role_edge_index"],
                "role_bond_type": sample["role_bond_type"],
            }
        )
        samp = {
            "N": sample["N"],
            "z": geo["z"],
            "role": geo.get("role", sample["role"]),
            "copy": geo.get("copy", sample["copy"]),
            "role_edge_index": sample["role_edge_index"],
            "role_bond_type": sample["role_bond_type"],
        }
        clean_cg = build_cg(samp, geo["pos"], sample["cell"], extra_mol=extra)
        noisy_cg = build_cg(samp, noisy.frac_coords_t, noisy.lattice_t, extra_mol=extra)
        geom = geometry_step_loss(
            model=model,
            loss_fn=loss_fn,
            corruption=noise.corruption,
            clean_cg=clean_cg,
            noisy_cg=noisy_cg,
            t=noisy.t,
            state_at_t=st0,
        )
        w = model.scf_assignment_weight(t_f)
        return geom, st0, t_f, w, sample["id"]

    log_every = int(cfg.get("log_every_steps", 50))
    val_dump_steps = [int(x) for x in (cfg.get("val_dump_steps") or [2000, 4000, 6000, 8000])]
    with (out / "training_trace.jsonl").open("w", buffering=1) as stream:
        for step in range(steps):
            sample_cpu = train_samples[order[step % len(order)]]
            t_f = float(torch.rand((), generator=g).item())
            opt.zero_grad(set_to_none=True)
            geom, st0, t_f, w, sid = one_geom(sample_cpu, t_f)
            L = geom["L_geom"]
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            pos_v = geom.get("geom_pos", geom.get("pos"))
            cell_v = geom.get("geom_cell", geom.get("cell"))
            def _f(x):
                if torch.is_tensor(x):
                    return float(x.detach())
                return float(x) if x is not None else 0.0
            row = {
                "step": step,
                "id": sid,
                "global_t": t_f,
                "t_bin": _tbin(t_f),
                "geometry_loss": float(L.detach()),
                "pos_loss": _f(pos_v),
                "cell_loss": _f(cell_v),
                "total_loss": float(L.detach()),
                "L_G": 0.0,
                "L_R": 0.0,
                "scf_weight": float(w),
                "scf_uses_clean_a0": 1.0 if args.ablation_arm == "clean_g" else 0.0,
                "cond_legal": float(bool(st0.validate()["legal"])),
                "ablation_arm": args.ablation_arm,
                "t_ge_half": float(t_f >= thresh),
            }
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == steps:
                print(json.dumps(row), flush=True)
            if (step + 1) in val_dump_steps:
                _run_val(model, val_samples, noise, loss_fn, g, device, thresh, out, step + 1)

    _dump_bins(out / "training_trace.jsonl", out / "geometry_bin_summary.json")
    cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"joint_state_dict": cpu_sd, "provenance": prov, "steps": steps}, out / "final_checkpoint.pt")
    best = out / "best_checkpoint.pt"
    if best.exists() or best.is_symlink():
        best.unlink()
    try:
        best.symlink_to("final_checkpoint.pt")
    except OSError:
        pass
    print(json.dumps({"event": "scaled_clean_g_train_done", "out": str(out)}), flush=True)


def _run_val(model, val_samples, noise, loss_fn, g, device, thresh, out, step):
    model.eval()
    n_t = 4
    rows = []
    with torch.no_grad():
        for i, sample_cpu in enumerate(val_samples):
            sample = crystal_to_tensors(sample_cpu, device=device)
            st0, partition = clean_state_from_sample(sample)
            model.set_orbit_relations(partition, sample["role_edge_index"], sample["role_bond_type"])
            for _ in range(n_t):
                t_f = float(torch.rand((), generator=g).item())
                t = torch.tensor([t_f], device=device, dtype=torch.float32)
                noisy = noise.corrupt_fixed_sample(
                    frac_coords_0=sample["pos"],
                    lattice_0=sample["cell"],
                    num_atoms=int(sample["N"]),
                    t=t,
                    generator=g,
                )
                extra = build_mol_conditioning_from_sample(
                    {
                        "z": sample["z"],
                        "role": sample["role"],
                        "copy": sample["copy"],
                        "role_edge_index": sample["role_edge_index"],
                        "role_bond_type": sample["role_bond_type"],
                    }
                )
                samp = {k: sample[k] for k in ("N", "z", "role", "copy", "role_edge_index", "role_bond_type")}
                clean_cg = build_cg(samp, sample["pos"], sample["cell"], extra_mol=extra)
                noisy_cg = build_cg(samp, noisy.frac_coords_t, noisy.lattice_t, extra_mol=extra)
                geom = geometry_step_loss(
                    model=model,
                    loss_fn=loss_fn,
                    corruption=noise.corruption,
                    clean_cg=clean_cg,
                    noisy_cg=noisy_cg,
                    t=noisy.t,
                    state_at_t=st0,
                )
                pv = geom.get("geom_pos", geom.get("pos"))
                cv = geom.get("geom_cell", geom.get("cell"))
                def _f(x):
                    if torch.is_tensor(x):
                        return float(x.detach())
                    return float(x) if x is not None else 0.0
                rows.append(
                    {
                        "step": step,
                        "id": sample["id"],
                        "global_t": t_f,
                        "geometry_loss": float(geom["L_geom"].detach()),
                        "pos_loss": _f(pv),
                        "cell_loss": _f(cv),
                        "t_ge_half": float(t_f >= thresh),
                    }
                )
    model.train()
    with (out / "val_trace.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    bins = {}
    for k in ("[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"):
        xs = [r for r in rows if _tbin(r["global_t"]) == k]
        bins[k] = {
            "n": len(xs),
            "geometry_loss": mean([r["geometry_loss"] for r in xs]),
            "pos_loss": mean([r["pos_loss"] for r in xs]),
            "cell_loss": mean([r["cell_loss"] for r in xs]),
        }
    summary = {
        "step": step,
        "n": len(rows),
        "overall": {
            "geometry_loss": mean([r["geometry_loss"] for r in rows]),
            "pos_loss": mean([r["pos_loss"] for r in rows]),
            "cell_loss": mean([r["cell_loss"] for r in rows]),
        },
        "t_ge_0.5": {
            "n": sum(1 for r in rows if r["t_ge_half"]),
            "geometry_loss": mean([r["geometry_loss"] for r in rows if r["t_ge_half"]]),
            "pos_loss": mean([r["pos_loss"] for r in rows if r["t_ge_half"]]),
            "cell_loss": mean([r["cell_loss"] for r in rows if r["t_ge_half"]]),
        },
        "by_bin": bins,
    }
    (out / f"val_step_{step}.json").write_text(json.dumps(summary, indent=2))
    conv_path = out / "convergence.json"
    conv = json.loads(conv_path.read_text()) if conv_path.exists() else {"by_step": {}}
    conv["by_step"][str(step)] = summary["overall"]
    steps_have = sorted(int(s) for s in conv["by_step"])
    if 4000 in steps_have and 8000 in steps_have:
        g4 = conv["by_step"]["4000"]["geometry_loss"]
        g8 = conv["by_step"]["8000"]["geometry_loss"]
        rel = (g8 - g4) / max(abs(g4), 1e-8) if g4 is not None and g8 is not None else None
        conv["from_4k_to_8k_geom_rel"] = rel
        conv["still_improving_at_8k"] = bool(rel is not None and rel < -0.03)
        conv["plateaued"] = bool(rel is not None and abs(rel) <= 0.03)
    conv_path.write_text(json.dumps(conv, indent=2))
    print(json.dumps({"event": "val", "step": step, "n": len(rows), **summary["overall"]}), flush=True)


def _dump_bins(trace: Path, dest: Path):
    rows = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    buckets = {k: [] for k in ("[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]")}
    pos_b = {k: [] for k in buckets}
    cell_b = {k: [] for k in buckets}
    for r in rows:
        k = r.get("t_bin") or _tbin(float(r["global_t"]))
        buckets[k].append(float(r["geometry_loss"]))
        pos_b[k].append(float(r["pos_loss"]) if r.get("pos_loss") is not None else 0.0)
        cell_b[k].append(float(r["cell_loss"]) if r.get("cell_loss") is not None else 0.0)

    def pack(xs):
        return sum(xs) / len(xs) if xs else 0.0

    out = {}
    for k, v in buckets.items():
        out[k] = {"n": len(v), "geometry_loss_mean": pack(v), "pos_loss_mean": pack(pos_b[k]), "cell_loss_mean": pack(cell_b[k])}
    out["overall"] = {
        "n": len(rows),
        "geometry_loss_mean": pack([r["geometry_loss"] for r in rows]),
        "pos_loss_mean": pack([r.get("pos_loss") or 0.0 for r in rows]),
        "cell_loss_mean": pack([r.get("cell_loss") or 0.0 for r in rows]),
    }
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
