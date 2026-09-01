#!/usr/bin/env python3
"""Held-out validation denoising for scaled Clean-G / Original."""
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

from mattergen.assignment.joint_assignment_diffusion.geometry_ablation import resolve_geometry_ablation_arm
from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
from mattergen.assignment.joint_assignment_diffusion.losses import geometry_step_loss
from mattergen.assignment.joint_assignment_diffusion.scaled_dataset import clean_state_from_sample, crystal_to_tensors
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import build_mol_conditioning_from_sample, load_molecular_csp_gemnet
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_joint_axl_diffusion_j1", ROOT / "scripts/assignment_diffusion_mvp/train_joint_axl_diffusion_j1.py"
)
_j1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_j1)
build_cg = _j1.build_cg


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
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ablation-arm", type=str, required=True, choices=["original", "clean_g"])
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--t-per-crystal", type=int, default=8)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["scaled_clean_g"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    samples = torch.load(Path(cfg["dataset_dir"]) / f"{args.split}.pt", map_location="cpu", weights_only=False)
    ablation = resolve_geometry_ablation_arm(ablation_arm=args.ablation_arm)
    gem = cfg.get("gemnet") or {}
    bundle = load_molecular_csp_gemnet(
        model_path=args.mattergen_model_path or gem["model_path"],
        load_epoch=args.mattergen_load_epoch if args.mattergen_load_epoch is not None else gem.get("load_epoch", 294),
        checkpoint_path=args.mattergen_checkpoint or gem.get("checkpoint_path"),
        freeze=False,
        strict=True,
    )
    gate = str(cfg.get("scf_time_gate") or "hard") if args.ablation_arm == "clean_g" else None
    model = JointAXLModel(
        bundle.denoiser.to(device),
        num_orbits=int(cfg.get("max_orbits", 64)),
        schedule=AsyncJumpSchedule(),
        geometry_assignment_conditioning=bool(ablation["geometry_assignment_conditioning"]),
        scf_time_gate=gate,
        scf_gate_threshold=float(cfg.get("scf_gate_threshold", 0.5)),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["joint_state_dict"], strict=False)
    model.eval()
    from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import MatterGenNativeNoiseAdapter

    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    loss_fn = bundle.lightning_module.to(device).diffusion_module.loss_fn
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)) + 17)
    thresh = float(cfg.get("scf_gate_threshold", 0.5))
    rows = []
    with torch.no_grad():
        for sample_cpu in samples:
            sample = crystal_to_tensors(sample_cpu, device=device)
            st0, partition = clean_state_from_sample(sample)
            model.set_orbit_relations(partition, sample["role_edge_index"], sample["role_bond_type"])
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
            for _ in range(int(args.t_per_crystal)):
                t_f = float(torch.rand((), generator=g).item())
                t = torch.tensor([t_f], device=device, dtype=torch.float32)
                noisy = noise.corrupt_fixed_sample(
                    frac_coords_0=sample["pos"], lattice_0=sample["cell"], num_atoms=int(sample["N"]), t=t, generator=g
                )
                clean_cg = build_cg(samp, sample["pos"], sample["cell"], extra_mol=extra)
                noisy_cg = build_cg(samp, noisy.frac_coords_t, noisy.lattice_t, extra_mol=extra)
                geom = geometry_step_loss(
                    model=model, loss_fn=loss_fn, corruption=noise.corruption,
                    clean_cg=clean_cg, noisy_cg=noisy_cg, t=noisy.t, state_at_t=st0,
                )
                pv = geom.get("geom_pos", geom.get("pos"))
                cv = geom.get("geom_cell", geom.get("cell"))
                rows.append(
                    {
                        "id": sample["id"],
                        "global_t": t_f,
                        "t_bin": _tbin(t_f),
                        "t_ge_half": float(t_f >= thresh),
                        "geometry_loss": float(geom["L_geom"].detach()),
                        "pos_loss": float(pv.detach()) if torch.is_tensor(pv) else float(pv or 0.0),
                        "cell_loss": float(cv.detach()) if torch.is_tensor(cv) else float(cv or 0.0),
                        "K": int(sample["K"]),
                        "N": int(sample["N"]),
                        "M": int(sample["M"]),
                    }
                )
    dest = out / f"denoising_{args.split}_trace.jsonl"
    with dest.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {
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
        "t_lt_0.5": {
            "n": sum(1 for r in rows if not r["t_ge_half"]),
            "geometry_loss": mean([r["geometry_loss"] for r in rows if not r["t_ge_half"]]),
        },
        "by_bin": {},
    }
    for k in ("[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"):
        xs = [r for r in rows if r["t_bin"] == k]
        summary["by_bin"][k] = {
            "n": len(xs),
            "geometry_loss": mean([r["geometry_loss"] for r in xs]),
            "pos_loss": mean([r["pos_loss"] for r in xs]),
            "cell_loss": mean([r["cell_loss"] for r in xs]),
        }
    (out / f"denoising_{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "denoising_eval_done", "split": args.split, **summary["overall"], "n": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
