#!/usr/bin/env python3
"""Paired reverse sampling: frozen clean A0, SCF gated by t."""
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
from mattergen.assignment.joint_assignment_diffusion.metrics import (
    crystal_geometry_vs_target,
    snapshot_inter_copy_metrics,
)
from mattergen.assignment.joint_assignment_diffusion.sampler import a_first_lie_step
from mattergen.assignment.joint_assignment_diffusion.scaled_dataset import clean_state_from_sample, crystal_to_tensors
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import build_mol_conditioning_from_sample, load_molecular_csp_gemnet
from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import MatterGenNativeNoiseAdapter
from mattergen.common.diffusion.predictors_correctors import LatticeAncestralSamplingPredictor
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition
from mattergen.diffusion.wrapped.wrapped_predictors_correctors import WrappedAncestralSamplingPredictor
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "train_joint_axl_diffusion_j1", ROOT / "scripts/assignment_diffusion_mvp/train_joint_axl_diffusion_j1.py"
)
_j1 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_j1)
build_cg = _j1.build_cg


def _as_cell_batch(cell: torch.Tensor) -> torch.Tensor:
    return cell.reshape(-1, 3, 3) if cell.numel() == 9 else cell.reshape(-1, 3, 3)


def _make_score_to_prev(noise: MatterGenNativeNoiseAdapter):
    """MatterGen ancestral predictor for pos (wrapped VE) + cell (lattice VP).

    The previous Euler ``x - dt * score`` is not the SDE reverse: predicted
    lattice noise can explode the cell, leaving GemNet with edges but no
    triplets (empty id_ragged_idx).
    """
    pos_pred = WrappedAncestralSamplingPredictor(corruption=noise.corruption.sdes["pos"], score_fn=None)
    cell_pred = LatticeAncestralSamplingPredictor(corruption=noise.corruption.sdes["cell"], score_fn=None)

    def score_to_prev(frac_t, cell_t, scores, t, s):
        device = frac_t.device
        t_ten = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
        dt = torch.as_tensor(float(s) - float(t), device=device, dtype=torch.float32)
        cell_b = _as_cell_batch(cell_t).to(device)
        n = int(frac_t.shape[0])
        pos_idx = torch.zeros(n, dtype=torch.long, device=device)
        # Dummy batch for LatticeVPSDE limit_mean (needs num_atoms).
        from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import _BatchView

        batch = _BatchView(
            {
                "pos": frac_t,
                "cell": cell_b,
                "num_atoms": torch.tensor([n], device=device, dtype=torch.long),
            }
        )
        frac_s, _ = pos_pred.update_given_score(
            x=frac_t,
            t=t_ten,
            dt=dt,
            batch_idx=pos_idx,
            score=scores["pos"],
            batch=batch,
        )
        cell_score = scores["cell"]
        if cell_score.ndim == 2:
            cell_score = cell_score.unsqueeze(0)
        cell_s, _ = cell_pred.update_given_score(
            x=cell_b,
            t=t_ten,
            dt=dt,
            batch_idx=None,
            score=cell_score,
            batch=batch,
        )
        cell_s = compute_lattice_polar_decomposition(cell_s)
        vol = torch.abs(torch.linalg.det(cell_s.reshape(3, 3)))
        # Keep the previous cell if the update collapsed / exploded the lattice.
        if not torch.isfinite(vol) or float(vol.item()) < 1e-2 or float(vol.item()) > 1e6:
            cell_s = cell_b
        return frac_s.remainder(1.0), cell_s.reshape(3, 3)

    return score_to_prev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ablation-arm", type=str, required=True, choices=["original", "clean_g"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["scaled_clean_g"]
    scfg = cfg.get("sampling") or {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    test = torch.load(Path(cfg["dataset_dir"]) / "test.pt", map_location="cpu", weights_only=False)
    n_crystals = min(int(scfg.get("n_crystals", 50)), len(test))
    n_traj = int(scfg.get("n_traj_per_crystal", 2))
    n_steps = int(scfg.get("n_reverse_steps", 100))
    snap_every = int(scfg.get("snapshot_every", 10))
    clash_cut = float(scfg.get("clash_cutoff", 1.2))
    overlap_ok = float(scfg.get("overlap_ok", 1.5))
    min_dist_cut = float(scfg.get("min_dist_cutoff", 0.7))

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

    times = [1.0 - i / float(n_steps) for i in range(n_steps + 1)]
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    score_to_prev = _make_score_to_prev(noise)
    rows = []
    with torch.no_grad():
        for ci in range(n_crystals):
            sample_cpu = test[ci]
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
            samp = dict(sample)
            for ti in range(n_traj):
                g = torch.Generator(device="cpu")
                g.manual_seed(10_000 + ci * 100 + ti)
                n = int(sample["N"])
                # t=1 native prior (same SDEs as training), not a deterministic cube.
                prior = noise.corrupt_fixed_sample(
                    frac_coords_0=sample["pos"],
                    lattice_0=sample["cell"],
                    num_atoms=n,
                    t=torch.tensor([1.0], dtype=torch.float32),
                    generator=g,
                )
                frac = prior.frac_coords_t.to(device)
                cell = prior.lattice_t.to(device)
                if cell.ndim == 3:
                    cell = cell.reshape(3, 3)
                state = st0.clone()
                snaps = []
                rec_clash = None
                rec_overlap = None
                for k in range(len(times) - 1):
                    t, s = times[k], times[k + 1]
                    state, frac, cell, _ev, _d = a_first_lie_step(
                        model=model,
                        chemgraph_builder=lambda st, fr, ce, extra=extra: build_cg(st, fr, ce, extra_mol=extra),
                        sample_tensors=samp,
                        state=state,
                        frac_t=frac,
                        cell_t=cell,
                        t=t,
                        s=s,
                        generator=g,
                        static_A=True,
                        score_to_prev=score_to_prev,
                    )
                    if k % snap_every == 0 or k + 1 == len(times) - 1:
                        met = snapshot_inter_copy_metrics(frac, cell, st0.copy_of(), clash_cutoff=clash_cut)
                        met["t"] = float(s)
                        snaps.append(met)
                        if rec_clash is None and met["E_clash"] < 1e-6:
                            rec_clash = float(s)
                        if rec_overlap is None and (
                            met.get("copy_overlap_max") == met.get("copy_overlap_max")
                            and met.get("copy_overlap_max", 1e9) <= overlap_ok
                        ):
                            rec_overlap = float(s)
                final_geo = crystal_geometry_vs_target(
                    frac.detach().cpu(),
                    cell.detach().cpu(),
                    sample["pos"].detach().cpu(),
                    sample["cell"].detach().cpu(),
                    copy_of=st0.copy_of().detach().cpu(),
                    min_dist_cutoff=min_dist_cut,
                )

                def _snap_mean(key):
                    xs = [float(s[key]) for s in snaps if s.get(key) is not None and s.get(key) == s.get(key)]
                    return sum(xs) / len(xs) if xs else None

                last = snaps[-1] if snaps else {}
                row = {
                    "crystal_index": ci,
                    "id": sample["id"],
                    "traj_index": ti,
                    "K": int(sample["K"]),
                    "N": int(sample["N"]),
                    "M": int(sample["M"]),
                    "ablation_arm": args.ablation_arm,
                    "recovery_t_clash": rec_clash,
                    "recovery_t_overlap": rec_overlap,
                    "final_E_clash": last.get("E_clash"),
                    "final_inter_copy_min_dist": last.get("inter_copy_min_dist"),
                    "final_inter_copy_p5": last.get("inter_copy_p5"),
                    "final_inter_copy_p10": last.get("inter_copy_p10"),
                    "final_copy_com_min": last.get("copy_com_min"),
                    "final_copy_radius_mean": last.get("copy_radius_mean"),
                    "final_copy_overlap_max": last.get("copy_overlap_max"),
                    "traj_mean_E_clash": _snap_mean("E_clash"),
                    "traj_mean_inter_copy_min_dist": _snap_mean("inter_copy_min_dist"),
                    "traj_mean_inter_copy_p5": _snap_mean("inter_copy_p5"),
                    "traj_mean_copy_com_min": _snap_mean("copy_com_min"),
                    "traj_mean_copy_radius_mean": _snap_mean("copy_radius_mean"),
                    "traj_mean_copy_overlap_max": _snap_mean("copy_overlap_max"),
                    **{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in final_geo.items()},
                }
                rows.append(row)
                torch.save(
                    {"row": row, "snapshots": snaps, "frac": frac.detach().cpu(), "cell": cell.detach().cpu()},
                    out / f"traj_{ci}_{ti}.pt",
                )
                print(json.dumps({"event": "sample_traj", **{k: row[k] for k in ("id", "traj_index", "no_clash", "min_dist", "final_E_clash") if k in row}}), flush=True)
    (out / "sample_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps({"event": "sample_done", "n": len(rows), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
