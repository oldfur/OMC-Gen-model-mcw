#!/usr/bin/env python3
"""Sample joint A+X+L with A-first Lie reverse sampler (RHODIN01 MVP)."""
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

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
from mattergen.assignment.joint_assignment_diffusion.metrics import (
    lock_schedule_checks,
    trajectory_legality,
)
from mattergen.assignment.joint_assignment_diffusion.sampler import sample_joint_prior_and_trajectory
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.ctmc import CTMCTrajectory
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    build_mol_conditioning_from_sample,
    load_molecular_csp_gemnet,
)
from mattergen.common.data.chemgraph import ChemGraph


def build_cg(sample, frac, cell):
    n = int(sample["N"])
    lat = cell if cell.ndim == 3 else cell.unsqueeze(0)
    device = frac.device
    mol_extra = build_mol_conditioning_from_sample(
        {
            "z": sample["z"],
            "role": sample["role"],
            "copy": sample["copy"],
            "role_edge_index": sample["role_edge_index"],
            "role_bond_type": sample["role_bond_type"],
        }
    )
    kw = dict(
        atomic_numbers=sample["z"].long().to(device),
        pos=frac,
        cell=lat.to(device),
        num_atoms=torch.tensor([n], dtype=torch.long, device=device),
        num_nodes=n,
    )
    for k, v in mol_extra.items():
        if k == "mol_copy_id":
            continue
        if torch.is_tensor(v):
            kw[k] = v.to(device)
    return Batch.from_data_list([ChemGraph(**kw)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["joint_j1"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}

    gem = cfg.get("gemnet") or {}
    bundle = load_molecular_csp_gemnet(
        model_path=args.mattergen_model_path or gem["model_path"],
        load_epoch=args.mattergen_load_epoch if args.mattergen_load_epoch is not None else gem.get("load_epoch", 294),
        checkpoint_path=args.mattergen_checkpoint or gem.get("checkpoint_path"),
        freeze=False,
        strict=True,
    )
    sch_cfg = cfg.get("schedule") or {}
    schedule = AsyncJumpSchedule(
        r_lock=float(sch_cfg.get("r_lock", 0.72)),
        g_lock=float(sch_cfg.get("g_lock", 0.52)),
        kappa_r=float(sch_cfg.get("kappa_r", 4.0)),
        kappa_g=float(sch_cfg.get("kappa_g", 6.0)),
    )
    model = JointAXLModel(bundle.denoiser.to(device), num_orbits=partition.J, schedule=schedule).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["joint_state_dict"], strict=False)
    model.eval()
    model.set_orbit_relations(partition, sample["role_edge_index"], sample["role_bond_type"])

    timesteps = (cfg.get("evaluation") or {}).get("timesteps") or [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    results = []
    for si in range(args.num_samples):
        g = torch.Generator(device="cpu")
        g.manual_seed(1000 + si)
        traj = sample_joint_prior_and_trajectory(
            model=model,
            partition=partition,
            atomic_numbers=sample["z"],
            role_z=sample["role_z"],
            K=int(sample["Z"]),
            sample_tensors=sample,
            chemgraph_builder=build_cg,
            timesteps=timesteps,
            generator=g,
        )
        # wrap for metrics
        ctmc = CTMCTrajectory(
            times=traj.times,
            states=traj.assignments,
            events=traj.events,
        )
        leg = trajectory_legality(ctmc)
        locks = lock_schedule_checks(ctmc, schedule)
        final = traj.assignments[-1]
        row = {
            "sample_index": si,
            "final_legal": final.validate()["legal"],
            "num_events": len(traj.events),
            **leg,
            **locks,
        }
        results.append(row)
        torch.save(
            {
                "frac": traj.frac_list[-1].cpu(),
                "cell": traj.cell_list[-1].cpu(),
                "A": final.A.cpu(),
                "orbit": final.orbit_of().cpu(),
                "copy": final.copy_of().cpu(),
            },
            out / f"sample_{si}.pt",
        )
        print(json.dumps({"event": "j1_sample", **row}), flush=True)

    (out / "sample_summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"event": "j1_sample_done", "n": len(results), "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
