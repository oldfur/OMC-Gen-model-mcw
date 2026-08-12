#!/usr/bin/env python3
"""Evaluate J1: legality, lock schedule, metrics vs clean target."""
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

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.joint_assignment_diffusion.ctmc import simulate_forward_ctmc
from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
from mattergen.assignment.joint_assignment_diffusion.metrics import (
    assignment_vs_target,
    lock_schedule_checks,
    trajectory_legality,
)
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.state import a_from_role_and_copy
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import load_molecular_csp_gemnet
# evaluate path currently uses CTMC-only metrics; geometry ChemGraph not required here


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
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

    target = a_from_role_and_copy(
        role=sample["role"],
        copy=sample["copy"],
        partition=partition,
        atomic_numbers=sample["z"],
        role_z=sample["role_z"],
        K=int(sample["Z"]),
    )

    rows = []
    for seed in (cfg.get("evaluation") or {}).get("seeds", [0, 1, 2, 3]):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        traj = simulate_forward_ctmc(target, schedule=schedule, generator=g)
        leg = trajectory_legality(traj)
        locks = lock_schedule_checks(traj, schedule)
        # metrics at selected times
        for tf in [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]:
            st = traj.state_at(tf)
            m = assignment_vs_target(st, target)
            rows.append(
                {
                    "seed": int(seed),
                    "t": tf,
                    **leg,
                    **locks,
                    **m,
                }
            )
        print(json.dumps({"event": "j1_eval_seed", "seed": seed, **leg, **locks}), flush=True)

    with (out / "eval_trace.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "all_legal": all(r["all_legal"] for r in rows),
        "r_lock_ok": all(r["r_lock_ok"] for r in rows),
        "g_lock_ok": all(r["g_lock_ok"] for r in rows),
        "mean_exact_C_at_0": sum(1 for r in rows if r["t"] == 0.0 and r.get("exact_C")) / max(1, sum(1 for r in rows if r["t"] == 0.0)),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "j1_eval_done", **summary}), flush=True)


if __name__ == "__main__":
    main()
