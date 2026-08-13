#!/usr/bin/env python3
"""Evaluate J1.1: legality, mobility windows, jump budgets, metrics vs clean target."""
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

from torch_geometric.data import Batch

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.joint_assignment_diffusion.ctmc import simulate_forward_ctmc
from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
from mattergen.assignment.joint_assignment_diffusion.metrics import (
    assignment_vs_target,
    jump_budget_diagnostics,
    lock_schedule_checks,
    trajectory_legality,
)
from mattergen.assignment.joint_assignment_diffusion.reverse_eval import (
    evaluate_local_reverse_pair,
    summarize_reverse_eval,
)
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.state import a_from_role_and_copy
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
    schedule = AsyncJumpSchedule.from_config(cfg.get("schedule") or {})
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
    seed_summaries = []
    for seed in (cfg.get("evaluation") or {}).get("seeds", [0, 1, 2, 3]):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        traj = simulate_forward_ctmc(target, schedule=schedule, generator=g)
        leg = trajectory_legality(traj)
        locks = lock_schedule_checks(traj, schedule)
        budget = jump_budget_diagnostics(traj, schedule)
        seed_summaries.append({"seed": int(seed), **leg, **locks, **budget})
        for tf in [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]:
            st = traj.state_at(tf)
            m = assignment_vs_target(st, target)
            rows.append(
                {
                    "seed": int(seed),
                    "t": tf,
                    **leg,
                    **locks,
                    **budget,
                    **m,
                }
            )
        print(json.dumps({"event": "j1_eval_seed", "seed": seed, **leg, **locks, **budget}), flush=True)

    with (out / "eval_trace.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "all_legal": all(r["all_legal"] for r in rows),
        "r_lock_ok": all(r["r_lock_ok"] for r in rows),
        "g_lock_ok": all(r["g_lock_ok"] for r in rows),
        "mean_exact_C_at_0": sum(1 for r in rows if r["t"] == 0.0 and r.get("exact_C"))
        / max(1, sum(1 for r in rows if r["t"] == 0.0)),
        "mean_n_R": sum(s["n_R"] for s in seed_summaries) / max(1, len(seed_summaries)),
        "mean_n_G": sum(s["n_G"] for s in seed_summaries) / max(1, len(seed_summaries)),
        "expected_R": schedule.kappa_r,
        "expected_G": schedule.kappa_g,
        "r_window": list(schedule.r_window),
        "g_window": list(schedule.g_window),
        "seeds_differ": len({(s["n_R"], s["n_G"], s["num_events"]) for s in seed_summaries}) > 1,
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "j1_eval_done", **summary}), flush=True)

    # ---- learned vs uniform local reverse (held-out symmetry + forward traj) ----
    eval_cfg = cfg.get("evaluation") or {}
    rev_cfg = eval_cfg.get("reverse_eval") or {}
    rev_seeds = rev_cfg.get("seeds") or [101, 102, 103, 104]
    grid = list(eval_cfg.get("timesteps") or [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    grid = sorted({float(x) for x in grid}, reverse=True)
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    reverse_rows = []
    for seed in rev_seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        aug = sample_symmetry_augment(atomic_numbers=sample["z"], K=int(sample["Z"]), generator=g)
        st0 = apply_symmetry_to_state(target, aug)
        geo = apply_symmetry_to_geometry(
            frac=sample["pos"],
            atomic_numbers=sample["z"],
            role=sample["role"],
            copy=sample["copy"],
            aug=aug,
        )
        traj = simulate_forward_ctmc(st0, schedule=schedule, generator=g)
        samp_aug = dict(sample)
        samp_aug["z"] = geo["z"]
        samp_aug["pos"] = geo["pos"]
        if "role" in geo:
            samp_aug["role"] = geo["role"]
        if "copy" in geo:
            samp_aug["copy"] = geo["copy"]
        mol_extra = build_mol_conditioning_from_sample(
            {
                "z": samp_aug["z"],
                "role": samp_aug.get("role", sample["role"]),
                "copy": samp_aug.get("copy", sample["copy"]),
                "role_edge_index": sample["role_edge_index"],
                "role_bond_type": sample["role_bond_type"],
            }
        )
        for t, s in zip(grid[:-1], grid[1:]):
            if t <= s:
                continue
            state_t = traj.state_at(t)
            state_s = traj.state_at(s)
            t_ten = torch.tensor([t], device=device, dtype=torch.float32)
            noisy = noise.corrupt_fixed_sample(
                frac_coords_0=geo["pos"],
                lattice_0=sample["cell"],
                num_atoms=int(sample["N"]),
                t=t_ten,
                generator=g,
            )
            kw = dict(
                atomic_numbers=samp_aug["z"].long().to(device),
                pos=noisy.frac_coords_t,
                cell=(noisy.lattice_t if noisy.lattice_t.ndim == 3 else noisy.lattice_t.unsqueeze(0)),
                num_atoms=torch.tensor([int(sample["N"])], dtype=torch.long, device=device),
                num_nodes=int(sample["N"]),
            )
            for k, v in mol_extra.items():
                if k == "mol_copy_id":
                    continue
                if torch.is_tensor(v):
                    kw[k] = v.to(device)
            cg = Batch.from_data_list([ChemGraph(**kw)])
            pair_seed = int(seed) * 10007 + int(round(t * 1000))
            row = evaluate_local_reverse_pair(
                model=model,
                chemgraph_t=cg,
                state_t=state_t,
                state_s=state_s,
                t=t,
                s=s,
                seed=pair_seed,
                c0=st0.C(),
            )
            reverse_rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "j1_reverse_eval_pair",
                        "seed": seed,
                        "t": t,
                        "s": s,
                        "win_ari": row["win_ari"],
                        "win_orbit": row["win_orbit"],
                        "dARI_L": row["delta_d_ari_learned"],
                        "dARI_U": row["delta_d_ari_uniform"],
                    }
                ),
                flush=True,
            )
    with (out / "reverse_eval_trace.jsonl").open("w") as f:
        for r in reverse_rows:
            f.write(json.dumps(r) + "\n")
    rev_sum = summarize_reverse_eval(reverse_rows)
    (out / "reverse_eval_summary.json").write_text(json.dumps(rev_sum, indent=2))
    print(json.dumps({"event": "j1_reverse_eval_done", **{k: v for k, v in rev_sum.items() if k not in ("by_g_bin", "by_r_bin")}}), flush=True)


if __name__ == "__main__":
    main()
