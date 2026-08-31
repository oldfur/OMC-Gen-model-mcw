#!/usr/bin/env python3
"""Smoke: Original vs Clean-G geometry paths (A_cond = A_0^GT, no learned G).

Sanity:
  1. Clean-G SCF enabled and consumes clean A_0, not noisy A_t
  2. Clean A_0 is legal (copy membership / cardinality)
  3. Legal G-swap on A_0 changes pos score only when SCF is on
  4. L_geom-only graph: G/R heads get no grads; Clean-G adapters do
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.joint_assignment_diffusion.geometry_ablation import (
    CLEAN_G_CONSTRUCTION,
    CLEAN_G_NOT,
)

TRACE = {
    "event": "clean_g_geometry_architecture_trace",
    "clean_G_construction": CLEAN_G_CONSTRUCTION,
    "clean_G_not": CLEAN_G_NOT,
    "G_to_geometry": [
        "A_cond = A_0^GT (clean copy_of / C / orbit_of) at every t",
        "X_t/L_t still noised; SCF never sees traj.state_at(t)",
        "JointAXLModel._build_a_feedback -> GemNet scf -> pos/cell scores",
        "loss = L_geom only; GJumpHead is not in the graph",
    ],
    "original_arm": "geometry_assignment_conditioning=false => scf.enabled=False; L_geom only",
    "clean_g_arm": "geometry_assignment_conditioning=true; A_cond=A_0^GT; L_geom only",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/assignment_diffusion_mvp/joint_axl_diffusion_j1.yaml")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()
    print(json.dumps(TRACE, indent=2), flush=True)
    if not args.execute:
        print(json.dumps({"event": "smoke_trace_only", "hint": "pass --execute to run two-arm forward check"}), flush=True)
        return

    import torch
    import yaml
    from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
    from mattergen.assignment.joint_assignment_diffusion.ctmc import oracle_assignment_at_t
    from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
    from mattergen.assignment.joint_assignment_diffusion.legal_moves import (
        apply_move,
        enumerate_g_moves,
    )
    from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
    from mattergen.assignment.joint_assignment_diffusion.state import a_from_role_and_copy
    from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
        build_mol_conditioning_from_sample,
        load_molecular_csp_gemnet,
    )
    from mattergen.common.data.chemgraph import ChemGraph
    from torch_geometric.data import Batch

    def build_cg(sample, frac, cell):
        n = int(sample["N"])
        lat = cell if cell.ndim == 3 else cell.unsqueeze(0)
        extra = build_mol_conditioning_from_sample(
            {
                "z": sample["z"],
                "role": sample["role"],
                "copy": sample["copy"],
                "role_edge_index": sample["role_edge_index"],
                "role_bond_type": sample["role_bond_type"],
            }
        )
        kw = dict(
            atomic_numbers=sample["z"].long(),
            pos=frac,
            cell=lat,
            num_atoms=torch.tensor([n], dtype=torch.long, device=frac.device),
            num_nodes=n,
        )
        for k, v in extra.items():
            if k != "mol_copy_id" and torch.is_tensor(v):
                kw[k] = v.to(frac.device)
        return Batch.from_data_list([ChemGraph(**kw)])

    cfg = yaml.safe_load(args.config.read_text())["joint_j1"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.load(cfg["fixed_sample_path"], map_location=device, weights_only=False)
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)
    gem = cfg.get("gemnet") or {}
    bundle = load_molecular_csp_gemnet(
        model_path=args.mattergen_model_path or gem["model_path"],
        load_epoch=args.mattergen_load_epoch if args.mattergen_load_epoch is not None else gem.get("load_epoch", 294),
        checkpoint_path=args.mattergen_checkpoint or gem.get("checkpoint_path"),
        freeze=True,
        strict=True,
    )
    schedule = AsyncJumpSchedule.from_config(cfg.get("schedule") or {})

    st0 = a_from_role_and_copy(
        role=sample["role"],
        copy=sample["copy"],
        partition=partition,
        atomic_numbers=sample["z"],
        role_z=sample["role_z"],
        K=int(sample["Z"]),
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.get("seed", 17)))
    st_t, traj = oracle_assignment_at_t(st0, schedule=schedule, t=0.6, generator=gen)
    cond_is_not_at = not bool(torch.equal(st0.A, st_t.A))
    legal0 = st0.validate()
    g_moves = enumerate_g_moves(st0)
    if not g_moves:
        raise SystemExit("smoke failed: no legal G swap on clean A_0")
    st_swap = apply_move(st0, g_moves[0])
    cg = build_cg(sample, sample["pos"], sample["cell"])
    t = torch.tensor([0.5], device=device)

    def _break_zero_init_adapters(model):
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(model.copy_to_node.weight)
            last = None
            for m in model.spatial_edge.proj.modules():
                if isinstance(m, torch.nn.Linear):
                    last = m
            if last is not None:
                torch.nn.init.xavier_uniform_(last.weight)
            last_mp = None
            for m in model.assign_mp.upd.modules():
                if isinstance(m, torch.nn.Linear):
                    last_mp = m
            if last_mp is not None:
                torch.nn.init.xavier_uniform_(last_mp.weight)

    def _grad_norm(params) -> tuple[float, int]:
        acc = 0.0
        n = 0
        for p in params:
            n += 1
            if p.grad is not None:
                acc += float(p.grad.detach().float().pow(2).sum())
        return acc ** 0.5, n

    rows = []
    for name, flag in (("original", False), ("clean_g", True)):
        model = JointAXLModel(
            bundle.denoiser.to(device),
            num_orbits=partition.J,
            schedule=schedule,
            g_copy_context_mode=str(cfg.get("g_copy_context_mode") or "template_counterfactual"),
            geometry_assignment_conditioning=flag,
        ).to(device)
        model.set_orbit_relations(partition, sample["role_edge_index"], sample["role_bond_type"])
        _break_zero_init_adapters(model)
        model.train()
        scf, _meta = model._build_a_feedback(st0, t_scalar=0.5)
        copy_of = _meta.get("copy_of")
        with torch.no_grad():
            o1 = model(cg, t, st0, compute_jumps=False)
            o2 = model(cg, t, st_swap, compute_jumps=False)
        dpos = float((o1.chemgraph_scores["pos"] - o2.chemgraph_scores["pos"]).abs().mean())

        model.zero_grad(set_to_none=True)
        out = model(cg, t, st0, compute_jumps=False)
        dummy = out.chemgraph_scores["pos"].float().pow(2).mean() + out.chemgraph_scores["cell"].float().pow(2).mean()
        dummy_req = bool(dummy.requires_grad and dummy.grad_fn is not None)
        if dummy_req:
            dummy.backward()
            g_head_gn, _ = _grad_norm(model.g_head.parameters())
            r_head_gn, _ = _grad_norm(model.r_head.parameters())
            adapter_gn, _ = _grad_norm(
                list(model.copy_to_node.parameters()) + list(model.assign_mp.parameters())
            )
        else:
            if flag:
                raise SystemExit("smoke failed: Clean-G geometry scores have no grad_fn")
            g_head_gn = 0.0
            r_head_gn = 0.0
            adapter_gn = 0.0
        rows.append(
            {
                "arm": name,
                "scf_enabled_flag": flag,
                "scf_enabled_runtime": bool(scf.get("enabled", False)),
                "scf_state_is_clean_a0": True,
                "clean_a0_legal": bool(legal0.get("legal")),
                "clean_capacity_ok": bool(legal0.get("capacity_ok")),
                "n_fwd_events_on_unused_At": len(traj.events),
                "clean_a0_differs_from_At": cond_is_not_at,
                "g_swap": [g_moves[0].i, g_moves[0].j],
                "pos_score_l1_under_g_swap": dpos,
                "compute_jumps": False,
                "dummy_requires_grad": dummy_req,
                "g_head_grad_norm_from_Lgeom": g_head_gn,
                "r_head_grad_norm_from_Lgeom": r_head_gn,
                "adapter_grad_norm_from_Lgeom": adapter_gn,
                "copy_of_unique": int(torch.unique(copy_of).numel()) if copy_of is not None else -1,
            }
        )
        del model
    orig_l1 = rows[0]["pos_score_l1_under_g_swap"]
    cln_l1 = rows[1]["pos_score_l1_under_g_swap"]
    ok_orig = (not rows[0]["scf_enabled_runtime"]) and orig_l1 < 1e-5
    ok_clean = (
        bool(rows[1]["scf_enabled_runtime"])
        and cln_l1 > 1e-4
        and cln_l1 > 10.0 * max(orig_l1, 1e-12)
        and bool(legal0.get("legal"))
        and cond_is_not_at
    )
    ok_lg_zero_graph = (
        (not rows[0]["dummy_requires_grad"])
        and rows[0]["adapter_grad_norm_from_Lgeom"] == 0.0
        and rows[1]["dummy_requires_grad"]
        and rows[1]["adapter_grad_norm_from_Lgeom"] > 0.0
        and rows[0]["g_head_grad_norm_from_Lgeom"] == 0.0
        and rows[1]["g_head_grad_norm_from_Lgeom"] == 0.0
        and rows[0]["r_head_grad_norm_from_Lgeom"] == 0.0
        and rows[1]["r_head_grad_norm_from_Lgeom"] == 0.0
    )
    out = {
        "event": "smoke_clean_g_geometry_ablation",
        "ok_original_invariant": ok_orig,
        "ok_clean_g_affects_geometry": ok_clean,
        "ok_L_geom_only_no_jump_head_grads": ok_lg_zero_graph,
        "ok_scf_uses_a0_not_At": cond_is_not_at and bool(legal0.get("legal")),
        "orig_l1": orig_l1,
        "clean_g_l1": cln_l1,
        "clean_construction": CLEAN_G_CONSTRUCTION,
        "arms": rows,
    }
    print(json.dumps(out), flush=True)
    if not (ok_orig and ok_clean and ok_lg_zero_graph):
        raise SystemExit("smoke failed: Clean-G geometry path mismatch")


if __name__ == "__main__":
    main()
