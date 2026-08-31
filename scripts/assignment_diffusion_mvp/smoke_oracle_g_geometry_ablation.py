#!/usr/bin/env python3
"""Smoke: Original vs Oracle-G geometry paths (no learned G).

Sanity:
  1. Oracle-G SCF enabled; Original SCF disabled
  2. Legal G-swap changes pos score only when SCF is on
  3. L_geom-only graph: GJumpHead / RJumpHead get no grads; Oracle adapters do
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
    ORACLE_G_CONSTRUCTION,
    ORACLE_G_NOT,
)

TRACE = {
    "event": "oracle_g_geometry_architecture_trace",
    "oracle_G_construction": ORACLE_G_CONSTRUCTION,
    "oracle_G_not": ORACLE_G_NOT,
    "G_to_geometry": [
        "oracle A_t = forward_CTMC(GT A_0).state_at(t)  (copy_of / C / orbit_of)",
        "JointAXLModel._build_a_feedback: node_delta, edge_adapter(C), mid_block(assign_mp+copy_pool)",
        "GemNetT.forward(soft_c_feedback=scf) if scf.enabled",
        "forces/stress -> ChemGraph pos/cell scores",
        "loss = L_geom only; GJumpHead is not in the graph",
    ],
    "original_arm": "geometry_assignment_conditioning=false => scf.enabled=False; L_geom only",
    "oracle_g_arm": "geometry_assignment_conditioning=true; A_t=oracle forward; L_geom only",
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
    st_t, traj = oracle_assignment_at_t(st0, schedule=schedule, t=0.5, generator=gen)
    g_moves = enumerate_g_moves(st_t)
    if not g_moves:
        g_moves = enumerate_g_moves(st0)
        st_t = st0
    if not g_moves:
        raise SystemExit("smoke failed: no legal G swap to probe copy partition")
    st_swap = apply_move(st_t, g_moves[0])
    cg = build_cg(sample, sample["pos"], sample["cell"])
    t = torch.tensor([0.5], device=device)

    def _break_zero_init_adapters(model):
        """Smoke-only: last-layer residuals are zero-init, so un-zero weights."""
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
    for name, flag in (("original", False), ("oracle_g", True)):
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
        scf, _meta = model._build_a_feedback(st_t, t_scalar=0.5)
        with torch.no_grad():
            o1 = model(cg, t, st_t, compute_jumps=False)
            o2 = model(cg, t, st_swap, compute_jumps=False)
        dpos = float((o1.chemgraph_scores["pos"] - o2.chemgraph_scores["pos"]).abs().mean())

        model.zero_grad(set_to_none=True)
        out = model(cg, t, st_t, compute_jumps=False)
        dummy = out.chemgraph_scores["pos"].float().pow(2).mean() + out.chemgraph_scores["cell"].float().pow(2).mean()
        dummy.backward()
        g_head_gn, _ = _grad_norm(model.g_head.parameters())
        r_head_gn, _ = _grad_norm(model.r_head.parameters())
        adapter_gn, _ = _grad_norm(list(model.copy_to_node.parameters()) + list(model.assign_mp.parameters()))
        rows.append(
            {
                "arm": name,
                "scf_enabled_flag": flag,
                "scf_enabled_runtime": bool(scf.get("enabled", False)),
                "oracle_state_legal": bool(st_t.validate()["legal"]),
                "n_fwd_events_to_build_A_t": len(traj.events),
                "g_swap": [g_moves[0].i, g_moves[0].j],
                "pos_score_l1_under_g_swap": dpos,
                "compute_jumps": False,
                "g_head_grad_norm_from_Lgeom": g_head_gn,
                "r_head_grad_norm_from_Lgeom": r_head_gn,
                "adapter_grad_norm_from_Lgeom": adapter_gn,
            }
        )
        del model
    orig_l1 = rows[0]["pos_score_l1_under_g_swap"]
    ora_l1 = rows[1]["pos_score_l1_under_g_swap"]
    ok_orig = (not rows[0]["scf_enabled_runtime"]) and orig_l1 < 1e-5
    ok_oracle = (
        bool(rows[1]["scf_enabled_runtime"])
        and ora_l1 > 1e-4
        and ora_l1 > 10.0 * max(orig_l1, 1e-12)
    )
    ok_lg_zero_graph = (
        rows[0]["g_head_grad_norm_from_Lgeom"] == 0.0
        and rows[1]["g_head_grad_norm_from_Lgeom"] == 0.0
        and rows[0]["r_head_grad_norm_from_Lgeom"] == 0.0
        and rows[1]["r_head_grad_norm_from_Lgeom"] == 0.0
        and rows[0]["adapter_grad_norm_from_Lgeom"] == 0.0
        and rows[1]["adapter_grad_norm_from_Lgeom"] > 0.0
    )
    out = {
        "event": "smoke_oracle_g_geometry_ablation",
        "ok_original_invariant": ok_orig,
        "ok_oracle_g_affects_geometry": ok_oracle,
        "ok_L_geom_only_no_jump_head_grads": ok_lg_zero_graph,
        "orig_l1": orig_l1,
        "oracle_g_l1": ora_l1,
        "oracle_construction": ORACLE_G_CONSTRUCTION,
        "arms": rows,
    }
    print(json.dumps(out), flush=True)
    if not (ok_orig and ok_oracle and ok_lg_zero_graph):
        raise SystemExit("smoke failed: Oracle-G geometry path mismatch")


if __name__ == "__main__":
    main()
