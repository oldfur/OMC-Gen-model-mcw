#!/usr/bin/env python3
"""Smoke: Original vs G-conditioned geometry paths (G must change scores only when scf on)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACE = {
    "event": "g_geometry_architecture_trace",
    "G_to_geometry": [
        "JointAssignmentState.copy_of / C()  (G / copy membership)",
        "JointAXLModel._build_a_feedback: node_delta (orbit+clock), edge_adapter(C,orbit,rho), mid_block(assign_mp+copy_pool)",
        "GemNetT.forward(soft_c_feedback=scf) if scf.enabled",
        "forces/stress -> ChemGraph pos/cell scores",
    ],
    "reverse": [
        "sampler.a_first_lie_step: Gillespie A on (s,t] -> state_s (G written back)",
        "model(chemgraph at X_t, t, state_s, compute_jumps=False) -> geometry scores",
        "Euler X_t,L_t -> X_s,L_s; next macrostep consumes state_s",
    ],
    "heads": "GJumpHead != GemNet forces/stress; L from lattice_out_blocks; X from forces",
    "original_arm": "geometry_assignment_conditioning=false => scf.enabled=False",
    "g_conditioned_arm": "geometry_assignment_conditioning=true => C/copy_of condition GemNet",
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
    from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
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
    st = a_from_role_and_copy(
        role=sample["role"],
        copy=sample["copy"],
        partition=partition,
        atomic_numbers=sample["z"],
        role_z=sample["role_z"],
        K=int(sample["Z"]),
    )
    # permute copies (G) without touching orbits (Rbar)
    perm = torch.roll(torch.arange(st.K, device=st.A.device), 1)
    st_perm = st.clone()
    st_perm.A = st.A.index_select(-1, perm)
    cg = build_cg(sample, sample["pos"], sample["cell"])
    t = torch.tensor([0.5], device=device)
    rows = []
    for name, flag in (("original", False), ("g_conditioned", True)):
        model = JointAXLModel(
            bundle.denoiser.to(device),
            num_orbits=partition.J,
            schedule=schedule,
            g_copy_context_mode=str(cfg.get("g_copy_context_mode") or "template_counterfactual"),
            geometry_assignment_conditioning=flag,
        ).to(device)
        model.eval()
        with torch.no_grad():
            o1 = model(cg, t, st, compute_jumps=False)
            o2 = model(cg, t, st_perm, compute_jumps=False)
        dpos = float((o1.chemgraph_scores["pos"] - o2.chemgraph_scores["pos"]).abs().mean())
        rows.append(
            {
                "arm": name,
                "scf_enabled": flag,
                "pos_score_l1_under_copy_perm": dpos,
                "g_affects_geometry": dpos > 1e-8,
            }
        )
        del model
    ok_orig = (not rows[0]["g_affects_geometry"])
    ok_g = bool(rows[1]["g_affects_geometry"])
    out = {"event": "smoke_g_geometry_ablation", "ok_original_invariant": ok_orig, "ok_g_affects_geometry": ok_g, "arms": rows}
    print(json.dumps(out), flush=True)
    if not (ok_orig and ok_g):
        raise SystemExit("smoke failed: G-conditioning path mismatch")


if __name__ == "__main__":
    main()
