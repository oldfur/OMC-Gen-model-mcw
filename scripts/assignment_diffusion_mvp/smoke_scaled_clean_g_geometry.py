#!/usr/bin/env python3
"""Smoke: Original vs gated Clean-G (A0 + hard t-gate at 0.5)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.joint_assignment_diffusion.joint_model import scf_hard_weight


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/assignment_diffusion_mvp/scaled_clean_g_geometry.yaml")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()

    gate_rows = []
    for t in (0.0, 0.49, 0.5, 0.51, 1.0):
        gate_rows.append(
            {
                "t": t,
                "original": scf_hard_weight(t, enabled=False, gate="hard", threshold=0.5),
                "clean_g": scf_hard_weight(t, enabled=True, gate="hard", threshold=0.5),
            }
        )
    print(json.dumps({"event": "scf_hard_gate_table", "rows": gate_rows}, indent=2), flush=True)
    assert all(r["original"] == 0.0 for r in gate_rows)
    assert gate_rows[0]["clean_g"] == 0.0 and gate_rows[1]["clean_g"] == 0.0
    assert gate_rows[2]["clean_g"] == 1.0 and gate_rows[3]["clean_g"] == 1.0
    if not args.execute:
        print(json.dumps({"event": "smoke_gate_only", "hint": "pass --execute for GemNet G-swap checks"}), flush=True)
        return

    import torch
    import yaml
    from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
    from mattergen.assignment.joint_assignment_diffusion.legal_moves import apply_move, enumerate_g_moves
    from mattergen.assignment.joint_assignment_diffusion.scaled_dataset import clean_state_from_sample, crystal_to_tensors
    from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
    from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
        build_mol_conditioning_from_sample,
        load_molecular_csp_gemnet,
    )
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "train_joint_axl_diffusion_j1", ROOT / "scripts/assignment_diffusion_mvp/train_joint_axl_diffusion_j1.py"
    )
    _j1 = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_j1)
    build_cg = _j1.build_cg

    cfg = yaml.safe_load(args.config.read_text())["scaled_clean_g"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Path(cfg["dataset_dir"]) / "train.pt"
    if ds.exists():
        samples = torch.load(ds, map_location="cpu", weights_only=False)
        sample_cpu = samples[0]
        sample = crystal_to_tensors(sample_cpu, device=device)
        st0, partition = clean_state_from_sample(sample)
    else:
        # fallback: RHODIN01 fixed sample
        jcfg = yaml.safe_load((ROOT / "configs/assignment_diffusion_mvp/joint_axl_diffusion_j1.yaml").read_text())["joint_j1"]
        raw = torch.load(jcfg["fixed_sample_path"], map_location=device, weights_only=False)
        from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
        import json as _json
        orbits = _json.loads(Path(jcfg["automorphism_orbits_path"]).read_text())
        per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
        partition = build_orbit_partition(per_role)
        from mattergen.assignment.joint_assignment_diffusion.state import a_from_role_and_copy
        st0 = a_from_role_and_copy(
            role=raw["role"], copy=raw["copy"], partition=partition,
            atomic_numbers=raw["z"], role_z=raw["role_z"], K=int(raw["Z"]),
        )
        sample = raw
        sample["K"] = int(raw["Z"])
        sample["role_edge_index"] = raw["role_edge_index"]
        sample["role_bond_type"] = raw["role_bond_type"]

    gem = cfg.get("gemnet") or {}
    bundle = load_molecular_csp_gemnet(
        model_path=args.mattergen_model_path or gem["model_path"],
        load_epoch=args.mattergen_load_epoch if args.mattergen_load_epoch is not None else gem.get("load_epoch", 294),
        checkpoint_path=args.mattergen_checkpoint or gem.get("checkpoint_path"),
        freeze=True,
        strict=True,
    )
    g_moves = enumerate_g_moves(st0)
    if not g_moves:
        raise SystemExit("smoke failed: no legal G swap on A0")
    st_swap = apply_move(st0, g_moves[0])
    extra = build_mol_conditioning_from_sample(
        {
            "z": sample["z"],
            "role": sample["role"],
            "copy": sample["copy"],
            "role_edge_index": sample["role_edge_index"],
            "role_bond_type": sample["role_bond_type"],
        }
    )
    cg = build_cg(sample, sample["pos"], sample["cell"], extra_mol=extra)

    def _xavier(model):
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

    rows = []
    for name, flag, gate in (("original", False, None), ("clean_g", True, "hard")):
        model = JointAXLModel(
            bundle.denoiser.to(device),
            num_orbits=max(int(partition.J), 8),
            schedule=AsyncJumpSchedule(),
            geometry_assignment_conditioning=flag,
            scf_time_gate=gate,
            scf_gate_threshold=0.5,
        ).to(device)
        model.set_orbit_relations(partition, sample["role_edge_index"], sample["role_bond_type"])
        _xavier(model)
        model.eval()
        for tval in (0.3, 0.7):
            t = torch.tensor([tval], device=device)
            scf, meta = model._build_a_feedback(st0, t_scalar=tval)
            with torch.no_grad():
                o1 = model(cg, t, st0, compute_jumps=False)
                o2 = model(cg, t, st_swap, compute_jumps=False)
            dpos = float((o1.chemgraph_scores["pos"] - o2.chemgraph_scores["pos"]).abs().mean())
            dcell = float((o1.chemgraph_scores["cell"] - o2.chemgraph_scores["cell"]).abs().mean())
            rows.append(
                {
                    "arm": name,
                    "t": tval,
                    "scf_enabled_runtime": bool(meta.get("scf_enabled_runtime", scf.get("enabled"))),
                    "scf_weight": float(meta.get("scf_weight", 0.0)),
                    "cond_legal": bool(st0.validate()["legal"]),
                    "dpos": dpos,
                    "dcell": dcell,
                }
            )
        del model

    orig_lo = next(r for r in rows if r["arm"] == "original" and r["t"] == 0.3)
    orig_hi = next(r for r in rows if r["arm"] == "original" and r["t"] == 0.7)
    cln_lo = next(r for r in rows if r["arm"] == "clean_g" and r["t"] == 0.3)
    cln_hi = next(r for r in rows if r["arm"] == "clean_g" and r["t"] == 0.7)
    ok = (
        orig_lo["dpos"] < 1e-5 and orig_hi["dpos"] < 1e-5
        and orig_lo["dcell"] < 1e-5 and orig_hi["dcell"] < 1e-5
        and (not orig_lo["scf_enabled_runtime"]) and (not orig_hi["scf_enabled_runtime"])
        and (not cln_lo["scf_enabled_runtime"]) and cln_lo["dpos"] < 1e-5 and cln_lo["dcell"] < 1e-5
        and cln_lo["scf_weight"] == 0.0
        and cln_hi["scf_enabled_runtime"] and cln_hi["dpos"] > 1e-4 and cln_hi["scf_weight"] == 1.0
        and cln_hi["dpos"] > 10.0 * max(cln_lo["dpos"], 1e-12)
        and bool(st0.validate()["legal"])
    )
    out = {"event": "smoke_scaled_clean_g", "ok": ok, "rows": rows}
    print(json.dumps(out), flush=True)
    if not ok:
        raise SystemExit("smoke failed: gated Clean-G path mismatch")


if __name__ == "__main__":
    main()
