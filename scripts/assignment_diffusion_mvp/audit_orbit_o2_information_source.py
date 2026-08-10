#!/usr/bin/env python3
"""Frozen O2 checkpoint information-source / ordering-leakage audit (NO TRAINING)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_global_copy_assembly_orbit_o2 import load_setup, move_o2_target, resolve_device
from mattergen.assignment.global_copy_assembly.information_source_audit import (
    ATOM_PERM_SEEDS,
    SHUFFLE_SEEDS,
    NEAR_TIE_TOLS,
    assert_no_oracle_inputs,
    audit_legacy_vs_strict_zero_features,
    classify_audit,
    decode_o2,
    freeze_model,
    permute_atom_indices,
    render_markdown_report,
    shuffle_orbit_candidate_order,
    shuffle_singleton_role_instance_order,
    summarize_seed_metrics,
    unpermute_C,
)
from mattergen.assignment.global_copy_assembly.orbit_targets import build_orbit_aware_target
from mattergen.assignment.global_copy_assembly.orbit_metrics import evaluate_orbit_assembly


def _metric_row(condition: str, decoded: dict) -> dict:
    m = decoded["metrics"]
    sm = decoded.get("singleton_margins") or {}
    om = {}
    if decoded.get("orbit_margins"):
        # first non-singleton orbit
        om = next(iter(decoded["orbit_margins"].values()))
    return {
        "condition": condition,
        "exact_C": m.get("exact_C"),
        "copy_pair_f1": m.get("copy_pair_f1"),
        "ARI": m.get("ARI"),
        "projected_bond_f1": m.get("projected_bond_f1"),
        "singleton_map_gap": sm.get("map_gap"),
        "orbit_map_gap": om.get("attachment_MAP_gap"),
        "singleton_tree_energy": decoded.get("singleton_tree_energy"),
        "cross_copy_false_molecular_edge_rate": m.get("cross_copy_false_molecular_edge_rate"),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="O2 checkpoint (default: prefer final then best under output_dir)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mismatch-sample", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--tie-break-mode",
        type=str,
        default="all",
        choices=["default", "reverse", "random", "all"],
    )
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to run information-source audit without --execute")

    cfg, sample, o2_target, backbone, model, meta = load_setup(args.config)
    out_root = Path(args.output_dir or cfg["output_dir"])
    audit_dir = out_root / "information_source_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint without overwriting existing O2 metrics
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        map_metrics = out_root / "map_evaluation_metrics.json"
        if map_metrics.exists():
            try:
                pref = json.loads(map_metrics.read_text()).get("checkpoint")
                if pref and Path(pref).exists():
                    ckpt_path = Path(pref)
                else:
                    ckpt_path = out_root / "final_checkpoint.pt"
            except Exception:
                ckpt_path = out_root / "final_checkpoint.pt"
        else:
            ckpt_path = out_root / "final_checkpoint.pt"
        if not ckpt_path.exists():
            alt = out_root / "best_checkpoint.pt"
            if alt.exists():
                ckpt_path = alt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"O2 checkpoint missing: {ckpt_path}")

    # Load frozen weights — never train
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = freeze_model(model)
    oracle_flags = assert_no_oracle_inputs(model)

    device = resolve_device(str(args.device or cfg.get("device", "auto")))
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    o2_target = move_o2_target(o2_target, device)
    model = model.to(device)

    # Role assignment used to build o2_target (from artifact)
    # Recover labels from bar_r / singleton sets is incomplete; reload artifact path.
    from train_global_copy_assembly_orbit_o2 import _resolve

    root = ROOT_DIR
    artifact_path = _resolve(
        cfg,
        "predicted_role_artifact_path",
        root / "outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/geometry_only_hard_r.jsonl",
    )
    art = [json.loads(l) for l in artifact_path.read_text().splitlines() if l.strip()][-1]
    role_assignment = torch.tensor(art["role_assignment"], dtype=torch.long, device=device)

    mismatch = None
    mismatch_status = "MISMATCH_SAMPLE_NOT_AVAILABLE"
    if args.mismatch_sample is not None and args.mismatch_sample.exists():
        mismatch = torch.load(args.mismatch_sample, map_location=device, weights_only=False)
        if mismatch["pos"].shape == sample["pos"].shape:
            mismatch_status = "LOADED"
        else:
            mismatch = None
            mismatch_status = "MISMATCH_SAMPLE_INCOMPATIBLE_SHAPE"

    notes: list[str] = []
    condition_table: list[dict] = []
    baseline: dict = {}

    # ---------- Baseline geometry modes ----------
    for mode in ("correct_geometry", "legacy_zero_geometry", "strict_zero_geometry"):
        dec = decode_o2(model, o2_target, backbone, sample, geometry_mode=mode)
        baseline[mode] = {
            **_metric_row(mode, dec),
            "geometry_note": dec["geometry"]["geometry_note"],
            "singleton_margins": dec["singleton_margins"],
            "orbit_margins": _jsonable(dec["orbit_margins"]),
        }
        condition_table.append(_metric_row(mode, dec))
        print(json.dumps({"event": "baseline", "mode": mode, **_metric_row(mode, dec)}), flush=True)

    if mismatch is not None:
        dec = decode_o2(
            model, o2_target, backbone, sample,
            geometry_mode="mismatched_geometry", mismatch_sample=mismatch,
        )
        baseline["mismatched_geometry"] = {
            **_metric_row("mismatched_geometry", dec),
            "geometry_note": dec["geometry"]["geometry_note"],
            "status": "EVALUATED",
            "singleton_margins": dec["singleton_margins"],
            "orbit_margins": _jsonable(dec["orbit_margins"]),
        }
        condition_table.append(_metric_row("mismatched_geometry", dec))
    else:
        baseline["mismatched_geometry"] = {"status": mismatch_status}
        notes.append(f"mismatched_geometry: {mismatch_status}")

    # ---------- Gate D: zero geometry feature audit ----------
    zero_feat = audit_legacy_vs_strict_zero_features(
        {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sample.items()}
    )
    (audit_dir / "zero_geometry_feature_audit.json").write_text(json.dumps(zero_feat, indent=2))

    # ---------- Gate A: atom permutation ----------
    atom_rows = []
    for geom in ("correct_geometry", "strict_zero_geometry", "legacy_zero_geometry"):
        for seed in ATOM_PERM_SEEDS:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            sigma = torch.randperm(int(sample["N"]), generator=g).to(device)
            sample_p, roles_p, inv = permute_atom_indices(sample, role_assignment, sigma)
            # Rebuild O2 target on permuted roles/copy (supervision metrics only for C0')
            anchor = backbone.singleton_roles[backbone.tree.root]
            o2_p = build_orbit_aware_target(
                roles_p.cpu(),
                sample_p["copy"].cpu(),
                partition=o2_target.partition,
                K=int(sample["Z"]),
                anchor_role=anchor,
            )
            o2_p = move_o2_target(o2_p, device)
            dec = decode_o2(model, o2_p, backbone, sample_p, geometry_mode=geom)
            C_back = unpermute_C(dec["C"], inv)
            # Metrics vs original C0
            metrics_back = evaluate_orbit_assembly(
                G=dec["G"][inv],  # also unpermute G rows
                C=C_back,
                copy=sample["copy"],
                bar_r=o2_target.bar_r,
                partition=o2_target.partition,
                role_assignment_for_projection=sample["role"],
                role_edge_index=sample["role_edge_index"],
                role_bond_type=sample["role_bond_type"],
                M=int(sample["M"]),
            )
            # invariance: compare unpermuted C to baseline C of same geometry
            base_C = decode_o2(model, o2_target, backbone, sample, geometry_mode=geom)["C"]
            invariant = bool(torch.equal(C_back.bool(), base_C.bool()))
            row = {
                "geometry_mode": geom,
                "seed": seed,
                "exact_C": metrics_back["exact_C"],
                "copy_pair_f1": metrics_back["copy_pair_f1"],
                "ARI": metrics_back["ARI"],
                "projected_bond_f1": metrics_back["projected_bond_f1"],
                "prediction_invariant_after_undo": invariant,
                "singleton_map_gap": (dec.get("singleton_margins") or {}).get("map_gap"),
                "orbit_map_gap": (
                    next(iter(dec["orbit_margins"].values())).get("attachment_MAP_gap")
                    if dec.get("orbit_margins")
                    else None
                ),
            }
            atom_rows.append(row)
            print(json.dumps({"event": "atom_perm", **row}), flush=True)

    with (audit_dir / "atom_permutation_results.jsonl").open("w") as f:
        for row in atom_rows:
            f.write(json.dumps(row) + "\n")

    def _rate(rows, geom, key="exact_C"):
        sub = [r for r in rows if r["geometry_mode"] == geom]
        if not sub:
            return None
        return sum(1.0 if r[key] else 0.0 for r in sub) / len(sub)

    atom_summary = {
        "correct_exact_C_rate": _rate(atom_rows, "correct_geometry"),
        "strict_zero_exact_C_rate": _rate(atom_rows, "strict_zero_geometry"),
        "legacy_zero_exact_C_rate": _rate(atom_rows, "legacy_zero_geometry"),
        "correct_invariance_rate": _rate(atom_rows, "correct_geometry", "prediction_invariant_after_undo"),
        "strict_zero_invariance_rate": _rate(atom_rows, "strict_zero_geometry", "prediction_invariant_after_undo"),
        "by_geometry": {
            g: summarize_seed_metrics([r for r in atom_rows if r["geometry_mode"] == g])
            for g in ("correct_geometry", "strict_zero_geometry", "legacy_zero_geometry")
        },
    }

    # Add representative atom-shuffle rows to condition table (seed=0)
    for geom, label in (
        ("correct_geometry", "correct + atom shuffle (seed0)"),
        ("strict_zero_geometry", "strict zero + atom shuffle (seed0)"),
    ):
        hit = next((r for r in atom_rows if r["geometry_mode"] == geom and r["seed"] == 0), None)
        if hit:
            condition_table.append(
                {
                    "condition": label,
                    "exact_C": hit["exact_C"],
                    "copy_pair_f1": hit["copy_pair_f1"],
                    "ARI": hit["ARI"],
                    "projected_bond_f1": hit["projected_bond_f1"],
                    "singleton_map_gap": hit.get("singleton_map_gap"),
                    "orbit_map_gap": hit.get("orbit_map_gap"),
                }
            )

    # ---------- Gate B: singleton role-instance order shuffle ----------
    role_rows = []
    for geom in ("correct_geometry", "strict_zero_geometry"):
        for seed in SHUFFLE_SEEDS:
            o2_s = shuffle_singleton_role_instance_order(o2_target, seed=seed)
            dec = decode_o2(model, o2_s, backbone, sample, geometry_mode=geom)
            base = decode_o2(model, o2_target, backbone, sample, geometry_mode=geom)
            invar = bool(torch.equal(dec["C"].bool(), base["C"].bool()))
            m = dec["metrics"]
            row = {
                "geometry_mode": geom,
                "seed": seed,
                "exact_C": m["exact_C"],
                "copy_pair_f1": m["copy_pair_f1"],
                "ARI": m["ARI"],
                "projected_bond_f1": m["projected_bond_f1"],
                "prediction_invariant_to_q_shuffle": invar,
                "singleton_map_gap": (dec.get("singleton_margins") or {}).get("map_gap"),
                "orbit_map_gap": (
                    next(iter(dec["orbit_margins"].values())).get("attachment_MAP_gap")
                    if dec.get("orbit_margins")
                    else None
                ),
                "MAP_score_singleton": dec.get("singleton_tree_energy"),
            }
            role_rows.append(row)
            print(json.dumps({"event": "role_order_shuffle", **row}), flush=True)
    with (audit_dir / "role_instance_shuffle_results.jsonl").open("w") as f:
        for row in role_rows:
            f.write(json.dumps(row) + "\n")
    role_summary = {
        "correct_exact_C_rate": _rate(role_rows, "correct_geometry"),
        "zero_exact_C_rate": _rate(role_rows, "strict_zero_geometry"),
        "correct_invariance_rate": _rate(role_rows, "correct_geometry", "prediction_invariant_to_q_shuffle"),
        "zero_invariance_rate": _rate(role_rows, "strict_zero_geometry", "prediction_invariant_to_q_shuffle"),
        "by_geometry": {
            g: summarize_seed_metrics([r for r in role_rows if r["geometry_mode"] == g])
            for g in ("correct_geometry", "strict_zero_geometry")
        },
    }
    for geom, label in (
        ("correct_geometry", "correct + role-order shuffle (seed0)"),
        ("strict_zero_geometry", "strict zero + role-order shuffle (seed0)"),
    ):
        hit = next((r for r in role_rows if r["geometry_mode"] == geom and r["seed"] == 0), None)
        if hit:
            condition_table.append(
                {
                    "condition": label,
                    "exact_C": hit["exact_C"],
                    "copy_pair_f1": hit["copy_pair_f1"],
                    "ARI": hit["ARI"],
                    "projected_bond_f1": hit["projected_bond_f1"],
                    "singleton_map_gap": hit.get("singleton_map_gap"),
                    "orbit_map_gap": hit.get("orbit_map_gap"),
                }
            )

    # ---------- Gate C: orbit candidate order shuffle ----------
    orbit_rows = []
    for geom in ("correct_geometry", "strict_zero_geometry"):
        for seed in SHUFFLE_SEEDS:
            o2_o = shuffle_orbit_candidate_order(o2_target, seed=seed)
            dec = decode_o2(model, o2_o, backbone, sample, geometry_mode=geom)
            base = decode_o2(model, o2_target, backbone, sample, geometry_mode=geom)
            invar = bool(torch.equal(dec["C"].bool(), base["C"].bool()))
            m = dec["metrics"]
            row = {
                "geometry_mode": geom,
                "seed": seed,
                "exact_C": m["exact_C"],
                "copy_pair_f1": m["copy_pair_f1"],
                "prediction_invariant_to_candidate_order": invar,
                "orbit_map_gap": (
                    next(iter(dec["orbit_margins"].values())).get("attachment_MAP_gap")
                    if dec.get("orbit_margins")
                    else None
                ),
            }
            orbit_rows.append(row)
            print(json.dumps({"event": "orbit_candidate_shuffle", **row}), flush=True)
    with (audit_dir / "orbit_candidate_shuffle_results.jsonl").open("w") as f:
        for row in orbit_rows:
            f.write(json.dumps(row) + "\n")

    # ---------- Gate E: map margins (already in baseline; write dedicated file) ----------
    map_margin_results = {
        mode: {
            "singleton_margins": baseline[mode].get("singleton_margins"),
            "orbit_margins": baseline[mode].get("orbit_margins"),
            "exact_C": baseline[mode].get("exact_C"),
            "singleton_MAP_gap": baseline[mode].get("singleton_map_gap"),
            "orbit_MAP_gap": baseline[mode].get("orbit_map_gap"),
        }
        for mode in ("correct_geometry", "legacy_zero_geometry", "strict_zero_geometry")
        if mode in baseline
    }
    # multi-tol near-tie sensitivity on strict zero
    sensitivity = {}
    for tol in NEAR_TIE_TOLS:
        dec = decode_o2(
            model, o2_target, backbone, sample,
            geometry_mode="strict_zero_geometry", near_tie_tol=tol,
        )
        om = next(iter(dec["orbit_margins"].values())) if dec.get("orbit_margins") else {}
        sensitivity[str(tol)] = {
            "number_of_near_ties": om.get("number_of_near_ties"),
            "attachment_MAP_gap": om.get("attachment_MAP_gap"),
            "exact_C": dec["metrics"]["exact_C"],
        }
    map_margin_results["near_tie_sensitivity_strict_zero"] = sensitivity
    (audit_dir / "map_margin_results.json").write_text(json.dumps(_jsonable(map_margin_results), indent=2))

    # ---------- Gate F: tie-break perturbation ----------
    tie_rows = []
    modes = ["default", "reverse", "random"] if args.tie_break_mode == "all" else [args.tie_break_mode]
    for geom in ("correct_geometry", "strict_zero_geometry"):
        for mode in modes:
            seeds = (0, 1, 2, 3) if mode == "random" else (0,)
            for seed in seeds:
                dec = decode_o2(
                    model,
                    o2_target,
                    backbone,
                    sample,
                    geometry_mode=geom,
                    attachment_pair_order=mode,
                    attachment_tie_seed=seed,
                    near_tie_tol=1e-6,
                )
                m = dec["metrics"]
                row = {
                    "geometry_mode": geom,
                    "tie_break_mode": mode,
                    "seed": seed,
                    "exact_C": m["exact_C"],
                    "copy_pair_f1": m["copy_pair_f1"],
                    "map_pairs": _jsonable(dec.get("orbit_attachments")),
                }
                tie_rows.append(row)
                print(json.dumps({"event": "tie_break", **{k: v for k, v in row.items() if k != "map_pairs"}}), flush=True)
    with (audit_dir / "tie_break_results.jsonl").open("w") as f:
        for row in tie_rows:
            f.write(json.dumps(row) + "\n")

    def _tie_varies(geom: str) -> bool:
        sub = [r for r in tie_rows if r["geometry_mode"] == geom]
        if len(sub) < 2:
            return False
        # compare exact_C and map_pairs
        ref = json.dumps(sub[0].get("map_pairs"), sort_keys=True)
        return any(json.dumps(r.get("map_pairs"), sort_keys=True) != ref or r["exact_C"] != sub[0]["exact_C"] for r in sub[1:])

    tie_summary = {
        "correct_changes_with_tie_break": _tie_varies("correct_geometry"),
        "zero_changes_with_tie_break": _tie_varies("strict_zero_geometry"),
        "rows": len(tie_rows),
    }

    # ---------- Mismatch file ----------
    (audit_dir / "mismatched_geometry_results.json").write_text(
        json.dumps(_jsonable(baseline.get("mismatched_geometry", {})), indent=2)
    )

    # ---------- Summary / classification ----------
    summary = {
        "checkpoint": str(ckpt_path),
        "role_source": meta.get("role_source"),
        "oracle_input_audit": oracle_flags,
        "baseline": _jsonable(baseline),
        "atom_permutation": atom_summary,
        "role_instance_shuffle": role_summary,
        "orbit_candidate_shuffle": {
            "correct_invariance_rate": _rate(orbit_rows, "correct_geometry", "prediction_invariant_to_candidate_order"),
            "zero_invariance_rate": _rate(orbit_rows, "strict_zero_geometry", "prediction_invariant_to_candidate_order"),
            "correct_exact_C_rate": _rate(orbit_rows, "correct_geometry"),
            "zero_exact_C_rate": _rate(orbit_rows, "strict_zero_geometry"),
        },
        "zero_geometry_feature_audit": zero_feat,
        "map_margin_results": _jsonable(map_margin_results),
        "tie_break": tie_summary,
        "condition_table": condition_table,
        "notes": notes,
        "no_retraining": True,
        "model_eval_mode": True,
    }
    summary["classifications"] = classify_audit(summary)
    (audit_dir / "audit_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    (audit_dir / "information_source_audit.md").write_text(render_markdown_report(summary))
    print(json.dumps({"event": "audit_done", "classifications": summary["classifications"], "dir": str(audit_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
