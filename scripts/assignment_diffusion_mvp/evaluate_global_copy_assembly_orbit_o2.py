#!/usr/bin/env python3
"""Independent MAP evaluation for orbit-aware O2 assembly."""
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
from mattergen.assignment.global_copy_assembly.orbit_metrics import evaluate_orbit_assembly


def geometry_for_mode(sample, mode: str, mismatch_sample=None):
    if mode == "correct_geometry":
        return sample["pos"]
    if mode == "zero_geometry":
        return torch.zeros_like(sample["pos"])
    if mode == "mismatched_geometry":
        if mismatch_sample is None or mismatch_sample["pos"].shape != sample["pos"].shape:
            raise ValueError("mismatched_geometry requires same-shape second sample")
        return mismatch_sample["pos"]
    raise ValueError(mode)


@torch.no_grad()
def evaluate_once(model, o2_target, backbone, sample, mode, mismatch_sample=None):
    local = dict(sample)
    local["pos"] = geometry_for_mode(sample, mode, mismatch_sample)
    decoded = model.map_decode(
        o2_target=o2_target,
        backbone=backbone,
        z=local["z"],
        frac=local["pos"],
        cell=local["cell"],
        role_z=local["role_z"],
        role_edge_index=local["role_edge_index"],
        role_bond_type=local["role_bond_type"],
    )
    metrics = evaluate_orbit_assembly(
        G=decoded["G"],
        C=decoded["C"],
        copy=sample["copy"],
        bar_r=o2_target.bar_r,
        partition=o2_target.partition,
        role_assignment_for_projection=sample["role"],
        role_edge_index=sample["role_edge_index"],
        role_bond_type=sample["role_bond_type"],
        M=int(sample["M"]),
    )
    return {
        "status": decoded["status"],
        "geometry_mode": mode,
        **metrics,
        "singleton_tree_energy": float(decoded["singleton_tree_energy"]),
        "orbit_attachments": decoded["orbit_attachments"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mismatch-sample", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to evaluate without --execute")

    cfg, sample, o2_target, backbone, model, meta = load_setup(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = resolve_device(str(cfg.get("device", "auto")))
    sample_dev = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()
    }
    o2_target = move_o2_target(o2_target, device)
    model = model.to(device)

    results = {
        mode: evaluate_once(model, o2_target, backbone, sample_dev, mode)
        for mode in ("correct_geometry", "zero_geometry")
    }
    if args.mismatch_sample is None:
        results["mismatched_geometry"] = {"status": "REQUIRES_EXPLICIT_MISMATCH_SAMPLE"}
    else:
        mismatch = torch.load(args.mismatch_sample, map_location=device, weights_only=False)
        results["mismatched_geometry"] = evaluate_once(
            model, o2_target, backbone, sample_dev, "mismatched_geometry", mismatch
        )

    available = [v["projected_bond_f1"] for v in results.values() if "projected_bond_f1" in v]
    results["delta_projected_bond_f1"] = (
        results["correct_geometry"]["projected_bond_f1"] - max(available[1:]) if len(available) > 1 else None
    )

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    correct = results["correct_geometry"]
    map_evaluation = {
        "checkpoint": str(args.checkpoint),
        "status": correct["status"],
        "role_source": meta["role_source"],
        "exact_C": correct["exact_C"],
        "copy_pair_f1": correct["copy_pair_f1"],
        "ARI": correct["ARI"],
        "orbit_copy_capacity_valid": correct["orbit_copy_capacity_valid"],
        "projected_bond_f1": correct["projected_bond_f1"],
        "projected_molecular_graph_exact": correct["projected_molecular_graph_exact"],
        "complete_copy_rate": correct["complete_copy_rate"],
        "copy_graph_isomorphism_rate": correct["copy_graph_isomorphism_rate"],
        "cross_copy_false_molecular_edge_rate": correct["cross_copy_false_molecular_edge_rate"],
        "meta": meta,
    }
    (output / "map_evaluation_metrics.json").write_text(json.dumps(map_evaluation, indent=2))
    (output / "evaluation_metrics.json").write_text(json.dumps(results, indent=2))
    (output / "condition_ablation.json").write_text(json.dumps(results, indent=2))
    (output / "per_sample_map_results.jsonl").write_text(
        json.dumps({"geometry_mode": "correct_geometry", "exact_C": correct["exact_C"]}) + "\n"
    )
    (output / "global_copy_assembly_orbit_o2_report.md").write_text(
        "# Orbit-aware O2 evaluation\n\n"
        f"Status: `{correct['status']}`\n\n"
        f"- exact C: {correct['exact_C']}\n"
        f"- copy-pair F1: {correct['copy_pair_f1']}\n"
        f"- ARI: {correct['ARI']}\n"
        f"- orbit-copy capacity valid: {correct['orbit_copy_capacity_valid']}\n"
        f"- projected-bond F1: {correct['projected_bond_f1']}\n"
        f"- cross-copy false molecular-edge rate: {correct['cross_copy_false_molecular_edge_rate']}\n"
        f"- role source: {meta['role_source']}\n"
        f"- bar R shape: {meta['bar_R_shape']}\n"
    )
    print(json.dumps(map_evaluation, sort_keys=True))


if __name__ == "__main__":
    main()
