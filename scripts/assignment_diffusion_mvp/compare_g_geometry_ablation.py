#!/usr/bin/env python3
"""Compare Original vs G-conditioned geometry ablation outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path, name: str):
    fp = p / name
    if not fp.exists():
        return None
    if fp.suffix == ".jsonl":
        return [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    return json.loads(fp.read_text())


def _mean(rows, key):
    xs = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(xs) / len(xs) if xs else None


def _rate(rows, key):
    xs = [bool(r.get(key)) for r in rows]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--g-conditioned", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    def pack(d: Path) -> dict:
        geom = _load(d, "geometry_bin_summary.json") or {}
        gteach = _load(d, "g_teacher_bin_summary.json") or {}
        samples = _load(d, "sample_summary.json") or []
        prov = _load(d, "runtime_provenance.json") or {}
        return {
            "arm": prov.get("ABLATION_ARM"),
            "geometry_assignment_conditioning": prov.get("GEOMETRY_ASSIGNMENT_CONDITIONING"),
            "geometry_loss_overall": (geom.get("overall") or {}).get("geometry_loss_mean"),
            "geometry_bins": geom,
            "delta_CE_G": (gteach.get("all") or {}).get("delta_teacher_CE"),
            "P_beneficial": (gteach.get("all") or {}).get("P_beneficial"),
            "sample_n": len(samples),
            "no_clash_rate": _rate(samples, "no_clash"),
            "valid_cell_rate": _rate(samples, "valid_cell"),
            "pass_basic_rate": _rate(samples, "pass_basic"),
            "min_dist_mean": _mean(samples, "min_dist"),
            "density_ratio_mean": _mean(samples, "density_ratio"),
            "volume_ratio_mean": _mean(samples, "volume_ratio"),
            "inter_copy_min_dist_mean": _mean(samples, "inter_copy_min_dist"),
        }

    orig = pack(args.original)
    garm = pack(args.g_conditioned)

    def delta(a, b):
        if a is None or b is None:
            return None
        return b - a

    table = {
        "geometry_loss_overall": {
            "Original": orig["geometry_loss_overall"],
            "G-conditioned": garm["geometry_loss_overall"],
            "delta": delta(orig["geometry_loss_overall"], garm["geometry_loss_overall"]),
        },
        "no_clash_rate": {
            "Original": orig["no_clash_rate"],
            "G-conditioned": garm["no_clash_rate"],
            "delta": delta(orig["no_clash_rate"], garm["no_clash_rate"]),
        },
        "valid_cell_rate": {
            "Original": orig["valid_cell_rate"],
            "G-conditioned": garm["valid_cell_rate"],
            "delta": delta(orig["valid_cell_rate"], garm["valid_cell_rate"]),
        },
        "min_dist_mean": {
            "Original": orig["min_dist_mean"],
            "G-conditioned": garm["min_dist_mean"],
            "delta": delta(orig["min_dist_mean"], garm["min_dist_mean"]),
        },
        "density_ratio_mean": {
            "Original": orig["density_ratio_mean"],
            "G-conditioned": garm["density_ratio_mean"],
            "delta": delta(orig["density_ratio_mean"], garm["density_ratio_mean"]),
        },
        "delta_CE_G": {
            "Original": orig["delta_CE_G"],
            "G-conditioned": garm["delta_CE_G"],
            "delta": delta(orig["delta_CE_G"], garm["delta_CE_G"]),
        },
        "P_beneficial": {
            "Original": orig["P_beneficial"],
            "G-conditioned": garm["P_beneficial"],
            "delta": delta(orig["P_beneficial"], garm["P_beneficial"]),
        },
    }
    out = {"original": orig, "g_conditioned": garm, "table": table}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"event": "g_geometry_ablation_comparison", "out": str(args.out), "table": table}, indent=2))


if __name__ == "__main__":
    main()
