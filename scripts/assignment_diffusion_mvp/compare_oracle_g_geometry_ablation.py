#!/usr/bin/env python3
"""Compare Original vs Oracle-G geometry-denoising ablation (no sampling required)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BINS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]", "overall"]


def _load(p: Path, name: str):
    fp = p / name
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def _bin_metric(geom: dict, key: str, field: str):
    row = (geom or {}).get(key) or {}
    return row.get(field)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--oracle-g", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    def pack(d: Path) -> dict:
        geom = _load(d, "geometry_bin_summary.json") or {}
        prov = _load(d, "runtime_provenance.json") or {}
        overall = geom.get("overall") or {}
        return {
            "arm": prov.get("ABLATION_ARM"),
            "geometry_assignment_conditioning": prov.get("GEOMETRY_ASSIGNMENT_CONDITIONING"),
            "train_assignment_heads": prov.get("TRAIN_ASSIGNMENT_HEADS"),
            "oracle_g": prov.get("ORACLE_G"),
            "oracle_g_construction": prov.get("ORACLE_G_CONSTRUCTION"),
            "assignment_loss": prov.get("ASSIGNMENT_LOSS"),
            "geometry_loss_overall": overall.get("geometry_loss_mean"),
            "pos_loss_overall": overall.get("pos_loss_mean"),
            "cell_loss_overall": overall.get("cell_loss_mean"),
            "geometry_bins": geom,
        }

    orig = pack(args.original)
    ora = pack(args.oracle_g)

    def delta(a, b):
        if a is None or b is None:
            return None
        return b - a

    table: dict = {}
    for field, orig_key, ora_key in (
        ("geometry_loss_overall", orig["geometry_loss_overall"], ora["geometry_loss_overall"]),
        ("pos_loss_overall", orig["pos_loss_overall"], ora["pos_loss_overall"]),
        ("cell_loss_overall", orig["cell_loss_overall"], ora["cell_loss_overall"]),
    ):
        table[field] = {
            "Original": orig_key,
            "Oracle-G": ora_key,
            "delta": delta(orig_key, ora_key),
        }
    for b in BINS:
        for metric, field in (
            ("geometry_loss", "geometry_loss_mean"),
            ("pos_loss", "pos_loss_mean"),
            ("cell_loss", "cell_loss_mean"),
        ):
            ov = _bin_metric(orig["geometry_bins"], b, field)
            av = _bin_metric(ora["geometry_bins"], b, field)
            table[f"{metric} {b}"] = {
                "Original": ov,
                "Oracle-G": av,
                "delta": delta(ov, av),
            }

    gl_o = orig["geometry_loss_overall"]
    gl_a = ora["geometry_loss_overall"]
    if gl_o is None or gl_a is None:
        diagnosis = "incomplete (missing geometry_bin_summary)"
    elif gl_a < gl_o * 0.97:
        diagnosis = (
            "Correct assignment information has downstream geometry value; "
            "the previous learned-G failure comes from prediction/joint-training/"
            "conditioning quality rather than the G concept itself."
        )
    elif gl_a > gl_o * 1.03:
        diagnosis = (
            "Even oracle assignment provides no measurable geometry benefit on this "
            "MVP (Oracle-G worse); explicit G conditioning itself is likely not useful "
            "in the current formulation."
        )
    else:
        diagnosis = (
            "Even oracle assignment provides no measurable geometry benefit on this "
            "MVP; explicit G conditioning itself is likely not useful in the current formulation."
        )

    out = {
        "original": orig,
        "oracle_g": ora,
        "table": table,
        "diagnosis": diagnosis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"event": "oracle_g_geometry_ablation_comparison", "out": str(args.out), "table": table, "diagnosis": diagnosis}, indent=2))


if __name__ == "__main__":
    main()
