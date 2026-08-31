#!/usr/bin/env python3
"""Compare Original vs Clean-G. Post-warmup (steps 200-999) is primary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BINS = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
PHASES = {
    "overall": (None, None),
    "steps_0_99": (0, 100),
    "steps_100_199": (100, 200),
    "steps_200_999": (200, None),  # primary
    "last_500": (-500, None),
    "last_200": (-200, None),
}


def _load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _num(row, key, default=0.0):
    """Read a numeric trace flag. Do not use `x or default`: 0.0 is valid."""
    v = row.get(key, default)
    if v is None:
        return float(default)
    return float(v)


def _tbin(t: float) -> str:
    if t < 0.2:
        return "[0.0,0.2)"
    if t < 0.4:
        return "[0.2,0.4)"
    if t < 0.6:
        return "[0.4,0.6)"
    if t < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def _pos(r):
    v = r.get("geom_pos", r.get("pos"))
    return None if v is None else float(v)


def _cell(r):
    v = r.get("geom_cell", r.get("cell"))
    return None if v is None else float(v)


def _pack_metrics(rows):
    return {
        "n": len(rows),
        "geometry_loss": _mean([r.get("geometry_loss") for r in rows]),
        "pos_loss": _mean([_pos(r) for r in rows]),
        "cell_loss": _mean([_cell(r) for r in rows]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--clean-g", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    orig_tr = _load_jsonl(args.original / "training_trace.jsonl")
    cln_tr = _load_jsonl(args.clean_g / "training_trace.jsonl")
    orig_prov = _load_json(args.original / "runtime_provenance.json") or {}
    cln_prov = _load_json(args.clean_g / "runtime_provenance.json") or {}

    n = min(len(orig_tr), len(cln_tr))
    orig_tr, cln_tr = orig_tr[:n], cln_tr[:n]
    aligned = [i for i in range(n) if abs(float(orig_tr[i]["global_t"]) - float(cln_tr[i]["global_t"])) <= 1e-6]
    aligned_set = set(aligned)
    unaligned = [i for i in range(n) if i not in aligned_set]

    def _phase_indices(lo, hi):
        idxs = list(range(n))
        if lo is None and hi is None:
            return idxs
        if lo is not None and lo < 0:
            return idxs[lo:]
        if hi is None:
            return idxs[lo:]
        return idxs[lo:hi]

    def phase_table(require_aligned: bool):
        out = {}
        for name, (lo, hi) in PHASES.items():
            idxs = _phase_indices(lo, hi)
            if require_aligned:
                idxs = [i for i in idxs if i in aligned_set]
            o = [orig_tr[i] for i in idxs]
            c = [cln_tr[i] for i in idxs]
            po, pc = _pack_metrics(o), _pack_metrics(c)

            def dlt(a, b):
                if a is None or b is None:
                    return None
                return b - a

            out[name] = {
                "n": po["n"],
                "geometry_loss": {"Original": po["geometry_loss"], "Clean-G": pc["geometry_loss"], "delta": dlt(po["geometry_loss"], pc["geometry_loss"])},
                "pos_loss": {"Original": po["pos_loss"], "Clean-G": pc["pos_loss"], "delta": dlt(po["pos_loss"], pc["pos_loss"])},
                "cell_loss": {"Original": po["cell_loss"], "Clean-G": pc["cell_loss"], "delta": dlt(po["cell_loss"], pc["cell_loss"])},
            }
        return out

    def tbin_table(rows_o, rows_c, *, paired: bool):
        from collections import defaultdict
        bo, bc = defaultdict(list), defaultdict(list)
        if paired:
            for a, b in zip(rows_o, rows_c):
                k = _tbin(float(a["global_t"]))
                bo[k].append(a)
                bc[k].append(b)
        else:
            for a in rows_o:
                bo[_tbin(float(a["global_t"]))].append(a)
            for b in rows_c:
                bc[_tbin(float(b["global_t"]))].append(b)
        out = {}
        for k in BINS:
            po, pc = _pack_metrics(bo[k]), _pack_metrics(bc[k])

            def dlt(a, b):
                if a is None or b is None:
                    return None
                return b - a

            out[k] = {
                "n_original": po["n"],
                "n_clean_g": pc["n"],
                "geometry_loss": {"Original": po["geometry_loss"], "Clean-G": pc["geometry_loss"], "delta": dlt(po["geometry_loss"], pc["geometry_loss"])},
                "pos_loss": {"Original": po["pos_loss"], "Clean-G": pc["pos_loss"], "delta": dlt(po["pos_loss"], pc["pos_loss"])},
                "cell_loss": {"Original": po["cell_loss"], "Clean-G": pc["cell_loss"], "delta": dlt(po["cell_loss"], pc["cell_loss"])},
            }
        return out

    def sanity():
        def always_zero(rows, key):
            return bool(rows) and all(_num(r, key, 0.0) == 0.0 for r in rows)

        def total_eq_geom(rows):
            return bool(rows) and all(abs(float(r["total_loss"]) - float(r["geometry_loss"])) < 1e-6 for r in rows)

        step0_t = None
        if orig_tr and cln_tr:
            step0_t = abs(float(orig_tr[0]["global_t"]) - float(cln_tr[0]["global_t"])) < 1e-8
        clean_uses_a0 = bool(cln_tr) and all(_num(r, "scf_uses_clean_a0", 0.0) == 1.0 for r in cln_tr)
        clean_eq_a0 = bool(cln_tr) and all(_num(r, "cond_equals_clean_a0", 0.0) == 1.0 for r in cln_tr)
        clean_legal = bool(cln_tr) and all(_num(r, "cond_legal", 0.0) == 1.0 for r in cln_tr)
        # At least some steps where A_t != A_0, proving we are not accidentally only hitting t with no jumps.
        some_at_differs = bool(cln_tr) and any(_num(r, "cond_equals_noisy_At", 1.0) == 0.0 for r in cln_tr)
        return {
            "orig_L_G_L_R_zero": always_zero(orig_tr, "L_G") and always_zero(orig_tr, "L_R"),
            "clean_L_G_L_R_zero": always_zero(cln_tr, "L_G") and always_zero(cln_tr, "L_R"),
            "orig_total_eq_geom": total_eq_geom(orig_tr),
            "clean_total_eq_geom": total_eq_geom(cln_tr),
            "step0_t_aligned": step0_t,
            "clean_scf_uses_a0_every_step": clean_uses_a0,
            "clean_cond_equals_a0_every_step": clean_eq_a0,
            "clean_cond_legal_every_step": clean_legal,
            "clean_sometimes_differs_from_At": some_at_differs,
            "n_paired_t_aligned": len(aligned),
            "n_t_unaligned": len(unaligned),
        }

    paired_all = phase_table(False)
    paired_aligned = phase_table(True)
    post = paired_aligned.get("steps_200_999") or paired_all.get("steps_200_999") or {}
    geom_o = ((post.get("geometry_loss") or {}).get("Original"))
    geom_c = ((post.get("geometry_loss") or {}).get("Clean-G"))
    pos_o = ((post.get("pos_loss") or {}).get("Original"))
    pos_c = ((post.get("pos_loss") or {}).get("Clean-G"))

    def rel(a, b):
        if a is None or b is None or abs(a) < 1e-18:
            return None
        return (b - a) / abs(a)

    geom_rel = rel(geom_o, geom_c)
    pos_rel = rel(pos_o, pos_c)
    # "Clearly better" = post-warmup geom AND pos both down by >3% on t-aligned pairs.
    clearly = (
        geom_rel is not None
        and pos_rel is not None
        and geom_rel < -0.03
        and pos_rel < -0.03
    )
    if clearly:
        diagnosis = "continue G route"
        diagnosis_text = (
            "Exact clean partition has downstream geometry value. Previous noisy-oracle / "
            "learned-G failure is caused by the assignment trajectory or joint-training "
            "formulation rather than lack of useful G information."
        )
    else:
        diagnosis = "stop current G→geometry route"
        diagnosis_text = (
            "Even permanently correct clean assignment does not provide a persistent "
            "geometry benefit under the current GemNet+SCF formulation."
        )

    out = {
        "original": {
            "arm": orig_prov.get("ABLATION_ARM"),
            "geometry_assignment_conditioning": orig_prov.get("GEOMETRY_ASSIGNMENT_CONDITIONING"),
            "train_assignment_heads": orig_prov.get("TRAIN_ASSIGNMENT_HEADS"),
            "clean_g": orig_prov.get("CLEAN_G"),
        },
        "clean_g": {
            "arm": cln_prov.get("ABLATION_ARM"),
            "geometry_assignment_conditioning": cln_prov.get("GEOMETRY_ASSIGNMENT_CONDITIONING"),
            "train_assignment_heads": cln_prov.get("TRAIN_ASSIGNMENT_HEADS"),
            "clean_g": cln_prov.get("CLEAN_G"),
            "construction": cln_prov.get("CLEAN_G_CONSTRUCTION"),
        },
        "sanity": sanity(),
        "paired_all_steps": paired_all,
        "paired_t_aligned": paired_aligned,
        "t_unaligned_n": len(unaligned),
        "t_bins_unpaired": tbin_table(orig_tr, cln_tr, paired=False),
        "t_bins_paired_aligned": tbin_table(
            [orig_tr[i] for i in aligned], [cln_tr[i] for i in aligned], paired=True
        ),
        "t_bins_post_warmup_unpaired": tbin_table(orig_tr[200:], cln_tr[200:], paired=False),
        "primary": "paired_t_aligned.steps_200_999",
        "post_warmup_geom_rel": geom_rel,
        "post_warmup_pos_rel": pos_rel,
        "diagnosis": diagnosis,
        "diagnosis_text": diagnosis_text,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {
                "event": "clean_g_geometry_ablation_comparison",
                "out": str(args.out),
                "sanity": out["sanity"],
                "primary": out["paired_t_aligned"].get("steps_200_999"),
                "diagnosis": diagnosis,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
