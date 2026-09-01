#!/usr/bin/env python3
"""Compare Original vs gated Clean-G: denoising + sampling + per-crystal win rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return json.loads(p.read_text())


def _num(x):
    """Keep legitimate 0.0; only drop None / NaN."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def _mean(xs):
    xs = [_num(x) for x in xs]
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None and x == x)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _bootstrap_ci(xs, *, n_boot=1000, seed=17):
    xs = [float(x) for x in xs if x is not None and x == x]
    if not xs:
        return None
    rng = __import__("random").Random(seed)
    means = []
    n = len(xs)
    for _ in range(n_boot):
        draw = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return {"low": lo, "high": hi, "n_crystals": n, "n_boot": n_boot}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--clean-g", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    def pack_denoise(d: Path):
        s = _load(d / "denoising_val_summary.json") or {}
        tr = _load(d / "training_trace.jsonl") or []
        n = len(tr)
        post = tr[n // 5 :] if n else []  # skip first 20%
        last = tr[n // 2 :] if n else []
        return {
            "val": s,
            "train_overall_geom": _mean([r.get("geometry_loss") for r in tr]),
            "train_post_warmup_geom": _mean([r.get("geometry_loss") for r in post]),
            "train_last50_geom": _mean([r.get("geometry_loss") for r in last]),
            "L_G_zero": all(float(r.get("L_G", 0) or 0) == 0 for r in tr) if tr else None,
            "L_R_zero": all(float(r.get("L_R", 0) or 0) == 0 for r in tr) if tr else None,
        }

    def pack_sample(d: Path):
        rows = _load(d / "sample_summary.json") or []
        return rows

    od, cd = pack_denoise(args.original), pack_denoise(args.clean_g)
    osamp, csamp = pack_sample(args.original), pack_sample(args.clean_g)
    PAIR_KEYS = [
        ("d_E_clash", "final_E_clash", "lower"),
        ("d_inter_min", "final_inter_copy_min_dist", "higher"),
        ("d_inter_p5", "final_inter_copy_p5", "higher"),
        ("d_overlap", "final_copy_overlap_max", "lower"),
        ("d_com_min", "final_copy_com_min", "higher"),
        ("d_radius", "final_copy_radius_mean", "either"),
        ("d_min_dist", "min_dist", "higher"),
        ("d_rec_clash", "recovery_t_clash", "higher"),
        ("d_rec_overlap", "recovery_t_overlap", "higher"),
        ("d_traj_E_clash", "traj_mean_E_clash", "lower"),
        ("d_traj_inter_min", "traj_mean_inter_copy_min_dist", "higher"),
        ("d_traj_overlap", "traj_mean_copy_overlap_max", "lower"),
        ("d_traj_com", "traj_mean_copy_com_min", "higher"),
    ]

    cmap = {(r["id"], r["traj_index"]): r for r in csamp}
    traj_deltas = []
    for r in osamp:
        q = cmap.get((r["id"], r["traj_index"]))
        if not q:
            continue
        rec = {
            "id": r["id"],
            "traj_index": r["traj_index"],
            "orig_no_clash": bool(r.get("no_clash")),
            "clean_no_clash": bool(q.get("no_clash")),
            "K": r.get("K"),
            "N": r.get("N"),
        }
        for dkey, src, _ in PAIR_KEYS:
            qv, ov = _num(q.get(src)), _num(r.get(src))
            if src == "final_inter_copy_min_dist":
                qv = qv if qv is not None else _num(q.get("inter_copy_min_dist"))
                ov = ov if ov is not None else _num(r.get("inter_copy_min_dist"))
            rec[dkey] = (qv - ov) if qv is not None and ov is not None else None
        traj_deltas.append(rec)
    # Crystal is the independent unit: average trajectories per id first.
    by_id = {}
    for d in traj_deltas:
        by_id.setdefault(d["id"], []).append(d)

    def _avg(rows, key):
        return _mean([r[key] for r in rows])

    crystals = []
    for cid, rows in by_id.items():
        rec = {
            "id": cid,
            "n_traj": len(rows),
            "K": rows[0].get("K"),
            "N": rows[0].get("N"),
        }
        for dkey, _src, _dir in PAIR_KEYS:
            rec[dkey] = _avg(rows, dkey)
        crystals.append(rec)

    def win_rate(key, better="lower"):
        xs = [d[key] for d in crystals if d[key] is not None]
        if not xs:
            return None
        if better == "lower":
            return sum(1 for x in xs if x < 0) / len(xs)
        return sum(1 for x in xs if x > 0) / len(xs)

    val_o = (od.get("val") or {}).get("overall") or {}
    val_c = (cd.get("val") or {}).get("overall") or {}
    geom_o, geom_c = val_o.get("geometry_loss"), val_c.get("geometry_loss")
    clash_wr = win_rate("d_E_clash", "lower")
    sep_wr = win_rate("d_inter_min", "higher")
    denoise_close = (
        geom_o is None or geom_c is None or abs(geom_c - geom_o) / max(abs(geom_o), 1e-8) < 0.03
    )
    samp_close = (clash_wr is None or abs(clash_wr - 0.5) < 0.1) and (sep_wr is None or abs(sep_wr - 0.5) < 0.1)
    if (not denoise_close) and clash_wr is not None and clash_wr > 0.6 and sep_wr is not None and sep_wr > 0.55:
        diagnosis = "continue"
        text = (
            "Exact assignment has downstream value mainly through improved reverse geometry dynamics."
        )
    elif denoise_close and samp_close:
        diagnosis = "stop"
        text = (
            "Even exact clean molecular-copy assignment, restricted to the high-noise regime, "
            "does not provide meaningful geometry value under the current GemNet+SCF formulation."
        )
    else:
        diagnosis = "stop"
        text = (
            "Held-out denoising and/or sampling do not show a stable Clean-G advantage; "
            "stop the current assignment diffusion + SCF geometry-conditioning route."
        )
    conv_c = _load(args.clean_g / "convergence.json") or {}
    if conv_c.get("still_improving_at_8k"):
        diagnosis = "inconclusive due to insufficient convergence"
        text = (
            "Clean-G validation geometry_loss still dropped >3% from 4k to 8k. "
            "Do not stop or continue; both arms used the same 8000-step budget."
        )

    out = {
        "original": od,
        "clean_g": cd,
        "n_paired_traj": len(traj_deltas),
        "n_paired_crystals": len(crystals),
        "unit": "crystal (trajectories averaged first)",
        "win_rate": {
            dkey: win_rate(dkey, "lower" if direction == "lower" else "higher")
            for dkey, _src, direction in PAIR_KEYS
            if direction != "either"
        },
        "paired": {
            k: {
                "mean": _mean([d[k] for d in crystals]),
                "median": _median([d[k] for d in crystals]),
                "ci95": _bootstrap_ci([d[k] for d in crystals]),
            }
            for k, _src, _dir in PAIR_KEYS
        },
        "per_crystal": crystals,
        "diagnosis": diagnosis,
        "diagnosis_text": text,
        "convergence_clean_g": conv_c,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"event": "scaled_clean_g_comparison", "diagnosis": diagnosis, "n_paired_crystals": len(crystals), "win_rate": out["win_rate"]}, indent=2))


if __name__ == "__main__":
    main()
