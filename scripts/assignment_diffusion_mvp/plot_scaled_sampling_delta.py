#!/usr/bin/env python3
"""Plot crystal-averaged Δ(t) = Clean-G − Original sampling metrics vs reverse t."""
from __future__ import annotations

import io
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class _Holder:
    def __init__(self, *a, **k):
        pass

    def append(self, x):
        pass


def load_pt(path: Path):
    zf = zipfile.ZipFile(path)
    prefix = zf.namelist()[0].split("/")[0] + "/"

    class U(pickle.Unpickler):
        def persistent_load(self, pid):
            if isinstance(pid, tuple) and pid[0] == "storage":
                raw = zf.read(f"{prefix}data/{pid[2]}")
                dt = np.int64 if ("Long" in str(pid[1]) or "Int" in str(pid[1])) else np.float32
                return np.frombuffer(raw, dtype=dt)
            return pid

        def find_class(self, module, name):
            if module.startswith("torch"):
                if name in ("_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_parameter"):
                    def rebuild(storage, storage_offset, size, stride, *rest):
                        n = int(np.prod(size)) if size else 0
                        view = np.asarray(storage)[storage_offset : storage_offset + max(n, 1)]
                        try:
                            return view.reshape(size)
                        except Exception:
                            return view

                    return rebuild
                return _Holder
            return super().find_class(module, name)

    return U(io.BytesIO(zf.read(f"{prefix}data.pkl"))).load()


KEYS = ("E_clash", "inter_copy_min_dist", "copy_overlap_max")


def collect(arm_dir: Path):
    by_id = defaultdict(list)
    for p in sorted(arm_dir.glob("traj_*_*.pt")):
        obj = load_pt(p)
        series = {}
        for s in obj["snapshots"]:
            t = round(float(s["t"]), 5)
            series[t] = {k: float(s[k]) if s.get(k) == s.get(k) else float("nan") for k in KEYS}
        by_id[obj["row"]["id"]].append(series)
    times = None
    crystals = {}
    for cid, trajs in by_id.items():
        ts = sorted(trajs[0].keys(), reverse=True)
        times = times or ts
        rec = {}
        for t in times:
            rec[t] = {}
            for k in KEYS:
                xs = [tr[t][k] for tr in trajs if t in tr and tr[t][k] == tr[t][k]]
                rec[t][k] = float(np.mean(xs)) if xs else float("nan")
        crystals[cid] = rec
    return times, crystals


def bootstrap_ci(xs, n_boot=1000, seed=17):
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    n = len(xs)
    means = [float(np.mean(rng.choice(xs, size=n, replace=True))) for _ in range(n_boot)]
    means.sort()
    return means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--clean-g", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    t_grid, orig_c = collect(args.original)
    _, cln_c = collect(args.clean_g)
    ids = sorted(set(orig_c) & set(cln_c))
    deltas = {k: [] for k in KEYS}
    for t in t_grid:
        for k in KEYS:
            xs = [cln_c[i][t][k] - orig_c[i][t][k] for i in ids]
            mu = float(np.mean(xs))
            lo, hi = bootstrap_ci(xs)
            p25, p75 = float(np.percentile(xs, 25)), float(np.percentile(xs, 75))
            deltas[k].append({"t": t, "mean": mu, "ci_low": lo, "ci_high": hi, "p25": p25, "p75": p75, "n": len(xs)})

    high = [d for d in deltas["E_clash"] if d["t"] >= 0.5]
    low = [d for d in deltas["E_clash"] if d["t"] < 0.5]
    note = {
        "n_crystals": len(ids),
        "t": t_grid,
        "E_clash_mean_t_ge_0.5": float(np.mean([d["mean"] for d in high])) if high else None,
        "E_clash_mean_t_lt_0.5": float(np.mean([d["mean"] for d in low])) if low else None,
        "E_clash_final": deltas["E_clash"][-1]["mean"] if deltas["E_clash"] else None,
        "inter_min_final": deltas["inter_copy_min_dist"][-1]["mean"] if deltas["inter_copy_min_dist"] else None,
        "curve": deltas,
    }
    hi_neg = high and all(d["ci_high"] < 0 for d in high)
    lo_kept = low and all(d["ci_high"] < 0 for d in low)
    if high and high[0]["mean"] < 0 and deltas["E_clash"][-1]["mean"] < 0:
        note["curve_conclusion"] = (
            "Clean-G lowers clash already in the SCF-on region (t>=0.5) and the "
            "advantage is still present after the gate turns off (t<0.5)."
            if (np.mean([d["mean"] for d in high]) < 0 and np.mean([d["mean"] for d in low]) < 0)
            else "Clash delta sign changes across the gate; see curve."
        )
    else:
        note["curve_conclusion"] = "No persistent Clean-G clash advantage across reverse t."
    note["scf_on_all_ci_better"] = bool(hi_neg)
    note["scf_off_all_ci_better"] = bool(lo_kept)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / (args.out.stem + "_delta.json")).write_text(json.dumps(note, indent=2))

    # SVG
    W, H = 1000, 780
    pad_l, pad_r, pad_t, pad_b = 70, 24, 58, 48
    panels = [
        ("E_clash", "Δ E_clash(t)  (Clean-G − Original)", "negative = Clean-G less clash"),
        ("inter_copy_min_dist", "Δ inter-copy min dist(t)", "positive = Clean-G more separated"),
        ("copy_overlap_max", "Δ copy overlap max(t)", "negative = Clean-G less overlap"),
    ]
    gh = (H - pad_t - pad_b) / 3

    def xmap(t, x0, x1):
        return x0 + (1.0 - t) * (x1 - x0)

    def ymap(v, y0, y1, vmin, vmax):
        return y1 - (v - vmin) / max(vmax - vmin, 1e-12) * (y1 - y0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="26" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="17" font-weight="700" fill="#222">Paired sampling Δ(t): Clean-G − Original</text>',
        f'<text x="500" y="44" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#666">{len(ids)} crystals, trajs averaged first. Band = crystal-level bootstrap 95% CI. Blue dashed = SCF gate t=0.5.</text>',
    ]
    col = "#1f77b4"
    for i, (key, title, hint) in enumerate(panels):
        y0 = pad_t + i * gh + 10
        y1 = y0 + gh - 36
        x0, x1 = pad_l, W - pad_r
        rows = deltas[key]
        vals = [d["ci_low"] for d in rows] + [d["ci_high"] for d in rows] + [0.0]
        vmin, vmax = min(vals), max(vals)
        pad = 0.08 * (vmax - vmin if vmax > vmin else 1)
        vmin, vmax = vmin - pad, vmax + pad
        parts.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#fafafa" stroke="#ddd"/>')
        # SCF regions
        xg = xmap(0.5, x0, x1)
        parts.append(f'<rect x="{x0}" y="{y0}" width="{xg-x0}" height="{y1-y0}" fill="#2ca02c" opacity="0.06"/>')
        parts.append(f'<rect x="{xg}" y="{y0}" width="{x1-xg}" height="{y1-y0}" fill="#999" opacity="0.06"/>')
        parts.append(f'<text x="{x0+8}" y="{y0+14}" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" fill="#2a7">t≥0.5 SCF on</text>')
        parts.append(f'<text x="{x1-8}" y="{y0+14}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="10" fill="#666">t&lt;0.5 SCF off</text>')
        zero = ymap(0.0, y0, y1, vmin, vmax)
        parts.append(f'<line x1="{x0}" y1="{zero:.1f}" x2="{x1}" y2="{zero:.1f}" stroke="#aaa" stroke-width="1"/>')
        parts.append(f'<line x1="{xg:.1f}" y1="{y0}" x2="{xg:.1f}" y2="{y1}" stroke="#1f77b4" stroke-dasharray="4 3"/>')
        for k in range(5):
            v = vmin + k / 4 * (vmax - vmin)
            yy = ymap(v, y0, y1, vmin, vmax)
            parts.append(f'<text x="{x0-6}" y="{yy+3:.1f}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="9" fill="#666">{v:.2f}</text>')
        for tv in (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0):
            xx = xmap(tv, x0, x1)
            parts.append(f'<text x="{xx:.1f}" y="{y1+12}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="9" fill="#666">{tv:g}</text>')
        xs = [xmap(d["t"], x0, x1) for d in rows]
        y_hi = [ymap(d["ci_high"], y0, y1, vmin, vmax) for d in rows]
        y_lo = [ymap(d["ci_low"], y0, y1, vmin, vmax) for d in rows]
        ym = [ymap(d["mean"], y0, y1, vmin, vmax) for d in rows]
        band = list(zip(xs, y_hi)) + list(zip(xs[::-1], y_lo[::-1]))
        parts.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in band)}" fill="{col}" opacity="0.22"/>')
        parts.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in zip(xs, ym))}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0-6}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" font-weight="600" fill="#222">{title}</text>')
        parts.append(f'<text x="{x1-4}" y="{y1-6}" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif" font-size="9" fill="#888">{hint}</text>')
        if i == 2:
            parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+28}" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#444">t (noise)  → reverse sampling</text>')
    parts.append("</svg>")
    args.out.write_text("\n".join(parts))
    print(json.dumps({"event": "delta_curve", "n_crystals": len(ids), "out": str(args.out), "conclusion": note["curve_conclusion"]}))


if __name__ == "__main__":
    main()
