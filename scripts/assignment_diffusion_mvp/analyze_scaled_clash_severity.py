#!/usr/bin/env python3
"""Post-hoc: is Clean-G E_clash gain from a few extreme inter-copy collisions?"""
from __future__ import annotations

import io
import json
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

CUTOFF = 1.2
SEVERE = 0.7
VERY = 0.4


class _H:
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
                tname = str(pid[1])
                if "Long" in tname or "Int" in tname:
                    dt = np.int64
                elif "Byte" in tname or "Char" in tname:
                    dt = np.uint8
                else:
                    dt = np.float32
                return np.frombuffer(raw, dtype=dt)
            return pid

        def find_class(self, module, name):
            if module.startswith("torch"):
                if name in ("_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_parameter"):
                    def rebuild(storage, storage_offset, size, stride, *rest):
                        n = int(np.prod(size)) if size else 0
                        view = np.asarray(storage)[storage_offset : storage_offset + max(n, 1)]
                        try:
                            return np.array(view.reshape(size))
                        except Exception:
                            return np.array(view)

                    return rebuild
                return _H
            return super().find_class(module, name)

    return U(io.BytesIO(zf.read(f"{prefix}data.pkl"))).load()


def pbc_pair_dist(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    frac = np.asarray(frac, dtype=np.float64)
    d = frac[:, None, :] - frac[None, :, :]
    d = d - np.round(d)
    return np.linalg.norm(d @ cell, axis=-1)


def pair_stats(frac, cell, copy, cutoff=CUTOFF):
    copy = np.asarray(copy).reshape(-1)
    dist = pbc_pair_dist(frac, cell)
    n = dist.shape[0]
    iu = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = iu & (copy[:, None] != copy[None, :])
    d = dist[mask]
    if d.size == 0:
        z = dict(
            E_clash=0.0,
            n_soft=0.0,
            n_severe=0.0,
            n_very=0.0,
            max_pen=0.0,
            min_d=float("nan"),
            top1_E=0.0,
            top3_E=0.0,
            top5_E=0.0,
            top1_pen=0.0,
            top3_pen_mean=0.0,
            share_top1=0.0,
            share_top3=0.0,
            n_pairs=0.0,
        )
        return z
    pen = np.clip(cutoff - d, 0.0, None)
    e = pen * pen
    order = np.argsort(-e)
    e_sorted = e[order]
    pen_sorted = pen[order]
    E = float(e.sum())
    top1 = float(e_sorted[0])
    top3 = float(e_sorted[:3].sum())
    top5 = float(e_sorted[:5].sum())
    return {
        "E_clash": E,
        "n_soft": float((d < cutoff).sum()),
        "n_severe": float((d < SEVERE).sum()),
        "n_very": float((d < VERY).sum()),
        "max_pen": float(pen.max()),
        "min_d": float(d.min()),
        "top1_E": top1,
        "top3_E": top3,
        "top5_E": top5,
        "top1_pen": float(pen_sorted[0]),
        "top3_pen_mean": float(pen_sorted[: min(3, pen_sorted.size)].mean()),
        "share_top1": (top1 / E) if E > 1e-12 else 0.0,
        "share_top3": (top3 / E) if E > 1e-12 else 0.0,
        "n_pairs": float(d.size),
        "E_rest": E - top1,
    }


def collect_arm(arm_dir: Path, copies: dict):
    by_id = defaultdict(list)
    for p in sorted(arm_dir.glob("traj_*_*.pt")):
        obj = load_pt(p)
        cid = obj["row"]["id"]
        st = pair_stats(obj["frac"], obj["cell"], copies[cid])
        st["row_E"] = float(obj["row"].get("final_E_clash") or 0.0)
        by_id[cid].append(st)
    crystals = {}
    keys = [k for k in next(iter(by_id.values()))[0] if k != "row_E"]
    for cid, rows in by_id.items():
        crystals[cid] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        crystals[cid]["n_traj"] = len(rows)
    return crystals


def bootstrap_ci(xs, n_boot=1000, seed=17):
    xs = np.asarray([x for x in xs if x == x], dtype=float)
    if xs.size == 0:
        return None
    rng = np.random.RandomState(seed)
    n = xs.size
    means = [float(xs[rng.randint(0, n, n)].mean()) for _ in range(n_boot)]
    means.sort()
    return {"low": means[int(0.025 * (n_boot - 1))], "high": means[int(0.975 * (n_boot - 1))], "n": int(n)}


def ecdf(xs):
    xs = np.sort(np.asarray(xs, dtype=float))
    y = np.arange(1, xs.size + 1) / xs.size
    return xs, y


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-pt", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--clean-g", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    test = load_pt(args.test_pt)
    copies = {s["id"]: np.asarray(s["copy"]) for s in test}
    orig = collect_arm(args.original, copies)
    cln = collect_arm(args.clean_g, copies)
    ids = sorted(set(orig) & set(cln))

    keys = [
        "E_clash",
        "n_soft",
        "n_severe",
        "n_very",
        "max_pen",
        "min_d",
        "top1_E",
        "top3_E",
        "top5_E",
        "top1_pen",
        "share_top1",
        "share_top3",
        "E_rest",
    ]
    paired = {}
    for k in keys:
        d = np.array([cln[i][k] - orig[i][k] for i in ids], dtype=float)
        paired[k] = {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "ci95": bootstrap_ci(d),
            "win_rate_lower": float((d < 0).mean()),
            "win_rate_higher": float((d > 0).mean()),
        }

    oE = np.array([orig[i]["E_clash"] for i in ids])
    cE = np.array([cln[i]["E_clash"] for i in ids])
    dE = cE - oE
    d_top = np.array([cln[i]["top1_E"] - orig[i]["top1_E"] for i in ids])
    d_rest = np.array([cln[i]["E_rest"] - orig[i]["E_rest"] for i in ids])
    # share of improvement from top-1 (only where original E>0)
    tot_imp = float((-dE).sum())  # positive if Clean-G lower
    top_imp = float((-d_top).sum())
    rest_imp = float((-d_rest).sum())
    # worst 20% original E_clash crystals
    thr = np.quantile(oE, 0.8)
    hi = oE >= thr
    lo = ~hi

    summary = {
        "n_crystals": len(ids),
        "cutoff": CUTOFF,
        "severe_dist": SEVERE,
        "very_severe_dist": VERY,
        "paired": paired,
        "mean_original": {k: float(np.mean([orig[i][k] for i in ids])) for k in keys},
        "mean_clean_g": {k: float(np.mean([cln[i][k] for i in ids])) for k in keys},
        "E_clash_reduction_from_top1_pair": top_imp / tot_imp if tot_imp > 1e-12 else None,
        "E_clash_reduction_from_rest": rest_imp / tot_imp if tot_imp > 1e-12 else None,
        "mean_share_top1_original": float(np.mean([orig[i]["share_top1"] for i in ids])),
        "mean_share_top1_clean_g": float(np.mean([cln[i]["share_top1"] for i in ids])),
        "mean_share_top3_original": float(np.mean([orig[i]["share_top3"] for i in ids])),
        "mean_share_top3_clean_g": float(np.mean([cln[i]["share_top3"] for i in ids])),
        "worst20pct_orig_E_mean_delta": float(dE[hi].mean()) if hi.any() else None,
        "other80pct_orig_E_mean_delta": float(dE[lo].mean()) if lo.any() else None,
        "worst20pct_n": int(hi.sum()),
    }
    # interpretation
    frac_top = summary["E_clash_reduction_from_top1_pair"]
    if frac_top is not None and frac_top >= 0.6:
        verdict = (
            "Most of the E_clash drop comes from the single worst inter-copy pair "
            f"({frac_top:.0%} of total reduction). Gain is dominated by fewer extreme collisions."
        )
    elif frac_top is not None and frac_top >= 0.35:
        verdict = (
            f"The worst pair accounts for {frac_top:.0%} of the E_clash reduction; "
            "the remainder is a broader (but smaller) shift. Mixed: extremes matter, not exclusively."
        )
    else:
        verdict = (
            "E_clash reduction is not concentrated in the worst pair "
            f"(top-1 share of reduction={frac_top}). More consistent with a bulk shift."
        )
    summary["verdict"] = verdict
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    # ---- SVG ----
    W, H = 1120, 900
    C_O, C_C = "#4d4d4d", "#2ca02c"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="26" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="17" font-weight="700" fill="#222">Clash severity: Original vs Clean-G</text>',
        '<text x="480" y="44" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="11" fill="#666">150 crystals, 2 traj averaged. E=Σ max(0,1.2−d)² on inter-copy pairs. Severe: d&lt;0.7 Å.</text>',
        f'<rect x="900" y="14" width="12" height="12" fill="{C_O}"/>',
        '<text x="918" y="24" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">Original</text>',
        f'<rect x="1000" y="14" width="12" height="12" fill="{C_C}"/>',
        '<text x="1018" y="24" font-family="DejaVu Sans, Arial, sans-serif" font-size="12">Clean-G</text>',
    ]

    def panel_box(x0, y0, x1, y1):
        parts.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#fafafa" stroke="#ddd"/>')

    def draw_ecdf(x0, y0, x1, y1, key, title, logx=False):
        ov = np.array([orig[i][key] for i in ids], dtype=float)
        cv = np.array([cln[i][key] for i in ids], dtype=float)
        if logx:
            ov = np.clip(ov, 1e-4, None)
            cv = np.clip(cv, 1e-4, None)
            xmin, xmax = min(ov.min(), cv.min()), max(ov.max(), cv.max())
            def xmap(v):
                return x0 + (np.log10(v) - np.log10(xmin)) / max(np.log10(xmax) - np.log10(xmin), 1e-9) * (x1 - x0)
        else:
            xmin, xmax = 0.0, max(ov.max(), cv.max())
            pad = 0.05 * (xmax - xmin if xmax > xmin else 1)
            xmax += pad
            def xmap(v):
                return x0 + (v - xmin) / max(xmax - xmin, 1e-9) * (x1 - x0)
        def ymap(v):
            return y1 - v * (y1 - y0)
        panel_box(x0, y0, x1, y1)
        for yv in (0.25, 0.5, 0.75, 1.0):
            yy = ymap(yv)
            parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#eee"/>')
            parts.append(f'<text x="{x0-4}" y="{yy+3:.1f}" text-anchor="end" font-size="8" font-family="DejaVu Sans, Arial, sans-serif" fill="#888">{yv:g}</text>')
        for xs, col in ((ov, C_O), (cv, C_C)):
            x, y = ecdf(xs)
            pts = " ".join(f"{xmap(a):.1f},{ymap(b):.1f}" for a, b in zip(x, y))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y0-6}" text-anchor="middle" font-size="12" font-weight="600" font-family="DejaVu Sans, Arial, sans-serif">{title}</text>')
        # a couple of x ticks
        if logx:
            ticks = []
            for e in range(int(np.floor(np.log10(xmin))), int(np.ceil(np.log10(xmax))) + 1):
                ticks.append(10.0 ** e)
        else:
            ticks = [xmin, 0.5 * (xmin + xmax), xmax]
        for t in ticks:
            if t < xmin or t > xmax:
                continue
            xx = xmap(t)
            lab = f"{t:.2g}" if t < 10 else f"{t:.1f}"
            parts.append(f'<text x="{xx:.1f}" y="{y1+12}" text-anchor="middle" font-size="8" font-family="DejaVu Sans, Arial, sans-serif" fill="#666">{lab}</text>')

    # 2x3 layout
    pad_l, pad_t = 58, 62
    gw, gh = 350, 215
    panels = [
        (0, 0, "E_clash", "ECDF of E_clash", True),
        (0, 1, "max_pen", "ECDF of max penetration (Å)", False),
        (0, 2, "n_severe", "ECDF of # severe pairs (d<0.7 Å)", False),
        (1, 0, "share_top1", "ECDF of E share from worst pair", False),
        (1, 1, "top1_E", "ECDF of worst-pair energy", True),
        (1, 2, "n_soft", "ECDF of # soft-clash pairs (d<1.2)", False),
    ]
    for r, c, key, title, logx in panels:
        x0 = pad_l + c * gw
        y0 = pad_t + r * gh
        draw_ecdf(x0, y0, x0 + 300, y0 + 165, key, title, logx=logx)

    # bottom annotation bar with decomposition
    o_top = float(np.mean([orig[i]["top1_E"] for i in ids]))
    o_rest = float(np.mean([orig[i]["E_rest"] for i in ids]))
    c_top = float(np.mean([cln[i]["top1_E"] for i in ids]))
    c_rest = float(np.mean([cln[i]["E_rest"] for i in ids]))
    mx = max(o_top + o_rest, c_top + c_rest, 1e-6)
    def bar(x, y, w, h, col, label=None):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{col}"/>')
    bx, by, bw, bh = 70, 520, 420, 28
    parts.append('<text x="70" y="508" font-size="12" font-weight="600" font-family="DejaVu Sans, Arial, sans-serif">Mean E_clash split: worst pair vs the rest</text>')
    # original
    parts.append('<text x="58" y="520" text-anchor="end" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">Orig</text>')
    bar(bx, by, (o_top / mx) * bw, bh, "#c44e52")
    bar(bx + (o_top / mx) * bw, by, (o_rest / mx) * bw, bh, C_O)
    parts.append(f'<text x="{bx+bw+8}" y="{by+18}" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">top1={o_top:.3f}  rest={o_rest:.3f}</text>')
    parts.append('<text x="58" y="556" text-anchor="end" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">Clean</text>')
    bar(bx, by + 36, (c_top / mx) * bw, bh, "#7ac47a")
    bar(bx + (c_top / mx) * bw, by + 36, (c_rest / mx) * bw, bh, C_C)
    parts.append(f'<text x="{bx+bw+8}" y="{by+54}" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">top1={c_top:.3f}  rest={c_rest:.3f}</text>')
    parts.append('<rect x="70" y="600" width="12" height="12" fill="#c44e52"/>')
    parts.append('<text x="86" y="610" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">worst-pair energy</text>')
    parts.append(f'<rect x="220" y="600" width="12" height="12" fill="{C_O}"/>')
    parts.append('<text x="236" y="610" font-size="11" font-family="DejaVu Sans, Arial, sans-serif">remaining pairs</text>')

    # text stats
    tx, ty = 620, 520
    lines = [
        f"n = {len(ids)} crystals (2 traj averaged)",
        f"Δ E_clash mean {paired['E_clash']['mean']:.3f}  CI [{paired['E_clash']['ci95']['low']:.3f}, {paired['E_clash']['ci95']['high']:.3f}]",
        f"Δ max_pen mean {paired['max_pen']['mean']:.3f}  CI [{paired['max_pen']['ci95']['low']:.3f}, {paired['max_pen']['ci95']['high']:.3f}]",
        f"Δ n_severe mean {paired['n_severe']['mean']:.3f}  CI [{paired['n_severe']['ci95']['low']:.3f}, {paired['n_severe']['ci95']['high']:.3f}]",
        f"Δ top1_E mean {paired['top1_E']['mean']:.3f}  rest {paired['E_rest']['mean']:.3f}",
        f"Reduction from worst pair: {100*(frac_top or 0):.0f}%   from rest: {100*(summary['E_clash_reduction_from_rest'] or 0):.0f}%",
        f"Worst 20% orig crystals ΔE={summary['worst20pct_orig_E_mean_delta']:.3f} vs other {summary['other80pct_orig_E_mean_delta']:.3f}",
        f"Mean top-1 share of E: orig {summary['mean_share_top1_original']:.2f}  clean {summary['mean_share_top1_clean_g']:.2f}",
    ]
    for i, line in enumerate(lines):
        parts.append(f'<text x="{tx}" y="{ty+i*16}" font-size="11" font-family="DejaVu Sans, Arial, sans-serif" fill="#333">{line}</text>')
    # wrap verdict
    v = verdict
    yv = ty + 8 * 16 + 10
    # split verdict into ~70 char lines
    words = v.split()
    cur = ""
    row = 0
    for w in words:
        if len(cur) + len(w) + 1 > 70:
            parts.append(f'<text x="{tx}" y="{yv+row*15}" font-size="11" font-family="DejaVu Sans, Arial, sans-serif" fill="#111" font-weight="600">{cur}</text>')
            row += 1
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        parts.append(f'<text x="{tx}" y="{yv+row*15}" font-size="11" font-family="DejaVu Sans, Arial, sans-serif" fill="#111" font-weight="600">{cur}</text>')

    parts.append("</svg>")
    svg_path = args.out.with_suffix(".svg")
    svg_path.write_text("\n".join(parts))
    print(json.dumps({"event": "clash_severity", "n": len(ids), "verdict": verdict, "json": str(args.out), "svg": str(svg_path)}))


if __name__ == "__main__":
    main()
