#!/usr/bin/env python3
"""Plot t-binned geometry/pos/cell loss: Original, G-cond, Oracle-G, Clean-G."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "outputs/assignment_diffusion_mvp"
BINS = [
    (0.0, 0.2, "[0.0,0.2)", 0.1),
    (0.2, 0.4, "[0.2,0.4)", 0.3),
    (0.4, 0.6, "[0.4,0.6)", 0.5),
    (0.6, 0.8, "[0.6,0.8)", 0.7),
    (0.8, 1.01, "[0.8,1.0]", 0.9),
]
LABELS = [b[2] for b in BINS]
XS = [b[3] for b in BINS]

COLORS = {
    "Original": "#444444",
    "G-conditioned": "#d62728",
    "Oracle-G": "#1f77b4",
    "Clean-G": "#2ca02c",
}


def tbin(t: float) -> str:
    for a, b, k, _ in BINS:
        if a <= t < b:
            return k
    return "[0.8,1.0]"


def load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pos_of(r: dict):
    v = r.get("geom_pos", r.get("pos"))
    return None if v is None else float(v)


def cell_of(r: dict):
    v = r.get("geom_cell", r.get("cell"))
    return None if v is None else float(v)


def aggregate(rows: list[dict], start: int = 0) -> dict:
    g: dict[str, list] = defaultdict(list)
    p: dict[str, list] = defaultdict(list)
    c: dict[str, list] = defaultdict(list)
    for r in rows[start:]:
        k = tbin(float(r["global_t"]))
        g[k].append(float(r["geometry_loss"]))
        pv, cv = pos_of(r), cell_of(r)
        if pv is not None:
            p[k].append(pv)
        if cv is not None:
            c[k].append(cv)
    out = {}
    for _, _, k, _ in BINS:
        out[k] = {
            "geom": sum(g[k]) / len(g[k]) if g[k] else None,
            "pos": sum(p[k]) / len(p[k]) if p[k] else None,
            "cell": sum(c[k]) / len(c[k]) if c[k] else None,
            "n": len(g[k]),
        }
    return out


def mean_aggs(aggs: list[dict]) -> dict:
    out = {}
    for _, _, k, _ in BINS:
        rec = {}
        for field in ("geom", "pos", "cell"):
            xs = [a[k][field] for a in aggs if a[k][field] is not None]
            rec[field] = sum(xs) / len(xs) if xs else None
            rec[field + "_min"] = min(xs) if xs else None
            rec[field + "_max"] = max(xs) if xs else None
        rec["n"] = sum(a[k]["n"] for a in aggs)
        out[k] = rec
    return out


def _mapx(x, x0, x1, px0, px1):
    return px0 + (x - x0) / (x1 - x0) * (px1 - px0)


def _mapy(y, y0, y1, py0, py1):
    return py1 - (y - y0) / (y1 - y0) * (py1 - py0)


def polyline(xs, ys, x0, x1, y0, y1, px0, px1, py0, py1):
    pts = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        pts.append(f"{_mapx(x,x0,x1,px0,px1):.1f},{_mapy(y,y0,y1,py0,py1):.1f}")
    return " ".join(pts)


def panel(svg, title, field, series, x0, y0, w, h):
    pad_l, pad_r, pad_t, pad_b = 58, 16, 28, 42
    px0, py0 = x0 + pad_l, y0 + pad_t
    px1, py1 = x0 + w - pad_r, y0 + h - pad_b
    vals = []
    for agg in series.values():
        for _, _, k, _ in BINS:
            v = agg[k][field]
            if v is not None:
                vals.append(v)
            lo, hi = agg[k].get(field + "_min"), agg[k].get(field + "_max")
            if lo is not None:
                vals.append(lo)
            if hi is not None:
                vals.append(hi)
    ymin = 0.0
    ymax = max(vals) * 1.12 if vals else 1.0
    xmin, xmax = 0.0, 1.0

    svg.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fff" stroke="#ddd"/>')
    svg.append(
        f'<text x="{x0 + w/2:.1f}" y="{y0 + 18}" text-anchor="middle" '
        f'font-family="DejaVu Sans,Helvetica,Arial" font-size="13" font-weight="600">{title}</text>'
    )
    # grid
    for i in range(5):
        yy = py0 + i * (py1 - py0) / 4
        gv = ymax * (1 - i / 4)
        svg.append(
            f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{px0-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'font-family="DejaVu Sans,Helvetica,Arial" font-size="10" fill="#555">{gv:.3f}</text>'
        )
    for x, lab in zip(XS, LABELS):
        xx = _mapx(x, xmin, xmax, px0, px1)
        svg.append(
            f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py1}" '
            f'stroke="#f3f3f3" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{xx:.1f}" y="{py1+16}" text-anchor="middle" '
            f'font-family="DejaVu Sans,Helvetica,Arial" font-size="9" fill="#444">{lab}</text>'
        )
    svg.append(
        f'<text x="{x0 + w/2:.1f}" y="{y0+h-8}" text-anchor="middle" '
        f'font-family="DejaVu Sans,Helvetica,Arial" font-size="11" fill="#333">diffusion time t</text>'
    )
    svg.append(
        f'<text x="{x0+14}" y="{y0+h/2:.1f}" text-anchor="middle" transform="rotate(-90 {x0+14} {y0+h/2:.1f})" '
        f'font-family="DejaVu Sans,Helvetica,Arial" font-size="11" fill="#333">loss</text>'
    )
    svg.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="#333" stroke-width="1.2"/>')
    svg.append(f'<line x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke="#333" stroke-width="1.2"/>')

    styles = [
        ("Original (mean of 3)", "#444444", "6,4", True),
        ("G-conditioned", "#d62728", None, False),
        ("Oracle-G (noisy At)", "#1f77b4", None, False),
        ("Clean-G (A0)", "#2ca02c", None, False),
    ]
    names = list(series.keys())
    for (label, color, dash, band), name in zip(styles, names):
        agg = series[name]
        ys = [agg[k][field] for _, _, k, _ in BINS]
        if band and agg[LABELS[0]].get(field + "_min") is not None:
            # min-max band
            ylo = [agg[k][field + "_min"] for _, _, k, _ in BINS]
            yhi = [agg[k][field + "_max"] for _, _, k, _ in BINS]
            up = [
                f"{_mapx(x,xmin,xmax,px0,px1):.1f},{_mapy(y,ymin,ymax,py0,py1):.1f}"
                for x, y in zip(XS, yhi)
                if y is not None
            ]
            dn = [
                f"{_mapx(x,xmin,xmax,px0,px1):.1f},{_mapy(y,ymin,ymax,py0,py1):.1f}"
                for x, y in zip(reversed(XS), reversed(ylo))
                if y is not None
            ]
            svg.append(
                f'<polygon points="{" ".join(up+dn)}" fill="{color}" fill-opacity="0.12" stroke="none"/>'
            )
        dashattr = f' stroke-dasharray="{dash}"' if dash else ""
        pts = polyline(XS, ys, xmin, xmax, ymin, ymax, px0, px1, py0, py1)
        svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dashattr} '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y in zip(XS, ys):
            if y is None:
                continue
            svg.append(
                f'<circle cx="{_mapx(x,xmin,xmax,px0,px1):.1f}" cy="{_mapy(y,ymin,ymax,py0,py1):.1f}" '
                f'r="3.2" fill="#fff" stroke="{color}" stroke-width="1.6"/>'
            )


def legend(svg, x, y):
    items = [
        ("Original (mean of 3 runs, band = min–max)", "#444444", "6,4"),
        ("G-conditioned (learned G + L_G/L_R)", "#d62728", None),
        ("Oracle-G (noisy At, L_geom only)", "#1f77b4", None),
        ("Clean-G (A0 every t, L_geom only)", "#2ca02c", None),
    ]
    svg.append(
        f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Helvetica,Arial" font-size="12" font-weight="600">Methods</text>'
    )
    for i, (lab, color, dash) in enumerate(items):
        yy = y + 22 + i * 20
        dashattr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<line x1="{x}" y1="{yy-4}" x2="{x+28}" y2="{yy-4}" stroke="{color}" stroke-width="2.4"{dashattr}/>'
        )
        svg.append(f'<circle cx="{x+14}" cy="{yy-4}" r="3" fill="#fff" stroke="{color}" stroke-width="1.5"/>')
        svg.append(
            f'<text x="{x+36}" y="{yy}" font-family="DejaVu Sans,Helvetica,Arial" font-size="12" fill="#222">{lab}</text>'
        )


def write_figure(path: Path, series: dict, subtitle: str):
    W, H = 1100, 980
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="550" y="32" text-anchor="middle" font-family="DejaVu Sans,Helvetica,Arial" '
        'font-size="18" font-weight="700">Assignment → geometry ablations, loss vs diffusion time t</text>',
        f'<text x="550" y="52" text-anchor="middle" font-family="DejaVu Sans,Helvetica,Arial" '
        f'font-size="12" fill="#555">{subtitle}</text>',
    ]
    panel(svg, "geometry_loss (weighted pos+cell)", "geom", series, 40, 70, 700, 280)
    panel(svg, "pos_loss", "pos", series, 40, 360, 700, 280)
    panel(svg, "cell_loss", "cell", series, 40, 650, 700, 280)
    legend(svg, 760, 120)
    note = (
        "Each point is the mean training denoising loss of samples whose global t falls in that bin. "
        "Original is averaged over the three independent Original runs (G-cond / Oracle-G / Clean-G experiments). "
        "G-conditioned jointly trains L_G/L_R; Oracle-G and Clean-G train L_geom only."
    )
    svg.append(
        f'<foreignObject x="760" y="230" width="310" height="280">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" style="font-family:DejaVu Sans,Helvetica,Arial;font-size:11px;color:#444;line-height:1.45">{note}</div>'
        f'</foreignObject>'
    )
    # foreignObject can be flaky in convert; add wrapped tspans instead
    svg.pop()
    words = note.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) > 42:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines):
        svg.append(
            f'<text x="760" y="{250 + i*16}" font-family="DejaVu Sans,Helvetica,Arial" '
            f'font-size="11" fill="#555">{line}</text>'
        )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg))
    print("wrote", path)


def main():
    base = ROOT / "outputs/assignment_diffusion_mvp"
    origs = [
        aggregate(load_jsonl(base / "g_geometry_ablation/original/training_trace.jsonl")),
        aggregate(load_jsonl(base / "oracle_g_geometry_ablation/original/training_trace.jsonl")),
        aggregate(load_jsonl(base / "clean_g_geometry_ablation/original/training_trace.jsonl")),
    ]
    series_all = {
        "Original": mean_aggs(origs),
        "G-conditioned": aggregate(load_jsonl(base / "g_geometry_ablation/g_conditioned/training_trace.jsonl")),
        "Oracle-G": aggregate(load_jsonl(base / "oracle_g_geometry_ablation/oracle_g/training_trace.jsonl")),
        "Clean-G": aggregate(load_jsonl(base / "clean_g_geometry_ablation/clean_g/training_trace.jsonl")),
    }
    origs_pw = [
        aggregate(load_jsonl(base / "g_geometry_ablation/original/training_trace.jsonl"), 200),
        aggregate(load_jsonl(base / "oracle_g_geometry_ablation/original/training_trace.jsonl"), 200),
        aggregate(load_jsonl(base / "clean_g_geometry_ablation/original/training_trace.jsonl"), 200),
    ]
    series_pw = {
        "Original": mean_aggs(origs_pw),
        "G-conditioned": aggregate(load_jsonl(base / "g_geometry_ablation/g_conditioned/training_trace.jsonl"), 200),
        "Oracle-G": aggregate(load_jsonl(base / "oracle_g_geometry_ablation/oracle_g/training_trace.jsonl"), 200),
        "Clean-G": aggregate(load_jsonl(base / "clean_g_geometry_ablation/clean_g/training_trace.jsonl"), 200),
    }
    out1 = OUTDIR / "ablation_tbin_loss_all_steps.svg"
    out2 = OUTDIR / "ablation_tbin_loss_post_warmup.svg"
    write_figure(out1, series_all, "All 1000 training steps (includes warmup)")
    write_figure(out2, series_pw, "Post-warmup only: training steps 200–999 (primary)")
    table = {"all_steps": series_all, "post_warmup_200_999": series_pw}
    (OUTDIR / "ablation_tbin_loss_table.json").write_text(json.dumps(table, indent=2, default=str))
    print("wrote", OUTDIR / "ablation_tbin_loss_table.json")


if __name__ == "__main__":
    main()
