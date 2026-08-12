#!/usr/bin/env python3
"""Assemble N2.1 causal edge report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    prov = json.loads((out / "runtime_provenance.json").read_text()) if (out / "runtime_provenance.json").exists() else {}
    ab = json.loads((out / "ablation_summary.json").read_text()) if (out / "ablation_summary.json").exists() else {}
    reg = ab.get("region_summary") or {}
    curve = ab.get("by_t_fraction") or json.loads((out / "geometry_curve_summary.json").read_text()) if (out / "geometry_curve_summary.json").exists() else []

    lines = [
        "# N2.1 Strict Causal Edge Feedback Report",
        "",
        "## Formula",
        "",
        r"$$\Delta e_{ij} = g_{\mathrm{noise}}(t)\, q_{ij}\, s_{ij}\, F_\psi(e_{ij})$$",
        "",
        r"with $q_{ij}=2|p_{ij}-1/2|$, $s_{ij}=2p_{ij}-1$, $p_{ij}=\widetilde C_{ij}$.",
        "",
        "- $F_\\psi$ inputs: **edge embedding only** (no C concat)",
        "- No residual path bypassing $(g q s)$",
        "- Group context: **disabled**",
        "- B5 uses **same** B2-trained adapter weights; only soft-C is orbit-preserving shuffled at eval",
        "",
        f"- MATTERGEN_EPOCH: `{prov.get('MATTERGEN_EPOCH')}`",
        f"- SOFT_C_SEMANTICS: `{prov.get('SOFT_C_SEMANTICS')}`",
        f"- NOISE_GATE: full_on=`{prov.get('NOISE_GATE_FULL_ON')}` full_off=`{prov.get('NOISE_GATE_FULL_OFF')}`",
        "",
        "## Gates",
        "",
        f"- Gate A (t/T≥0.6 structural off B2=B5=B0): `{ab.get('gate_A_structural_off_ok')}`",
        f"- Gate B (Region A frac B2<B0): `{ab.get('gate_B_frac_B2_lt_B0_region_A')}`",
        f"- Gate C (Region A frac B2<B5): `{ab.get('gate_C_frac_B2_lt_B5_region_A')}`",
        f"- Claim support (A region + structural): `{ab.get('claim_support_region_A')}`",
        "",
        "## By region",
        "",
    ]
    for name in ("A_exact_c", "B_soft_info", "C_gate_off"):
        r = reg.get(name) or {}
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- I_B2: `{r.get('I_B2')}`")
        lines.append(f"- I_B5: `{r.get('I_B5')}`")
        lines.append(f"- D_C = L_B5−L_B2: `{r.get('D_C')}`")
        lines.append(f"- frac B2<B0: `{r.get('frac_B2_lt_B0')}`")
        lines.append(f"- frac B2<B5: `{r.get('frac_B2_lt_B5')}`")
        lines.append("")
    lines += [
        "## By t/T",
        "",
        "| t/T | region | L_B0 | L_B2 | L_B5 | I_B2 | I_B5 | D_C | B2<B0 | B2<B5 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in curve:
        lines.append(
            f"| {s.get('t_fraction')} | {s.get('region')} | {s.get('mean_L_B0')} | {s.get('mean_L_B2')} | "
            f"{s.get('mean_L_B5')} | {s.get('mean_I_B2')} | {s.get('mean_I_B5')} | {s.get('mean_D_C')} | "
            f"{s.get('frac_B2_lt_B0')} | {s.get('frac_B2_lt_B5')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Support causal C→geometry only if Gate A holds **and** Region A has B2<B0 and B2<B5 stably.",
        "- If B2≈B5 under strict modulation, that is a clean negative result (not adapter capacity).",
        "",
    ]
    (out / "n2_1_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(out / "n2_1_report.md")}))


if __name__ == "__main__":
    main()
