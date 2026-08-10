#!/usr/bin/env python3
"""Assemble N1 markdown report from curve summaries."""
from __future__ import annotations

import json
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    noise = json.loads((out / "noise_process_audit.json").read_text()) if (out / "noise_process_audit.json").exists() else {}
    prov = json.loads((out / "runtime_provenance.json").read_text()) if (out / "runtime_provenance.json").exists() else {}
    lines = [
        "# N1 Noisy Copy Assignment Report",
        "",
        "## Primary experiment",
        "",
        "- **PRIMARY_N1** = `gemnet` (frozen pretrained molecular-CSP GemNet node embeddings)",
        "- **ABLATION** = `context_encoder` (standalone; not run by default remote script)",
        "",
        f"- noise_source: `{noise.get('noise_source') or prov.get('NOISE_SOURCE')}`",
        f"- independent_noise_implementation: `{noise.get('independent_noise_implementation')}`",
        f"- HIDDEN_SOURCE: `{prov.get('HIDDEN_SOURCE', 'gemnet_node_embeddings')}`",
        f"- CONTEXT_CRYSTAL_ENCODER_USED: `{prov.get('CONTEXT_CRYSTAL_ENCODER_USED', False)}`",
        f"- MATTERGEN_MODEL_PATH: `{prov.get('MATTERGEN_MODEL_PATH')}`",
        f"- MATTERGEN_LOAD_EPOCH: `{prov.get('MATTERGEN_LOAD_EPOCH')}`",
        f"- MATTERGEN_CHECKPOINT: `{prov.get('MATTERGEN_CHECKPOINT')}`",
        f"- GEMNET_BACKBONE_FROZEN: `{prov.get('GEMNET_BACKBONE_FROZEN')}`",
        f"- GEOMETRY_FEEDBACK: `{prov.get('GEOMETRY_FEEDBACK', False)}`",
        f"- soft_c_semantics: `{prov.get('SOFT_C_SEMANTICS', 'c_soft_conditional_on_singleton_map')}`",
        f"- pos SDE: `{noise.get('pos_sde_path')}`",
        f"- cell SDE: `{noise.get('cell_sde_path')}`",
        f"- multi_corruption: `{noise.get('multi_corruption_path')}`",
        "",
        "Assignment branch is observational only (no geometry feedback).",
        "N1 answers: do pretrained molecular-CSP GemNet hiddens already contain",
        "decodable copy-assignment information?",
        "",
        "## Soft C definition (important)",
        "",
        "All soft same-copy metrics (`same_copy_pair_AUC`, Brier, etc.) use:",
        "",
        "> **conditional-on-singleton-MAP structured soft C**",
        "> (`soft_c_semantics = c_soft_conditional_on_singleton_map`)",
        "",
        "Construction:",
        "1. Fix singleton copy groups to hard MAP `G_singleton`.",
        "2. Boltzmann-average exact balanced orbit attachments conditional on that backbone.",
        "",
        "This is **not** full-joint structured `P(g_i=g_j)` over (tree-CRF × attachment),",
        "and **not** an unconstrained pair-sigmoid classifier.",
        "Singleton-tree uncertainty is **not** fully marginalized in this MVP.",
        "",
    ]
    for mode in ("oracle_orbit", "predicted_orbit"):
        path = out / f"{mode}_summary.json"
        if not path.exists():
            lines.append(f"## {mode}\n\n_missing_\n")
            continue
        s = json.loads(path.read_text())
        lines.append(f"## {mode}")
        lines.append("")
        lines.append(f"- first_soft_signal_auc: `{s.get('first_soft_signal_auc')}`")
        lines.append(f"- useful_grouping_f1: `{s.get('useful_grouping_f1')}`")
        lines.append(f"- reliable_map: `{s.get('reliable_map')}`")
        lines.append(f"- exact_recovery_region: `{s.get('exact_recovery_region')}`")
        lines.append("")
        lines.append("| t/T | exact_C_rate | pair F1 | AUC | orbit acc | MAP gap |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for a in s.get("aggregated") or []:
            lines.append(
                f"| {a.get('t_fraction')} | {a.get('exact_C_rate')} | {a.get('copy_pair_f1')} | "
                f"{a.get('same_copy_pair_AUC')} | {a.get('orbit_atom_accuracy')} | {a.get('orbit_attachment_MAP_gap')} |"
            )
        lines.append("")
    (out / "n1_report.md").write_text("\n".join(lines) + "\n")
    # thresholds aggregate
    thr = {
        "oracle_orbit": json.loads((out / "oracle_orbit_summary.json").read_text()) if (out / "oracle_orbit_summary.json").exists() else {},
        "predicted_orbit": json.loads((out / "predicted_orbit_summary.json").read_text()) if (out / "predicted_orbit_summary.json").exists() else {},
    }
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2))
    (out / "curve_summary.json").write_text(json.dumps(thr, indent=2))
    print(json.dumps({"report": str(out / "n1_report.md")}))


if __name__ == "__main__":
    main()
