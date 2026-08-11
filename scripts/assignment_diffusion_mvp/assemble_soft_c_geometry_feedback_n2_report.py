#!/usr/bin/env python3
"""Assemble N2 markdown report."""
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
    ha = json.loads((out / "parameter_hash_audit.json").read_text()) if (out / "parameter_hash_audit.json").exists() else {}
    summary = ab.get("by_t_fraction") or []

    lines = [
        "# N2 Soft-C Geometry Feedback Report",
        "",
        "## Architecture",
        "",
        "- Pass A: frozen unconditioned epoch294 GemNet + frozen N1 → soft C_t (stop-grad)",
        "- Pass B: same (X_t,L_t) + stopgrad(soft C_t) → GemNet + N2 adapters → geometry scores",
        "- **No independent G/C diffusion** (`ASSIGNMENT_TRAJECTORY=geometry_induced_inference`)",
        f"- soft_c_semantics: `{prov.get('SOFT_C_SEMANTICS')}`",
        f"- edge semantics: `{prov.get('EDGE_SEMANTICS')}`",
        f"- group context: `{prov.get('GROUP_CONTEXT')}`",
        f"- pairwise confidence q=2|p-0.5|; noise gate full_on={prov.get('NOISE_GATE_FULL_ON')} full_off={prov.get('NOISE_GATE_FULL_OFF')}",
        "",
        "## Insertion points",
        "",
        "- **Node/group adapter**: `mattergen/common/gemnet/gemnet.py` `GemNetT.forward` after atom embedding / node_condition, before edge embedding + interaction blocks",
        "- **Edge adapter**: same file, after `angle_edge_emb`, before `int_blocks`",
        "- Threaded via `GemNetTDenoiser.forward(..., soft_c_feedback=...)`",
        "",
        "## Provenance",
        "",
        f"- MATTERGEN_EPOCH: `{prov.get('MATTERGEN_EPOCH')}`",
        f"- MATTERGEN_CHECKPOINT_SHA: `{prov.get('MATTERGEN_CHECKPOINT_SHA')}`",
        f"- N1_CHECKPOINT_SHA: `{prov.get('N1_CHECKPOINT_SHA')}`",
        f"- BASE_GEMNET_FROZEN: `{prov.get('BASE_GEMNET_FROZEN')}`",
        f"- ASSIGNMENT_FROZEN: `{prov.get('ASSIGNMENT_FROZEN')}`",
        f"- GEOMETRY_OBJECTIVE: `{prov.get('GEOMETRY_OBJECTIVE')}`",
        f"- N1 hash unchanged: `{ha.get('n1_unchanged')}`",
        f"- GemNet hash unchanged: `{ha.get('gemnet_unchanged')}`",
        f"- N2 adapters updated: `{ha.get('n2_updated')}`",
        "",
        "## Ablation summary (paired Δloss vs B0; negative = better)",
        "",
        "| t/T | B0 loss | ΔB2 combined | ΔB3 edge | ΔB4 group | ΔB5 shuffled | B1≡B0 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary:
        lines.append(
            f"| {s.get('t_fraction')} | {s.get('mean_loss_B0')} | {s.get('mean_delta_B2')} | "
            f"{s.get('mean_delta_B3')} | {s.get('mean_delta_B4')} | {s.get('mean_delta_B5')} | "
            f"{s.get('b1_match_rate')} |"
        )
    lines += [
        "",
        f"- overall mean ΔB2: `{ab.get('overall_mean_delta_B2')}`",
        f"- overall mean ΔB5 (shuffled): `{ab.get('overall_mean_delta_B5_shuffled')}`",
        "",
        "## Interpretation checklist",
        "",
        "1. Soft C is geometry-induced per timestep (not G diffusion).",
        "2. Soft C is stop-grad into geometry loss.",
        "3. B1 (no feedback) must match B0 geometry scores.",
        "4. If B2 improves but B5 shuffled does not, copy structure is causal.",
        "5. High-noise (t/T≥0.6) gate should null feedback.",
        "",
    ]
    (out / "n2_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(out / "n2_report.md")}))


if __name__ == "__main__":
    main()
