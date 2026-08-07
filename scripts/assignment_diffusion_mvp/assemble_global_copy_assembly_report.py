#!/usr/bin/env python3
"""Build the final report markdown for the geometry-R workflow."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ROOT = Path("outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r")
ROOT.mkdir(parents=True, exist_ok=True)

artifact_path = ROOT / "geometry_only_hard_r.jsonl"
audit_path = ROOT / "geometry_only_r_audit.json"
map_path = ROOT / "map_evaluation_metrics.json"
eval_path = ROOT / "evaluation_metrics.json"


def _load_json_or_jsonl(path: Path):
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    # JSONL: use last non-empty record; plain JSON object: parse whole file.
    if path.suffix == ".jsonl":
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else None
    return json.loads(text)


artifact = _load_json_or_jsonl(artifact_path)
audit = _load_json_or_jsonl(audit_path)
map_eval = _load_json_or_jsonl(map_path)
evaluation = _load_json_or_jsonl(eval_path)

summary_lines = [
    "# Global copy assembly geometry-R report",
    "",
    "This report is generated from the real geometry-only predicted-R artifacts produced by the exporter/auditor chain.",
    "",
    f"- Exported artifact: {artifact_path}",
    f"- Audit artifact: {audit_path}",
]
if artifact is not None:
    summary_lines.extend([
        f"- Sample: {artifact.get('sample_id')} ({artifact.get('split')})",
        f"- Role source: {artifact.get('role_source')}",
        f"- Decoder: {artifact.get('decoder')}",
        f"- Literal accuracy: {artifact.get('metrics', {}).get('literal_accuracy', 'n/a')}",
    ])
if audit is not None:
    summary_lines.extend([
        f"- Audit classification: {audit.get('classification')}",
        f"- Role capacity valid: {audit.get('role_capacity_valid')}",
        f"- Element compatible: {audit.get('element_compatible')}",
        f"- Orbit role exact: {audit.get('orbit_role_exact')}",
        f"- Target defined: {audit.get('target_defined')}",
        f"- Structural R error: {audit.get('structural_r_error')}",
        f"- Audit projected bond F1 (oracle C0): {audit.get('projected_bond_f1')}",
    ])
if map_eval is not None:
    summary_lines.extend([
        "",
        "## Independent MAP evaluation",
        f"- Checkpoint: {map_eval.get('checkpoint')}",
        f"- exact C: {map_eval.get('exact_C')}",
        f"- copy-pair F1: {map_eval.get('copy_pair_f1')}",
        f"- projected-bond F1: {map_eval.get('projected_bond_f1')}",
        f"- projected graph exact: {map_eval.get('projected_molecular_graph_exact')}",
        f"- complete-copy rate: {map_eval.get('complete_copy_rate')}",
        f"- copy graph-isomorphism rate: {map_eval.get('copy_graph_isomorphism_rate')}",
        f"- cross-copy false molecular-edge rate: {map_eval.get('cross_copy_false_molecular_edge_rate')}",
    ])
if evaluation is not None and "correct_geometry" in evaluation:
    correct = evaluation["correct_geometry"]
    summary_lines.extend([
        "",
        f"- Evaluation status: {correct.get('status')}",
        f"- Geometry ablation delta projected-bond F1: {evaluation.get('delta_projected_bond_f1')}",
    ])

report = "\n".join(summary_lines) + "\n"
(ROOT / "global_copy_assembly_geometry_r_report.md").write_text(report)
(ROOT / "config_audit.json").write_text(json.dumps({
    "artifact": str(artifact_path),
    "audit": str(audit_path),
    "classification": audit.get("classification") if audit is not None else None,
    "map_exact_C": map_eval.get("exact_C") if map_eval is not None else None,
    "map_copy_pair_f1": map_eval.get("copy_pair_f1") if map_eval is not None else None,
}, indent=2))
