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
artifact = json.loads(artifact_path.read_text()) if artifact_path.exists() else None
audit = json.loads(audit_path.read_text()) if audit_path.exists() else None

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
        f"- Projected bond F1: {audit.get('projected_bond_f1')}",
    ])

report = "\n".join(summary_lines) + "\n"
(ROOT / "global_copy_assembly_geometry_r_report.md").write_text(report)
(ROOT / "config_audit.json").write_text(json.dumps({
    "artifact": str(artifact_path),
    "audit": str(audit_path),
    "classification": audit.get("classification") if audit is not None else None,
}, indent=2))
