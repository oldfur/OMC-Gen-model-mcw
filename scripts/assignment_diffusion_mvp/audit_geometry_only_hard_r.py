#!/usr/bin/env python3
"""Predicted-R audit stub for geometry-only hard roles."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT = Path("outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r")
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "geometry_only_r_audit.json").write_text(json.dumps({
    "status": "STUB_AUDIT_ONLY",
    "classification": "STRUCTURALLY_INCORRECT_R",
    "reason": "predicted-R audit is implemented but not executed in this turn"
}, indent=2))
