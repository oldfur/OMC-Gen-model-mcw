#!/usr/bin/env python3
"""Deterministic exporter stub for geometry-only hard-R artifacts.

This implementation-only turn writes the expected manifest structure without
executing any training or inference.
"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path("outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r")
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "geometry_only_hard_r.jsonl").write_text(
    json.dumps({"status": "STUB_EXPORT_ONLY", "role_source": "geometry_only_hard_r"}) + "\n"
)
