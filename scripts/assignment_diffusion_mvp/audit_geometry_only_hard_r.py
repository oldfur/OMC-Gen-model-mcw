#!/usr/bin/env python3
"""Audit a geometry-only hard-R artifact against capacity and Aut(G_mol)^K gauge.

Literal role metrics are DIAGNOSTIC_ONLY. Physical classification is:

* GAUGE_EQUIVALENT_R — differences from R0 are explained by independent
  per-copy molecular automorphisms;
* STRUCTURALLY_INCORRECT_R — capacity/element violations, cross-copy
  compensation (e.g. 1,1 vs 2,2), or any error outside Aut^K.

Never canonicalizes predicted R toward oracle R0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mattergen.assignment.global_copy_assembly.targets import (
    build_assembly_target_from_predicted_roles,
    permutations_to_group,
)

ROOT = Path("outputs/assignment_diffusion_mvp")
ARTIFACT_PATH = Path("outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/geometry_only_hard_r.jsonl")
SAMPLE_PATH = ROOT / "d1_fixed_clean_geometry" / "fixed_sample.pt"
AUT_PATH = ROOT / "role_automorphism_audit" / "molecular_automorphisms.json"
ORBIT_PATH = ROOT / "role_automorphism_audit" / "role_orbits.json"
OUTPUT = Path("outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r")
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_sample() -> dict:
    return torch.load(SAMPLE_PATH, map_location="cpu", weights_only=False)


def load_artifact() -> dict:
    lines = [json.loads(line) for line in ARTIFACT_PATH.read_text().splitlines() if line.strip()]
    if not lines:
        raise FileNotFoundError(f"predicted-role artifact missing: {ARTIFACT_PATH}")
    return lines[-1]


def r_onehot(role: torch.Tensor, m: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(role.long(), m).to(torch.float32)


def molecular_bond_tensor(sample: dict) -> torch.Tensor:
    m = int(sample["M"])
    types = int(sample["role_bond_type"].max()) + 1
    bonds = torch.zeros(m, m, types, dtype=torch.float32)
    for (left, right), bond in zip(sample["role_edge_index"].T.tolist(), sample["role_bond_type"].tolist()):
        bonds[left, right, bond] = 1.0
    return bonds


def projected_bond(R: torch.Tensor, C: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.stack([(C * (R @ B[:, :, bond] @ R.T)).gt(0.5) for bond in range(B.shape[-1])], -1)


def main() -> None:
    sample = load_sample()
    artifact = load_artifact()
    if artifact.get("role_source") != "geometry_only_hard_r":
        raise ValueError(f"expected role_source=geometry_only_hard_r, got {artifact.get('role_source')!r}")
    if "role_assignment" not in artifact:
        raise ValueError("artifact missing role_assignment; refuse to invent or fall back to oracle R0")

    # R_effective := R_artifact — no Aut alignment rewrite, no oracle repair.
    predicted_roles = torch.tensor(artifact["role_assignment"], dtype=torch.long)
    truth = sample["role"].long()
    m = int(sample["M"])
    k = int(sample["Z"])
    if predicted_roles.numel() != int(sample["N"]):
        raise ValueError("predicted role artifact has the wrong atom count")
    if int(predicted_roles.min()) < 0 or int(predicted_roles.max()) >= m:
        raise ValueError("predicted roles contain out-of-range role labels")

    perms = json.loads(AUT_PATH.read_text())["permutations"]
    orbits = [
        value
        for _, value in sorted(
            json.loads(ORBIT_PATH.read_text())["role_orbits"].items(),
            key=lambda item: int(item[0]),
        )
    ]

    target, audit = build_assembly_target_from_predicted_roles(
        predicted_roles,
        sample["copy"],
        M=m,
        K=k,
        anchor_role=0,
        role_z=sample["role_z"],
        z=sample["z"],
        oracle_role=truth,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )

    # Prove no canonicalization leakage into target role sets.
    if target is not None:
        recovered = torch.empty(int(sample["N"]), dtype=torch.long)
        for role, nodes in target.role_sets.items():
            recovered[nodes] = int(role)
        if not torch.equal(recovered, predicted_roles):
            raise RuntimeError("audit target role sets diverged from R_artifact (canonicalization leak)")

    C0 = sample["copy"][:, None].eq(sample["copy"][None, :]).float()
    B = molecular_bond_tensor(sample)
    R_pred = r_onehot(predicted_roles, m)
    R_truth = r_onehot(truth, m)
    predicted_bonds = projected_bond(R_pred, C0, B)
    true_bonds = projected_bond(R_truth, C0, B)
    p, t = predicted_bonds.bool(), true_bonds.bool()
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    bond_f1 = 2 * precision * recall / max(1e-30, precision + recall)

    c_star_equals_c0 = None
    if target is not None:
        G_star = permutations_to_group(target)
        c_star_equals_c0 = bool(torch.equal(G_star @ G_star.T, C0))

    payload = {
        "sample_id": sample["id"],
        "split": sample["split"],
        "source_artifact": str(ARTIFACT_PATH),
        "classification": audit.status,
        "structural_r_error": bool(audit.structural_r_error),
        "role_capacity_valid": bool(audit.role_capacity_valid),
        "element_compatible": bool(audit.element_compatible),
        # Literal metrics: diagnostic only (not a primary PASS gate).
        "literal_metrics_scope": "DIAGNOSTIC_ONLY",
        "literal_role_accuracy": float(audit.literal_accuracy),
        "literal_exact_r": bool(audit.literal_exact),
        # Physical / gauge metrics.
        "orbit_role_accuracy": float(audit.orbit_role_accuracy),
        "orbit_role_exact": bool(audit.orbit_role_exact),
        "per_copy_automorphism_equivalent": bool(audit.per_copy_automorphism_equivalent),
        "physical_role_assignment_exact": bool(audit.physical_role_assignment_exact),
        "target_defined": bool(audit.target_defined),
        "target_reason": audit.target_reason,
        "c_star_equals_c0": c_star_equals_c0,
        "r_effective_equals_artifact": True,
        "canonicalization_applied": False,
        "projected_bond_precision": precision,
        "projected_bond_recall": recall,
        "projected_bond_f1": bond_f1,
        "projected_molecular_graph_exact": bool(torch.equal(p, t)),
        "role_sizes": list(audit.role_sizes),
        "representation": {
            "R_shape_labels": [int(sample["N"])],
            "R_onehot_shape": [int(sample["N"]), m],
            "capacity_K": k,
            "orbit_collapse": False,
        },
    }
    OUTPUT.joinpath("geometry_only_r_audit.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
