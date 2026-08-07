#!/usr/bin/env python3
"""Audit a geometry-only hard-R artifact against structural and automorphism constraints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mattergen.assignment.global_copy_assembly.targets import build_assembly_target_from_predicted_roles

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
    predicted_roles = torch.tensor(artifact["role_assignment"], dtype=torch.long)
    truth = sample["role"].long()
    m = int(sample["M"])
    k = int(sample["Z"])
    if predicted_roles.numel() != int(sample["N"]):
        raise ValueError("predicted role artifact has the wrong atom count")
    if int(predicted_roles.min()) < 0 or int(predicted_roles.max()) >= m:
        raise ValueError("predicted roles contain out-of-range role labels")

    role_sets = {role: (predicted_roles == role).nonzero().flatten().sort().values for role in range(m)}
    role_capacity_valid = all(len(nodes) == k for nodes in role_sets.values())
    element_compatible = bool(torch.equal(torch.sort(sample["role_z"][predicted_roles]).values, torch.sort(sample["z"]).values))
    literal_accuracy = float((predicted_roles == truth).float().mean())

    orbits = [value for _, value in sorted(json.loads(ORBIT_PATH.read_text())["role_orbits"].items(), key=lambda item: int(item[0]))]
    orbit_ok = torch.tensor([int(int(predicted_roles[i])) in orbits[int(truth[i])] for i in range(len(truth))], dtype=torch.bool)
    orbit_role_exact = bool(orbit_ok.all())

    perms = json.loads(AUT_PATH.read_text())["permutations"]
    per_copy_automorphism_equivalent = True
    copy = sample["copy"]
    for c in range(int(copy.max()) + 1):
        idx = (copy == c).nonzero().flatten()
        if len(idx) == 0:
            continue
        truth_block = truth[idx]
        block_ok = any(torch.equal(predicted_roles[idx], torch.tensor([perm[int(x)] for x in truth_block], dtype=torch.long)) for perm in perms)
        per_copy_automorphism_equivalent &= block_ok

    R_pred = r_onehot(predicted_roles, m)
    R_truth = r_onehot(truth, m)
    C0 = sample["copy"][:, None].eq(sample["copy"][None, :]).float()
    B = molecular_bond_tensor(sample)
    predicted_bonds = projected_bond(R_pred, C0, B)
    true_bonds = projected_bond(R_truth, C0, B)
    p, t = predicted_bonds.bool(), true_bonds.bool()
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    bond_f1 = 2 * precision * recall / max(1e-30, precision + recall)

    target, audit = build_assembly_target_from_predicted_roles(
        predicted_roles,
        sample["copy"],
        M=m,
        K=k,
        anchor_role=0,
        role_z=sample["role_z"],
        z=sample["z"],
        oracle_role=truth,
    )
    classification = "GAUGE_EQUIVALENT_R" if audit.target_defined and not audit.structural_r_error else "STRUCTURALLY_INCORRECT_R"
    payload = {
        "sample_id": sample["id"],
        "split": sample["split"],
        "source_artifact": str(ARTIFACT_PATH),
        "classification": classification,
        "structural_r_error": bool(audit.structural_r_error),
        "role_capacity_valid": role_capacity_valid,
        "element_compatible": element_compatible,
        "literal_role_accuracy": literal_accuracy,
        "orbit_role_accuracy": float(orbit_ok.float().mean()),
        "orbit_role_exact": orbit_role_exact,
        "per_copy_automorphism_equivalent": per_copy_automorphism_equivalent,
        "target_defined": bool(audit.target_defined),
        "target_reason": audit.target_reason,
        "projected_bond_precision": precision,
        "projected_bond_recall": recall,
        "projected_bond_f1": bond_f1,
        "projected_molecular_graph_exact": bool(torch.equal(p, t)),
        "role_sizes": [int(nodes.numel()) for nodes in role_sets.values()],
    }
    OUTPUT.joinpath("geometry_only_r_audit.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
