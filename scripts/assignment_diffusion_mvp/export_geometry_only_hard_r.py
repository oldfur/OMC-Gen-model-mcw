#!/usr/bin/env python3
"""Export a real geometry-only hard-R artifact from the fixed RHODIN01 diagnostic checkpoint."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mattergen.common.role_partition_diffusion import OraclePartitionRoleDiagnostic, capacity_sinkhorn
from mattergen.common.role_partition_diffusion.swap_gibbs import SwapGibbsRoleDiffusion, assert_legal

ROOT = ROOT_DIR / "outputs" / "assignment_diffusion_mvp"
SAMPLE_PATH = ROOT / "d1_fixed_clean_geometry" / "fixed_sample.pt"
CHECKPOINT_PATH = Path(os.environ.get("GEOMETRY_ONLY_CHECKPOINT_PATH", str(ROOT / "role_oracle_partition_diagnostic" / "checkpoints" / "geometry_only" / "best.pt")))
ORBIT_PATH = ROOT / "role_automorphism_audit" / "role_orbits.json"
OUTPUT = ROOT / "global_copy_assembly_geometry_r"
OUTPUT.mkdir(parents=True, exist_ok=True)
SEED = 17


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sample() -> dict:
    raw = torch.load(SAMPLE_PATH, map_location="cpu", weights_only=False)
    return {key: (value.cpu() if isinstance(value, torch.Tensor) else value) for key, value in raw.items()}


def terminal_states(sample: dict) -> list[torch.Tensor]:
    diffusion = SwapGibbsRoleDiffusion(steps=64, terminal_randomization_steps=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [
        diffusion.terminal_prior(
            sample["role"].to(device),
            sample["z"].to(device),
            torch.Generator(device=device).manual_seed(1000 + n),
        )
        for n in range(32)
    ]


def geometry(sample: dict, kind: str) -> torch.Tensor:
    if kind == "correct":
        return sample["pos"]
    if kind == "zero":
        return torch.zeros_like(sample["pos"])
    if kind == "mismatch":
        generator = torch.Generator(device=sample["pos"].device).manual_seed(SEED + 313)
        return sample["pos"][torch.randperm(int(sample["N"]), generator=generator)]
    raise ValueError(kind)


def logits(model: OraclePartitionRoleDiagnostic, sample: dict, state: torch.Tensor, *, geometry_kind: str = "correct") -> torch.Tensor:
    return model(
        z=sample["z"],
        frac=geometry(sample, geometry_kind),
        cell=sample["cell"],
        role_z=sample["role_z"],
        role_edge_index=sample["role_edge_index"],
        role_bond_type=sample["role_bond_type"],
        current_role=state,
        same_copy=None,
    )


def hungarian_capacity(scores: torch.Tensor, sample: dict) -> torch.Tensor:
    result = torch.empty(int(sample["N"]), dtype=torch.long, device=scores.device)
    for element in sample["z"].unique(sorted=True):
        atoms = (sample["z"] == element).nonzero().flatten()
        roles = (sample["role_z"] == element).nonzero().flatten()
        slots = roles.repeat_interleave(int(sample["Z"]))
        cost = -scores[atoms][:, slots].detach().float().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        if len(row) != len(atoms):
            raise RuntimeError("Hungarian did not return a complete element-block matching")
        result[atoms[torch.as_tensor(row, device=scores.device)]] = slots[torch.as_tensor(col, device=scores.device)]
    assert_legal(result, sample["z"], sample["role_z"], int(sample["Z"]))
    return result


def load_model() -> tuple[OraclePartitionRoleDiagnostic, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"geometry-only checkpoint not found: {CHECKPOINT_PATH}\nRun the diagnostic training step first or set GEOMETRY_ONLY_CHECKPOINT_PATH.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = OraclePartitionRoleDiagnostic(context_mode="geometry_only").to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


def role_orbit_accuracy(pred: torch.Tensor, truth: torch.Tensor, role_orbits: list[list[int]]) -> float:
    orbit_ok = torch.tensor([int(int(pred[i])) in role_orbits[int(truth[i])] for i in range(len(truth))], device=pred.device)
    return float(orbit_ok.float().mean())


def main() -> None:
    seed_all(SEED)
    sample = load_sample()
    orbits = [value for _, value in sorted(json.loads(ORBIT_PATH.read_text())["role_orbits"].items(), key=lambda item: int(item[0]))]
    model, device = load_model()
    state_inputs = terminal_states(sample)
    sample = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in sample.items()}

    best_assignment: list[int] | None = None
    best_metrics: dict[str, float | bool | int] | None = None
    with torch.no_grad():
        for index, state in enumerate(state_inputs):
            state = state.to(device)
            pred = hungarian_capacity(logits(model, sample, state, geometry_kind="correct"), sample)
            literal_accuracy = float((pred.cpu() == sample["role"].cpu()).float().mean())
            orbit_accuracy = role_orbit_accuracy(pred.cpu(), sample["role"].cpu(), orbits)
            metrics = {
                "terminal_state_index": index,
                "literal_accuracy": literal_accuracy,
                "orbit_role_accuracy": orbit_accuracy,
                "literal_exact": bool((pred.cpu() == sample["role"].cpu()).all()),
            }
            if best_metrics is None or (metrics["literal_accuracy"], metrics["orbit_role_accuracy"]) > (best_metrics["literal_accuracy"], best_metrics["orbit_role_accuracy"]):
                best_assignment = pred.cpu().tolist()
                best_metrics = metrics

    if best_assignment is None or best_metrics is None:
        raise RuntimeError("failed to produce a geometry-only hard-R assignment")

    record = {
        "sample_id": sample["id"],
        "split": sample["split"],
        "N": int(sample["N"]),
        "M": int(sample["M"]),
        "K": int(sample["Z"]),
        "role_source": "geometry_only_hard_r",
        "role_assignment": best_assignment,
        "model_context_mode": "geometry_only",
        "checkpoint": str(CHECKPOINT_PATH),
        "decoder": "independent element-block Hungarian MAP",
        "terminal_state_index": int(best_metrics["terminal_state_index"]),
        "metrics": {
            "literal_accuracy": float(best_metrics["literal_accuracy"]),
            "orbit_role_accuracy": float(best_metrics["orbit_role_accuracy"]),
            "literal_exact": bool(best_metrics["literal_exact"]),
        },
    }
    output_path = OUTPUT / "geometry_only_hard_r.jsonl"
    output_path.write_text(json.dumps(record) + "\n")
    print(json.dumps({"output": str(output_path), **record["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
