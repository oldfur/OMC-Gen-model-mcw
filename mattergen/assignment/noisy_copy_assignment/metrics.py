"""N1 evaluation metrics aggregation."""
from __future__ import annotations

import torch

from mattergen.assignment.global_copy_assembly.orbit_membership import OrbitPartition
from mattergen.assignment.global_copy_assembly.orbit_metrics import evaluate_orbit_assembly
from .soft_c import soft_c_metrics


def orbit_accuracy(pred_bar: torch.Tensor, oracle_bar: torch.Tensor) -> dict[str, float | bool]:
    pred = pred_bar.argmax(-1)
    truth = oracle_bar.argmax(-1)
    acc = float((pred == truth).float().mean())
    exact = bool(torch.equal(pred, truth))
    return {"orbit_atom_accuracy": acc, "orbit_exact": exact}


def evaluate_n1_once(
    *,
    output,
    sample: dict,
    oracle_bar: torch.Tensor,
    partition: OrbitPartition,
) -> dict:
    metrics = evaluate_orbit_assembly(
        G=output.group_map,
        C=output.c_map,
        copy=sample["copy"],
        bar_r=output.orbit_map,
        partition=partition,
        role_assignment_for_projection=sample["role"],
        role_edge_index=sample["role_edge_index"],
        role_bond_type=sample["role_bond_type"],
        M=int(sample["M"]),
    )
    orbit = orbit_accuracy(output.orbit_map, oracle_bar.to(output.orbit_map.device))
    soft = {}
    if output.c_soft is not None:
        C0 = sample["copy"][:, None].eq(sample["copy"][None, :]).float().to(output.c_soft.device)
        soft = soft_c_metrics(output.c_soft, C0)
    # structured confidence from diagnostics
    conf = {}
    om = (output.diagnostics or {}).get("orbit_attachment_margins") or {}
    conf["orbit_attachment_MAP_gap"] = om.get("attachment_MAP_gap")
    conf["orbit_target_log_probability"] = om.get("target_log_probability")
    conf["structured_entropy"] = om.get("entropy")
    conf["number_of_ties"] = om.get("number_of_exact_ties")
    conf["singleton_tree_energy"] = output.diagnostics.get("singleton_tree_energy")
    return {**metrics, **orbit, **soft, **conf}
