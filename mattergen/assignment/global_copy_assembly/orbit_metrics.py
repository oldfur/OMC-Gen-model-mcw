"""Evaluation metrics for orbit-aware O2 assembly."""
from __future__ import annotations

import torch

from .metrics import pair_partition_metrics, projected_bond_metrics, projected_molecular_bonds
from .orbit_membership import OrbitPartition, validate_orbit_copy_capacity


def adjusted_rand_index(labels_true: torch.Tensor, labels_pred: torch.Tensor) -> float:
    """ARI for two integer labelings of the same N points (no sklearn)."""
    y = labels_true.long().flatten()
    p = labels_pred.long().flatten()
    if y.numel() != p.numel():
        raise ValueError("label length mismatch")
    n = int(y.numel())
    if n == 0:
        return 1.0
    # contingency
    yt = {int(v): i for i, v in enumerate(torch.unique(y).tolist())}
    pt = {int(v): i for i, v in enumerate(torch.unique(p).tolist())}
    cont = torch.zeros(len(yt), len(pt), dtype=torch.float64)
    for a, b in zip(y.tolist(), p.tolist()):
        cont[yt[int(a)], pt[int(b)]] += 1.0
    sum_comb_c = torch.sum(cont * (cont - 1) / 2)
    row = cont.sum(1)
    col = cont.sum(0)
    sum_comb_row = torch.sum(row * (row - 1) / 2)
    sum_comb_col = torch.sum(col * (col - 1) / 2)
    comb_n = n * (n - 1) / 2
    if comb_n == 0:
        return 1.0
    expected = sum_comb_row * sum_comb_col / comb_n
    max_index = 0.5 * (sum_comb_row + sum_comb_col)
    denom = max_index - expected
    if abs(float(denom)) < 1e-12:
        return 1.0 if float(sum_comb_c) == float(expected) else 0.0
    return float((sum_comb_c - expected) / denom)


def copy_labels_from_G(G: torch.Tensor) -> torch.Tensor:
    return G.argmax(-1)


def complete_copy_rate(G: torch.Tensor, copy: torch.Tensor) -> float:
    true_sets = []
    for c in range(int(copy.max().item()) + 1):
        atoms = (copy == c).nonzero(as_tuple=False).flatten().sort().values
        if atoms.numel():
            true_sets.append(atoms)
    complete = 0
    for k in range(G.shape[1]):
        atoms = (G[:, k] > 0.5).nonzero(as_tuple=False).flatten().sort().values
        if any(torch.equal(atoms, t) for t in true_sets):
            complete += 1
    return complete / max(1, G.shape[1])


def evaluate_orbit_assembly(
    *,
    G: torch.Tensor,
    C: torch.Tensor,
    copy: torch.Tensor,
    bar_r: torch.Tensor,
    partition: OrbitPartition,
    role_assignment_for_projection: torch.Tensor,
    role_edge_index: torch.Tensor,
    role_bond_type: torch.Tensor,
    M: int,
) -> dict[str, object]:
    """Gauge-invariant C metrics + orbit capacity + projected bonds via R for projection.

    Projected bonds use the provided role_assignment (typically oracle R0 for a
    pure-C test, or predicted R with Aut-orbit-safe interpretation). Literal
    1/2 identity is not a PASS gate.
    """
    C0 = copy[:, None].eq(copy[None, :]).float().to(C.device)
    pair = pair_partition_metrics(C, C0)
    pred_labels = copy_labels_from_G(G)
    ari = adjusted_rand_index(copy.to(pred_labels.device), pred_labels)
    cap = validate_orbit_copy_capacity(G, bar_r.to(G.device), partition)
    # molecular bond tensor
    types = int(role_bond_type.max().item()) + 1
    B = torch.zeros(M, M, types, device=C.device)
    for (left, right), bond in zip(role_edge_index.T.tolist(), role_bond_type.tolist()):
        B[left, right, int(bond)] = 1.0
    # Projected bonds evaluate C under a fixed molecular R (typically oracle R0).
    # Literal roles 1/2 identity is not a PASS gate for assembly success.
    if role_assignment_for_projection.ndim == 1:
        R_proj = torch.nn.functional.one_hot(role_assignment_for_projection.long(), M).float().to(C.device)
    else:
        R_proj = role_assignment_for_projection.float().to(C.device)
    pred_bonds = projected_molecular_bonds(R_proj, C, B)
    true_bonds = projected_molecular_bonds(R_proj, C0, B)
    bond = projected_bond_metrics(pred_bonds, true_bonds, C0)
    rate = complete_copy_rate(G, copy.to(G.device))
    return {
        **pair,
        "ARI": ari,
        "orbit_copy_capacity_valid": cap["valid"],
        "orbit_copy_capacity": cap,
        "complete_copy_rate": rate,
        "copy_graph_isomorphism_rate": 1.0 if bond["projected_molecular_graph_exact"] else float(rate),
        **bond,
        "row_capacity_valid": bool(torch.allclose(G.sum(-1), torch.ones(G.shape[0], device=G.device))),
        "group_size_valid": bool(torch.allclose(G.sum(0), torch.full((G.shape[1],), float(G.shape[0] // G.shape[1]), device=G.device))),
    }
