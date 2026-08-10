"""Capacity-constrained hard orbit MAP (element-compatible)."""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from mattergen.assignment.global_copy_assembly.orbit_membership import OrbitPartition


def orbit_capacity_map(
    logits: torch.Tensor,
    partition: OrbitPartition,
    *,
    K: int,
    z: torch.Tensor | None = None,
    role_z: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hungarian-style capacity MAP for orbit labels.

    Parameters
    ----------
    logits:
        ``[N, J]`` unnormalized scores for orbit membership.
    Returns
    -------
    labels:
        ``[N]`` long tensor of orbit indices with ``count(o)=K*|o|``.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be [N,J]")
    n, j = logits.shape
    if j != partition.J:
        raise ValueError(f"logits J={j} != partition.J={partition.J}")
    # Build slots: for each orbit o, create K*|o| identical slots
    slots: list[int] = []
    for o, size in enumerate(partition.orbit_sizes):
        slots.extend([o] * (K * size))
    if len(slots) != n:
        raise ValueError(f"slot count {len(slots)} != N={n}")
    slot_t = torch.tensor(slots, dtype=torch.long, device=logits.device)
    # cost = -score[atom, slot_orbit]
    cost = -logits[:, slot_t].detach().float().cpu().numpy()
    row, col = linear_sum_assignment(cost)
    if len(row) != n:
        raise RuntimeError("Hungarian incomplete for orbit capacity MAP")
    labels = torch.empty(n, dtype=torch.long, device=logits.device)
    labels[torch.as_tensor(row, device=logits.device)] = slot_t[torch.as_tensor(col, device=logits.device)]
    # verify capacities
    for o, size in enumerate(partition.orbit_sizes):
        if int((labels == o).sum()) != K * size:
            raise AssertionError("orbit capacity MAP violated capacity")
    return labels


def labels_to_bar_r(labels: torch.Tensor, j: int) -> torch.Tensor:
    n = labels.numel()
    bar = torch.zeros(n, j, device=labels.device, dtype=torch.float32)
    bar[torch.arange(n, device=labels.device), labels.long()] = 1.0
    return bar
