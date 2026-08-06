"""The single permutation convention used by global copy assembly.

``P_r[q] = k`` means that crystal instance ``q`` in the stable role-instance
list ``V_r`` belongs to copy-gauge label ``k``.  Rows are instances; values are
copy labels.  This convention is used in targets, CRF factors and decoding.
"""
from __future__ import annotations

import itertools
import torch


def enumerate_permutations(k: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"K must be a positive integer, got {k!r}")
    return torch.tensor(list(itertools.permutations(range(k))), dtype=torch.long, device=device)


def inverse_permutations(permutations: torch.Tensor) -> torch.Tensor:
    if permutations.ndim != 2:
        raise ValueError(f"permutations must have shape [S,K], got {tuple(permutations.shape)}")
    s, k = permutations.shape
    expected = torch.arange(k, device=permutations.device).expand(s, k)
    if not torch.equal(torch.sort(permutations, dim=-1).values, expected):
        raise ValueError("every permutation state must be a bijection over 0..K-1")
    inverse = torch.empty_like(permutations)
    inverse.scatter_(1, permutations, torch.arange(k, device=permutations.device).expand(s, k))
    return inverse


def identity_index(permutations: torch.Tensor) -> int:
    identity = torch.arange(permutations.shape[1], device=permutations.device)
    found = (permutations == identity).all(dim=1).nonzero().flatten()
    if len(found) != 1:
        raise ValueError("permutation table must contain exactly one identity state")
    return int(found.item())


def compose(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return ``left[right[q]]`` under the shared instance->copy convention."""
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("composition expects two K-vector permutations")
    return left[right]
