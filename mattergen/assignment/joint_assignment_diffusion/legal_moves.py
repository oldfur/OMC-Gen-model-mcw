"""Legal R-move and G-move enumeration for joint assignment CTMC."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import JointAssignmentState


@dataclass(frozen=True)
class LegalMove:
    kind: str  # "R" or "G"
    i: int
    j: int

    def as_tuple(self) -> tuple[str, int, int]:
        a, b = (self.i, self.j) if self.i < self.j else (self.j, self.i)
        return (self.kind, a, b)


def enumerate_r_moves(state: JointAssignmentState) -> list[LegalMove]:
    """R-move: same copy, same element, different orbit → swap orbit labels.

    k_i = k_j, z_i = z_j, o_i ≠ o_j.
    """
    n = state.N
    if n < 2:
        return []
    orbit = state.orbit_of()
    copy = state.copy_of()
    z = state.atomic_numbers
    device = state.A.device
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    mask = (copy[ii] == copy[jj]) & (z[ii] == z[jj]) & (orbit[ii] != orbit[jj])
    ii = ii[mask]
    jj = jj[mask]
    if ii.numel() == 0:
        return []
    return [LegalMove("R", int(i), int(j)) for i, j in zip(ii.tolist(), jj.tolist())]


def enumerate_g_moves(state: JointAssignmentState) -> list[LegalMove]:
    """G-move: same orbit, different copy → swap copy membership.

    o_i = o_j, k_i ≠ k_j.
    """
    n = state.N
    if n < 2:
        return []
    orbit = state.orbit_of()
    copy = state.copy_of()
    device = state.A.device
    ii, jj = torch.triu_indices(n, n, offset=1, device=device)
    mask = (orbit[ii] == orbit[jj]) & (copy[ii] != copy[jj])
    ii = ii[mask]
    jj = jj[mask]
    if ii.numel() == 0:
        return []
    return [LegalMove("G", int(i), int(j)) for i, j in zip(ii.tolist(), jj.tolist())]


def enumerate_legal_moves(state: JointAssignmentState) -> dict[str, list[LegalMove]]:
    return {"R": enumerate_r_moves(state), "G": enumerate_g_moves(state)}


def apply_move(state: JointAssignmentState, move: LegalMove) -> JointAssignmentState:
    return state.apply_swap(move.i, move.j)
