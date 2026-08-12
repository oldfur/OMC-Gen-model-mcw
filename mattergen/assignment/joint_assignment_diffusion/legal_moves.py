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
    orbit = state.orbit_of()
    copy = state.copy_of()
    z = state.atomic_numbers
    moves: list[LegalMove] = []
    for i in range(n):
        for j in range(i + 1, n):
            if int(copy[i]) != int(copy[j]):
                continue
            if int(z[i]) != int(z[j]):
                continue
            if int(orbit[i]) == int(orbit[j]):
                continue
            moves.append(LegalMove("R", i, j))
    return moves


def enumerate_g_moves(state: JointAssignmentState) -> list[LegalMove]:
    """G-move: same orbit, different copy → swap copy membership.

    o_i = o_j, k_i ≠ k_j.
    """
    n = state.N
    orbit = state.orbit_of()
    copy = state.copy_of()
    moves: list[LegalMove] = []
    for i in range(n):
        for j in range(i + 1, n):
            if int(orbit[i]) != int(orbit[j]):
                continue
            if int(copy[i]) == int(copy[j]):
                continue
            moves.append(LegalMove("G", i, j))
    return moves


def enumerate_legal_moves(state: JointAssignmentState) -> dict[str, list[LegalMove]]:
    return {"R": enumerate_r_moves(state), "G": enumerate_g_moves(state)}


def apply_move(state: JointAssignmentState, move: LegalMove) -> JointAssignmentState:
    return state.apply_swap(move.i, move.j)
