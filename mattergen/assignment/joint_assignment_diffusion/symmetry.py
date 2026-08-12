"""Symmetry augmentation: element-wise atom perm + copy-column perm."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import JointAssignmentState


@dataclass
class SymmetryAugment:
    atom_perm: torch.Tensor  # [N] long
    copy_perm: torch.Tensor  # [K] long


def sample_symmetry_augment(
    *,
    atomic_numbers: torch.Tensor,
    K: int,
    generator: torch.Generator | None = None,
) -> SymmetryAugment:
    n = int(atomic_numbers.numel())
    device = atomic_numbers.device
    atom_perm = torch.arange(n, device=device)
    # permute within each element block
    for z in torch.unique(atomic_numbers):
        idx = (atomic_numbers == z).nonzero(as_tuple=False).flatten()
        if idx.numel() <= 1:
            continue
        if generator is None:
            local = torch.randperm(idx.numel(), device=device)
        else:
            local = torch.randperm(idx.numel(), generator=generator)
        atom_perm[idx] = idx[local]
    if generator is None:
        copy_perm = torch.randperm(K, device=device)
    else:
        copy_perm = torch.randperm(K, generator=generator)
    return SymmetryAugment(atom_perm=atom_perm, copy_perm=copy_perm)


def apply_symmetry_to_state(state: JointAssignmentState, aug: SymmetryAugment) -> JointAssignmentState:
    """Permute atoms then copy columns: same physical crystal, different gauge."""
    A = state.A[aug.atom_perm]  # atom axis
    A = A[:, :, aug.copy_perm]  # copy columns
    return JointAssignmentState(
        A=A,
        partition=state.partition,
        atomic_numbers=state.atomic_numbers[aug.atom_perm],
        element_by_orbit=state.element_by_orbit,
    )


def apply_symmetry_to_geometry(
    *,
    frac: torch.Tensor,
    atomic_numbers: torch.Tensor,
    role: torch.Tensor | None,
    copy: torch.Tensor | None,
    aug: SymmetryAugment,
) -> dict[str, torch.Tensor]:
    out = {
        "pos": frac[aug.atom_perm],
        "z": atomic_numbers[aug.atom_perm],
    }
    if role is not None:
        out["role"] = role[aug.atom_perm]
    if copy is not None:
        # copy labels remap by inverse of copy_perm
        inv = torch.empty_like(aug.copy_perm)
        inv[aug.copy_perm] = torch.arange(aug.copy_perm.numel(), device=aug.copy_perm.device)
        out["copy"] = inv[copy[aug.atom_perm]]
    return out
