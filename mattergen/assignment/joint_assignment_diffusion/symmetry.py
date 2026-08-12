"""Symmetry augmentation: element-wise atom perm + copy-column perm."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import JointAssignmentState


@dataclass
class SymmetryAugment:
    atom_perm: torch.Tensor  # [N] long
    copy_perm: torch.Tensor  # [K] long


def _randperm(n: int, *, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
    """torch.randperm with generator always materializes on CPU; move to ``device``."""
    if generator is None:
        return torch.randperm(n, device=device)
    return torch.randperm(n, generator=generator).to(device=device)


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
        local = _randperm(int(idx.numel()), device=device, generator=generator)
        atom_perm[idx] = idx[local]
    copy_perm = _randperm(int(K), device=device, generator=generator)
    return SymmetryAugment(atom_perm=atom_perm, copy_perm=copy_perm)


def apply_symmetry_to_state(state: JointAssignmentState, aug: SymmetryAugment) -> JointAssignmentState:
    """Permute atoms then copy columns: same physical crystal, different gauge."""
    device = state.A.device
    atom_perm = aug.atom_perm.to(device=device)
    copy_perm = aug.copy_perm.to(device=device)
    A = state.A[atom_perm]  # atom axis
    A = A[:, :, copy_perm]  # copy columns
    return JointAssignmentState(
        A=A,
        partition=state.partition,
        atomic_numbers=state.atomic_numbers[atom_perm],
        element_by_orbit=state.element_by_orbit.to(device=device),
    )


def apply_symmetry_to_geometry(
    *,
    frac: torch.Tensor,
    atomic_numbers: torch.Tensor,
    role: torch.Tensor | None,
    copy: torch.Tensor | None,
    aug: SymmetryAugment,
) -> dict[str, torch.Tensor]:
    device = frac.device
    atom_perm = aug.atom_perm.to(device=device)
    copy_perm = aug.copy_perm.to(device=device)
    out = {
        "pos": frac[atom_perm],
        "z": atomic_numbers[atom_perm],
    }
    if role is not None:
        out["role"] = role.to(device=device)[atom_perm]
    if copy is not None:
        # copy labels remap by inverse of copy_perm
        inv = torch.empty_like(copy_perm)
        inv[copy_perm] = torch.arange(copy_perm.numel(), device=device)
        out["copy"] = inv[copy.to(device=device)[atom_perm]]
    return out
