"""Decode exact tree-MAP permutation states into G and C."""
from __future__ import annotations

import torch

from .targets import AssemblyTarget, permutations_to_group


def states_to_permutations(target: AssemblyTarget, permutation_table: torch.Tensor, state_indices: dict[int,int]) -> dict[int,torch.Tensor]:
    if set(state_indices) != set(range(target.M)):
        raise ValueError("MAP state indices must cover every molecular role")
    output={role:permutation_table[index] for role,index in state_indices.items()}
    if not torch.equal(output[target.anchor_role],torch.arange(target.K,device=permutation_table.device)):
        raise AssertionError("decoded anchor state is not identity")
    return output


def decode_copy_assembly(target: AssemblyTarget, permutation_table: torch.Tensor, state_indices: dict[int,int]) -> tuple[torch.Tensor,torch.Tensor]:
    G=permutations_to_group(target,states_to_permutations(target,permutation_table,state_indices))
    C=G@G.T
    if not torch.equal(C,C.T) or not torch.equal(torch.diag(C),torch.ones(len(C),device=C.device)):
        raise AssertionError("decoded C violates symmetry or diagonal invariants")
    return G,C
