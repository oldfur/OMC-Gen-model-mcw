"""Clean-PBC atom-pair potentials for molecular tree edges."""
from __future__ import annotations

import torch
from torch import nn


class BondPairPotential(nn.Module):
    """Maps K-by-K role-instance pairs to scalar molecular-bond compatibilities.

    The forward API contains no copy relation, copy identifier, Q target or
    canonical-role index embedding.  Role identity is represented only by the
    permutation-equivariant molecular encoder output supplied as ``h_role``.
    """
    def __init__(self, hidden: int = 256, pair_hidden: int | None = None, rbf_dim: int = 32, bond_types: int = 8, cutoff: float = 6.0):
        super().__init__()
        self.bond_embedding = nn.Embedding(bond_types, hidden)
        self.register_buffer("centres", torch.linspace(0, cutoff, rbf_dim))
        self.cutoff = cutoff
        pair_hidden=hidden if pair_hidden is None else pair_hidden
        self.net = nn.Sequential(nn.Linear(8 * hidden + rbf_dim, pair_hidden), nn.SiLU(), nn.Linear(pair_hidden, pair_hidden), nn.SiLU(), nn.Linear(pair_hidden, 1))

    def forward(self, h_left: torch.Tensor, h_right: torch.Tensor, h_role_left: torch.Tensor, h_role_right: torch.Tensor, bond_type: int | torch.Tensor, frac_left: torch.Tensor, frac_right: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if h_left.ndim != 2 or h_right.ndim != 2 or h_left.shape != h_right.shape:
            raise ValueError("tree pair potential expects equal [K,H] instance embeddings")
        if frac_left.shape != frac_right.shape or frac_left.shape != (len(h_left), 3):
            raise ValueError("role-instance positions must have shape [K,3]")
        delta=frac_right[None,:,:]-frac_left[:,None,:]; delta=delta-torch.round(delta)
        distance=torch.linalg.norm(delta@cell,dim=-1)
        rbf=torch.exp(-((distance[...,None]-self.centres)/(self.cutoff/max(1,len(self.centres))))**2)
        left=h_left[:,None,:].expand(-1,len(h_right),-1); right=h_right[None,:,:].expand(len(h_left),-1,-1)
        bond=self.bond_embedding(torch.as_tensor(bond_type,device=h_left.device).long().clamp(0,self.bond_embedding.num_embeddings-1)).view(1,1,-1).expand_as(left)
        roles=torch.cat([h_role_left,h_role_right],-1).view(1,1,-1).expand(len(h_left),len(h_right),-1)
        features=torch.cat([left,right,left+right,(left-right).abs(),left*right,roles,bond,rbf],-1)
        return self.net(features).squeeze(-1)


def permutation_factor(pair_score: torch.Tensor, permutations: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """Build Phi[parent_state, child_state] without repeated inverse operations.

    For parent state ``u`` and child state ``v``, the child instance matching
    parent instance q is ``inverse[v, u[q]]`` under P_r[q]=copy-label.
    """
    if pair_score.ndim != 2 or pair_score.shape[0] != pair_score.shape[1]:
        raise ValueError("pair score must have shape [K,K]")
    states,k=permutations.shape
    if inverse.shape != permutations.shape or pair_score.shape[0] != k:
        raise ValueError("factor shape is incompatible with the permutation table")
    labels=permutations[:,None,:].expand(states,states,k)
    child_inverse=inverse[None,:,:].expand(states,states,k)
    child_instance=torch.gather(child_inverse,2,labels)
    parent_instance=torch.arange(k,device=pair_score.device).view(1,1,k).expand_as(child_instance)
    return pair_score[parent_instance,child_instance].sum(-1)
