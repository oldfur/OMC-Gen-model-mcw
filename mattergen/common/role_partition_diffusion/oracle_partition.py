"""Fixed-sample oracle copy-partition identifiability diagnostic.

This module is deliberately separate from the R/Q sampler.  Its only purpose
is to test whether a *given* copy-equivalence relation would make a one-step,
capacity-constrained R prediction identifiable.  It never receives numeric
copy IDs and it is not used by production sampling.
"""
from __future__ import annotations

import torch
from torch import nn


def periodic_edges(frac: torch.Tensor, cell: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    delta = frac[None, :, :] - frac[:, None, :]
    delta = delta - torch.round(delta)
    distance = torch.linalg.norm(delta @ cell, dim=-1)
    return (distance > 0) & (distance < cutoff), distance


class ContextCrystalEncoder(nn.Module):
    """PBC-invariant scalar encoder with optional relation-only context.

    ``same_copy`` is an N-by-N equivalence relation, never a copy identifier.
    In ``oracle_same_copy`` mode it produces separate same/inter edge types;
    in ``oracle_copy_local`` mode it only gates message passing.
    """

    def __init__(self, hidden: int = 256, layers: int = 4, rbf_dim: int = 64, cutoff: float = 6.0):
        super().__init__()
        self.atom = nn.Embedding(119, hidden)
        self.register_buffer("centres", torch.linspace(0, cutoff, rbf_dim))
        self.cutoff = cutoff
        self.edge = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden + rbf_dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.node = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )

    def forward(self, z: torch.Tensor, frac: torch.Tensor, cell: torch.Tensor, *, context_mode: str, same_copy: torch.Tensor | None = None) -> torch.Tensor:
        edge, distance = periodic_edges(frac, cell, self.cutoff)
        if context_mode == "oracle_copy_local":
            if same_copy is None:
                raise ValueError("oracle_copy_local requires a same-copy relation")
            edge = edge & same_copy.bool()
        elif context_mode not in {"geometry_only", "oracle_same_copy"}:
            raise ValueError(f"unknown role_context_mode={context_mode!r}")
        rbf = torch.exp(-((distance[..., None] - self.centres) / (self.cutoff / len(self.centres))) ** 2)
        relation = torch.zeros((*edge.shape, 2), dtype=frac.dtype, device=frac.device)
        if context_mode == "oracle_same_copy" and same_copy is not None:
            relation[..., 0] = same_copy
            relation[..., 1] = (~same_copy.bool()).to(frac.dtype)
            relation.diagonal(dim1=0, dim2=1).zero_()
        h = self.atom(z)
        for edge_net, node_net in zip(self.edge, self.node):
            source = h[:, None, :].expand(-1, len(z), -1)
            target = h[None, :, :].expand(len(z), -1, -1)
            message = edge_net(torch.cat([source, target, rbf, relation], dim=-1)) * edge[..., None]
            pooled = message.sum(1) / edge.sum(1, keepdim=True).clamp_min(1)
            h = h + node_net(torch.cat([h, pooled], dim=-1))
        return h


class CapacityRoleHead(nn.Module):
    """Permutation-equivariant compatibility head; no absolute role/node IDs."""

    def __init__(self, hidden: int = 256, layers: int = 4):
        super().__init__()
        self.current = nn.Linear(hidden, hidden)
        self.inp = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(5 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, hx: torch.Tensor, hm: torch.Tensor, current_role: torch.Tensor) -> torch.Tensor:
        atom = hx[:, None, :].expand(-1, len(hm), -1)
        role = hm[None, :, :].expand(len(hx), -1, -1)
        current = self.current(hm[current_role])[:, None, :].expand_as(atom)
        pair = self.inp(torch.cat([atom, role, current, (atom - role).abs(), atom * role], dim=-1))
        for block in self.blocks:
            row_mean, row_max = pair.mean(1), pair.max(1).values
            col_mean, col_max = pair.mean(0), pair.max(0).values
            pair = pair + block(torch.cat([
                pair,
                row_mean[:, None, :].expand_as(pair),
                row_max[:, None, :].expand_as(pair),
                col_mean[None, :, :].expand_as(pair),
                col_max[None, :, :].expand_as(pair),
            ], dim=-1))
        return self.out(pair).squeeze(-1)


def capacity_sinkhorn(scores: torch.Tensor, role_z: torch.Tensor, crystal_z: torch.Tensor, k: int, temperature: float = 0.20, iterations: int = 100) -> torch.Tensor:
    """Generalized element-block Sinkhorn with row marginals 1 and role marginals K.

    Each role is expanded into K unit-capacity slots.  The hidden network never
    sees forbidden logits; the element constraint is imposed by solving each
    compatible element block independently.
    """
    n, m = scores.shape
    result = scores.new_zeros(n, m)
    for element in crystal_z.unique(sorted=True):
        atom_idx = (crystal_z == element).nonzero().flatten()
        role_idx = (role_z == element).nonzero().flatten()
        if len(atom_idx) != len(role_idx) * k:
            raise ValueError("element block does not satisfy capacity N_e=K*M_e")
        slot_role = role_idx.repeat_interleave(k)
        logp = scores[atom_idx][:, slot_role] / temperature
        for _ in range(iterations):
            logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
            logp = logp - torch.logsumexp(logp, dim=0, keepdim=True)
        p = logp.exp()
        for role in role_idx:
            result[atom_idx, role] = p[:, slot_role == role].sum(1)
    return result


class OraclePartitionRoleDiagnostic(nn.Module):
    """One-step structured R predictor for the three oracle-context modes."""

    def __init__(self, *, context_mode: str = "geometry_only", hidden: int = 256, layers: int = 4):
        super().__init__()
        if context_mode not in {"geometry_only", "oracle_same_copy", "oracle_copy_local"}:
            raise ValueError(f"unsupported role_context_mode={context_mode!r}")
        self.context_mode = context_mode
        self.crystal_encoder = ContextCrystalEncoder(hidden=hidden, layers=layers)
        from .molecule_encoder import MolecularGraphEncoder
        self.molecule_encoder = MolecularGraphEncoder(hidden=hidden, layers=layers)
        self.role_head = CapacityRoleHead(hidden=hidden, layers=layers)

    def forward(self, *, z: torch.Tensor, frac: torch.Tensor, cell: torch.Tensor, role_z: torch.Tensor, role_edge_index: torch.Tensor, role_bond_type: torch.Tensor, current_role: torch.Tensor, same_copy: torch.Tensor | None = None) -> torch.Tensor:
        hx = self.crystal_encoder(z, frac, cell, context_mode=self.context_mode, same_copy=same_copy)
        hm = self.molecule_encoder(role_z, role_edge_index, role_bond_type)
        return self.role_head(hx, hm, current_role)
