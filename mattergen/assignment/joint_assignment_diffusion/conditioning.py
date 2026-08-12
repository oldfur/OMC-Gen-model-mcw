"""A-conditioning features: orbit nodes, spatial-edge C/orbit, nonlocal assignment graph."""
from __future__ import annotations

import torch
from torch import nn


class OrbitRelationTable(nn.Module):
    """Learned embedding of orbit-pair molecular relation ρ_{oo'}."""

    def __init__(self, num_orbits: int, dim: int = 32, max_path: int = 16):
        super().__init__()
        self.num_orbits = num_orbits
        self.bond_emb = nn.Embedding(8, dim)
        self.dist_emb = nn.Embedding(max_path, dim)
        self.out = nn.Linear(2 * dim, dim)
        # buffers filled externally
        self.register_buffer("bond_type", torch.zeros(num_orbits, num_orbits, dtype=torch.long))
        self.register_buffer("graph_dist", torch.full((num_orbits, num_orbits), max_path - 1, dtype=torch.long))

    def set_from_role_graph(
        self,
        *,
        partition,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
    ) -> None:
        j = self.num_orbits
        bond = torch.zeros(j, j, dtype=torch.long, device=role_edge_index.device)
        # adjacency on orbits
        adj = torch.zeros(j, j, dtype=torch.float32, device=role_edge_index.device)
        for e in range(role_edge_index.shape[1]):
            r0 = int(role_edge_index[0, e].item())
            r1 = int(role_edge_index[1, e].item())
            o0 = partition.role_to_orbit[r0]
            o1 = partition.role_to_orbit[r1]
            bt = int(role_bond_type[e].item()) if role_bond_type.ndim == 1 else int(role_bond_type[e].reshape(-1)[0])
            bond[o0, o1] = max(int(bond[o0, o1]), min(bt, 7))
            bond[o1, o0] = bond[o0, o1]
            if o0 != o1:
                adj[o0, o1] = 1.0
                adj[o1, o0] = 1.0
        # BFS distances on orbit quotient graph
        dist = torch.full((j, j), self.dist_emb.num_embeddings - 1, dtype=torch.long, device=adj.device)
        for s in range(j):
            dist[s, s] = 0
            q = [s]
            seen = {s}
            while q:
                u = q.pop(0)
                for v in range(j):
                    if adj[u, v] > 0 and v not in seen:
                        nd = int(dist[s, u].item()) + 1
                        if nd < int(dist[s, v].item()):
                            dist[s, v] = nd
                        seen.add(v)
                        q.append(v)
        self.bond_type = bond
        self.graph_dist = dist.clamp(0, self.dist_emb.num_embeddings - 1)

    def forward(self, o_i: torch.Tensor, o_j: torch.Tensor) -> torch.Tensor:
        bt = self.bond_type[o_i, o_j]
        gd = self.graph_dist[o_i, o_j]
        return self.out(torch.cat([self.bond_emb(bt), self.dist_emb(gd)], dim=-1))


class OrbitSiteEncoder(nn.Module):
    """Orbit embeddings z_o from molecular role GINE-like features (lightweight)."""

    def __init__(self, hidden: int = 128, num_elements: int = 128):
        super().__init__()
        self.elem = nn.Embedding(num_elements, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, element_by_orbit: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.elem(element_by_orbit.long().clamp(0, 127)))


class ClockEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor, beta_r: torch.Tensor, beta_g: torch.Tensor, lock_flags: torch.Tensor) -> torch.Tensor:
        # t, beta_r, beta_g, lock_r||lock_g as 4-vector
        x = torch.stack([t, beta_r, beta_g, lock_flags], dim=-1)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class AssignmentGraphMP(nn.Module):
    """Nonlocal same-copy invariant message passing (no distance/angle).

    Vectorized over same-copy ordered pairs (i≠j); math identical to the
    previous per-pair Python loop.
    """

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        # zero-init last residual
        nn.init.zeros_(self.upd[-1].weight)
        nn.init.zeros_(self.upd[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        *,
        copy_of: torch.Tensor,
        orbit_of: torch.Tensor,
        z_orbit: torch.Tensor,
        rho: OrbitRelationTable,
    ) -> torch.Tensor:
        n = h.shape[0]
        device = h.device
        dtype = h.dtype
        msgs = torch.zeros_like(h)
        counts = torch.zeros(n, 1, device=device, dtype=dtype)
        if n <= 1:
            return self.upd(torch.cat([h, msgs], dim=-1))

        copy_of = copy_of.long()
        orbit_of = orbit_of.long()
        # Group by copy id (K is small); form all ordered pairs i≠j within each copy.
        max_k = int(copy_of.max().item()) if copy_of.numel() else -1
        for k in range(max_k + 1):
            idx = (copy_of == k).nonzero(as_tuple=True)[0]
            m = int(idx.numel())
            if m < 2:
                continue
            # Cartesian product of indices in this copy, drop diagonal.
            src = idx.repeat_interleave(m)
            dst = idx.repeat(m)
            keep = src != dst
            src = src[keep]
            dst = dst[keep]
            rel = rho(orbit_of[src], orbit_of[dst])
            feat = torch.cat([h[src], h[dst], rel], dim=-1)
            m_ij = self.msg(feat)
            msgs.index_add_(0, src, m_ij)
            counts.index_add_(
                0, src, torch.ones(src.shape[0], 1, device=device, dtype=dtype)
            )
        msgs = msgs / counts.clamp_min(1.0)
        return self.upd(torch.cat([h, msgs], dim=-1))


class CopyContextPool(nn.Module):
    """Copy contexts v_k without copy-ID embeddings (vectorized index_add)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.psi = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.phi = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(
        self,
        h: torch.Tensor,
        orbit_of: torch.Tensor,
        copy_of: torch.Tensor,
        z_orbit: torch.Tensor,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, hid = h.shape
        device = h.device
        dtype = h.dtype
        k = copy_of.long()
        o = orbit_of.long()
        feat = torch.cat([h, z_orbit[o]], dim=-1)
        psi = self.psi(feat)
        v = torch.zeros(K, hid, device=device, dtype=dtype)
        counts = torch.zeros(K, 1, device=device, dtype=dtype)
        if n > 0:
            v.index_add_(0, k, psi)
            counts.index_add_(0, k, torch.ones(n, 1, device=device, dtype=dtype))
        v = v / counts.clamp_min(1.0)
        c_i = v[k] if n > 0 else torch.zeros(0, hid, device=device, dtype=dtype)
        v_a = self.phi(v.mean(0, keepdim=False))
        return v, c_i, v_a


class SpatialEdgeAssignmentFeaturizer(nn.Module):
    """Scalar features for GemNet PBC edges from A (C, orbits, rho, clocks)."""

    def __init__(self, hidden: int, rho_dim: int = 32, clock_dim: int = 64):
        super().__init__()
        self.orbit_emb = nn.Embedding(64, 32)
        self.proj = nn.Sequential(
            nn.Linear(1 + 32 * 2 + rho_dim + clock_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        *,
        edge_index: torch.Tensor,
        C: torch.Tensor,
        orbit_of: torch.Tensor,
        rho: OrbitRelationTable,
        clock: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        cij = C[src, dst].unsqueeze(-1)
        oi, oj = orbit_of[src], orbit_of[dst]
        zo = torch.cat([self.orbit_emb(oi.clamp(0, 63)), self.orbit_emb(oj.clamp(0, 63))], dim=-1)
        rel = rho(oi, oj)
        clk = clock.expand(src.shape[0], -1) if clock.ndim == 1 else clock[0].expand(src.shape[0], -1)
        return self.proj(torch.cat([cij, zo, rel, clk], dim=-1))
