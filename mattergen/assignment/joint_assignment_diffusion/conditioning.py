"""A-conditioning features: orbit nodes, spatial-edge C/orbit, nonlocal assignment graph."""
from __future__ import annotations

import torch
from torch import nn

from mattergen.assignment.global_copy_assembly.pair_potential import (
    gaussian_radial_basis,
    pbc_minimum_image_distance,
)


class OrbitRelationTable(nn.Module):
    """Learned embedding of orbit-pair molecular relation ρ_{oo'}.

    Discrete template buffers (bond existence / type / multiplicity / binned
    graph distance) are automorphism-orbit aggregates of the clean molecular
    role graph.  ``forward`` is the historical R/GemNet embedding and is left
    unchanged.  B3 G-policy reads the raw buffers via ``raw_template`` and
    embeds them with a G-specific encoder.
    """

    # dist_bin: 0=self, 1, 2, 3, 4=reachable ≥4, 5=disconnected
    DIST_BIN_DISCONNECTED = 5

    def __init__(self, num_orbits: int, dim: int = 32, max_path: int = 16):
        super().__init__()
        self.num_orbits = num_orbits
        self.bond_emb = nn.Embedding(8, dim)
        self.dist_emb = nn.Embedding(max_path, dim)
        self.out = nn.Linear(2 * dim, dim)
        # buffers filled externally
        self.register_buffer("bond_type", torch.zeros(num_orbits, num_orbits, dtype=torch.long))
        self.register_buffer("graph_dist", torch.full((num_orbits, num_orbits), max_path - 1, dtype=torch.long))
        self.register_buffer("bond_exists", torch.zeros(num_orbits, num_orbits))
        self.register_buffer("edge_mult", torch.zeros(num_orbits, num_orbits))
        self.register_buffer(
            "dist_bin", torch.full((num_orbits, num_orbits), self.DIST_BIN_DISCONNECTED, dtype=torch.long)
        )

    def set_from_role_graph(
        self,
        *,
        partition,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
    ) -> None:
        j = self.num_orbits
        device = role_edge_index.device
        bond = torch.zeros(j, j, dtype=torch.long, device=device)
        # adjacency on orbits
        adj = torch.zeros(j, j, dtype=torch.float32, device=device)
        edge_mult = torch.zeros(j, j, dtype=torch.float32, device=device)
        seen_undirected: set[tuple[int, int]] = set()
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
            key = (min(r0, r1), max(r0, r1))
            if key in seen_undirected:
                continue
            seen_undirected.add(key)
            edge_mult[o0, o1] += 1.0
            if o0 != o1:
                edge_mult[o1, o0] += 1.0
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
        disconnected = self.dist_emb.num_embeddings - 1
        dist = dist.clamp(0, disconnected)
        dist_bin = torch.full((j, j), self.DIST_BIN_DISCONNECTED, dtype=torch.long, device=device)
        reachable = dist < disconnected
        dist_bin = torch.where(reachable, dist.clamp(max=4), dist_bin)
        dist_bin.fill_diagonal_(0)
        self.bond_type = bond
        self.graph_dist = dist
        self.bond_exists = (edge_mult > 0).to(dtype=torch.float32)
        self.edge_mult = edge_mult
        self.dist_bin = dist_bin

    def raw_template(self, o_i: torch.Tensor, o_j: torch.Tensor) -> dict[str, torch.Tensor]:
        """Discrete orbit-pair template (no learned params, no copy id)."""
        return {
            "bond_type": self.bond_type[o_i, o_j],
            "dist_bin": self.dist_bin[o_i, o_j],
            "bond_exists": self.bond_exists[o_i, o_j],
            "edge_mult": self.edge_mult[o_i, o_j],
        }

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


class OrbitSlotCopyContext(nn.Module):
    """Copy × orbit slot table U[k,o] = pool_{i: copy=k, orbit=o} φ(h_i, z_o).

    Shared φ across copies; no copy-ID embedding. Equivariant to copy-column perm.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.phi_slot = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def atom_slot_feat(
        self, h: torch.Tensor, orbit_of: torch.Tensor, z_orbit: torch.Tensor
    ) -> torch.Tensor:
        o = orbit_of.long()
        return self.phi_slot(torch.cat([h, z_orbit[o]], dim=-1))

    def slot_table(
        self,
        h: torch.Tensor,
        *,
        orbit_of: torch.Tensor,
        copy_of: torch.Tensor,
        z_orbit: torch.Tensor,
        K: int,
        J: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return U[K,J,H], atom_feat[N,H], counts[K,J,1]."""
        n, hid = h.shape
        device, dtype = h.device, h.dtype
        feat = self.atom_slot_feat(h, orbit_of, z_orbit)
        U = torch.zeros(K, J, hid, device=device, dtype=dtype)
        cnt = torch.zeros(K, J, 1, device=device, dtype=dtype)
        if n == 0:
            return U, feat, cnt
        k = copy_of.long()
        o = orbit_of.long()
        flat = k * J + o
        U.view(K * J, hid).index_add_(0, flat, feat)
        ones = torch.ones(n, 1, device=device, dtype=dtype)
        cnt.view(K * J, 1).index_add_(0, flat, ones)
        U = U / cnt.clamp_min(1.0)
        return U, feat, cnt

    def exclude_atom_slots(
        self,
        U: torch.Tensor,
        cnt: torch.Tensor,
        feat: torch.Tensor,
        orbit_of: torch.Tensor,
        copy_of: torch.Tensor,
    ) -> torch.Tensor:
        """U_excl[i, o', :] = copy-of-i slot table with atom i removed from its own orbit."""
        n = feat.shape[0]
        k = copy_of.long()
        o = orbit_of.long()
        U_i = U[k]  # [N, J, H]
        cnt_i = cnt[k]  # [N, J, 1]
        ar = torch.arange(n, device=feat.device)
        own = U_i[ar, o]
        own_c = cnt_i[ar, o]
        own_ex = (own * own_c - feat) / (own_c - 1.0).clamp_min(1.0)
        own_ex = torch.where(own_c <= 1.0, torch.zeros_like(own_ex), own_ex)
        U_ex = U_i.clone()
        U_ex[ar, o] = own_ex
        return U_ex

    def slot_diagnostics(self, U: torch.Tensor) -> dict[str, float]:
        """Norms and orbit-wise variance (collapse check)."""
        if U.numel() == 0:
            return {
                "slot_embedding_norm_mean": 0.0,
                "slot_embedding_norm_std": 0.0,
                "slot_orbit_pairwise_var": 0.0,
            }
        norms = U.norm(dim=-1)
        # variance of slot vectors across orbits, averaged over copies
        # U: [K, J, H]
        var_o = U.var(dim=1, unbiased=False).mean()
        return {
            "slot_embedding_norm_mean": float(norms.mean().detach()),
            "slot_embedding_norm_std": float(norms.std(unbiased=False).detach()) if norms.numel() > 1 else 0.0,
            "slot_orbit_pairwise_var": float(var_o.detach()),
        }


class CandidateCopyGeometry(nn.Module):
    """PBC min-image Gaussian RBF from a query atom to copy×orbit slots.

    Distance / RBF convention is imported from ``pair_potential`` (same wrap
    and Gaussian width as BondPairPotential). Scalar invariant features only;
    no Cartesian direction vectors.
    """

    def __init__(self, rbf_dim: int = 32, cutoff: float = 6.0):
        super().__init__()
        self.rbf_dim = int(rbf_dim)
        self.cutoff = float(cutoff)
        self.register_buffer("centres", torch.linspace(0.0, self.cutoff, self.rbf_dim))

    def all_pairs_rbf(self, frac: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """RBF[i, r] = Gaussian-RBF(d_PBC(i, r)). Shape [N, N, R]."""
        if frac.ndim != 2 or frac.shape[-1] != 3:
            raise ValueError(f"frac must be [N,3], got {tuple(frac.shape)}")
        n = frac.shape[0]
        if n == 0:
            return frac.new_zeros(0, 0, self.rbf_dim)
        cell = cell.to(device=frac.device, dtype=frac.dtype)
        dist = pbc_minimum_image_distance(frac[:, None, :], frac[None, :, :], cell)
        return gaussian_radial_basis(dist, self.centres.to(device=frac.device, dtype=frac.dtype), self.cutoff)

    def pool_to_slots(
        self,
        rbf: torch.Tensor,
        *,
        copy_of: torch.Tensor,
        orbit_of: torch.Tensor,
        K: int,
        J: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool target atoms into copy×orbit slots.

        Returns:
            G: [N, K, J, R]  query i → slot (k, o')
            cnt: [K, J]      slot multiplicity (before any exclusion)
        """
        n, n2, rdim = rbf.shape
        device, dtype = rbf.device, rbf.dtype
        G = torch.zeros(n, K * J, rdim, device=device, dtype=dtype)
        cnt = torch.zeros(K * J, device=device, dtype=dtype)
        if n == 0:
            return G.view(n, K, J, rdim), cnt.view(K, J)
        flat = copy_of.long() * int(J) + orbit_of.long()
        G.index_add_(1, flat, rbf)
        cnt.index_add_(0, flat, torch.ones(n, device=device, dtype=dtype))
        G = G / cnt.clamp_min(1.0).view(1, -1, 1)
        return G.view(n, K, J, rdim), cnt.view(K, J)

    def exclude_atom(
        self,
        G: torch.Tensor,
        rbf: torch.Tensor,
        cnt: torch.Tensor,
        *,
        query: torch.Tensor,
        dest_copy: torch.Tensor,
        exclude: torch.Tensor,
        copy_of: torch.Tensor,
        orbit_of: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slot geometry G[query, dest_copy] with ``exclude`` removed if it lives there.

        Empty slots after remove-one become a zero RBF and occupancy 0 (no NaN,
        no leftover self-distance-0 peak).
        """
        g0 = G[query, dest_copy]  # [P, J, R]
        occ0 = cnt[dest_copy]  # [P, J]
        if query.numel() == 0:
            return g0, occ0
        o_ex = orbit_of[exclude].long()
        in_dest = copy_of[exclude].long() == dest_copy.long()
        r_ex = rbf[query, exclude]  # [P, R]
        p = int(query.shape[0])
        ar = torch.arange(p, device=query.device)
        g_slot = g0[ar, o_ex]
        c_slot = occ0[ar, o_ex]
        new_slot = (c_slot.unsqueeze(-1) * g_slot - r_ex) / (c_slot - 1.0).clamp_min(1.0).unsqueeze(-1)
        empty = in_dest & (c_slot <= 1.0)
        new_slot = torch.where(
            (~in_dest).unsqueeze(-1),
            g_slot,
            torch.where(empty.unsqueeze(-1), torch.zeros_like(new_slot), new_slot),
        )
        g_out = g0.clone()
        g_out[ar, o_ex] = new_slot
        occ_out = occ0.clone()
        occ_out[ar, o_ex] = torch.where(in_dest, (c_slot - 1.0).clamp_min(0.0), c_slot)
        return g_out, occ_out

    def self_excluded_tables(
        self,
        G: torch.Tensor,
        rbf: torch.Tensor,
        cnt: torch.Tensor,
        *,
        copy_of: torch.Tensor,
        orbit_of: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-atom, all-copy geometry with each atom removed from its own slot.

        Dest copies that do not contain the query are unchanged. Used for
        representation diagnostics (variance across copies / candidates).
        """
        n, k, j, _r = G.shape
        g_all = G.clone()
        occ_all = cnt.unsqueeze(0).expand(n, -1, -1).clone()
        if n == 0:
            return g_all, occ_all
        ar = torch.arange(n, device=G.device)
        k_i = copy_of.long()
        o_i = orbit_of.long()
        c_slot = cnt[k_i, o_i]
        g_slot = g_all[ar, k_i, o_i]
        r_self = rbf[ar, ar]
        new_slot = (c_slot.unsqueeze(-1) * g_slot - r_self) / (c_slot - 1.0).clamp_min(1.0).unsqueeze(-1)
        new_slot = torch.where(c_slot.unsqueeze(-1) <= 1.0, torch.zeros_like(new_slot), new_slot)
        g_all[ar, k_i, o_i] = new_slot
        occ_all = occ_all.clone()
        occ_all[ar, k_i, o_i] = (c_slot - 1.0).clamp_min(0.0)
        return g_all, occ_all


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
