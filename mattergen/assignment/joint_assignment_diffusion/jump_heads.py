"""Direct legal jump logits for R and G moves (fixed total exit rate)."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .conditioning import CandidateCopyGeometry
from .legal_moves import LegalMove
from .state import JointAssignmentState

_G_COPY_CONTEXT_MODES = (
    "mean",
    "orbit_slot",
    "orbit_slot_geometry",
    "template_counterfactual",
)

_G_SLOT_MODES = ("orbit_slot", "orbit_slot_geometry", "template_counterfactual")
_G_GEOM_MODES = ("orbit_slot_geometry", "template_counterfactual")


class TemplateRhoEncoder(nn.Module):
    """G-specific embedding of discrete molecular-template ρ_{oo'}.

    Inputs are orbit-aggregated (automorphism-invariant) and contain no copy
    id or canonical role label.  Shared ``OrbitRelationTable.forward`` is not
    used here so L_G cannot update the R/GemNet rho path.
    """

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = int(dim)
        self.bond_type_emb = nn.Embedding(8, dim)
        self.dist_emb = nn.Embedding(6, dim)
        self.out = nn.Sequential(
            nn.Linear(2 * dim + 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        bond_type: torch.Tensor,
        dist_bin: torch.Tensor,
        bond_exists: torch.Tensor,
        edge_mult: torch.Tensor,
    ) -> torch.Tensor:
        feat = torch.cat(
            [
                self.bond_type_emb(bond_type.long().clamp(0, 7)),
                self.dist_emb(dist_bin.long().clamp(0, 5)),
                bond_exists.unsqueeze(-1).to(dtype=self.out[0].weight.dtype),
                torch.log1p(edge_mult.clamp_min(0.0)).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.out(feat)

    def encode_table(self, rho_table) -> torch.Tensor:
        """ρ[o, o', :] for the full orbit-pair table."""
        return self.forward(
            rho_table.bond_type,
            rho_table.dist_bin,
            rho_table.bond_exists,
            rho_table.edge_mult,
        )


class RJumpHead(nn.Module):
    """Symmetric pair head for legal R-moves."""

    def __init__(self, hidden: int, rho_dim: int = 32):
        super().__init__()
        # h_i+h_j, |h_i-h_j|, zo_i, zo_j, c_i, c_j, rho → logit
        in_dim = 2 * hidden + 2 * hidden + 2 * hidden + rho_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # zero-init logits → uniform π at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def pair_logit(
        self,
        h: torch.Tensor,
        i: int,
        j: int,
        *,
        z_orbit: torch.Tensor,
        c_i: torch.Tensor,
        orbit_of: torch.Tensor,
        rho_vec: torch.Tensor,
    ) -> torch.Tensor:
        hi, hj = h[i], h[j]
        feat = torch.cat(
            [
                hi + hj,
                (hi - hj).abs(),
                z_orbit[int(orbit_of[i])],
                z_orbit[int(orbit_of[j])],
                c_i[i],
                c_i[j],
                rho_vec,
            ],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)

    def batch_logits(
        self,
        h: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        z_orbit: torch.Tensor,
        c_i: torch.Tensor,
        orbit_of: torch.Tensor,
        rho_vecs: torch.Tensor,
    ) -> torch.Tensor:
        hi, hj = h[ii], h[jj]
        oi, oj = orbit_of[ii].long(), orbit_of[jj].long()
        feat = torch.cat(
            [
                hi + hj,
                (hi - hj).abs(),
                z_orbit[oi],
                z_orbit[oj],
                c_i[ii],
                c_i[jj],
                rho_vecs,
            ],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)


class GJumpHead(nn.Module):
    """Symmetric pair head for legal G-moves.

    ``mean``: exclusion mean copy context (J1.3-A / B0).
    ``orbit_slot``: structured U[k,o] slots with remove-one exclusion (B1).
    ``orbit_slot_geometry``: B1 slots + candidate→copy PBC radial relations (B2).
    ``template_counterfactual``: B2 + ρ template + explicit ΔS (B3).
    """

    def __init__(self, hidden: int, *, copy_context_mode: str = "mean"):
        super().__init__()
        if copy_context_mode not in _G_COPY_CONTEXT_MODES:
            raise ValueError(f"unknown g copy_context_mode={copy_context_mode}")
        self.copy_context_mode = copy_context_mode
        in_dim = 2 * hidden + 2 * hidden + 2 * hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Orbit-slot pair MLP: local pair + slot pair + z_o + z_o' + t
        self.t_enc = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        slot_in = 2 * hidden + 2 * hidden + 2 * hidden + 32
        self.slot_pair = nn.Sequential(
            nn.Linear(slot_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.slot_out = nn.Sequential(
            nn.Linear(2 * hidden + 2 * hidden + hidden + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.slot_out[-1].weight)
        nn.init.zeros_(self.slot_out[-1].bias)
        # B2: shared phi_rel(h_i, U_{k,o'}, z_o, z_{o'}, g, occ, t) + symmetric G MLP.
        self.cand_geom = CandidateCopyGeometry(rbf_dim=32, cutoff=6.0)
        rel_in = 4 * hidden + int(self.cand_geom.rbf_dim) + 1 + 32
        self.phi_rel = nn.Sequential(
            nn.Linear(rel_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # local pair + current/cross symmetric groups + t
        geom_in = 8 * hidden + 32
        self.geom_out = nn.Sequential(
            nn.Linear(geom_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.geom_out[-1].weight)
        nn.init.zeros_(self.geom_out[-1].bias)
        # B3: G-specific template ρ, compatibility, counterfactual aggregator.
        self.template_rho = TemplateRhoEncoder(dim=32)
        rel_b3 = 4 * hidden + int(self.cand_geom.rbf_dim) + 1 + 32 + 32
        self.phi_rel_b3 = nn.Sequential(
            nn.Linear(rel_b3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.psi_compat = nn.Sequential(
            nn.Linear(hidden, max(hidden // 4, 32)),
            nn.SiLU(),
            nn.Linear(max(hidden // 4, 32), 1),
        )
        self.phi_w = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
        self.phi_cf = nn.Sequential(nn.Linear(1 + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        cf_in = 1 + hidden + 4 * hidden + 32
        self.cf_out = nn.Sequential(
            nn.Linear(cf_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.cf_out[-1].weight)
        nn.init.zeros_(self.cf_out[-1].bias)

    def pair_logit(
        self,
        h: torch.Tensor,
        i: int,
        j: int,
        *,
        z_orbit: torch.Tensor,
        orbit_of: torch.Tensor,
        copy_of: torch.Tensor,
        v_copies: torch.Tensor,
    ) -> torch.Tensor:
        def excl(idx: int) -> torch.Tensor:
            k = int(copy_of[idx])
            mask = (copy_of == k) & (torch.arange(h.shape[0], device=h.device) != idx)
            if mask.any():
                return h[mask].mean(0)
            return v_copies[k]

        hi, hj = h[i], h[j]
        ei, ej = excl(i), excl(j)
        oi, oj = int(orbit_of[i]), int(orbit_of[j])
        feat = torch.cat([hi + hj, (hi - hj).abs(), ei + ej, (ei - ej).abs(), z_orbit[oi], z_orbit[oj]], dim=-1)
        return self.net(feat).squeeze(-1)

    def batch_logits(
        self,
        h: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        z_orbit: torch.Tensor,
        orbit_of: torch.Tensor,
        excl: torch.Tensor,
    ) -> torch.Tensor:
        hi, hj = h[ii], h[jj]
        ei, ej = excl[ii], excl[jj]
        oi, oj = orbit_of[ii].long(), orbit_of[jj].long()
        feat = torch.cat(
            [
                hi + hj,
                (hi - hj).abs(),
                ei + ej,
                (ei - ej).abs(),
                z_orbit[oi],
                z_orbit[oj],
            ],
            dim=-1,
        )
        return self.net(feat).squeeze(-1)

    def batch_logits_orbit_slot(
        self,
        h: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        z_orbit: torch.Tensor,
        orbit_of: torch.Tensor,
        u_excl: torch.Tensor,
        t_scalar: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Orbit-slot G logits. u_excl[n, J, H] is per-atom exclusion slot table.

        Returns (logits[P], q_slot[P, H]) for diagnostics.
        """
        hi, hj = h[ii], h[jj]
        oi = orbit_of[ii].long()
        uk, ul = u_excl[ii], u_excl[jj]  # [P, J, H]
        p, j_orb, hid = uk.shape
        tenc = self.t_enc(
            torch.tensor([[float(t_scalar)]], device=h.device, dtype=h.dtype)
        ).expand(p, j_orb, -1)
        zo_c = z_orbit[oi].unsqueeze(1).expand(p, j_orb, -1)
        zo_s = z_orbit.unsqueeze(0).expand(p, j_orb, -1)
        hi_e = (hi + hj).unsqueeze(1).expand(p, j_orb, -1)
        hd_e = (hi - hj).abs().unsqueeze(1).expand(p, j_orb, -1)
        slot_in = torch.cat(
            [hi_e, hd_e, uk + ul, (uk - ul).abs(), zo_c, zo_s, tenc],
            dim=-1,
        )
        q_o = self.slot_pair(slot_in)  # [P, J, H]
        q_pool = q_o.mean(dim=1)
        t_pair = self.t_enc(
            torch.tensor([[float(t_scalar)]], device=h.device, dtype=h.dtype)
        ).expand(p, -1)
        local = torch.cat([hi + hj, (hi - hj).abs(), z_orbit[oi], z_orbit[orbit_of[jj].long()]], dim=-1)
        logits = self.slot_out(torch.cat([local, q_pool, t_pair], dim=-1)).squeeze(-1)
        return logits, q_pool

    def _encode_t(self, t_scalar: float, n: int, device, dtype) -> torch.Tensor:
        t = torch.tensor([[float(t_scalar)]], device=device, dtype=dtype)
        enc = self.t_enc(t)
        return enc.expand(max(n, 1), -1)[:n] if n > 0 else enc[:0]

    def _phi_rel_pool(
        self,
        h_i: torch.Tensor,
        U: torch.Tensor,
        z_o: torch.Tensor,
        z_orbit: torch.Tensor,
        g: torch.Tensor,
        occ: torch.Tensor,
        tenc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q_{i→k,o'} = φ_rel(...); q_{i→k} = masked mean over orbit slots."""
        p, j_orb, hid = U.shape
        hi_e = h_i.unsqueeze(1).expand(p, j_orb, -1)
        zo_e = z_o.unsqueeze(1).expand(p, j_orb, -1)
        zop_e = z_orbit.unsqueeze(0).expand(p, j_orb, -1)
        t_e = tenc.unsqueeze(1).expand(p, j_orb, -1)
        occ_e = occ.unsqueeze(-1)
        feat = torch.cat([hi_e, U, zo_e, zop_e, g, occ_e, t_e], dim=-1)
        q = self.phi_rel(feat)
        mask = (occ > 0).to(dtype=q.dtype).unsqueeze(-1)
        q = q * mask
        q_pool = q.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return q, q_pool

    def batch_logits_orbit_slot_geometry(
        self,
        h: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        z_orbit: torch.Tensor,
        orbit_of: torch.Tensor,
        copy_of: torch.Tensor,
        U: torch.Tensor,
        u_excl: torch.Tensor,
        slot_cnt: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        t_scalar: float,
        K: int,
        J: int,
    ) -> tuple[torch.Tensor, dict]:
        """Exclusion-aware candidate→copy geometry G logits (J1.3-B2).

        For swap (i∈k, j∈l) the shared MLP sees four atom→copy relations:
            i→k\\{i}, i→l\\{j}, j→l\\{j}, j→k\\{i}
        combined as symmetric current / cross groups. No scalar ΔS / template.
        """
        empty = {
            "candidate_copy_geom_norm_mean": 0.0,
            "candidate_copy_geom_norm_std": 0.0,
            "candidate_copy_relation_norm_mean": 0.0,
            "candidate_copy_relation_norm_std": 0.0,
            "candidate_copy_relation_variance_across_copies": 0.0,
            "candidate_copy_relation_variance_across_candidates": 0.0,
            "current_vs_cross_relation_distance": 0.0,
            "g_logit_std_across_legal_moves": 0.0,
        }
        p = int(ii.numel())
        if p == 0:
            return h.new_zeros(0), empty

        rbf = self.cand_geom.all_pairs_rbf(frac, cell)
        G, cnt_geom = self.cand_geom.pool_to_slots(
            rbf, copy_of=copy_of, orbit_of=orbit_of, K=K, J=J
        )
        ki = copy_of[ii].long()
        li = copy_of[jj].long()
        # Four exclusion-aware slot geometries.
        g_ik, occ_ik = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=ii, dest_copy=ki, exclude=ii, copy_of=copy_of, orbit_of=orbit_of
        )
        g_il, occ_il = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=ii, dest_copy=li, exclude=jj, copy_of=copy_of, orbit_of=orbit_of
        )
        g_jl, occ_jl = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=jj, dest_copy=li, exclude=jj, copy_of=copy_of, orbit_of=orbit_of
        )
        g_jk, occ_jk = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=jj, dest_copy=ki, exclude=ii, copy_of=copy_of, orbit_of=orbit_of
        )
        # Matching U tables: own copy remove-self; dest copy remove-partner.
        u_ik = u_excl[ii]
        u_il = u_excl[jj]
        u_jl = u_excl[jj]
        u_jk = u_excl[ii]

        hi, hj = h[ii], h[jj]
        oi = orbit_of[ii].long()
        oj = orbit_of[jj].long()
        tenc = self._encode_t(t_scalar, p, h.device, h.dtype)
        zo_i = z_orbit[oi]
        zo_j = z_orbit[oj]

        _, q_ik = self._phi_rel_pool(hi, u_ik, zo_i, z_orbit, g_ik, occ_ik, tenc)
        _, q_il = self._phi_rel_pool(hi, u_il, zo_i, z_orbit, g_il, occ_il, tenc)
        _, q_jl = self._phi_rel_pool(hj, u_jl, zo_j, z_orbit, g_jl, occ_jl, tenc)
        _, q_jk = self._phi_rel_pool(hj, u_jk, zo_j, z_orbit, g_jk, occ_jk, tenc)

        q_current_sum = q_ik + q_jl
        q_current_dif = (q_ik - q_jl).abs()
        q_cross_sum = q_il + q_jk
        q_cross_dif = (q_il - q_jk).abs()
        local = torch.cat([hi + hj, (hi - hj).abs(), zo_i, zo_j], dim=-1)
        feat = torch.cat(
            [local, q_current_sum, q_current_dif, q_cross_sum, q_cross_dif, tenc],
            dim=-1,
        )
        logits = self.geom_out(feat).squeeze(-1)

        # --- representation diagnostics (no extra learned scores) ---
        g_stack = torch.stack([g_ik, g_il, g_jl, g_jk], dim=0)
        g_norms = g_stack.norm(dim=-1)
        q_stack = torch.stack([q_ik, q_il, q_jl, q_jk], dim=0)
        q_norms = q_stack.norm(dim=-1)
        cur_vs_crs = (q_current_sum - q_cross_sum).norm(dim=-1)
        var_copies = 0.0
        var_cands = 0.0
        n = int(h.shape[0])
        if n > 0 and K > 1:
            g_all, occ_all = self.cand_geom.self_excluded_tables(
                G, rbf, cnt_geom, copy_of=copy_of, orbit_of=orbit_of
            )
            u_all = U.unsqueeze(0).expand(n, -1, -1, -1).clone()
            u_all[torch.arange(n, device=h.device), copy_of.long()] = u_excl
            # Flatten (atom, copy) as independent queries for phi_rel.
            nk = n * int(K)
            h_f = h.unsqueeze(1).expand(n, K, -1).reshape(nk, -1)
            u_f = u_all.reshape(nk, J, -1)
            g_f = g_all.reshape(nk, J, -1)
            occ_f = occ_all.reshape(nk, J)
            zo_f = z_orbit[orbit_of.long()].unsqueeze(1).expand(n, K, -1).reshape(nk, -1)
            t_all = self._encode_t(t_scalar, nk, h.device, h.dtype)
            _, q_all = self._phi_rel_pool(h_f, u_f, zo_f, z_orbit, g_f, occ_f, t_all)
            q_all = q_all.view(n, K, -1)
            var_copies = float(q_all.var(dim=1, unbiased=False).mean().detach())
            var_cands = float(q_all.var(dim=0, unbiased=False).mean().detach()) if n > 1 else 0.0

        diag = {
            "candidate_copy_geom_norm_mean": float(g_norms.mean().detach()),
            "candidate_copy_geom_norm_std": float(g_norms.std(unbiased=False).detach())
            if g_norms.numel() > 1
            else 0.0,
            "candidate_copy_relation_norm_mean": float(q_norms.mean().detach()),
            "candidate_copy_relation_norm_std": float(q_norms.std(unbiased=False).detach())
            if q_norms.numel() > 1
            else 0.0,
            "candidate_copy_relation_variance_across_copies": var_copies,
            "candidate_copy_relation_variance_across_candidates": var_cands,
            "current_vs_cross_relation_distance": float(cur_vs_crs.mean().detach()),
            "g_logit_std_across_legal_moves": float(logits.detach().std(unbiased=False))
            if p > 1
            else 0.0,
        }
        return logits, diag

    def _phi_rel_pool_b3(
        self,
        h_i: torch.Tensor,
        U: torch.Tensor,
        z_o: torch.Tensor,
        z_orbit: torch.Tensor,
        g: torch.Tensor,
        occ: torch.Tensor,
        rho_po: torch.Tensor,
        tenc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, c, w, S for one candidate→copy table. c/w are per-orbit."""
        p, j_orb, hid = U.shape
        hi_e = h_i.unsqueeze(1).expand(p, j_orb, -1)
        zo_e = z_o.unsqueeze(1).expand(p, j_orb, -1)
        zop_e = z_orbit.unsqueeze(0).expand(p, j_orb, -1)
        t_e = tenc.unsqueeze(1).expand(p, j_orb, -1)
        occ_e = occ.unsqueeze(-1)
        feat = torch.cat([hi_e, U, zo_e, zop_e, g, occ_e, rho_po, t_e], dim=-1)
        q = self.phi_rel_b3(feat)
        mask = (occ > 0).to(dtype=q.dtype)
        q = q * mask.unsqueeze(-1)
        c = self.psi_compat(q).squeeze(-1) * mask
        w = torch.nn.functional.softplus(self.phi_w(rho_po).squeeze(-1))
        s = (w * c).sum(dim=-1)
        return q, c, w, s

    def batch_logits_template_counterfactual(
        self,
        h: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        z_orbit: torch.Tensor,
        orbit_of: torch.Tensor,
        copy_of: torch.Tensor,
        U: torch.Tensor,
        u_excl: torch.Tensor,
        slot_cnt: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        t_scalar: float,
        K: int,
        J: int,
        rho_table,
    ) -> tuple[torch.Tensor, dict]:
        """Template-conditioned counterfactual G logits (J1.3-B3).

        ΔS = S_swap − S_current is a relational feature only; rates stay β·π.
        """
        empty = {
            "candidate_copy_geom_norm_mean": 0.0,
            "candidate_copy_geom_norm_std": 0.0,
            "candidate_copy_relation_norm_mean": 0.0,
            "candidate_copy_relation_norm_std": 0.0,
            "candidate_copy_relation_variance_across_copies": 0.0,
            "candidate_copy_relation_variance_across_candidates": 0.0,
            "current_vs_cross_relation_distance": 0.0,
            "g_logit_std_across_legal_moves": 0.0,
            "template_relation_norm": 0.0,
            "compatibility_S_mean": 0.0,
            "compatibility_S_std": 0.0,
            "delta_S_mean": 0.0,
            "delta_S_std": 0.0,
            "abs_delta_S_mean": 0.0,
            "counterfactual_feature_norm": 0.0,
            "counterfactual_feature_var": 0.0,
        }
        p = int(ii.numel())
        if p == 0:
            empty["delta_S_vec"] = h.new_zeros(0)
            return h.new_zeros(0), empty

        rbf = self.cand_geom.all_pairs_rbf(frac, cell)
        G, cnt_geom = self.cand_geom.pool_to_slots(
            rbf, copy_of=copy_of, orbit_of=orbit_of, K=K, J=J
        )
        ki = copy_of[ii].long()
        li = copy_of[jj].long()
        g_ik, occ_ik = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=ii, dest_copy=ki, exclude=ii, copy_of=copy_of, orbit_of=orbit_of
        )
        g_il, occ_il = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=ii, dest_copy=li, exclude=jj, copy_of=copy_of, orbit_of=orbit_of
        )
        g_jl, occ_jl = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=jj, dest_copy=li, exclude=jj, copy_of=copy_of, orbit_of=orbit_of
        )
        g_jk, occ_jk = self.cand_geom.exclude_atom(
            G, rbf, cnt_geom, query=jj, dest_copy=ki, exclude=ii, copy_of=copy_of, orbit_of=orbit_of
        )
        u_ik = u_excl[ii]
        u_il = u_excl[jj]
        u_jl = u_excl[jj]
        u_jk = u_excl[ii]

        hi, hj = h[ii], h[jj]
        oi = orbit_of[ii].long()
        oj = orbit_of[jj].long()
        tenc = self._encode_t(t_scalar, p, h.device, h.dtype)
        zo_i = z_orbit[oi]
        zo_j = z_orbit[oj]
        rho_all = self.template_rho.encode_table(rho_table)
        rho_i = rho_all[oi]
        rho_j = rho_all[oj]

        q_ik, c_ik, _w_ik, s_ik = self._phi_rel_pool_b3(
            hi, u_ik, zo_i, z_orbit, g_ik, occ_ik, rho_i, tenc
        )
        q_il, c_il, _w_il, s_il = self._phi_rel_pool_b3(
            hi, u_il, zo_i, z_orbit, g_il, occ_il, rho_i, tenc
        )
        q_jl, c_jl, _w_jl, s_jl = self._phi_rel_pool_b3(
            hj, u_jl, zo_j, z_orbit, g_jl, occ_jl, rho_j, tenc
        )
        q_jk, c_jk, _w_jk, s_jk = self._phi_rel_pool_b3(
            hj, u_jk, zo_j, z_orbit, g_jk, occ_jk, rho_j, tenc
        )

        s_current = s_ik + s_jl
        s_swap = s_il + s_jk
        delta_s = s_swap - s_current
        # Orbit-wise counterfactual residual (symmetric in (i,k)<->(j,l)).
        delta_c = (c_il + c_jk) - (c_ik + c_jl)
        # G-moves share orbit(i)=orbit(j); use that orbit's template vs o'.
        cf_in = torch.cat([delta_c.unsqueeze(-1), rho_i], dim=-1)
        q_cf_o = self.phi_cf(cf_in)
        q_cf = q_cf_o.mean(dim=1)
        local = torch.cat([hi + hj, (hi - hj).abs(), zo_i, zo_j], dim=-1)
        feat = torch.cat([delta_s.unsqueeze(-1), q_cf, local, tenc], dim=-1)
        logits = self.cf_out(feat).squeeze(-1)

        g_stack = torch.stack([g_ik, g_il, g_jl, g_jk], dim=0)
        g_norms = g_stack.norm(dim=-1)
        q_stack = torch.stack([q_ik, q_il, q_jl, q_jk], dim=0)
        q_norms = q_stack.norm(dim=-1)
        s_stack = torch.stack([s_ik, s_il, s_jl, s_jk], dim=0)
        cur_vs_crs = (s_il - s_ik).abs()
        var_copies = 0.0
        var_cands = 0.0
        n = int(h.shape[0])
        if n > 0 and K > 1:
            g_all, occ_all = self.cand_geom.self_excluded_tables(
                G, rbf, cnt_geom, copy_of=copy_of, orbit_of=orbit_of
            )
            u_all = U.unsqueeze(0).expand(n, -1, -1, -1).clone()
            u_all[torch.arange(n, device=h.device), copy_of.long()] = u_excl
            nk = n * int(K)
            h_f = h.unsqueeze(1).expand(n, K, -1).reshape(nk, -1)
            u_f = u_all.reshape(nk, J, -1)
            g_f = g_all.reshape(nk, J, -1)
            occ_f = occ_all.reshape(nk, J)
            zo_f = z_orbit[orbit_of.long()].unsqueeze(1).expand(n, K, -1).reshape(nk, -1)
            rho_f = rho_all[orbit_of.long()].unsqueeze(1).expand(n, K, -1, -1).reshape(nk, J, -1)
            t_all = self._encode_t(t_scalar, nk, h.device, h.dtype)
            _, _, _, s_all = self._phi_rel_pool_b3(h_f, u_f, zo_f, z_orbit, g_f, occ_f, rho_f, t_all)
            s_all = s_all.view(n, K)
            var_copies = float(s_all.var(dim=1, unbiased=False).mean().detach())
            var_cands = float(s_all.var(dim=0, unbiased=False).mean().detach()) if n > 1 else 0.0

        diag = {
            "candidate_copy_geom_norm_mean": float(g_norms.mean().detach()),
            "candidate_copy_geom_norm_std": float(g_norms.std(unbiased=False).detach())
            if g_norms.numel() > 1
            else 0.0,
            "candidate_copy_relation_norm_mean": float(q_norms.mean().detach()),
            "candidate_copy_relation_norm_std": float(q_norms.std(unbiased=False).detach())
            if q_norms.numel() > 1
            else 0.0,
            "candidate_copy_relation_variance_across_copies": var_copies,
            "candidate_copy_relation_variance_across_candidates": var_cands,
            "current_vs_cross_relation_distance": float(cur_vs_crs.mean().detach()),
            "g_logit_std_across_legal_moves": float(logits.detach().std(unbiased=False))
            if p > 1
            else 0.0,
            "template_relation_norm": float(rho_all.norm(dim=-1).mean().detach()),
            "compatibility_S_mean": float(s_stack.mean().detach()),
            "compatibility_S_std": float(s_stack.std(unbiased=False).detach())
            if s_stack.numel() > 1
            else 0.0,
            "delta_S_mean": float(delta_s.mean().detach()),
            "delta_S_std": float(delta_s.std(unbiased=False).detach()) if p > 1 else 0.0,
            "abs_delta_S_mean": float(delta_s.abs().mean().detach()),
            "counterfactual_feature_norm": float(q_cf.norm(dim=-1).mean().detach()),
            "counterfactual_feature_var": float(q_cf.var(dim=0, unbiased=False).mean().detach())
            if p > 1
            else 0.0,
            "delta_S_vec": delta_s.detach(),
        }
        return logits, diag


def _exclusion_contexts(
    h: torch.Tensor,
    copy_of: torch.Tensor,
    v_copies: torch.Tensor,
) -> torch.Tensor:
    n, hid = h.shape
    device = h.device
    dtype = h.dtype
    k_all = copy_of.long()
    K = int(v_copies.shape[0])
    copy_sum = torch.zeros(K, hid, device=device, dtype=dtype)
    copy_cnt = torch.zeros(K, device=device, dtype=dtype)
    if n > 0:
        copy_sum.index_add_(0, k_all, h)
        copy_cnt.index_add_(0, k_all, torch.ones(n, device=device, dtype=dtype))
    cnt_i = copy_cnt[k_all]
    denom = (cnt_i - 1.0).clamp_min(1.0).unsqueeze(-1)
    excl = (copy_sum[k_all] - h) / denom
    singleton = cnt_i <= 1
    if singleton.any():
        excl = torch.where(singleton.unsqueeze(-1), v_copies[k_all], excl)
    return excl


def compute_move_logits(
    *,
    moves: dict[str, list[LegalMove]],
    h: torch.Tensor,
    state: JointAssignmentState,
    z_orbit: torch.Tensor,
    c_i: torch.Tensor,
    v_copies: torch.Tensor,
    rho_table,
    r_head: RJumpHead,
    g_head: GJumpHead,
    slot_ctx=None,
    t_scalar: float = 0.0,
    frac: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    h_g: torch.Tensor | None = None,
    z_orbit_g: torch.Tensor | None = None,
) -> tuple[dict[str, list[tuple[LegalMove, torch.Tensor]]], dict]:
    """Score all legal moves; batched MLP forwards.

    R uses ``h`` / ``z_orbit`` (live trunk).  G uses ``h_g`` / ``z_orbit_g``
    when provided (detached under B3 isolation).
    """
    orbit_of = state.orbit_of()
    copy_of = state.copy_of()
    out: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
    slot_diag: dict = {"g_copy_context_mode": getattr(g_head, "copy_context_mode", "mean")}
    device = h.device
    h_g = h if h_g is None else h_g
    z_g = z_orbit if z_orbit_g is None else z_orbit_g

    r_moves = moves["R"]
    if r_moves:
        ii = torch.tensor([m.i for m in r_moves], device=device, dtype=torch.long)
        jj = torch.tensor([m.j for m in r_moves], device=device, dtype=torch.long)
        oi = orbit_of[ii].long()
        oj = orbit_of[jj].long()
        rel = rho_table(oi, oj)
        logits = r_head.batch_logits(
            h, ii, jj, z_orbit=z_orbit, c_i=c_i, orbit_of=orbit_of, rho_vecs=rel
        )
        for m, logit in zip(r_moves, logits):
            out["R"].append((m, logit))

    g_moves = moves["G"]
    if g_moves:
        ii = torch.tensor([m.i for m in g_moves], device=device, dtype=torch.long)
        jj = torch.tensor([m.j for m in g_moves], device=device, dtype=torch.long)
        mode = getattr(g_head, "copy_context_mode", "mean")
        use_slots = mode in _G_SLOT_MODES and slot_ctx is not None
        if use_slots:
            U, feat, cnt = slot_ctx.slot_table(
                h_g,
                orbit_of=orbit_of,
                copy_of=copy_of,
                z_orbit=z_g,
                K=state.K,
                J=state.J,
            )
            u_excl = slot_ctx.exclude_atom_slots(U, cnt, feat, orbit_of, copy_of)
            slot_diag.update(slot_ctx.slot_diagnostics(U))
            if mode == "template_counterfactual" and frac is not None and cell is not None:
                logits, rel_diag = g_head.batch_logits_template_counterfactual(
                    h_g,
                    ii,
                    jj,
                    z_orbit=z_g,
                    orbit_of=orbit_of,
                    copy_of=copy_of,
                    U=U,
                    u_excl=u_excl,
                    slot_cnt=cnt,
                    frac=frac,
                    cell=cell,
                    t_scalar=t_scalar,
                    K=state.K,
                    J=state.J,
                    rho_table=rho_table,
                )
                slot_diag.update(rel_diag)
            elif mode == "orbit_slot_geometry" and frac is not None and cell is not None:
                logits, rel_diag = g_head.batch_logits_orbit_slot_geometry(
                    h_g,
                    ii,
                    jj,
                    z_orbit=z_g,
                    orbit_of=orbit_of,
                    copy_of=copy_of,
                    U=U,
                    u_excl=u_excl,
                    slot_cnt=cnt,
                    frac=frac,
                    cell=cell,
                    t_scalar=t_scalar,
                    K=state.K,
                    J=state.J,
                )
                slot_diag.update(rel_diag)
            else:
                logits, q_pool = g_head.batch_logits_orbit_slot(
                    h_g, ii, jj, z_orbit=z_g, orbit_of=orbit_of, u_excl=u_excl, t_scalar=t_scalar
                )
                slot_diag["slot_pair_feature_norm"] = float(q_pool.norm(dim=-1).mean().detach())
        else:
            v_g = v_copies.detach() if h_g is not h else v_copies
            excl = _exclusion_contexts(h_g, copy_of, v_g)
            logits = g_head.batch_logits(
                h_g, ii, jj, z_orbit=z_g, orbit_of=orbit_of, excl=excl
            )
        for m, logit in zip(g_moves, logits):
            out["G"].append((m, logit))
        if "g_logit_std_across_legal_moves" not in slot_diag:
            if len(g_moves) > 1:
                stacked = torch.stack([lg.detach() for _, lg in out["G"]])
                slot_diag["g_logit_std_across_legal_moves"] = float(stacked.std(unbiased=False))
            else:
                slot_diag["g_logit_std_across_legal_moves"] = 0.0
    return out, slot_diag


def logits_to_pi(
    scored: dict[str, list[tuple[LegalMove, torch.Tensor]]],
    *,
    clip: float = 8.0,
) -> dict[str, list[tuple[LegalMove, torch.Tensor]]]:
    """π_m = softmax_m ℓ_m over legal moves of each kind (empty → [])."""
    pi: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
    for kind in ("R", "G"):
        pool = scored.get(kind, [])
        if not pool:
            continue
        logits = torch.stack([logit for _, logit in pool]).clamp(-clip, clip)
        probs = torch.softmax(logits, dim=0)
        pi[kind] = [(m, probs[i]) for i, (m, _) in enumerate(pool)]
    return pi


def logits_to_rates(
    scored: dict[str, list[tuple[LegalMove, torch.Tensor]]],
    *,
    beta_r: float,
    beta_g: float,
    clip: float = 8.0,
) -> dict[str, list[tuple[LegalMove, torch.Tensor]]]:
    """Fixed total exit rate: r_m^a = β_a(t) · π_m^a,  Σ_m r_m^a = β_a(t).

    Illegal moves are absent from the pool (rate 0). Empty pool → no rates.
    """
    pi = logits_to_pi(scored, clip=clip)
    rates: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
    for kind, beta in (("R", beta_r), ("G", beta_g)):
        pool = pi[kind]
        if not pool:
            continue
        if beta <= 0:
            rates[kind] = [(m, p * 0.0) for m, p in pool]
        else:
            rates[kind] = [(m, float(beta) * p) for m, p in pool]
    return rates


def jump_pool_diagnostics(
    scored: dict[str, list[tuple[LegalMove, torch.Tensor]]],
    *,
    beta_r: float,
    beta_g: float,
    clip: float = 8.0,
) -> dict[str, Any]:
    """R/G legal counts, logit moments, entropy, uniform NLL, total rate."""
    out: dict[str, Any] = {}
    pi = logits_to_pi(scored, clip=clip)
    for kind, beta in (("R", beta_r), ("G", beta_g)):
        pool = scored.get(kind, [])
        n = len(pool)
        out[f"num_{kind}_moves"] = n
        out[f"beta_{kind}"] = float(beta)
        if n == 0:
            out[f"logit_mean_{kind}"] = 0.0
            out[f"logit_std_{kind}"] = 0.0
            out[f"entropy_{kind}"] = 0.0
            out[f"uniform_nll_{kind}"] = 0.0
            out[f"total_rate_{kind}"] = 0.0
            continue
        logits = torch.stack([logit.detach() for _, logit in pool]).clamp(-clip, clip)
        out[f"logit_mean_{kind}"] = float(logits.mean())
        out[f"logit_std_{kind}"] = float(logits.std(unbiased=False)) if n > 1 else 0.0
        probs = torch.softmax(logits, dim=0)
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
        out[f"entropy_{kind}"] = ent
        out[f"uniform_nll_{kind}"] = float(math.log(n))
        # Σ r = β when n>0
        out[f"total_rate_{kind}"] = float(beta) if beta > 0 else 0.0
        out[f"pi_max_{kind}"] = float(probs.max())
    out["g_logit_std_across_legal_moves"] = float(out.get("logit_std_G", 0.0))
    return out
