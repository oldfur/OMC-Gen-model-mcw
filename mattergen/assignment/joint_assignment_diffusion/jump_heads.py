"""Direct legal jump logits for R and G moves (fixed total exit rate)."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .legal_moves import LegalMove
from .state import JointAssignmentState


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

    ``mean``: exclusion mean copy context (J1.3-A baseline).
    ``orbit_slot``: structured U[k,o] slots with remove-one exclusion (J1.3-B1).
    """

    def __init__(self, hidden: int, *, copy_context_mode: str = "mean"):
        super().__init__()
        if copy_context_mode not in ("mean", "orbit_slot"):
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
) -> tuple[dict[str, list[tuple[LegalMove, torch.Tensor]]], dict]:
    """Score all legal moves; batched MLP forwards.

    Returns (scored, slot_diag). slot_diag is empty in mean mode.
    """
    orbit_of = state.orbit_of()
    copy_of = state.copy_of()
    out: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
    slot_diag: dict = {"g_copy_context_mode": getattr(g_head, "copy_context_mode", "mean")}
    device = h.device

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
        if mode == "orbit_slot" and slot_ctx is not None:
            U, feat, cnt = slot_ctx.slot_table(
                h,
                orbit_of=orbit_of,
                copy_of=copy_of,
                z_orbit=z_orbit,
                K=state.K,
                J=state.J,
            )
            u_excl = slot_ctx.exclude_atom_slots(U, cnt, feat, orbit_of, copy_of)
            logits, q_pool = g_head.batch_logits_orbit_slot(
                h, ii, jj, z_orbit=z_orbit, orbit_of=orbit_of, u_excl=u_excl, t_scalar=t_scalar
            )
            slot_diag.update(slot_ctx.slot_diagnostics(U))
            slot_diag["slot_pair_feature_norm"] = float(q_pool.norm(dim=-1).mean().detach())
        else:
            excl = _exclusion_contexts(h, copy_of, v_copies)
            logits = g_head.batch_logits(
                h, ii, jj, z_orbit=z_orbit, orbit_of=orbit_of, excl=excl
            )
        for m, logit in zip(g_moves, logits):
            out["G"].append((m, logit))
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
    return out
