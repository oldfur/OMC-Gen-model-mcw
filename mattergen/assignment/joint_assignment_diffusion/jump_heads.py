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
    """Symmetric pair head for legal G-moves with exclusion copy contexts."""

    def __init__(self, hidden: int):
        super().__init__()
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
) -> dict[str, list[tuple[LegalMove, torch.Tensor]]]:
    """Score all legal moves; batched MLP forwards."""
    orbit_of = state.orbit_of()
    copy_of = state.copy_of()
    out: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
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
        excl = _exclusion_contexts(h, copy_of, v_copies)
        logits = g_head.batch_logits(
            h, ii, jj, z_orbit=z_orbit, orbit_of=orbit_of, excl=excl
        )
        for m, logit in zip(g_moves, logits):
            out["G"].append((m, logit))
    return out


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
