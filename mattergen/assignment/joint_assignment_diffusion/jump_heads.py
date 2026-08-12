"""Direct legal jump logits for R and G moves (no energy model)."""
from __future__ import annotations

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
        # zero-init logits → exp(0)=1 → uniform CTMC proposal at start
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


class GJumpHead(nn.Module):
    """Symmetric pair head for legal G-moves with exclusion copy contexts."""

    def __init__(self, hidden: int):
        super().__init__()
        in_dim = 2 * hidden + 2 * hidden + 2 * hidden  # sum/diff + two exclusion contexts + two zo
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
        # exclusion: mean of other atoms in same copy, fallback to v_k if singleton
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
    orbit_of = state.orbit_of()
    copy_of = state.copy_of()
    out: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
    for m in moves["R"]:
        rel = rho_table(
            torch.tensor(int(orbit_of[m.i]), device=h.device),
            torch.tensor(int(orbit_of[m.j]), device=h.device),
        )
        logit = r_head.pair_logit(h, m.i, m.j, z_orbit=z_orbit, c_i=c_i, orbit_of=orbit_of, rho_vec=rel)
        out["R"].append((m, logit))
    for m in moves["G"]:
        logit = g_head.pair_logit(
            h, m.i, m.j, z_orbit=z_orbit, orbit_of=orbit_of, copy_of=copy_of, v_copies=v_copies
        )
        out["G"].append((m, logit))
    return out


def logits_to_rates(
    scored: dict[str, list[tuple[LegalMove, torch.Tensor]]],
    *,
    beta_r: float,
    beta_g: float,
    clip: float = 8.0,
) -> dict[str, list[tuple[LegalMove, torch.Tensor]]]:
    """r = β/|M| * exp(clip(ℓ))."""
    rates: dict[str, list[tuple[LegalMove, torch.Tensor]]] = {"R": [], "G": []}
    for kind, beta in (("R", beta_r), ("G", beta_g)):
        pool = scored[kind]
        msize = max(len(pool), 1)
        for m, logit in pool:
            if beta <= 0:
                r = logit * 0.0
            else:
                r = (beta / float(msize)) * torch.exp(logit.clamp(-clip, clip))
            rates[kind].append((m, r))
    return rates
