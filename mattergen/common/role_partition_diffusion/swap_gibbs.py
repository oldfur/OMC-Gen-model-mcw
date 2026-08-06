"""Capacity-preserving swap/Gibbs diffusion for canonical roles.

The state is always a legal role labelling.  No MASK state or independent role
sampling is used in this module.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def legal_actions(role: torch.Tensor, atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Return unordered same-element, unequal-role swap pairs [E,2]."""
    pairs = []
    for element in atomic_numbers.unique(sorted=True):
        index = (atomic_numbers == element).nonzero().flatten()
        for left in range(len(index)):
            for right in range(left + 1, len(index)):
                i, j = index[left], index[right]
                if role[i] != role[j]:
                    pairs.append(torch.stack([i, j]))
    return torch.stack(pairs) if pairs else torch.empty((0, 2), dtype=torch.long, device=role.device)


def apply_action(role: torch.Tensor, action: torch.Tensor | None) -> torch.Tensor:
    out = role.clone()
    if action is not None:
        i, j = action.tolist()
        out[i], out[j] = role[j], role[i]
    return out


def assert_legal(role: torch.Tensor, z: torch.Tensor, role_z: torch.Tensor, k: int) -> None:
    if role.ndim != 1 or not bool(((role >= 0) & (role < len(role_z))).all()):
        raise ValueError("role state has invalid labels")
    if not bool(torch.equal(z, role_z[role])):
        raise ValueError("role state violates the element hard mask")
    counts = torch.bincount(role, minlength=len(role_z))
    if not bool((counts == k).all()):
        raise ValueError(f"role state violates capacity K={k}: {counts.tolist()}")


class SwapGibbsRoleDiffusion:
    def __init__(self, steps: int = 64, terminal_randomization_steps: int = 128, s_max: int | None = None):
        self.steps = steps
        self.terminal_randomization_steps = terminal_randomization_steps
        self.s_max = terminal_randomization_steps if s_max is None else s_max

    def corrupt(self, clean: torch.Tensor, z: torch.Tensor, t: int, generator=None) -> torch.Tensor:
        """Forward chain with floor(s_max * t/T) uniformly chosen legal swaps."""
        state = clean.clone()
        n = int(math.floor(self.s_max * (float(t) / self.steps)))
        for _ in range(n):
            actions = legal_actions(state, z)
            if not len(actions):
                break
            state = apply_action(state, actions[torch.randint(len(actions), (), device=state.device, generator=generator)])
        return state

    def terminal_prior(self, clean: torch.Tensor, z: torch.Tensor, generator=None) -> torch.Tensor:
        """Uniformly shuffle each element's legal role multiset."""
        state = clean.clone()
        for element in z.unique(sorted=True):
            index = (z == element).nonzero().flatten()
            state[index] = clean[index][torch.randperm(len(index), device=state.device, generator=generator)]
        return state

    @staticmethod
    def agreement(state: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        return (state == clean).sum()

    def target_distribution(self, state: torch.Tensor, clean: torch.Tensor, z: torch.Tensor, temperature: float):
        actions = legal_actions(state, z)
        before = self.agreement(state, clean)
        improve = [torch.zeros((), device=state.device)]
        for pair in actions:
            improve.append(self.agreement(apply_action(state, pair), clean) - before)
        improve = torch.stack(improve).to(torch.float32)
        return actions, improve, torch.softmax(improve / temperature, dim=0)


class SwapScoreHead(nn.Module):
    """Compatibility-derived logits for no-op plus legal same-element swaps."""
    def __init__(self, hidden: int = 256, steps: int = 64, rbf_dim: int = 32):
        super().__init__()
        self.time = nn.Embedding(steps + 1, hidden)
        self.register_buffer("centres", torch.linspace(0, 6, rbf_dim))
        self.compat = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.action = nn.Sequential(nn.Linear(7 * hidden + rbf_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.noop = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def compatibility(self, hx: torch.Tensor, hm: torch.Tensor) -> torch.Tensor:
        a = hx[:, None, :].expand(-1, len(hm), -1); b = hm[None, :, :].expand(len(hx), -1, -1)
        return self.compat(torch.cat([a, b, (a - b).abs(), a * b], -1)).squeeze(-1)

    def forward(self, hx: torch.Tensor, hm: torch.Tensor, role: torch.Tensor, pairs: torch.Tensor, frac: torch.Tensor, cell: torch.Tensor, t: int):
        comp = self.compatibility(hx, hm)
        te = self.time(torch.as_tensor(t, device=hx.device))
        noop = self.noop(torch.cat([hx.mean(0), hm.mean(0), te], -1)).view(1)
        if not len(pairs):
            return noop, comp
        i, j = pairs[:, 0], pairs[:, 1]; ri, rj = role[i], role[j]
        delta = comp[i, rj] + comp[j, ri] - comp[i, ri] - comp[j, rj]
        d = frac[j] - frac[i]; d = d - torch.round(d); dist = torch.linalg.norm(d @ cell, dim=-1)
        rbf = torch.exp(-((dist[:, None] - self.centres) / 0.1875) ** 2)
        feature = torch.cat([hx[i], hx[j], hm[ri], hm[rj], (hx[i] - hx[j]).abs(), hx[i] * hx[j], te[None].expand(len(i), -1), rbf, delta[:, None]], -1)
        return torch.cat([noop, self.action(feature).squeeze(-1)]), comp


class SwapStopHead(nn.Module):
    """Independent stop classifier; it is intentionally not in the swap softmax."""
    def __init__(self, hidden: int = 256, steps: int = 64):
        super().__init__()
        self.time = nn.Embedding(steps + 1, hidden)
        self.net = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # Conservative before supervision: continuing is safer than an unseen
        # false stop because inference uses a 0.99 threshold.
        nn.init.constant_(self.net[-1].bias, -5.0)

    def forward(self, hx: torch.Tensor, hm: torch.Tensor, t: int) -> torch.Tensor:
        te = self.time(torch.as_tensor(t, device=hx.device))
        return self.net(torch.cat([hx.mean(0), hm.mean(0), te], -1)).squeeze(-1)
