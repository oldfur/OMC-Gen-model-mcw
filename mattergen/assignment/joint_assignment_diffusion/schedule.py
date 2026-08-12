"""Asynchronous R/G lock schedules for joint CTMC."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AsyncJumpSchedule:
    """β_a(t) = 6 κ_a / (1-t_lock) * u (1-u), u = clip((t-t_lock)/(1-t_lock), 0, 1).

    Exactly zero for t <= t_lock.
    """

    r_lock: float = 0.72
    g_lock: float = 0.52
    kappa_r: float = 4.0
    kappa_g: float = 6.0

    def beta(self, t: float | torch.Tensor, *, kind: str) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=torch.float32)
        if kind == "R":
            t_lock, kappa = self.r_lock, self.kappa_r
        elif kind == "G":
            t_lock, kappa = self.g_lock, self.kappa_g
        else:
            raise ValueError(kind)
        denom = max(1.0 - float(t_lock), 1e-8)
        u = ((t - t_lock) / denom).clamp(0.0, 1.0)
        # for t <= t_lock, u=0 → beta=0
        beta = (6.0 * kappa / denom) * u * (1.0 - u)
        return beta

    def beta_r(self, t: float | torch.Tensor) -> torch.Tensor:
        return self.beta(t, kind="R")

    def beta_g(self, t: float | torch.Tensor) -> torch.Tensor:
        return self.beta(t, kind="G")

    def is_r_locked(self, t: float) -> bool:
        return float(t) <= self.r_lock + 1e-12

    def is_g_locked(self, t: float) -> bool:
        return float(t) <= self.g_lock + 1e-12
