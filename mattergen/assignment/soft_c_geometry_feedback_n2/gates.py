"""Noise gate and pair-wise confidence for soft-C geometry feedback (N2)."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoiseGateConfig:
    enabled: bool = True
    full_on_t_fraction: float = 0.30
    full_off_t_fraction: float = 0.60

    def __post_init__(self):
        if self.full_on_t_fraction < 0 or self.full_off_t_fraction > 1:
            raise ValueError("noise gate fractions must be in [0,1]")
        if self.full_on_t_fraction > self.full_off_t_fraction:
            raise ValueError("full_on_t_fraction must be <= full_off_t_fraction")


def noise_gate(
    t_fraction: torch.Tensor | float,
    *,
    full_on: float = 0.30,
    full_off: float = 0.60,
) -> torch.Tensor:
    """Piecewise noise gate g_noise(t/T).

    g = 1 for t/T <= full_on
    g = (full_off - t/T) / (full_off - full_on) for full_on < t/T < full_off
    g = 0 for t/T >= full_off
    """
    t = torch.as_tensor(t_fraction, dtype=torch.float32)
    span = max(float(full_off) - float(full_on), 1e-8)
    g = torch.where(
        t <= full_on,
        torch.ones_like(t),
        torch.where(
            t >= full_off,
            torch.zeros_like(t),
            (full_off - t) / span,
        ),
    )
    return g.clamp(0.0, 1.0)


def noise_gate_value(
    t_fraction: float | torch.Tensor,
    cfg: NoiseGateConfig,
) -> torch.Tensor:
    """If noise_gate disabled, return 1 (always on); else piecewise g_noise."""
    t = torch.as_tensor(t_fraction, dtype=torch.float32)
    if not cfg.enabled:
        return torch.ones_like(t, dtype=torch.float32)
    return noise_gate(
        t_fraction,
        full_on=cfg.full_on_t_fraction,
        full_off=cfg.full_off_t_fraction,
    )


def pairwise_confidence(p: torch.Tensor) -> torch.Tensor:
    """q_ij = 2 |p_ij - 1/2| clamped to [0, 1]."""
    return (2.0 * (p - 0.5).abs()).clamp(0.0, 1.0)


def pair_gate(
    p: torch.Tensor,
    t_fraction: float | torch.Tensor,
    cfg: NoiseGateConfig,
    *,
    pairwise_confidence_enabled: bool = True,
) -> torch.Tensor:
    """a_ij = g_noise(t) * q_ij."""
    g = noise_gate_value(t_fraction, cfg).to(device=p.device, dtype=p.dtype)
    if pairwise_confidence_enabled:
        q = pairwise_confidence(p)
    else:
        q = torch.ones_like(p)
    while g.ndim < p.ndim:
        g = g.reshape(*g.shape, *([1] * (p.ndim - g.ndim)))
    return (g * q).clamp(0.0, 1.0)
