"""Asynchronous R/G mobility windows with fixed integrated jump budget κ."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


def _as_window(w: Sequence[float] | None, default: tuple[float, float]) -> tuple[float, float]:
    if w is None:
        return float(default[0]), float(default[1])
    if len(w) != 2:
        raise ValueError(f"window must be [low, high], got {w}")
    lo, hi = float(w[0]), float(w[1])
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(f"window must satisfy 0 <= low < high <= 1, got {[lo, hi]}")
    return lo, hi


@dataclass
class AsyncJumpSchedule:
    """Compact sin² mobility bumps with fixed total exit rate β_a(t).

    β_a(t) = (κ_a / Z_a) * sin²(π (t - t_low)/(t_high - t_low))
    for t ∈ (t_low, t_high), else 0.

    Z_a = (t_high - t_low) / 2 so that ∫_0^1 β_a(t) dt = κ_a.

    Legacy fields ``r_lock`` / ``g_lock`` equal the lower window edge (strict lock below).
    """

    r_window: tuple[float, float] = (0.60, 0.95)
    g_window: tuple[float, float] = (0.35, 0.75)
    kappa_r: float = 4.0
    kappa_g: float = 6.0
    # Optional legacy constructor aliases
    r_lock: float | None = None
    g_lock: float | None = None

    def __post_init__(self) -> None:
        self.r_window = _as_window(self.r_window, (0.60, 0.95))
        self.g_window = _as_window(self.g_window, (0.35, 0.75))
        # Back-compat: if only locks provided (old configs), treat as [lock, 1.0)
        if self.r_lock is not None and self.r_window == (0.60, 0.95):
            # only override when user passed r_lock intentionally via field and kept default window —
            # Prefer explicit r_window from config. If r_lock set without custom window in old code paths:
            pass
        # Expose lock = lower edge for diagnostics / metrics
        object.__setattr__(self, "r_lock", float(self.r_window[0]))
        object.__setattr__(self, "g_lock", float(self.g_window[0]))

    @classmethod
    def from_config(cls, sch_cfg: dict | None) -> "AsyncJumpSchedule":
        cfg = dict(sch_cfg or {})
        r_win = cfg.get("r_window")
        g_win = cfg.get("g_window")
        if r_win is None and "r_lock" in cfg:
            # migrate old lock-only configs to [lock, 0.99]
            r_win = [float(cfg["r_lock"]), 0.99]
        if g_win is None and "g_lock" in cfg:
            g_win = [float(cfg["g_lock"]), 0.99]
        return cls(
            r_window=tuple(r_win) if r_win is not None else (0.60, 0.95),
            g_window=tuple(g_win) if g_win is not None else (0.35, 0.75),
            kappa_r=float(cfg.get("kappa_r", 4.0)),
            kappa_g=float(cfg.get("kappa_g", 6.0)),
        )

    def _window(self, kind: str) -> tuple[float, float]:
        if kind == "R":
            return self.r_window
        if kind == "G":
            return self.g_window
        raise ValueError(kind)

    def _kappa(self, kind: str) -> float:
        return self.kappa_r if kind == "R" else self.kappa_g

    def _Z(self, kind: str) -> float:
        lo, hi = self._window(kind)
        # ∫ sin²(πu) du over u∈[0,1] = 1/2; dt = Δ du ⇒ ∫sin² dt = Δ/2
        return max((hi - lo) / 2.0, 1e-12)

    def is_active(self, t: float, *, kind: str) -> bool:
        lo, hi = self._window(kind)
        tt = float(t)
        return lo < tt < hi

    def is_r_active(self, t: float) -> bool:
        return self.is_active(t, kind="R")

    def is_g_active(self, t: float) -> bool:
        return self.is_active(t, kind="G")

    def is_r_locked(self, t: float) -> bool:
        """True when R rate is strictly zero (outside open window)."""
        return not self.is_r_active(t)

    def is_g_locked(self, t: float) -> bool:
        return not self.is_g_active(t)

    def beta(self, t: float | torch.Tensor, *, kind: str) -> torch.Tensor:
        """Mobility β_a(t); 0 outside (t_low, t_high)."""
        t = torch.as_tensor(t, dtype=torch.float32)
        lo, hi = self._window(kind)
        kappa = self._kappa(kind)
        Z = self._Z(kind)
        delta = max(hi - lo, 1e-12)
        # interior mask (strict)
        u = (t - lo) / delta
        inside = (t > lo) & (t < hi)
        # sin²(π u); at boundaries u=0 or 1 → 0
        bump = torch.sin(torch.pi * u).pow(2)
        beta = (kappa / Z) * bump
        return torch.where(inside, beta, torch.zeros_like(beta))

    def beta_r(self, t: float | torch.Tensor) -> torch.Tensor:
        return self.beta(t, kind="R")

    def beta_g(self, t: float | torch.Tensor) -> torch.Tensor:
        return self.beta(t, kind="G")

    def _cum_sin2_from_low(self, t: float, *, kind: str) -> float:
        """∫_{t_low}^{clip(t)} sin²(π(u)) dτ  with u=(τ-low)/Δ (0 if t<=low)."""
        lo, hi = self._window(kind)
        delta = hi - lo
        if t <= lo:
            return 0.0
        tt = min(float(t), hi)
        u = (tt - lo) / delta
        # ∫_0^u sin²(π v) Δ dv = Δ ∫_0^u (1-cos(2πv))/2 dv = (Δ/2)(u - sin(2πu)/(2π))
        return (delta / 2.0) * (u - (torch.sin(torch.tensor(2.0 * torch.pi * u)).item() / (2.0 * torch.pi)))

    def integrated_beta(self, t0: float, t1: float, *, kind: str) -> float:
        """H_a(t0,t1) = ∫_{t0}^{t1} β_a(τ) dτ for t0 <= t1 (0 if inverted)."""
        a, b = float(t0), float(t1)
        if b <= a:
            return 0.0
        kappa = self._kappa(kind)
        Z = self._Z(kind)
        # ∫ β = (κ/Z) * ∫ sin² dt
        integ_sin = self._cum_sin2_from_low(b, kind=kind) - self._cum_sin2_from_low(a, kind=kind)
        return (kappa / Z) * integ_sin

    def integrated_hazard_total(
        self,
        t0: float,
        t1: float,
        *,
        r_on: bool = True,
        g_on: bool = True,
    ) -> float:
        """∫_{t0}^{t1} (1_R β_R + 1_G β_G) dτ."""
        h = 0.0
        if r_on:
            h += self.integrated_beta(t0, t1, kind="R")
        if g_on:
            h += self.integrated_beta(t0, t1, kind="G")
        return h

    def inverse_integrated_hazard(
        self,
        t_high: float,
        hazard: float,
        *,
        t_low: float,
        r_on: bool = True,
        g_on: bool = True,
        n_bisect: int = 48,
    ) -> float | None:
        """Reverse-time inverse: find τ ∈ [t_low, t_high] s.t. ∫_τ^{t_high} λ = hazard.

        Used by reverse Gillespie (time decreasing). Returns None if no event.
        """
        H_full = self.integrated_hazard_total(t_low, t_high, r_on=r_on, g_on=g_on)
        if hazard > H_full + 1e-12 or H_full <= 1e-30:
            return None
        if hazard <= 1e-15:
            return float(t_high)
        lo, hi = float(t_low), float(t_high)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            H = self.integrated_hazard_total(mid, t_high, r_on=r_on, g_on=g_on)
            # want H(mid, t_high) = hazard; larger mid → smaller H
            if H > hazard:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def inverse_integrated_hazard_forward(
        self,
        t_low: float,
        hazard: float,
        *,
        t_high: float,
        r_on: bool = True,
        g_on: bool = True,
        n_bisect: int = 48,
    ) -> float | None:
        """Forward-time inverse: find τ ∈ [t_low, t_high] s.t. ∫_{t_low}^τ λ = hazard.

        Used by forward CTMC (time increasing). Returns None if hazard exceeds
        segment mass (no event before t_high).
        """
        H_full = self.integrated_hazard_total(t_low, t_high, r_on=r_on, g_on=g_on)
        if hazard > H_full + 1e-12 or H_full <= 1e-30:
            return None
        if hazard <= 1e-15:
            return float(t_low)
        # Equivalent: remaining reverse hazard from τ to t_high is H_full - hazard
        remaining = H_full - float(hazard)
        if remaining <= 1e-15:
            return float(t_high)
        return self.inverse_integrated_hazard(
            t_high, remaining, t_low=t_low, r_on=r_on, g_on=g_on, n_bisect=n_bisect
        )

    def next_mobility_time(self, t: float, t_end: float) -> float:
        """Next time > t where a mobility window opens or closes, or t_end."""
        candidates = [float(t_end)]
        for lo, hi in (self.r_window, self.g_window):
            for x in (lo, hi):
                if t < x < t_end:
                    candidates.append(float(x))
        return min(candidates)

    def sample_t_proportional_to_beta(
        self,
        *,
        kind: str,
        generator: torch.Generator | None = None,
    ) -> float:
        """Sample t ~ p(t) ∝ β_a(t) on the active window (inverse CDF of sin² bump)."""
        lo, hi = self._window(kind)
        delta = hi - lo
        # CDF of density ∝ sin²(πu) on [0,1]: F(u) = 2 * ∫_0^u sin² = u - sin(2πu)/(2π)
        # sample U~Unif, invert F by bisection

        def _rand() -> float:
            if generator is None:
                return float(torch.rand(()).item())
            return float(torch.rand((), generator=generator).item())

        u_target = _rand()
        a, b = 0.0, 1.0
        for _ in range(48):
            m = 0.5 * (a + b)
            Fm = m - (torch.sin(torch.tensor(2.0 * torch.pi * m)).item() / (2.0 * torch.pi))
            # F(1)=1, F(0)=0; normalize already F(1)=1
            if Fm < u_target:
                a = m
            else:
                b = m
        u = 0.5 * (a + b)
        # stay strictly inside (lo, hi)
        t = lo + u * delta
        eps = 1e-6 * delta
        return float(min(max(t, lo + eps), hi - eps))

    def schedule_knots(self, t_start: float, t_end: float) -> list[float]:
        """Critical times in (t_start, t_end) where mobility pieces change."""
        knots = []
        for lo, hi in (self.r_window, self.g_window):
            for x in (lo, hi):
                if t_start < x < t_end:
                    knots.append(float(x))
        return sorted(set(knots))

    def expected_jump_budget(self) -> dict[str, float]:
        """E[N_a] = κ_a when legal moves always exist (upper reference)."""
        return {"R": float(self.kappa_r), "G": float(self.kappa_g), "total": float(self.kappa_r + self.kappa_g)}
