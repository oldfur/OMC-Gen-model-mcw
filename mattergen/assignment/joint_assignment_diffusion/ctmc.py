"""Forward stochastic CTMC on legal assignment states (fixed total exit rate)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .legal_moves import LegalMove, apply_move, enumerate_legal_moves
from .schedule import AsyncJumpSchedule
from .state import JointAssignmentState


@dataclass
class CTMCEvent:
    time: float
    kind: str
    i: int
    j: int


@dataclass
class CTMCTrajectory:
    times: list[float]
    states: list[JointAssignmentState]
    events: list[CTMCEvent] = field(default_factory=list)

    def state_at(self, t: float) -> JointAssignmentState:
        """Piecewise-constant left-continuous state for time t in [0,1]."""
        for k in range(len(self.times) - 1):
            if self.times[k] <= t < self.times[k + 1]:
                return self.states[k]
        return self.states[-1]


def simulate_forward_ctmc(
    state0: JointAssignmentState,
    *,
    schedule: AsyncJumpSchedule,
    t_start: float = 0.0,
    t_end: float = 1.0,
    generator: torch.Generator | None = None,
    max_events: int = 100_000,
) -> CTMCTrajectory:
    """Gillespie CTMC with fixed total exit rates β_a(t).

    Forward proposal is uniform on legal moves of each kind:
      r_m^a = β_a(t) / |M_a|   ⇒   Σ_m r_m^a = β_a(t)
    (matches reverse training of π; forward prior uses π_uniform).

    Time-dependent β uses integrated-hazard-aware piecewise knots at window edges.
    """
    if t_end <= t_start:
        raise ValueError("t_end must exceed t_start")
    t = float(t_start)
    state = state0.clone()
    times = [t]
    states = [state.clone()]
    events: list[CTMCEvent] = []

    def _rand() -> float:
        if generator is None:
            return float(torch.rand(()).item())
        return float(torch.rand((), generator=generator).item())

    n_events = 0
    while t < t_end - 1e-12 and n_events < max_events:
        moves = enumerate_legal_moves(state)
        r_on = len(moves["R"]) > 0
        g_on = len(moves["G"]) > 0
        beta_r = float(schedule.beta_r(t).item()) if r_on else 0.0
        beta_g = float(schedule.beta_g(t).item()) if g_on else 0.0
        total = 0.0
        rates: list[tuple[LegalMove, float]] = []
        for kind, beta, pool in (("R", beta_r, moves["R"]), ("G", beta_g, moves["G"])):
            if beta <= 0.0 or not pool:
                continue
            r_each = beta / float(len(pool))
            for m in pool:
                rates.append((m, r_each))
                total += r_each

        if total <= 1e-30:
            # jump to next mobility opening or end (do not freeze forever)
            next_candidates = [t_end]
            for lo, hi in (schedule.r_window, schedule.g_window):
                if t < lo < t_end:
                    next_candidates.append(lo + 1e-9)
            next_t = min(next_candidates)
            if next_t <= t + 1e-12:
                break
            t = min(next_t, t_end)
            times.append(t)
            states.append(state.clone())
            continue

        # non-homogeneous: use integrated hazard over [t, next_knot]
        knots = schedule.schedule_knots(t, t_end) + [t_end]
        t_next_knot = min(k for k in knots if k > t + 1e-15)
        # Between knots, β varies continuously; invert cumulative hazard with on-flags
        E = -float(torch.log(torch.tensor(max(_rand(), 1e-12))).item())
        H_seg = schedule.integrated_hazard_total(t, t_next_knot, r_on=r_on, g_on=g_on)
        if E > H_seg + 1e-12:
            # no event before next knot
            t = t_next_knot
            times.append(t)
            states.append(state.clone())
            continue
        t_ev = schedule.inverse_integrated_hazard(
            t_next_knot, E, t_low=t, r_on=r_on, g_on=g_on
        )
        if t_ev is None or t_ev <= t + 1e-15:
            t = t_next_knot
            times.append(t)
            states.append(state.clone())
            continue
        t = float(t_ev)
        # Recompute rates at event time for kind choice (β(t) may differ)
        beta_r = float(schedule.beta_r(t).item()) if r_on else 0.0
        beta_g = float(schedule.beta_g(t).item()) if g_on else 0.0
        rates = []
        total = 0.0
        for kind, beta, pool in (("R", beta_r, moves["R"]), ("G", beta_g, moves["G"])):
            if beta <= 0.0 or not pool:
                continue
            r_each = beta / float(len(pool))
            for m in pool:
                rates.append((m, r_each))
                total += r_each
        if total <= 1e-30:
            times.append(t)
            states.append(state.clone())
            continue
        pick = _rand() * total
        acc = 0.0
        chosen: LegalMove | None = None
        for m, r in rates:
            acc += r
            if pick <= acc:
                chosen = m
                break
        if chosen is None:
            chosen = rates[-1][0]
        state = apply_move(state, chosen)
        events.append(CTMCEvent(time=t, kind=chosen.kind, i=chosen.i, j=chosen.j))
        times.append(t)
        states.append(state.clone())
        n_events += 1
        v = state.validate()
        if not v["legal"]:
            raise RuntimeError(f"CTMC left legal space: {v}")

    if times[-1] < t_end:
        times.append(t_end)
        states.append(state.clone())
    return CTMCTrajectory(times=times, states=states, events=events)
