"""Forward stochastic CTMC on legal assignment states."""
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
        # states[k] holds on [times[k], times[k+1])
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
    """Exact Gillespie CTMC with time-dependent rates β_a(t)/|M_a|.

    Q_a(A, A^m; t) = β_a(t) / |M_a(A)| for each legal move m of type a.
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
        rates: list[tuple[LegalMove, float]] = []
        total = 0.0
        for kind in ("R", "G"):
            beta = float(schedule.beta(t, kind=kind).item())
            pool = moves[kind]
            msize = len(pool)
            if beta <= 0.0 or msize == 0:
                continue
            r_each = beta / float(msize)
            for m in pool:
                rates.append((m, r_each))
                total += r_each
        if total <= 1e-30:
            # no jumps possible until some unlock; jump to next unlock or end
            next_t = t_end
            if schedule.is_g_locked(t) and schedule.g_lock < t_end:
                next_t = min(next_t, schedule.g_lock + 1e-9)
            if schedule.is_r_locked(t) and schedule.r_lock < t_end:
                next_t = min(next_t, schedule.r_lock + 1e-9)
            if next_t <= t + 1e-12:
                break
            t = min(next_t, t_end)
            times.append(t)
            states.append(state.clone())
            continue
        # waiting time with frozen rates (piecewise approx between unlocks)
        u = max(_rand(), 1e-12)
        dt = -torch.log(torch.tensor(u)).item() / total
        # clamp to next schedule knot
        knots = [t_end]
        if t < schedule.g_lock < t_end:
            knots.append(schedule.g_lock)
        if t < schedule.r_lock < t_end:
            knots.append(schedule.r_lock)
        t_next_knot = min(knots)
        if t + dt >= t_next_knot:
            t = t_next_knot
            times.append(t)
            states.append(state.clone())
            continue
        # accept a move
        t = t + dt
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
