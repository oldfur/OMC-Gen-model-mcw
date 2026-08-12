"""A-first Lie splitting reverse sampler (stochastic CTMC + MatterGen geometry)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .ctmc import CTMCEvent
from .legal_moves import LegalMove, apply_move, enumerate_legal_moves
from .schedule import AsyncJumpSchedule
from .state import JointAssignmentState, sample_uniform_legal_prior


@dataclass
class SampleTrajectory:
    times: list[float] = field(default_factory=list)
    assignments: list[JointAssignmentState] = field(default_factory=list)
    events: list[CTMCEvent] = field(default_factory=list)
    frac_list: list[torch.Tensor] = field(default_factory=list)
    cell_list: list[torch.Tensor] = field(default_factory=list)


def _gillespie_step_a(
    model,
    chemgraph,
    t: float,
    t_target: float,
    state: JointAssignmentState,
    *,
    generator: torch.Generator | None = None,
    max_events: int = 10_000,
) -> tuple[JointAssignmentState, list[CTMCEvent]]:
    """Stochastic A reverse/forward dynamics with fixed geometry; recompute logits each jump."""
    events: list[CTMCEvent] = []
    cur_t = float(t)
    st = state.clone()
    schedule: AsyncJumpSchedule = model.schedule

    def _rand() -> float:
        if generator is None:
            return float(torch.rand(()).item())
        return float(torch.rand((), generator=generator).item())

    n_ev = 0
    # Reverse time: we integrate from t down to t_target (t_target < t)
    while cur_t > t_target + 1e-12 and n_ev < max_events:
        t_ten = torch.tensor([cur_t], device=chemgraph["pos"].device)
        out = model(chemgraph, t_ten, st, compute_jumps=True)
        rates_flat: list[tuple[LegalMove, float, str]] = []
        total = 0.0
        for kind in ("R", "G"):
            for m, r in out.move_rates[kind]:
                rv = float(r.detach().item())
                if rv > 0:
                    rates_flat.append((m, rv, kind))
                    total += rv
        if total <= 1e-30:
            # freeze until next unlock going downward or stop
            # when going reverse, unlocks are crossed as time decreases past lock
            cur_t = t_target
            break
        u = max(_rand(), 1e-12)
        dt = -float(torch.log(torch.tensor(u)).item()) / total
        # reverse: decrease time
        if cur_t - dt <= t_target:
            cur_t = t_target
            break
        cur_t = cur_t - dt
        pick = _rand() * total
        acc = 0.0
        chosen = rates_flat[-1]
        for item in rates_flat:
            acc += item[1]
            if pick <= acc:
                chosen = item
                break
        m, _r, kind = chosen
        st = apply_move(st, m)
        events.append(CTMCEvent(time=cur_t, kind=kind, i=m.i, j=m.j))
        n_ev += 1
        if not st.validate()["legal"]:
            raise RuntimeError("sampler left legal assignment space")
    return st, events


def a_first_lie_step(
    *,
    model,
    chemgraph_builder,
    sample_tensors: dict,
    state: JointAssignmentState,
    frac_t: torch.Tensor,
    cell_t: torch.Tensor,
    t: float,
    s: float,
    generator: torch.Generator | None = None,
    noise_adapter=None,
    score_to_prev=None,
) -> tuple[JointAssignmentState, torch.Tensor, torch.Tensor, list[CTMCEvent]]:
    """One macrostep t→s: A-step then geometry-step."""
    # A-step with fixed X_t,L_t
    cg_t = chemgraph_builder(sample_tensors, frac_t, cell_t)
    state_s, events = _gillespie_step_a(model, cg_t, t, s, state, generator=generator)
    # Geometry-step with A_s
    cg_s_cond = chemgraph_builder(sample_tensors, frac_t, cell_t)
    t_ten = torch.tensor([t], device=frac_t.device)
    out = model(cg_s_cond, t_ten, state_s, compute_jumps=False)
    # Native geometry reverse: if score_to_prev provided use it; else Euler-like residual
    if score_to_prev is not None:
        frac_s, cell_s = score_to_prev(frac_t, cell_t, out.chemgraph_scores, t, s)
    else:
        # lightweight fallback: small step along -score (not production MatterGen PC)
        pos_score = out.chemgraph_scores["pos"]
        cell_score = out.chemgraph_scores["cell"]
        dt = t - s
        frac_s = (frac_t - dt * pos_score).remainder(1.0)
        cell_s = cell_t - dt * (cell_score.squeeze(0) if cell_score.ndim == 3 else cell_score)
    return state_s, frac_s, cell_s, events


def sample_joint_prior_and_trajectory(
    *,
    model,
    partition,
    atomic_numbers: torch.Tensor,
    role_z: torch.Tensor,
    K: int,
    sample_tensors: dict,
    chemgraph_builder,
    timesteps: list[float],
    generator: torch.Generator | None = None,
    noise_adapter=None,
) -> SampleTrajectory:
    """Full reverse from T=1 to 0 with A-first Lie splitting."""
    # Geometry prior ~ isotropic noise in frac (MVP); cell ~ identity scale
    n = int(atomic_numbers.numel())
    device = atomic_numbers.device
    if generator is None:
        frac = torch.rand(n, 3, device=device)
        cell = torch.eye(3, device=device) * (float(n) / 0.05) ** (1.0 / 3.0)
    else:
        frac = torch.rand(n, 3, generator=generator)
        # move to device
        frac = frac.to(device)
        cell = torch.eye(3, device=device) * (float(n) / 0.05) ** (1.0 / 3.0)
    state = sample_uniform_legal_prior(
        partition=partition,
        atomic_numbers=atomic_numbers,
        role_z=role_z,
        K=K,
        generator=generator,
    )
    traj = SampleTrajectory()
    times = sorted(timesteps, reverse=True)
    if times[0] < 1.0:
        times = [1.0] + times
    if times[-1] > 0.0:
        times = times + [0.0]
    traj.times.append(times[0])
    traj.assignments.append(state.clone())
    traj.frac_list.append(frac.clone())
    traj.cell_list.append(cell.clone())
    for t, s in zip(times[:-1], times[1:]):
        state, frac, cell, events = a_first_lie_step(
            model=model,
            chemgraph_builder=chemgraph_builder,
            sample_tensors=sample_tensors,
            state=state,
            frac_t=frac,
            cell_t=cell,
            t=t,
            s=s,
            generator=generator,
            noise_adapter=noise_adapter,
        )
        traj.times.append(s)
        traj.assignments.append(state.clone())
        traj.frac_list.append(frac.clone())
        traj.cell_list.append(cell.clone())
        traj.events.extend(events)
    return traj
