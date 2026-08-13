"""A-first Lie reverse sampler with integrated-hazard CTMC (J1.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .ctmc import CTMCEvent
from .jump_heads import logits_to_pi
from .legal_moves import LegalMove, apply_move
from .schedule import AsyncJumpSchedule
from .state import JointAssignmentState, sample_uniform_legal_prior


@dataclass
class SampleTrajectory:
    times: list[float] = field(default_factory=list)
    assignments: list[JointAssignmentState] = field(default_factory=list)
    events: list[CTMCEvent] = field(default_factory=list)
    frac_list: list[torch.Tensor] = field(default_factory=list)
    cell_list: list[torch.Tensor] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _rand(generator: torch.Generator | None) -> float:
    if generator is None:
        return float(torch.rand(()).item())
    return float(torch.rand((), generator=generator).item())


def _uniformize_pi(pi: dict) -> dict:
    """Replace learned π with uniform 1/|M| on each legal pool (fixed β still applies)."""
    out = {"R": [], "G": []}
    for kind in ("R", "G"):
        pool = pi.get(kind) or []
        n = len(pool)
        if n == 0:
            continue
        u = 1.0 / float(n)
        out[kind] = [(m, p * 0.0 + u) for m, p in pool]
    return out


def _gillespie_step_a_integrated(
    model,
    chemgraph,
    t: float,
    t_target: float,
    state: JointAssignmentState,
    *,
    generator: torch.Generator | None = None,
    max_events: int = 10_000,
    static_A: bool = False,
    policy: str = "learned",
) -> tuple[JointAssignmentState, list[CTMCEvent], dict[str, Any]]:
    """Reverse A dynamics on (t_target, t] with non-homogeneous integrated hazard.

    Does **not** freeze β at the left macrostep endpoint. Event times solve
    ∫_τ^{cur} λ(u) du = E for E ~ Exp(1), with λ = 1_{|M_R|>0} β_R + 1_{|M_G|>0} β_G.
    After each event, legal moves and π are recomputed (stochastic CTMC).
    """
    events: list[CTMCEvent] = []
    cur_t = float(t)
    st = state.clone()
    schedule: AsyncJumpSchedule = model.schedule
    diag = {
        "attempted_events": 0,
        "H_R_macro": schedule.integrated_beta(t_target, t, kind="R"),
        "H_G_macro": schedule.integrated_beta(t_target, t, kind="G"),
    }
    if static_A:
        return st, events, diag

    n_ev = 0
    while cur_t > t_target + 1e-12 and n_ev < max_events:
        t_ten = torch.tensor([cur_t], device=chemgraph["pos"].device, dtype=torch.float32)
        out = model(chemgraph, t_ten, st, compute_jumps=True)
        n_r = len(out.move_logits.get("R", []))
        n_g = len(out.move_logits.get("G", []))
        r_on = n_r > 0
        g_on = n_g > 0
        # If both inactive for remaining interval, done
        H_left = schedule.integrated_hazard_total(t_target, cur_t, r_on=r_on, g_on=g_on)
        if H_left <= 1e-30:
            cur_t = t_target
            break
        E = -float(torch.log(torch.tensor(max(_rand(generator), 1e-12))).item())
        diag["attempted_events"] += 1
        if E > H_left + 1e-12:
            cur_t = t_target
            break
        tau = schedule.inverse_integrated_hazard(
            cur_t, E, t_low=t_target, r_on=r_on, g_on=g_on
        )
        if tau is None or tau >= cur_t - 1e-15:
            cur_t = t_target
            break
        cur_t = float(tau)
        # Kind probabilities ∝ β_a(cur) among active non-empty pools
        beta_r = float(schedule.beta_r(cur_t).item()) if r_on else 0.0
        beta_g = float(schedule.beta_g(cur_t).item()) if g_on else 0.0
        # Re-evaluate π at event time (state unchanged until jump)
        t_ten = torch.tensor([cur_t], device=chemgraph["pos"].device, dtype=torch.float32)
        out = model(chemgraph, t_ten, st, compute_jumps=True)
        pi = logits_to_pi(out.move_logits)
        if policy == "uniform":
            pi = _uniformize_pi(pi)
        mass_r = beta_r if pi["R"] else 0.0
        mass_g = beta_g if pi["G"] else 0.0
        tot = mass_r + mass_g
        if tot <= 1e-30:
            # β≈0 at this τ (window edge); step slightly earlier and continue
            # (must decrease cur_t or reverse loop can stall)
            cur_t = max(t_target, cur_t - 1e-6)
            continue
        pick = _rand(generator) * tot
        if pick <= mass_r and pi["R"]:
            kind = "R"
            pool = pi["R"]
        else:
            kind = "G"
            pool = pi["G"]
            if not pool and pi["R"]:
                kind = "R"
                pool = pi["R"]
        if not pool:
            cur_t = max(t_target, cur_t - 1e-6)
            continue
        # Sample move ~ π
        u = _rand(generator)
        acc = 0.0
        chosen_m: LegalMove | None = None
        for m, p in pool:
            acc += float(p.detach().item())
            if u <= acc:
                chosen_m = m
                break
        if chosen_m is None:
            chosen_m = pool[-1][0]
        st = apply_move(st, chosen_m)
        events.append(CTMCEvent(time=cur_t, kind=kind, i=chosen_m.i, j=chosen_m.j))
        n_ev += 1
        if not st.validate()["legal"]:
            raise RuntimeError("sampler left legal assignment space")
    diag["n_events_macro"] = len(events)
    diag["n_R_macro"] = sum(1 for e in events if e.kind == "R")
    diag["n_G_macro"] = sum(1 for e in events if e.kind == "G")
    return st, events, diag


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
    static_A: bool = False,
) -> tuple[JointAssignmentState, torch.Tensor, torch.Tensor, list[CTMCEvent], dict[str, Any]]:
    """One macrostep t→s: A-step (integrated hazard) then geometry-step."""
    cg_t = chemgraph_builder(sample_tensors, frac_t, cell_t)
    state_s, events, a_diag = _gillespie_step_a_integrated(
        model, cg_t, t, s, state, generator=generator, static_A=static_A, policy="learned"
    )
    cg_s_cond = chemgraph_builder(sample_tensors, frac_t, cell_t)
    t_ten = torch.tensor([t], device=frac_t.device, dtype=torch.float32)
    out = model(cg_s_cond, t_ten, state_s, compute_jumps=False)
    if score_to_prev is not None:
        frac_s, cell_s = score_to_prev(frac_t, cell_t, out.chemgraph_scores, t, s)
    else:
        pos_score = out.chemgraph_scores["pos"]
        cell_score = out.chemgraph_scores["cell"]
        dt = t - s
        frac_s = (frac_t - dt * pos_score).remainder(1.0)
        cell_s = cell_t - dt * (cell_score.squeeze(0) if cell_score.ndim == 3 else cell_score)
    return state_s, frac_s, cell_s, events, a_diag


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
    assignment_mode: str = "ctmc_A",
) -> SampleTrajectory:
    """Full reverse from T=1 to 0 with A-first Lie splitting.

    assignment_mode:
      - ``ctmc_A``: stochastic reverse CTMC on A (default)
      - ``static_A``: A_t = A_T for all t (persistent random legal assignment)
    """
    if assignment_mode not in ("ctmc_A", "static_A"):
        raise ValueError(f"unknown assignment_mode={assignment_mode}")
    static_A = assignment_mode == "static_A"

    n = int(atomic_numbers.numel())
    device = atomic_numbers.device
    if generator is None:
        frac = torch.rand(n, 3, device=device)
        cell = torch.eye(3, device=device) * (float(n) / 0.05) ** (1.0 / 3.0)
    else:
        frac = torch.rand(n, 3, generator=generator).to(device)
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
    # dedupe while preserving order
    dedup = []
    for x in times:
        if not dedup or abs(dedup[-1] - x) > 1e-12:
            dedup.append(float(x))
    times = dedup

    schedule: AsyncJumpSchedule = model.schedule
    traj.diagnostics = {
        "assignment_mode": assignment_mode,
        "expected_jumps": schedule.expected_jump_budget(),
        "H_R_full": schedule.integrated_beta(0.0, 1.0, kind="R"),
        "H_G_full": schedule.integrated_beta(0.0, 1.0, kind="G"),
        "jumps_per_bin": {},
        "macro_H": [],
    }

    traj.times.append(times[0])
    traj.assignments.append(state.clone())
    traj.frac_list.append(frac.clone())
    traj.cell_list.append(cell.clone())

    for t, s in zip(times[:-1], times[1:]):
        state, frac, cell, events, a_diag = a_first_lie_step(
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
            static_A=static_A,
        )
        traj.times.append(s)
        traj.assignments.append(state.clone())
        traj.frac_list.append(frac.clone())
        traj.cell_list.append(cell.clone())
        traj.events.extend(events)
        bin_key = f"{t:.2f}->{s:.2f}"
        traj.diagnostics["jumps_per_bin"][bin_key] = {
            "n": len(events),
            "n_R": sum(1 for e in events if e.kind == "R"),
            "n_G": sum(1 for e in events if e.kind == "G"),
            "H_R_segment": a_diag.get("H_R_macro", 0.0),
            "H_G_segment": a_diag.get("H_G_macro", 0.0),
            "expected_R_segment": a_diag.get("H_R_macro", 0.0),
            "expected_G_segment": a_diag.get("H_G_macro", 0.0),
        }
        traj.diagnostics["macro_H"].append(a_diag)

    n_R = sum(1 for e in traj.events if e.kind == "R")
    n_G = sum(1 for e in traj.events if e.kind == "G")
    traj.diagnostics["n_R"] = n_R
    traj.diagnostics["n_G"] = n_G
    traj.diagnostics["n_total"] = n_R + n_G
    return traj
