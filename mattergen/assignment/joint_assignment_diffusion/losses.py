"""Joint losses: MatterGen geometry + reverse CTMC point-process NLL for R/G."""
from __future__ import annotations

from typing import Any

import torch

from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss

from .ctmc import CTMCTrajectory
from .jump_heads import logits_to_rates
from .legal_moves import enumerate_legal_moves
from .schedule import AsyncJumpSchedule
from .state import JointAssignmentState


def reverse_ctmc_segment_nll(
    *,
    model,
    chemgraph_t,
    t_start: float,
    t_end: float,
    traj: CTMCTrajectory,
    t_geom: torch.Tensor,
    clip: float = 8.0,
    n_quad: int = 4,
) -> dict[str, torch.Tensor]:
    """Reverse-rate point-process NLL on forward segment [t_start, t_end].

    Forward events are reversed: integrating from t_end → t_start with
    reverse rates evaluated at fixed (X_t, L_t) geometry (A-first Lie).

    L = ∫ λ_θ dτ − Σ log r_m(τ_e)  (for reverse events).
    """
    device = t_geom.device
    # collect reverse events in (s,t] = (t_start, t_end]
    rev_events = [e for e in traj.events if t_start < e.time <= t_end]
    # reverse order
    rev_events = list(reversed(rev_events))

    # Survival / integrated rate: midpoint quadrature on reverse time
    # reverse time u from 0..Δ maps to forward time t_end - u
    delta = t_end - t_start
    if delta <= 0:
        z = torch.zeros((), device=device)
        return {"L_R": z, "L_G": z, "L_A": z}

    integrated_r = torch.zeros((), device=device)
    integrated_g = torch.zeros((), device=device)
    for q in range(n_quad):
        # midpoint
        u = (q + 0.5) / n_quad * delta
        t_fwd = t_end - u
        # state just after reverse progress: walk reverse events
        # approx: use traj.state_at(t_fwd)
        st = traj.state_at(t_fwd)
        out = model(chemgraph_t, t_geom, st, compute_jumps=True)
        # total rate mass
        for kind, integ in (("R", "r"), ("G", "g")):
            rates = out.move_rates[kind]
            if not rates:
                continue
            tot = torch.stack([r for _, r in rates]).sum()
            if kind == "R":
                integrated_r = integrated_r + tot * (delta / n_quad)
            else:
                integrated_g = integrated_g + tot * (delta / n_quad)

    # event log-likelihood (reverse)
    log_r = torch.zeros((), device=device)
    log_g = torch.zeros((), device=device)
    # reverse path states: start from traj.state_at(t_end)
    st = traj.state_at(t_end).clone()
    for e in rev_events:
        # reverse event undoes forward swap: same swap
        out = model(chemgraph_t, t_geom, st, compute_jumps=True)
        found = None
        kind = e.kind
        for m, r in out.move_rates[kind]:
            if {m.i, m.j} == {e.i, e.j}:
                found = r
                break
        if found is None:
            # illegal under reverse state — large penalty
            pen = torch.tensor(20.0, device=device)
            if kind == "R":
                log_r = log_r - pen
            else:
                log_g = log_g - pen
        else:
            if kind == "R":
                log_r = log_r + torch.log(found.clamp_min(1e-12))
            else:
                log_g = log_g + torch.log(found.clamp_min(1e-12))
        from .legal_moves import LegalMove, apply_move

        st = apply_move(st, LegalMove(kind, e.i, e.j))

    L_R = integrated_r - log_r
    L_G = integrated_g - log_g
    return {"L_R": L_R, "L_G": L_G, "L_A": L_R + L_G}


def joint_training_step_losses(
    *,
    model,
    loss_fn,
    corruption,
    clean_cg,
    noisy_cg,
    t: torch.Tensor,
    state_for_geometry: JointAssignmentState,
    traj: CTMCTrajectory,
    t_s: float,
    t_t: float,
    lambda_r: float = 1.0,
    lambda_g: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Geometry branch uses A_s; assignment reverse NLL on segment with fixed X_t,L_t."""
    # NoiseLevelEncoding requires 1D batch time: shape [B], not scalar 0-dim.
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1)
    if noisy_cg["pos"].device.type != "cpu":
        t = t.to(device=noisy_cg["pos"].device)
    # geometry with A_s (cleaner assignment for A-first Lie)
    geom_out = model(noisy_cg, t, state_for_geometry, compute_jumps=False)
    L_geom, metrics = mattergen_geometry_loss(
        loss_fn=loss_fn,
        corruption=corruption,
        clean_batch=clean_cg,
        noisy_batch=noisy_cg,
        score_model_output=geom_out.chemgraph_scores,
        t=t,
    )
    nll = reverse_ctmc_segment_nll(
        model=model,
        chemgraph_t=noisy_cg,
        t_start=t_s,
        t_end=t_t,
        traj=traj,
        t_geom=t,
    )
    total = L_geom + lambda_r * nll["L_R"] + lambda_g * nll["L_G"]
    return {
        "loss": total,
        "L_geom": L_geom,
        "L_R": nll["L_R"],
        "L_G": nll["L_G"],
        **{f"geom_{k}": torch.tensor(v, device=L_geom.device) if not torch.is_tensor(v) else v for k, v in metrics.items()},
    }
