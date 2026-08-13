"""J1 validation metrics: legality, mobility windows, segment hazard, copy metrics."""
from __future__ import annotations

from typing import Any

import torch

from mattergen.assignment.global_copy_assembly.orbit_metrics import adjusted_rand_index
from mattergen.assignment.global_copy_assembly.metrics import pair_partition_metrics

from .ctmc import CTMCTrajectory
from .schedule import AsyncJumpSchedule
from .state import JointAssignmentState


def trajectory_legality(traj: CTMCTrajectory) -> dict[str, Any]:
    illegal = 0
    for st in traj.states:
        if not st.validate()["legal"]:
            illegal += 1
    n_R = sum(1 for e in traj.events if e.kind == "R")
    n_G = sum(1 for e in traj.events if e.kind == "G")
    return {
        "num_states": len(traj.states),
        "illegal_state_count": illegal,
        "all_legal": illegal == 0,
        "num_events": len(traj.events),
        "n_R": n_R,
        "n_G": n_G,
    }


def lock_schedule_checks(traj: CTMCTrajectory, schedule: AsyncJumpSchedule) -> dict[str, Any]:
    """Jumps must lie strictly inside active mobility windows."""
    r_outside = sum(1 for e in traj.events if e.kind == "R" and not schedule.is_r_active(e.time))
    g_outside = sum(1 for e in traj.events if e.kind == "G" and not schedule.is_g_active(e.time))
    return {
        "r_jumps_outside_window": r_outside,
        "g_jumps_outside_window": g_outside,
        "r_jumps_at_or_below_lock": r_outside,
        "g_jumps_at_or_below_lock": g_outside,
        "r_lock_ok": r_outside == 0,
        "g_lock_ok": g_outside == 0,
        "r_lock": schedule.r_lock,
        "g_lock": schedule.g_lock,
        "r_window": list(schedule.r_window),
        "g_window": list(schedule.g_window),
    }


def jump_budget_diagnostics(
    traj: CTMCTrajectory,
    schedule: AsyncJumpSchedule,
    *,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> dict[str, Any]:
    """Compare actual jumps to **segment** integrated hazard H_a(s,t)=∫_s^t β_a.

    Also reports full-window integrals for reference (κ when legal always exist).
    """
    n_R = sum(1 for e in traj.events if e.kind == "R")
    n_G = sum(1 for e in traj.events if e.kind == "G")
    H_R_full = schedule.integrated_beta(0.0, 1.0, kind="R")
    H_G_full = schedule.integrated_beta(0.0, 1.0, kind="G")
    H_R_seg = schedule.integrated_beta(t_start, t_end, kind="R")
    H_G_seg = schedule.integrated_beta(t_start, t_end, kind="G")
    return {
        "n_R": n_R,
        "n_G": n_G,
        "H_R_full": H_R_full,
        "H_G_full": H_G_full,
        "H_R_segment": H_R_seg,
        "H_G_segment": H_G_seg,
        "expected_R_segment": H_R_seg,
        "expected_G_segment": H_G_seg,
        "expected_R": schedule.kappa_r,
        "expected_G": schedule.kappa_g,
        "ratio_R_vs_segment": n_R / max(H_R_seg, 1e-8),
        "ratio_G_vs_segment": n_G / max(H_G_seg, 1e-8),
        # legacy keys (full-window)
        "H_R": H_R_full,
        "H_G": H_G_full,
        "ratio_R": n_R / max(schedule.kappa_r, 1e-8),
        "ratio_G": n_G / max(schedule.kappa_g, 1e-8),
    }


def assignment_vs_target(state: JointAssignmentState, target: JointAssignmentState) -> dict[str, float | bool]:
    C = state.C()
    C0 = target.C()
    pair = pair_partition_metrics(C, C0)
    ari = adjusted_rand_index(target.copy_of(), state.copy_of())
    orbit_acc = float((state.orbit_of() == target.orbit_of()).float().mean())
    return {
        **{k: float(v) if not isinstance(v, bool) else v for k, v in pair.items()},
        "ARI": float(ari),
        "orbit_atom_accuracy": orbit_acc,
    }
