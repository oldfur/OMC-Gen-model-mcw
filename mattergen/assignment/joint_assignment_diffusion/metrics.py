"""J1 validation metrics: legality, lock schedules, copy metrics."""
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
    return {
        "num_states": len(traj.states),
        "illegal_state_count": illegal,
        "all_legal": illegal == 0,
        "num_events": len(traj.events),
    }


def lock_schedule_checks(traj: CTMCTrajectory, schedule: AsyncJumpSchedule) -> dict[str, Any]:
    r_after_lock = sum(1 for e in traj.events if e.kind == "R" and e.time <= schedule.r_lock + 1e-9)
    g_after_lock = sum(1 for e in traj.events if e.kind == "G" and e.time <= schedule.g_lock + 1e-9)
    return {
        "r_jumps_at_or_below_lock": r_after_lock,
        "g_jumps_at_or_below_lock": g_after_lock,
        "r_lock_ok": r_after_lock == 0,
        "g_lock_ok": g_after_lock == 0,
        "r_lock": schedule.r_lock,
        "g_lock": schedule.g_lock,
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
