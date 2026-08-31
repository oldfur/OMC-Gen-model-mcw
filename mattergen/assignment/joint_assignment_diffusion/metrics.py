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


def assignment_distances(pred: JointAssignmentState, target: JointAssignmentState) -> dict[str, float]:
    """Scalar distances used by reverse-eval Δd (lower is closer to target)."""
    m = assignment_vs_target(pred, target)
    orbit_acc = float(m["orbit_atom_accuracy"])
    ari = float(m["ARI"])
    f1 = float(m["copy_pair_f1"])
    return {
        "d_orbit": 1.0 - orbit_acc,
        "d_ari": 1.0 - ari,
        "d_f1": 1.0 - f1,
        "orbit_atom_accuracy": orbit_acc,
        "orbit_exact": 1.0 if orbit_acc >= 1.0 - 1e-12 else 0.0,
        "ARI": ari,
        "copy_pair_f1": f1,
        "exact_C": 1.0 if bool(m["exact_C"]) else 0.0,
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


def pbc_min_dist(frac: torch.Tensor, cell: torch.Tensor) -> float:
    """Minimum-image pairwise distance (self excluded)."""
    n = int(frac.shape[0])
    if n < 2:
        return float("inf")
    if cell.ndim == 3:
        cell = cell.reshape(-1, 3, 3)[0]
    delta = frac[:, None, :] - frac[None, :, :]
    delta = delta - torch.round(delta)
    dist = torch.linalg.norm(delta @ cell, dim=-1)
    dist = dist + torch.eye(n, device=dist.device, dtype=dist.dtype) * 1e9
    return float(dist.min().item())


def crystal_geometry_vs_target(
    frac: torch.Tensor,
    cell: torch.Tensor,
    target_frac: torch.Tensor,
    target_cell: torch.Tensor,
    *,
    copy_of: torch.Tensor | None = None,
    min_dist_cutoff: float = 0.7,
    density_range: tuple[float, float] = (0.04, 0.25),
) -> dict[str, float | bool]:
    """Existing crystal-generation sanity: clash, density, volume vs clean target."""
    cell = cell.reshape(3, 3) if cell.numel() == 9 else cell.reshape(-1, 3, 3)[0]
    tcell = target_cell.reshape(3, 3) if target_cell.numel() == 9 else target_cell.reshape(-1, 3, 3)[0]
    n = int(frac.shape[0])
    vol = abs(float(torch.det(cell).item()))
    tvol = abs(float(torch.det(tcell).item()))
    dens = n / vol if vol > 1e-12 else float("inf")
    tdens = n / tvol if tvol > 1e-12 else float("inf")
    mind = pbc_min_dist(frac, cell) if vol > 1e-12 else 0.0
    inter = None
    if copy_of is not None and int(copy_of.max().item()) > int(copy_of.min().item()):
        # min distance between different copies (intermolecular contact)
        k = copy_of.long()
        delta = frac[:, None, :] - frac[None, :, :]
        delta = delta - torch.round(delta)
        dist = torch.linalg.norm(delta @ cell, dim=-1)
        mask = k[:, None] != k[None, :]
        if bool(mask.any()):
            inter = float(dist[mask].min().item())
    no_clash = mind >= float(min_dist_cutoff)
    valid_cell = density_range[0] <= dens <= density_range[1]
    return {
        "volume": vol,
        "atom_density": dens,
        "min_dist": mind,
        "no_clash": bool(no_clash),
        "valid_cell": bool(valid_cell),
        "pass_basic": bool(valid_cell and no_clash),
        "volume_ratio": (vol / tvol) if tvol > 1e-12 else float("nan"),
        "density_ratio": (dens / tdens) if tdens > 0 and tdens != float("inf") else float("nan"),
        "inter_copy_min_dist": float(inter) if inter is not None else float("nan"),
    }
