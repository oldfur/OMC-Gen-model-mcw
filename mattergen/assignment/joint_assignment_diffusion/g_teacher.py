"""Improvement-weighted set-valued G denoising teacher (J1.3-A).

Teacher reference is the gauge-invariant clean copy partition C_0 = G_0 G_0^T.
Utility is ΔF1 only; ΔARI is diagnostic. Does not change G-head or rates.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from mattergen.assignment.global_copy_assembly.metrics import pair_partition_metrics
from mattergen.assignment.global_copy_assembly.orbit_metrics import adjusted_rand_index

from .legal_moves import LegalMove, apply_move, enumerate_g_moves
from .state import JointAssignmentState


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def f1_vs_c0(state: JointAssignmentState, c0: torch.Tensor) -> float:
    return float(pair_partition_metrics(state.C(), c0)["copy_pair_f1"])


def ari_vs_clean(state: JointAssignmentState, copy0: torch.Tensor) -> float:
    return float(adjusted_rand_index(copy0, state.copy_of()))


def g_move_utilities(
    a_plus: JointAssignmentState,
    *,
    c0: torch.Tensor,
    copy0: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Score every legal G swap by clean-directed ΔF1 (and ΔARI for logs)."""
    moves = enumerate_g_moves(a_plus)
    f1_plus = f1_vs_c0(a_plus, c0)
    ari_plus = ari_vs_clean(a_plus, copy0) if copy0 is not None else 0.0
    rows: list[dict[str, Any]] = []
    for m in moves:
        am = apply_move(a_plus, m)
        f1 = f1_vs_c0(am, c0)
        d_ari = 0.0
        if copy0 is not None:
            d_ari = ari_vs_clean(am, copy0) - ari_plus
        rows.append(
            {
                "move": m,
                "key": _pair_key(m.i, m.j),
                "u": f1 - f1_plus,
                "delta_ari": d_ari,
            }
        )
    return rows


def improvement_weighted_teacher(
    utils: list[dict[str, Any]],
    *,
    temperature: float,
    eps_best: float = 1e-8,
) -> dict[str, Any]:
    """Soft q on beneficial moves; fallback softmax over all if none improve F1."""
    n = len(utils)
    if n == 0:
        return {
            "q": torch.zeros(0),
            "no_beneficial": True,
            "u_max": 0.0,
            "num_beneficial": 0,
            "num_best": 0,
            "best_keys": set(),
        }
    u = torch.tensor([float(r["u"]) for r in utils], dtype=torch.float32)
    u_max = float(u.max())
    pos = (u > 0).nonzero(as_tuple=True)[0]
    t = max(float(temperature), 1e-8)
    no_ben = int(pos.numel()) == 0
    if no_ben:
        q = torch.softmax(u / t, dim=0)
    else:
        q = torch.zeros(n, dtype=torch.float32)
        shifted = (u[pos] - u_max) / t
        q[pos] = torch.softmax(shifted, dim=0)
    best_mask = u >= (u_max - eps_best)
    best_keys = {utils[i]["key"] for i in best_mask.nonzero(as_tuple=True)[0].tolist()}
    return {
        "q": q,
        "u": u,
        "no_beneficial": no_ben,
        "u_max": u_max,
        "num_beneficial": int((u > 0).sum()),
        "num_best": int(best_mask.sum()),
        "best_keys": best_keys,
        "best_mask": best_mask,
    }


def teacher_entropy(q: torch.Tensor) -> float:
    p = q[q > 1e-12]
    if p.numel() == 0:
        return 0.0
    return float((-(p * p.log()).sum()))


def teacher_diagnostics(
    *,
    utils: list[dict[str, Any]],
    teacher: dict[str, Any],
    hist_key: tuple[int, int] | None,
) -> dict[str, Any]:
    q = teacher["q"]
    u = teacher.get("u")
    n = len(utils)
    h = teacher_entropy(q)
    n_eff = float(math.exp(h)) if h > 0 else (1.0 if n else 0.0)
    pos_u = [float(r["u"]) for r in utils if r["u"] > 0]
    hist_idx = None
    if hist_key is not None:
        for i, r in enumerate(utils):
            if r["key"] == hist_key:
                hist_idx = i
                break
    hist_u = float(utils[hist_idx]["u"]) if hist_idx is not None else None
    hist_q = float(q[hist_idx]) if hist_idx is not None and q.numel() else 0.0
    # rank by utility (1 = best ΔF1)
    hist_rank = None
    hist_best = False
    hist_ben = False
    if hist_idx is not None and u is not None:
        order = torch.argsort(u, descending=True)
        hist_rank = int((order == hist_idx).nonzero(as_tuple=True)[0].item()) + 1
        hist_best = hist_key in teacher["best_keys"]
        hist_ben = hist_u is not None and hist_u > 0
    return {
        "num_legal_G": n,
        "num_beneficial_G": teacher["num_beneficial"],
        "num_positive_teacher_support": int((q > 1e-12).sum()) if q.numel() else 0,
        "teacher_entropy_G": h,
        "teacher_effective_support_G": n_eff,
        "teacher_max_prob_G": float(q.max()) if q.numel() else 0.0,
        "best_delta_F1": teacher["u_max"],
        "mean_positive_delta_F1": float(sum(pos_u) / len(pos_u)) if pos_u else 0.0,
        "historical_delta_F1": hist_u,
        "historical_inverse_rank_by_utility": hist_rank,
        "historical_inverse_is_beneficial": hist_ben,
        "historical_inverse_is_best": hist_best,
        "historical_inverse_teacher_mass": hist_q,
        "g_teacher_no_beneficial": bool(teacher["no_beneficial"]),
    }


def policy_quality(
    *,
    utils: list[dict[str, Any]],
    teacher: dict[str, Any],
    pi: torch.Tensor,
) -> dict[str, Any]:
    """Learning metrics of π vs teacher utilities (no hist CE)."""
    u = teacher["u"]
    n = int(u.numel()) if u is not None else 0
    if n == 0 or pi.numel() == 0:
        return {
            "CE_teacher": 0.0,
            "uniform_CE_teacher": 0.0,
            "delta_CE_teacher": 0.0,
            "P_beneficial": 0.0,
            "P_best": 0.0,
            "top1_is_beneficial": 0.0,
            "top1_is_best": 0.0,
            "top1_delta_F1": 0.0,
            "top1_delta_ARI": 0.0,
            "expected_delta_F1": 0.0,
            "uniform_expected_delta_F1": 0.0,
        }
    q = teacher["q"].to(device=pi.device, dtype=pi.dtype)
    pi = pi / pi.sum().clamp_min(1e-12)
    ce = float((-(q * pi.clamp_min(1e-12).log()).sum()).detach())
    uni = float(math.log(n))
    ben = u > 0
    p_ben = float(pi[ben].sum().detach()) if ben.any() else 0.0
    p_best = float(pi[teacher["best_mask"]].sum().detach())
    top = int(torch.argmax(pi).item())
    e_pi = float((pi * u.to(device=pi.device)).sum().detach())
    e_uni = float(u.mean())
    return {
        "CE_teacher": ce,
        "uniform_CE_teacher": uni,
        "delta_CE_teacher": ce - uni,
        "P_beneficial": p_ben,
        "P_best": p_best,
        "top1_is_beneficial": 1.0 if float(u[top]) > 0 else 0.0,
        "top1_is_best": 1.0 if utils[top]["key"] in teacher["best_keys"] else 0.0,
        "top1_delta_F1": float(u[top]),
        "top1_delta_ARI": float(utils[top]["delta_ari"]),
        "expected_delta_F1": e_pi,
        "uniform_expected_delta_F1": e_uni,
    }


def align_pi_to_utils(
    pi_pool: list[tuple[LegalMove, torch.Tensor]],
    utils: list[dict[str, Any]],
) -> torch.Tensor:
    """π vector in teacher/move-utility order."""
    amap = {_pair_key(m.i, m.j): p for m, p in pi_pool}
    if not utils:
        return torch.zeros(0)
    device = pi_pool[0][1].device if pi_pool else "cpu"
    vec = []
    for r in utils:
        p = amap.get(r["key"])
        if p is None:
            vec.append(torch.zeros((), device=device))
        else:
            vec.append(p)
    stacked = torch.stack(vec)
    return stacked / stacked.sum().clamp_min(1e-12)
