"""Joint losses: MatterGen geometry + event-mean CE on reverse segment (s, t]."""
from __future__ import annotations

import math
from typing import Any

import torch

from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss

from .ctmc import CTMCTrajectory
from .g_teacher import (
    align_pi_to_utils,
    g_move_utilities,
    improvement_weighted_teacher,
    policy_quality,
    teacher_diagnostics,
)
from .jump_heads import jump_pool_diagnostics, logits_to_pi
from .legal_moves import LegalMove, apply_move
from .state import JointAssignmentState


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def reverse_categorical_jump_nll(
    *,
    model,
    chemgraph_t,
    traj: CTMCTrajectory,
    t_geom: torch.Tensor,
    t_start: float,
    t_end: float,
    start_state: JointAssignmentState | None = None,
    kinds: tuple[str, ...] = ("R", "G"),
    clip: float = 8.0,
    h_segment: dict[str, float] | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """Event-mean legal CE on reverse events in (t_start, t_end] only.

        L_a = (1/N_a) Σ_{e ∈ E_{s:t} ∩ a} -log π^a_{m_e}     (0 if N_a=0)
        L_a^uniform = mean_e log |M_a|
        ΔL_a = L_a - L_a^uniform

    Hard constraint: if H_a^{segment}=0 then N_a=0 and L_a=0.
    Reverse walk starts at A_t (not A_1) and only undoes E_{s:t}.
    """
    device = t_geom.device
    zero = torch.zeros((), device=device)
    empty = {
        "L_R": zero,
        "L_G": zero.clone(),
        "L_A": zero.clone(),
        "L_R_uniform": 0.0,
        "L_G_uniform": 0.0,
        "delta_L_R": 0.0,
        "delta_L_G": 0.0,
        "CE_R": 0.0,
        "CE_G": 0.0,
        "uniform_CE_R": 0.0,
        "uniform_CE_G": 0.0,
        "delta_CE_R": 0.0,
        "delta_CE_G": 0.0,
        "n_R_events": 0,
        "n_G_events": 0,
        "n_R_miss": 0,
        "n_G_miss": 0,
        "num_legal_R": 0,
        "num_legal_G": 0,
    }

    # Strict local segment: (s, t]
    seg = traj.events_on_segment(float(t_start), float(t_end))
    h_seg = h_segment or {}
    # Drop kinds whose segment hazard is (numerically) zero
    allowed = set(kinds)
    for kind in ("R", "G"):
        if float(h_seg.get(kind, 1.0)) <= 1e-12:
            allowed.discard(kind)
    if not seg:
        return empty

    st = (start_state or traj.state_at(float(t_end))).clone()
    pen = torch.tensor(20.0, device=device)
    L_sum = {k: zero.clone() for k in ("R", "G")}
    L_uni_sum = {k: 0.0 for k in ("R", "G")}
    n_ev = {k: 0 for k in ("R", "G")}
    n_miss = {k: 0 for k in ("R", "G")}
    last_diag: dict[str, Any] = {}

    for e in reversed(seg):
        kind = e.kind
        score_this = kind in allowed
        if score_this:
            out = model(chemgraph_t, t_geom, st, compute_jumps=True)
            scored = out.move_logits
            pi_pool = logits_to_pi(scored, clip=clip).get(kind, [])
            n_legal = len(pi_pool)
            last_diag.update(
                jump_pool_diagnostics(
                    scored,
                    beta_r=float(out.diagnostics.get("beta_r", 0.0)),
                    beta_g=float(out.diagnostics.get("beta_g", 0.0)),
                    clip=clip,
                )
            )
            last_diag["num_legal_R"] = int(last_diag.get("num_R_moves", 0))
            last_diag["num_legal_G"] = int(last_diag.get("num_G_moves", 0))
            pi_map = {_pair_key(m.i, m.j): p for m, p in pi_pool}
            found = pi_map.get(_pair_key(e.i, e.j))
            n_ev[kind] += 1
            if n_legal > 0:
                L_uni_sum[kind] += math.log(n_legal)
            if found is None:
                n_miss[kind] += 1
                L_sum[kind] = L_sum[kind] + pen
            else:
                L_sum[kind] = L_sum[kind] - torch.log(found.clamp_min(1e-12))
        st = apply_move(st, LegalMove(kind, e.i, e.j))

    L = {}
    L_uni = {}
    for kind in ("R", "G"):
        n = n_ev[kind]
        if n > 0:
            L[kind] = L_sum[kind] / float(n)
            L_uni[kind] = L_uni_sum[kind] / float(n)
        else:
            L[kind] = zero.clone()
            L_uni[kind] = 0.0

    def _delta(kind: str) -> float:
        return float(L[kind].detach()) - float(L_uni[kind])

    return {
        "L_R": L["R"],
        "L_G": L["G"],
        "L_A": L["R"] + L["G"],
        "L_R_uniform": L_uni["R"],
        "L_G_uniform": L_uni["G"],
        "delta_L_R": _delta("R"),
        "delta_L_G": _delta("G"),
        "CE_R": float(L["R"].detach()),
        "CE_G": float(L["G"].detach()),
        "uniform_CE_R": float(L_uni["R"]),
        "uniform_CE_G": float(L_uni["G"]),
        "delta_CE_R": _delta("R"),
        "delta_CE_G": _delta("G"),
        "n_R_events": n_ev["R"],
        "n_G_events": n_ev["G"],
        "n_R_miss": n_miss["R"],
        "n_G_miss": n_miss["G"],
        "num_legal_R": int(last_diag.get("num_legal_R", last_diag.get("num_R_moves", 0))),
        "num_legal_G": int(last_diag.get("num_legal_G", last_diag.get("num_G_moves", 0))),
        **{k: v for k, v in last_diag.items() if not torch.is_tensor(v)},
    }


def event_conditioned_assignment_ce(
    *,
    model,
    chemgraph_t,
    t: torch.Tensor,
    state_after: JointAssignmentState,
    kind: str,
    i: int,
    j: int,
    clip: float = 8.0,
) -> dict[str, torch.Tensor | float | int]:
    """Single reverse-event CE at global time τ: input (X_τ, L_τ, A_{τ+}).

    Target is the inverse swap m^{-1}=(i,j) (same pair as the forward event).
    L = -log π_a(m^{-1}); rates remain r=βπ (unused in this CE).
    """
    t = torch.as_tensor(t, dtype=torch.float32, device=chemgraph_t["pos"].device).reshape(-1)
    out = model(chemgraph_t, t, state_after, compute_jumps=True)
    scored = out.move_logits
    # scores cached on output for optional same-τ geometry loss (single GemNet)
    pi_pool = logits_to_pi(scored, clip=clip).get(kind, [])
    n_legal = len(pi_pool)
    device = t.device
    zero = torch.zeros((), device=device)
    if n_legal == 0:
        return {
            "CE": zero + 20.0,
            "uniform_CE": 0.0,
            "delta_CE": 20.0,
            "target_probability": 0.0,
            "target_rank": -1,
            "top1": 0.0,
            "top5": 0.0,
            "entropy": 0.0,
            "num_legal": 0,
            "hit": 0,
        }
    key = _pair_key(i, j)
    probs = []
    target_p = None
    target_idx = None
    for idx, (m, p) in enumerate(pi_pool):
        probs.append(p)
        if _pair_key(m.i, m.j) == key:
            target_p = p
            target_idx = idx
    stacked = torch.stack(probs)
    order = torch.argsort(stacked, descending=True)
    if target_p is None:
        ce = zero + 20.0
        tprob = 0.0
        rank = n_legal + 1
        hit = 0
    else:
        ce = -torch.log(target_p.clamp_min(1e-12))
        tprob = float(target_p.detach())
        rank = int((order == target_idx).nonzero(as_tuple=True)[0].item()) + 1
        hit = 1
    uni = float(math.log(n_legal))
    ent = float((-(stacked.detach() * stacked.detach().clamp_min(1e-12).log()).sum()))
    return {
        "CE": ce,
        "uniform_CE": uni,
        "delta_CE": float(ce.detach()) - uni,
        "target_probability": tprob,
        "target_rank": rank,
        "top1": 1.0 if rank == 1 else 0.0,
        "top5": 1.0 if 1 <= rank <= 5 else 0.0,
        "entropy": ent,
        "num_legal": n_legal,
        "hit": hit,
        "chemgraph_scores": out.chemgraph_scores,
        "g_copy_context_mode": out.diagnostics.get("g_copy_context_mode", "mean"),
        "slot_embedding_norm_mean": out.diagnostics.get("slot_embedding_norm_mean", 0.0),
        "slot_embedding_norm_std": out.diagnostics.get("slot_embedding_norm_std", 0.0),
        "slot_pair_feature_norm": out.diagnostics.get("slot_pair_feature_norm", 0.0),
        "slot_orbit_pairwise_var": out.diagnostics.get("slot_orbit_pairwise_var", 0.0),
    }


def event_conditioned_g_teacher_ce(
    *,
    model,
    chemgraph_t,
    t: torch.Tensor,
    state_after: JointAssignmentState,
    c0: torch.Tensor,
    copy0: torch.Tensor,
    hist_i: int,
    hist_j: int,
    temperature: float,
    clip: float = 8.0,
) -> dict[str, Any]:
    """G soft-target CE at (X_τ, L_τ, A_{τ+}) with improvement-weighted teacher.

    L_G = -Σ_m q_m log π_G(m); rates still r=βπ (not trained as total hazard).
    """
    t = torch.as_tensor(t, dtype=torch.float32, device=chemgraph_t["pos"].device).reshape(-1)
    out = model(chemgraph_t, t, state_after, compute_jumps=True)
    utils = g_move_utilities(state_after, c0=c0, copy0=copy0)
    teacher = improvement_weighted_teacher(utils, temperature=temperature)
    pi_pool = logits_to_pi(out.move_logits, clip=clip).get("G", [])
    pi = align_pi_to_utils(pi_pool, utils)
    device = t.device
    zero = torch.zeros((), device=device)
    if pi.numel() == 0 or teacher["q"].numel() == 0:
        ce = zero + 20.0
        q_dev = teacher["q"]
    else:
        q_dev = teacher["q"].to(device=pi.device, dtype=pi.dtype)
        ce = -(q_dev * pi.clamp_min(1e-12).log()).sum()
    tdiag = teacher_diagnostics(utils=utils, teacher=teacher, hist_key=_pair_key(hist_i, hist_j))
    pqual = policy_quality(utils=utils, teacher=teacher, pi=pi if pi.numel() else zero)
    return {
        "CE": ce,
        "uniform_CE": pqual["uniform_CE_teacher"],
        "delta_CE": pqual["delta_CE_teacher"],
        "chemgraph_scores": out.chemgraph_scores,
        "g_copy_context_mode": out.diagnostics.get("g_copy_context_mode", "mean"),
        "slot_embedding_norm_mean": out.diagnostics.get("slot_embedding_norm_mean", 0.0),
        "slot_embedding_norm_std": out.diagnostics.get("slot_embedding_norm_std", 0.0),
        "slot_pair_feature_norm": out.diagnostics.get("slot_pair_feature_norm", 0.0),
        "slot_orbit_pairwise_var": out.diagnostics.get("slot_orbit_pairwise_var", 0.0),
        **tdiag,
        **pqual,
    }


def geometry_step_loss(
    *,
    model,
    loss_fn,
    corruption,
    clean_cg,
    noisy_cg,
    t: torch.Tensor,
    state_at_t: JointAssignmentState,
) -> dict[str, torch.Tensor]:
    """MatterGen geometry loss at a single global t with A_t."""
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1)
    if noisy_cg["pos"].device.type != "cpu":
        t = t.to(device=noisy_cg["pos"].device)
    geom_out = model(noisy_cg, t, state_at_t, compute_jumps=False)
    L_geom, metrics = mattergen_geometry_loss(
        loss_fn=loss_fn,
        corruption=corruption,
        clean_batch=clean_cg,
        noisy_batch=noisy_cg,
        score_model_output=geom_out.chemgraph_scores,
        t=t,
    )
    out: dict[str, torch.Tensor] = {"L_geom": L_geom, "loss": L_geom}
    for k, v in metrics.items():
        out[f"geom_{k}"] = torch.tensor(v, device=L_geom.device) if not torch.is_tensor(v) else v
    return out


def joint_training_step_losses(
    *,
    model,
    loss_fn,
    corruption,
    clean_cg,
    noisy_cg,
    t: torch.Tensor,
    t_s: float,
    state_at_t: JointAssignmentState,
    traj: CTMCTrajectory,
    lambda_r: float = 1.0,
    lambda_g: float = 1.0,
    h_r_segment: float = 0.0,
    h_g_segment: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Joint sample S_t=(A_t,X_t,L_t); assignment CE only on E_{s:t}.

    Reverse targets are events with s < τ_e ≤ t, walked from A_t toward A_s.
    """
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1)
    if noisy_cg["pos"].device.type != "cpu":
        t = t.to(device=noisy_cg["pos"].device)
    t_end = float(t[0].item())

    geom_out = model(noisy_cg, t, state_at_t, compute_jumps=False)
    L_geom, metrics = mattergen_geometry_loss(
        loss_fn=loss_fn,
        corruption=corruption,
        clean_batch=clean_cg,
        noisy_batch=noisy_cg,
        score_model_output=geom_out.chemgraph_scores,
        t=t,
    )

    device = L_geom.device
    nll = reverse_categorical_jump_nll(
        model=model,
        chemgraph_t=noisy_cg,
        traj=traj,
        t_geom=t,
        t_start=float(t_s),
        t_end=t_end,
        start_state=state_at_t,
        kinds=("R", "G"),
        h_segment={"R": float(h_r_segment), "G": float(h_g_segment)},
    )
    L_R = nll["L_R"]
    L_G = nll["L_G"]
    total = L_geom + lambda_r * L_R + lambda_g * L_G

    out: dict[str, torch.Tensor] = {
        "loss": total,
        "L_geom": L_geom,
        "L_R": L_R,
        "L_G": L_G,
        "CE_R": L_R,
        "CE_G": L_G,
        "L_R_uniform": torch.tensor(float(nll["L_R_uniform"]), device=device),
        "L_G_uniform": torch.tensor(float(nll["L_G_uniform"]), device=device),
        "uniform_CE_R": torch.tensor(float(nll["uniform_CE_R"]), device=device),
        "uniform_CE_G": torch.tensor(float(nll["uniform_CE_G"]), device=device),
        "delta_L_R": torch.tensor(float(nll["delta_L_R"]), device=device),
        "delta_L_G": torch.tensor(float(nll["delta_L_G"]), device=device),
        "delta_CE_R": torch.tensor(float(nll["delta_CE_R"]), device=device),
        "delta_CE_G": torch.tensor(float(nll["delta_CE_G"]), device=device),
        "n_R_events": torch.tensor(float(nll["n_R_events"]), device=device),
        "n_G_events": torch.tensor(float(nll["n_G_events"]), device=device),
        "num_legal_R": torch.tensor(float(nll.get("num_legal_R", 0)), device=device),
        "num_legal_G": torch.tensor(float(nll.get("num_legal_G", 0)), device=device),
    }
    for k, v in metrics.items():
        out[f"geom_{k}"] = torch.tensor(v, device=device) if not torch.is_tensor(v) else v
    for key in (
        "logit_mean_R",
        "logit_mean_G",
        "logit_std_R",
        "logit_std_G",
        "entropy_R",
        "entropy_G",
    ):
        if key in nll and not torch.is_tensor(nll[key]):
            out[key] = torch.tensor(float(nll[key]), device=device)
    return out
