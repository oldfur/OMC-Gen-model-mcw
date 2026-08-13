"""Joint losses: MatterGen geometry + event-mean categorical reverse jump NLL."""
from __future__ import annotations

import math
from typing import Any

import torch

from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss

from .ctmc import CTMCTrajectory
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
    kinds: tuple[str, ...] = ("R", "G"),
    clip: float = 8.0,
) -> dict[str, torch.Tensor | float | int]:
    """Event-mean legal categorical NLL on reverse CTMC events.

    With fixed total hazard Σ r = β(t), training only fits π:

        L_a = (1/N_a) Σ_e -log π^a_{m_e}     (0 if N_a=0)
        L_a^uniform = mean_e log |M_a|
        ΔL_a = L_a - L_a^uniform
    """
    device = t_geom.device
    rev_events = list(reversed(list(traj.events)))

    zero = torch.zeros((), device=device)
    L_sum = {k: zero.clone() for k in ("R", "G")}
    L_uni_sum = {k: 0.0 for k in ("R", "G")}
    n_ev = {k: 0 for k in ("R", "G")}
    n_miss = {k: 0 for k in ("R", "G")}
    last_diag: dict[str, Any] = {}

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
    if not rev_events:
        return empty

    st = traj.states[-1].clone()
    pen = torch.tensor(20.0, device=device)

    for e in rev_events:
        kind = e.kind
        if kind in kinds:
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
        # aliases for remote logs
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


def joint_training_step_losses(
    *,
    model,
    loss_fn,
    corruption,
    clean_cg,
    noisy_cg,
    t: torch.Tensor,
    state_at_t: JointAssignmentState,
    traj: CTMCTrajectory,
    lambda_r: float = 1.0,
    lambda_g: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Single global time t: joint state (X_t, L_t, A_t).

    Geometry and assignment heads share the same (noisy_cg, t, A_t).
    Async mobility is only via β_R(t), β_G(t) inside the CTMC / schedule.
    """
    t = torch.as_tensor(t, dtype=torch.float32).reshape(-1)
    if noisy_cg["pos"].device.type != "cpu":
        t = t.to(device=noisy_cg["pos"].device)

    # Geometry conditioned on A_t at the same global t
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
    # Single joint reverse-path categorical NLL for both R and G at same (X_t,t)
    nll = reverse_categorical_jump_nll(
        model=model,
        chemgraph_t=noisy_cg,
        traj=traj,
        t_geom=t,
        kinds=("R", "G"),
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
