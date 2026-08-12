"""Joint losses: MatterGen geometry + categorical reverse jump NLL (fixed hazard)."""
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
    """Legal categorical NLL on reverse CTMC events (fixed total hazard).

    Survival ∫λ is parameter-free when Σ r = β(t). Training only fits
    π_m = softmax(ℓ) on observed reverse swaps:

        L_a = - Σ_{e ∈ a} log π^a_{m_e}

    Also reports uniform baseline L_a^uniform = Σ log |M_a| and ΔL = L - L_uniform.
    """
    device = t_geom.device
    rev_events = list(reversed(list(traj.events)))

    zero = torch.zeros((), device=device)
    L = {k: zero.clone() for k in ("R", "G")}
    L_uni = {k: 0.0 for k in ("R", "G")}
    n_ev = {k: 0 for k in ("R", "G")}
    n_miss = {k: 0 for k in ("R", "G")}
    last_diag: dict[str, Any] = {}

    if not rev_events:
        return {
            "L_R": L["R"],
            "L_G": L["G"],
            "L_A": L["R"] + L["G"],
            "L_R_uniform": 0.0,
            "L_G_uniform": 0.0,
            "delta_L_R": 0.0,
            "delta_L_G": 0.0,
            "n_R_events": 0,
            "n_G_events": 0,
            "n_R_miss": 0,
            "n_G_miss": 0,
            **last_diag,
        }

    # Reverse path from end state
    st = traj.state_at(1.0).clone() if traj.times[-1] >= 1.0 - 1e-9 else traj.states[-1].clone()
    # Prefer explicit end: last state after all events
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
            pi_map = {_pair_key(m.i, m.j): p for m, p in pi_pool}
            found = pi_map.get(_pair_key(e.i, e.j))
            n_ev[kind] += 1
            if n_legal > 0:
                L_uni[kind] += math.log(n_legal)
            if found is None:
                n_miss[kind] += 1
                L[kind] = L[kind] + pen
            else:
                L[kind] = L[kind] - torch.log(found.clamp_min(1e-12))
        # Always apply reverse swap so mixed R/G path stays consistent
        st = apply_move(st, LegalMove(kind, e.i, e.j))

    def _delta(kind: str) -> float:
        # L is tensor; detach for scalar report
        return float(L[kind].detach()) - L_uni[kind]

    return {
        "L_R": L["R"],
        "L_G": L["G"],
        "L_A": L["R"] + L["G"],
        "L_R_uniform": L_uni["R"],
        "L_G_uniform": L_uni["G"],
        "delta_L_R": _delta("R"),
        "delta_L_G": _delta("G"),
        "n_R_events": n_ev["R"],
        "n_G_events": n_ev["G"],
        "n_R_miss": n_miss["R"],
        "n_G_miss": n_miss["G"],
        **{k: v for k, v in last_diag.items() if not torch.is_tensor(v)},
    }


def assignment_kind_nll(
    *,
    model,
    chemgraph,
    t_geom: torch.Tensor,
    traj: CTMCTrajectory,
    kind: str,
    clip: float = 8.0,
) -> dict[str, torch.Tensor | float | int]:
    """Single-kind reverse categorical NLL (active-window training call)."""
    full = reverse_categorical_jump_nll(
        model=model,
        chemgraph_t=chemgraph,
        traj=traj,
        t_geom=t_geom,
        kinds=(kind,),
        clip=clip,
    )
    # zero out the other kind's contribution in returned L_* tensors already only kind events
    return full


def joint_training_step_losses(
    *,
    model,
    loss_fn,
    corruption,
    clean_cg,
    noisy_cg_geom,
    t_geom: torch.Tensor,
    state_for_geometry: JointAssignmentState,
    traj: CTMCTrajectory,
    noisy_cg_r=None,
    t_r: torch.Tensor | None = None,
    noisy_cg_g=None,
    t_g: torch.Tensor | None = None,
    lambda_r: float = 1.0,
    lambda_g: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Geometry at t_X ~ U; assignment categorical NLL at active t_R / t_G.

    Survival hazard is constant w.r.t. parameters (Σ r = β). Only π is trained.
    """
    t_geom = torch.as_tensor(t_geom, dtype=torch.float32).reshape(-1)
    if noisy_cg_geom["pos"].device.type != "cpu":
        t_geom = t_geom.to(device=noisy_cg_geom["pos"].device)

    geom_out = model(noisy_cg_geom, t_geom, state_for_geometry, compute_jumps=False)
    L_geom, metrics = mattergen_geometry_loss(
        loss_fn=loss_fn,
        corruption=corruption,
        clean_batch=clean_cg,
        noisy_batch=noisy_cg_geom,
        score_model_output=geom_out.chemgraph_scores,
        t=t_geom,
    )

    device = L_geom.device
    # R head: geometry at t_R (active), reverse categorical on R events
    if noisy_cg_r is None or t_r is None:
        nll_r = reverse_categorical_jump_nll(
            model=model, chemgraph_t=noisy_cg_geom, traj=traj, t_geom=t_geom, kinds=("R",)
        )
    else:
        t_r = torch.as_tensor(t_r, dtype=torch.float32, device=device).reshape(-1)
        nll_r = reverse_categorical_jump_nll(
            model=model, chemgraph_t=noisy_cg_r, traj=traj, t_geom=t_r, kinds=("R",)
        )

    if noisy_cg_g is None or t_g is None:
        nll_g = reverse_categorical_jump_nll(
            model=model, chemgraph_t=noisy_cg_geom, traj=traj, t_geom=t_geom, kinds=("G",)
        )
    else:
        t_g = torch.as_tensor(t_g, dtype=torch.float32, device=device).reshape(-1)
        nll_g = reverse_categorical_jump_nll(
            model=model, chemgraph_t=noisy_cg_g, traj=traj, t_geom=t_g, kinds=("G",)
        )

    L_R = nll_r["L_R"]
    L_G = nll_g["L_G"]
    total = L_geom + lambda_r * L_R + lambda_g * L_G

    out: dict[str, torch.Tensor] = {
        "loss": total,
        "L_geom": L_geom,
        "L_R": L_R,
        "L_G": L_G,
        "L_R_uniform": torch.tensor(float(nll_r["L_R_uniform"]), device=device),
        "L_G_uniform": torch.tensor(float(nll_g["L_G_uniform"]), device=device),
        "delta_L_R": torch.tensor(float(nll_r["delta_L_R"]), device=device),
        "delta_L_G": torch.tensor(float(nll_g["delta_L_G"]), device=device),
        "n_R_events": torch.tensor(float(nll_r["n_R_events"]), device=device),
        "n_G_events": torch.tensor(float(nll_g["n_G_events"]), device=device),
    }
    for k, v in metrics.items():
        out[f"geom_{k}"] = torch.tensor(v, device=device) if not torch.is_tensor(v) else v
    # optional last-event diagnostics (scalars)
    for src, prefix in ((nll_r, "r"), (nll_g, "g")):
        for key in (
            f"num_R_moves",
            f"num_G_moves",
            f"logit_mean_R",
            f"logit_mean_G",
            f"logit_std_R",
            f"logit_std_G",
            f"entropy_R",
            f"entropy_G",
        ):
            if key in src and not torch.is_tensor(src[key]):
                out[f"diag_{key}"] = torch.tensor(float(src[key]), device=device)
    return out
