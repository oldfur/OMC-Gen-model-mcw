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


_G_REPR_KEYS = (
    "g_copy_context_mode",
    "g_relation_detach_trunk",
    "slot_embedding_norm_mean",
    "slot_embedding_norm_std",
    "slot_pair_feature_norm",
    "slot_orbit_pairwise_var",
    "candidate_copy_geom_norm_mean",
    "candidate_copy_geom_norm_std",
    "candidate_copy_relation_norm_mean",
    "candidate_copy_relation_norm_std",
    "candidate_copy_relation_variance_across_copies",
    "candidate_copy_relation_variance_across_candidates",
    "current_vs_cross_relation_distance",
    "g_logit_std_across_legal_moves",
    "template_relation_norm",
    "compatibility_S_mean",
    "compatibility_S_std",
    "delta_S_mean",
    "delta_S_std",
    "abs_delta_S_mean",
    "counterfactual_feature_norm",
    "counterfactual_feature_var",
)


def _g_repr_fields(out) -> dict[str, Any]:
    diag = getattr(out, "diagnostics", {}) or {}
    fields: dict[str, Any] = {}
    for k in _G_REPR_KEYS:
        v = diag.get(k, 0.0 if k != "g_copy_context_mode" else "mean")
        if k == "g_relation_detach_trunk":
            fields[k] = float(bool(v))
        else:
            fields[k] = v
    return fields


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based average ranks (tied values share the mean rank)."""
    n = int(x.numel())
    ranks = torch.zeros(n, dtype=torch.float32)
    if n == 0:
        return ranks
    order = torch.argsort(x, stable=True)
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and float(xs[j]) == float(xs[i]):
            j += 1
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_tied(x: torch.Tensor, y: torch.Tensor) -> float:
    """Tied-rank Spearman correlation; 0 if undefined."""
    if x.numel() < 2 or y.numel() != x.numel():
        return 0.0
    rx = _average_ranks(x.detach().float().reshape(-1).cpu())
    ry = _average_ranks(y.detach().float().reshape(-1).cpu())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(rx.norm() * ry.norm())
    if den < 1e-12:
        return 0.0
    return float((rx * ry).sum() / den)


def isolation_grad_norms(model, ce_g: torch.Tensor) -> dict[str, float]:
    """Grad norms of L_G w.r.t. G-specific / shared trunk / R-head."""
    zeros = {
        "grad_norm_G_specific_from_LG": 0.0,
        "grad_norm_shared_trunk_from_LG": 0.0,
        "grad_norm_R_head_from_LG": 0.0,
    }
    if not (torch.is_tensor(ce_g) and ce_g.requires_grad and ce_g.grad_fn is not None):
        return zeros
    groups = {
        "grad_norm_G_specific_from_LG": [p for p in model.g_specific_parameters() if p.requires_grad],
        "grad_norm_shared_trunk_from_LG": [p for p in model.shared_trunk_parameters() if p.requires_grad],
        "grad_norm_R_head_from_LG": [p for p in model.r_head.parameters() if p.requires_grad],
    }
    all_p = groups["grad_norm_G_specific_from_LG"] + groups["grad_norm_shared_trunk_from_LG"] + groups[
        "grad_norm_R_head_from_LG"
    ]
    if not all_p:
        return zeros
    grads = torch.autograd.grad(ce_g, all_p, retain_graph=True, allow_unused=True)
    out = {}
    i = 0
    for name, params in groups.items():
        acc = 0.0
        for p in params:
            g = grads[i]
            i += 1
            if g is not None:
                acc += float(g.detach().float().pow(2).sum())
        out[name] = acc ** 0.5
    return out


def flatten_param_grads(params, grads) -> torch.Tensor:
    """Concatenate per-parameter grads; unused → zeros of the same numel."""
    chunks = []
    for p, g in zip(params, grads):
        if g is None:
            chunks.append(torch.zeros(p.numel(), dtype=torch.float32))
        else:
            chunks.append(g.detach().float().reshape(-1).cpu())
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks)


def trunk_rg_interference_metrics(
    g_r: torch.Tensor,
    g_g: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """cos_RG and D_G/R on flattened shared-trunk gradients."""
    if g_r.numel() == 0 or g_g.numel() == 0:
        return {
            "cos_RG": 0.0,
            "D_G_over_R": 0.0,
            "norm_g_R": 0.0,
            "norm_g_G": 0.0,
            "destructive_dominant": 0.0,
        }
    nr = float(g_r.norm())
    ng = float(g_g.norm())
    cos = float((g_r * g_g).sum() / (nr * ng + eps))
    dgr = ng / (nr + eps)
    return {
        "cos_RG": cos,
        "D_G_over_R": dgr,
        "norm_g_R": nr,
        "norm_g_G": ng,
        "destructive_dominant": 1.0 if (cos < 0.0 and dgr > 1.0) else 0.0,
    }


def _snapshot_named_buffers(model) -> dict[str, torch.Tensor]:
    return {n: b.detach().clone() for n, b in model.named_buffers()}


def _restore_named_buffers(model, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, b in model.named_buffers():
            if n in snap:
                b.copy_(snap[n])


def coupled_trunk_interference_audit(
    *,
    model,
    ce_g: torch.Tensor,
    probe: dict[str, Any],
    param_lrs: dict[int, float],
    eps: float = 1e-12,
) -> dict[str, float]:
    """Shared-trunk R vs G interference on a fixed R probe.

    Does not write ``.grad``, does not step the optimizer, and restores
    parameter data + buffers so the training trajectory is unchanged.
    ``Δ_G L_R`` is an isolated SGD step ``θ_s ← θ_s − lr ⊙ g_G`` (no Adam
    moments, no L_geom) so damage is attributable to L_G coupling.
    """
    empty = {
        "cos_RG": 0.0,
        "D_G_over_R": 0.0,
        "norm_g_R": 0.0,
        "norm_g_G": 0.0,
        "destructive_dominant": 0.0,
        "delta_G_LR": 0.0,
        "delta_G_LR_linear": 0.0,
        "L_R_probe_pre": 0.0,
        "L_R_probe_post": 0.0,
    }
    if not (torch.is_tensor(ce_g) and ce_g.requires_grad and ce_g.grad_fn is not None):
        return empty
    if not probe:
        return empty
    trunk = [p for p in model.shared_trunk_parameters() if p.requires_grad]
    if not trunk:
        return empty

    g_g_list = torch.autograd.grad(ce_g, trunk, retain_graph=True, allow_unused=True)
    g_g = flatten_param_grads(trunk, g_g_list)

    was_training = model.training
    buf_snap = _snapshot_named_buffers(model)
    backups = [p.detach().clone() for p in trunk]
    try:
        model.eval()
        r_pre = event_conditioned_assignment_ce(
            model=model,
            chemgraph_t=probe["chemgraph_t"],
            t=probe["t"],
            state_after=probe["state"],
            kind="R",
            i=int(probe["i"]),
            j=int(probe["j"]),
        )
        ce_r = r_pre["CE"]
        l_pre = float(ce_r.detach())
        if torch.is_tensor(ce_r) and ce_r.requires_grad and ce_r.grad_fn is not None:
            g_r_list = torch.autograd.grad(ce_r, trunk, retain_graph=False, allow_unused=True)
        else:
            g_r_list = [None] * len(trunk)
        g_r = flatten_param_grads(trunk, g_r_list)
        mets = trunk_rg_interference_metrics(g_r, g_g, eps=eps)

        # first-order: ⟨g_R, −lr ⊙ g_G⟩
        lin = 0.0
        with torch.no_grad():
            for p, gr, gg in zip(trunk, g_r_list, g_g_list):
                lr = float(param_lrs.get(id(p), 0.0))
                if gr is None or gg is None or lr == 0.0:
                    continue
                lin += float((-lr) * (gr.detach().float() * gg.detach().float()).sum())

        with torch.no_grad():
            for p, gg in zip(trunk, g_g_list):
                lr = float(param_lrs.get(id(p), 0.0))
                if gg is None or lr == 0.0:
                    continue
                p.add_(gg.detach(), alpha=-lr)

        r_post = event_conditioned_assignment_ce(
            model=model,
            chemgraph_t=probe["chemgraph_t"],
            t=probe["t"],
            state_after=probe["state"],
            kind="R",
            i=int(probe["i"]),
            j=int(probe["j"]),
        )
        l_post = float(r_post["CE"].detach())
        mets.update(
            {
                "delta_G_LR": l_post - l_pre,
                "delta_G_LR_linear": lin,
                "L_R_probe_pre": l_pre,
                "L_R_probe_post": l_post,
            }
        )
        return mets
    finally:
        with torch.no_grad():
            for p, b in zip(trunk, backups):
                p.copy_(b)
        _restore_named_buffers(model, buf_snap)
        model.train(was_training)


def aggregate_r_forgetting(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Orbit-invariant R CE / top1 forgetting: final − best (loss convention)."""
    rrecs = [r for r in records if r.get("kind") == "R"]
    if not rrecs:
        return {"n": 0, "best_delta_CE_R": 0.0, "final_delta_CE_R": 0.0, "forgetting_delta_CE_R": 0.0}

    def _win(recs: list[dict], k: str, default: float = 0.0) -> float:
        xs = [float(r.get(k, default)) for r in recs if r.get(k) is not None]
        return float(sum(xs) / len(xs)) if xs else default

    dces = [float(r.get("delta_CE", 0.0)) for r in rrecs]
    top1s = [float(r.get("top1", 0.0)) for r in rrecs]
    best_dce = min(dces)
    best_top1 = max(top1s)
    # final = last 100 R events (or all if fewer)
    tail = rrecs[-100:] if len(rrecs) >= 20 else rrecs
    final_dce = _win(tail, "delta_CE")
    final_top1 = _win(tail, "top1")
    # running-best prefix then last-window
    run_best = dces[0]
    run_bests = []
    for v in dces:
        run_best = min(run_best, v)
        run_bests.append(run_best)
    return {
        "n": len(rrecs),
        "n_final_window": len(tail),
        "best_delta_CE_R": best_dce,
        "final_delta_CE_R": final_dce,
        "forgetting_delta_CE_R": final_dce - best_dce,
        "best_top1_R": best_top1,
        "final_top1_R": final_top1,
        "forgetting_top1_R": best_top1 - final_top1,
        "overall_delta_CE_R": _win(rrecs, "delta_CE"),
        "overall_top1_R": _win(rrecs, "top1"),
        "last100_delta_CE_R": final_dce,
        "last100_top1_R": final_top1,
    }


def summarize_interference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-G-step interference rows (early <200, late ≥800)."""
    if not rows:
        return {"n": 0}

    def _agg(recs: list[dict]) -> dict[str, Any]:
        if not recs:
            return {"n": 0}
        cos = [float(r["cos_RG"]) for r in recs]
        dgr = [float(r["D_G_over_R"]) for r in recs]
        dmg = [float(r["delta_G_LR"]) for r in recs]
        cos_s = sorted(cos)
        dmg_s = sorted(dmg)
        mid = len(cos_s) // 2
        med_cos = cos_s[mid] if len(cos_s) % 2 == 1 else 0.5 * (cos_s[mid - 1] + cos_s[mid])
        mid_d = len(dmg_s) // 2
        med_dmg = dmg_s[mid_d] if len(dmg_s) % 2 == 1 else 0.5 * (dmg_s[mid_d - 1] + dmg_s[mid_d])
        n = float(len(recs))
        return {
            "n": len(recs),
            "cos_RG_mean": sum(cos) / n,
            "cos_RG_median": med_cos,
            "cos_RG_negative_fraction": sum(1.0 for c in cos if c < 0.0) / n,
            "cos_RG_lt_m0_2_fraction": sum(1.0 for c in cos if c < -0.2) / n,
            "D_G_over_R_mean": sum(dgr) / n,
            "D_G_over_R_median": sorted(dgr)[len(dgr) // 2],
            "destructive_dominant_fraction": sum(float(r.get("destructive_dominant", 0.0)) for r in recs) / n,
            "delta_G_LR_mean": sum(dmg) / n,
            "delta_G_LR_median": med_dmg,
            "delta_G_LR_positive_fraction": sum(1.0 for d in dmg if d > 0.0) / n,
            "cumulative_delta_G_LR": sum(dmg),
            "norm_g_G_mean": sum(float(r.get("norm_g_G", 0.0)) for r in recs) / n,
            "norm_g_R_mean": sum(float(r.get("norm_g_R", 0.0)) for r in recs) / n,
        }

    early = [r for r in rows if int(r.get("step", 0)) < 200]
    late = [r for r in rows if int(r.get("step", 0)) >= 800]
    return {"all": _agg(rows), "early": _agg(early), "late": _agg(late)}


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
        **_g_repr_fields(out),
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
    u = teacher.get("u")
    if u is None:
        u = torch.tensor([float(r["u"]) for r in utils], dtype=torch.float32)
    logit_list = []
    ds_list = []
    ds_vec = out.diagnostics.get("delta_S_vec")
    scored_g = out.move_logits.get("G", []) if out.move_logits else []
    ds_map = {}
    if torch.is_tensor(ds_vec) and ds_vec.numel() == len(scored_g):
        for (m, _), dsv in zip(scored_g, ds_vec):
            ds_map[_pair_key(m.i, m.j)] = dsv
    u_ord = []
    for r in utils:
        key = r["key"]
        u_ord.append(float(r["u"]))
        hit = None
        for m, lg in scored_g:
            if _pair_key(m.i, m.j) == key:
                hit = lg
                break
        logit_list.append(hit if hit is not None else zero)
        if key in ds_map:
            ds_list.append(ds_map[key])
    if logit_list:
        logits_al = torch.stack([lg if torch.is_tensor(lg) else zero for lg in logit_list])
        spear_lg = spearman_tied(logits_al, torch.tensor(u_ord, dtype=torch.float32))
    else:
        spear_lg = 0.0
    spear_ds = 0.0
    ds_ben = 0.0
    ds_harm = 0.0
    if ds_list and len(ds_list) == len(u_ord):
        ds_t = torch.stack([d.reshape(()) if d.ndim else d for d in ds_list]).detach().float().cpu()
        u_t = torch.tensor(u_ord, dtype=torch.float32)
        spear_ds = spearman_tied(ds_t, u_t)
        ben = u_t > 0
        if bool(ben.any()):
            ds_ben = float(ds_t[ben].mean())
        if bool((~ben).any()):
            ds_harm = float(ds_t[~ben].mean())
    return {
        "CE": ce,
        "uniform_CE": pqual["uniform_CE_teacher"],
        "delta_CE": pqual["delta_CE_teacher"],
        "chemgraph_scores": out.chemgraph_scores,
        "spearman_logit_vs_utility": spear_lg,
        "spearman_deltaS_vs_utility": spear_ds,
        "delta_S_beneficial_mean": ds_ben,
        "delta_S_harmful_mean": ds_harm,
        **_g_repr_fields(out),
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
