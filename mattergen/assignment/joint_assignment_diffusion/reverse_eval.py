"""Learned local reverse vs uniform-π control (same A_t, X_t, β, [s,t])."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .g_teacher import f1_vs_c0
from .legal_moves import LegalMove, apply_move
from .metrics import assignment_distances
from .sampler import _gillespie_step_a_integrated
from .state import JointAssignmentState


G_BINS: list[tuple[float, float, str]] = [
    (0.35, 0.40, "[0.35,0.40)"),
    (0.40, 0.45, "[0.40,0.45)"),
    (0.45, 0.50, "[0.45,0.50)"),
    (0.50, 0.55, "[0.50,0.55)"),
    (0.55, 0.60, "[0.55,0.60)"),
    (0.60, 0.65, "[0.60,0.65)"),
    (0.65, 0.70, "[0.65,0.70)"),
    (0.70, 0.751, "[0.70,0.75]"),
]

R_BINS: list[tuple[float, float, str]] = [
    (0.60, 0.65, "[0.60,0.65)"),
    (0.65, 0.70, "[0.65,0.70)"),
    (0.70, 0.75, "[0.70,0.75)"),
    (0.75, 0.80, "[0.75,0.80)"),
    (0.80, 0.85, "[0.80,0.85)"),
    (0.85, 0.90, "[0.85,0.90)"),
    (0.90, 0.951, "[0.90,0.95]"),
]


def bin_label(t: float, bins: list[tuple[float, float, str]]) -> str | None:
    for lo, hi, name in bins:
        if lo <= float(t) < hi:
            return name
    return None


def _run_reverse(
    *,
    model,
    chemgraph,
    state_t: JointAssignmentState,
    t: float,
    s: float,
    generator: torch.Generator,
    policy: str,
):
    st, ev, _d = _gillespie_step_a_integrated(
        model,
        chemgraph,
        t,
        s,
        state_t,
        generator=generator,
        static_A=False,
        policy=policy,
    )
    return st, ev


def _g_jump_c0_stats(state0: JointAssignmentState, events, c0: torch.Tensor) -> dict[str, float]:
    """Clean-directed ΔF1 of executed G jumps along a reverse path."""
    st = state0.clone()
    n_g = 0
    n_imp = 0
    sum_df1 = 0.0
    for e in events:
        if e.kind != "G":
            st = apply_move(st, LegalMove(e.kind, e.i, e.j))
            continue
        f_before = f1_vs_c0(st, c0)
        st = apply_move(st, LegalMove("G", e.i, e.j))
        df1 = f1_vs_c0(st, c0) - f_before
        n_g += 1
        sum_df1 += df1
        if df1 > 0:
            n_imp += 1
    return {
        "n_g_jumps": float(n_g),
        "frac_g_jumps_improving_c0": (n_imp / n_g) if n_g else 0.0,
        "mean_g_jump_delta_f1_c0": (sum_df1 / n_g) if n_g else 0.0,
    }


def evaluate_local_reverse_pair(
    *,
    model,
    chemgraph_t,
    state_t: JointAssignmentState,
    state_s: JointAssignmentState,
    t: float,
    s: float,
    seed: int,
    c0: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Paired learned vs uniform reverse on the same (A_t, X_t, [s,t], seed)."""
    g_learn = torch.Generator().manual_seed(int(seed))
    g_uni = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        hat, ev_hat = _run_reverse(
            model=model, chemgraph=chemgraph_t, state_t=state_t, t=t, s=s,
            generator=g_learn, policy="learned",
        )
        uni, ev_uni = _run_reverse(
            model=model, chemgraph=chemgraph_t, state_t=state_t, t=t, s=s,
            generator=g_uni, policy="uniform",
        )
    d_start = assignment_distances(state_t, state_s)
    d_hat = assignment_distances(hat, state_s)
    d_uni = assignment_distances(uni, state_s)
    row: dict[str, Any] = {
        "t": float(t),
        "s": float(s),
        "seed": int(seed),
        "hat_legal": hat.validate()["legal"],
        "uni_legal": uni.validate()["legal"],
        "start_orbit_acc": d_start["orbit_atom_accuracy"],
        "start_ARI": d_start["ARI"],
        "start_f1": d_start["copy_pair_f1"],
        "start_orbit_exact": d_start["orbit_exact"],
        "start_exact_C": d_start["exact_C"],
        "learned_orbit_acc": d_hat["orbit_atom_accuracy"],
        "learned_orbit_exact": d_hat["orbit_exact"],
        "learned_ARI": d_hat["ARI"],
        "learned_f1": d_hat["copy_pair_f1"],
        "learned_exact_C": d_hat["exact_C"],
        "uniform_orbit_acc": d_uni["orbit_atom_accuracy"],
        "uniform_orbit_exact": d_uni["orbit_exact"],
        "uniform_ARI": d_uni["ARI"],
        "uniform_f1": d_uni["copy_pair_f1"],
        "uniform_exact_C": d_uni["exact_C"],
        "delta_d_orbit_learned": d_start["d_orbit"] - d_hat["d_orbit"],
        "delta_d_orbit_uniform": d_start["d_orbit"] - d_uni["d_orbit"],
        "delta_d_ari_learned": d_start["d_ari"] - d_hat["d_ari"],
        "delta_d_ari_uniform": d_start["d_ari"] - d_uni["d_ari"],
        "delta_d_f1_learned": d_start["d_f1"] - d_hat["d_f1"],
        "delta_d_f1_uniform": d_start["d_f1"] - d_uni["d_f1"],
        "g_bin": bin_label(t, G_BINS),
        "r_bin": bin_label(t, R_BINS),
    }
    row["win_orbit"] = float(row["delta_d_orbit_learned"] > row["delta_d_orbit_uniform"] + 1e-12)
    row["win_ari"] = float(row["delta_d_ari_learned"] > row["delta_d_ari_uniform"] + 1e-12)
    row["win_f1"] = float(row["delta_d_f1_learned"] > row["delta_d_f1_uniform"] + 1e-12)
    if c0 is not None:
        lg = _g_jump_c0_stats(state_t, ev_hat, c0)
        ug = _g_jump_c0_stats(state_t, ev_uni, c0)
        row["learned_frac_g_improving_c0"] = lg["frac_g_jumps_improving_c0"]
        row["uniform_frac_g_improving_c0"] = ug["frac_g_jumps_improving_c0"]
        row["learned_mean_g_jump_delta_f1_c0"] = lg["mean_g_jump_delta_f1_c0"]
        row["uniform_mean_g_jump_delta_f1_c0"] = ug["mean_g_jump_delta_f1_c0"]
        row["learned_n_g_jumps"] = lg["n_g_jumps"]
        row["uniform_n_g_jumps"] = ug["n_g_jumps"]
    return row


def summarize_reverse_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(key: str, subset: list[dict] | None = None) -> float:
        src = subset if subset is not None else rows
        if not src:
            return 0.0
        return float(sum(float(r[key]) for r in src) / len(src))

    out: dict[str, Any] = {
        "n_pairs": len(rows),
        "mean_delta_d_orbit_learned": _mean("delta_d_orbit_learned"),
        "mean_delta_d_orbit_uniform": _mean("delta_d_orbit_uniform"),
        "mean_delta_d_ari_learned": _mean("delta_d_ari_learned"),
        "mean_delta_d_ari_uniform": _mean("delta_d_ari_uniform"),
        "mean_delta_d_f1_learned": _mean("delta_d_f1_learned"),
        "mean_delta_d_f1_uniform": _mean("delta_d_f1_uniform"),
        "win_rate_orbit": _mean("win_orbit"),
        "win_rate_ari": _mean("win_ari"),
        "win_rate_f1": _mean("win_f1"),
        "learned_beats_uniform_orbit": _mean("delta_d_orbit_learned") > _mean("delta_d_orbit_uniform"),
        "learned_beats_uniform_ari": _mean("delta_d_ari_learned") > _mean("delta_d_ari_uniform"),
        "learned_beats_uniform_f1": _mean("delta_d_f1_learned") > _mean("delta_d_f1_uniform"),
        "learned_frac_g_improving_c0": _mean("learned_frac_g_improving_c0") if rows and "learned_frac_g_improving_c0" in rows[0] else 0.0,
        "uniform_frac_g_improving_c0": _mean("uniform_frac_g_improving_c0") if rows and "uniform_frac_g_improving_c0" in rows[0] else 0.0,
        "learned_mean_g_jump_delta_f1_c0": _mean("learned_mean_g_jump_delta_f1_c0") if rows and "learned_mean_g_jump_delta_f1_c0" in rows[0] else 0.0,
        "uniform_mean_g_jump_delta_f1_c0": _mean("uniform_mean_g_jump_delta_f1_c0") if rows and "uniform_mean_g_jump_delta_f1_c0" in rows[0] else 0.0,
        "by_g_bin": {},
        "by_r_bin": {},
    }
    for bins, key, dest in ((G_BINS, "g_bin", "by_g_bin"), (R_BINS, "r_bin", "by_r_bin")):
        for _lo, _hi, name in bins:
            sub = [r for r in rows if r.get(key) == name]
            out[dest][name] = {
                "n": len(sub),
                "win_rate_orbit": _mean("win_orbit", sub),
                "win_rate_ari": _mean("win_ari", sub),
                "win_rate_f1": _mean("win_f1", sub),
                "delta_d_ari_learned": _mean("delta_d_ari_learned", sub),
                "delta_d_ari_uniform": _mean("delta_d_ari_uniform", sub),
                "delta_d_orbit_learned": _mean("delta_d_orbit_learned", sub),
                "delta_d_orbit_uniform": _mean("delta_d_orbit_uniform", sub),
                "learned_orbit_acc": _mean("learned_orbit_acc", sub),
                "uniform_orbit_acc": _mean("uniform_orbit_acc", sub),
                "learned_ARI": _mean("learned_ARI", sub),
                "uniform_ARI": _mean("uniform_ARI", sub),
                "learned_f1": _mean("learned_f1", sub),
                "uniform_f1": _mean("uniform_f1", sub),
            }
    return out


def event_bin_name(kind: str, t: float) -> str:
    bins = G_BINS if kind == "G" else R_BINS
    return bin_label(t, bins) or "other"


def aggregate_g_teacher_bins(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate G teacher-learning metrics into time bins + 0.40-0.50 / 0.50-0.60."""
    g_recs = [r for r in records if r.get("kind") == "G"]

    def _agg(recs: list[dict]) -> dict[str, Any]:
        if not recs:
            return {"n": 0}
        def m(k, default=0.0):
            xs = [float(r.get(k, default)) for r in recs if r.get(k) is not None]
            return float(sum(xs) / len(xs)) if xs else 0.0
        return {
            "n": len(recs),
            "delta_teacher_CE": m("delta_CE_teacher"),
            "P_beneficial": m("P_beneficial"),
            "P_best": m("P_best"),
            "top1_beneficial_rate": m("top1_is_beneficial"),
            "top1_best_rate": m("top1_is_best"),
            "mean_top1_delta_F1": m("top1_delta_F1"),
            "expected_delta_F1_under_pi": m("expected_delta_F1"),
            "uniform_expected_delta_F1": m("uniform_expected_delta_F1"),
            "teacher_entropy": m("teacher_entropy_G"),
            "teacher_effective_support": m("teacher_effective_support_G"),
            "num_beneficial_G": m("num_beneficial_G"),
            "historical_inverse_is_best": m("historical_inverse_is_best"),
            "g_teacher_no_beneficial": m("g_teacher_no_beneficial"),
        }

    out: dict[str, Any] = {"all": _agg(g_recs), "by_bin": {}}
    for _lo, _hi, name in G_BINS:
        out["by_bin"][name] = _agg([r for r in g_recs if r.get("bin") == name])
    out["t_040_050"] = _agg([r for r in g_recs if 0.40 <= float(r.get("tau", -1)) < 0.50])
    out["t_050_060"] = _agg([r for r in g_recs if 0.50 <= float(r.get("tau", -1)) < 0.60])
    return out


def aggregate_event_bins(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-event CE diagnostics into R/G time bins."""
    by: dict[str, list] = defaultdict(list)
    for rec in records:
        kind = rec["kind"]
        name = event_bin_name(kind, rec["tau"])
        by[f"{kind}:{name}"].append(rec)

    def _m(recs, key):
        xs = [float(r[key]) for r in recs if r.get(key) is not None]
        if not xs:
            return 0.0
        return float(sum(xs) / len(xs))

    out: dict[str, Any] = {}
    for key, recs in sorted(by.items()):
        out[key] = {
            "event_count": len(recs),
            "CE": _m(recs, "CE"),
            "uniform_CE": _m(recs, "uniform_CE"),
            "delta_CE": _m(recs, "delta_CE"),
            "target_probability": _m(recs, "target_probability"),
            "target_rank": _m(recs, "target_rank"),
            "top1": _m(recs, "top1"),
            "top5": _m(recs, "top5"),
            "entropy": _m(recs, "entropy"),
        }
    return out
