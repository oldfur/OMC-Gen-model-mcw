#!/usr/bin/env python3
"""G-event target ambiguity audit (no training, no GemNet).

Question: is the single historical inverse-swap CE a reasonable G supervision
target, relative to clean/gauge-invariant copy partition C_0?

Does not import JointAXLModel / denoiser. Does not modify CTMC / legal moves.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.global_copy_assembly.metrics import pair_partition_metrics
from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.global_copy_assembly.orbit_metrics import adjusted_rand_index
from mattergen.assignment.joint_assignment_diffusion.ctmc import simulate_forward_ctmc
from mattergen.assignment.joint_assignment_diffusion.legal_moves import (
    LegalMove,
    apply_move,
    enumerate_g_moves,
)
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.state import (
    JointAssignmentState,
    a_from_role_and_copy,
)
from mattergen.assignment.joint_assignment_diffusion.symmetry import (
    apply_symmetry_to_state,
    sample_symmetry_augment,
)


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

EPS_F1 = 1e-4
EPS_ARI = 1e-4
NEAR_F1 = 0.01
NEAR_ARI = 0.02


def _bin_name(t: float) -> str:
    for lo, hi, name in G_BINS:
        if lo <= float(t) < hi:
            return name
    return "other"


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _offdiag_hamming(C: torch.Tensor, C0: torch.Tensor) -> float:
    n = C.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=C.device)
    return float((C.bool()[off] != C0.bool()[off]).float().mean())


def _scores(C: torch.Tensor, C0: torch.Tensor, copy: torch.Tensor, copy0: torch.Tensor) -> dict[str, float | bool]:
    pair = pair_partition_metrics(C, C0)
    return {
        "f1": float(pair["copy_pair_f1"]),
        "exact_C": bool(pair["exact_C"]),
        "ari": float(adjusted_rand_index(copy0, copy)),
        "hamming": _offdiag_hamming(C, C0),
    }


def _audit_one_event(
    *,
    a_plus: JointAssignmentState,
    a_minus: JointAssignmentState,
    event_i: int,
    event_j: int,
    c0: torch.Tensor,
    copy0: torch.Tensor,
    tau: float,
    seed: int,
) -> dict:
    hist_key = _pair_key(event_i, event_j)
    # sanity: historical inverse reconstructs A_{τ-}
    recon = apply_move(a_plus, LegalMove("G", event_i, event_j))
    exact_recon = bool(torch.equal(recon.A, a_minus.A))

    c_plus = a_plus.C()
    base = _scores(c_plus, c0, a_plus.copy_of(), copy0)
    moves = enumerate_g_moves(a_plus)

    rows = []
    hist_idx = None
    for k, m in enumerate(moves):
        am = apply_move(a_plus, m)
        sc = _scores(am.C(), c0, am.copy_of(), copy0)
        d_f1 = float(sc["f1"] - base["f1"])
        d_ari = float(sc["ari"] - base["ari"])
        d_ham = float(base["hamming"] - sc["hamming"])  # positive = fewer disagreements
        rec = {
            "i": m.i,
            "j": m.j,
            "delta_f1": d_f1,
            "delta_ari": d_ari,
            "delta_hamming": d_ham,
            "f1": sc["f1"],
            "ari": sc["ari"],
            "exact_C": sc["exact_C"],
            "is_historical": _pair_key(m.i, m.j) == hist_key,
        }
        if rec["is_historical"]:
            hist_idx = k
        rows.append(rec)

    n = len(rows)
    if n == 0:
        return {
            "seed": seed,
            "tau": tau,
            "bin": _bin_name(tau),
            "num_legal_G": 0,
            "exact_recon_A_minus": exact_recon,
            "base_f1": base["f1"],
            "base_ari": base["ari"],
            "base_exact_C": base["exact_C"],
        }

    best_f1 = max(r["delta_f1"] for r in rows)
    best_ari = max(r["delta_ari"] for r in rows)
    n_ben_f1 = sum(1 for r in rows if r["delta_f1"] > EPS_F1)
    n_ben_ari = sum(1 for r in rows if r["delta_ari"] > EPS_ARI)
    n_best_f1 = sum(1 for r in rows if r["delta_f1"] >= best_f1 - EPS_F1)
    n_best_ari = sum(1 for r in rows if r["delta_ari"] >= best_ari - EPS_ARI)
    n_near_f1 = sum(1 for r in rows if r["delta_f1"] >= best_f1 - NEAR_F1)
    n_near_ari = sum(1 for r in rows if r["delta_ari"] >= best_ari - NEAR_ARI)

    def _rank(key: str) -> int:
        # rank 1 = largest delta; ties take min rank
        order = sorted(range(n), key=lambda i: -rows[i][key])
        if hist_idx is None:
            return n + 1
        return int(order.index(hist_idx)) + 1

    hist = rows[hist_idx] if hist_idx is not None else None
    hist_best_f1 = bool(hist and hist["delta_f1"] >= best_f1 - EPS_F1)
    hist_best_ari = bool(hist and hist["delta_ari"] >= best_ari - EPS_ARI)
    hist_near_f1 = bool(hist and hist["delta_f1"] >= best_f1 - NEAR_F1)
    hist_near_ari = bool(hist and hist["delta_ari"] >= best_ari - NEAR_ARI)
    hist_ben_f1 = bool(hist and hist["delta_f1"] > EPS_F1)
    hist_ben_ari = bool(hist and hist["delta_ari"] > EPS_ARI)

    return {
        "seed": seed,
        "tau": tau,
        "bin": _bin_name(tau),
        "event_i": event_i,
        "event_j": event_j,
        "num_legal_G": n,
        "exact_recon_A_minus": exact_recon,
        "base_f1": base["f1"],
        "base_ari": base["ari"],
        "base_exact_C": bool(base["exact_C"]),
        "base_hamming": base["hamming"],
        "num_beneficial_F1": n_ben_f1,
        "num_beneficial_ARI": n_ben_ari,
        "num_best_F1": n_best_f1,
        "num_best_ARI": n_best_ari,
        "num_near_best_F1": n_near_f1,
        "num_near_best_ARI": n_near_ari,
        "historical_found": hist is not None,
        "historical_inverse_rank_F1": _rank("delta_f1") if hist else None,
        "historical_inverse_rank_ARI": _rank("delta_ari") if hist else None,
        "historical_inverse_is_beneficial_F1": hist_ben_f1,
        "historical_inverse_is_beneficial_ARI": hist_ben_ari,
        "historical_inverse_is_beneficial": hist_ben_f1 or hist_ben_ari,
        "historical_inverse_is_best_F1": hist_best_f1,
        "historical_inverse_is_best_ARI": hist_best_ari,
        "historical_in_near_best_F1": hist_near_f1,
        "historical_in_near_best_ARI": hist_near_ari,
        "best_delta_F1": best_f1,
        "historical_delta_F1": None if hist is None else hist["delta_f1"],
        "best_delta_ARI": best_ari,
        "historical_delta_ARI": None if hist is None else hist["delta_ari"],
        "historical_delta_hamming": None if hist is None else hist["delta_hamming"],
    }


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _rate(flags: list[bool]) -> float:
    return float(sum(1 for x in flags if x) / len(flags)) if flags else 0.0


def _aggregate(events: list[dict]) -> dict:
    def take(key):
        return [e[key] for e in events if e.get(key) is not None]

    out = {
        "n_events": len(events),
        "mean_num_legal_G": _mean(take("num_legal_G")),
        "mean_num_beneficial_F1": _mean(take("num_beneficial_F1")),
        "mean_num_beneficial_ARI": _mean(take("num_beneficial_ARI")),
        "mean_num_best_F1": _mean(take("num_best_F1")),
        "mean_num_best_ARI": _mean(take("num_best_ARI")),
        "mean_num_near_best_F1": _mean(take("num_near_best_F1")),
        "mean_num_near_best_ARI": _mean(take("num_near_best_ARI")),
        "P_any_beneficial_F1": _rate([e.get("num_beneficial_F1", 0) > 0 for e in events]),
        "P_any_beneficial_ARI": _rate([e.get("num_beneficial_ARI", 0) > 0 for e in events]),
        "P_historical_best_F1": _rate([bool(e.get("historical_inverse_is_best_F1")) for e in events]),
        "P_historical_best_ARI": _rate([bool(e.get("historical_inverse_is_best_ARI")) for e in events]),
        "P_historical_near_best_F1": _rate([bool(e.get("historical_in_near_best_F1")) for e in events]),
        "P_historical_near_best_ARI": _rate([bool(e.get("historical_in_near_best_ARI")) for e in events]),
        "P_historical_beneficial_F1": _rate([bool(e.get("historical_inverse_is_beneficial_F1")) for e in events]),
        "P_historical_beneficial_ARI": _rate([bool(e.get("historical_inverse_is_beneficial_ARI")) for e in events]),
        "mean_historical_rank_F1": _mean(take("historical_inverse_rank_F1")),
        "mean_historical_rank_ARI": _mean(take("historical_inverse_rank_ARI")),
        "mean_best_delta_F1": _mean(take("best_delta_F1")),
        "mean_historical_delta_F1": _mean(take("historical_delta_F1")),
        "mean_best_delta_ARI": _mean(take("best_delta_ARI")),
        "mean_historical_delta_ARI": _mean(take("historical_delta_ARI")),
        "P_exact_recon_A_minus": _rate([bool(e.get("exact_recon_A_minus")) for e in events]),
        "frac_near_best_F1_gt_1": _rate([e.get("num_near_best_F1", 0) > 1 for e in events]),
        "frac_near_best_ARI_gt_1": _rate([e.get("num_near_best_ARI", 0) > 1 for e in events]),
        "frac_beneficial_F1_gt_1": _rate([e.get("num_beneficial_F1", 0) > 1 for e in events]),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--seed0", type=int, default=2001)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["joint_j1"]
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    if sample["id"] != cfg["fixed_sample_id"]:
        raise ValueError("sample id mismatch")
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)
    schedule = AsyncJumpSchedule.from_config(cfg.get("schedule") or {})

    clean0 = a_from_role_and_copy(
        role=sample["role"],
        copy=sample["copy"],
        partition=partition,
        atomic_numbers=sample["z"],
        role_z=sample["role_z"],
        K=int(sample["Z"]),
    )

    events: list[dict] = []
    per_seed_counts: list[dict] = []
    for k in range(int(args.n_seeds)):
        if len(events) >= int(args.max_events):
            break
        seed = int(args.seed0) + k
        g = torch.Generator().manual_seed(seed)
        aug = sample_symmetry_augment(
            atomic_numbers=clean0.atomic_numbers, K=clean0.K, generator=g
        )
        st0 = apply_symmetry_to_state(clean0, aug)
        # clean partition after the same symmetry (copy-col perm leaves C invariant)
        c0 = st0.C()
        copy0 = st0.copy_of()
        traj = simulate_forward_ctmc(st0, schedule=schedule, generator=g)
        n_this = 0
        for e in traj.events:
            if e.kind != "G":
                continue
            if not (0.35 <= e.time <= 0.75):
                continue
            a_plus = traj.state_at(e.time)
            a_minus = apply_move(a_plus, LegalMove("G", e.i, e.j))
            rec = _audit_one_event(
                a_plus=a_plus,
                a_minus=a_minus,
                event_i=e.i,
                event_j=e.j,
                c0=c0,
                copy0=copy0,
                tau=float(e.time),
                seed=seed,
            )
            events.append(rec)
            n_this += 1
            if len(events) >= int(args.max_events):
                break
        per_seed_counts.append({"seed": seed, "n_g_events_kept": n_this, "n_traj_events": len(traj.events)})
        print(json.dumps({"event": "g_amb_seed", "seed": seed, "n_kept": n_this, "total": len(events)}), flush=True)

    bins: dict[str, list] = defaultdict(list)
    mid: list[dict] = []
    for e in events:
        bins[e["bin"]].append(e)
        if 0.40 <= e["tau"] < 0.60:
            mid.append(e)

    summary = {
        "sample": sample["id"],
        "n_seeds_used": len(per_seed_counts),
        "eps_f1": EPS_F1,
        "eps_ari": EPS_ARI,
        "near_f1": NEAR_F1,
        "near_ari": NEAR_ARI,
        "all": _aggregate(events),
        "t_040_060": _aggregate(mid),
        "by_bin": {name: _aggregate(bins[name]) for _, _, name in G_BINS},
        "per_seed": per_seed_counts,
    }

    ev_path = out / "g_event_ambiguity_events.jsonl"
    with ev_path.open("w") as f:
        for rec in events:
            f.write(json.dumps(rec) + "\n")
    sum_path = out / "g_event_ambiguity_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "event": "g_amb_done",
                "n_events": len(events),
                "n_040_060": len(mid),
                "P_hist_best_F1": summary["all"]["P_historical_best_F1"],
                "P_hist_best_ARI": summary["all"]["P_historical_best_ARI"],
                "mean_near_best_F1": summary["all"]["mean_num_near_best_F1"],
                "mean_beneficial_F1": summary["all"]["mean_num_beneficial_F1"],
                "events": str(ev_path),
                "summary": str(sum_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
