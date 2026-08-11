#!/usr/bin/env python3
"""Paired geometry eval for N2 modes (B0–B5) under held-out noise seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss
from mattergen.assignment.soft_c_geometry_feedback_n2.setup_utils import (
    build_chemgraph_batch,
    load_fixed_sample_bundle,
    load_n2_stack,
)

MODES = [
    "B0_baseline",
    "B1_no_feedback",
    "B2_combined",
    "B3_edge_only",
    "B4_group_only",
    "B5_shuffled_c",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--adapter-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    p.add_argument("--n1-checkpoint", type=str, default=None)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing evaluation without --execute")

    cfg = yaml.safe_load(args.config.read_text())["assignment_n2"]
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    n1_ckpt = args.n1_checkpoint or cfg.get("n1_checkpoint")
    model, pl_module, _ = load_n2_stack(
        cfg=cfg,
        device=device,
        n1_ckpt_path=n1_ckpt,
        mattergen_model_path=args.mattergen_model_path,
        mattergen_load_epoch=args.mattergen_load_epoch,
        mattergen_checkpoint=args.mattergen_checkpoint,
    )
    ckpt = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    model.edge_adapter.load_state_dict(ckpt["edge_adapter"])
    model.group_adapter.load_state_dict(ckpt["group_adapter"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    sample, partition, backbone, o2_target, oracle_bar = load_fixed_sample_bundle(cfg, device)
    del partition
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    loss_fn = pl_module.diffusion_module.loss_fn
    T = float(noise.T)

    eval_cfg = cfg.get("evaluation") or {}
    fracs = eval_cfg.get("timestep_fractions") or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    seeds = eval_cfg.get("seeds") or [1001, 1002, 1003, 1004, 1017, 1042, 1123, 2026]

    rows = []
    gate_rows = []
    with torch.no_grad():
        for tf in fracs:
            for seed in seeds:
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed) + int(1000 * float(tf)))
                noisy = noise.corrupt_at_fraction(
                    frac_coords_0=sample["pos"],
                    lattice_0=sample["cell"],
                    num_atoms=int(sample["N"]),
                    t_fraction=float(tf),
                    generator=g,
                )
                clean_cg = build_chemgraph_batch(sample, sample["pos"], sample["cell"])
                noisy_cg = build_chemgraph_batch(sample, noisy.frac_coords_t, noisy.lattice_t)
                losses = {}
                scores = {}
                for mode in MODES:
                    soft_c, h_a, _ = model.pass_a_assignment(
                        sample=sample,
                        frac_t=noisy.frac_coords_t,
                        cell_t=noisy.lattice_t,
                        t=noisy.t,
                        o2_target=o2_target,
                        backbone_tree=backbone,
                        oracle_bar=oracle_bar,
                        mode=mode,  # type: ignore[arg-type]
                    )
                    score_out = model.forward_geometry(
                        chemgraph=noisy_cg,
                        t=noisy.t.reshape(-1),
                        soft_c=soft_c,
                        h_a=h_a,
                        t_fraction=float(tf),
                        mode=mode,  # type: ignore[arg-type]
                    )
                    loss, metrics = mattergen_geometry_loss(
                        loss_fn=loss_fn,
                        corruption=noise.corruption,
                        clean_batch=clean_cg,
                        noisy_batch=noisy_cg,
                        score_model_output=score_out,
                        t=noisy.t.reshape(-1),
                    )
                    losses[mode] = float(loss)
                    scores[mode] = {
                        "pos": score_out["pos"].detach().cpu(),
                        "cell": score_out["cell"].detach().cpu(),
                    }
                    if model._last_diag is not None:
                        gate_rows.append(
                            {
                                "t_fraction": float(tf),
                                "seed": int(seed),
                                "mode": mode,
                                "g_noise": model._last_diag.g_noise,
                                "edge_audit": model._last_diag.edge_audit,
                                "group_audit": model._last_diag.group_audit,
                            }
                        )
                # paired deltas vs B0
                b0 = losses["B0_baseline"]
                row = {
                    "t_fraction": float(tf),
                    "seed": int(seed),
                    "sigma_x": float(noisy.sigma_x),
                    "sigma_l": float(noisy.sigma_l),
                    "log_snr_x": float(noisy.log_snr_x),
                    "losses": losses,
                    "delta_vs_B0": {m: losses[m] - b0 for m in MODES},
                    # B1 must match B0 scores
                    "b1_matches_b0_pos": bool(
                        torch.allclose(scores["B1_no_feedback"]["pos"], scores["B0_baseline"]["pos"], atol=1e-5, rtol=1e-4)
                    ),
                    "b1_matches_b0_cell": bool(
                        torch.allclose(scores["B1_no_feedback"]["cell"], scores["B0_baseline"]["cell"], atol=1e-5, rtol=1e-4)
                    ),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "n2_eval_point",
                            "t_fraction": tf,
                            "seed": seed,
                            "B0": b0,
                            "B2": losses["B2_combined"],
                            "delta_B2": losses["B2_combined"] - b0,
                            "b1_match": row["b1_matches_b0_pos"] and row["b1_matches_b0_cell"],
                        }
                    ),
                    flush=True,
                )

    with (out / "paired_geometry_eval.jsonl").open("w") as f:
        for r in rows:
            # drop large tensors
            f.write(json.dumps({k: v for k, v in r.items() if k != "scores"}) + "\n")
    with (out / "feedback_gate_audit.jsonl").open("w") as f:
        for r in gate_rows:
            f.write(json.dumps(r) + "\n")

    # Aggregate by t_fraction
    from collections import defaultdict

    by_t = defaultdict(list)
    for r in rows:
        by_t[r["t_fraction"]].append(r)
    summary = []
    for tf in sorted(by_t):
        chunk = by_t[tf]
        def mean_delta(mode):
            return sum(c["delta_vs_B0"][mode] for c in chunk) / len(chunk)
        summary.append(
            {
                "t_fraction": tf,
                "mean_loss_B0": sum(c["losses"]["B0_baseline"] for c in chunk) / len(chunk),
                "mean_loss_B2": sum(c["losses"]["B2_combined"] for c in chunk) / len(chunk),
                "mean_delta_B2": mean_delta("B2_combined"),
                "mean_delta_B3": mean_delta("B3_edge_only"),
                "mean_delta_B4": mean_delta("B4_group_only"),
                "mean_delta_B5": mean_delta("B5_shuffled_c"),
                "b1_match_rate": sum(1 for c in chunk if c["b1_matches_b0_pos"] and c["b1_matches_b0_cell"]) / len(chunk),
                "n": len(chunk),
            }
        )
    (out / "geometry_curve_summary.json").write_text(json.dumps(summary, indent=2))
    ablation = {
        "modes": MODES,
        "note": "negative delta_vs_B0 means lower geometry loss than baseline (improvement)",
        "by_t_fraction": summary,
        "overall_mean_delta_B2": sum(s["mean_delta_B2"] for s in summary) / max(1, len(summary)),
        "overall_mean_delta_B5_shuffled": sum(s["mean_delta_B5"] for s in summary) / max(1, len(summary)),
    }
    (out / "ablation_summary.json").write_text(json.dumps(ablation, indent=2))
    print(json.dumps({"event": "n2_eval_done", "output": str(out)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
