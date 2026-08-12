#!/usr/bin/env python3
"""N2.1 paired eval: B0 / B2 correct-C / B5 orbit-shuffle / B6 oracle on shared noisy batches.

Uses ONE trained B2 adapter for B2/B5/B6 (only soft-C changes at eval).
Improvement I = L_B0 - L_mode; D_C = L_B5 - L_B2.
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

from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss
from mattergen.assignment.soft_c_geometry_feedback_n2_1.setup_utils import (
    build_chemgraph_batch,
    load_fixed_sample_bundle,
    load_n21_stack,
)

MODES = ["B0_baseline", "B2_correct_c", "B5_shuffled_c", "B6_oracle_c"]


def _region(tf: float) -> str:
    if tf <= 0.30:
        return "A_exact_c"
    if tf <= 0.50:
        return "B_soft_info"
    return "C_gate_off"


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
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["assignment_n21"]
    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    )
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    model, pl_module, _ = load_n21_stack(
        cfg=cfg,
        device=device,
        n1_ckpt_path=args.n1_checkpoint or cfg["n1_checkpoint"],
        mattergen_model_path=args.mattergen_model_path,
        mattergen_load_epoch=args.mattergen_load_epoch,
        mattergen_checkpoint=args.mattergen_checkpoint,
    )
    ckpt = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
    key = "edge_modulator" if "edge_modulator" in ckpt else "edge_adapter"
    model.edge_modulator.load_state_dict(ckpt[key] if key in ckpt else ckpt["edge_modulator"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    sample, _part, backbone, o2_target, oracle_bar = load_fixed_sample_bundle(cfg, device)
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    loss_fn = pl_module.diffusion_module.loss_fn
    eval_cfg = cfg.get("evaluation") or {}
    fracs = eval_cfg.get("timestep_fractions") or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    seeds = eval_cfg.get("seeds") or [1001, 1002, 1003, 1004, 1017, 1042, 1123, 2026]

    rows = []
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

                soft_c, meta = model.pass_a_soft_c(
                    sample=sample,
                    frac_t=noisy.frac_coords_t,
                    cell_t=noisy.lattice_t,
                    t=noisy.t,
                    o2_target=o2_target,
                    backbone_tree=backbone,
                    oracle_bar=oracle_bar,
                    mode="B2_correct_c",
                )
                g_shuf = torch.Generator(device="cpu")
                g_shuf.manual_seed(int(seed) + 17 + int(1000 * float(tf)))
                soft_c_shuf = (
                    model.apply_orbit_preserving_shuffle(soft_c, oracle_bar, generator=g_shuf)
                    if soft_c is not None
                    else None
                )
                copy = sample["copy"]
                soft_c_oracle = copy[:, None].eq(copy[None, :]).float()

                losses = {}
                scores = {}
                for mode in MODES:
                    if mode == "B0_baseline":
                        sc = None
                    elif mode == "B2_correct_c":
                        sc = soft_c
                    elif mode == "B5_shuffled_c":
                        sc = soft_c_shuf
                    else:
                        sc = soft_c_oracle
                    out_sc = model.forward_geometry(
                        chemgraph=noisy_cg,
                        t=noisy.t.reshape(-1),
                        soft_c=sc,
                        t_fraction=float(tf),
                        mode=mode,  # type: ignore[arg-type]
                    )
                    loss, _ = mattergen_geometry_loss(
                        loss_fn=loss_fn,
                        corruption=noise.corruption,
                        clean_batch=clean_cg,
                        noisy_batch=noisy_cg,
                        score_model_output=out_sc,
                        t=noisy.t.reshape(-1),
                    )
                    losses[mode] = float(loss)
                    scores[mode] = {
                        "pos": out_sc["pos"].detach().cpu(),
                        "cell": out_sc["cell"].detach().cpu(),
                    }

                b0 = losses["B0_baseline"]
                imp = {m: b0 - losses[m] for m in MODES}
                d_c = losses["B5_shuffled_c"] - losses["B2_correct_c"]
                gate_off = float(tf) >= 0.60
                b2_eq_b0 = bool(
                    torch.allclose(scores["B2_correct_c"]["pos"], scores["B0_baseline"]["pos"], atol=1e-5, rtol=1e-4)
                    and torch.allclose(scores["B2_correct_c"]["cell"], scores["B0_baseline"]["cell"], atol=1e-5, rtol=1e-4)
                )
                b5_eq_b0 = bool(
                    torch.allclose(scores["B5_shuffled_c"]["pos"], scores["B0_baseline"]["pos"], atol=1e-5, rtol=1e-4)
                    and torch.allclose(scores["B5_shuffled_c"]["cell"], scores["B0_baseline"]["cell"], atol=1e-5, rtol=1e-4)
                )
                row = {
                    "t_fraction": float(tf),
                    "region": _region(float(tf)),
                    "seed": int(seed),
                    "sigma_x": float(noisy.sigma_x),
                    "log_snr_x": float(noisy.log_snr_x),
                    "losses": losses,
                    "improvement_vs_B0": imp,
                    "D_C": d_c,
                    "b2_lt_b0": bool(losses["B2_correct_c"] < b0),
                    "b2_lt_b5": bool(losses["B2_correct_c"] < losses["B5_shuffled_c"]),
                    "gate_off_region": gate_off,
                    "b2_eq_b0": b2_eq_b0,
                    "b5_eq_b0": b5_eq_b0,
                    "structural_off_ok": (not gate_off) or (b2_eq_b0 and b5_eq_b0),
                    "soft_c_source_b2": meta.get("soft_c_source"),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "n21_eval_point",
                            "t_fraction": tf,
                            "seed": seed,
                            "region": row["region"],
                            "I_B2": imp["B2_correct_c"],
                            "I_B5": imp["B5_shuffled_c"],
                            "D_C": d_c,
                            "b2_lt_b5": row["b2_lt_b5"],
                            "structural_off_ok": row["structural_off_ok"],
                        }
                    ),
                    flush=True,
                )

    with (out / "paired_geometry_eval.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def _stats(vals):
        if not vals:
            return {"mean": None, "median": None, "std": None, "n": 0}
        t = torch.tensor(vals, dtype=torch.float64)
        return {
            "mean": float(t.mean()),
            "median": float(t.median()),
            "std": float(t.std(unbiased=False)),
            "n": len(vals),
        }

    by_t = defaultdict(list)
    by_reg = defaultdict(list)
    for r in rows:
        by_t[r["t_fraction"]].append(r)
        by_reg[r["region"]].append(r)

    curve = []
    for tf in sorted(by_t):
        ch = by_t[tf]
        curve.append(
            {
                "t_fraction": tf,
                "region": _region(tf),
                "mean_L_B0": sum(c["losses"]["B0_baseline"] for c in ch) / len(ch),
                "mean_L_B2": sum(c["losses"]["B2_correct_c"] for c in ch) / len(ch),
                "mean_L_B5": sum(c["losses"]["B5_shuffled_c"] for c in ch) / len(ch),
                "mean_L_B6": sum(c["losses"]["B6_oracle_c"] for c in ch) / len(ch),
                "mean_I_B2": sum(c["improvement_vs_B0"]["B2_correct_c"] for c in ch) / len(ch),
                "mean_I_B5": sum(c["improvement_vs_B0"]["B5_shuffled_c"] for c in ch) / len(ch),
                "mean_D_C": sum(c["D_C"] for c in ch) / len(ch),
                "frac_B2_lt_B0": sum(1 for c in ch if c["b2_lt_b0"]) / len(ch),
                "frac_B2_lt_B5": sum(1 for c in ch if c["b2_lt_b5"]) / len(ch),
                "structural_off_ok_rate": sum(1 for c in ch if c["structural_off_ok"]) / len(ch),
                "n": len(ch),
            }
        )
    (out / "geometry_curve_summary.json").write_text(json.dumps(curve, indent=2))

    region_summary = {}
    for reg, ch in by_reg.items():
        i2 = [c["improvement_vs_B0"]["B2_correct_c"] for c in ch]
        i5 = [c["improvement_vs_B0"]["B5_shuffled_c"] for c in ch]
        dc = [c["D_C"] for c in ch]
        region_summary[reg] = {
            "I_B2": _stats(i2),
            "I_B5": _stats(i5),
            "D_C": _stats(dc),
            "frac_B2_lt_B0": sum(1 for c in ch if c["b2_lt_b0"]) / len(ch),
            "frac_B2_lt_B5": sum(1 for c in ch if c["b2_lt_b5"]) / len(ch),
            "structural_off_ok_rate": sum(1 for c in ch if c["structural_off_ok"]) / len(ch),
            "n": len(ch),
        }
    (out / "region_summary.json").write_text(json.dumps(region_summary, indent=2))

    # Gate A structural
    gate_a_ok = all(c["structural_off_ok"] for c in rows if c["gate_off_region"])
    a_rows = [c for c in rows if c["region"] == "A_exact_c"]
    gate_b = sum(1 for c in a_rows if c["b2_lt_b0"]) / max(1, len(a_rows))
    gate_c = sum(1 for c in a_rows if c["b2_lt_b5"]) / max(1, len(a_rows))
    summary = {
        "formula": "delta_e = g * q * s * F_psi(e)",
        "group_context": False,
        "b5_same_adapter_weights": True,
        "b5_shuffle": "orbit_preserving",
        "gate_A_structural_off_ok": gate_a_ok,
        "gate_B_frac_B2_lt_B0_region_A": gate_b,
        "gate_C_frac_B2_lt_B5_region_A": gate_c,
        "region_summary": region_summary,
        "by_t_fraction": curve,
        "claim_support_region_A": bool(gate_b > 0.5 and gate_c > 0.5 and gate_a_ok),
    }
    (out / "ablation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "n21_eval_done", "summary": summary}, default=str), flush=True)


if __name__ == "__main__":
    main()
