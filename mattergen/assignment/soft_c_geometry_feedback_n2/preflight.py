"""Minimal baseline-equivalence preflight for N2 training."""
from __future__ import annotations

from typing import Any

import torch

from mattergen.assignment.soft_c_geometry_feedback_n2.gates import noise_gate_value
from mattergen.assignment.soft_c_geometry_feedback_n2.module import SoftCGeometryFeedbackN2
from mattergen.assignment.soft_c_geometry_feedback_n2.setup_utils import build_chemgraph_batch


def _score_close(a, b, *, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    return bool(
        torch.allclose(a["pos"], b["pos"], atol=atol, rtol=rtol)
        and torch.allclose(a["cell"], b["cell"], atol=atol, rtol=rtol)
    )


@torch.no_grad()
def run_equality_preflight(
    *,
    model: SoftCGeometryFeedbackN2,
    sample: dict,
    backbone_tree,
    o2_target,
    oracle_bar: torch.Tensor,
    noise_adapter,
    seed: int = 12345,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Three gates before training:

    1. B0 vs B1 geometry scores (same noisy batch)
    2. Zero-init B2 vs B0 (adapters start at 0 residual)
    3. High-noise t/T>=full_off: B2 vs B0 (gate off)
    """
    device = sample["pos"].device
    T = float(noise_adapter.T)
    full_off = float(model.config.noise_gate.full_off_t_fraction)
    report: dict[str, Any] = {"ok": True, "checks": []}

    def _one(tf: float, label: str) -> dict[str, Any]:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + int(1000 * tf))
        noisy = noise_adapter.corrupt_at_fraction(
            frac_coords_0=sample["pos"],
            lattice_0=sample["cell"],
            num_atoms=int(sample["N"]),
            t_fraction=float(tf),
            generator=g,
        )
        noisy_cg = build_chemgraph_batch(sample, noisy.frac_coords_t, noisy.lattice_t)
        t_vec = noisy.t.reshape(-1)
        # Shared Pass-A soft C / H^A (geometry fixed)
        soft_c, h_a, _ = model.pass_a_assignment(
            sample=sample,
            frac_t=noisy.frac_coords_t,
            cell_t=noisy.lattice_t,
            t=noisy.t,
            o2_target=o2_target,
            backbone_tree=backbone_tree,
            oracle_bar=oracle_bar,
            mode="B2_combined",
        )

        def _fwd(mode: str, sc=soft_c, ha=h_a):
            out = model.forward_geometry(
                chemgraph=noisy_cg,
                t=t_vec,
                soft_c=sc,
                h_a=ha,
                t_fraction=float(tf),
                mode=mode,  # type: ignore[arg-type]
            )
            return {"pos": out["pos"].detach(), "cell": out["cell"].detach()}

        s_b0 = _fwd("B0_baseline", sc=None, ha=h_a)
        s_b1 = _fwd("B1_no_feedback", sc=soft_c, ha=h_a)
        s_b2 = _fwd("B2_combined", sc=soft_c, ha=h_a)
        g_noise = float(noise_gate_value(tf, model.config.noise_gate).reshape(-1)[0].item())
        return {
            "label": label,
            "t_fraction": float(tf),
            "g_noise": g_noise,
            "b0_vs_b1": _score_close(s_b0, s_b1, atol=atol, rtol=rtol),
            "b0_vs_b2": _score_close(s_b0, s_b2, atol=atol, rtol=rtol),
            "pos_max_abs_b0_b1": float((s_b0["pos"] - s_b1["pos"]).abs().max()),
            "pos_max_abs_b0_b2": float((s_b0["pos"] - s_b2["pos"]).abs().max()),
        }

    # Gate 1+2: low noise (feedback can be active; zero-init ⇒ B2≈B0; B1≡B0 always)
    low = _one(0.10, "low_noise_zero_init")
    report["checks"].append(low)
    if not low["b0_vs_b1"]:
        report["ok"] = False
        report["failure"] = "PREFLIGHT_B0_B1_MISMATCH"
    if not low["b0_vs_b2"]:
        report["ok"] = False
        report["failure"] = report.get("failure") or "PREFLIGHT_ZERO_INIT_B2_NEQ_B0"

    # Gate 3: high noise gate off
    high_tf = max(full_off, 0.60)
    high = _one(high_tf, "high_noise_gate_off")
    report["checks"].append(high)
    if high["g_noise"] > 1e-8:
        report["ok"] = False
        report["failure"] = "PREFLIGHT_HIGH_NOISE_GATE_NOT_ZERO"
    elif not high["b0_vs_b2"]:
        report["ok"] = False
        report["failure"] = report.get("failure") or "PREFLIGHT_HIGH_NOISE_B2_NEQ_B0"
    if not high["b0_vs_b1"]:
        report["ok"] = False
        report["failure"] = report.get("failure") or "PREFLIGHT_B0_B1_MISMATCH_HIGH_NOISE"

    return report
