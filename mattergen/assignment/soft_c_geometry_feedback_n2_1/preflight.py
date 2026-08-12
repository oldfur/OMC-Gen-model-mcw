"""Minimal N2.1 equality / structural-off preflight."""
from __future__ import annotations

from typing import Any

import torch

from mattergen.assignment.soft_c_geometry_feedback_n2.gates import noise_gate_value
from mattergen.assignment.soft_c_geometry_feedback_n2_1.module import SoftCCausalEdgeN21
from mattergen.assignment.soft_c_geometry_feedback_n2_1.setup_utils import build_chemgraph_batch


def _close(a, b, *, atol=1e-5, rtol=1e-4) -> bool:
    return bool(
        torch.allclose(a["pos"], b["pos"], atol=atol, rtol=rtol)
        and torch.allclose(a["cell"], b["cell"], atol=atol, rtol=rtol)
    )


@torch.no_grad()
def run_n21_preflight(
    *,
    model: SoftCCausalEdgeN21,
    sample: dict,
    backbone_tree,
    o2_target,
    oracle_bar: torch.Tensor,
    noise_adapter,
    seed: int = 12345,
) -> dict[str, Any]:
    full_off = float(model.config.noise_gate.full_off_t_fraction)
    report: dict[str, Any] = {"ok": True, "checks": []}

    def _scores(tf: float):
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
        soft_c, meta = model.pass_a_soft_c(
            sample=sample,
            frac_t=noisy.frac_coords_t,
            cell_t=noisy.lattice_t,
            t=noisy.t,
            o2_target=o2_target,
            backbone_tree=backbone_tree,
            oracle_bar=oracle_bar,
            mode="B2_correct_c",
        )
        t_vec = noisy.t.reshape(-1)

        def fwd(mode, sc):
            out = model.forward_geometry(
                chemgraph=noisy_cg,
                t=t_vec,
                soft_c=sc,
                t_fraction=float(tf),
                mode=mode,  # type: ignore[arg-type]
            )
            return {"pos": out["pos"].detach(), "cell": out["cell"].detach()}

        s0 = fwd("B0_baseline", None)
        s2 = fwd("B2_correct_c", soft_c)
        g_noise = float(noise_gate_value(tf, model.config.noise_gate).reshape(-1)[0].item())
        return {
            "t_fraction": tf,
            "g_noise": g_noise,
            "b0_vs_b2": _close(s0, s2),
            "pos_max_abs": float((s0["pos"] - s2["pos"]).abs().max()),
            "soft_c_source": meta.get("soft_c_source"),
        }

    # Zero-init / low-noise: B2 == B0 (F_ψ last layer zero ⇒ Δe=0 even if g q s ≠ 0)
    low = _scores(0.10)
    low["label"] = "zero_init_or_low_noise"
    report["checks"].append(low)
    if not low["b0_vs_b2"]:
        report["ok"] = False
        report["failure"] = "PREFLIGHT_ZERO_INIT_B2_NEQ_B0"

    # High-noise structural off
    high = _scores(max(full_off, 0.60))
    high["label"] = "high_noise_gate_off"
    report["checks"].append(high)
    if high["g_noise"] > 1e-8:
        report["ok"] = False
        report["failure"] = "PREFLIGHT_GATE_NOT_ZERO"
    elif not high["b0_vs_b2"]:
        report["ok"] = False
        report["failure"] = report.get("failure") or "PREFLIGHT_HIGH_NOISE_B2_NEQ_B0"

    return report
