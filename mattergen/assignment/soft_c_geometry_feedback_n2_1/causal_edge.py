"""Strict C-dependent edge residual for N2.1.

Δe_ij = g_noise(t) · q_ij · s_ij · F_ψ(e_ij)

with q = 2|p-1/2|, s = 2p-1, p = soft_C_ij.

No C concat into F_ψ. No residual path that bypasses (g q s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mattergen.assignment.soft_c_geometry_feedback_n2.gates import (
    NoiseGateConfig,
    noise_gate_value,
    pairwise_confidence,
)


@dataclass
class CausalEdgeAudit:
    num_edges: int = 0
    g_noise: float = 0.0
    mean_abs_coeff: float = 0.0
    frac_zero_coeff: float = 1.0
    num_zeroed_self_image: int = 0
    num_nonzero_image_edges: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_edges": self.num_edges,
            "g_noise": self.g_noise,
            "mean_abs_coeff": self.mean_abs_coeff,
            "frac_zero_coeff": self.frac_zero_coeff,
            "num_zeroed_self_image": self.num_zeroed_self_image,
            "num_nonzero_image_edges": self.num_nonzero_image_edges,
            "formula": "delta_e = g * q * s * F_psi(e)",
            "f_psi_inputs": "edge_emb_only",
        }


def causal_edge_coefficients(
    *,
    soft_c: torch.Tensor,
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor | None,
    t_fraction: float,
    noise_gate_cfg: NoiseGateConfig,
) -> tuple[torch.Tensor, CausalEdgeAudit]:
    """Per-edge scalar coeff = g * q * s; PBC self-image forced to 0."""
    soft_c = soft_c.detach().float()
    src = edge_index[0].long()
    dst = edge_index[1].long()
    e = int(src.numel())
    audit = CausalEdgeAudit(num_edges=e)
    if e == 0:
        return soft_c.new_zeros(0), audit

    p = soft_c[src, dst].clamp(0.0, 1.0)
    q = pairwise_confidence(p)
    s = 2.0 * p - 1.0
    g = noise_gate_value(t_fraction, noise_gate_cfg).to(device=p.device, dtype=p.dtype)
    if g.ndim > 0:
        g = g.reshape(-1)[0]
    audit.g_noise = float(g.item()) if torch.is_tensor(g) else float(g)
    coeff = (g * q * s).to(dtype=p.dtype)

    if cell_offsets is not None:
        off = cell_offsets.reshape(e, -1)
        nonzero_img = off.abs().sum(dim=-1) > 0
        audit.num_nonzero_image_edges = int(nonzero_img.sum().item())
        self_img = src.eq(dst) & nonzero_img
        if self_img.any():
            coeff = coeff.clone()
            coeff[self_img] = 0.0
            audit.num_zeroed_self_image = int(self_img.sum().item())

    audit.mean_abs_coeff = float(coeff.abs().mean().item())
    audit.frac_zero_coeff = float((coeff.abs() < 1e-12).float().mean().item())
    return coeff, audit


def apply_causal_edge_residual(
    edge_emb: torch.Tensor,
    *,
    soft_c: torch.Tensor,
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor | None,
    t_fraction: float,
    noise_gate_cfg: NoiseGateConfig,
    f_psi: nn.Module,
) -> tuple[torch.Tensor, CausalEdgeAudit]:
    """Δe = coeff[..., None] * F_ψ(edge_emb)."""
    coeff, audit = causal_edge_coefficients(
        soft_c=soft_c,
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        t_fraction=t_fraction,
        noise_gate_cfg=noise_gate_cfg,
    )
    if edge_emb.shape[0] == 0:
        return edge_emb, audit
    direction = f_psi(edge_emb)
    if direction.shape != edge_emb.shape:
        raise RuntimeError(
            f"F_ψ output shape {tuple(direction.shape)} != edge_emb {tuple(edge_emb.shape)}"
        )
    delta = coeff.unsqueeze(-1).to(dtype=edge_emb.dtype) * direction
    return delta, audit
