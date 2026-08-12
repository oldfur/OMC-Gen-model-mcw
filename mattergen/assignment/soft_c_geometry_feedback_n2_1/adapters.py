"""N2.1 causal edge adapter: F_ψ(e) only; C enters solely as scalar multiplier."""
from __future__ import annotations

import torch
from torch import nn

from mattergen.assignment.soft_c_geometry_feedback_n2.adapters import ZeroInitLinear


class CausalEdgeModulator(nn.Module):
    """F_ψ(edge_emb) → residual direction; C cannot enter this MLP.

    Δe = (g · q · s) * F_ψ(e)  is applied by the caller.
    Final layer zero-init ⇒ step-0 residual direction is 0 ⇒ Δe=0.
    """

    def __init__(self, emb_size_edge: int, bottleneck: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size_edge, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
            ZeroInitLinear(bottleneck, emb_size_edge),
        )

    def forward(self, edge_emb: torch.Tensor) -> torch.Tensor:
        """Return F_ψ(e) only (same shape as edge_emb). No C inputs."""
        if edge_emb.shape[0] == 0:
            return edge_emb
        return self.net(edge_emb)
