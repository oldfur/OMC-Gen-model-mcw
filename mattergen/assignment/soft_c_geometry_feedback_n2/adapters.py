"""Zero-init residual adapters for soft-C edge and group feedback (N2)."""
from __future__ import annotations

import torch
from torch import nn


class ZeroInitLinear(nn.Linear):
    """Linear layer with zero weight and bias (identity residual at init)."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class SoftCEdgeAdapter(nn.Module):
    """Edge residual: e' = e + Adapter(e, c_intra, c_inter, c_signed).

    Final layer zero-init ⇒ Δe=0 at start (epoch294 baseline).
    """

    def __init__(self, emb_size_edge: int, bottleneck: int = 64, n_channels: int = 3):
        super().__init__()
        self.n_channels = n_channels
        in_dim = emb_size_edge + n_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
            ZeroInitLinear(bottleneck, emb_size_edge),
        )

    def forward(
        self,
        edge_emb: torch.Tensor,
        c_intra: torch.Tensor,
        c_inter: torch.Tensor,
        c_signed: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual Δe (same shape as edge_emb)."""
        if edge_emb.shape[0] == 0:
            return edge_emb
        ch = torch.stack([c_intra, c_inter, c_signed], dim=-1)  # [E, 3]
        x = torch.cat([edge_emb, ch.to(dtype=edge_emb.dtype)], dim=-1)
        return self.net(x)


class SoftCGroupAdapter(nn.Module):
    """Node residual from expected same-copy context.

    Inputs: H_i^A, h_group, h_group-H_i^A, g_group (scalar).
    Final layer zero-init.
    """

    def __init__(self, hidden: int, bottleneck: int = 128):
        super().__init__()
        # 3 * hidden + 1 confidence scalar
        in_dim = 3 * hidden + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
            nn.SiLU(),
            ZeroInitLinear(bottleneck, hidden),
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_group: torch.Tensor,
        g_group: torch.Tensor,
    ) -> torch.Tensor:
        """Return unscaled residual Δh (caller multiplies by g_group)."""
        diff = h_group - h_a
        g = g_group.reshape(-1, 1).to(dtype=h_a.dtype)
        x = torch.cat([h_a, h_group, diff, g], dim=-1)
        return self.net(x)
