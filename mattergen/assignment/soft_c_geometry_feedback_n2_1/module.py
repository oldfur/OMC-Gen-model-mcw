"""N2.1: strict C-dependent causal edge feedback (no group, no unconditional residual)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from mattergen.assignment.global_copy_assembly.orbit_membership import OrbitPartition
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import freeze_module, parameter_sha256
from mattergen.assignment.noisy_copy_assignment.module import NoisyCopyAssignmentN1
from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS
from mattergen.assignment.soft_c_geometry_feedback_n2.feedback import (
    orbit_preserving_shuffle_soft_c,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.gates import NoiseGateConfig, noise_gate_value
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.denoiser import GemNetTDenoiser

from .adapters import CausalEdgeModulator
from .causal_edge import apply_causal_edge_residual

N21Mode = Literal["B0_baseline", "B2_correct_c", "B5_shuffled_c", "B6_oracle_c"]


@dataclass
class CausalEdgeConfig:
    enabled: bool = True
    noise_gate: NoiseGateConfig = field(default_factory=NoiseGateConfig)
    edge_bottleneck: int = 64
    freeze_base_gemnet: bool = True
    freeze_n1: bool = True
    soft_c_semantics: str = SOFT_C_SEMANTICS
    mode: N21Mode = "B2_correct_c"
    # Group context permanently off for N2.1 main experiment
    group_context_enabled: bool = False


@dataclass
class N21Diagnostics:
    mode: str
    g_noise: float
    feedback_enabled: bool
    edge_audit: dict[str, Any] = field(default_factory=dict)
    soft_c_source: str = ""
    formula: str = "delta_e = g * q * s * F_psi(e)"
    group_context: bool = False
    g_diffusion: bool = False


class SoftCCausalEdgeN21(nn.Module):
    """Pass A frozen N1 soft-C; Pass B C-modulated edge residual only."""

    def __init__(
        self,
        *,
        denoiser: GemNetTDenoiser,
        n1_model: NoisyCopyAssignmentN1,
        partition: OrbitPartition,
        config: CausalEdgeConfig | None = None,
        mattergen_provenance: dict | None = None,
        n1_provenance: dict | None = None,
    ):
        super().__init__()
        self.config = config or CausalEdgeConfig()
        self.partition = partition
        self.denoiser = denoiser
        self.n1 = n1_model
        self.mattergen_provenance = dict(mattergen_provenance or {})
        self.n1_provenance = dict(n1_provenance or {})
        if self.config.freeze_base_gemnet:
            freeze_module(self.denoiser)
        if self.config.freeze_n1:
            freeze_module(self.n1)

        emb_edge = int(getattr(self.denoiser.gemnet, "emb_size_edge", self.denoiser.hidden_dim))
        ae = getattr(self.denoiser.gemnet, "angle_edge_emb", None)
        if ae is not None:
            last = list(ae.modules())[-1]
            if isinstance(last, nn.Linear):
                emb_edge = int(last.out_features)
        self.edge_modulator = CausalEdgeModulator(
            emb_size_edge=emb_edge,
            bottleneck=self.config.edge_bottleneck,
        )
        self._last_diag: N21Diagnostics | None = None

    def trainable_n21_parameters(self):
        for p in self.edge_modulator.parameters():
            if p.requires_grad:
                yield p

    def param_audit(self) -> dict[str, int]:
        return {
            "trainable_base_gemnet_params": sum(
                p.numel() for p in self.denoiser.parameters() if p.requires_grad
            ),
            "trainable_n1_params": sum(p.numel() for p in self.n1.parameters() if p.requires_grad),
            "trainable_n21_params": sum(p.numel() for p in self.trainable_n21_parameters()),
            "base_gemnet_total_params": sum(p.numel() for p in self.denoiser.parameters()),
        }

    def n1_parameter_hash(self) -> str:
        return parameter_sha256(self.n1)

    def gemnet_parameter_hash(self) -> str:
        return parameter_sha256(self.denoiser)

    def n21_parameter_hash(self) -> str:
        return parameter_sha256(self.edge_modulator)

    def _mode_flags(self, mode: N21Mode | None = None) -> dict[str, bool]:
        mode = mode or self.config.mode
        if mode == "B0_baseline":
            return dict(feedback=False, use_n1=False, shuffle=False, oracle=False)
        if mode == "B2_correct_c":
            return dict(feedback=True, use_n1=True, shuffle=False, oracle=False)
        if mode == "B5_shuffled_c":
            return dict(feedback=True, use_n1=True, shuffle=True, oracle=False)
        if mode == "B6_oracle_c":
            return dict(feedback=True, use_n1=False, shuffle=False, oracle=True)
        raise ValueError(mode)

    @torch.no_grad()
    def pass_a_soft_c(
        self,
        *,
        sample: dict,
        frac_t: torch.Tensor,
        cell_t: torch.Tensor,
        t: torch.Tensor,
        o2_target,
        backbone_tree,
        oracle_bar: torch.Tensor,
        orbit_mode: str = "predicted_orbit",
        mode: N21Mode | None = None,
        oracle_c: torch.Tensor | None = None,
        soft_c_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict]:
        """Frozen N1 (or oracle) → soft C. Geometry/H^A extraction as needed for N1."""
        flags = self._mode_flags(mode)
        meta: dict[str, Any] = {"mode": mode or self.config.mode, "flags": flags}
        if soft_c_override is not None:
            soft_c = soft_c_override.detach().float()
            meta["soft_c_source"] = "override"
        elif flags["oracle"]:
            if oracle_c is None:
                copy = sample["copy"]
                soft_c = copy[:, None].eq(copy[None, :]).float()
            else:
                soft_c = oracle_c.float()
            meta["soft_c_source"] = "ORACLE_UPPER_BOUND_ONLY"
        elif flags["use_n1"]:
            out = self.n1.map_decode(
                o2_target=o2_target,
                backbone_tree=backbone_tree,
                z=sample["z"],
                frac_t=frac_t,
                cell_t=cell_t,
                t=t,
                role_z=sample["role_z"],
                role_edge_index=sample["role_edge_index"],
                role_bond_type=sample["role_bond_type"],
                oracle_bar_r=oracle_bar,
                atomic_numbers=sample["z"],
                orbit_mode=orbit_mode,
                produce_soft_c=True,
            )
            soft_c = out.c_soft
            if soft_c is None:
                soft_c = out.c_map.float()
                meta["soft_c_fallback"] = "hard_c_map"
            meta["soft_c_source"] = "frozen_n1"
        else:
            return None, meta

        soft_c = soft_c.detach().clamp(0.0, 1.0)
        soft_c.fill_diagonal_(1.0)
        if flags["shuffle"]:
            orbit_labels = oracle_bar.argmax(dim=-1).long()
            soft_c = orbit_preserving_shuffle_soft_c(soft_c, orbit_labels)
            soft_c = soft_c.clamp(0.0, 1.0)
            soft_c.fill_diagonal_(1.0)
            meta["soft_c_source"] = "orbit_preserving_shuffled_n1"
            meta["shuffle_kind"] = "orbit_preserving"
        return soft_c, meta

    def apply_orbit_preserving_shuffle(
        self,
        soft_c: torch.Tensor,
        oracle_bar: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        orbit_labels = oracle_bar.argmax(dim=-1).long()
        out = orbit_preserving_shuffle_soft_c(soft_c.detach(), orbit_labels, generator=generator)
        out = out.clamp(0.0, 1.0)
        out.fill_diagonal_(1.0)
        return out

    def _build_feedback_state(
        self,
        *,
        soft_c: torch.Tensor,
        t_fraction: float,
        mode: str,
        soft_c_source: str,
    ) -> dict:
        g_noise = float(noise_gate_value(t_fraction, self.config.noise_gate).reshape(-1)[0].item())
        diag = N21Diagnostics(
            mode=mode,
            g_noise=g_noise,
            feedback_enabled=True,
            soft_c_source=soft_c_source,
        )

        def edge_adapter_fn(m, edge_index, cell_offsets, soft_c=soft_c, t_fraction=t_fraction):
            delta, audit = apply_causal_edge_residual(
                m,
                soft_c=soft_c,
                edge_index=edge_index,
                cell_offsets=cell_offsets,
                t_fraction=t_fraction,
                noise_gate_cfg=self.config.noise_gate,
                f_psi=self.edge_modulator,
            )
            diag.edge_audit = audit.as_dict()
            return delta

        self._last_diag = diag
        return {
            "enabled": True,
            "node_delta": None,  # group permanently off
            "edge_adapter": edge_adapter_fn,
        }

    def forward_geometry(
        self,
        *,
        chemgraph: ChemGraph,
        t: torch.Tensor,
        soft_c: torch.Tensor | None,
        t_fraction: float,
        mode: N21Mode | None = None,
        soft_c_source: str = "",
    ) -> ChemGraph:
        flags = self._mode_flags(mode)
        if not flags["feedback"] or soft_c is None or not self.config.enabled:
            self._last_diag = N21Diagnostics(
                mode=mode or self.config.mode,
                g_noise=float(noise_gate_value(t_fraction, self.config.noise_gate).reshape(-1)[0]),
                feedback_enabled=False,
                soft_c_source=soft_c_source or "none",
            )
            return self.denoiser(chemgraph, t, soft_c_feedback=None)
        scf = self._build_feedback_state(
            soft_c=soft_c,
            t_fraction=t_fraction,
            mode=mode or self.config.mode,
            soft_c_source=soft_c_source,
        )
        return self.denoiser(chemgraph, t, soft_c_feedback=scf)
