"""N2: confidence-gated soft-C feedback into MatterGen geometry denoising.

Pass A — frozen unconditioned GemNet + frozen N1 → soft C_t (stop-grad)
Pass B — same (X_t,L_t) + soft C_t → GemNet + N2 adapters → geometry scores

No independent G/C diffusion trajectory (geometry-induced assignment only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from mattergen.assignment.global_copy_assembly.orbit_membership import (
    OrbitPartition,
    build_orbit_partition,
    collapse_roles_to_orbit_membership,
)
from mattergen.assignment.global_copy_assembly.orbit_module import prepare_backbone
from mattergen.assignment.global_copy_assembly.orbit_targets import build_orbit_aware_target
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    freeze_module,
    load_molecular_csp_gemnet,
    parameter_sha256,
)
from mattergen.assignment.noisy_copy_assignment.module import (
    NoisyCopyAssignmentConfig,
    NoisyCopyAssignmentN1,
)
from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.denoiser import GemNetTDenoiser

from .adapters import SoftCEdgeAdapter, SoftCGroupAdapter
from .feedback import (
    build_edge_soft_c_channels,
    expected_same_copy_representation,
    shuffle_soft_c,
)
from .gates import NoiseGateConfig, noise_gate_value

N2Mode = Literal[
    "B0_baseline",
    "B1_no_feedback",
    "B2_combined",
    "B3_edge_only",
    "B4_group_only",
    "B5_shuffled_c",
    "B6_oracle_c",
]


@dataclass
class SoftCFeedbackConfig:
    enabled: bool = True
    edge_semantics_enabled: bool = True
    group_context_enabled: bool = True
    pairwise_confidence_enabled: bool = True
    noise_gate: NoiseGateConfig = field(default_factory=NoiseGateConfig)
    edge_bottleneck: int = 64
    group_bottleneck: int = 128
    freeze_base_gemnet: bool = True
    freeze_n1: bool = True
    soft_c_semantics: str = SOFT_C_SEMANTICS
    molecules_atoms_m: int = 10  # RHODIN01 M
    # Modes control which channels fire
    mode: N2Mode = "B2_combined"


@dataclass
class N2ForwardDiagnostics:
    mode: str
    soft_c_semantics: str
    g_noise: float
    edge_audit: dict[str, Any] = field(default_factory=dict)
    group_audit: dict[str, Any] = field(default_factory=dict)
    feedback_enabled: bool = False
    edge_enabled: bool = False
    group_enabled: bool = False
    assignment_trajectory: str = "geometry_induced_inference"
    g_diffusion: bool = False


class SoftCGeometryFeedbackN2(nn.Module):
    """Two-pass soft-C geometry feedback model."""

    def __init__(
        self,
        *,
        denoiser: GemNetTDenoiser,
        n1_model: NoisyCopyAssignmentN1,
        partition: OrbitPartition,
        config: SoftCFeedbackConfig | None = None,
        mattergen_provenance: dict | None = None,
        n1_provenance: dict | None = None,
    ):
        super().__init__()
        self.config = config or SoftCFeedbackConfig()
        self.partition = partition
        self.denoiser = denoiser
        self.n1 = n1_model
        self.mattergen_provenance = dict(mattergen_provenance or {})
        self.n1_provenance = dict(n1_provenance or {})

        if self.config.freeze_base_gemnet:
            freeze_module(self.denoiser)
        if self.config.freeze_n1:
            freeze_module(self.n1)

        # After angle_edge_emb, edge reps have emb_size_edge (== hidden_dim for molCSP).
        emb_edge = int(getattr(self.denoiser.gemnet, "emb_size_edge", self.denoiser.hidden_dim))
        # Infer from angle_edge_emb last Linear if available
        ae = getattr(self.denoiser.gemnet, "angle_edge_emb", None)
        if ae is not None:
            last = list(ae.modules())[-1]
            if isinstance(last, nn.Linear):
                emb_edge = int(last.out_features)
        emb_atom = int(getattr(self.denoiser.gemnet, "emb_size_atom", self.denoiser.hidden_dim))
        self.edge_adapter = SoftCEdgeAdapter(
            emb_size_edge=emb_edge,
            bottleneck=self.config.edge_bottleneck,
        )
        self.group_adapter = SoftCGroupAdapter(
            hidden=emb_atom,
            bottleneck=self.config.group_bottleneck,
        )
        self._emb_atom = emb_atom
        # Pass-A extractor reuses N1's injected gemnet path
        self._last_diag: N2ForwardDiagnostics | None = None

    # ------------------------------------------------------------------ params
    def trainable_n2_parameters(self):
        for p in self.edge_adapter.parameters():
            if p.requires_grad:
                yield p
        for p in self.group_adapter.parameters():
            if p.requires_grad:
                yield p

    def param_audit(self) -> dict[str, int]:
        base = sum(p.numel() for p in self.denoiser.parameters())
        base_tr = sum(p.numel() for p in self.denoiser.parameters() if p.requires_grad)
        n1_tr = sum(p.numel() for p in self.n1.parameters() if p.requires_grad)
        n2_tr = sum(p.numel() for p in self.trainable_n2_parameters())
        return {
            "base_gemnet_total_params": int(base),
            "trainable_base_gemnet_params": int(base_tr),
            "trainable_n1_params": int(n1_tr),
            "trainable_n2_params": int(n2_tr),
        }

    def n1_parameter_hash(self) -> str:
        return parameter_sha256(self.n1)

    def gemnet_parameter_hash(self) -> str:
        return parameter_sha256(self.denoiser)

    def n2_parameter_hash(self) -> str:
        return parameter_sha256(nn.ModuleList([self.edge_adapter, self.group_adapter]))

    # ------------------------------------------------------------------ modes
    def _mode_flags(self, mode: N2Mode | None = None) -> dict[str, bool]:
        mode = mode or self.config.mode
        edge = self.config.edge_semantics_enabled
        group = self.config.group_context_enabled
        feedback = self.config.enabled
        if mode == "B0_baseline":
            return dict(feedback=False, edge=False, group=False, use_n1=False, shuffle=False, oracle=False)
        if mode == "B1_no_feedback":
            return dict(feedback=False, edge=False, group=False, use_n1=True, shuffle=False, oracle=False)
        if mode == "B2_combined":
            return dict(feedback=feedback, edge=edge, group=group, use_n1=True, shuffle=False, oracle=False)
        if mode == "B3_edge_only":
            return dict(feedback=feedback, edge=True, group=False, use_n1=True, shuffle=False, oracle=False)
        if mode == "B4_group_only":
            return dict(feedback=feedback, edge=False, group=True, use_n1=True, shuffle=False, oracle=False)
        if mode == "B5_shuffled_c":
            return dict(feedback=feedback, edge=edge, group=group, use_n1=True, shuffle=True, oracle=False)
        if mode == "B6_oracle_c":
            return dict(
                feedback=feedback,
                edge=edge,
                group=group,
                use_n1=False,
                shuffle=False,
                oracle=True,
            )
        raise ValueError(mode)

    # ------------------------------------------------------------------ Pass A
    @torch.no_grad()
    def pass_a_assignment(
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
        oracle_c: torch.Tensor | None = None,
        mode: N2Mode | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, dict]:
        """Frozen unconditioned GemNet + N1 → soft C (detached). Also returns H^A."""
        flags = self._mode_flags(mode)
        # Always extract H^A without feedback
        h_a_out = self.n1.extract_atom_hidden(
            z=sample["z"],
            frac=frac_t,
            cell=cell_t,
            t=t,
            atomic_numbers=sample["z"],
        )
        # N1 extract already applies gemnet_proj; for group context we need
        # full GemNet hidden dim. Re-extract raw node embeddings from extractor.
        h_a_raw = None
        if self.n1._gemnet_extractor is not None:
            raw = self.n1._gemnet_extractor.extract(
                frac=frac_t,
                cell=cell_t,
                atomic_numbers=sample["z"].long(),
                t=t,
                chemgraph_extra=self.n1._chemgraph_extra,
                detach=True,
            )
            h_a_raw = raw.node_embeddings  # [N, 512]

        soft_c = None
        meta: dict[str, Any] = {"mode": mode or self.config.mode, "flags": flags}
        if flags["oracle"]:
            if oracle_c is None:
                copy = sample["copy"]
                soft_c = copy[:, None].eq(copy[None, :]).float()
            else:
                soft_c = oracle_c.float()
            meta["soft_c_source"] = "ORACLE_UPPER_BOUND_ONLY"
        elif flags["use_n1"]:
            # B1 observation and all feedback modes that need soft C
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
            meta["n1_diag"] = {
                k: out.diagnostics.get(k)
                for k in (
                    "soft_c_semantics",
                    "soft_C_kind",
                    "hidden_source",
                    "orbit_mode",
                )
                if k in (out.diagnostics or {})
            }
        else:
            meta["soft_c_source"] = "none_B0_baseline"
        if soft_c is not None:
            soft_c = soft_c.detach()
            if flags["shuffle"]:
                soft_c = shuffle_soft_c(soft_c)
                meta["soft_c_source"] = "shuffled_n1"
            soft_c = soft_c.clamp(0.0, 1.0)
            soft_c.fill_diagonal_(1.0)
        h_ret = h_a_raw if h_a_raw is not None else h_a_out.detach()
        return soft_c, h_ret, meta

    # ------------------------------------------------------------------ Pass B
    def _build_soft_c_feedback_state(
        self,
        *,
        soft_c: torch.Tensor | None,
        h_a: torch.Tensor,
        t_fraction: float,
        flags: dict[str, bool],
    ) -> tuple[dict | None, N2ForwardDiagnostics]:
        g_noise = float(noise_gate_value(t_fraction, self.config.noise_gate).reshape(-1)[0].item())
        diag = N2ForwardDiagnostics(
            mode=self.config.mode,
            soft_c_semantics=self.config.soft_c_semantics,
            g_noise=g_noise,
            feedback_enabled=bool(flags["feedback"]),
            edge_enabled=bool(flags["edge"] and flags["feedback"]),
            group_enabled=bool(flags["group"] and flags["feedback"]),
        )
        if not flags["feedback"] or soft_c is None:
            self._last_diag = diag
            return None, diag

        # Precompute group node residual (independent of edge graph)
        node_delta = None
        if flags["group"]:
            h_group, g_group, mass, gmeta = expected_same_copy_representation(
                h_a=h_a,
                soft_c=soft_c,
                t_fraction=t_fraction,
                noise_gate_cfg=self.config.noise_gate,
                molecules_atoms_m=self.config.molecules_atoms_m,
                pairwise_confidence_enabled=self.config.pairwise_confidence_enabled,
            )
            # Project h_a (512) may match group adapter; ensure dims
            ha = h_a
            if ha.shape[-1] != self.group_adapter.net[-1].out_features:
                # If H^A dim differs, use identity pad/project — should match gemnet hidden
                pass
            delta = self.group_adapter(ha, h_group, g_group)
            node_delta = g_group.unsqueeze(-1) * delta
            diag.group_audit = gmeta

        # Edge adapter closure captures soft_c; channels built with actual edge_index
        edge_adapter_fn = None
        if flags["edge"]:

            def edge_adapter_fn(m, edge_index, cell_offsets, soft_c=soft_c, t_fraction=t_fraction):
                c_intra, c_inter, c_signed, audit = build_edge_soft_c_channels(
                    soft_c=soft_c,
                    edge_index=edge_index,
                    cell_offsets=cell_offsets,
                    t_fraction=t_fraction,
                    noise_gate_cfg=self.config.noise_gate,
                    pairwise_confidence_enabled=self.config.pairwise_confidence_enabled,
                )
                diag.edge_audit = audit.as_dict()
                return self.edge_adapter(m, c_intra, c_inter, c_signed)

        state = {
            "enabled": True,
            "node_delta": node_delta,
            "edge_adapter": edge_adapter_fn,
        }
        self._last_diag = diag
        return state, diag

    def forward_geometry(
        self,
        *,
        chemgraph: ChemGraph,
        t: torch.Tensor,
        soft_c: torch.Tensor | None,
        h_a: torch.Tensor | None,
        t_fraction: float,
        mode: N2Mode | None = None,
    ) -> ChemGraph:
        """Pass B (or baseline): geometry scores with optional soft-C feedback."""
        flags = self._mode_flags(mode)
        if not flags["feedback"] or soft_c is None or h_a is None:
            scf = None
            self._last_diag = N2ForwardDiagnostics(
                mode=mode or self.config.mode,
                soft_c_semantics=self.config.soft_c_semantics,
                g_noise=float(noise_gate_value(t_fraction, self.config.noise_gate).reshape(-1)[0]),
                feedback_enabled=False,
            )
        else:
            scf, _ = self._build_soft_c_feedback_state(
                soft_c=soft_c, h_a=h_a, t_fraction=t_fraction, flags=flags
            )
        # Gradients flow into adapters only (base frozen)
        return self.denoiser(chemgraph, t, soft_c_feedback=scf)

    def geometry_scores_dict(self, chemgraph_out: ChemGraph) -> dict[str, torch.Tensor]:
        return {
            "pos": chemgraph_out["pos"],
            "cell": chemgraph_out["cell"],
            "atomic_numbers": chemgraph_out["atomic_numbers"],
        }
