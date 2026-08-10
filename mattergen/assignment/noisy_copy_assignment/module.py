"""N1 module: frozen pretrained molecular-CSP GemNet hiddens → O2 structured assignment.

Default hidden source is GemNetTDenoiser.node_embeddings (epoch294 le50 molCSP).
ContextCrystalEncoder is retained only as an explicit ablation
(``hidden_source=context_encoder``) and must never be a silent fallback.

Observational branch only: never modifies geometry score tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mattergen.assignment.global_copy_assembly.orbit_attachment import (
    attachment_map_margins,
)
from mattergen.assignment.global_copy_assembly.orbit_membership import (
    OrbitPartition,
    validate_orbit_copy_capacity,
)
from mattergen.assignment.global_copy_assembly.orbit_module import (
    OrbitAwareAssemblyConfig,
    OrbitAwareCopyAssembly,
)
from mattergen.assignment.global_copy_assembly.orbit_targets import (
    OrbitAwareAssemblyTarget,
)
from mattergen.assignment.global_copy_assembly.singleton_backbone import SingletonBackbone
from mattergen.common.role_partition_diffusion.oracle_partition import ContextCrystalEncoder
from mattergen.denoiser import GemNetTDenoiser
from mattergen.diffusion.model_utils import NoiseLevelEncoding

from .gemnet_loader import (
    GemNetHiddenExtractor,
    GemNetHiddenOutput,
    build_mol_conditioning_from_sample,
    count_params,
    freeze_module,
    parameter_sha256,
)
from .orbit_capacity import labels_to_bar_r, orbit_capacity_map
from .soft_c import SOFT_C_KIND, SOFT_C_SEMANTICS, soft_c_from_singleton_map_and_attachment


@dataclass
class NoisyCopyAssignmentConfig:
    enabled: bool = True
    geometry_feedback: bool = False  # must stay False in N1
    freeze_gemnet_backbone: bool = True
    noise_source: str = "mattergen_native"
    # PRIMARY_N1 default: frozen pretrained GemNet node embeddings.
    # Ablation only: context_encoder (standalone, not pretrained mol-CSP).
    hidden_source: str = "gemnet"  # gemnet | context_encoder
    orbit_mode: str = "oracle_orbit"  # oracle_orbit | predicted_orbit
    produce_soft_c: bool = True
    hidden_dim: int = 256  # assignment-head width (GemNet may be 512 → projected)
    crystal_num_layers: int = 4
    pair_hidden_dim: int = 256
    distance_rbf_dim: int = 32
    pair_score_scale: float = 20.0
    orbit_weight: float = 1.0
    singleton_tree_weight: float = 1.0
    orbit_attachment_weight: float = 1.0
    pair_aux_weight: float = 0.1
    gauge_marginalization: str = "logsumexp"
    virtual_bond_type_id: int = 7
    use_copy_id_as_input: bool = False
    use_oracle_C_as_input: bool = False
    use_oracle_role_assignment: bool = False
    limit_density: float = 0.05
    # Fail loudly if gemnet requested but not injected / load fails.
    fail_on_gemnet_fallback: bool = True
    # Soft C naming for provenance.
    soft_c_semantics: str = SOFT_C_SEMANTICS


@dataclass
class AssignmentOutput:
    orbit_logits: torch.Tensor | None
    orbit_map: torch.Tensor  # bar_R [N,J]
    group_map: torch.Tensor  # G [N,K]
    c_map: torch.Tensor
    # conditional-on-singleton-MAP structured soft C (not full-joint soft C)
    c_soft: torch.Tensor | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TimestepConditionedCrystalEncoder(nn.Module):
    """Standalone context encoder + timestep fusion (ablation only).

    Not used when ``hidden_source=gemnet``.  Does not load MatterGen weights.
    """

    def __init__(self, hidden: int = 256, layers: int = 4):
        super().__init__()
        self.crystal = ContextCrystalEncoder(hidden=hidden, layers=layers, rbf_dim=64)
        self.t_enc = NoiseLevelEncoding(hidden)
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward_geometry_probe(self, z, frac, cell, t: torch.Tensor) -> torch.Tensor:
        hx = self.crystal(z, frac, cell, context_mode="geometry_only")
        te = self.t_enc(t.reshape(1).to(hx.device)).expand(hx.shape[0], -1)
        return self.fuse(torch.cat([hx, te], dim=-1)).sum()

    def extract_atom_hidden(
        self,
        *,
        z: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        t: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del atomic_numbers
        hx = self.crystal(z, frac, cell, context_mode="geometry_only")
        te = self.t_enc(t.reshape(1).to(hx.device)).expand(hx.shape[0], -1)
        return self.fuse(torch.cat([hx, te], dim=-1))


class OrbitLogitHead(nn.Module):
    def __init__(self, hidden: int, num_orbits: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_orbits),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class NoisyCopyAssignmentN1(nn.Module):
    """N1 observational assignment branch on noisy MatterGen geometry features."""

    def __init__(self, config: NoisyCopyAssignmentConfig, partition: OrbitPartition):
        super().__init__()
        if config.geometry_feedback:
            raise ValueError("N1 forbids geometry_feedback=true")
        if config.use_copy_id_as_input or config.use_oracle_C_as_input:
            raise ValueError("N1 forbids copy/C0 as model input")
        if config.hidden_source not in {"gemnet", "context_encoder"}:
            raise ValueError(
                f"hidden_source must be 'gemnet' or 'context_encoder', got {config.hidden_source!r}"
            )
        self.config = config
        self.partition = partition
        h = config.hidden_dim

        # GemNet path (primary): injected after construction via set_gemnet_denoiser.
        self._gemnet_extractor: GemNetHiddenExtractor | None = None
        self._gemnet_denoiser: GemNetTDenoiser | None = None
        self._last_hidden_meta: dict[str, Any] = {}
        self._chemgraph_extra: dict[str, Any] | None = None
        # Trainable projection GemNet-H → assignment-H (created on inject if dims differ).
        self.gemnet_proj: nn.Module | None = None

        # Ablation-only context encoder. Never used as silent fallback for gemnet.
        if config.hidden_source == "context_encoder":
            self.context_backbone: TimestepConditionedCrystalEncoder | None = (
                TimestepConditionedCrystalEncoder(hidden=h, layers=config.crystal_num_layers)
            )
        else:
            self.context_backbone = None

        self.orbit_head = OrbitLogitHead(h, partition.J)
        o2_cfg = OrbitAwareAssemblyConfig(
            crystal_hidden_dim=h,
            molecular_hidden_dim=h,
            pair_hidden_dim=config.pair_hidden_dim,
            distance_rbf_dim=config.distance_rbf_dim,
            pair_score_scale=config.pair_score_scale,
            gauge_marginalization=config.gauge_marginalization,
            virtual_bond_type_id=config.virtual_bond_type_id,
            use_copy_id_as_input=False,
            use_oracle_role_assignment=False,
        )
        self.o2 = OrbitAwareCopyAssembly(o2_cfg)
        # Drop O2 crystal encoder from optimization when using GemNet features:
        # we override hx via _inject_hx.  Freeze O2's unused crystal encoder always
        # when primary path is gemnet.
        self.mol_encoder = self.o2.molecule_encoder
        self.pair_potential = self.o2.pair_potential
        self.orbit_attach = self.o2.orbit_head
        self.path_length_embedding = self.o2.path_length_embedding
        if config.hidden_source == "gemnet":
            freeze_module(self.o2.crystal_encoder)
        if config.freeze_gemnet_backbone and self.context_backbone is not None:
            freeze_module(self.context_backbone)

    # ------------------------------------------------------------------ DI API
    def set_gemnet_denoiser(
        self,
        denoiser: GemNetTDenoiser | nn.Module,
        *,
        freeze: bool | None = None,
        chemgraph_extra: dict | None = None,
    ) -> GemNetHiddenExtractor:
        """Dependency-inject pretrained GemNetTDenoiser (replaces global hack).

        Equivalent formal API to the prior ``set_gemnet_denoiser`` on the
        context encoder; lives on the N1 module so trainers do not rely on
        module-global state.
        """
        if not isinstance(denoiser, GemNetTDenoiser):
            # Allow duck-typed denoisers that expose the GemNetTDenoiser forward surface
            # (gemnet + noise_level_encoding) for tests and thin adapters.
            if not isinstance(denoiser, nn.Module) or not (
                hasattr(denoiser, "gemnet") and hasattr(denoiser, "noise_level_encoding")
            ):
                raise TypeError(
                    f"set_gemnet_denoiser expects GemNetTDenoiser (or duck-type with "
                    f"gemnet + noise_level_encoding), got {type(denoiser)!r}"
                )
        freeze = self.config.freeze_gemnet_backbone if freeze is None else freeze
        if freeze:
            freeze_module(denoiser)
            cond = getattr(denoiser, "molecule_conditioner", None)
            if isinstance(cond, nn.Module):
                freeze_module(cond)
        self._gemnet_denoiser = denoiser  # type: ignore[assignment]
        self._gemnet_extractor = GemNetHiddenExtractor(denoiser)  # type: ignore[arg-type]
        gdim = int(getattr(denoiser, "hidden_dim", self._gemnet_extractor.hidden_dim))
        adim = int(self.config.hidden_dim)
        if gdim != adim:
            self.gemnet_proj = nn.Sequential(
                nn.Linear(gdim, adim),
                nn.SiLU(),
                nn.Linear(adim, adim),
            )
        else:
            self.gemnet_proj = nn.Identity()
        if chemgraph_extra is not None:
            self._chemgraph_extra = chemgraph_extra
        return self._gemnet_extractor

    def set_chemgraph_extra(self, extra: dict | None) -> None:
        self._chemgraph_extra = extra

    def prepare_mol_conditioning_from_sample(self, sample: dict) -> dict[str, torch.Tensor]:
        extra = build_mol_conditioning_from_sample(sample)
        self._chemgraph_extra = extra
        return extra

    def freeze_backbone(self) -> None:
        if self._gemnet_denoiser is not None:
            freeze_module(self._gemnet_denoiser)
            cond = getattr(self._gemnet_denoiser, "molecule_conditioner", None)
            if isinstance(cond, nn.Module):
                freeze_module(cond)
        if self._gemnet_extractor is not None:
            freeze_module(self._gemnet_extractor.denoiser)
        if self.context_backbone is not None:
            freeze_module(self.context_backbone)
        if self.config.hidden_source == "gemnet":
            freeze_module(self.o2.crystal_encoder)

    def gemnet_parameter_hash(self) -> str | None:
        if self._gemnet_denoiser is None:
            return None
        return parameter_sha256(self._gemnet_denoiser)

    def assignment_parameter_hash(self) -> str:
        """Hash only assignment-trainable parameters (heads + proj + mol encoders)."""
        mods: list[nn.Module] = [
            self.orbit_head,
            self.mol_encoder,
            self.pair_potential,
            self.orbit_attach,
            self.path_length_embedding,
            self.o2.virtual_mix,
        ]
        if self.gemnet_proj is not None and not isinstance(self.gemnet_proj, nn.Identity):
            mods.append(self.gemnet_proj)
        if self.config.hidden_source == "context_encoder" and self.context_backbone is not None:
            if any(p.requires_grad for p in self.context_backbone.parameters()):
                mods.append(self.context_backbone)
        return parameter_sha256(nn.ModuleList(mods))

    def trainable_assignment_parameters(self):
        """Explicit generator of parameters that the N1 optimizer may update.

        Never yields GemNet / pretrained molecule_conditioner parameters.
        """
        yielded = set()

        def _yield_from(module: nn.Module | None, *, require_grad: bool = True):
            if module is None:
                return
            for p in module.parameters():
                if require_grad and not p.requires_grad:
                    continue
                pid = id(p)
                if pid in yielded:
                    continue
                yielded.add(pid)
                yield p

        # Never train GemNet backbone
        gemnet_ids = set()
        if self._gemnet_denoiser is not None:
            gemnet_ids = {id(p) for p in self._gemnet_denoiser.parameters()}

        for p in _yield_from(self.orbit_head):
            yield p
        for p in _yield_from(self.gemnet_proj):
            if id(p) not in gemnet_ids:
                yield p
        for p in _yield_from(self.mol_encoder):
            yield p
        for p in _yield_from(self.pair_potential):
            yield p
        for p in _yield_from(self.orbit_attach):
            yield p
        for p in _yield_from(self.path_length_embedding):
            yield p
        for p in _yield_from(self.o2.virtual_mix):
            yield p
        # context encoder ablation: only if not frozen and selected
        if (
            self.config.hidden_source == "context_encoder"
            and self.context_backbone is not None
            and not self.config.freeze_gemnet_backbone
        ):
            for p in _yield_from(self.context_backbone):
                yield p

    def param_audit(self) -> dict[str, Any]:
        gemnet_trainable = 0
        gemnet_total = 0
        if self._gemnet_denoiser is not None:
            c = count_params(self._gemnet_denoiser)
            gemnet_trainable = c["trainable_params"]
            gemnet_total = c["total_params"]
        assign = list(self.trainable_assignment_parameters())
        assign_n = sum(p.numel() for p in assign)
        return {
            "gemnet_total_params": gemnet_total,
            "gemnet_trainable_params": gemnet_trainable,
            "assignment_trainable_params": assign_n,
            "hidden_source": self.config.hidden_source,
            "gemnet_injected": self._gemnet_extractor is not None,
            "context_encoder_present": self.context_backbone is not None,
            "geometry_feedback": self.config.geometry_feedback,
            "freeze_gemnet_backbone": self.config.freeze_gemnet_backbone,
        }

    def _require_gemnet(self) -> GemNetHiddenExtractor:
        if self.config.hidden_source != "gemnet":
            raise RuntimeError("_require_gemnet called with hidden_source!=gemnet")
        if self._gemnet_extractor is None:
            raise RuntimeError(
                "HIDDEN_SOURCE=gemnet but GemNetTDenoiser was not injected. "
                "Call set_gemnet_denoiser(...) after loading MatterGenCheckpointInfo "
                f"(load_epoch). fail_on_gemnet_fallback={self.config.fail_on_gemnet_fallback}. "
                "No ContextCrystalEncoder fallback."
            )
        return self._gemnet_extractor

    def extract_atom_hidden(
        self,
        *,
        z: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        t: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
        chemgraph_extra: dict | None = None,
    ) -> torch.Tensor:
        """Return per-atom hidden [N, H_assignment] from configured source.

        For gemnet: GemNetTDenoiser.gemnet(...).node_embeddings → optional proj.
        Fails loudly if gemnet required but missing (no silent fallback).
        """
        atom_z = atomic_numbers if atomic_numbers is not None else z
        if self.config.hidden_source == "gemnet":
            extractor = self._require_gemnet()
            extra = chemgraph_extra if chemgraph_extra is not None else self._chemgraph_extra
            out: GemNetHiddenOutput = extractor.extract(
                frac=frac,
                cell=cell,
                atomic_numbers=atom_z.long(),
                t=t,
                chemgraph_extra=extra,
                detach=True,
            )
            self._last_hidden_meta = dict(out.metadata)
            h = out.node_embeddings
            if self.gemnet_proj is None:
                # Dims may match after inject; if not injected proj, identity only if dims equal
                if h.shape[-1] != self.config.hidden_dim:
                    raise RuntimeError(
                        f"GemNet hidden dim {h.shape[-1]} != assignment hidden_dim "
                        f"{self.config.hidden_dim} and gemnet_proj is None. "
                        "Call set_gemnet_denoiser before forward."
                    )
                return h
            return self.gemnet_proj(h)

        # Explicit ablation path only.
        if self.context_backbone is None:
            raise RuntimeError(
                "hidden_source=context_encoder but context_backbone was not built"
            )
        h = self.context_backbone.extract_atom_hidden(
            z=z, frac=frac, cell=cell, t=t, atomic_numbers=atom_z
        )
        self._last_hidden_meta = {
            "hidden_source": "context_crystal_encoder",
            "context_crystal_encoder_used": True,
            "timestep_conditioning": True,
            "hidden_dim": int(h.shape[-1]),
            "num_atoms": int(h.shape[0]),
        }
        return h

    def _inject_hx(self, hx: torch.Tensor):
        """Monkey-patch O2 encode to return (hx, hm) with our features."""

        def encode(**kwargs):
            hm = self.mol_encoder(
                kwargs["role_z"], kwargs["role_edge_index"], kwargs["role_bond_type"]
            )
            return hx, hm

        self.o2.encode = encode  # type: ignore[method-assign]

    def geometry_probe(self, z, frac, cell, t) -> torch.Tensor:
        """Scalar probe used to assert assignment does not alter geometry branch.

        With gemnet: sum of node embeddings from frozen extractor (assignment heads
        do not enter). With context encoder: frozen context probe.
        """
        if self.config.hidden_source == "gemnet":
            extractor = self._require_gemnet()
            out = extractor.extract(
                frac=frac,
                cell=cell,
                atomic_numbers=z.long(),
                t=t,
                chemgraph_extra=self._chemgraph_extra,
                detach=True,
            )
            return out.node_embeddings.sum()
        assert self.context_backbone is not None
        return self.context_backbone.forward_geometry_probe(z, frac, cell, t)

    def predict_orbit_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.orbit_head(h)

    def resolve_bar_r(
        self,
        *,
        h: torch.Tensor,
        K: int,
        oracle_bar_r: torch.Tensor | None,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        mode = mode or self.config.orbit_mode
        if mode == "oracle_orbit":
            if oracle_bar_r is None:
                raise ValueError("oracle_orbit mode requires oracle_bar_r")
            return oracle_bar_r, None, None
        if mode != "predicted_orbit":
            raise ValueError(mode)
        logits = self.predict_orbit_logits(h)
        labels = orbit_capacity_map(logits, self.partition, K=K)
        bar = labels_to_bar_r(labels, self.partition.J)
        return bar, logits, labels

    def loss(
        self,
        *,
        o2_target: OrbitAwareAssemblyTarget,
        backbone_tree: SingletonBackbone,
        z: torch.Tensor,
        frac_t: torch.Tensor,
        cell_t: torch.Tensor,
        t: torch.Tensor,
        role_z: torch.Tensor,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
        oracle_bar_r: torch.Tensor,
        atomic_numbers: torch.Tensor | None = None,
        orbit_mode: str | None = None,
        chemgraph_extra: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.config.use_copy_id_as_input or self.config.use_oracle_C_as_input:
            raise RuntimeError("oracle copy leakage")
        h = self.extract_atom_hidden(
            z=z,
            frac=frac_t,
            cell=cell_t,
            t=t,
            atomic_numbers=atomic_numbers,
            chemgraph_extra=chemgraph_extra,
        )
        mode = orbit_mode or self.config.orbit_mode
        bar, logits, _ = self.resolve_bar_r(
            h=h, K=o2_target.K, oracle_bar_r=oracle_bar_r, mode=mode
        )
        del bar
        oracle_labels = oracle_bar_r.argmax(-1)
        orbit_loss = h.new_zeros(())
        if logits is not None:
            orbit_loss = nn.functional.cross_entropy(logits, oracle_labels)
        elif mode == "oracle_orbit":
            orbit_loss = h.new_zeros(())

        self._inject_hx(h)
        values = self.o2.loss(
            o2_target=o2_target,
            backbone=backbone_tree,
            z=z,
            frac=frac_t,
            cell=cell_t if cell_t.ndim == 2 else cell_t.squeeze(0),
            role_z=role_z,
            role_edge_index=role_edge_index,
            role_bond_type=role_bond_type,
        )
        total = (
            self.config.orbit_weight * orbit_loss
            + self.config.singleton_tree_weight * values["tree_nll"]
            + self.config.orbit_attachment_weight * values["orbit_nll"]
            + self.config.pair_aux_weight * values["pair_ce"]
        )
        return {
            "loss": total,
            "orbit_loss": orbit_loss.detach() if torch.is_tensor(orbit_loss) else orbit_loss,
            "singleton_nll": values["tree_nll"],
            "orbit_attachment_nll": values["orbit_nll"],
            "pair_ce": values["pair_ce"],
            "total_assignment_loss": total,
        }

    @torch.no_grad()
    def map_decode(
        self,
        *,
        o2_target: OrbitAwareAssemblyTarget,
        backbone_tree: SingletonBackbone,
        z: torch.Tensor,
        frac_t: torch.Tensor,
        cell_t: torch.Tensor,
        t: torch.Tensor,
        role_z: torch.Tensor,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
        oracle_bar_r: torch.Tensor | None,
        atomic_numbers: torch.Tensor | None = None,
        orbit_mode: str | None = None,
        produce_soft_c: bool | None = None,
        chemgraph_extra: dict | None = None,
    ) -> AssignmentOutput:
        produce_soft = self.config.produce_soft_c if produce_soft_c is None else produce_soft_c
        h = self.extract_atom_hidden(
            z=z,
            frac=frac_t,
            cell=cell_t,
            t=t,
            atomic_numbers=atomic_numbers,
            chemgraph_extra=chemgraph_extra,
        )
        mode = orbit_mode or self.config.orbit_mode
        bar, logits, pred_labels = self.resolve_bar_r(
            h=h, K=o2_target.K, oracle_bar_r=oracle_bar_r, mode=mode
        )
        self._inject_hx(h)
        cell = cell_t if cell_t.ndim == 2 else cell_t.squeeze(0)
        decoded = self.o2.map_decode(
            o2_target=o2_target,
            backbone=backbone_tree,
            z=z,
            frac=frac_t,
            cell=cell,
            role_z=role_z,
            role_edge_index=role_edge_index,
            role_bond_type=role_bond_type,
        )
        G = decoded["G"]
        C = decoded["C"]
        c_soft = None
        soft_diag: dict[str, Any] = {}
        if produce_soft and o2_target.orbit_targets:
            ot = o2_target.orbit_targets[0]
            singleton_mask = torch.zeros(o2_target.N, dtype=torch.bool, device=G.device)
            for nodes in o2_target.singleton_target.role_sets.values():
                singleton_mask[nodes] = True
            G_sing = G.clone()
            G_sing[~singleton_mask] = 0
            F = self.o2.build_attachment_scores_from_G(
                orbit_atoms=ot.atom_indices.to(G.device),
                G_singleton=G_sing,
                hx=h,
                frac=frac_t,
                cell=cell,
                singleton_mask=singleton_mask,
                orbit_id=ot.orbit_index,
                mode="max",
            )
            c_soft = soft_c_from_singleton_map_and_attachment(
                G_singleton=G_sing,
                singleton_mask=singleton_mask,
                orbit_atoms=ot.atom_indices.to(G.device),
                F_attach=F,
                atoms_per_copy=ot.atoms_per_copy,
            )
            margins = attachment_map_margins(
                F, atoms_per_copy=ot.atoms_per_copy, target_pairs=list(ot.pairs_local)
            )
            soft_diag["orbit_attachment_margins"] = margins
            soft_diag["soft_C_kind"] = SOFT_C_KIND
            soft_diag["soft_c_semantics"] = SOFT_C_SEMANTICS
            soft_diag["soft_C_note"] = (
                "c_soft is conditional-on-singleton-MAP structured soft C: "
                "singleton groups fixed at MAP; orbit attachments Boltzmann-averaged "
                "under that backbone. Singleton-tree uncertainty is NOT fully "
                "marginalized. Not full joint structured P(g_i=g_j)."
            )
        diag = {
            "status": "NOISY_COPY_ASSIGNMENT_N1",
            "n1_mode": "observational_noisy_copy_assignment",
            "orbit_mode": mode,
            "hidden_source": self._last_hidden_meta.get(
                "hidden_source",
                "gemnet_node_embeddings"
                if self.config.hidden_source == "gemnet"
                else "context_crystal_encoder",
            ),
            "hidden_meta": dict(self._last_hidden_meta),
            "context_crystal_encoder_used": self.config.hidden_source == "context_encoder",
            "singleton_tree_energy": decoded.get("singleton_tree_energy"),
            "orbit_attachments": decoded.get("orbit_attachments"),
            "orbit_copy_capacity": validate_orbit_copy_capacity(G, bar.to(G.device), self.partition),
            "use_copy_id_as_input": False,
            "use_oracle_C_as_input": False,
            "geometry_feedback": False,
            "freeze_gemnet_backbone": self.config.freeze_gemnet_backbone,
            "soft_c_semantics": SOFT_C_SEMANTICS,
            **soft_diag,
        }
        if pred_labels is not None:
            diag["orbit_map_labels"] = pred_labels.detach().cpu().tolist()
        return AssignmentOutput(
            orbit_logits=logits.detach() if logits is not None else None,
            orbit_map=bar.detach(),
            group_map=G.detach(),
            c_map=C.detach(),
            c_soft=c_soft.detach() if c_soft is not None else None,
            diagnostics=diag,
        )
