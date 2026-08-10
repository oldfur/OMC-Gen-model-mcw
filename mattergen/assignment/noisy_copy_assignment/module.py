"""N1 module: GemNet (or frozen crystal encoder) features → O2 structured assignment.

Observational branch only: never modifies geometry score tensors.
Default freezes the feature backbone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mattergen.assignment.global_copy_assembly.orbit_attachment import (
    OrbitAttachmentHead,
    attachment_map_margins,
    balanced_attachment_dp,
)
from mattergen.assignment.global_copy_assembly.orbit_membership import (
    OrbitPartition,
    collapse_roles_to_orbit_membership,
    validate_orbit_copy_capacity,
)
from mattergen.assignment.global_copy_assembly.orbit_module import (
    OrbitAwareAssemblyConfig,
    OrbitAwareCopyAssembly,
    prepare_backbone,
)
from mattergen.assignment.global_copy_assembly.orbit_targets import (
    OrbitAwareAssemblyTarget,
    build_orbit_aware_target,
)
from mattergen.assignment.global_copy_assembly.pair_potential import BondPairPotential, permutation_factor
from mattergen.assignment.global_copy_assembly.permutations import (
    enumerate_permutations,
    identity_index,
    inverse_permutations,
)
from mattergen.assignment.global_copy_assembly.singleton_backbone import SingletonBackbone, edge_meta_lookup
from mattergen.assignment.global_copy_assembly.targets import target_state_indices
from mattergen.assignment.global_copy_assembly.tree_crf import TreeCRF
from mattergen.common.role_partition_diffusion.oracle_partition import ContextCrystalEncoder
from mattergen.diffusion.model_utils import NoiseLevelEncoding

from .orbit_capacity import labels_to_bar_r, orbit_capacity_map
from .soft_c import SOFT_C_KIND, soft_c_from_singleton_map_and_attachment, soft_c_metrics


@dataclass
class NoisyCopyAssignmentConfig:
    enabled: bool = True
    geometry_feedback: bool = False  # must stay False in N1
    freeze_gemnet_backbone: bool = True
    noise_source: str = "mattergen_native"
    orbit_mode: str = "oracle_orbit"  # oracle_orbit | predicted_orbit
    produce_soft_c: bool = True
    hidden_dim: int = 256
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
    backbone_kind: str = "context_crystal_with_t"  # or "gemnet" when full denoiser injected
    limit_density: float = 0.05


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
    """Frozen-capable crystal encoder + timestep fusion (GemNet stand-in when no MG ckpt).

    When a full ``GemNetTDenoiser`` is attached via ``set_gemnet_denoiser``, atom
    hiddens come from ``gemnet(...).node_embeddings`` (MatterGen path).
    """

    def __init__(self, hidden: int = 256, layers: int = 4):
        super().__init__()
        self.crystal = ContextCrystalEncoder(hidden=hidden, layers=layers, rbf_dim=64)
        self.t_enc = NoiseLevelEncoding(hidden)
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self._gemnet_denoiser: nn.Module | None = None

    def set_gemnet_denoiser(self, denoiser: nn.Module | None) -> None:
        self._gemnet_denoiser = denoiser

    def forward_geometry_probe(self, z, frac, cell, t: torch.Tensor) -> torch.Tensor:
        """Return a fixed probe tensor for geometry-score invariance tests (not used in MG)."""
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
        if self._gemnet_denoiser is not None:
            # MatterGen GemNetTDenoiser path: node embeddings before fc_atom / force heads.
            gemnet = getattr(self._gemnet_denoiser, "gemnet", None)
            noise_enc = getattr(self._gemnet_denoiser, "noise_level_encoding", None)
            if gemnet is None or noise_enc is None:
                raise RuntimeError("attached denoiser missing gemnet / noise_level_encoding")
            n = z.shape[0]
            batch = torch.zeros(n, dtype=torch.long, device=z.device)
            num_atoms = torch.tensor([n], device=z.device, dtype=torch.long)
            atom_types = atomic_numbers if atomic_numbers is not None else z
            lat = cell if cell.ndim == 3 else cell.unsqueeze(0)
            t_enc = noise_enc(t.reshape(1).to(lat.device))
            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in gemnet.parameters())):
                out = gemnet(
                    z=t_enc,
                    frac_coords=frac,
                    atom_types=atom_types.long(),
                    num_atoms=num_atoms,
                    batch=batch,
                    lengths=None,
                    angles=None,
                    lattice=lat,
                    edge_index=None,
                    to_jimages=None,
                    num_bonds=None,
                    node_condition=None,
                )
            return out.node_embeddings
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
        self.config = config
        self.partition = partition
        h = config.hidden_dim
        self.backbone = TimestepConditionedCrystalEncoder(hidden=h, layers=config.crystal_num_layers)
        self.orbit_head = OrbitLogitHead(h, partition.J)
        # Reuse O2 pair/attachment heads; crystal encoder inside O2 is unused for features
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
        # Drop O2 crystal encoder params from optimization when freezing backbone:
        # we override hx via injected features in forward helpers.
        self.mol_encoder = self.o2.molecule_encoder
        self.pair_potential = self.o2.pair_potential
        self.orbit_attach = self.o2.orbit_head
        self.path_length_embedding = self.o2.path_length_embedding
        if config.freeze_gemnet_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def trainable_assignment_parameters(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p

    def _inject_hx(self, hx: torch.Tensor):
        """Monkey-patch O2 encode to return (hx, hm) with our features."""

        def encode(**kwargs):
            hm = self.mol_encoder(
                kwargs["role_z"], kwargs["role_edge_index"], kwargs["role_bond_type"]
            )
            return hx, hm

        self.o2.encode = encode  # type: ignore[method-assign]

    def geometry_probe(self, z, frac, cell, t) -> torch.Tensor:
        """Scalar probe used to assert assignment does not alter geometry branch."""
        return self.backbone.forward_geometry_probe(z, frac, cell, t)

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
    ) -> dict[str, torch.Tensor]:
        if self.config.use_copy_id_as_input or self.config.use_oracle_C_as_input:
            raise RuntimeError("oracle copy leakage")
        h = self.backbone.extract_atom_hidden(
            z=z, frac=frac_t, cell=cell_t, t=t, atomic_numbers=atomic_numbers
        )
        mode = orbit_mode or self.config.orbit_mode
        bar, logits, _ = self.resolve_bar_r(
            h=h, K=o2_target.K, oracle_bar_r=oracle_bar_r, mode=mode
        )
        # Orbit CE uses oracle bar_r labels
        oracle_labels = oracle_bar_r.argmax(-1)
        orbit_loss = h.new_zeros(())
        if logits is not None:
            orbit_loss = nn.functional.cross_entropy(logits, oracle_labels)
        elif mode == "oracle_orbit":
            orbit_loss = h.new_zeros(())

        self._inject_hx(h)
        # When predicted orbit, rebuild singleton/orbit targets from hard bar map
        # For training N1-B, structured target still uses mol_copy supervision via o2_target
        # built externally from hard-R; orbit membership for V sets should match training bar.
        # Use provided o2_target (from oracle collapse or predicted capacity MAP offline).
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
    ) -> AssignmentOutput:
        produce_soft = self.config.produce_soft_c if produce_soft_c is None else produce_soft_c
        h = self.backbone.extract_atom_hidden(
            z=z, frac=frac_t, cell=cell_t, t=t, atomic_numbers=atomic_numbers
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
        soft_diag = {}
        if produce_soft and o2_target.orbit_targets:
            # rebuild F for soft marginals with MAP singleton G
            ot = o2_target.orbit_targets[0]
            singleton_mask = torch.zeros(o2_target.N, dtype=torch.bool, device=G.device)
            for nodes in o2_target.singleton_target.role_sets.values():
                singleton_mask[nodes] = True
            # zero orbit rows of G for singleton-only mask path used in soft C
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
            margins = attachment_map_margins(F, atoms_per_copy=ot.atoms_per_copy, target_pairs=list(ot.pairs_local))
            soft_diag["orbit_attachment_margins"] = margins
            soft_diag["soft_C_kind"] = SOFT_C_KIND
            soft_diag["soft_C_note"] = (
                "c_soft is conditional-on-singleton-MAP structured soft C: "
                "singleton groups fixed at MAP; orbit attachments Boltzmann-averaged "
                "under that backbone. Not full joint structured P(g_i=g_j)."
            )
        diag = {
            "status": "NOISY_COPY_ASSIGNMENT_N1",
            "orbit_mode": mode,
            "singleton_tree_energy": decoded.get("singleton_tree_energy"),
            "orbit_attachments": decoded.get("orbit_attachments"),
            "orbit_copy_capacity": validate_orbit_copy_capacity(G, bar.to(G.device), self.partition),
            "use_copy_id_as_input": False,
            "use_oracle_C_as_input": False,
            "geometry_feedback": False,
            "freeze_gemnet_backbone": self.config.freeze_gemnet_backbone,
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
