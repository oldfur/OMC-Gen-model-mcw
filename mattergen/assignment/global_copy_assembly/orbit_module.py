"""Orbit-aware full copy assembly (experiment O2).

Stage A: singleton-role tree-CRF backbone (reuses BondPairPotential + TreeCRF).
Stage B: exact balanced attachment of non-singleton orbits (bitmask DP).
Final: merge rows into G[N,K], C=GG^T.

``mol_copy_id`` is never an input to any encoder / score / decoder forward.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from mattergen.common.role_partition_diffusion.molecule_encoder import MolecularGraphEncoder
from mattergen.common.role_partition_diffusion.oracle_partition import ContextCrystalEncoder

from .orbit_attachment import AttachmentResult, OrbitAttachmentHead, balanced_attachment_dp
from .orbit_membership import (
    OrbitPartition,
    validate_orbit_copy_capacity,
)
from .orbit_targets import OrbitAwareAssemblyTarget
from .pair_potential import BondPairPotential, permutation_factor
from .permutations import enumerate_permutations, identity_index, inverse_permutations
from .singleton_backbone import SingletonBackbone, build_singleton_backbone, edge_meta_lookup
from .targets import AssemblyTarget, target_state_indices
from .tree_crf import TreeCRF


@dataclass(frozen=True)
class OrbitAwareAssemblyConfig:
    enabled: bool = True
    mode: str = "clean_geometry_orbit_aware"
    role_source: str = "geometry_only_hard_r"
    non_singleton_strategy: str = "exact_capacity_attachment"
    parameterization: str = "singleton_tree_crf_plus_orbit_bitmask"
    anchor_policy: str = "highest_degree_singleton"
    anchor_role: int | None = None
    crystal_hidden_dim: int = 256
    crystal_num_layers: int = 4
    molecular_hidden_dim: int = 256
    pair_hidden_dim: int = 256
    distance_rbf_dim: int = 32
    pair_score_scale: float = 20.0
    singleton_tree_weight: float = 1.0
    orbit_attachment_weight: float = 1.0
    pair_aux_weight: float = 0.1
    gauge_marginalization: str = "logsumexp"
    exact_attachment_solver: str = "bitmask_dp"
    probability_tolerance: float = 1e-5
    use_oracle_role_assignment: bool = False
    use_oracle_copy_relation: bool = False
    use_copy_id_as_input: bool = False
    strict_finite_checks: bool = True
    virtual_bond_type_id: int = 7  # reserved embedding slot for virtual singleton edges


class OrbitAwareCopyAssembly(nn.Module):
    """O2 model: singleton backbone + orbit balanced attachment."""

    def __init__(self, config: OrbitAwareAssemblyConfig = OrbitAwareAssemblyConfig()):
        super().__init__()
        if config.mode != "clean_geometry_orbit_aware":
            raise ValueError("OrbitAwareCopyAssembly only supports mode=clean_geometry_orbit_aware")
        if config.non_singleton_strategy != "exact_capacity_attachment":
            raise ValueError("only exact_capacity_attachment is implemented")
        if config.exact_attachment_solver != "bitmask_dp":
            raise ValueError("only bitmask_dp solver is implemented")
        if config.use_oracle_copy_relation or config.use_copy_id_as_input:
            raise ValueError("copy supervision is forbidden as model input")
        if config.use_oracle_role_assignment:
            raise ValueError("O2 predicted-orbit path requires use_oracle_role_assignment=false")
        if config.crystal_hidden_dim != config.molecular_hidden_dim:
            raise ValueError("MVP requires matching crystal and molecular hidden dims")
        self.config = config
        h = config.crystal_hidden_dim
        self.crystal_encoder = ContextCrystalEncoder(
            hidden=h, layers=config.crystal_num_layers, rbf_dim=64
        )
        self.molecule_encoder = MolecularGraphEncoder(hidden=h, layers=config.crystal_num_layers)
        self.pair_potential = BondPairPotential(
            hidden=h,
            pair_hidden=config.pair_hidden_dim,
            rbf_dim=config.distance_rbf_dim,
            score_scale=config.pair_score_scale,
        )
        self.orbit_head = OrbitAttachmentHead(
            hidden=h,
            rbf_dim=config.distance_rbf_dim,
            score_scale=config.pair_score_scale,
            gauge_marginalization=config.gauge_marginalization,
        )
        # Virtual edge path-length embedding (scalar path length -> hidden)
        self.path_length_embedding = nn.Embedding(32, h)
        self.virtual_mix = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, h))

    def encode(
        self,
        *,
        z: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        role_z: torch.Tensor,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hx = self.crystal_encoder(z, frac, cell, context_mode="geometry_only")
        hm = self.molecule_encoder(role_z, role_edge_index, role_bond_type)
        return hx, hm

    def _singleton_bond_type(self, backbone: SingletonBackbone, role_a: int, role_b: int) -> int:
        edge = edge_meta_lookup(backbone, role_a, role_b)
        if edge.real:
            return int(edge.path_bond_types[0])
        return int(self.config.virtual_bond_type_id)

    def singleton_factors(
        self,
        *,
        singleton_target: AssemblyTarget,
        backbone: SingletonBackbone,
        hx: torch.Tensor,
        hm: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor, dict[tuple[int, int], torch.Tensor]]:
        """Tree factors on *local* singleton role indices."""
        tree = backbone.tree
        if singleton_target.anchor_role != tree.root:
            raise ValueError("singleton target anchor and backbone tree root disagree")
        states = enumerate_permutations(singleton_target.K, device=hx.device)
        inverse = inverse_permutations(states)
        factors: dict[tuple[int, int], torch.Tensor] = {}
        scores: dict[tuple[int, int], torch.Tensor] = {}
        for parent_local, child_local in tree.tree_edges:
            parent_role = backbone.singleton_roles[parent_local]
            child_role = backbone.singleton_roles[child_local]
            parent_nodes = singleton_target.role_sets[parent_local]
            child_nodes = singleton_target.role_sets[child_local]
            bond = self._singleton_bond_type(backbone, parent_role, child_role)
            score = self.pair_potential(
                hx[parent_nodes],
                hx[child_nodes],
                hm[parent_role],
                hm[child_role],
                bond,
                frac[parent_nodes],
                frac[child_nodes],
                cell,
            )
            # Soft path-length bias for virtual edges (geometry still dominates via RBF).
            edge = edge_meta_lookup(backbone, parent_role, child_role)
            if not edge.real:
                length = min(int(edge.path_length), self.path_length_embedding.num_embeddings - 1)
                path_emb = self.path_length_embedding(
                    torch.as_tensor(length, device=hx.device, dtype=torch.long)
                )
                # additive scalar from path embedding similarity (bounded)
                bias = self.score_scale_tanh(
                    (path_emb * (hm[parent_role] + hm[child_role])).sum() / max(1, path_emb.numel())
                )
                score = score + bias
            scores[(parent_local, child_local)] = score
            factors[(parent_local, child_local)] = permutation_factor(score, states, inverse)
        if self.config.strict_finite_checks and not all(torch.isfinite(v).all() for v in factors.values()):
            raise FloatingPointError("non-finite singleton tree factors")
        return factors, states, scores

    def score_scale_tanh(self, x: torch.Tensor) -> torch.Tensor:
        s = float(self.config.pair_score_scale)
        return s * torch.tanh(x / s)

    def build_attachment_scores_from_G(
        self,
        *,
        orbit_atoms: torch.Tensor,
        G_singleton: torch.Tensor,
        hx: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        singleton_mask: torch.Tensor,
        orbit_id: int,
        mode: str | None = None,
    ) -> torch.Tensor:
        """F[k,i,j] for orbit local indices i,j and copy k (out-of-place; autograd-safe)."""
        n = int(orbit_atoms.numel())
        K = int(G_singleton.shape[1])
        neg = hx.new_tensor(float("-inf"))
        mats: list[torch.Tensor] = []
        for k in range(K):
            copy_atoms = ((G_singleton[:, k] > 0.5) & singleton_mask).nonzero(as_tuple=False).flatten()
            if copy_atoms.numel() == 0:
                raise ValueError(f"singleton backbone copy {k} is empty")
            h_copy = hx[copy_atoms]
            frac_copy = frac[copy_atoms]
            # Upper triangle scores first (unordered pairs).
            upper: dict[tuple[int, int], torch.Tensor] = {}
            for i in range(n):
                ai = int(orbit_atoms[i].item())
                for j in range(i + 1, n):
                    aj = int(orbit_atoms[j].item())
                    upper[(i, j)] = self.orbit_head.pair_score(
                        h_i=hx[ai],
                        h_j=hx[aj],
                        frac_i=frac[ai],
                        frac_j=frac[aj],
                        h_copy=h_copy,
                        frac_copy=frac_copy,
                        cell=cell,
                        orbit_id=orbit_id,
                        mode=mode,
                    )
            rows: list[torch.Tensor] = []
            for i in range(n):
                cols: list[torch.Tensor] = []
                for j in range(n):
                    if i == j:
                        cols.append(neg)
                    elif i < j:
                        cols.append(upper[(i, j)])
                    else:
                        cols.append(upper[(j, i)])
                rows.append(torch.stack(cols, dim=0))
            mats.append(torch.stack(rows, dim=0))
        return torch.stack(mats, dim=0)

    def loss(
        self,
        *,
        o2_target: OrbitAwareAssemblyTarget,
        backbone: SingletonBackbone,
        z: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        role_z: torch.Tensor,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hx, hm = self.encode(
            z=z, frac=frac, cell=cell, role_z=role_z,
            role_edge_index=role_edge_index, role_bond_type=role_bond_type,
        )
        singleton_target = o2_target.singleton_target
        factors, states, scores = self.singleton_factors(
            singleton_target=singleton_target, backbone=backbone, hx=hx, hm=hm, frac=frac, cell=cell
        )
        crf = TreeCRF(backbone.tree, num_states=len(states), identity_state=identity_index(states))
        target_states = target_state_indices(singleton_target, states)
        log_z, target_score = crf.log_partition_and_target_score(factors, target_states)
        tree_nll = (log_z - target_score) / max(1, len(backbone.tree.tree_edges))
        # pair aux CE on singleton tree edges
        pair_losses = []
        for parent, child in backbone.tree.tree_edges:
            p_parent = singleton_target.permutations[parent]
            p_child = singleton_target.permutations[child]
            inv_child = torch.empty_like(p_child)
            inv_child[p_child] = torch.arange(singleton_target.K, device=p_child.device)
            match = inv_child[p_parent]
            pair_losses.extend(
                [
                    nn.functional.cross_entropy(scores[(parent, child)], match),
                    nn.functional.cross_entropy(scores[(parent, child)].T, torch.argsort(match)),
                ]
            )
        pair_ce = torch.stack(pair_losses).mean() if pair_losses else tree_nll.new_zeros(())

        # Teacher-forced singleton G for orbit attachment scores (Stage B).
        G_sing = torch.zeros(o2_target.N, o2_target.K, dtype=torch.float32, device=hx.device)
        for role, nodes in singleton_target.role_sets.items():
            G_sing[nodes, singleton_target.permutations[role].to(hx.device)] = 1.0
        singleton_mask = torch.zeros(o2_target.N, dtype=torch.bool, device=hx.device)
        for nodes in singleton_target.role_sets.values():
            singleton_mask[nodes] = True

        orbit_nlls = []
        orbit_logz = []
        orbit_target_scores = []
        for ot in o2_target.orbit_targets:
            F = self.build_attachment_scores_from_G(
                orbit_atoms=ot.atom_indices.to(hx.device),
                G_singleton=G_sing,
                hx=hx,
                frac=frac,
                cell=cell,
                singleton_mask=singleton_mask,
                orbit_id=ot.orbit_index,
                mode="logsumexp",
            )
            result = balanced_attachment_dp(
                F,
                target_pairs=list(ot.pairs_local),
                atoms_per_copy=ot.atoms_per_copy,
            )
            if result.target_score is None:
                raise RuntimeError("attachment DP missing target_score")
            nll = result.log_partition - result.target_score
            orbit_nlls.append(nll)
            orbit_logz.append(result.log_partition)
            orbit_target_scores.append(result.target_score)
        orbit_nll = torch.stack(orbit_nlls).mean() if orbit_nlls else tree_nll.new_zeros(())

        tol = self.config.probability_tolerance
        if tree_nll.detach() < -tol:
            raise FloatingPointError(f"singleton tree NLL negative: {float(tree_nll.detach())}")
        if orbit_nll.detach() < -tol:
            raise FloatingPointError(f"orbit attachment NLL negative: {float(orbit_nll.detach())}")

        total = (
            self.config.singleton_tree_weight * tree_nll
            + self.config.orbit_attachment_weight * orbit_nll
            + self.config.pair_aux_weight * pair_ce
        )
        if not torch.isfinite(total):
            raise FloatingPointError("O2 loss is non-finite")
        return {
            "loss": total,
            "tree_nll": tree_nll,
            "orbit_nll": orbit_nll,
            "pair_ce": pair_ce,
            "singleton_log_partition": log_z,
            "singleton_target_score": target_score,
            "orbit_log_partition": torch.stack(orbit_logz).mean() if orbit_logz else total.new_zeros(()),
            "orbit_target_score": torch.stack(orbit_target_scores).mean() if orbit_target_scores else total.new_zeros(()),
        }

    @torch.no_grad()
    def map_decode(
        self,
        *,
        o2_target: OrbitAwareAssemblyTarget,
        backbone: SingletonBackbone,
        z: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        role_z: torch.Tensor,
        role_edge_index: torch.Tensor,
        role_bond_type: torch.Tensor,
    ) -> dict[str, object]:
        hx, hm = self.encode(
            z=z, frac=frac, cell=cell, role_z=role_z,
            role_edge_index=role_edge_index, role_bond_type=role_bond_type,
        )
        singleton_target = o2_target.singleton_target
        factors, states, scores = self.singleton_factors(
            singleton_target=singleton_target, backbone=backbone, hx=hx, hm=hm, frac=frac, cell=cell
        )
        crf = TreeCRF(backbone.tree, num_states=len(states), identity_state=identity_index(states))
        decoded = crf.map_decode(factors)
        N, K = o2_target.N, o2_target.K
        G = torch.zeros(N, K, dtype=torch.float32, device=hx.device)
        for role, nodes in singleton_target.role_sets.items():
            perm = states[decoded.state_indices[role]].long()
            G[nodes, perm] = 1.0
        singleton_mask = torch.zeros(N, dtype=torch.bool, device=hx.device)
        for nodes in singleton_target.role_sets.values():
            singleton_mask[nodes] = True

        attachment_maps: dict[int, AttachmentResult] = {}
        for ot in o2_target.orbit_targets:
            F = self.build_attachment_scores_from_G(
                orbit_atoms=ot.atom_indices.to(hx.device),
                G_singleton=G,
                hx=hx,
                frac=frac,
                cell=cell,
                singleton_mask=singleton_mask,
                orbit_id=ot.orbit_index,
                mode="max",
            )
            result = balanced_attachment_dp(F, atoms_per_copy=ot.atoms_per_copy)
            attachment_maps[ot.orbit_index] = result
            atoms = ot.atom_indices.to(hx.device)
            for k, (i, j) in enumerate(result.map_pairs):
                G[int(atoms[i].item()), k] = 1.0
                G[int(atoms[j].item()), k] = 1.0

        if not torch.allclose(G.sum(-1), torch.ones(N, device=G.device)):
            raise AssertionError("each atom must have exactly one copy in G")
        cap = validate_orbit_copy_capacity(G, o2_target.bar_r.to(G.device), o2_target.partition)
        if not cap["valid"]:
            raise AssertionError(f"orbit-copy capacity violated: {cap}")
        C = G @ G.T
        return {
            "G": G,
            "C": C,
            "singleton_state_indices": decoded.state_indices,
            "singleton_tree_energy": decoded.score,
            "orbit_attachments": {
                j: {"map_pairs": res.map_pairs, "map_score": float(res.map_score.detach())}
                for j, res in attachment_maps.items()
            },
            "orbit_copy_capacity": cap,
            "status": "CLEAN_GEOMETRY_ORBIT_AWARE_O2",
        }


def prepare_backbone(
    partition: OrbitPartition,
    role_edge_index: torch.Tensor,
    role_bond_type: torch.Tensor,
    *,
    anchor_role: int | None = None,
) -> SingletonBackbone:
    from .tree_builder import select_anchor_role

    # select_anchor_role expects per-role orbits list
    per_role = [list(partition.orbits[partition.role_to_orbit[r]]) for r in range(partition.M)]
    anchor = select_anchor_role(per_role, role_edge_index, explicit_anchor=anchor_role)
    return build_singleton_backbone(
        role_edge_index,
        role_bond_type,
        M=partition.M,
        singleton_roles=partition.singleton_roles(),
        anchor_role=anchor,
    )
