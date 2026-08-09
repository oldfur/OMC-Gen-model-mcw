"""Supervision builders for orbit-aware O2 assembly.

``mol_copy_id`` is consumed only here (and evaluation).  It never enters model
forward tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .orbit_membership import (
    OrbitPartition,
    collapse_roles_to_orbit_membership,
    orbit_atom_sets,
    validate_orbit_global_capacity,
)
from .targets import AssemblyTarget


@dataclass(frozen=True)
class OrbitAttachmentTarget:
    """Balanced attachment target for one non-singleton orbit."""

    orbit_index: int
    atom_indices: torch.Tensor  # [m*K] sorted crystal indices (V_o)
    # For each copy label k in singleton gauge order 0..K-1: two local indices into atom_indices
    pairs_local: tuple[tuple[int, int], ...]
    atoms_per_copy: int


@dataclass(frozen=True)
class OrbitAwareAssemblyTarget:
    """Full O2 supervision: singleton P* + orbit attachment A*."""

    partition: OrbitPartition
    bar_r: torch.Tensor  # [N,J]
    singleton_roles: tuple[int, ...]
    singleton_target: AssemblyTarget
    # Map singleton local tree role index -> molecular role id
    local_to_role: tuple[int, ...]
    role_to_local: dict[int, int]
    orbit_targets: tuple[OrbitAttachmentTarget, ...]
    K: int
    N: int


def build_singleton_target_from_roles(
    role_assignment: torch.Tensor,
    mol_copy_id: torch.Tensor,
    *,
    singleton_roles: list[int] | tuple[int, ...],
    M: int,
    K: int,
    anchor_role: int,
) -> AssemblyTarget:
    """Build P* only on singleton canonical roles (each |V_r|=K).

    Non-singleton role labels are ignored for Stage A sets.  Capacity on each
    singleton role must still be K under the provided assignment.
    """
    roles = tuple(sorted(int(r) for r in singleton_roles))
    if anchor_role not in roles:
        raise ValueError("anchor_role must be a singleton role")
    # extract_role_sets validates all M roles; filter by building labels only
    # for singleton check via manual sets.
    if role_assignment.ndim == 1:
        labels = role_assignment.long()
    else:
        labels = role_assignment.argmax(-1).long()
    role_sets = {}
    for role in roles:
        nodes = (labels == role).nonzero(as_tuple=False).flatten().sort().values
        if int(nodes.numel()) != K:
            raise ValueError(
                f"TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: singleton role {role} "
                f"has size {int(nodes.numel())}, expected K={K}"
            )
        role_sets[role] = nodes
    # Reuse build_assembly_target by constructing a dense assignment that only
    # needs singleton roles... build_assembly_target requires full M*K capacity.
    # Instead inline permutation construction for the singleton subset, then
    # wrap as AssemblyTarget with M_eff = number of singletons via local remap.
    # We keep molecular role ids as keys by building a *local* AssemblyTarget
    # with roles remapped to 0..S-1 for tree-CRF compatibility.
    role_to_local = {role: i for i, role in enumerate(roles)}
    local_sets = {role_to_local[r]: role_sets[r] for r in roles}
    local_anchor = role_to_local[anchor_role]
    S = len(roles)
    # Build permutations in local id space using the same gauge as build_assembly_target.
    anchor_nodes = local_sets[local_anchor]
    anchor_copy = mol_copy_id[anchor_nodes]
    if len(torch.unique(anchor_copy)) != K:
        raise ValueError("singleton anchor role instances must cover each copy exactly once")
    copy_to_label = {int(c): q for q, c in enumerate(anchor_copy.tolist())}
    permutations: dict[int, torch.Tensor] = {}
    for local_role, nodes in local_sets.items():
        values = torch.tensor(
            [copy_to_label.get(int(c), -1) for c in mol_copy_id[nodes].tolist()],
            dtype=torch.long,
            device=nodes.device,
        )
        if (values < 0).any() or not torch.equal(torch.sort(values).values, torch.arange(K, device=values.device)):
            raise ValueError(
                f"TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: singleton local role {local_role} "
                "does not contain a bijective copy permutation"
            )
        permutations[local_role] = values
    if not torch.equal(permutations[local_anchor], torch.arange(K, device=anchor_nodes.device)):
        raise AssertionError("singleton anchor convention P_anchor[q]=q failed")
    return AssemblyTarget(
        role_sets=local_sets,
        permutations=permutations,
        anchor_role=local_anchor,
        K=K,
        M=S,
    )


def build_orbit_attachment_target(
    bar_r: torch.Tensor,
    mol_copy_id: torch.Tensor,
    partition: OrbitPartition,
    *,
    orbit_index: int,
    copy_label_from_atom: torch.Tensor,
    K: int,
) -> OrbitAttachmentTarget:
    """Build balanced attachment A* for one orbit using true copy labels.

    ``copy_label_from_atom`` is the singleton-gauge copy id for each crystal
    atom (length N), derived from singleton P*/G — or equivalently mol_copy_id
    remapped through the anchor gauge.  For supervision we use the same gauge
    as singleton assembly: label q for each true copy.
    """
    m = partition.orbit_sizes[orbit_index]
    atoms = (bar_r[:, orbit_index] > 0.5).nonzero(as_tuple=False).flatten().sort().values
    if int(atoms.numel()) != m * K:
        raise ValueError(
            f"TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: orbit {orbit_index} size "
            f"{int(atoms.numel())} != m*K={m * K}"
        )
    # Map crystal atom -> local index in V_o
    local_of = {int(a): i for i, a in enumerate(atoms.tolist())}
    pairs: list[tuple[int, int]] = []
    for k in range(K):
        members = [local_of[int(a)] for a in atoms.tolist() if int(copy_label_from_atom[int(a)]) == k]
        if len(members) != m:
            raise ValueError(
                f"TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: orbit {orbit_index} copy {k} "
                f"has {len(members)} atoms, expected {m}"
            )
        if m == 2:
            i, j = sorted(members)
            pairs.append((i, j))
        else:
            raise NotImplementedError("MVP orbit attachment target supports m=2 only")
    return OrbitAttachmentTarget(
        orbit_index=orbit_index,
        atom_indices=atoms,
        pairs_local=tuple(pairs),
        atoms_per_copy=m,
    )


def singleton_copy_labels_from_target(singleton_target: AssemblyTarget, N: int) -> torch.Tensor:
    """Scatter P* on singleton role sets to a length-N copy-gauge label tensor.

    Atoms not in any singleton set receive -1.
    """
    labels = torch.full((N,), -1, dtype=torch.long, device=next(iter(singleton_target.role_sets.values())).device)
    for role, nodes in singleton_target.role_sets.items():
        labels[nodes] = singleton_target.permutations[role].to(labels.device)
    return labels


def build_orbit_aware_target(
    role_assignment: torch.Tensor,
    mol_copy_id: torch.Tensor,
    *,
    partition: OrbitPartition,
    K: int,
    anchor_role: int,
) -> OrbitAwareAssemblyTarget:
    """Full O2 supervision from hard canonical R + mol_copy_id."""
    bar = collapse_roles_to_orbit_membership(role_assignment, partition)
    cap = validate_orbit_global_capacity(bar, partition, K=K)
    if not cap["valid"]:
        raise ValueError(f"TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: orbit capacity {cap}")
    singleton_roles = tuple(partition.singleton_roles())
    singleton_target = build_singleton_target_from_roles(
        role_assignment,
        mol_copy_id,
        singleton_roles=singleton_roles,
        M=partition.M,
        K=K,
        anchor_role=anchor_role,
    )
    role_to_local = {role: i for i, role in enumerate(singleton_roles)}
    N = int(bar.shape[0])
    # Anchor gauge copy labels for all atoms via mol_copy_id:
    # Use singleton anchor mapping from true copies to gauge labels q.
    anchor_local = role_to_local[anchor_role]
    anchor_nodes = singleton_target.role_sets[anchor_local]
    anchor_copy = mol_copy_id[anchor_nodes]
    copy_to_q = {int(c): q for q, c in enumerate(anchor_copy.tolist())}
    copy_gauge = torch.tensor(
        [copy_to_q[int(c)] for c in mol_copy_id.tolist()],
        dtype=torch.long,
        device=mol_copy_id.device,
    )
    orbit_targets = []
    for j in partition.non_singleton_orbit_indices():
        orbit_targets.append(
            build_orbit_attachment_target(
                bar, mol_copy_id, partition, orbit_index=j, copy_label_from_atom=copy_gauge, K=K
            )
        )
    return OrbitAwareAssemblyTarget(
        partition=partition,
        bar_r=bar,
        singleton_roles=singleton_roles,
        singleton_target=singleton_target,
        local_to_role=singleton_roles,
        role_to_local=role_to_local,
        orbit_targets=tuple(orbit_targets),
        K=K,
        N=N,
    )
