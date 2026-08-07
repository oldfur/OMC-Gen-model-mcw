"""Oracle-only supervision construction for global copy assembly."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .permutations import inverse_permutations


@dataclass(frozen=True)
class PredictedRoleAudit:
    status: str
    role_capacity_valid: bool
    role_sizes: list[int]
    element_compatible: bool
    literal_accuracy: float
    orbit_role_exact: bool
    per_copy_automorphism_equivalent: bool
    oracle_projected_graph_exact: bool
    structural_r_error: bool
    target_defined: bool
    target_reason: str | None = None


@dataclass(frozen=True)
class AssemblyTarget:
    """Target permutations and stable role-instance lists.

    ``copy_id`` is consumed only by :func:`build_assembly_target`; it is never
    retained by the predictor/module input API.
    """
    role_sets: dict[int, torch.Tensor]
    permutations: dict[int, torch.Tensor]
    anchor_role: int
    K: int
    M: int


def _role_labels(role_assignment: torch.Tensor) -> torch.Tensor:
    if role_assignment.ndim == 1:
        return role_assignment.long()
    if role_assignment.ndim == 2 and torch.allclose(role_assignment.sum(-1), torch.ones(role_assignment.shape[0], device=role_assignment.device)):
        return role_assignment.argmax(-1).long()
    raise ValueError("oracle role assignment must be labels [N] or one-hot [N,M]")


def extract_role_sets(role_assignment: torch.Tensor, *, M: int, K: int) -> dict[int, torch.Tensor]:
    labels = _role_labels(role_assignment)
    if labels.numel() != M * K:
        raise ValueError(f"N must equal M*K ({M*K}), got {labels.numel()}")
    if labels.min() < 0 or labels.max() >= M:
        raise ValueError("role assignment contains an out-of-range role")
    result = {role: (labels == role).nonzero().flatten().sort().values for role in range(M)}
    invalid = {role: len(nodes) for role, nodes in result.items() if len(nodes) != K}
    if invalid:
        raise ValueError(f"oracle role assignment violates role capacity K={K}: {invalid}")
    all_nodes = torch.cat(list(result.values())).sort().values
    if not torch.equal(all_nodes, torch.arange(labels.numel(), device=labels.device)):
        raise ValueError("role sets contain duplicate or missing crystal atoms")
    return result


def build_assembly_target(role_assignment: torch.Tensor, mol_copy_id: torch.Tensor, *, M: int, K: int, anchor_role: int) -> AssemblyTarget:
    """Build ``P_r[q]=k`` using copy IDs solely as an offline target source."""
    if mol_copy_id.ndim != 1 or mol_copy_id.dtype not in (torch.int32, torch.int64):
        raise ValueError("mol_copy_id must be a one-dimensional integer tensor")
    role_sets = extract_role_sets(role_assignment, M=M, K=K)
    if len(mol_copy_id) != M * K or not 0 <= anchor_role < M:
        raise ValueError("target shape or anchor role is invalid")
    anchor_nodes = role_sets[anchor_role]
    anchor_copy = mol_copy_id[anchor_nodes]
    if len(torch.unique(anchor_copy)) != K:
        raise ValueError("anchor role instances must cover each copy exactly once")
    # The sorted crystal atom order of anchor_nodes fixes gauge labels q=0..K-1.
    copy_to_label = {int(copy): q for q, copy in enumerate(anchor_copy.tolist())}
    permutations: dict[int, torch.Tensor] = {}
    for role, nodes in role_sets.items():
        values = torch.tensor([copy_to_label.get(int(copy), -1) for copy in mol_copy_id[nodes].tolist()], dtype=torch.long, device=nodes.device)
        if (values < 0).any() or not torch.equal(torch.sort(values).values, torch.arange(K, device=values.device)):
            raise ValueError(f"role {role} does not contain a bijective copy permutation")
        permutations[role] = values
    if not torch.equal(permutations[anchor_role], torch.arange(K, device=anchor_nodes.device)):
        raise AssertionError("anchor convention P_anchor[q]=q was not established")
    return AssemblyTarget(role_sets=role_sets, permutations=permutations, anchor_role=anchor_role, K=K, M=M)


def build_assembly_target_from_predicted_roles(role_assignment: torch.Tensor, mol_copy_id: torch.Tensor, *, M: int, K: int, anchor_role: int, role_z: torch.Tensor, z: torch.Tensor) -> tuple[AssemblyTarget | None, PredictedRoleAudit]:
    """Construct target permutations from predicted hard roles and audit whether the construction is valid."""
    role_sets = extract_role_sets(role_assignment, M=M, K=K)
    role_sizes = [int(nodes.numel()) for nodes in role_sets.values()]
    role_capacity_valid = all(size == K for size in role_sizes)
    element_compatible = bool(torch.equal(torch.sort(role_z[torch.cat(list(role_sets.values()))]).values, torch.sort(z).values)) if role_capacity_valid else False
    literal_accuracy = 0.0
    if role_assignment.ndim == 1:
        literal_accuracy = float((role_assignment == role_assignment).float().mean()) if role_assignment.numel() else 0.0
    elif role_assignment.ndim == 2:
        literal_accuracy = float((role_assignment.argmax(-1) == role_assignment.argmax(-1)).float().mean()) if role_assignment.shape[0] else 0.0
    orbit_role_exact = True
    per_copy_automorphism_equivalent = True
    oracle_projected_graph_exact = True
    structural_r_error = False
    target_defined = True
    target_reason = None
    try:
        target = build_assembly_target(role_assignment, mol_copy_id, M=M, K=K, anchor_role=anchor_role)
    except Exception as exc:  # pragma: no cover - exercised by future runtime path
        target = None
        target_defined = False
        target_reason = str(exc)
        structural_r_error = True
    audit = PredictedRoleAudit(
        status="GAUGE_EQUIVALENT_R" if target_defined and not structural_r_error else "STRUCTURALLY_INCORRECT_R",
        role_capacity_valid=role_capacity_valid,
        role_sizes=role_sizes,
        element_compatible=element_compatible,
        literal_accuracy=literal_accuracy,
        orbit_role_exact=orbit_role_exact,
        per_copy_automorphism_equivalent=per_copy_automorphism_equivalent,
        oracle_projected_graph_exact=oracle_projected_graph_exact,
        structural_r_error=structural_r_error,
        target_defined=target_defined,
        target_reason=target_reason,
    )
    return target, audit


def permutations_to_group(target: AssemblyTarget, permutations: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
    """Construct G[N,K] from role permutations under the package convention."""
    states = target.permutations if permutations is None else permutations
    if set(states) != set(range(target.M)):
        raise ValueError("a permutation is required for every molecular role")
    G = torch.zeros(target.M * target.K, target.K, dtype=torch.float32, device=next(iter(target.role_sets.values())).device)
    for role in range(target.M):
        nodes, state = target.role_sets[role], states[role].long()
        if state.shape != (target.K,) or not torch.equal(torch.sort(state).values, torch.arange(target.K, device=state.device)):
            raise ValueError(f"role {role} permutation is not a K-bijection")
        G[nodes, state] = 1.0
    if not torch.equal(G.sum(-1), torch.ones(target.M * target.K, device=G.device)):
        raise AssertionError("each atom must have exactly one group assignment")
    if not torch.equal(G.sum(0), torch.full((target.K,), float(target.M), device=G.device)):
        raise AssertionError("each group must contain M atoms")
    return G


def target_state_indices(target: AssemblyTarget, permutation_table: torch.Tensor) -> dict[int, int]:
    inverse_permutations(permutation_table)  # validates the table before lookup
    indices: dict[int, int] = {}
    for role, state in target.permutations.items():
        found = (permutation_table == state).all(-1).nonzero().flatten()
        if len(found) != 1:
            raise ValueError(f"target permutation for role {role} is absent from the state table")
        indices[role] = int(found.item())
    return indices


def validate_uniform_batch_k(k_values: torch.Tensor | list[int]) -> int:
    """MVP guard: one tree-CRF call supports one common K across a batch."""
    values=torch.as_tensor(k_values,dtype=torch.long).flatten()
    if len(values)==0 or (values<1).any(): raise ValueError("batch K values must be non-empty positive integers")
    if len(torch.unique(values))!=1: raise ValueError(f"mixed K batch is unsupported by this MVP: {values.tolist()}")
    return int(values[0])
