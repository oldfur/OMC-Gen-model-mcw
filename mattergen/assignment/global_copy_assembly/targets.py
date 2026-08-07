"""Supervision construction for global copy assembly.

Role representation is always full canonical ``R ∈ {0,1}^{N×M}`` with
``|V_r| = K`` for every role. Molecular automorphisms are a *gauge* only:
they define physical equivalence and audit labels. They never collapse
orbits, never rewrite predicted R, and never substitute oracle ``R_0``.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .permutations import inverse_permutations


@dataclass(frozen=True)
class PredictedRoleAudit:
    """Audit of a predicted hard role assignment against capacity and Aut gauge.

    Literal metrics are retained for diagnostics only
    (``literal_metrics_scope == "DIAGNOSTIC_ONLY"``). Physical acceptance
    uses per-copy molecular-automorphism equivalence and assembly target
    constructibility under the *predicted* gauge.
    """

    status: str
    role_capacity_valid: bool
    role_sizes: list[int]
    element_compatible: bool
    literal_accuracy: float
    literal_exact: bool
    orbit_role_accuracy: float
    orbit_role_exact: bool
    per_copy_automorphism_equivalent: bool
    physical_role_assignment_exact: bool
    oracle_projected_graph_exact: bool
    structural_r_error: bool
    target_defined: bool
    target_reason: str | None = None
    literal_metrics_scope: str = "DIAGNOSTIC_ONLY"


@dataclass(frozen=True)
class AssemblyTarget:
    """Target permutations and stable role-instance lists.

    ``copy_id`` is consumed only by :func:`build_assembly_target` / predicted
    target builders as offline supervision for ``P*``. It is never retained by
    the predictor/module input API and is never used to rewrite predicted R.
    """

    role_sets: dict[int, torch.Tensor]
    permutations: dict[int, torch.Tensor]
    anchor_role: int
    K: int
    M: int


def _role_labels(role_assignment: torch.Tensor) -> torch.Tensor:
    if role_assignment.ndim == 1:
        return role_assignment.long()
    if role_assignment.ndim == 2 and torch.allclose(
        role_assignment.sum(-1), torch.ones(role_assignment.shape[0], device=role_assignment.device)
    ):
        return role_assignment.argmax(-1).long()
    raise ValueError("role assignment must be labels [N] or one-hot [N,M]")


def extract_role_sets(role_assignment: torch.Tensor, *, M: int, K: int) -> dict[int, torch.Tensor]:
    """Extract capacity-K role sets from a hard assignment (oracle or predicted)."""
    labels = _role_labels(role_assignment)
    if labels.numel() != M * K:
        raise ValueError(f"N must equal M*K ({M*K}), got {labels.numel()}")
    if labels.min() < 0 or labels.max() >= M:
        raise ValueError("role assignment contains an out-of-range role")
    result = {role: (labels == role).nonzero().flatten().sort().values for role in range(M)}
    invalid = {role: len(nodes) for role, nodes in result.items() if len(nodes) != K}
    if invalid:
        raise ValueError(f"role assignment violates role capacity K={K}: {invalid}")
    all_nodes = torch.cat(list(result.values())).sort().values
    if not torch.equal(all_nodes, torch.arange(labels.numel(), device=labels.device)):
        raise ValueError("role sets contain duplicate or missing crystal atoms")
    return result


def build_assembly_target(
    role_assignment: torch.Tensor,
    mol_copy_id: torch.Tensor,
    *,
    M: int,
    K: int,
    anchor_role: int,
) -> AssemblyTarget:
    """Build ``P_r[q]=k`` on the *given* role assignment's role sets.

    Role sets come from ``role_assignment`` only (predicted or oracle).
    ``mol_copy_id`` supplies supervision for which true copy each instance
    belongs to. This function never rewrites roles toward oracle ``R_0``.
    """
    if mol_copy_id.ndim != 1 or mol_copy_id.dtype not in (torch.int32, torch.int64):
        raise ValueError("mol_copy_id must be a one-dimensional integer tensor")
    role_sets = extract_role_sets(role_assignment, M=M, K=K)
    if len(mol_copy_id) != M * K or not 0 <= anchor_role < M:
        raise ValueError("target shape or anchor role is invalid")
    anchor_nodes = role_sets[anchor_role]
    anchor_copy = mol_copy_id[anchor_nodes]
    if len(torch.unique(anchor_copy)) != K:
        raise ValueError("anchor role instances must cover each copy exactly once")
    # Sorted crystal atom order of anchor_nodes fixes gauge labels q=0..K-1.
    copy_to_label = {int(copy): q for q, copy in enumerate(anchor_copy.tolist())}
    permutations: dict[int, torch.Tensor] = {}
    for role, nodes in role_sets.items():
        values = torch.tensor(
            [copy_to_label.get(int(copy), -1) for copy in mol_copy_id[nodes].tolist()],
            dtype=torch.long,
            device=nodes.device,
        )
        if (values < 0).any() or not torch.equal(torch.sort(values).values, torch.arange(K, device=values.device)):
            raise ValueError(
                "TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: "
                f"role {role} does not contain a bijective copy permutation"
            )
        permutations[role] = values
    if not torch.equal(permutations[anchor_role], torch.arange(K, device=anchor_nodes.device)):
        raise AssertionError("anchor convention P_anchor[q]=q was not established")
    return AssemblyTarget(role_sets=role_sets, permutations=permutations, anchor_role=anchor_role, K=K, M=M)


def _apply_role_permutation(role_block: torch.Tensor, perm: list[int] | tuple[int, ...]) -> torch.Tensor:
    mapping = torch.as_tensor(list(perm), dtype=torch.long, device=role_block.device)
    if mapping.ndim != 1:
        raise ValueError("automorphism permutation must be a 1-D role map")
    return mapping[role_block.long()]


def per_copy_automorphism_equivalent(
    predicted_roles: torch.Tensor,
    oracle_roles: torch.Tensor,
    mol_copy_id: torch.Tensor,
    automorphism_permutations: list[list[int]] | list[tuple[int, ...]],
) -> bool:
    """True iff each true copy differs from oracle R by some ``π ∈ Aut(G_mol)``.

    Independent per-copy gauges are allowed (``Aut(G_mol)^K``). A single global
    automorphism alignment is *not* a substitute: each copy is checked alone.
    """
    pred = _role_labels(predicted_roles)
    truth = _role_labels(oracle_roles)
    if pred.shape != truth.shape or pred.shape != mol_copy_id.shape:
        raise ValueError("predicted roles, oracle roles, and mol_copy_id must share shape [N]")
    if not automorphism_permutations:
        raise ValueError("automorphism_permutations must be non-empty")
    copy = mol_copy_id.long()
    for copy_id in range(int(copy.max().item()) + 1):
        idx = (copy == copy_id).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        truth_block = truth[idx]
        pred_block = pred[idx]
        block_ok = any(
            torch.equal(pred_block, _apply_role_permutation(truth_block, perm))
            for perm in automorphism_permutations
        )
        if not block_ok:
            return False
    return True


def orbit_role_metrics(
    predicted_roles: torch.Tensor,
    oracle_roles: torch.Tensor,
    role_orbits: list[list[int]],
) -> tuple[float, bool]:
    """Atom-wise orbit membership accuracy (diagnostic physical soft metric)."""
    pred = _role_labels(predicted_roles)
    truth = _role_labels(oracle_roles)
    if pred.shape != truth.shape:
        raise ValueError("predicted and oracle roles must share shape [N]")
    if not role_orbits:
        raise ValueError("role_orbits must be non-empty")
    ok = []
    for i in range(int(pred.numel())):
        true_role = int(truth[i].item())
        if true_role < 0 or true_role >= len(role_orbits):
            raise ValueError(f"oracle role {true_role} is outside role_orbits")
        ok.append(int(pred[i].item()) in set(int(r) for r in role_orbits[true_role]))
    accuracy = float(sum(ok) / max(1, len(ok)))
    return accuracy, bool(all(ok))


def build_assembly_target_from_predicted_roles(
    role_assignment: torch.Tensor,
    mol_copy_id: torch.Tensor,
    *,
    M: int,
    K: int,
    anchor_role: int,
    role_z: torch.Tensor,
    z: torch.Tensor,
    oracle_role: torch.Tensor | None = None,
    automorphism_permutations: list[list[int]] | list[tuple[int, ...]] | None = None,
    role_orbits: list[list[int]] | None = None,
) -> tuple[AssemblyTarget | None, PredictedRoleAudit]:
    """Build ``P*`` on predicted role sets and audit Aut-gauge equivalence.

    Critical invariants
    -------------------
    * ``R`` stays shape ``[N]`` labels / ``[N,M]`` one-hot with capacity ``K``.
    * Role sets are ``V_r = {i : R_ir = 1}`` from the *artifact* assignment.
    * No canonicalization, no oracle-R rewrite, no orbit collapse.
    * ``mol_copy_id`` is used only to read which true copy each predicted
      instance belongs to when forming ``P*_r[q]``.
    * If a role set has duplicate/missing true copies, returns
      ``TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR`` without substituting ``R_0``.
    """
    labels = _role_labels(role_assignment)
    if role_z.ndim != 1 or int(role_z.numel()) != M:
        raise ValueError(f"role_z must have shape [M={M}], got shape {tuple(role_z.shape)}")
    if z.ndim != 1 or int(z.numel()) != int(labels.numel()):
        raise ValueError(f"z must have shape [N={labels.numel()}], got shape {tuple(z.shape)}")

    # Soft role-set sizes: capacity failures become audit flags, not early hard exits.
    role_sets_soft = {
        role: (labels == role).nonzero(as_tuple=False).flatten().sort().values for role in range(M)
    }
    role_sizes = [int(role_sets_soft[role].numel()) for role in range(M)]
    in_range = bool(labels.numel() > 0 and int(labels.min()) >= 0 and int(labels.max()) < M)
    role_capacity_valid = bool(
        in_range and labels.numel() == M * K and all(size == K for size in role_sizes)
    )
    # Element hard-mask: index role_z by *role labels*, never crystal atom indices.
    element_compatible = bool(torch.equal(z, role_z[labels])) if in_range else False

    literal_accuracy = 0.0
    literal_exact = False
    if oracle_role is not None:
        oracle_labels = _role_labels(oracle_role)
        if oracle_labels.shape == labels.shape:
            literal_accuracy = float((labels == oracle_labels).float().mean())
            literal_exact = bool(torch.equal(labels, oracle_labels))

    orbit_role_accuracy = 0.0
    orbit_role_exact = False
    if oracle_role is not None and role_orbits is not None:
        orbit_role_accuracy, orbit_role_exact = orbit_role_metrics(labels, oracle_role, role_orbits)

    per_copy_auto = False
    if oracle_role is not None and automorphism_permutations is not None:
        per_copy_auto = per_copy_automorphism_equivalent(
            labels, oracle_role, mol_copy_id, automorphism_permutations
        )
    physical_role_assignment_exact = per_copy_auto

    # Assembly target on the predicted gauge only — never on oracle role sets.
    structural_r_error = False
    target_defined = False
    target_reason: str | None = None
    target: AssemblyTarget | None = None
    try:
        target = build_assembly_target(
            role_assignment, mol_copy_id, M=M, K=K, anchor_role=anchor_role
        )
        target_defined = True
        # Prove role_sets recover the artifact labels (no silent R rewrite).
        recovered = torch.empty_like(labels)
        for role, nodes in target.role_sets.items():
            recovered[nodes] = int(role)
        if not torch.equal(recovered, labels):
            raise RuntimeError("internal error: target.role_sets do not match predicted R artifact")
    except Exception as exc:
        target = None
        target_defined = False
        reason = str(exc)
        if not reason.startswith("TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR"):
            reason = f"TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: {reason}"
        target_reason = reason
        structural_r_error = True

    if target_defined and not element_compatible:
        target = None
        target_defined = False
        structural_r_error = True
        target_reason = (
            "TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR: "
            "element-incompatible predicted role assignment"
        )

    # Physical classification: Aut(G_mol)^K gauge, not literal equality.
    if oracle_role is not None and automorphism_permutations is not None:
        if per_copy_auto and target_defined and not structural_r_error:
            status = "GAUGE_EQUIVALENT_R"
            structural_r_error = False
        else:
            status = "STRUCTURALLY_INCORRECT_R"
            structural_r_error = True
            if not target_defined and target_reason is None:
                target_reason = "TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR"
            elif target_defined and not per_copy_auto:
                # Copy-bijective but not Aut-equivalent (e.g. non-automorphic role perm).
                target_reason = (
                    target_reason
                    or "STRUCTURALLY_INCORRECT_R: not equivalent under Aut(G_mol)^K"
                )
    else:
        # Without Aut tables, assembly constructibility is the only structural signal.
        if target_defined and not structural_r_error:
            status = "GAUGE_EQUIVALENT_R"
        else:
            status = "STRUCTURALLY_INCORRECT_R"
            structural_r_error = True
            if target_reason is None:
                target_reason = "TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR"

    # Oracle projected-graph exact is evaluation-side; leave false unless caller fills it.
    oracle_projected_graph_exact = False

    audit = PredictedRoleAudit(
        status=status,
        role_capacity_valid=role_capacity_valid,
        role_sizes=role_sizes,
        element_compatible=element_compatible,
        literal_accuracy=literal_accuracy,
        literal_exact=literal_exact,
        orbit_role_accuracy=orbit_role_accuracy,
        orbit_role_exact=orbit_role_exact,
        per_copy_automorphism_equivalent=per_copy_auto,
        physical_role_assignment_exact=physical_role_assignment_exact,
        oracle_projected_graph_exact=oracle_projected_graph_exact,
        structural_r_error=structural_r_error,
        target_defined=target_defined,
        target_reason=target_reason,
        literal_metrics_scope="DIAGNOSTIC_ONLY",
    )
    return target, audit


def permutations_to_group(
    target: AssemblyTarget, permutations: dict[int, torch.Tensor] | None = None
) -> torch.Tensor:
    """Construct G[N,K] from role permutations under the package convention."""
    states = target.permutations if permutations is None else permutations
    if set(states) != set(range(target.M)):
        raise ValueError("a permutation is required for every molecular role")
    G = torch.zeros(
        target.M * target.K,
        target.K,
        dtype=torch.float32,
        device=next(iter(target.role_sets.values())).device,
    )
    for role in range(target.M):
        nodes, state = target.role_sets[role], states[role].long()
        if state.shape != (target.K,) or not torch.equal(
            torch.sort(state).values, torch.arange(target.K, device=state.device)
        ):
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
    values = torch.as_tensor(k_values, dtype=torch.long).flatten()
    if len(values) == 0 or (values < 1).any():
        raise ValueError("batch K values must be non-empty positive integers")
    if len(torch.unique(values)) != 1:
        raise ValueError(f"mixed K batch is unsupported by this MVP: {values.tolist()}")
    return int(values[0])
