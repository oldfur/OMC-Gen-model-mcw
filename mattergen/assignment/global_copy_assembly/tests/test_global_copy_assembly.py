"""Tests intentionally authored but not executed in the implementation-only turn."""
from __future__ import annotations

import itertools
import pytest
import torch

from mattergen.assignment.global_copy_assembly.decoder import decode_copy_assembly
from mattergen.assignment.global_copy_assembly.metrics import projected_molecular_bonds
from mattergen.assignment.global_copy_assembly.module import GlobalCopyAssemblyConfig, GlobalStructuredCopyAssembly
from mattergen.assignment.global_copy_assembly.pair_potential import BondPairPotential, permutation_factor
from mattergen.assignment.global_copy_assembly.permutations import compose, enumerate_permutations, identity_index, inverse_permutations
from mattergen.assignment.global_copy_assembly.targets import (
    build_assembly_target,
    build_assembly_target_from_predicted_roles,
    per_copy_automorphism_equivalent,
    permutations_to_group,
    target_state_indices,
    validate_uniform_batch_k,
)
from mattergen.assignment.global_copy_assembly.tree_builder import build_bfs_tree, select_anchor_role
from mattergen.assignment.global_copy_assembly.tree_crf import TreeCRF


def synthetic_target(M=3,K=2):
    # Stable atom indices are deliberately interleaved across roles/copies.
    role=torch.tensor([0,1,2,0,1,2]);copy=torch.tensor([0,1,0,1,0,1])
    return build_assembly_target(role,copy,M=M,K=K,anchor_role=0),role,copy


def _rhodin_like_layout(*, M: int = 3, K: int = 4):
    """Atoms grouped by copy then role: index = copy * M + role."""
    roles = []
    copies = []
    for copy_id in range(K):
        for role in range(M):
            roles.append(role)
            copies.append(copy_id)
    oracle = torch.tensor(roles, dtype=torch.long)
    copy = torch.tensor(copies, dtype=torch.long)
    # Roles 1/2 share an element so Aut swap(1,2) preserves the element mask.
    role_z = torch.tensor([1, 1, 1][:M], dtype=torch.long)
    z = role_z[oracle]
    # Aut(G_mol) = {id, swap(1,2)}; orbit partition [[0],[1,2]].
    perms = [[0, 1, 2], [0, 2, 1]]
    orbits = [[0], [1, 2], [1, 2]]
    return oracle, copy, role_z, z, perms, orbits


def _apply_per_copy_swaps(oracle: torch.Tensor, copy: torch.Tensor, swap_copies: set[int]) -> torch.Tensor:
    pred = oracle.clone()
    for i in range(int(oracle.numel())):
        if int(copy[i]) in swap_copies and int(oracle[i]) in (1, 2):
            pred[i] = 2 if int(oracle[i]) == 1 else 1
    return pred


def test_permutation_convention_round_trip_and_identity():
    states=enumerate_permutations(3);inverse=inverse_permutations(states)
    assert len(states)==6 and torch.equal(states[identity_index(states)],torch.arange(3))
    for p,inv in zip(states,inverse): assert torch.equal(compose(p,inv),torch.arange(3))


def test_target_constructs_permutations_and_exact_C():
    target,_,copy=synthetic_target();G=permutations_to_group(target);C=G@G.T;C0=copy[:,None].eq(copy[None,:]).float()
    assert torch.equal(C,C0) and torch.equal(target.permutations[target.anchor_role],torch.arange(target.K))


def brute_force_logz(tree,factors,states,identity):
    values=[]
    for choice in itertools.product(range(states),repeat=len(tree.preorder)-1):
        assignment={tree.root:identity};assignment.update({role:value for role,value in zip(tree.preorder[1:],choice)})
        values.append(sum(factors[e][assignment[e[0]],assignment[e[1]]] for e in tree.tree_edges))
    return torch.logsumexp(torch.stack(values),0),torch.stack(values).max()


def test_tree_crf_matches_bruteforce_and_has_finite_gradient():
    tree=build_bfs_tree(torch.tensor([[0,1,1],[1,2,3]]),M=4,root=0);states=enumerate_permutations(2);factors={edge:torch.randn(2,2,requires_grad=True) for edge in tree.tree_edges};crf=TreeCRF(tree,num_states=2,identity_state=identity_index(states))
    logz,best=brute_force_logz(tree,factors,2,crf.identity_state)
    assert torch.allclose(crf.log_partition(factors),logz)
    decoded=crf.map_decode(factors);assert torch.allclose(decoded.score,best)
    target={0:crf.identity_state,1:0,2:1,3:0};loss=crf.nll(factors,target);loss.backward();assert all(torch.isfinite(x.grad).all() for x in factors.values())


def test_tree_crf_nll_is_stable_under_large_per_edge_score_shifts():
    tree=build_bfs_tree(torch.tensor([[0],[1]]),M=2,root=0);states=enumerate_permutations(2);crf=TreeCRF(tree,num_states=2,identity_state=identity_index(states));target={0:crf.identity_state,1:0}
    base={(0,1):torch.tensor([[0.0,-3.0],[-2.0,1.0]],dtype=torch.float64,requires_grad=True)}
    shifted={(0,1):base[(0,1)]+1.0e12}
    base_nll=crf.nll(base,target);shifted_nll=crf.nll(shifted,target);logz,target_score=crf.log_partition_and_target_score(shifted,target)
    assert torch.allclose(base_nll,shifted_nll,atol=1e-8)
    assert shifted_nll >= 0 and target_score-logz <= 1e-10
    shifted_nll.backward();assert torch.isfinite(base[(0,1)].grad).all()


def test_bond_pair_potential_bounds_pair_scores():
    potential=BondPairPotential(hidden=4,pair_hidden=8,rbf_dim=4,score_scale=3.0)
    score=potential(torch.randn(2,4),torch.randn(2,4),torch.randn(4),torch.randn(4),1,torch.rand(2,3),torch.rand(2,3),torch.eye(3))
    assert score.shape == (2,2) and score.abs().max() <= 3.0


def test_oracle_pair_scores_recover_C_and_projected_graph():
    target,role,copy=synthetic_target();states=enumerate_permutations(2);inverse=inverse_permutations(states);tree=build_bfs_tree(torch.tensor([[0,1],[1,2]]),M=3,root=0)
    factors={}
    for parent,child in tree.tree_edges:
        score=torch.full((2,2),-10.);p,q=target.permutations[parent],target.permutations[child];invq=torch.empty_like(q);invq[q]=torch.arange(2);score[torch.arange(2),invq[p]]=10.;factors[(parent,child)]=permutation_factor(score,states,inverse)
    crf=TreeCRF(tree,num_states=len(states),identity_state=identity_index(states));G,C=decode_copy_assembly(target,states,crf.map_decode(factors).state_indices);assert torch.equal(C,copy[:,None].eq(copy[None,:]).float()) and torch.equal(G.sum(0),torch.full((2,),3.))
    R=torch.nn.functional.one_hot(role,3).float();B=torch.zeros(3,3,1);B[0,1]=B[1,0]=B[1,2]=B[2,1]=1.;assert projected_molecular_bonds(R,C,B).any()


def test_copy_label_gauge_leaves_C_invariant():
    target,_,_=synthetic_target();G=permutations_to_group(target);sigma=torch.tensor([1,0]);assert torch.equal(G@G.T,(G[:,sigma])@(G[:,sigma]).T)


def test_bond_type_lookup_accepts_consistent_bidirectional_bond():
    edges=torch.tensor([[9,1],[1,9]])
    bond_types=torch.tensor([1,1])
    assert GlobalStructuredCopyAssembly._bond_type(edges,bond_types,9,1)==1


def test_bond_type_lookup_rejects_inconsistent_duplicate_bond():
    edges=torch.tensor([[9,1],[1,9]])
    bond_types=torch.tensor([1,2])
    with pytest.raises(ValueError,match="inconsistent molecular bond types"):
        GlobalStructuredCopyAssembly._bond_type(edges,bond_types,9,1)


def test_non_bijective_decoded_permutation_fails_loudly():
    target,_,_=synthetic_target();bad={**target.permutations,1:torch.tensor([0,0])}
    with pytest.raises(ValueError): permutations_to_group(target,bad)


@pytest.mark.parametrize("bad",[torch.tensor([0,0,0,1,1,2]),torch.tensor([0,1,2,0,1])])
def test_bad_role_capacity_fails_loudly(bad):
    with pytest.raises(ValueError): build_assembly_target(bad,torch.tensor([0,1,0,1,0,1]),M=3,K=2,anchor_role=0)


def test_failure_modes_fail_loudly():
    with pytest.raises(ValueError): select_anchor_role([[0,1]],torch.tensor([[0],[1]]))
    with pytest.raises(ValueError): build_bfs_tree(torch.tensor([[0],[1]]),M=3,root=0)
    with pytest.raises(ValueError): validate_uniform_batch_k([2,3])
    with pytest.raises(ValueError): build_assembly_target(torch.tensor([0,1,2,0,1,2]),torch.tensor([0,0,0,0,0,0]),M=3,K=2,anchor_role=0)
    states=enumerate_permutations(2);tree=build_bfs_tree(torch.tensor([[0],[1]]),M=2,root=0);crf=TreeCRF(tree,num_states=2,identity_state=0)
    with pytest.raises(FloatingPointError): crf.log_partition({(0,1):torch.full((2,2),float("nan"))})


def test_predicted_role_audit_marks_gauge_equivalent_roles():
    target, _, copy = synthetic_target()
    oracle = torch.tensor([0, 1, 2, 0, 1, 2])
    perms = [[0, 1, 2], [0, 2, 1]]
    orbits = [[0], [1, 2], [1, 2]]
    # role_z is the molecular table [M], never an atom-length vector.
    target2, audit = build_assembly_target_from_predicted_roles(
        oracle,
        copy,
        M=3,
        K=2,
        anchor_role=0,
        role_z=torch.tensor([1, 1, 1]),
        z=torch.tensor([1, 1, 1, 1, 1, 1]),
        oracle_role=oracle,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )
    assert target2 is not None and audit.status == "GAUGE_EQUIVALENT_R"
    assert audit.element_compatible and audit.target_defined
    assert audit.literal_metrics_scope == "DIAGNOSTIC_ONLY"
    assert audit.literal_accuracy == 1.0
    assert audit.physical_role_assignment_exact and audit.per_copy_automorphism_equivalent
    G = permutations_to_group(target2)
    assert torch.equal(G @ G.T, copy[:, None].eq(copy[None, :]).float())


def test_predicted_role_element_check_indexes_role_z_by_role_labels_not_atom_ids():
    """Regression: role_sets atom indices (e.g. 11) must not index role_z [M=10]."""
    M, K, N = 10, 4, 40
    role = torch.arange(N) % M
    copy = torch.arange(N) // M
    role_z = torch.arange(M) + 1
    z = role_z[role]
    # Atom index 11 appears in some role set; indexing role_z with atom ids would OOB.
    assert int((role == 1).nonzero().flatten().max()) >= 11 or N > M
    identity = [list(range(M))]
    orbits = [[r] for r in range(M)]
    target, audit = build_assembly_target_from_predicted_roles(
        role,
        copy,
        M=M,
        K=K,
        anchor_role=0,
        role_z=role_z,
        z=z,
        oracle_role=role,
        automorphism_permutations=identity,
        role_orbits=orbits,
    )
    assert target is not None
    assert audit.element_compatible
    assert audit.target_defined
    assert audit.status == "GAUGE_EQUIVALENT_R"


def test_A_independent_per_copy_automorphism_gauge():
    """Test A: independent per-copy 1↔2 swaps remain GAUGE_EQUIVALENT_R with C*=C0."""
    M, K = 3, 4
    oracle, copy, role_z, z, perms, orbits = _rhodin_like_layout(M=M, K=K)
    # copy0 identity, copy1 swap, copy2 identity, copy3 swap
    predicted = _apply_per_copy_swaps(oracle, copy, swap_copies={1, 3})
    assert not torch.equal(predicted, oracle)
    # Capacity preserved: each role still has exactly K instances.
    for role in range(M):
        assert int((predicted == role).sum()) == K
    target, audit = build_assembly_target_from_predicted_roles(
        predicted,
        copy,
        M=M,
        K=K,
        anchor_role=0,
        role_z=role_z,
        z=z,
        oracle_role=oracle,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )
    assert audit.status == "GAUGE_EQUIVALENT_R"
    assert audit.per_copy_automorphism_equivalent
    assert audit.physical_role_assignment_exact
    assert audit.target_defined and target is not None
    assert audit.literal_exact is False
    assert audit.literal_metrics_scope == "DIAGNOSTIC_ONLY"
    G = permutations_to_group(target)
    C0 = copy[:, None].eq(copy[None, :]).float()
    assert torch.equal(G @ G.T, C0)


def test_B_capacity_violation_fails_structural():
    """Test B: role1=K+1, role2=K-1 is illegal; Aut cannot legalize it."""
    M, K = 3, 4
    oracle, copy, role_z, z, perms, orbits = _rhodin_like_layout(M=M, K=K)
    bad = oracle.clone()
    # Move one role-2 atom onto role-1 → capacity (K+1, K-1).
    idx_role2 = (bad == 2).nonzero(as_tuple=False).flatten()[0]
    bad[idx_role2] = 1
    assert int((bad == 1).sum()) == K + 1
    assert int((bad == 2).sum()) == K - 1
    target, audit = build_assembly_target_from_predicted_roles(
        bad,
        copy,
        M=M,
        K=K,
        anchor_role=0,
        role_z=role_z,
        z=role_z[bad.clamp(0, M - 1)],
        oracle_role=oracle,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )
    assert target is None
    assert audit.status == "STRUCTURALLY_INCORRECT_R"
    assert not audit.role_capacity_valid
    assert audit.structural_r_error
    assert not audit.target_defined
    assert audit.target_reason is not None
    assert "TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR" in audit.target_reason


def test_C_cross_copy_compensation_is_structurally_incorrect():
    """Test C: global capacity OK but copy has 1,1 and another 2,2 → structural."""
    M, K = 3, 4
    oracle, copy, role_z, z, perms, orbits = _rhodin_like_layout(M=M, K=K)
    bad = oracle.clone()
    # Within copy 0: force both former roles 1 and 2 to role 1.
    # Within copy 2: force both to role 2. Global |V_1|=|V_2|=K still holds.
    for i in range(int(bad.numel())):
        if int(copy[i]) == 0 and int(oracle[i]) in (1, 2):
            bad[i] = 1
        if int(copy[i]) == 2 and int(oracle[i]) in (1, 2):
            bad[i] = 2
    assert int((bad == 1).sum()) == K
    assert int((bad == 2).sum()) == K
    assert not per_copy_automorphism_equivalent(bad, oracle, copy, perms)
    target, audit = build_assembly_target_from_predicted_roles(
        bad,
        copy,
        M=M,
        K=K,
        anchor_role=0,
        role_z=role_z,
        z=role_z[bad],
        oracle_role=oracle,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )
    assert audit.status == "STRUCTURALLY_INCORRECT_R"
    assert audit.structural_r_error
    assert not audit.per_copy_automorphism_equivalent
    assert not audit.physical_role_assignment_exact
    assert target is None
    assert not audit.target_defined
    assert audit.target_reason is not None
    assert "TARGET_UNDEFINED_DUE_TO_STRUCTURAL_R_ERROR" in audit.target_reason


def test_D_no_canonicalization_leakage_r_effective_equals_artifact():
    """Test D: R_effective from target.role_sets equals artifact and differs from R0."""
    M, K = 3, 4
    oracle, copy, role_z, z, perms, orbits = _rhodin_like_layout(M=M, K=K)
    artifact = _apply_per_copy_swaps(oracle, copy, swap_copies={1})
    assert not torch.equal(artifact, oracle)
    target, audit = build_assembly_target_from_predicted_roles(
        artifact,
        copy,
        M=M,
        K=K,
        anchor_role=0,
        role_z=role_z,
        z=z,
        oracle_role=oracle,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )
    assert target is not None
    effective = torch.empty_like(artifact)
    for role, nodes in target.role_sets.items():
        effective[nodes] = int(role)
    assert torch.equal(effective, artifact)
    assert not torch.equal(effective, oracle)
    assert audit.status == "GAUGE_EQUIVALENT_R"
    assert audit.literal_exact is False


def test_E_predicted_gauge_target_round_trip_to_C0():
    """Test E: legal gauge-swapped R yields P*→G*→C*=C0 under predicted gauge."""
    M, K = 3, 4
    oracle, copy, role_z, z, perms, orbits = _rhodin_like_layout(M=M, K=K)
    predicted = _apply_per_copy_swaps(oracle, copy, swap_copies={0, 2, 3})
    target, audit = build_assembly_target_from_predicted_roles(
        predicted,
        copy,
        M=M,
        K=K,
        anchor_role=0,
        role_z=role_z,
        z=z,
        oracle_role=oracle,
        automorphism_permutations=perms,
        role_orbits=orbits,
    )
    assert audit.target_defined and target is not None
    assert audit.status == "GAUGE_EQUIVALENT_R"
    for role, state in target.permutations.items():
        assert torch.equal(torch.sort(state).values, torch.arange(K))
    G = permutations_to_group(target)
    C0 = copy[:, None].eq(copy[None, :]).float()
    assert torch.equal(G @ G.T, C0)
    # Role sets remain predicted gauge (not oracle R0 sets).
    for role in range(M):
        assert torch.equal(
            target.role_sets[role],
            (predicted == role).nonzero(as_tuple=False).flatten().sort().values,
        )


def test_predicted_role_structural_capacity_error_is_audited_not_raised():
    copy = torch.tensor([0, 1, 0, 1, 0, 1])
    bad = torch.tensor([0, 0, 0, 0, 0, 0])
    target, audit = build_assembly_target_from_predicted_roles(
        bad,
        copy,
        M=3,
        K=2,
        anchor_role=0,
        role_z=torch.tensor([1, 1, 1]),
        z=torch.tensor([1, 1, 1, 1, 1, 1]),
        oracle_role=torch.tensor([0, 1, 2, 0, 1, 2]),
        automorphism_permutations=[[0, 1, 2], [0, 2, 1]],
        role_orbits=[[0], [1, 2], [1, 2]],
    )
    assert target is None
    assert audit.structural_r_error
    assert not audit.target_defined
    assert audit.status == "STRUCTURALLY_INCORRECT_R"
    assert not audit.role_capacity_valid


def test_predicted_role_audit_fails_loudly_when_target_cannot_be_constructed():
    with pytest.raises(ValueError):
        build_assembly_target(
            torch.tensor([0, 0, 0, 0, 0, 0]),
            torch.tensor([0, 1, 0, 1, 0, 1]),
            M=3,
            K=2,
            anchor_role=0,
        )


def test_predicted_role_source_isolated_from_oracle_copy_supervision():
    cfg = GlobalCopyAssemblyConfig(
        mode="clean_geometry_predicted_r",
        role_source="geometry_only_hard_r",
        use_oracle_role_assignment=False,
        use_oracle_copy_relation=False,
        use_copy_id_as_input=False,
    )
    assert cfg.mode == "clean_geometry_predicted_r"
    assert cfg.role_source == "geometry_only_hard_r"
    assert not cfg.use_oracle_role_assignment
    assert not cfg.use_oracle_copy_relation
    assert not cfg.use_copy_id_as_input
