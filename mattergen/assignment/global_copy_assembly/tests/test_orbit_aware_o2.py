"""Tests for orbit-aware O2 assembly (authored; not executed in this implementation turn)."""
from __future__ import annotations

import inspect
import torch

from mattergen.assignment.global_copy_assembly.orbit_attachment import (
    OrbitAttachmentHead,
    balanced_attachment_dp,
    brute_force_balanced_logz_map,
)
from mattergen.assignment.global_copy_assembly.orbit_membership import (
    build_orbit_partition,
    collapse_roles_to_orbit_membership,
    validate_orbit_copy_capacity,
    validate_orbit_global_capacity,
)
from mattergen.assignment.global_copy_assembly.orbit_module import OrbitAwareAssemblyConfig, OrbitAwareCopyAssembly
from mattergen.assignment.global_copy_assembly.orbit_targets import (
    build_orbit_aware_target,
    build_singleton_target_from_roles,
)
from mattergen.assignment.global_copy_assembly.singleton_backbone import build_singleton_backbone


def _layout_m3_k4():
    """Atoms: copy-major then role. Roles 1,2 form Aut orbit."""
    M, K = 3, 4
    roles, copies = [], []
    for c in range(K):
        for r in range(M):
            roles.append(r)
            copies.append(c)
    oracle = torch.tensor(roles, dtype=torch.long)
    copy = torch.tensor(copies, dtype=torch.long)
    # edges 0-1, 0-2, 1-2 so singleton {0} connects to 1,2; without 1,2 singleton graph is isolated
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    bond_type = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    role_z = torch.tensor([1, 1, 1], dtype=torch.long)
    orbits = [[0], [1, 2]]
    return oracle, copy, edge_index, bond_type, role_z, orbits, M, K


def test_1_orbit_collapse_from_cross_copy_compensation():
    """Canonical 1,1 / 2,2 still collapses to |V_[1,2]|=2K."""
    oracle, copy, _edge, _bt, _rz, _orb, M, K = _layout_m3_k4()
    bad = oracle.clone()
    # copy0: both 1; copy2: both 2; global capacity still K each
    for i in range(len(bad)):
        if int(copy[i]) == 0 and int(oracle[i]) in (1, 2):
            bad[i] = 1
        if int(copy[i]) == 2 and int(oracle[i]) in (1, 2):
            bad[i] = 2
    assert int((bad == 1).sum()) == K
    assert int((bad == 2).sum()) == K
    partition = build_orbit_partition([[0], [1, 2]])
    bar = collapse_roles_to_orbit_membership(bad, partition)
    assert bar.shape == (M * K, 2)
    v12 = int((bar[:, 1] > 0.5).sum())
    assert v12 == 2 * K
    cap = validate_orbit_global_capacity(bar, partition, K=K)
    assert cap["valid"] is True


def test_2_orbit_copy_capacity_on_merged_G():
    oracle, copy, edge_index, bond_type, role_z, orbits, M, K = _layout_m3_k4()
    partition = build_orbit_partition(orbits)
    N = M * K
    G = torch.zeros(N, K)
    for i in range(N):
        G[i, int(copy[i])] = 1.0
    bar = collapse_roles_to_orbit_membership(oracle, partition)
    cap = validate_orbit_copy_capacity(G, bar, partition)
    assert cap["valid"] is True
    # singleton orbit 0: 1 per copy; orbit 1,2: 2 per copy
    assert cap["expected_per_copy"] == [1, 2]


def test_3_gauge_invariance_of_attachment_score():
    head = OrbitAttachmentHead(hidden=8, rbf_dim=4, score_scale=5.0, gauge_marginalization="logsumexp")
    h_i = torch.randn(8)
    h_j = torch.randn(8)
    frac_i = torch.rand(3)
    frac_j = torch.rand(3)
    h_copy = torch.randn(3, 8)
    frac_copy = torch.rand(3, 3)
    cell = torch.eye(3)
    s_ij = head.pair_score(
        h_i=h_i, h_j=h_j, frac_i=frac_i, frac_j=frac_j,
        h_copy=h_copy, frac_copy=frac_copy, cell=cell, orbit_id=0,
    )
    s_ji = head.pair_score(
        h_i=h_j, h_j=h_i, frac_i=frac_j, frac_j=frac_i,
        h_copy=h_copy, frac_copy=frac_copy, cell=cell, orbit_id=0,
    )
    assert torch.allclose(s_ij, s_ji, atol=1e-5)


def test_4_bitmask_dp_matches_bruteforce():
    torch.manual_seed(0)
    K, n = 3, 6
    raw = torch.randn(K, n, n)
    F = 0.5 * (raw + raw.transpose(-1, -2))
    F = F + torch.diag_embed(torch.full((K, n), float("-inf")))
    dp = balanced_attachment_dp(F, atoms_per_copy=2)
    log_z_bf, map_bf, assign_bf = brute_force_balanced_logz_map(F, atoms_per_copy=2)
    assert torch.allclose(dp.log_partition, log_z_bf, atol=1e-4)
    assert torch.allclose(dp.map_score, map_bf, atol=1e-4)
    # MAP score of reconstructed pairs equals map_score
    recon = sum(F[k, i, j] for k, (i, j) in enumerate(dp.map_pairs))
    assert torch.allclose(recon, dp.map_score, atol=1e-4)


def test_5_oracle_attachment_recovers_true_partition():
    K, n = 2, 4
    F = torch.full((K, n, n), -10.0)
    # true pairs: copy0 -> (0,1), copy1 -> (2,3)
    F[0, 0, 1] = F[0, 1, 0] = 5.0
    F[1, 2, 3] = F[1, 3, 2] = 5.0
    F = F + torch.diag_embed(torch.full((K, n), float("-inf")))
    result = balanced_attachment_dp(F, atoms_per_copy=2)
    assert set(result.map_pairs) == {(0, 1), (2, 3)}
    assert result.map_pairs[0] == (0, 1)
    assert result.map_pairs[1] == (2, 3)


def test_6_full_G_merge_row_and_orbit_capacity():
    oracle, copy, edge_index, bond_type, role_z, orbits, M, K = _layout_m3_k4()
    partition = build_orbit_partition(orbits)
    bar = collapse_roles_to_orbit_membership(oracle, partition)
    N = M * K
    G = torch.zeros(N, K)
    # singleton role 0
    for i in range(N):
        if int(oracle[i]) == 0:
            G[i, int(copy[i])] = 1.0
    # orbit atoms: 2 per copy
    for i in range(N):
        if int(oracle[i]) in (1, 2):
            G[i, int(copy[i])] = 1.0
    assert torch.allclose(G.sum(-1), torch.ones(N))
    assert validate_orbit_copy_capacity(G, bar, partition)["valid"] is True


def test_7_no_copy_id_in_model_forward_signature():
    cfg = OrbitAwareAssemblyConfig()
    model = OrbitAwareCopyAssembly(cfg)
    # encode / loss / map_decode signatures must not accept mol_copy_id
    for name in ("encode", "loss", "map_decode", "build_attachment_scores_from_G"):
        sig = inspect.signature(getattr(model, name))
        assert "mol_copy_id" not in sig.parameters
        assert "copy" not in sig.parameters
        assert "C0" not in sig.parameters
    assert cfg.use_copy_id_as_input is False


def test_singleton_target_builds_under_gauge_swap_on_orbit_roles_only():
    """Singleton Stage A target still defined when only 1/2 are gauge-swapped."""
    oracle, copy, edge_index, bond_type, role_z, orbits, M, K = _layout_m3_k4()
    pred = oracle.clone()
    for i in range(len(pred)):
        if int(copy[i]) in (1, 3) and int(oracle[i]) in (1, 2):
            pred[i] = 2 if int(oracle[i]) == 1 else 1
    # singleton role 0 unchanged and bijective
    target = build_singleton_target_from_roles(
        pred, copy, singleton_roles=[0], M=M, K=K, anchor_role=0
    )
    assert target.M == 1
    assert torch.equal(target.permutations[0], torch.arange(K))


def test_o2_target_from_cross_copy_compensation_orbit_ok_singleton_ok():
    """After orbit collapse path: singleton target + orbit attachment targets."""
    oracle, copy, edge_index, bond_type, role_z, orbits, M, K = _layout_m3_k4()
    bad = oracle.clone()
    for i in range(len(bad)):
        if int(copy[i]) == 0 and int(oracle[i]) in (1, 2):
            bad[i] = 1
        if int(copy[i]) == 2 and int(oracle[i]) in (1, 2):
            bad[i] = 2
    partition = build_orbit_partition(orbits)
    # Singleton still fine; orbit attachment target uses true copies of orbit atoms.
    o2 = build_orbit_aware_target(bad, copy, partition=partition, K=K, anchor_role=0)
    assert o2.bar_r.shape[1] == 2
    assert len(o2.orbit_targets) == 1
    ot = o2.orbit_targets[0]
    assert int(ot.atom_indices.numel()) == 2 * K
    assert len(ot.pairs_local) == K


def test_singleton_backbone_allows_virtual_edges_when_disconnected():
    # Only role 0 is singleton; roles 1,2 removed from singleton set — graph among {0} is trivial.
    # Two singletons 0 and 3 with no direct edge; path 0-1-3 virtual.
    edge_index = torch.tensor([[0, 1, 1, 3], [1, 0, 3, 1]], dtype=torch.long)
    bond_type = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    backbone = build_singleton_backbone(
        edge_index, bond_type, M=4, singleton_roles=[0, 3], anchor_role=0
    )
    assert backbone.tree.root == 0
    assert len(backbone.tree.tree_edges) == 1
    assert len(backbone.virtual_tree_edges) == 1
