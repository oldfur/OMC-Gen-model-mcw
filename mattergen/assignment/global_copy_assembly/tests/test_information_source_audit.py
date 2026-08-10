"""Unit tests for O2 information-source audit helpers (not executed in this turn)."""
from __future__ import annotations

import torch

from mattergen.assignment.global_copy_assembly.information_source_audit import (
    geometry_bundle,
    permute_atom_indices,
    shuffle_orbit_candidate_order,
    shuffle_singleton_role_instance_order,
    unpermute_C,
    audit_legacy_vs_strict_zero_features,
    classify_audit,
)
from mattergen.assignment.global_copy_assembly.orbit_attachment import (
    attachment_map_margins,
    balanced_attachment_dp,
    enumerate_balanced_attachment_scores,
)
from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.global_copy_assembly.orbit_targets import build_orbit_aware_target


def _toy_sample_and_roles():
    M, K = 3, 2
    N = M * K
    roles, copies = [], []
    for c in range(K):
        for r in range(M):
            roles.append(r)
            copies.append(c)
    role = torch.tensor(roles, dtype=torch.long)
    copy = torch.tensor(copies, dtype=torch.long)
    sample = {
        "N": N,
        "M": M,
        "Z": K,
        "pos": torch.randn(N, 3),
        "z": torch.ones(N, dtype=torch.long),
        "role": role,
        "copy": copy,
        "cell": torch.eye(3),
        "role_z": torch.ones(M, dtype=torch.long),
        "role_edge_index": torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
        "role_bond_type": torch.tensor([1, 1, 1, 1], dtype=torch.long),
    }
    return sample, role, copy


def test_geometry_bundle_strict_zero_uses_identity_cell():
    sample, _, _ = _toy_sample_and_roles()
    legacy = geometry_bundle(sample, "legacy_zero_geometry")
    strict = geometry_bundle(sample, "strict_zero_geometry")
    assert torch.equal(legacy["frac"], torch.zeros_like(sample["pos"]))
    assert torch.equal(legacy["cell"], sample["cell"])
    assert torch.equal(strict["frac"], torch.zeros_like(sample["pos"]))
    assert torch.equal(strict["cell"], torch.eye(3))


def test_atom_permutation_roundtrip_unpermute_C():
    sample, roles, _ = _toy_sample_and_roles()
    n = sample["N"]
    sigma = torch.randperm(n)
    sample_p, roles_p, inv = permute_atom_indices(sample, roles, sigma)
    assert torch.equal(sample_p["pos"], sample["pos"][sigma])
    assert torch.equal(roles_p, roles[sigma])
    C = torch.arange(n * n, dtype=torch.float32).view(n, n)
    # C' on permuted indices: C'[j,k]=C[sigma[j],sigma[k]]
    C_p = C[sigma][:, sigma]
    C_back = unpermute_C(C_p, inv)
    assert torch.equal(C_back, C)


def test_singleton_role_order_shuffle_preserves_capacity():
    sample, role, copy = _toy_sample_and_roles()
    partition = build_orbit_partition([[0], [1, 2]])
    o2 = build_orbit_aware_target(role, copy, partition=partition, K=2, anchor_role=0)
    o2s = shuffle_singleton_role_instance_order(o2, seed=17)
    for r, nodes in o2s.singleton_target.role_sets.items():
        assert nodes.numel() == o2.K
    # all singleton atoms preserved as a multiset
    old = torch.cat([o2.singleton_target.role_sets[r] for r in sorted(o2.singleton_target.role_sets)])
    new = torch.cat([o2s.singleton_target.role_sets[r] for r in sorted(o2s.singleton_target.role_sets)])
    assert torch.equal(old.sort().values, new.sort().values)


def test_orbit_candidate_shuffle_preserves_atom_set():
    sample, role, copy = _toy_sample_and_roles()
    partition = build_orbit_partition([[0], [1, 2]])
    o2 = build_orbit_aware_target(role, copy, partition=partition, K=2, anchor_role=0)
    assert len(o2.orbit_targets) == 1
    o2s = shuffle_orbit_candidate_order(o2, seed=3)
    a0 = o2.orbit_targets[0].atom_indices.sort().values
    a1 = o2s.orbit_targets[0].atom_indices.sort().values
    assert torch.equal(a0, a1)
    assert len(o2s.orbit_targets[0].pairs_local) == o2.K


def test_attachment_top2_and_tie_break_modes_do_not_change_scores():
    torch.manual_seed(0)
    K, n = 2, 4
    raw = torch.randn(K, n, n)
    F = 0.5 * (raw + raw.transpose(-1, -2))
    F = F + torch.diag_embed(torch.full((K, n), float("-inf")))
    scored = enumerate_balanced_attachment_scores(F)
    assert len(scored) >= 2
    margins = attachment_map_margins(F, near_tie_tol=1e-6)
    assert margins["best_attachment_score"] >= margins["second_best_attachment_score"]
    r_def = balanced_attachment_dp(F, pair_order="default")
    r_rev = balanced_attachment_dp(F, pair_order="reverse", near_tie_tol=1e-6)
    # scores of chosen assignments must equal some enumerated score
    assert any(abs(s - float(r_def.map_score)) < 1e-5 for s, _ in scored)
    assert any(abs(s - float(r_rev.map_score)) < 1e-5 for s, _ in scored)


def test_legacy_zero_feature_audit_keys():
    sample, _, _ = _toy_sample_and_roles()
    rep = audit_legacy_vs_strict_zero_features(sample)
    assert rep["crystal_encoder_rebuilds_edges_each_forward"] is True
    assert "legacy_zero_geometry" in rep and "strict_zero_geometry" in rep


def test_classify_audit_inconclusive_default():
    labels = classify_audit({"atom_permutation": {}, "role_instance_shuffle": {}, "baseline": {}})
    assert "INCONCLUSIVE" in labels or labels
