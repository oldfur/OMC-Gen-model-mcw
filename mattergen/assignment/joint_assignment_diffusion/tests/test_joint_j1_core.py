"""J1 core unit tests (authored; not executed in this commit)."""
from __future__ import annotations

import torch

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.joint_assignment_diffusion.legal_moves import (
    enumerate_g_moves,
    enumerate_r_moves,
)
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.state import (
    a_from_role_and_copy,
    sample_uniform_legal_prior,
)
from mattergen.assignment.joint_assignment_diffusion.ctmc import simulate_forward_ctmc
from mattergen.assignment.joint_assignment_diffusion.symmetry import (
    apply_symmetry_to_state,
    sample_symmetry_augment,
)


def _toy_state():
    # M=4 roles, orbits [[0],[1,2],[3]], K=2 → N = sum m_o * K = (1+2+1)*2 = 8
    partition = build_orbit_partition([[0], [1, 2], [3]])
    role_z = torch.tensor([6, 1, 1, 8])  # C, H, H, O
    # expand atoms
    roles = []
    copies = []
    zs = []
    for k in range(2):
        for o, orb in enumerate(partition.orbits):
            for r in orb:
                roles.append(r)
                copies.append(k)
                zs.append(int(role_z[r]))
    role = torch.tensor(roles)
    copy = torch.tensor(copies)
    z = torch.tensor(zs)
    st = a_from_role_and_copy(
        role=role, copy=copy, partition=partition, atomic_numbers=z, role_z=role_z, K=2
    )
    return st, partition, role_z


def test_state_capacity_legal():
    st, _, _ = _toy_state()
    v = st.validate()
    assert v["legal"]


def test_r_move_requires_same_copy_diff_orbit():
    st, _, _ = _toy_state()
    moves = enumerate_r_moves(st)
    for m in moves:
        assert int(st.copy_of()[m.i]) == int(st.copy_of()[m.j])
        assert int(st.orbit_of()[m.i]) != int(st.orbit_of()[m.j])
        assert int(st.atomic_numbers[m.i]) == int(st.atomic_numbers[m.j])


def test_g_move_requires_same_orbit_diff_copy():
    st, _, _ = _toy_state()
    moves = enumerate_g_moves(st)
    for m in moves:
        assert int(st.orbit_of()[m.i]) == int(st.orbit_of()[m.j])
        assert int(st.copy_of()[m.i]) != int(st.copy_of()[m.j])


def test_async_schedule_windows_and_integral():
    sch = AsyncJumpSchedule(r_window=(0.60, 0.95), g_window=(0.35, 0.75), kappa_r=4.0, kappa_g=6.0)
    assert float(sch.beta_r(0.5)) == 0.0
    assert float(sch.beta_g(0.2)) == 0.0
    assert float(sch.beta_r(0.80)) > 0.0
    assert float(sch.beta_g(0.55)) > 0.0
    # endpoints of open interval → 0
    assert float(sch.beta_r(0.60)) == 0.0
    assert float(sch.beta_r(0.95)) == 0.0
    # ∫ β ≈ κ
    assert abs(sch.integrated_beta(0.0, 1.0, kind="R") - 4.0) < 1e-4
    assert abs(sch.integrated_beta(0.0, 1.0, kind="G") - 6.0) < 1e-4


def test_fixed_exit_rate_softmax_sums_to_beta():
    from mattergen.assignment.joint_assignment_diffusion.jump_heads import logits_to_rates
    from mattergen.assignment.joint_assignment_diffusion.legal_moves import LegalMove

    scored = {
        "R": [(LegalMove("R", 0, 1), torch.tensor(0.0)), (LegalMove("R", 2, 3), torch.tensor(1.0))],
        "G": [(LegalMove("G", 0, 2), torch.tensor(-1.0))],
    }
    rates = logits_to_rates(scored, beta_r=3.0, beta_g=2.0)
    assert abs(sum(float(r) for _, r in rates["R"]) - 3.0) < 1e-5
    assert abs(sum(float(r) for _, r in rates["G"]) - 2.0) < 1e-5


def test_forward_ctmc_stays_legal_and_respects_window():
    st, _, _ = _toy_state()
    sch = AsyncJumpSchedule(r_window=(0.60, 0.95), g_window=(0.35, 0.75), kappa_r=2.0, kappa_g=3.0)
    g = torch.Generator().manual_seed(0)
    traj = simulate_forward_ctmc(st, schedule=sch, generator=g)
    for s in traj.states:
        assert s.validate()["legal"]
    for e in traj.events:
        if e.kind == "R":
            assert sch.is_r_active(e.time)
        if e.kind == "G":
            assert sch.is_g_active(e.time)


def test_forward_ctmc_produces_g_and_r_events():
    """G and R share integrated-hazard plumbing; both should fire under κ>0."""
    st, _, _ = _toy_state()
    sch = AsyncJumpSchedule(r_window=(0.60, 0.95), g_window=(0.35, 0.75), kappa_r=4.0, kappa_g=6.0)
    n_R = n_G = 0
    for seed in range(8):
        traj = simulate_forward_ctmc(st, schedule=sch, generator=torch.Generator().manual_seed(seed))
        n_R += sum(1 for e in traj.events if e.kind == "R")
        n_G += sum(1 for e in traj.events if e.kind == "G")
    assert n_G > 0, "G events must not systematically vanish"
    assert n_R > 0, "R events must not systematically vanish"


def test_orbit_slot_copy_perm_equivariance():
    """U and G logits follow copy-column permutation; no copy-id features."""
    import torch.nn as nn

    from mattergen.assignment.joint_assignment_diffusion.conditioning import OrbitSlotCopyContext
    from mattergen.assignment.joint_assignment_diffusion.jump_heads import GJumpHead

    torch.manual_seed(0)
    n, h, k, j = 6, 8, 2, 3
    orbit_of = torch.tensor([0, 1, 2, 0, 1, 2])
    copy_of = torch.tensor([0, 0, 0, 1, 1, 1])
    hid = torch.randn(n, h)
    z_orbit = torch.randn(j, h)
    ctx = OrbitSlotCopyContext(h)
    U, feat, cnt = ctx.slot_table(
        hid, orbit_of=orbit_of, copy_of=copy_of, z_orbit=z_orbit, K=k, J=j
    )
    perm = torch.tensor([1, 0])
    copy2 = perm[copy_of]
    U2, _, _ = ctx.slot_table(
        hid, orbit_of=orbit_of, copy_of=copy2, z_orbit=z_orbit, K=k, J=j
    )
    assert torch.allclose(U2[1], U[0], atol=1e-5)
    assert torch.allclose(U2[0], U[1], atol=1e-5)
    head = GJumpHead(h, copy_context_mode="orbit_slot")
    for p in head.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)
    nn.init.zeros_(head.slot_out[-1].weight)
    nn.init.zeros_(head.slot_out[-1].bias)
    # after last-layer zero, logits are 0 (uniform start) regardless of perm
    u_ex = ctx.exclude_atom_slots(U, cnt, feat, orbit_of, copy_of)
    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([3, 4, 5])
    logits, _ = head.batch_logits_orbit_slot(
        hid, ii, jj, z_orbit=z_orbit, orbit_of=orbit_of, u_excl=u_ex, t_scalar=0.5
    )
    assert torch.allclose(logits, torch.zeros_like(logits), atol=1e-6)
    # nonzero last layer: logits invariant to copy relabel
    torch.nn.init.xavier_uniform_(head.slot_out[-1].weight)
    logits_a, _ = head.batch_logits_orbit_slot(
        hid, ii, jj, z_orbit=z_orbit, orbit_of=orbit_of, u_excl=u_ex, t_scalar=0.5
    )
    feat2 = ctx.atom_slot_feat(hid, orbit_of, z_orbit)
    U2, feat2, cnt2 = ctx.slot_table(
        hid, orbit_of=orbit_of, copy_of=copy2, z_orbit=z_orbit, K=k, J=j
    )
    u_ex2 = ctx.exclude_atom_slots(U2, cnt2, feat2, orbit_of, copy2)
    logits_b, _ = head.batch_logits_orbit_slot(
        hid, ii, jj, z_orbit=z_orbit, orbit_of=orbit_of, u_excl=u_ex2, t_scalar=0.5
    )
    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_g_teacher_support_only_beneficial_and_fallback():
    from mattergen.assignment.joint_assignment_diffusion.g_teacher import improvement_weighted_teacher

    utils = [
        {"key": (0, 1), "u": 0.05, "delta_ari": 0.0, "move": None},
        {"key": (0, 2), "u": 0.04, "delta_ari": 0.0, "move": None},
        {"key": (1, 2), "u": -0.01, "delta_ari": 0.0, "move": None},
    ]
    tch = improvement_weighted_teacher(utils, temperature=0.02)
    q = tch["q"]
    assert abs(float(q.sum()) - 1.0) < 1e-6
    assert float(q[2]) == 0.0
    assert float(q[0]) > float(q[1]) > 0
    assert tch["num_beneficial"] == 2
    utils_none = [{"key": (0, 1), "u": -0.1, "delta_ari": 0.0, "move": None}]
    tch2 = improvement_weighted_teacher(utils_none, temperature=0.02)
    assert tch2["no_beneficial"]
    assert abs(float(tch2["q"].sum()) - 1.0) < 1e-6


def test_event_bin_labels_cover_g_window():
    from mattergen.assignment.joint_assignment_diffusion.reverse_eval import event_bin_name

    assert event_bin_name("G", 0.37) == "[0.35,0.40)"
    assert event_bin_name("G", 0.52) == "[0.50,0.55)"
    assert event_bin_name("G", 0.70) == "[0.70,0.75]"
    assert event_bin_name("R", 0.81) == "[0.80,0.85)"


def test_next_reverse_grid_s_matches_sampler_macrostep():
    from mattergen.assignment.joint_assignment_diffusion.schedule import next_reverse_grid_s

    grid = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    assert abs(next_reverse_grid_s(0.73, grid) - 0.7) < 1e-12
    assert abs(next_reverse_grid_s(0.9, grid) - 0.8) < 1e-12
    assert abs(next_reverse_grid_s(0.05, grid) - 0.0) < 1e-12
    assert abs(next_reverse_grid_s(0.0, grid) - 0.0) < 1e-12


def test_events_on_segment_excludes_outside():
    st, _, _ = _toy_state()
    sch = AsyncJumpSchedule(r_window=(0.60, 0.95), g_window=(0.35, 0.75), kappa_r=4.0, kappa_g=6.0)
    traj = simulate_forward_ctmc(st, schedule=sch, generator=torch.Generator().manual_seed(0))
    # t below R window: no R events on (s, t]
    t, s = 0.55, 0.5
    seg = traj.events_on_segment(s, t)
    assert all(e.kind != "R" for e in seg)
    # empty / inverted
    assert traj.events_on_segment(0.8, 0.8) == []


def test_forward_inverse_hazard_matches_integral():
    sch = AsyncJumpSchedule(r_window=(0.60, 0.95), g_window=(0.35, 0.75), kappa_r=4.0, kappa_g=6.0)
    t0, t1 = 0.40, 0.70
    H = sch.integrated_hazard_total(t0, t1, r_on=False, g_on=True)
    assert H > 0
    # mid-hazard event time
    tau = sch.inverse_integrated_hazard_forward(t0, 0.5 * H, t_high=t1, r_on=False, g_on=True)
    assert tau is not None
    H_left = sch.integrated_hazard_total(t0, float(tau), r_on=False, g_on=True)
    assert abs(H_left - 0.5 * H) < 1e-3


def test_uniform_prior_legal():
    st, partition, role_z = _toy_state()
    prior = sample_uniform_legal_prior(
        partition=partition,
        atomic_numbers=st.atomic_numbers,
        role_z=role_z,
        K=st.K,
        generator=torch.Generator().manual_seed(1),
    )
    assert prior.validate()["legal"]


def test_symmetry_preserves_capacity():
    st, _, _ = _toy_state()
    aug = sample_symmetry_augment(atomic_numbers=st.atomic_numbers, K=st.K, generator=torch.Generator().manual_seed(2))
    st2 = apply_symmetry_to_state(st, aug)
    assert st2.validate()["legal"]


def test_vectorized_assign_mp_matches_loop():
    """Engineering: AssignmentGraphMP vector path == O(N²) reference loop."""
    from mattergen.assignment.joint_assignment_diffusion.conditioning import (
        AssignmentGraphMP,
        OrbitRelationTable,
    )

    st, partition, _ = _toy_state()
    H = 16
    mp = AssignmentGraphMP(hidden=H, edge_dim=32)
    rho = OrbitRelationTable(num_orbits=partition.J, dim=32)
    for p in mp.parameters():
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.uniform_(p, -0.1, 0.1)
    h = torch.randn(st.N, H)
    copy_of, orbit_of = st.copy_of(), st.orbit_of()
    # reference loop (pre-vectorization semantics)
    msgs = torch.zeros_like(h)
    counts = torch.zeros(st.N, 1)
    for i in range(st.N):
        for j in range(st.N):
            if i == j or int(copy_of[i]) != int(copy_of[j]):
                continue
            rel = rho(
                torch.tensor(int(orbit_of[i])),
                torch.tensor(int(orbit_of[j])),
            )
            feat = torch.cat([h[i], h[j], rel], dim=-1)
            msgs[i] = msgs[i] + mp.msg(feat)
            counts[i] += 1
    ref = mp.upd(torch.cat([h, msgs / counts.clamp_min(1.0)], dim=-1))
    got = mp(h, copy_of=copy_of, orbit_of=orbit_of, z_orbit=torch.randn(partition.J, H), rho=rho)
    assert torch.allclose(ref, got, atol=1e-5)


def test_enumerate_moves_vectorized_matches_bruteforce():
    st, _, _ = _toy_state()
    from mattergen.assignment.joint_assignment_diffusion.legal_moves import (
        enumerate_g_moves,
        enumerate_r_moves,
    )

    orbit, copy, z = st.orbit_of(), st.copy_of(), st.atomic_numbers
    n = st.N
    brute_r = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if int(copy[i]) == int(copy[j])
        and int(z[i]) == int(z[j])
        and int(orbit[i]) != int(orbit[j])
    }
    brute_g = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if int(orbit[i]) == int(orbit[j]) and int(copy[i]) != int(copy[j])
    }
    assert {(m.i, m.j) for m in enumerate_r_moves(st)} == brute_r
    assert {(m.i, m.j) for m in enumerate_g_moves(st)} == brute_g


def _init_g_head_nonzero(head):
    import torch.nn as nn

    for p in head.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)
    if hasattr(head, "geom_out"):
        nn.init.xavier_uniform_(head.geom_out[-1].weight)
        nn.init.zeros_(head.geom_out[-1].bias)
    if hasattr(head, "cf_out"):
        nn.init.xavier_uniform_(head.cf_out[-1].weight)
        nn.init.zeros_(head.cf_out[-1].bias)
    if hasattr(head, "slot_out"):
        nn.init.xavier_uniform_(head.slot_out[-1].weight)
        nn.init.zeros_(head.slot_out[-1].bias)
    if hasattr(head, "net"):
        nn.init.xavier_uniform_(head.net[-1].weight)
        nn.init.zeros_(head.net[-1].bias)


def test_pbc_min_image_matches_pair_potential_convention():
    """B2 reuses BondPairPotential wrap: Δu = round(frac_b-frac_a), ||Δu @ cell||."""
    from mattergen.assignment.global_copy_assembly.pair_potential import (
        gaussian_radial_basis,
        pbc_minimum_image_distance,
    )

    frac_a = torch.tensor([[0.1, 0.2, 0.3]])
    frac_b = torch.tensor([[0.9, 0.2, 0.3]])
    cell = torch.eye(3) * 10.0
    dist = pbc_minimum_image_distance(frac_a, frac_b, cell)
    assert torch.allclose(dist, torch.tensor([2.0]), atol=1e-5)
    centres = torch.linspace(0.0, 6.0, 32)
    rbf = gaussian_radial_basis(dist, centres, 6.0)
    assert rbf.shape == (1, 32)
    assert torch.isfinite(rbf).all()
    # batched [N,N] same formula as BondPairPotential.forward
    frac = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    d2 = pbc_minimum_image_distance(frac[:, None, :], frac[None, :, :], cell)
    assert torch.allclose(d2[0, 1], torch.tensor(4.0), atol=1e-5)


def test_candidate_copy_geometry_empty_slot_and_no_self_distance():
    """Singleton remove-one → zero RBF (not d=0 peak); no NaN."""
    from mattergen.assignment.joint_assignment_diffusion.conditioning import CandidateCopyGeometry

    torch.manual_seed(0)
    geom = CandidateCopyGeometry(rbf_dim=32, cutoff=6.0)
    # two copies, two singleton orbits
    frac = torch.tensor(
        [
            [0.10, 0.10, 0.10],
            [0.12, 0.20, 0.10],
            [0.60, 0.10, 0.10],
            [0.62, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )
    copy_of = torch.tensor([0, 0, 1, 1])
    orbit_of = torch.tensor([0, 1, 0, 1])
    cell = torch.eye(3) * 8.0
    rbf = geom.all_pairs_rbf(frac, cell)
    G, cnt = geom.pool_to_slots(rbf, copy_of=copy_of, orbit_of=orbit_of, K=2, J=2)
    assert (cnt == 1).all()
    q = torch.tensor([0])
    g_own, occ_own = geom.exclude_atom(
        G, rbf, cnt, query=q, dest_copy=torch.tensor([0]), exclude=q, copy_of=copy_of, orbit_of=orbit_of
    )
    assert occ_own[0, 0].item() == 0.0
    assert torch.allclose(g_own[0, 0], torch.zeros_like(g_own[0, 0]))
    # leftover self-distance would be RBF(0) ≠ 0
    rbf0 = geom.all_pairs_rbf(frac[0:1], cell)[0, 0]
    assert rbf0.norm() > 0.5
    assert not torch.allclose(g_own[0, 0], rbf0)
    assert torch.isfinite(g_own).all()
    # dest copy excluding partner: atom 0 → copy 1 \ {2}
    g_dst, occ_dst = geom.exclude_atom(
        G,
        rbf,
        cnt,
        query=q,
        dest_copy=torch.tensor([1]),
        exclude=torch.tensor([2]),
        copy_of=copy_of,
        orbit_of=orbit_of,
    )
    assert occ_dst[0, 0].item() == 0.0
    assert torch.allclose(g_dst[0, 0], torch.zeros_like(g_dst[0, 0]))
    assert occ_dst[0, 1].item() == 1.0
    assert g_dst[0, 1].norm() > 0


def test_candidate_copy_geometry_slot_multiplicity_invariance():
    """Permuting atoms inside one orbit slot does not change pooled g."""
    from mattergen.assignment.joint_assignment_diffusion.conditioning import CandidateCopyGeometry

    torch.manual_seed(1)
    geom = CandidateCopyGeometry(rbf_dim=16, cutoff=6.0)
    # copy 0: atoms 0,1 in orbit 0 (multiplicity 2); atom 2 orbit 1
    # copy 1: atoms 3,4 orbit 0; atom 5 orbit 1
    frac = torch.tensor(
        [
            [0.05, 0.05, 0.00],
            [0.08, 0.15, 0.02],
            [0.10, 0.40, 0.00],
            [0.55, 0.05, 0.00],
            [0.70, 0.18, 0.03],
            [0.58, 0.42, 0.00],
        ],
        dtype=torch.float32,
    )
    copy_of = torch.tensor([0, 0, 0, 1, 1, 1])
    orbit_of = torch.tensor([0, 0, 1, 0, 0, 1])
    cell = torch.eye(3) * 10.0
    rbf = geom.all_pairs_rbf(frac, cell)
    G, cnt = geom.pool_to_slots(rbf, copy_of=copy_of, orbit_of=orbit_of, K=2, J=2)
    assert int(cnt[1, 0].item()) == 2
    # query atom 2 (external to dest slot orbit 0 of copy 1)
    g_a, _ = geom.exclude_atom(
        G,
        rbf,
        cnt,
        query=torch.tensor([2]),
        dest_copy=torch.tensor([1]),
        exclude=torch.tensor([5]),
        copy_of=copy_of,
        orbit_of=orbit_of,
    )
    frac_swap = frac.clone()
    frac_swap[3], frac_swap[4] = frac[4].clone(), frac[3].clone()
    rbf_b = geom.all_pairs_rbf(frac_swap, cell)
    G_b, cnt_b = geom.pool_to_slots(rbf_b, copy_of=copy_of, orbit_of=orbit_of, K=2, J=2)
    g_b, _ = geom.exclude_atom(
        G_b,
        rbf_b,
        cnt_b,
        query=torch.tensor([2]),
        dest_copy=torch.tensor([1]),
        exclude=torch.tensor([5]),
        copy_of=copy_of,
        orbit_of=orbit_of,
    )
    assert torch.allclose(g_a, g_b, atol=1e-5)


def test_candidate_copy_geometry_pair_and_copy_symmetry():
    """(i,j) vs (j,i) same logit; copy-column permutation leaves logits invariant."""
    import torch.nn as nn

    from mattergen.assignment.joint_assignment_diffusion.conditioning import OrbitSlotCopyContext
    from mattergen.assignment.joint_assignment_diffusion.jump_heads import GJumpHead

    torch.manual_seed(2)
    n, hid, k, j = 6, 8, 2, 3
    orbit_of = torch.tensor([0, 1, 2, 0, 1, 2])
    copy_of = torch.tensor([0, 0, 0, 1, 1, 1])
    h = torch.randn(n, hid)
    z_orbit = torch.randn(j, hid)
    frac = torch.rand(n, 3)
    cell = torch.eye(3) * 9.0
    ctx = OrbitSlotCopyContext(hid)
    U, feat, cnt = ctx.slot_table(h, orbit_of=orbit_of, copy_of=copy_of, z_orbit=z_orbit, K=k, J=j)
    u_excl = ctx.exclude_atom_slots(U, cnt, feat, orbit_of, copy_of)
    head = GJumpHead(hid, copy_context_mode="orbit_slot_geometry")
    _init_g_head_nonzero(head)

    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([3, 4, 5])
    logits, diag = head.batch_logits_orbit_slot_geometry(
        h,
        ii,
        jj,
        z_orbit=z_orbit,
        orbit_of=orbit_of,
        copy_of=copy_of,
        U=U,
        u_excl=u_excl,
        slot_cnt=cnt,
        frac=frac,
        cell=cell,
        t_scalar=0.45,
        K=k,
        J=j,
    )
    logits_swap, _ = head.batch_logits_orbit_slot_geometry(
        h,
        jj,
        ii,
        z_orbit=z_orbit,
        orbit_of=orbit_of,
        copy_of=copy_of,
        U=U,
        u_excl=u_excl,
        slot_cnt=cnt,
        frac=frac,
        cell=cell,
        t_scalar=0.45,
        K=k,
        J=j,
    )
    assert torch.allclose(logits, logits_swap, atol=1e-5)
    assert torch.isfinite(logits).all()
    assert diag["candidate_copy_geom_norm_mean"] > 0.0
    assert diag["candidate_copy_relation_variance_across_copies"] >= 0.0

    # copy-column permutation: physical pairs unchanged → same logits
    perm = torch.tensor([1, 0])
    copy2 = perm[copy_of]
    U2, feat2, cnt2 = ctx.slot_table(h, orbit_of=orbit_of, copy_of=copy2, z_orbit=z_orbit, K=k, J=j)
    u_ex2 = ctx.exclude_atom_slots(U2, cnt2, feat2, orbit_of, copy2)
    logits_p, _ = head.batch_logits_orbit_slot_geometry(
        h,
        ii,
        jj,
        z_orbit=z_orbit,
        orbit_of=orbit_of,
        copy_of=copy2,
        U=U2,
        u_excl=u_ex2,
        slot_cnt=cnt2,
        frac=frac,
        cell=cell,
        t_scalar=0.45,
        K=k,
        J=j,
    )
    assert torch.allclose(logits, logits_p, atol=1e-5)

    # zero last layer still uniform (start-of-training)
    nn.init.zeros_(head.geom_out[-1].weight)
    nn.init.zeros_(head.geom_out[-1].bias)
    logits0, _ = head.batch_logits_orbit_slot_geometry(
        h,
        ii,
        jj,
        z_orbit=z_orbit,
        orbit_of=orbit_of,
        copy_of=copy_of,
        U=U,
        u_excl=u_excl,
        slot_cnt=cnt,
        frac=frac,
        cell=cell,
        t_scalar=0.45,
        K=k,
        J=j,
    )
    assert torch.allclose(logits0, torch.zeros_like(logits0), atol=1e-6)


def test_g_copy_context_modes_b0_b1_b2_b3_still_registered():
    """Ablation switch keeps mean / orbit_slot / geometry / template_counterfactual."""
    from mattergen.assignment.joint_assignment_diffusion.jump_heads import (
        GJumpHead,
        _G_COPY_CONTEXT_MODES,
    )

    assert _G_COPY_CONTEXT_MODES == (
        "mean",
        "orbit_slot",
        "orbit_slot_geometry",
        "template_counterfactual",
    )
    for mode in _G_COPY_CONTEXT_MODES:
        head = GJumpHead(8, copy_context_mode=mode)
        assert head.copy_context_mode == mode


def _toy_rho_table(j: int = 3):
    from mattergen.assignment.joint_assignment_diffusion.conditioning import OrbitRelationTable

    rho = OrbitRelationTable(num_orbits=j, dim=32)
    st, partition, _ = _toy_state()
    # undirected molecular path on roles 0-1-2-3
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    btype = torch.ones(edges.shape[1], dtype=torch.long)
    rho.set_from_role_graph(partition=partition, role_edge_index=edges, role_bond_type=btype)
    return rho, st


def test_template_rho_orbit_invariant_and_disconnected_bin():
    """ρ is orbit-aggregated; disconnected ≠ ordinary hop count."""
    from mattergen.assignment.joint_assignment_diffusion.conditioning import OrbitRelationTable

    rho = OrbitRelationTable(num_orbits=3, dim=16)
    # two roles in orbit 1 are equivalent: edges 0-1 and 0-2 must pool
    # partition: [[0],[1,2],[3]] from _toy_state — rebuild explicitly
    from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition

    partition = build_orbit_partition([[0], [1, 2], [3]])
    edges_a = torch.tensor([[0, 1], [1, 0]])
    edges_b = torch.tensor([[0, 2], [2, 0]])
    bt = torch.tensor([2, 2])
    rho.set_from_role_graph(partition=partition, role_edge_index=edges_a, role_bond_type=bt)
    exists_a = rho.bond_exists.clone()
    mult_a = rho.edge_mult.clone()
    rho.set_from_role_graph(partition=partition, role_edge_index=edges_b, role_bond_type=bt)
    assert torch.allclose(exists_a, rho.bond_exists)
    assert torch.allclose(mult_a, rho.edge_mult)
    # 0 bonded to orbit-1; orbit-2 disconnected
    assert float(rho.bond_exists[0, 1]) == 1.0
    assert float(rho.bond_exists[0, 2]) == 0.0
    assert int(rho.dist_bin[0, 0]) == 0
    assert int(rho.dist_bin[0, 2]) == OrbitRelationTable.DIST_BIN_DISCONNECTED


def test_template_counterfactual_pair_and_copy_symmetry():
    """g_swap_logit(i,j,k,l) == g_swap_logit(j,i,l,k); copy relabel invariant."""
    import torch.nn as nn

    from mattergen.assignment.joint_assignment_diffusion.conditioning import OrbitSlotCopyContext
    from mattergen.assignment.joint_assignment_diffusion.jump_heads import GJumpHead

    torch.manual_seed(3)
    n, hid, k, j = 6, 8, 2, 3
    orbit_of = torch.tensor([0, 1, 2, 0, 1, 2])
    copy_of = torch.tensor([0, 0, 0, 1, 1, 1])
    h = torch.randn(n, hid)
    z_orbit = torch.randn(j, hid)
    frac = torch.rand(n, 3)
    cell = torch.eye(3) * 9.0
    rho, _ = _toy_rho_table(j)
    ctx = OrbitSlotCopyContext(hid)
    U, feat, cnt = ctx.slot_table(h, orbit_of=orbit_of, copy_of=copy_of, z_orbit=z_orbit, K=k, J=j)
    u_excl = ctx.exclude_atom_slots(U, cnt, feat, orbit_of, copy_of)
    head = GJumpHead(hid, copy_context_mode="template_counterfactual")
    _init_g_head_nonzero(head)
    nn.init.xavier_uniform_(head.cf_out[-1].weight)
    nn.init.zeros_(head.cf_out[-1].bias)

    ii = torch.tensor([0, 1, 2])
    jj = torch.tensor([3, 4, 5])
    kwargs = dict(
        z_orbit=z_orbit,
        orbit_of=orbit_of,
        copy_of=copy_of,
        U=U,
        u_excl=u_excl,
        slot_cnt=cnt,
        frac=frac,
        cell=cell,
        t_scalar=0.45,
        K=k,
        J=j,
        rho_table=rho,
    )
    logits, diag = head.batch_logits_template_counterfactual(h, ii, jj, **kwargs)
    logits_swap, _ = head.batch_logits_template_counterfactual(h, jj, ii, **kwargs)
    assert torch.allclose(logits, logits_swap, atol=1e-5)
    assert torch.isfinite(logits).all()
    assert "delta_S_vec" in diag
    ds_swap = head.batch_logits_template_counterfactual(h, jj, ii, **kwargs)[1]["delta_S_vec"]
    assert torch.allclose(diag["delta_S_vec"], ds_swap, atol=1e-5)

    perm = torch.tensor([1, 0])
    copy2 = perm[copy_of]
    U2, feat2, cnt2 = ctx.slot_table(h, orbit_of=orbit_of, copy_of=copy2, z_orbit=z_orbit, K=k, J=j)
    u_ex2 = ctx.exclude_atom_slots(U2, cnt2, feat2, orbit_of, copy2)
    kwargs2 = dict(kwargs)
    kwargs2.update(copy_of=copy2, U=U2, u_excl=u_ex2, slot_cnt=cnt2)
    logits_p, _ = head.batch_logits_template_counterfactual(h, ii, jj, **kwargs2)
    assert torch.allclose(logits, logits_p, atol=1e-5)

    nn.init.zeros_(head.cf_out[-1].weight)
    nn.init.zeros_(head.cf_out[-1].bias)
    logits0, _ = head.batch_logits_template_counterfactual(h, ii, jj, **kwargs)
    assert torch.allclose(logits0, torch.zeros_like(logits0), atol=1e-6)


def test_template_counterfactual_detach_does_not_flow_to_trunk():
    """L_G through detached h^G must not populate trunk grads."""
    import torch.nn as nn

    from mattergen.assignment.joint_assignment_diffusion.conditioning import OrbitSlotCopyContext
    from mattergen.assignment.joint_assignment_diffusion.jump_heads import GJumpHead

    torch.manual_seed(4)
    n, hid, k, j = 6, 8, 2, 3
    orbit_of = torch.tensor([0, 1, 2, 0, 1, 2])
    copy_of = torch.tensor([0, 0, 0, 1, 1, 1])
    trunk = nn.Linear(3, hid)
    frac = torch.rand(n, 3)
    h_live = trunk(frac)
    h_g = h_live.detach()
    z_orbit = torch.randn(j, hid)
    cell = torch.eye(3) * 9.0
    rho, _ = _toy_rho_table(j)
    ctx = OrbitSlotCopyContext(hid)
    U, feat, cnt = ctx.slot_table(h_g, orbit_of=orbit_of, copy_of=copy_of, z_orbit=z_orbit, K=k, J=j)
    u_excl = ctx.exclude_atom_slots(U, cnt, feat, orbit_of, copy_of)
    head = GJumpHead(hid, copy_context_mode="template_counterfactual")
    _init_g_head_nonzero(head)
    nn.init.xavier_uniform_(head.cf_out[-1].weight)
    logits, _ = head.batch_logits_template_counterfactual(
        h_g,
        torch.tensor([0, 1, 2]),
        torch.tensor([3, 4, 5]),
        z_orbit=z_orbit,
        orbit_of=orbit_of,
        copy_of=copy_of,
        U=U,
        u_excl=u_excl,
        slot_cnt=cnt,
        frac=frac,
        cell=cell,
        t_scalar=0.45,
        K=k,
        J=j,
        rho_table=rho,
    )
    logits.sum().backward()
    assert trunk.weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.cf_out.parameters())


def test_spearman_tied_ranks():
    from mattergen.assignment.joint_assignment_diffusion.losses import spearman_tied

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 2.0, 3.0])
    assert abs(spearman_tied(x, y) - 1.0) < 1e-6
    assert abs(spearman_tied(x, -y) + 1.0) < 1e-6
    xt = torch.tensor([1.0, 1.0, 2.0])
    yt = torch.tensor([3.0, 3.0, 9.0])
    assert spearman_tied(xt, yt) > 0.9


def test_trunk_rg_interference_metrics_and_forgetting():
    from mattergen.assignment.joint_assignment_diffusion.losses import (
        aggregate_r_forgetting,
        flatten_param_grads,
        summarize_interference,
        trunk_rg_interference_metrics,
    )

    p = torch.zeros(3, requires_grad=False)
    g_r = flatten_param_grads([p], [torch.tensor([1.0, 0.0, 0.0])])
    g_g_anti = flatten_param_grads([p], [torch.tensor([-2.0, 0.0, 0.0])])
    m = trunk_rg_interference_metrics(g_r, g_g_anti)
    assert m["cos_RG"] < -0.99
    assert m["D_G_over_R"] > 1.0
    assert m["destructive_dominant"] == 1.0
    g_g_al = flatten_param_grads([p], [torch.tensor([0.5, 0.0, 0.0])])
    m2 = trunk_rg_interference_metrics(g_r, g_g_al)
    assert m2["cos_RG"] > 0.99
    assert m2["destructive_dominant"] == 0.0

    recs = [
        {"kind": "R", "delta_CE": -0.5, "top1": 0.4},
        {"kind": "R", "delta_CE": -1.0, "top1": 0.6},
        {"kind": "R", "delta_CE": -0.2, "top1": 0.3},
        {"kind": "G", "delta_CE": 0.0, "top1": 0.0},
    ]
    fr = aggregate_r_forgetting(recs)
    assert abs(fr["best_delta_CE_R"] + 1.0) < 1e-9
    assert fr["forgetting_delta_CE_R"] > 0.0
    inter = summarize_interference(
        [
            {"step": 10, "cos_RG": -0.5, "D_G_over_R": 2.0, "destructive_dominant": 1.0, "delta_G_LR": 0.1, "norm_g_G": 1.0, "norm_g_R": 0.5},
            {"step": 900, "cos_RG": 0.1, "D_G_over_R": 0.5, "destructive_dominant": 0.0, "delta_G_LR": -0.02, "norm_g_G": 0.2, "norm_g_R": 0.4},
        ]
    )
    assert inter["all"]["n"] == 2
    assert inter["early"]["n"] == 1
    assert inter["late"]["n"] == 1


def test_crystal_geometry_clash_metric():
    from mattergen.assignment.joint_assignment_diffusion.metrics import crystal_geometry_vs_target

    frac = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    cell = torch.eye(3) * 10.0
    ok = crystal_geometry_vs_target(frac, cell, frac, cell)
    assert ok["no_clash"]
    close = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    bad = crystal_geometry_vs_target(close, cell, frac, cell)
    assert not bad["no_clash"]


def test_geometry_assignment_conditioning_flag_on_model_signature():
    import inspect
    from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel

    sig = inspect.signature(JointAXLModel.__init__)
    assert "geometry_assignment_conditioning" in sig.parameters


def test_resolve_geometry_ablation_arm_original_vs_oracle_g():
    from mattergen.assignment.joint_assignment_diffusion.geometry_ablation import (
        resolve_geometry_ablation_arm,
    )

    orig = resolve_geometry_ablation_arm(ablation_arm="original")
    assert orig["geometry_assignment_conditioning"] is False
    assert orig["train_assignment_heads"] is False
    assert orig["oracle_g"] is False
    oracle = resolve_geometry_ablation_arm(ablation_arm="oracle_g")
    assert oracle["geometry_assignment_conditioning"] is True
    assert oracle["train_assignment_heads"] is False
    assert oracle["oracle_g"] is True
    learned = resolve_geometry_ablation_arm(ablation_arm="g_conditioned")
    assert learned["train_assignment_heads"] is True
    # geom on + no assignment heads ⇒ oracle-G, not learned G
    implied = resolve_geometry_ablation_arm(
        geometry_assignment_conditioning=True, train_assignment_heads=False
    )
    assert implied["ablation_arm"] == "oracle_g"


def test_oracle_assignment_is_forward_of_gt_not_clean_g0():
    from mattergen.assignment.joint_assignment_diffusion.ctmc import oracle_assignment_at_t
    from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule

    st0, _, _ = _toy_state()
    g = torch.Generator()
    g.manual_seed(0)
    schedule = AsyncJumpSchedule()
    st_t, traj = oracle_assignment_at_t(st0, schedule=schedule, t=0.6, generator=g)
    assert traj.times[0] == 0.0
    # Oracle at t is the CTMC state, not a jump-head prediction.
    assert st_t.validate()["legal"]
    g0 = torch.Generator()
    g0.manual_seed(1)
    st_early, _ = oracle_assignment_at_t(st0, schedule=schedule, t=0.0, generator=g0)
    # t=0 must be the GT assignment (no forward mass yet).
    assert torch.equal(st_early.A, st0.A)
