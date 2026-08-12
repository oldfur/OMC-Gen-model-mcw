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


def test_async_schedule_lock():
    sch = AsyncJumpSchedule(r_lock=0.72, g_lock=0.52)
    assert float(sch.beta_r(0.5)) == 0.0
    assert float(sch.beta_g(0.4)) == 0.0
    assert float(sch.beta_r(0.9)) > 0.0
    assert float(sch.beta_g(0.7)) > 0.0


def test_forward_ctmc_stays_legal_and_respects_lock():
    st, _, _ = _toy_state()
    sch = AsyncJumpSchedule(r_lock=0.72, g_lock=0.52, kappa_r=2.0, kappa_g=3.0)
    g = torch.Generator().manual_seed(0)
    traj = simulate_forward_ctmc(st, schedule=sch, generator=g)
    for s in traj.states:
        assert s.validate()["legal"]
    for e in traj.events:
        if e.kind == "R":
            assert e.time > sch.r_lock - 1e-9
        if e.kind == "G":
            assert e.time > sch.g_lock - 1e-9


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
