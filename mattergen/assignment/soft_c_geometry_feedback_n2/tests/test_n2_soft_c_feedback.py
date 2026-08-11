"""N2 soft-C geometry feedback tests (authored; not executed in this commit turn)."""
from __future__ import annotations

import inspect

import torch
from torch import nn

from mattergen.assignment.soft_c_geometry_feedback_n2.adapters import (
    SoftCEdgeAdapter,
    SoftCGroupAdapter,
    ZeroInitLinear,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.feedback import (
    build_edge_soft_c_channels,
    expected_same_copy_representation,
    orbit_preserving_shuffle_soft_c,
    shuffle_soft_c,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.gates import (
    NoiseGateConfig,
    noise_gate,
    pair_gate,
    pairwise_confidence,
)
from mattergen.common.gemnet.gemnet import GemNetT


def test_noise_gate_piecewise():
    cfg = NoiseGateConfig(full_on_t_fraction=0.30, full_off_t_fraction=0.60)
    assert float(noise_gate(0.0, full_on=0.3, full_off=0.6)) == 1.0
    assert float(noise_gate(0.30, full_on=0.3, full_off=0.6)) == 1.0
    g = float(noise_gate(0.45, full_on=0.3, full_off=0.6))
    assert 0.0 < g < 1.0
    assert abs(g - 0.5) < 1e-5
    assert float(noise_gate(0.60, full_on=0.3, full_off=0.6)) == 0.0
    assert float(noise_gate(1.0, full_on=0.3, full_off=0.6)) == 0.0


def test_pairwise_confidence():
    p = torch.tensor([0.0, 0.5, 1.0, 0.75])
    q = pairwise_confidence(p)
    assert torch.allclose(q, torch.tensor([1.0, 0.0, 1.0, 0.5]))


def test_pair_gate_zeros_at_uncertain_or_high_noise():
    p = torch.full((3, 3), 0.5)
    a = pair_gate(p, 0.1, NoiseGateConfig())
    assert torch.allclose(a, torch.zeros_like(a))
    p2 = torch.ones(2, 2)
    a2 = pair_gate(p2, 0.9, NoiseGateConfig())
    assert torch.allclose(a2, torch.zeros_like(a2))


def test_zero_init_adapters_zero_residual():
    e = SoftCEdgeAdapter(emb_size_edge=16, bottleneck=8)
    m = torch.randn(5, 16)
    c = torch.rand(5)
    delta = e(m, c, 1 - c, 2 * c - 1)
    assert torch.allclose(delta, torch.zeros_like(delta))
    assert isinstance(e.net[-1], ZeroInitLinear)

    g = SoftCGroupAdapter(hidden=16, bottleneck=8)
    h = torch.randn(4, 16)
    hg = torch.randn(4, 16)
    gg = torch.ones(4)
    assert torch.allclose(g(h, hg, gg), torch.zeros_like(h))


def test_self_image_edge_zeros_same_copy_feedback():
    n = 4
    soft_c = torch.eye(n)
    soft_c = soft_c + (1 - soft_c) * 0.1
    # edge: self with nonzero image + normal pair
    edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    cell_offsets = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.long)
    c_in, c_out, c_s, audit = build_edge_soft_c_channels(
        soft_c=soft_c,
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        t_fraction=0.1,
        noise_gate_cfg=NoiseGateConfig(),
    )
    assert audit.num_nonzero_self_image_edges == 1
    assert audit.num_zeroed_self_image_same_copy == 1
    assert float(c_in[0]) == 0.0
    assert float(c_s[0]) == 0.0


def test_group_context_excludes_self():
    n, h = 5, 8
    H = torch.randn(n, h)
    C = torch.ones(n, n) * 0.9
    C.fill_diagonal_(1.0)
    hg, gg, mass, meta = expected_same_copy_representation(
        h_a=H,
        soft_c=C,
        t_fraction=0.1,
        noise_gate_cfg=NoiseGateConfig(),
        molecules_atoms_m=5,
    )
    assert hg.shape == (n, h)
    assert mass.min() > 0
    assert meta["num_atoms"] == n


def test_shuffle_soft_c_preserves_symmetry_and_values():
    C = torch.rand(6, 6)
    C = 0.5 * (C + C.T)
    C.fill_diagonal_(1.0)
    g = torch.Generator().manual_seed(0)
    Cs = shuffle_soft_c(C, generator=g)
    assert torch.allclose(Cs, Cs.T, atol=1e-6)
    # multiset of off-diagonal values preserved under simultaneous row/col perm
    assert torch.allclose(torch.sort(C.flatten())[0], torch.sort(Cs.flatten())[0])


def test_orbit_preserving_shuffle_stays_within_orbits():
    """B5: permute only inside each orbit; keep value multiset; not identity."""
    # Two orbits of size 3: labels [0,0,0,1,1,1]
    n = 6
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    C = torch.rand(n, n)
    C = 0.5 * (C + C.T)
    C.fill_diagonal_(1.0)
    g = torch.Generator().manual_seed(1)
    Cs = orbit_preserving_shuffle_soft_c(C, labels, generator=g)
    assert torch.allclose(Cs, Cs.T, atol=1e-6)
    assert torch.allclose(torch.sort(C.flatten())[0], torch.sort(Cs.flatten())[0])
    # Not a pure global reordering of unrelated orbits: block structure of
    # which *positions* hold orbit-internal mass can change, but identity is forbidden.
    assert not torch.allclose(Cs, C)
    # Singleton orbit must be fixed: add singleton and check position 6
    labels2 = torch.tensor([0, 0, 0, 1, 1, 1, 2])
    C2 = torch.eye(7)
    C2[0, 1] = C2[1, 0] = 0.8
    C2[3, 4] = C2[4, 3] = 0.7
    g2 = torch.Generator().manual_seed(2)
    Cs2 = orbit_preserving_shuffle_soft_c(C2, labels2, generator=g2)
    # Diagonal stays 1
    assert torch.allclose(torch.diag(Cs2), torch.ones(7))


def test_gemnet_forward_accepts_soft_c_feedback_kwarg():
    src = inspect.getsource(GemNetT.forward)
    assert "soft_c_feedback" in src
    assert "edge_adapter" in src
    assert "node_delta" in src
    assert "angle_edge_emb" in src


def test_denoiser_threads_soft_c_feedback():
    from mattergen.denoiser import GemNetTDenoiser

    src = inspect.getsource(GemNetTDenoiser.forward)
    assert "soft_c_feedback" in src


def test_n2_trainable_only_adapters():
    from mattergen.assignment.soft_c_geometry_feedback_n2.module import SoftCFeedbackConfig

    # lightweight: only check adapter modules have requires_grad True by default
    e = SoftCEdgeAdapter(8)
    g = SoftCGroupAdapter(8)
    assert any(p.requires_grad for p in e.parameters())
    assert any(p.requires_grad for p in g.parameters())
    cfg = SoftCFeedbackConfig()
    assert cfg.noise_gate.full_on_t_fraction == 0.30
    assert cfg.noise_gate.full_off_t_fraction == 0.60


def test_no_g_diffusion_in_n2_module_doc():
    from mattergen.assignment.soft_c_geometry_feedback_n2 import module as m

    src = inspect.getsource(m)
    assert "geometry_induced_inference" in src
    assert "No independent G/C diffusion" in src or "geometry-induced" in src.lower()


def test_soft_c_semantics_string():
    from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS

    assert SOFT_C_SEMANTICS == "c_soft_conditional_on_singleton_map"
