"""N2.1 causal edge tests (authored; not run in this commit turn)."""
from __future__ import annotations

import inspect

import torch

from mattergen.assignment.soft_c_geometry_feedback_n2.gates import NoiseGateConfig
from mattergen.assignment.soft_c_geometry_feedback_n2_1.adapters import CausalEdgeModulator
from mattergen.assignment.soft_c_geometry_feedback_n2_1.causal_edge import (
    apply_causal_edge_residual,
    causal_edge_coefficients,
)


def test_f_psi_takes_edge_only():
    m = CausalEdgeModulator(8, bottleneck=4)
    e = torch.randn(5, 8)
    out = m(e)
    assert out.shape == e.shape
    sig = inspect.signature(m.forward)
    assert list(sig.parameters.keys()) == ["edge_emb"]


def test_zero_init_direction_zero():
    m = CausalEdgeModulator(8, bottleneck=4)
    e = torch.randn(5, 8)
    assert torch.allclose(m(e), torch.zeros_like(e))


def test_coeff_zero_when_uncertain_or_high_noise():
    C = torch.full((4, 4), 0.5)
    C.fill_diagonal_(1.0)
    ei = torch.tensor([[0, 1], [1, 2]])
    coeff, _ = causal_edge_coefficients(
        soft_c=C,
        edge_index=ei,
        cell_offsets=None,
        t_fraction=0.1,
        noise_gate_cfg=NoiseGateConfig(),
    )
    # off-diagonal p=0.5 ⇒ q=0 ⇒ coeff=0
    assert torch.allclose(coeff, torch.zeros_like(coeff))

    C2 = torch.ones(4, 4)
    coeff2, audit = causal_edge_coefficients(
        soft_c=C2,
        edge_index=ei,
        cell_offsets=None,
        t_fraction=0.9,
        noise_gate_cfg=NoiseGateConfig(),
    )
    assert audit.g_noise == 0.0
    assert torch.allclose(coeff2, torch.zeros_like(coeff2))


def test_delta_structurally_zero_when_g_or_q_zero():
    m = CausalEdgeModulator(8, bottleneck=4)
    # Break zero-init so F is nonzero
    with torch.no_grad():
        m.net[-1].weight.fill_(0.1)
        m.net[-1].bias.zero_()
    e = torch.randn(3, 8)
    C = torch.full((3, 3), 0.5)
    C.fill_diagonal_(1.0)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    delta, _ = apply_causal_edge_residual(
        e,
        soft_c=C,
        edge_index=ei,
        cell_offsets=None,
        t_fraction=0.1,
        noise_gate_cfg=NoiseGateConfig(),
        f_psi=m,
    )
    assert torch.allclose(delta, torch.zeros_like(delta))

    C2 = torch.eye(3)
    C2[0, 1] = C2[1, 0] = 1.0
    delta2, audit = apply_causal_edge_residual(
        e,
        soft_c=C2,
        edge_index=ei,
        cell_offsets=None,
        t_fraction=0.9,
        noise_gate_cfg=NoiseGateConfig(),
        f_psi=m,
    )
    assert audit.g_noise == 0.0
    assert torch.allclose(delta2, torch.zeros_like(delta2))


def test_self_image_forced_zero():
    m = CausalEdgeModulator(4, bottleneck=4)
    with torch.no_grad():
        m.net[-1].weight.fill_(0.2)
    e = torch.randn(2, 4)
    C = torch.ones(2, 2)
    ei = torch.tensor([[0, 0], [0, 1]])  # self + normal
    off = torch.tensor([[1, 0, 0], [0, 0, 0]])
    coeff, audit = causal_edge_coefficients(
        soft_c=C,
        edge_index=ei,
        cell_offsets=off,
        t_fraction=0.1,
        noise_gate_cfg=NoiseGateConfig(),
    )
    assert audit.num_zeroed_self_image == 1
    assert float(coeff[0]) == 0.0


def test_no_c_concat_in_modulator_source():
    src = inspect.getsource(CausalEdgeModulator.forward)
    assert "c_intra" not in src
    assert "soft_c" not in src
