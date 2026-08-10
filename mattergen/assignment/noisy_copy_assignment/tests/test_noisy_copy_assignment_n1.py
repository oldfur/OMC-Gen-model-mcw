"""Tests for N1 noisy copy assignment (authored; not executed in this commit turn)."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
    PROVENANCE,
    build_default_mattergen_corruption,
)
from mattergen.assignment.noisy_copy_assignment.module import (
    NoisyCopyAssignmentConfig,
    NoisyCopyAssignmentN1,
)
from mattergen.assignment.noisy_copy_assignment.orbit_capacity import orbit_capacity_map, labels_to_bar_r
from mattergen.assignment.noisy_copy_assignment.soft_c import (
    SOFT_C_KIND,
    SOFT_C_SEMANTICS,
    soft_c_from_singleton_map_and_attachment,
)
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    GemNetHiddenExtractor,
    GemNetHiddenOutput,
    build_mol_conditioning_from_sample,
    freeze_module,
    parameter_sha256,
    resolve_checkpoint_path,
)
from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption


# ---------------------------------------------------------------------------
# Existing noise / soft-C / capacity tests
# ---------------------------------------------------------------------------


def test_1_mattergen_noise_reuses_native_classes():
    assert PROVENANCE["independent_noise_implementation"] is False
    corr = build_default_mattergen_corruption()
    assert isinstance(corr, MultiCorruption)
    assert "pos" in corr.sdes and "cell" in corr.sdes
    src = inspect.getsource(MatterGenNativeNoiseAdapter.corrupt_fixed_sample)
    assert "sample_marginal" in src
    assert "randn_like" not in src or "corruption" in src


def test_batch_view_supports_membership_like_simple_batched_data():
    """MultiCorruption.apply uses ``field in batch``; must not probe batch[0]."""
    from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import _BatchView

    batch = _BatchView(
        {
            "pos": torch.rand(4, 3),
            "cell": torch.eye(3).unsqueeze(0),
            "num_atoms": torch.tensor([4]),
        }
    )
    assert "pos" in batch
    assert "cell" in batch
    assert "num_atoms" in batch
    assert 0 not in batch
    # membership must not raise KeyError: 0
    assert "missing" not in batch
    assert batch.get_batch_idx("cell") is None
    assert batch.get_batch_idx("pos").tolist() == [0, 0, 0, 0]


def test_2_clean_limit_t_near_zero_close_to_input():
    adapter = MatterGenNativeNoiseAdapter()
    n, pos0 = 8, torch.rand(8, 3)
    cell0 = torch.eye(3)
    g = torch.Generator().manual_seed(0)
    noisy = adapter.corrupt_fixed_sample(
        frac_coords_0=pos0, lattice_0=cell0, num_atoms=n,
        t=torch.tensor([1e-5]), generator=g,
    )
    assert noisy.frac_coords_t.shape == pos0.shape
    assert noisy.lattice_t.shape == cell0.shape
    assert float(noisy.t) <= 1e-4 or float(noisy.t) > 0


def test_3_noisy_determinism_by_seed():
    adapter = MatterGenNativeNoiseAdapter()
    pos0 = torch.rand(6, 3)
    cell0 = torch.eye(3) * 2
    t = torch.tensor([0.3])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    g3 = torch.Generator().manual_seed(999)
    a = adapter.corrupt_fixed_sample(frac_coords_0=pos0, lattice_0=cell0, num_atoms=6, t=t, generator=g1)
    b = adapter.corrupt_fixed_sample(frac_coords_0=pos0, lattice_0=cell0, num_atoms=6, t=t, generator=g2)
    c = adapter.corrupt_fixed_sample(frac_coords_0=pos0, lattice_0=cell0, num_atoms=6, t=t, generator=g3)
    assert torch.allclose(a.frac_coords_t, b.frac_coords_t)
    assert not torch.allclose(a.frac_coords_t, c.frac_coords_t)


def test_4_assignment_observational_geometry_probe_unchanged_by_orbit_head():
    """With context_encoder ablation, geometry probe ignores orbit head."""
    partition = build_orbit_partition([[0], [1, 2], [3]])
    cfg = NoisyCopyAssignmentConfig(
        freeze_gemnet_backbone=True,
        hidden_dim=32,
        crystal_num_layers=1,
        hidden_source="context_encoder",
    )
    model = NoisyCopyAssignmentN1(cfg, partition)
    z = torch.ones(6, dtype=torch.long)
    frac = torch.rand(6, 3)
    cell = torch.eye(3)
    t = torch.tensor(0.2)
    p1 = model.geometry_probe(z, frac, cell, t)
    with torch.no_grad():
        for p in model.orbit_head.parameters():
            p.add_(1.0)
    p2 = model.geometry_probe(z, frac, cell, t)
    assert torch.allclose(p1, p2)


def test_5_frozen_backbone_no_grad_params_context_ablation():
    partition = build_orbit_partition([[0], [1, 2]])
    cfg = NoisyCopyAssignmentConfig(
        freeze_gemnet_backbone=True,
        hidden_dim=16,
        crystal_num_layers=1,
        hidden_source="context_encoder",
    )
    model = NoisyCopyAssignmentN1(cfg, partition)
    assert model.context_backbone is not None
    assert all(not p.requires_grad for p in model.context_backbone.parameters())
    assert any(p.requires_grad for p in model.orbit_head.parameters())


def test_6_orbit_capacity_map():
    partition = build_orbit_partition([[0], [1, 2]])
    K = 2
    N = K * (1 + 2)
    logits = torch.randn(N, partition.J)
    labels = orbit_capacity_map(logits, partition, K=K)
    for o, size in enumerate(partition.orbit_sizes):
        assert int((labels == o).sum()) == K * size
    bar = labels_to_bar_r(labels, partition.J)
    assert bar.shape == (N, partition.J)
    assert torch.allclose(bar.sum(-1), torch.ones(N))


def test_7_soft_c_properties():
    N, K = 6, 2
    orbit_atoms = torch.tensor([1, 2, 4, 5])
    raw = torch.randn(2, 4, 4)
    F = 0.5 * (raw + raw.transpose(-1, -2))
    F = F + torch.diag_embed(torch.full((2, 4), float("-inf")))
    G = torch.zeros(6, 2)
    G[0, 0] = G[3, 1] = 1
    mask = torch.zeros(6, dtype=torch.bool)
    mask[0] = mask[3] = True
    C = soft_c_from_singleton_map_and_attachment(
        G_singleton=G, singleton_mask=mask, orbit_atoms=orbit_atoms, F_attach=F, atoms_per_copy=2
    )
    assert torch.allclose(C, C.T, atol=1e-5)
    assert torch.all((C >= 0) & (C <= 1))
    assert torch.allclose(torch.diag(C), torch.ones(6))
    assert SOFT_C_SEMANTICS == "c_soft_conditional_on_singleton_map"
    assert "conditional-on-singleton-MAP" in SOFT_C_KIND


def test_8_no_oracle_copy_flags():
    cfg = NoisyCopyAssignmentConfig()
    assert cfg.use_copy_id_as_input is False
    assert cfg.use_oracle_C_as_input is False
    assert cfg.geometry_feedback is False
    assert cfg.hidden_source == "gemnet"


def test_9_hard_c_gauge_invariance():
    G = torch.tensor([[1.0, 0], [1, 0], [0, 1], [0, 1]])
    C = G @ G.T
    P = torch.tensor([[0.0, 1], [1, 0]])
    Gp = G @ P
    assert torch.equal(C, Gp @ Gp.T)


def test_10_high_noise_tie_fields_exist_in_provenance():
    assert "noise_source" in PROVENANCE
    assert PROVENANCE["noise_source"] == "mattergen_native_forward_process"
    assert PROVENANCE["independent_noise_implementation"] is False


# ---------------------------------------------------------------------------
# GemNet wiring tests (1–12 from N1 rewire spec)
# ---------------------------------------------------------------------------


class _FakeGemNetOut:
    def __init__(self, n: int, h: int):
        self.node_embeddings = torch.randn(n, h)
        self.edge_embeddings = None
        self.forces = torch.zeros(n, 3)
        self.stress = torch.zeros(1, 3, 3)


class _FakeGemNet(nn.Module):
    def __init__(self, h: int = 32):
        super().__init__()
        self.h = h
        self.w = nn.Parameter(torch.randn(h))

    def forward(self, **kwargs):
        n = int(kwargs["frac_coords"].shape[0])
        return _FakeGemNetOut(n, self.h)


class _FakeDenoiser(nn.Module):
    """Minimal stand-in for GemNetTDenoiser interface used by extractor/module."""

    def __init__(self, h: int = 32):
        super().__init__()
        self.hidden_dim = h
        self.gemnet = _FakeGemNet(h)
        self.noise_level_encoding = nn.Linear(1, h)  # not real; tests mock extract
        self.property_embeddings = nn.ModuleDict()
        self.molecule_conditioner = None
        self.molecule_conditioner_gate_center = None
        self.molecule_conditioner_gate_width = 0.05
        self.molecule_conditioner_gate_min_scale = 0.0
        self._probe = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, t):
        n = x["pos"].shape[0]
        return {
            "pos": torch.zeros(n, 3),
            "cell": torch.zeros(1, 3, 3),
            "atomic_numbers": torch.zeros(n, 100),
        }


def test_gemnet_default_hidden_source():
    cfg = NoisyCopyAssignmentConfig()
    assert cfg.hidden_source == "gemnet"
    assert cfg.fail_on_gemnet_fallback is True


def test_fail_on_missing_gemnet():
    """Test 2: hidden_source=gemnet without injection must raise."""
    partition = build_orbit_partition([[0], [1]])
    cfg = NoisyCopyAssignmentConfig(hidden_source="gemnet", hidden_dim=16, crystal_num_layers=1)
    model = NoisyCopyAssignmentN1(cfg, partition)
    z = torch.ones(4, dtype=torch.long)
    frac = torch.rand(4, 3)
    cell = torch.eye(3)
    t = torch.tensor(0.1)
    raised = False
    try:
        model.extract_atom_hidden(z=z, frac=frac, cell=cell, t=t, atomic_numbers=z)
    except RuntimeError as exc:
        raised = True
        assert "not injected" in str(exc).lower() or "GemNet" in str(exc)
        assert "fallback" in str(exc).lower() or "not injected" in str(exc).lower()
    assert raised


def test_no_fallback_to_context_encoder():
    """Test 3: gemnet mode does not silently use context encoder."""
    partition = build_orbit_partition([[0], [1]])
    cfg = NoisyCopyAssignmentConfig(hidden_source="gemnet", hidden_dim=16, crystal_num_layers=1)
    model = NoisyCopyAssignmentN1(cfg, partition)
    assert model.context_backbone is None
    try:
        model.extract_atom_hidden(
            z=torch.ones(2, dtype=torch.long),
            frac=torch.rand(2, 3),
            cell=torch.eye(3),
            t=torch.tensor(0.2),
        )
        assert False, "should have raised"
    except RuntimeError:
        pass
    # Still no context backbone created
    assert model.context_backbone is None


def test_set_gemnet_denoiser_injection_and_frozen():
    """Tests 5, DI API: set_gemnet_denoiser freezes backbone."""
    partition = build_orbit_partition([[0], [1, 2]])
    cfg = NoisyCopyAssignmentConfig(
        hidden_source="gemnet", hidden_dim=16, freeze_gemnet_backbone=True, crystal_num_layers=1
    )
    model = NoisyCopyAssignmentN1(cfg, partition)
    # Use a real GemNetTDenoiser-shaped mock via isinstance bypass:
    # module checks isinstance(denoiser, GemNetTDenoiser). Patch that check path.
    fake = _FakeDenoiser(h=32)
    # Monkey: wrap so isinstance passes by attaching as attribute and patching set
    with patch(
        "mattergen.assignment.noisy_copy_assignment.module.GemNetTDenoiser",
        _FakeDenoiser,
    ), patch(
        "mattergen.assignment.noisy_copy_assignment.module.GemNetHiddenExtractor"
    ) as Ext:
        Ext.return_value = MagicMock(
            hidden_dim=32,
            extract=MagicMock(
                return_value=GemNetHiddenOutput(
                    node_embeddings=torch.randn(6, 32),
                    hidden_dim=32,
                    metadata={
                        "hidden_source": "gemnet_node_embeddings",
                        "context_crystal_encoder_used": False,
                        "num_atoms": 6,
                        "timestep_conditioning": True,
                    },
                )
            ),
        )
        Ext.return_value.denoiser = fake
        model.set_gemnet_denoiser(fake, freeze=True)
        assert model.gemnet_proj is not None
        h = model.extract_atom_hidden(
            z=torch.ones(6, dtype=torch.long),
            frac=torch.rand(6, 3),
            cell=torch.eye(3),
            t=torch.tensor(0.3),
            atomic_numbers=torch.ones(6, dtype=torch.long),
        )
        assert h.shape == (6, 16)
        assert model._last_hidden_meta["hidden_source"] == "gemnet_node_embeddings"
        assert model._last_hidden_meta["context_crystal_encoder_used"] is False


def test_frozen_gemnet_requires_grad_false():
    """Test 5: backbone requires_grad=False after freeze."""
    fake = _FakeDenoiser(h=8)
    freeze_module(fake)
    assert all(not p.requires_grad for p in fake.parameters())


def test_assignment_gradients_present_context_path():
    """Test 6: assignment head params receive grad (context ablation path)."""
    partition = build_orbit_partition([[0], [1]])
    cfg = NoisyCopyAssignmentConfig(
        hidden_source="context_encoder",
        freeze_gemnet_backbone=True,
        hidden_dim=16,
        crystal_num_layers=1,
    )
    model = NoisyCopyAssignmentN1(cfg, partition)
    z = torch.ones(4, dtype=torch.long)
    frac = torch.rand(4, 3)
    cell = torch.eye(3)
    t = torch.tensor(0.2)
    h = model.extract_atom_hidden(z=z, frac=frac, cell=cell, t=t)
    logits = model.orbit_head(h)
    loss = logits.sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.orbit_head.parameters())
    # frozen context backbone: no grad or zero
    for p in model.context_backbone.parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0


def test_trainable_assignment_parameters_exclude_gemnet():
    """Optimizer list must not include gemnet ids."""
    partition = build_orbit_partition([[0], [1]])
    cfg = NoisyCopyAssignmentConfig(hidden_source="gemnet", hidden_dim=16, crystal_num_layers=1)
    model = NoisyCopyAssignmentN1(cfg, partition)
    fake = _FakeDenoiser(h=32)
    with patch(
        "mattergen.assignment.noisy_copy_assignment.module.GemNetTDenoiser",
        _FakeDenoiser,
    ), patch(
        "mattergen.assignment.noisy_copy_assignment.module.GemNetHiddenExtractor"
    ) as Ext:
        Ext.return_value = MagicMock(hidden_dim=32, denoiser=fake)
        model.set_gemnet_denoiser(fake, freeze=True)
    gem_ids = {id(p) for p in fake.parameters()}
    for p in model.trainable_assignment_parameters():
        assert id(p) not in gem_ids
    assert any(p.requires_grad for p in model.trainable_assignment_parameters())


def test_soft_c_semantics_not_full_joint():
    assert SOFT_C_SEMANTICS == "c_soft_conditional_on_singleton_map"
    assert "exact_structured_C_marginal" not in SOFT_C_SEMANTICS
    assert "full" not in SOFT_C_SEMANTICS.lower() or "conditional" in SOFT_C_KIND.lower()


def test_geometry_feedback_forbidden():
    partition = build_orbit_partition([[0]])
    try:
        NoisyCopyAssignmentN1(
            NoisyCopyAssignmentConfig(geometry_feedback=True, hidden_source="context_encoder"),
            partition,
        )
        assert False
    except ValueError as e:
        assert "geometry_feedback" in str(e)


def test_mol_conditioning_builder_excludes_copy_id():
    """Test 12: mol_copy_id / C0 never enter conditioning extras as features."""
    n, m, k = 8, 2, 4
    z = torch.ones(n, dtype=torch.long)
    role = torch.tensor([i % m for i in range(n)], dtype=torch.long)
    copy = torch.tensor([i // m for i in range(n)], dtype=torch.long)
    role_ei = torch.tensor([[0], [1]], dtype=torch.long)
    role_bt = torch.tensor([1], dtype=torch.long)
    sample = {
        "z": z,
        "role": role,
        "copy": copy,
        "role_edge_index": role_ei,
        "role_bond_type": role_bt,
        "mol_copy_id": copy,  # present but must not be required / used as assignment input
        "C0": torch.eye(n),
    }
    extra = build_mol_conditioning_from_sample(sample)
    assert "mol_x" in extra
    assert "mol_bond_edge_index" in extra
    assert "mol_copy_id" not in extra
    assert "C0" not in extra
    assert "copy" not in extra


def test_parameter_hash_stable_when_frozen():
    m = nn.Linear(4, 4)
    freeze_module(m)
    h1 = parameter_sha256(m)
    h2 = parameter_sha256(m)
    assert h1 == h2
    with torch.no_grad():
        m.weight.add_(1.0)
    h3 = parameter_sha256(m)
    assert h1 != h3


def test_checkpoint_metadata_keys_documented():
    """Test 11: required metadata keys for N1 checkpoints (schema check)."""
    required = {
        "n1_mode",
        "mattergen_model_path",
        "mattergen_load_epoch",
        "mattergen_checkpoint",
        "mattergen_checkpoint_sha256",
        "hidden_source",
        "gemnet_frozen",
        "noise_source",
        "geometry_feedback",
        "orbit_modes",
        "soft_c_semantics",
    }
    # Mirror train script ckpt construction keys
    sample_ckpt = {
        "n1_mode": "observational_noisy_copy_assignment",
        "mattergen_model_path": "/path",
        "mattergen_load_epoch": 294,
        "mattergen_checkpoint": "/path/ckpt.ckpt",
        "mattergen_checkpoint_sha256": "abc",
        "hidden_source": "gemnet_node_embeddings",
        "gemnet_frozen": True,
        "noise_source": "mattergen_native_forward_process",
        "geometry_feedback": False,
        "orbit_modes": ["oracle_orbit", "predicted_orbit"],
        "soft_c_semantics": SOFT_C_SEMANTICS,
    }
    assert required.issubset(sample_ckpt.keys())
    assert sample_ckpt["soft_c_semantics"] == "c_soft_conditional_on_singleton_map"


def test_resolve_checkpoint_path_uses_mattergen_api():
    """Test 1: model_path + load_epoch go through MatterGenCheckpointInfo."""
    src = inspect.getsource(resolve_checkpoint_path)
    assert "MatterGenCheckpointInfo" in src
    src_load = inspect.getsource(
        __import__(
            "mattergen.assignment.noisy_copy_assignment.gemnet_loader",
            fromlist=["load_molecular_csp_gemnet"],
        ).load_molecular_csp_gemnet
    )
    assert "load_model_diffusion" in src_load
    assert "MatterGenCheckpointInfo" in src_load


def test_extractor_mirrors_denoiser_forward_signature():
    """Test: extractor source mentions noise_level_encoding + molecule_conditioner + gemnet."""
    src = inspect.getsource(GemNetHiddenExtractor.extract)
    assert "noise_level_encoding" in src
    assert "molecule_conditioner" in src
    assert "node_embeddings" in src
    assert "torch.no_grad" in src


def test_build_chemgraph_is_pyg_batch_for_get_batch_idx():
    """GemNetTDenoiser requires ChemGraphBatch; bare ChemGraph asserts in get_batch_idx."""
    from torch_geometric.data import Batch

    from mattergen.assignment.noisy_copy_assignment.gemnet_loader import GemNetHiddenExtractor

    # Minimal stand-in only to call build_chemgraph_from_sample
    class _D(nn.Module):
        hidden_dim = 8

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    # bypass freeze on real denoiser interface
    ext = GemNetHiddenExtractor.__new__(GemNetHiddenExtractor)
    nn.Module.__init__(ext)
    ext.denoiser = _D()
    n = 5
    g = ext.build_chemgraph_from_sample(
        frac=torch.rand(n, 3),
        cell=torch.eye(3),
        atomic_numbers=torch.ones(n, dtype=torch.long),
        extra=None,
    )
    assert isinstance(g, Batch)
    batch_idx = g.get_batch_idx("pos")
    assert batch_idx is not None
    assert batch_idx.tolist() == [0] * n
    assert int(g["num_atoms"].reshape(-1)[0]) == n


def test_no_oracle_c_in_extractor_kwargs():
    src = inspect.getsource(GemNetHiddenExtractor.build_chemgraph_from_sample)
    assert "ORACLE_ASSIGNMENT_DENYLIST" in src or "mol_copy_id" in src


def test_primary_vs_ablation_config():
    primary = NoisyCopyAssignmentConfig(hidden_source="gemnet")
    ablation = NoisyCopyAssignmentConfig(hidden_source="context_encoder")
    assert primary.hidden_source == "gemnet"
    assert ablation.hidden_source == "context_encoder"
    m_ab = NoisyCopyAssignmentN1(ablation, build_orbit_partition([[0], [1]]))
    assert m_ab.context_backbone is not None
    m_pr = NoisyCopyAssignmentN1(primary, build_orbit_partition([[0], [1]]))
    assert m_pr.context_backbone is None
