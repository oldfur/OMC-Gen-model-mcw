"""Load frozen molecular-CSP GemNet via official MatterGen checkpoint APIs.

Uses:
  MatterGenCheckpointInfo(model_path, load_epoch)
  load_model_diffusion(...)

Primary N1 source: le50 molCSP epoch294 (not mattergen_base / later FT models).

Paths are never hard-coded here; callers pass model_path / load_epoch from config/CLI.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.denoiser import GemNetTDenoiser
from mattergen.property_embeddings import get_property_embeddings

# Fields that may condition the pretrained molecular-CSP denoiser.
# mol_copy_id is *not* required by MolecularGraphConditioner and must never be
# treated as an assignment-head feature; we deliberately omit it from extras.
MOL_CONDITIONING_KEYS = (
    "mol_x",
    "mol_bond_edge_index",
    "mol_bond_attr",
    "mol_bond_d0",
    "mol_atom_id",
    "mol_num_molecules",
    "chemical_system",
)

# Explicit denylist: never pass these into GemNet ChemGraph for N1.
ORACLE_ASSIGNMENT_DENYLIST = (
    "mol_copy_id",
    "copy",
    "G0",
    "C0",
    "role",  # role is oracle partition; mol_atom_id is the conditioning surrogate when present
)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parameter_sha256(module: nn.Module) -> str:
    """Stable content hash of all parameters (name-ordered)."""
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in sorted(module.named_parameters(), key=lambda x: x[0]):
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_module(module: nn.Module) -> int:
    """Set requires_grad=False on all parameters; return count frozen."""
    n = 0
    for p in module.parameters():
        if p.requires_grad:
            n += 1
        p.requires_grad_(False)
    module.eval()
    return n


def count_params(module: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
    }


@dataclass(frozen=True)
class LoadedGemNetBundle:
    """Loaded DiffusionLightningModule + extracted GemNetTDenoiser score model."""

    lightning_module: nn.Module
    denoiser: GemNetTDenoiser
    model_path: str
    load_epoch: int | str
    checkpoint_path: str
    checkpoint_sha256: str
    hidden_dim: int
    provenance: dict[str, Any]


def resolve_checkpoint_path(model_path: str | Path, load_epoch: int | str = 294) -> Path:
    info = MatterGenCheckpointInfo(
        model_path=str(model_path),
        load_epoch=load_epoch,  # type: ignore[arg-type]
        strict_checkpoint_loading=True,
    )
    return Path(info.checkpoint_path)


def load_molecular_csp_gemnet(
    *,
    model_path: str | Path,
    load_epoch: int | str = 294,
    checkpoint_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    freeze: bool = True,
) -> LoadedGemNetBundle:
    """Load pretrained molecular-CSP diffusion module and return its score model (GemNetTDenoiser).

    Prefer official:
        MatterGenCheckpointInfo(model_path, load_epoch) + load_model_diffusion(info)
    over ad-hoc torch.load + load_state_dict.
    """
    del map_location  # load_model_diffusion chooses device via get_device()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"MatterGen model_path not found: {model_path}. "
            "N1 hidden_source=gemnet requires the le50 molCSP run directory. "
            "No ContextCrystalEncoder fallback."
        )
    info = MatterGenCheckpointInfo(
        model_path=str(model_path),
        load_epoch=load_epoch,  # type: ignore[arg-type]
        strict_checkpoint_loading=strict,
    )
    resolved = Path(info.checkpoint_path)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"explicit checkpoint_path missing: {checkpoint_path}")
        # Audit that epoch resolver and explicit path refer to the same ckpt file.
        if checkpoint_path.resolve() != resolved.resolve() and checkpoint_path.name != resolved.name:
            raise ValueError(
                "gemnet.checkpoint_path does not match MatterGenCheckpointInfo resolution:\n"
                f"  explicit={checkpoint_path}\n  resolved={resolved}\n"
                "Pass consistent model_path + load_epoch, or omit checkpoint_path."
            )
    if not resolved.exists():
        raise FileNotFoundError(
            f"Resolved checkpoint missing: {resolved}. "
            "N1 will not fall back to ContextCrystalEncoder."
        )

    pl_module = load_model_diffusion(info)
    # DiffusionLightningModule.diffusion_module.model is the ScoreModel / GemNetTDenoiser
    denoiser = pl_module.diffusion_module.model
    if not isinstance(denoiser, GemNetTDenoiser):
        inner = getattr(denoiser, "model", None)
        if isinstance(inner, GemNetTDenoiser):
            denoiser = inner
        else:
            raise TypeError(
                f"Expected GemNetTDenoiser score model, got {type(denoiser)!r}. "
                "N1 requires molecular-CSP GemNetTDenoiser."
            )
    pl_module.eval()
    if freeze:
        freeze_module(denoiser)
        # Freeze any property embeddings / molecule conditioner on the denoiser path.
        for attr in ("molecule_conditioner", "property_embeddings", "property_embeddings_adapt"):
            sub = getattr(denoiser, attr, None)
            if isinstance(sub, nn.Module):
                freeze_module(sub)

    hidden_dim = int(getattr(denoiser, "hidden_dim", 512))
    sha = file_sha256(resolved)
    prov = {
        "loader": "mattergen.common.utils.eval_utils.load_model_diffusion",
        "checkpoint_info": "mattergen.common.utils.data_classes.MatterGenCheckpointInfo",
        "model_path": str(model_path),
        "load_epoch": load_epoch,
        "checkpoint_path": str(resolved),
        "checkpoint_sha256": sha,
        "score_model_class": type(denoiser).__name__,
        "hidden_dim": hidden_dim,
        "not_mattergen_base": True,
        "hidden_source": "gemnet_node_embeddings",
        "gemnet_backbone_frozen": bool(freeze),
        "strict_checkpoint_loading": bool(strict),
    }
    return LoadedGemNetBundle(
        lightning_module=pl_module,
        denoiser=denoiser,
        model_path=str(model_path),
        load_epoch=load_epoch,
        checkpoint_path=str(resolved),
        checkpoint_sha256=sha,
        hidden_dim=hidden_dim,
        provenance=prov,
    )


def build_mol_conditioning_from_sample(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Build molecule-conditioner inputs without oracle assignment leakage.

    Prefer native ``mol_*`` tensors if the fixed sample already has them.
    Otherwise expand the molecular *role* graph across copies using ``role``
    and ``copy`` only to place edges (topology reconstruction), never as
    assignment-head features.
    """
    extra: dict[str, torch.Tensor] = {}
    if all(k in sample and sample[k] is not None for k in ("mol_x", "mol_bond_edge_index", "mol_bond_attr")):
        for k in MOL_CONDITIONING_KEYS:
            if k in sample and sample[k] is not None:
                val = sample[k]
                if torch.is_tensor(val):
                    extra[k] = val
        # Explicitly never forward copy id
        return extra

    # Reconstruct crystal-level mol graph from role graph × copies.
    if not all(k in sample for k in ("z", "role", "copy", "role_edge_index", "role_bond_type")):
        raise KeyError(
            "Cannot build mol conditioning: fixed sample missing both native mol_* fields "
            "and role/copy graph fields needed for reconstruction."
        )
    z = sample["z"].long()
    role = sample["role"].long()
    copy = sample["copy"].long()
    role_ei = sample["role_edge_index"].long()
    role_bt = sample["role_bond_type"].long()
    n = int(z.numel())
    device = z.device

    # Map (role, copy) -> atom index
    # Assume unique (role, copy) pairs for molecular crystal packing.
    index = {}
    for i in range(n):
        index[(int(role[i]), int(copy[i]))] = i

    edges_src: list[int] = []
    edges_dst: list[int] = []
    bond_attr_rows: list[list[int]] = []
    e_role = int(role_ei.shape[1]) if role_ei.ndim == 2 else 0
    unique_copies = sorted({int(c) for c in copy.tolist()})
    for e in range(e_role):
        r_a = int(role_ei[0, e])
        r_b = int(role_ei[1, e])
        bt = int(role_bt[e]) if role_bt.ndim == 1 else int(role_bt[e].reshape(-1)[0])
        bt = min(max(bt, 0), 7)
        aromatic = int(bt == 4)
        for c in unique_copies:
            ia = index.get((r_a, c))
            ib = index.get((r_b, c))
            if ia is None or ib is None:
                continue
            edges_src.append(ia)
            edges_dst.append(ib)
            bond_attr_rows.append([bt, aromatic])

    if edges_src:
        mol_bond_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
        mol_bond_attr = torch.tensor(bond_attr_rows, dtype=torch.long, device=device)
    else:
        mol_bond_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        mol_bond_attr = torch.empty((0, 2), dtype=torch.long, device=device)

    # OGB-style 6-col atom features; only atomic number + degree filled.
    degree = torch.zeros(n, dtype=torch.long, device=device)
    if mol_bond_edge_index.numel() > 0:
        degree.scatter_add_(
            0,
            mol_bond_edge_index[0],
            torch.ones(mol_bond_edge_index.shape[1], dtype=torch.long, device=device),
        )
    mol_x = torch.zeros(n, 6, dtype=torch.long, device=device)
    mol_x[:, 0] = z.clamp(0, 127)
    mol_x[:, 2] = degree.clamp(0, 11)

    extra = {
        "mol_x": mol_x,
        "mol_bond_edge_index": mol_bond_edge_index,
        "mol_bond_attr": mol_bond_attr,
        "mol_atom_id": role,
    }
    return extra


@dataclass
class GemNetHiddenOutput:
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor | None = None
    timestep_embedding: torch.Tensor | None = None
    hidden_dim: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class GemNetHiddenExtractor(nn.Module):
    """Extract ``node_embeddings`` via the same path as ``GemNetTDenoiser.forward``.

    Frozen pretrained denoiser; assignment heads train on detached hiddens.
    Mirrors denoiser.py GemNetTDenoiser.forward up to gemnet(...).node_embeddings
    so timestep encoding, property embeddings, molecule conditioning, and OTF
    graph construction all match molecular-CSP sampling.
    """

    def __init__(self, denoiser: GemNetTDenoiser):
        super().__init__()
        self.denoiser = denoiser
        freeze_module(self.denoiser)

    @property
    def hidden_dim(self) -> int:
        return int(getattr(self.denoiser, "hidden_dim", 512))

    def build_chemgraph_from_sample(
        self,
        *,
        frac: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
        extra: dict | None = None,
    ) -> ChemGraph:
        """Build a **batched** ChemGraph (batch size 1).

        ``GemNetTDenoiser.forward`` calls ``x.get_batch_idx("pos")``, which
        asserts ``isinstance(x, pyg Batch)``.  A bare single-structure ChemGraph
        therefore raises AssertionError; wrap with ``Batch.from_data_list``.
        """
        n = int(atomic_numbers.numel())
        lat = cell if cell.ndim == 3 else cell.unsqueeze(0)
        if lat.shape[0] != 1:
            raise ValueError(f"N1 extractor expects a single lattice, got cell shape {tuple(lat.shape)}")
        kwargs: dict[str, Any] = dict(
            atomic_numbers=atomic_numbers.long(),
            pos=frac,
            cell=lat,
            # graph-level: shape [1] so PyG batch keeps [B] after collate
            num_atoms=torch.tensor([n], dtype=torch.long, device=frac.device),
            num_nodes=n,
        )
        if extra:
            for key, val in extra.items():
                if key in ORACLE_ASSIGNMENT_DENYLIST:
                    continue
                if key in MOL_CONDITIONING_KEYS or key.startswith("mol_"):
                    if key == "mol_copy_id":
                        continue
                    if torch.is_tensor(val):
                        kwargs[key] = val.to(device=frac.device)
                    else:
                        kwargs[key] = val
        single = ChemGraph(**kwargs)
        # ChemGraphBatch via PyG DynamicInheritance (same as training collate path)
        batched = Batch.from_data_list([single])
        return batched  # type: ignore[return-value]

    def extract(
        self,
        *,
        frac: torch.Tensor,
        cell: torch.Tensor,
        atomic_numbers: torch.Tensor,
        t: torch.Tensor,
        chemgraph_extra: dict | None = None,
        detach: bool = True,
    ) -> GemNetHiddenOutput:
        """Run GemNetTDenoiser internal path (t + mol condition + OTF graph) → node_embeddings."""
        self.denoiser.eval()
        x = self.build_chemgraph_from_sample(
            frac=frac, cell=cell, atomic_numbers=atomic_numbers, extra=chemgraph_extra
        )
        t = t.to(device=frac.device).reshape(-1)
        if t.numel() != 1:
            raise ValueError("N1 fixed-sample extractor expects scalar t per crystal")

        # Mirror GemNetTDenoiser.forward (mattergen/denoiser.py) up to node_embeddings.
        # x is a ChemGraphBatch (batch size 1); get_batch_idx is valid.
        frac_coords, lattice, atom_types, num_atoms, batch = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        if batch is None:
            # dense fallback should not happen for per-atom pos; keep N1 robust
            batch = torch.zeros(
                int(frac_coords.shape[0]), dtype=torch.long, device=frac_coords.device
            )
        t_enc = self.denoiser.noise_level_encoding(t).to(lattice.device)
        z_per_crystal = t_enc
        property_embedding_values = get_property_embeddings(
            batch=x, property_embeddings=self.denoiser.property_embeddings
        )
        if len(property_embedding_values) > 0:
            z_per_crystal = torch.cat([z_per_crystal, property_embedding_values], dim=-1)

        node_condition = None
        if self.denoiser.molecule_conditioner is not None:
            try:
                node_condition = self.denoiser.molecule_conditioner(x)
            except Exception as exc:
                raise RuntimeError(
                    "GemNet molecule_conditioner failed on ChemGraph. "
                    "Provide mol_* fields for molecular-CSP conditioning "
                    f"(same as sampler). Original error: {exc}"
                ) from exc
            if (
                node_condition is not None
                and self.denoiser.molecule_conditioner_gate_center is not None
            ):
                gate = torch.sigmoid(
                    (float(self.denoiser.molecule_conditioner_gate_center) - t)
                    / float(self.denoiser.molecule_conditioner_gate_width)
                )
                min_scale = float(self.denoiser.molecule_conditioner_gate_min_scale)
                gate = min_scale + (1.0 - min_scale) * gate
                node_condition = node_condition * gate[batch].unsqueeze(-1)

        # Frozen backbone: no grad into GemNet; assignment heads train after detach.
        with torch.no_grad():
            output = self.denoiser.gemnet(
                z=z_per_crystal,
                frac_coords=frac_coords,
                atom_types=atom_types,
                num_atoms=num_atoms,
                batch=batch,
                lengths=None,
                angles=None,
                lattice=lattice,
                edge_index=None,
                to_jimages=None,
                num_bonds=None,
                node_condition=node_condition,
            )
            h = output.node_embeddings
            if detach:
                h = h.detach()

        if h.shape[0] != atomic_numbers.shape[0]:
            raise RuntimeError(
                f"GemNet node_embeddings N={h.shape[0]} != atomic_numbers N={atomic_numbers.shape[0]}"
            )
        edge_h = getattr(output, "edge_embeddings", None)
        if edge_h is not None and detach:
            edge_h = edge_h.detach()
        meta = {
            "hidden_source": "gemnet_node_embeddings",
            "tensor": "GemNetTDenoiser.gemnet(...).node_embeddings",
            "hidden_dim": int(h.shape[-1]),
            "num_atoms": int(h.shape[0]),
            "dtype": str(h.dtype),
            "device": str(h.device),
            "timestep_conditioning": True,
            "molecule_conditioner_used": node_condition is not None,
            "context_crystal_encoder_used": False,
            "atom_index_order": "matches_noisy_batch_atomic_numbers_order",
        }
        return GemNetHiddenOutput(
            node_embeddings=h,
            edge_embeddings=edge_h if torch.is_tensor(edge_h) else None,
            timestep_embedding=t_enc.detach() if detach else t_enc,
            hidden_dim=int(h.shape[-1]),
            metadata=meta,
        )

    def geometry_score_probe(self, x: ChemGraph, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full denoiser geometry outputs (for invariance tests). Assignment must not alter these."""
        self.denoiser.eval()
        with torch.no_grad():
            out = self.denoiser(x, t.reshape(-1))
        return {
            "pos_score": out["pos"].detach().clone(),
            "cell_score": out["cell"].detach().clone(),
            "atomic_logits": out["atomic_numbers"].detach().clone(),
        }
