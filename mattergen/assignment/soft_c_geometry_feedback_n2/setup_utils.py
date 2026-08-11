"""Shared N2 loading / target / chemgraph helpers."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from mattergen.assignment.global_copy_assembly.orbit_membership import (
    build_orbit_partition,
    collapse_roles_to_orbit_membership,
)
from mattergen.assignment.global_copy_assembly.orbit_module import prepare_backbone
from mattergen.assignment.global_copy_assembly.orbit_targets import (
    OrbitAwareAssemblyTarget,
    OrbitAttachmentTarget,
    build_orbit_aware_target,
)
from mattergen.assignment.global_copy_assembly.targets import AssemblyTarget
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    build_mol_conditioning_from_sample,
    file_sha256,
    freeze_module,
    load_molecular_csp_gemnet,
)
from mattergen.assignment.noisy_copy_assignment.module import (
    NoisyCopyAssignmentConfig,
    NoisyCopyAssignmentN1,
)
from mattergen.common.data.chemgraph import ChemGraph

from .gates import NoiseGateConfig
from .module import SoftCFeedbackConfig, SoftCGeometryFeedbackN2


def move_o2_target(o2_target, device):
    st = o2_target.singleton_target
    singleton = AssemblyTarget(
        role_sets={r: n.to(device) for r, n in st.role_sets.items()},
        permutations={r: p.to(device) for r, p in st.permutations.items()},
        anchor_role=st.anchor_role,
        K=st.K,
        M=st.M,
    )
    orbit_targets = tuple(
        OrbitAttachmentTarget(
            orbit_index=ot.orbit_index,
            atom_indices=ot.atom_indices.to(device),
            pairs_local=ot.pairs_local,
            atoms_per_copy=ot.atoms_per_copy,
        )
        for ot in o2_target.orbit_targets
    )
    return OrbitAwareAssemblyTarget(
        partition=o2_target.partition,
        bar_r=o2_target.bar_r.to(device),
        singleton_roles=o2_target.singleton_roles,
        singleton_target=singleton,
        local_to_role=o2_target.local_to_role,
        role_to_local=o2_target.role_to_local,
        orbit_targets=orbit_targets,
        K=o2_target.K,
        N=o2_target.N,
    )


def load_fixed_sample_bundle(cfg: dict, device: torch.device):
    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    if sample["id"] != cfg["fixed_sample_id"]:
        raise ValueError("sample id mismatch")
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)
    art = [
        json.loads(l)
        for l in Path(cfg["predicted_role_artifact_path"]).read_text().splitlines()
        if l.strip()
    ][-1]
    role_assignment = torch.tensor(art["role_assignment"], dtype=torch.long)
    oracle_bar = collapse_roles_to_orbit_membership(sample["role"].long(), partition)
    backbone = prepare_backbone(partition, sample["role_edge_index"], sample["role_bond_type"])
    anchor = backbone.singleton_roles[backbone.tree.root]
    o2_target = build_orbit_aware_target(
        role_assignment,
        sample["copy"],
        partition=partition,
        K=int(sample["Z"]),
        anchor_role=anchor,
    )
    sample_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    o2_target = move_o2_target(o2_target, device)
    oracle_bar = oracle_bar.to(device)
    return sample_d, partition, backbone, o2_target, oracle_bar


def build_chemgraph_batch(sample: dict, frac, cell, extra: dict | None = None) -> ChemGraph:
    n = int(sample["N"])
    lat = cell if cell.ndim == 3 else cell.unsqueeze(0)
    kwargs: dict[str, Any] = dict(
        atomic_numbers=sample["z"].long(),
        pos=frac,
        cell=lat,
        num_atoms=torch.tensor([n], dtype=torch.long, device=frac.device),
        num_nodes=n,
    )
    extra = extra or build_mol_conditioning_from_sample(sample)
    for k, v in extra.items():
        if k == "mol_copy_id":
            continue
        if torch.is_tensor(v):
            kwargs[k] = v.to(device=frac.device)
        else:
            kwargs[k] = v
    return Batch.from_data_list([ChemGraph(**kwargs)])  # type: ignore[return-value]


def load_n2_stack(
    *,
    cfg: dict,
    device: torch.device,
    n1_ckpt_path: str | Path,
    mattergen_model_path: str | None = None,
    mattergen_load_epoch: int | None = None,
    mattergen_checkpoint: str | None = None,
) -> tuple[SoftCGeometryFeedbackN2, Any, dict]:
    """Load MatterGen epoch294 + frozen N1 + N2 adapters."""
    gem_cfg = dict(cfg.get("gemnet") or {})
    model_path = mattergen_model_path or gem_cfg.get("model_path")
    load_epoch = mattergen_load_epoch if mattergen_load_epoch is not None else gem_cfg.get("load_epoch", 294)
    ckpt_path = mattergen_checkpoint or gem_cfg.get("checkpoint_path")
    if not model_path:
        raise RuntimeError("mattergen model_path required")

    bundle = load_molecular_csp_gemnet(
        model_path=model_path,
        load_epoch=load_epoch,
        checkpoint_path=ckpt_path,
        freeze=True,
        strict=True,
    )
    denoiser = bundle.denoiser.to(device)
    pl_module = bundle.lightning_module.to(device)

    # N1 model shell
    sample_probe = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)

    n1_cfg_raw = cfg.get("n1") or {}
    n1_flat = {
        "hidden_source": "gemnet",
        "freeze_gemnet_backbone": True,
        "geometry_feedback": False,
        "hidden_dim": int(n1_cfg_raw.get("hidden_dim", 256)),
        "produce_soft_c": True,
    }
    allowed = {f.name for f in fields(NoisyCopyAssignmentConfig)}
    n1_cfg = NoisyCopyAssignmentConfig(**{k: v for k, v in n1_flat.items() if k in allowed})
    n1 = NoisyCopyAssignmentN1(n1_cfg, partition).to(device)
    n1.set_gemnet_denoiser(denoiser, freeze=True)
    n1.prepare_mol_conditioning_from_sample(sample_probe)
    if n1.gemnet_proj is not None:
        n1.gemnet_proj.to(device)

    n1_ckpt = torch.load(n1_ckpt_path, map_location="cpu", weights_only=False)
    # Provenance: N1 must come from same MatterGen ckpt
    n1_src_sha = n1_ckpt.get("mattergen_checkpoint_sha256")
    if n1_src_sha and n1_src_sha != bundle.checkpoint_sha256:
        raise RuntimeError(
            f"N1_GEMNET_SOURCE_MISMATCH: N1 source sha={n1_src_sha} "
            f"!= N2 MatterGen sha={bundle.checkpoint_sha256}"
        )
    missing, unexpected = n1.load_state_dict(n1_ckpt["state_dict"], strict=False)
    freeze_module(n1)
    # re-inject frozen denoiser after load (state_dict may not include gemnet)
    n1.set_gemnet_denoiser(denoiser, freeze=True)
    n1.prepare_mol_conditioning_from_sample(
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_probe.items()}
    )

    fb = cfg.get("soft_c_feedback") or {}
    ng = fb.get("noise_gate") or {}
    n2_cfg = SoftCFeedbackConfig(
        enabled=bool(fb.get("enabled", True)),
        edge_semantics_enabled=bool((fb.get("edge_semantics") or {}).get("enabled", True)),
        group_context_enabled=bool((fb.get("group_context") or {}).get("enabled", True)),
        pairwise_confidence_enabled=bool((fb.get("pairwise_confidence") or {}).get("enabled", True)),
        noise_gate=NoiseGateConfig(
            enabled=bool(ng.get("enabled", True)),
            full_on_t_fraction=float(ng.get("full_on_t_fraction", 0.30)),
            full_off_t_fraction=float(ng.get("full_off_t_fraction", 0.60)),
        ),
        freeze_base_gemnet=bool(cfg.get("freeze_base_gemnet", True)),
        freeze_n1=True,
        molecules_atoms_m=int(sample_probe.get("M", 10)),
        mode=str(cfg.get("n2_mode", "B2_combined")),  # type: ignore[arg-type]
    )
    model = SoftCGeometryFeedbackN2(
        denoiser=denoiser,
        n1_model=n1,
        partition=partition,
        config=n2_cfg,
        mattergen_provenance=bundle.provenance,
        n1_provenance={
            "n1_checkpoint": str(n1_ckpt_path),
            "n1_checkpoint_sha256": file_sha256(Path(n1_ckpt_path)),
            "mattergen_checkpoint_sha256": n1_src_sha,
            "missing_keys": list(missing)[:20],
            "unexpected_keys": list(unexpected)[:20],
        },
    ).to(device)
    # adapters trainable
    for p in model.edge_adapter.parameters():
        p.requires_grad_(True)
    for p in model.group_adapter.parameters():
        p.requires_grad_(True)

    return model, pl_module, {
        "mattergen_checkpoint_sha256": bundle.checkpoint_sha256,
        "mattergen_model_path": bundle.model_path,
        "mattergen_load_epoch": bundle.load_epoch,
        "mattergen_checkpoint": bundle.checkpoint_path,
        "n1_checkpoint": str(n1_ckpt_path),
        "n1_checkpoint_sha256": file_sha256(Path(n1_ckpt_path)),
    }
