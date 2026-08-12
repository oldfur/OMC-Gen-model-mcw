"""Load MatterGen + frozen N1 + N2.1 causal edge modulator."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    file_sha256,
    freeze_module,
    load_molecular_csp_gemnet,
)
from mattergen.assignment.noisy_copy_assignment.module import (
    NoisyCopyAssignmentConfig,
    NoisyCopyAssignmentN1,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.gates import NoiseGateConfig
from mattergen.assignment.soft_c_geometry_feedback_n2.setup_utils import (
    build_chemgraph_batch,
    load_fixed_sample_bundle,
)

from .module import CausalEdgeConfig, SoftCCausalEdgeN21

# re-export for scripts
__all__ = [
    "build_chemgraph_batch",
    "load_fixed_sample_bundle",
    "load_n21_stack",
]


def load_n21_stack(
    *,
    cfg: dict,
    device: torch.device,
    n1_ckpt_path: str | Path,
    mattergen_model_path: str | None = None,
    mattergen_load_epoch: int | None = None,
    mattergen_checkpoint: str | None = None,
) -> tuple[SoftCCausalEdgeN21, Any, dict]:
    gem_cfg = dict(cfg.get("gemnet") or {})
    model_path = mattergen_model_path or gem_cfg.get("model_path")
    load_epoch = (
        mattergen_load_epoch
        if mattergen_load_epoch is not None
        else gem_cfg.get("load_epoch", 294)
    )
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

    sample_probe = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)

    n1_flat = {
        "hidden_source": "gemnet",
        "freeze_gemnet_backbone": True,
        "geometry_feedback": False,
        "hidden_dim": int((cfg.get("n1") or {}).get("hidden_dim", 256)),
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
    n1_src_sha = n1_ckpt.get("mattergen_checkpoint_sha256")
    if n1_src_sha and n1_src_sha != bundle.checkpoint_sha256:
        raise RuntimeError(
            f"N1_GEMNET_SOURCE_MISMATCH: N1 sha={n1_src_sha} != N2.1 sha={bundle.checkpoint_sha256}"
        )
    n1.load_state_dict(n1_ckpt["state_dict"], strict=False)
    freeze_module(n1)
    n1.set_gemnet_denoiser(denoiser, freeze=True)
    n1.prepare_mol_conditioning_from_sample(
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample_probe.items()}
    )

    fb = cfg.get("soft_c_feedback") or {}
    ng = fb.get("noise_gate") or {}
    n21_cfg = CausalEdgeConfig(
        enabled=bool(fb.get("enabled", True)),
        noise_gate=NoiseGateConfig(
            enabled=bool(ng.get("enabled", True)),
            full_on_t_fraction=float(ng.get("full_on_t_fraction", 0.30)),
            full_off_t_fraction=float(ng.get("full_off_t_fraction", 0.60)),
        ),
        freeze_base_gemnet=bool(cfg.get("freeze_base_gemnet", True)),
        freeze_n1=True,
        group_context_enabled=False,
        mode=str(cfg.get("n21_mode", "B2_correct_c")),  # type: ignore[arg-type]
    )
    model = SoftCCausalEdgeN21(
        denoiser=denoiser,
        n1_model=n1,
        partition=partition,
        config=n21_cfg,
        mattergen_provenance=bundle.provenance,
        n1_provenance={
            "n1_checkpoint": str(n1_ckpt_path),
            "n1_checkpoint_sha256": file_sha256(Path(n1_ckpt_path)),
            "mattergen_checkpoint_sha256": n1_src_sha,
        },
    ).to(device)
    for p in model.edge_modulator.parameters():
        p.requires_grad_(True)

    return model, pl_module, {
        "mattergen_checkpoint_sha256": bundle.checkpoint_sha256,
        "mattergen_model_path": bundle.model_path,
        "mattergen_load_epoch": bundle.load_epoch,
        "mattergen_checkpoint": bundle.checkpoint_path,
        "n1_checkpoint": str(n1_ckpt_path),
        "n1_checkpoint_sha256": file_sha256(Path(n1_ckpt_path)),
    }
