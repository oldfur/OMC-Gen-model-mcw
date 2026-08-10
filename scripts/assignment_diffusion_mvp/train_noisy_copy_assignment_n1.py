#!/usr/bin/env python3
"""Train N1 observational noisy copy assignment on frozen molecular-CSP GemNet hiddens.

Default: hidden_source=gemnet (le50 molCSP epoch294). Does not train pos/cell scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
import sys

import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mattergen.assignment.global_copy_assembly.orbit_membership import (
    build_orbit_partition,
    collapse_roles_to_orbit_membership,
)
from mattergen.assignment.global_copy_assembly.orbit_module import prepare_backbone
from mattergen.assignment.global_copy_assembly.orbit_targets import build_orbit_aware_target
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    load_molecular_csp_gemnet,
    parameter_sha256,
)
from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
    PROVENANCE,
)
from mattergen.assignment.noisy_copy_assignment.module import (
    NoisyCopyAssignmentConfig,
    NoisyCopyAssignmentN1,
)
from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(req)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return d


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["assignment_n1"]


def _config_hash(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _move_o2_target(o2_target, device):
    from mattergen.assignment.global_copy_assembly.orbit_targets import (
        OrbitAwareAssemblyTarget,
        OrbitAttachmentTarget,
    )
    from mattergen.assignment.global_copy_assembly.targets import AssemblyTarget

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


def _inject_gemnet(model: NoisyCopyAssignmentN1, cfg: dict, args, sample: dict, device) -> dict:
    """Load molecular-CSP GemNet and inject into N1. Fail loudly — no context fallback."""
    hidden_source = str(cfg.get("hidden_source", "gemnet"))
    if hidden_source != "gemnet":
        if hidden_source == "context_encoder":
            print(
                json.dumps(
                    {
                        "event": "n1_ablation_context_encoder",
                        "PRIMARY_N1": "gemnet",
                        "ABLATION": "context_encoder",
                        "note": "Running standalone context encoder ablation, not primary N1.",
                    }
                ),
                flush=True,
            )
            return {
                "hidden_source": "context_encoder",
                "context_crystal_encoder_used": True,
                "gemnet_backbone_frozen": False,
                "mattergen_model_path": None,
                "mattergen_load_epoch": None,
                "mattergen_checkpoint": None,
                "mattergen_checkpoint_sha256": None,
            }
        raise ValueError(f"unknown hidden_source={hidden_source}")

    gem_cfg = dict(cfg.get("gemnet") or {})
    model_path = args.mattergen_model_path or gem_cfg.get("model_path")
    load_epoch = args.mattergen_load_epoch
    if load_epoch is None:
        load_epoch = gem_cfg.get("load_epoch", 294)
    ckpt_path = args.mattergen_checkpoint or gem_cfg.get("checkpoint_path")
    if not model_path:
        raise RuntimeError(
            "hidden_source=gemnet requires gemnet.model_path or --mattergen-model-path. "
            "No ContextCrystalEncoder fallback."
        )
    if gem_cfg.get("require_checkpoint", True):
        from pathlib import Path as P

        if not P(model_path).exists():
            raise FileNotFoundError(
                f"MatterGen model_path missing: {model_path}. "
                "N1 primary experiment aborts (no context encoder fallback)."
            )
        if ckpt_path and not P(ckpt_path).exists():
            raise FileNotFoundError(
                f"MatterGen checkpoint missing: {ckpt_path}. "
                "N1 primary experiment aborts (no context encoder fallback)."
            )

    bundle = load_molecular_csp_gemnet(
        model_path=model_path,
        load_epoch=load_epoch,
        checkpoint_path=ckpt_path,
        freeze=bool(cfg.get("freeze_gemnet_backbone", True)),
        strict=True,
    )
    # Move denoiser to device
    denoiser = bundle.denoiser.to(device)
    model.set_gemnet_denoiser(denoiser, freeze=True)
    model.prepare_mol_conditioning_from_sample(
        {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    )
    # Move projection head weights to device if created after .to(device) earlier
    if model.gemnet_proj is not None:
        model.gemnet_proj.to(device)

    prov = {
        **bundle.provenance,
        "hidden_source": "gemnet_node_embeddings",
        "context_crystal_encoder_used": False,
        "gemnet_backbone_frozen": True,
        "geometry_feedback": False,
        "use_copy_id_as_input": False,
        "use_oracle_C_as_input": False,
        "noise_source": PROVENANCE["noise_source"],
        "mattergen_model_path": bundle.model_path,
        "mattergen_load_epoch": bundle.load_epoch,
        "mattergen_checkpoint": bundle.checkpoint_path,
        "mattergen_checkpoint_sha256": bundle.checkpoint_sha256,
        "soft_c_semantics": SOFT_C_SEMANTICS,
    }
    return prov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mattergen-model-path", type=str, default=None)
    parser.add_argument("--mattergen-load-epoch", type=int, default=None)
    parser.add_argument("--mattergen-checkpoint", type=str, default=None)
    parser.add_argument(
        "--hidden-source",
        type=str,
        default=None,
        help="Override config hidden_source (gemnet|context_encoder). Primary N1 must be gemnet.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to train N1 without --execute")

    cfg = load_cfg(args.config)
    if args.hidden_source is not None:
        cfg["hidden_source"] = args.hidden_source
    if not cfg.get("enabled", True):
        raise SystemExit("assignment_n1.enabled is false")
    if cfg.get("geometry_feedback"):
        raise RuntimeError("N1 forbids geometry_feedback")

    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    if sample["id"] != cfg["fixed_sample_id"]:
        raise ValueError("sample id mismatch")
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)

    art_path = Path(cfg["predicted_role_artifact_path"])
    art = [json.loads(l) for l in art_path.read_text().splitlines() if l.strip()][-1]
    if art.get("role_source") != "geometry_only_hard_r":
        raise ValueError("expected geometry_only_hard_r artifact")
    role_assignment = torch.tensor(art["role_assignment"], dtype=torch.long)
    oracle_bar = collapse_roles_to_orbit_membership(sample["role"].long(), partition)
    backbone = prepare_backbone(partition, sample["role_edge_index"], sample["role_bond_type"])
    anchor = backbone.singleton_roles[backbone.tree.root]
    o2_target = build_orbit_aware_target(
        role_assignment, sample["copy"], partition=partition, K=int(sample["Z"]), anchor_role=anchor
    )

    allowed = {f.name for f in fields(NoisyCopyAssignmentConfig)}
    loss_cfg = cfg.get("loss") or {}
    flat = {**cfg, **{k: loss_cfg[k] for k in loss_cfg}}
    # Drop nested tables that are not config fields
    flat.pop("gemnet", None)
    flat.pop("context_encoder", None)
    flat.pop("evaluation", None)
    flat.pop("structured_decoder", None)
    flat.pop("orbit_modes", None)
    flat.pop("loss", None)
    model_cfg = NoisyCopyAssignmentConfig(**{k: v for k, v in flat.items() if k in allowed})
    # Enforce primary defaults
    if model_cfg.hidden_source == "gemnet":
        model_cfg.fail_on_gemnet_fallback = True

    model = NoisyCopyAssignmentN1(model_cfg, partition)
    device = resolve_device(str(cfg.get("device", "auto")))
    sample_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    o2_target = _move_o2_target(o2_target, device)
    oracle_bar = oracle_bar.to(device)
    model = model.to(device)

    gem_prov = _inject_gemnet(model, cfg, args, sample, device)
    if model_cfg.freeze_gemnet_backbone:
        model.freeze_backbone()

    # Runtime provenance — abort if primary N1 is not GemNet
    if model_cfg.hidden_source == "gemnet":
        if gem_prov.get("hidden_source") != "gemnet_node_embeddings":
            raise RuntimeError(
                f"PRIMARY N1 requires hidden_source=gemnet_node_embeddings, got {gem_prov}"
            )
        if gem_prov.get("context_crystal_encoder_used"):
            raise RuntimeError("PRIMARY N1 must not use ContextCrystalEncoder")

    audit = model.param_audit()
    print(
        json.dumps(
            {
                "event": "n1_param_audit",
                **audit,
                "trainable GemNet params": audit["gemnet_trainable_params"],
                "trainable assignment params": audit["assignment_trainable_params"],
            }
        ),
        flush=True,
    )
    if model_cfg.hidden_source == "gemnet" and audit["gemnet_trainable_params"] != 0:
        raise RuntimeError(
            f"FROZEN_BACKBONE_VIOLATION: gemnet_trainable_params={audit['gemnet_trainable_params']}"
        )
    if audit["assignment_trainable_params"] <= 0:
        raise RuntimeError("ASSIGNMENT_HEAD_NOT_UPDATED: no trainable assignment parameters")

    params = list(model.trainable_assignment_parameters())
    if not params:
        raise RuntimeError("no trainable assignment parameters")

    # Hash before training
    gemnet_hash_before = model.gemnet_parameter_hash()
    assign_hash_before = model.assignment_parameter_hash()

    opt = torch.optim.AdamW(
        params,  # explicit assignment-only list — never model.parameters()
        lr=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    steps = int(args.steps or cfg.get("steps", 2000))
    clip = float(cfg.get("gradient_clip_norm", 1.0))
    log_every = int(cfg.get("log_every_steps", 20))
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))

    runtime_prov = {
        "NOISE_SOURCE": PROVENANCE["noise_source"],
        "MATTERGEN_MODEL_PATH": gem_prov.get("mattergen_model_path") or gem_prov.get("model_path"),
        "MATTERGEN_LOAD_EPOCH": gem_prov.get("mattergen_load_epoch") or gem_prov.get("load_epoch"),
        "MATTERGEN_CHECKPOINT": gem_prov.get("mattergen_checkpoint") or gem_prov.get("checkpoint_path"),
        "MATTERGEN_CHECKPOINT_SHA256": gem_prov.get("mattergen_checkpoint_sha256")
        or gem_prov.get("checkpoint_sha256"),
        "HIDDEN_SOURCE": "gemnet_node_embeddings"
        if model_cfg.hidden_source == "gemnet"
        else "context_crystal_encoder",
        "CONTEXT_CRYSTAL_ENCODER_USED": model_cfg.hidden_source == "context_encoder",
        "GEMNET_BACKBONE_FROZEN": bool(model_cfg.freeze_gemnet_backbone)
        and model_cfg.hidden_source == "gemnet",
        "GEOMETRY_FEEDBACK": False,
        "USE_COPY_ID_AS_INPUT": False,
        "USE_ORACLE_C_AS_INPUT": False,
        "SOFT_C_SEMANTICS": SOFT_C_SEMANTICS,
        "PRIMARY_N1": "gemnet",
        "ABLATION": "context_encoder",
        "run_config_hash": _config_hash(cfg),
        "noise_config_hash": _config_hash(PROVENANCE),
        "gemnet_parameter_hash_before": gemnet_hash_before,
        "assignment_parameter_hash_before": assign_hash_before,
        "param_audit": audit,
        "loader": gem_prov.get("loader"),
    }
    for k, v in runtime_prov.items():
        if k in {
            "NOISE_SOURCE",
            "MATTERGEN_MODEL_PATH",
            "MATTERGEN_LOAD_EPOCH",
            "MATTERGEN_CHECKPOINT",
            "HIDDEN_SOURCE",
            "CONTEXT_CRYSTAL_ENCODER_USED",
            "GEMNET_BACKBONE_FROZEN",
            "GEOMETRY_FEEDBACK",
            "USE_COPY_ID_AS_INPUT",
            "USE_ORACLE_C_AS_INPUT",
        }:
            print(f"{k}={v}", flush=True)

    (out / "noise_process_audit.json").write_text(
        json.dumps(
            {
                **PROVENANCE,
                "adapter": "MatterGenNativeNoiseAdapter.corrupt_fixed_sample",
                "independent_noise_implementation": False,
                "fields": ["pos", "cell"],
                "limit_density": cfg.get("limit_density", 0.05),
            },
            indent=2,
        )
    )
    (out / "runtime_provenance.json").write_text(json.dumps(runtime_prov, indent=2, default=str))
    (out / "config_audit.json").write_text(
        json.dumps(
            {
                "status": "NOISY_COPY_ASSIGNMENT_N1",
                "geometry_feedback": False,
                "freeze_gemnet_backbone": model_cfg.freeze_gemnet_backbone,
                "hidden_source": model_cfg.hidden_source,
                "use_copy_id_as_input": False,
                "use_oracle_C_as_input": False,
                "noise_source": "mattergen_native",
                "sample": sample["id"],
                "device": str(device),
                "trainable_param_count": sum(p.numel() for p in params),
                "soft_c_semantics": SOFT_C_SEMANTICS,
                **{k: runtime_prov[k] for k in runtime_prov if k.isupper()},
            },
            indent=2,
            default=str,
        )
    )

    print(
        json.dumps(
            {
                "event": "n1_training_start",
                "steps": steps,
                "device": str(device),
                "noise_source": PROVENANCE["noise_source"],
                "hidden_source": runtime_prov["HIDDEN_SOURCE"],
                "freeze_backbone": runtime_prov["GEMNET_BACKBONE_FROZEN"],
                "gemnet_hash_before": gemnet_hash_before,
            }
        ),
        flush=True,
    )

    # Hidden tensor contract audit on first forward
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)))
    with (out / "training_trace.jsonl").open("w", buffering=1) as stream:
        for step in range(steps):
            t = noise.sample_t(1, device=device)
            noisy = noise.corrupt_fixed_sample(
                frac_coords_0=sample_d["pos"],
                lattice_0=sample_d["cell"],
                num_atoms=int(sample_d["N"]),
                t=t,
                generator=g,
            )
            opt.zero_grad(set_to_none=True)
            mode = "oracle_orbit" if step % 2 == 0 else "predicted_orbit"
            vals = model.loss(
                o2_target=o2_target,
                backbone_tree=backbone,
                z=sample_d["z"],
                frac_t=noisy.frac_coords_t,
                cell_t=noisy.lattice_t,
                t=noisy.t,
                role_z=sample_d["role_z"],
                role_edge_index=sample_d["role_edge_index"],
                role_bond_type=sample_d["role_bond_type"],
                oracle_bar_r=oracle_bar,
                atomic_numbers=sample_d["z"],
                orbit_mode=mode,
            )
            if step == 0:
                meta = dict(model._last_hidden_meta)
                print(json.dumps({"event": "hidden_tensor_contract", **meta}), flush=True)
                if model_cfg.hidden_source == "gemnet":
                    if meta.get("hidden_source") != "gemnet_node_embeddings":
                        raise RuntimeError(
                            f"N1 main experiment abort: actual hidden is {meta.get('hidden_source')}"
                        )
                    if meta.get("context_crystal_encoder_used"):
                        raise RuntimeError("context encoder used under gemnet mode")
                    if int(meta.get("num_atoms", -1)) != int(sample_d["N"]):
                        raise RuntimeError(
                            f"hidden N={meta.get('num_atoms')} != sample N={sample_d['N']}"
                        )
                    if int(sample_d["N"]) == 40 and int(meta.get("num_atoms", -1)) != 40:
                        raise RuntimeError("RHODIN01 expects N=40 atom hiddens")

            vals["loss"].backward()
            # GemNet grads must be absent / zero
            if model._gemnet_denoiser is not None:
                for p in model._gemnet_denoiser.parameters():
                    if p.grad is not None and float(p.grad.abs().sum()) != 0.0:
                        raise RuntimeError(
                            "FROZEN_BACKBONE_VIOLATION: GemNet received non-zero gradients"
                        )
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            row = {
                "step": step,
                "t": float(noisy.t.detach()),
                "t_fraction": float(noisy.t.detach()) / noise.T,
                "sigma_x": float(noisy.sigma_x),
                "sigma_l": float(noisy.sigma_l),
                "log_snr_x": float(noisy.log_snr_x),
                "log_snr_l": float(noisy.log_snr_l),
                "orbit_mode": mode,
                "orbit_loss": float(vals["orbit_loss"]),
                "singleton_nll": float(vals["singleton_nll"].detach()),
                "orbit_attachment_nll": float(vals["orbit_attachment_nll"].detach()),
                "total_assignment_loss": float(vals["total_assignment_loss"].detach()),
                "gradient_norm": float(gn),
            }
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == steps:
                print(json.dumps(row), flush=True)

    # Hash after training
    gemnet_hash_after = model.gemnet_parameter_hash()
    assign_hash_after = model.assignment_parameter_hash()
    if model_cfg.hidden_source == "gemnet":
        if gemnet_hash_before != gemnet_hash_after:
            raise RuntimeError(
                "FROZEN_BACKBONE_VIOLATION: gemnet_parameter_hash changed during training\n"
                f"  before={gemnet_hash_before}\n  after={gemnet_hash_after}"
            )
    if assign_hash_before == assign_hash_after:
        raise RuntimeError(
            "ASSIGNMENT_HEAD_NOT_UPDATED: assignment_parameter_hash unchanged after training"
        )

    hash_audit = {
        "gemnet_parameter_hash_before": gemnet_hash_before,
        "gemnet_parameter_hash_after": gemnet_hash_after,
        "assignment_parameter_hash_before": assign_hash_before,
        "assignment_parameter_hash_after": assign_hash_after,
        "gemnet_unchanged": gemnet_hash_before == gemnet_hash_after,
        "assignment_updated": assign_hash_before != assign_hash_after,
    }
    (out / "parameter_hash_audit.json").write_text(json.dumps(hash_audit, indent=2))
    print(json.dumps({"event": "parameter_hash_audit", **hash_audit}), flush=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {f.name: getattr(model_cfg, f.name) for f in fields(NoisyCopyAssignmentConfig)},
        "noise_provenance": PROVENANCE,
        "steps": steps,
        "n1_mode": "observational_noisy_copy_assignment",
        "mattergen_model_path": runtime_prov.get("MATTERGEN_MODEL_PATH"),
        "mattergen_load_epoch": runtime_prov.get("MATTERGEN_LOAD_EPOCH"),
        "mattergen_checkpoint": runtime_prov.get("MATTERGEN_CHECKPOINT"),
        "mattergen_checkpoint_sha256": runtime_prov.get("MATTERGEN_CHECKPOINT_SHA256"),
        "hidden_source": runtime_prov["HIDDEN_SOURCE"],
        "gemnet_frozen": runtime_prov["GEMNET_BACKBONE_FROZEN"],
        "noise_source": PROVENANCE["noise_source"],
        "geometry_feedback": False,
        "orbit_modes": ["oracle_orbit", "predicted_orbit"],
        "soft_c_semantics": SOFT_C_SEMANTICS,
        "parameter_hash_audit": hash_audit,
        "runtime_provenance": runtime_prov,
    }
    torch.save(ckpt, out / "best_checkpoint.pt")
    torch.save(ckpt, out / "final_checkpoint.pt")
    (out / "checkpoint_selection.json").write_text(
        json.dumps({"best": "best_checkpoint.pt", "final": "final_checkpoint.pt"}, indent=2)
    )
    print(json.dumps({"event": "n1_training_done", "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
