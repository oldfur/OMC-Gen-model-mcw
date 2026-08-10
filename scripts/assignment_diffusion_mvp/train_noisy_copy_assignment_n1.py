#!/usr/bin/env python3
"""Train N1 observational noisy copy assignment (freeze GemNet/backbone by default)."""
from __future__ import annotations

import argparse
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
from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
    PROVENANCE,
)
from mattergen.assignment.noisy_copy_assignment.module import (
    NoisyCopyAssignmentConfig,
    NoisyCopyAssignmentN1,
)


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(req)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return d


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["assignment_n1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to train N1 without --execute")

    cfg = load_cfg(args.config)
    if not cfg.get("enabled", True):
        raise SystemExit("assignment_n1.enabled is false")
    if cfg.get("geometry_feedback"):
        raise RuntimeError("N1 forbids geometry_feedback")
    root = ROOT_DIR
    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    if sample["id"] != cfg["fixed_sample_id"]:
        raise ValueError("sample id mismatch")
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)

    # Geometry hard-R for building O2 targets (orbit collapse from hard labels)
    art_path = Path(cfg["predicted_role_artifact_path"])
    art = [json.loads(l) for l in art_path.read_text().splitlines() if l.strip()][-1]
    if art.get("role_source") != "geometry_only_hard_r":
        raise ValueError("expected geometry_only_hard_r artifact")
    role_assignment = torch.tensor(art["role_assignment"], dtype=torch.long)
    oracle_bar = collapse_roles_to_orbit_membership(sample["role"].long(), partition)
    # structured targets from hard-R (may be gauge-swapped); use sample copy for supervision only
    backbone = prepare_backbone(partition, sample["role_edge_index"], sample["role_bond_type"])
    anchor = backbone.singleton_roles[backbone.tree.root]
    o2_target = build_orbit_aware_target(
        role_assignment, sample["copy"], partition=partition, K=int(sample["Z"]), anchor_role=anchor
    )

    allowed = {f.name for f in fields(NoisyCopyAssignmentConfig)}
    loss_cfg = cfg.get("loss") or {}
    flat = {**cfg, **{k: loss_cfg[k] for k in loss_cfg}}
    model_cfg = NoisyCopyAssignmentConfig(**{k: v for k, v in flat.items() if k in allowed})
    model = NoisyCopyAssignmentN1(model_cfg, partition)
    if model_cfg.freeze_gemnet_backbone:
        model.freeze_backbone()

    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    device = resolve_device(str(cfg.get("device", "auto")))
    sample_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    from mattergen.assignment.global_copy_assembly.orbit_targets import (
        OrbitAwareAssemblyTarget,
        OrbitAttachmentTarget,
    )
    from mattergen.assignment.global_copy_assembly.targets import AssemblyTarget

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

    o2_target = move_o2_target(o2_target, device)
    oracle_bar = oracle_bar.to(device)
    model = model.to(device)

    params = list(model.trainable_assignment_parameters())
    if not params:
        raise RuntimeError("no trainable assignment parameters")
    # ensure backbone frozen
    for p in model.backbone.parameters():
        if p.requires_grad:
            raise RuntimeError("backbone not frozen under default N1 config")

    opt = torch.optim.AdamW(
        params,
        lr=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    steps = int(args.steps or cfg.get("steps", 2000))
    clip = float(cfg.get("gradient_clip_norm", 1.0))
    log_every = int(cfg.get("log_every_steps", 20))
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

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
    (out / "config_audit.json").write_text(
        json.dumps(
            {
                "status": "NOISY_COPY_ASSIGNMENT_N1",
                "geometry_feedback": False,
                "freeze_gemnet_backbone": True,
                "use_copy_id_as_input": False,
                "use_oracle_C_as_input": False,
                "noise_source": "mattergen_native",
                "sample": sample["id"],
                "device": str(device),
                "trainable_param_count": sum(p.numel() for p in params),
            },
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                "event": "n1_training_start",
                "steps": steps,
                "device": str(device),
                "noise_source": PROVENANCE["noise_source"],
                "freeze_backbone": True,
            }
        ),
        flush=True,
    )

    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)))
    with (out / "training_trace.jsonl").open("w", buffering=1) as stream:
        for step in range(steps):
            # sample t like MatterGen UniformTimestepSampler
            t = noise.sample_t(1, device=device)
            noisy = noise.corrupt_fixed_sample(
                frac_coords_0=sample_d["pos"],
                lattice_0=sample_d["cell"],
                num_atoms=int(sample_d["N"]),
                t=t,
                generator=g,
            )
            opt.zero_grad(set_to_none=True)
            # train primarily oracle_orbit path; alternate predicted_orbit CE via loss
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
            vals["loss"].backward()
            # backbone grads must be None/zero
            for p in model.backbone.parameters():
                if p.grad is not None and float(p.grad.abs().sum()) != 0.0:
                    raise RuntimeError("backbone received non-zero gradients")
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

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {f.name: getattr(model_cfg, f.name) for f in fields(NoisyCopyAssignmentConfig)},
        "noise_provenance": PROVENANCE,
        "steps": steps,
    }
    torch.save(ckpt, out / "best_checkpoint.pt")
    torch.save(ckpt, out / "final_checkpoint.pt")
    (out / "checkpoint_selection.json").write_text(
        json.dumps({"best": "best_checkpoint.pt", "final": "final_checkpoint.pt"}, indent=2)
    )


if __name__ == "__main__":
    main()
