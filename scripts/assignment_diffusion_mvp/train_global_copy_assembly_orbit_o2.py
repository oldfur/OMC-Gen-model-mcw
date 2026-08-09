#!/usr/bin/env python3
"""Train orbit-aware O2 global copy assembly (clean geometry + geometry hard-R)."""
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

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition, collapse_roles_to_orbit_membership
from mattergen.assignment.global_copy_assembly.orbit_module import (
    OrbitAwareAssemblyConfig,
    OrbitAwareCopyAssembly,
    prepare_backbone,
)
from mattergen.assignment.global_copy_assembly.orbit_targets import build_orbit_aware_target


def _resolve(cfg: dict, key: str, fallback: Path) -> Path:
    value = cfg.get(key)
    return Path(value) if value else fallback


def load_setup(config_path: Path):
    raw = yaml.safe_load(config_path.read_text())
    cfg = raw["orbit_copy_assembly"]
    if not cfg.get("enabled", False):
        raise ValueError("orbit_copy_assembly.enabled must be true")
    root = ROOT_DIR
    sample_path = _resolve(cfg, "fixed_sample_path", root / "outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt")
    orbits_path = _resolve(cfg, "automorphism_orbits_path", root / "outputs/assignment_diffusion_mvp/role_automorphism_audit/role_orbits.json")
    artifact_path = _resolve(
        cfg,
        "predicted_role_artifact_path",
        root / "outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/geometry_only_hard_r.jsonl",
    )
    sample = torch.load(sample_path, map_location="cpu", weights_only=False)
    if sample["id"] != cfg["fixed_sample_id"] or sample["split"] != cfg["split"]:
        raise ValueError("fixed sample identity/split mismatch")
    orbit_json = json.loads(orbits_path.read_text())
    per_role = [value for _, value in sorted(orbit_json["role_orbits"].items(), key=lambda item: int(item[0]))]
    partition = build_orbit_partition(per_role)

    if cfg.get("use_copy_id_as_input") or cfg.get("use_oracle_copy_relation"):
        raise RuntimeError("O2 forbids copy supervision as model input")
    if cfg.get("use_oracle_role_assignment"):
        raise RuntimeError("O2 geometry path requires use_oracle_role_assignment=false")

    # Role source: geometry hard-R artifact (canonical-shaped). Optional oracle-orbit ablation.
    ablation = cfg.get("ablation") or {}
    if ablation.get("C_oracle_orbit_membership"):
        role_assignment = sample["role"].long()
        role_source_note = "oracle_r0_collapsed_to_orbits"
    else:
        if not artifact_path.exists():
            raise FileNotFoundError(f"predicted-role artifact missing: {artifact_path}")
        lines = [json.loads(line) for line in artifact_path.read_text().splitlines() if line.strip()]
        artifact = lines[-1]
        if "role_assignment" not in artifact:
            raise ValueError("artifact missing role_assignment")
        if artifact.get("role_source") != "geometry_only_hard_r":
            raise ValueError(f"unexpected role_source {artifact.get('role_source')!r}")
        role_assignment = torch.tensor(artifact["role_assignment"], dtype=torch.long)
        role_source_note = "geometry_only_hard_r"
        if role_assignment.numel() != int(sample["N"]):
            raise ValueError("artifact N mismatch")

    # R_effective for orbit membership is the artifact (or oracle under ablation C).
    # Never canonicalize artifact toward R0 before collapse.
    allowed = {f.name for f in fields(OrbitAwareAssemblyConfig)}
    model_cfg = OrbitAwareAssemblyConfig(**{k: v for k, v in cfg.items() if k in allowed})
    model = OrbitAwareCopyAssembly(model_cfg)
    backbone = prepare_backbone(
        partition,
        sample["role_edge_index"],
        sample["role_bond_type"],
        anchor_role=model_cfg.anchor_role,
    )
    # tree.root is a *local* singleton index; map back to molecular role id.
    anchor_role = backbone.singleton_roles[backbone.tree.root]
    o2_target = build_orbit_aware_target(
        role_assignment,
        sample["copy"],
        partition=partition,
        K=int(sample["Z"]),
        anchor_role=anchor_role,
    )
    # Prove bar_r comes from role_assignment, not a rewritten R0 (unless ablation C).
    bar_check = collapse_roles_to_orbit_membership(role_assignment, partition)
    if not torch.equal(bar_check, o2_target.bar_r):
        raise RuntimeError("orbit membership mismatch vs role_assignment")
    meta = {
        "role_source": role_source_note,
        "partition_orbits": [list(o) for o in partition.orbits],
        "singleton_roles": list(o2_target.singleton_roles),
        "non_singleton_orbits": [list(partition.orbits[j]) for j in partition.non_singleton_orbit_indices()],
        "canonicalization_applied": False,
        "orbit_collapse": True,  # membership collapse only; decoder still canonical-shaped
        "R_shape_labels": [int(sample["N"])],
        "bar_R_shape": list(o2_target.bar_r.shape),
    }
    return cfg, sample, o2_target, backbone, model, meta


def resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def move_o2_target(o2_target, device: torch.device):
    from mattergen.assignment.global_copy_assembly.orbit_targets import OrbitAwareAssemblyTarget, OrbitAttachmentTarget
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing to train without --execute")

    cfg, sample, o2_target, backbone, model, meta = load_setup(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    device = resolve_device(str(cfg.get("device", "auto")))
    sample_dev = {
        k: (v.to(device) if torch.is_tensor(v) and k != "copy" else v)
        for k, v in sample.items()
        if k != "copy"
    }
    o2_target = move_o2_target(o2_target, device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    grad_clip = float(cfg.get("gradient_clip_norm", 1.0))
    log_every = int(cfg.get("log_every_steps", 10))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_audit.json").write_text(
        json.dumps(
            {
                "status": "CLEAN_GEOMETRY_ORBIT_AWARE_O2",
                "meta": meta,
                "device": str(device),
                "singleton_tree_weight": model.config.singleton_tree_weight,
                "orbit_attachment_weight": model.config.orbit_attachment_weight,
                "pair_aux_weight": model.config.pair_aux_weight,
                "gauge_marginalization": model.config.gauge_marginalization,
                "exact_attachment_solver": model.config.exact_attachment_solver,
                "use_copy_id_as_input": False,
                "anchor_role": backbone.singleton_roles[backbone.tree.root],
                "tree_edges_roles": backbone.tree_edges_roles,
                "virtual_tree_edges": backbone.virtual_tree_edges,
            },
            indent=2,
        )
    )
    (output_dir / "permutation_convention.json").write_text(
        json.dumps(
            {
                "singleton": "P_r[q]=k on singleton roles only; anchor identity gauge",
                "orbit_attachment": "unordered pairs of orbit atoms per copy; no canonical 1/2 identity",
                "final": "G[N,K] merge singleton rows + orbit rows; C=GG^T",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "event": "o2_training_start",
                "device": str(device),
                "steps": args.steps,
                "role_source": meta["role_source"],
                "bar_R_shape": meta["bar_R_shape"],
            }
        ),
        flush=True,
    )
    with (output_dir / "training_trace.jsonl").open("w", buffering=1) as stream:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            values = model.loss(
                o2_target=o2_target,
                backbone=backbone,
                z=sample_dev["z"],
                frac=sample_dev["pos"],
                cell=sample_dev["cell"],
                role_z=sample_dev["role_z"],
                role_edge_index=sample_dev["role_edge_index"],
                role_bond_type=sample_dev["role_bond_type"],
            )
            values["loss"].backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("O2 gradient non-finite")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("O2 grad norm non-finite")
            optimizer.step()
            row = {"step": step, **{k: float(v.detach()) for k, v in values.items()}, "gradient_norm": float(grad_norm)}
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == args.steps:
                print(json.dumps(row), flush=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "step": args.steps,
        "monitor": "val/projected_bond_f1",
        "tie_break": "val/copy_pair_f1",
        "config": {f.name: getattr(model.config, f.name) for f in fields(OrbitAwareAssemblyConfig)},
        "meta": meta,
    }
    torch.save(ckpt, output_dir / "best_checkpoint.pt")
    torch.save(ckpt, output_dir / "final_checkpoint.pt")
    (output_dir / "checkpoint_selection.json").write_text(
        json.dumps(
            {
                "monitor": "val/projected_bond_f1",
                "tie_break": "val/copy_pair_f1",
                "best_checkpoint": "best_checkpoint.pt",
                "final_checkpoint": "final_checkpoint.pt",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
