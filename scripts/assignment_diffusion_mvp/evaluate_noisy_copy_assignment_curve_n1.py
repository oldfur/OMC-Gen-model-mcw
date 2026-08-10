#!/usr/bin/env python3
"""N1 recovery curves vs MatterGen-native noise level (frozen assignment model).

Uses the same frozen epoch294 GemNet hidden source as training for both
oracle_orbit and predicted_orbit curves.
"""
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
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import load_molecular_csp_gemnet
from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import MatterGenNativeNoiseAdapter
from mattergen.assignment.noisy_copy_assignment.metrics import evaluate_n1_once
from mattergen.assignment.noisy_copy_assignment.module import NoisyCopyAssignmentConfig, NoisyCopyAssignmentN1
from mattergen.assignment.noisy_copy_assignment.soft_c import SOFT_C_SEMANTICS


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def crossing(curve: list[dict], key: str, thr: float, gap_key: str | None = None, gap_thr: float | None = None):
    for row in curve:
        if row.get(key) is None:
            continue
        if float(row[key]) > thr:
            if gap_key is not None and gap_thr is not None:
                g = row.get(gap_key)
                if g is None or float(g) <= gap_thr:
                    continue
            return {"t_fraction": row["t_fraction"], "log_snr_x": row.get("log_snr_x"), key: row[key]}
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mattergen-model-path", type=str, default=None)
    parser.add_argument("--mattergen-load-epoch", type=int, default=None)
    parser.add_argument("--mattergen-checkpoint", type=str, default=None)
    parser.add_argument("--hidden-source", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing evaluation without --execute")

    cfg = yaml.safe_load(args.config.read_text())["assignment_n1"]
    if args.hidden_source is not None:
        cfg["hidden_source"] = args.hidden_source
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)
    art = [json.loads(l) for l in Path(cfg["predicted_role_artifact_path"]).read_text().splitlines() if l.strip()][-1]
    role_assignment = torch.tensor(art["role_assignment"], dtype=torch.long)
    oracle_bar = collapse_roles_to_orbit_membership(sample["role"].long(), partition)
    backbone = prepare_backbone(partition, sample["role_edge_index"], sample["role_bond_type"])
    anchor = backbone.singleton_roles[backbone.tree.root]
    o2_target = build_orbit_aware_target(
        role_assignment, sample["copy"], partition=partition, K=int(sample["Z"]), anchor_role=anchor
    )

    loss_cfg = cfg.get("loss") or {}
    flat = {**cfg, **loss_cfg}
    for drop in ("gemnet", "context_encoder", "evaluation", "structured_decoder", "orbit_modes", "loss"):
        flat.pop(drop, None)
    allowed = {f.name for f in fields(NoisyCopyAssignmentConfig)}
    model_cfg = NoisyCopyAssignmentConfig(**{k: v for k, v in flat.items() if k in allowed})
    # Prefer checkpoint-stored hidden_source
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" in ckpt and isinstance(ckpt["config"], dict):
        hs = ckpt["config"].get("hidden_source")
        if hs:
            model_cfg.hidden_source = hs
    model = NoisyCopyAssignmentN1(model_cfg, partition)

    device = resolve_device(str(cfg.get("device", "auto")))
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    o2_target = _move_o2_target(o2_target, device)
    oracle_bar = oracle_bar.to(device)
    model = model.to(device)

    # Inject same GemNet source as training
    if model_cfg.hidden_source == "gemnet":
        gem_cfg = dict(cfg.get("gemnet") or {})
        model_path = (
            args.mattergen_model_path
            or ckpt.get("mattergen_model_path")
            or gem_cfg.get("model_path")
        )
        load_epoch = args.mattergen_load_epoch
        if load_epoch is None:
            load_epoch = ckpt.get("mattergen_load_epoch") or gem_cfg.get("load_epoch", 294)
        ckpt_path = (
            args.mattergen_checkpoint
            or ckpt.get("mattergen_checkpoint")
            or gem_cfg.get("checkpoint_path")
        )
        if not model_path:
            raise RuntimeError("eval hidden_source=gemnet requires MatterGen model_path")
        bundle = load_molecular_csp_gemnet(
            model_path=model_path,
            load_epoch=load_epoch,
            checkpoint_path=ckpt_path,
            freeze=True,
            strict=True,
        )
        model.set_gemnet_denoiser(bundle.denoiser.to(device), freeze=True)
        model.prepare_mol_conditioning_from_sample(sample)
        if model.gemnet_proj is not None:
            model.gemnet_proj.to(device)

    # Load assignment weights (strict=False allows missing gemnet keys in state_dict)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # GemNet weights come from MatterGen loader, not N1 ckpt — filter expected misses
    bad_missing = [k for k in missing if not k.startswith("_gemnet") and "denoiser" not in k]
    if unexpected:
        # Allow only non-critical
        print(json.dumps({"event": "state_dict_unexpected", "keys": list(unexpected)[:20]}), flush=True)
    if bad_missing:
        print(json.dumps({"event": "state_dict_missing", "keys": bad_missing[:40]}), flush=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Re-freeze gemnet after load
    if model_cfg.hidden_source == "gemnet":
        model.freeze_backbone()

    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    eval_cfg = cfg.get("evaluation") or {}
    fracs = eval_cfg.get("timestep_fractions") or [0.0, 0.5, 1.0]
    seeds = eval_cfg.get("seeds") or [0, 1, 2]
    thr = eval_cfg.get("thresholds") or {}

    for orbit_mode, out_name in (
        ("oracle_orbit", "oracle_orbit_curve.jsonl"),
        ("predicted_orbit", "predicted_orbit_curve.jsonl"),
    ):
        rows = []
        for tf in fracs:
            for seed in seeds:
                g = torch.Generator(device="cpu")
                g.manual_seed(int(seed) + int(1000 * float(tf)))
                noisy = noise.corrupt_at_fraction(
                    frac_coords_0=sample["pos"],
                    lattice_0=sample["cell"],
                    num_atoms=int(sample["N"]),
                    t_fraction=float(tf),
                    generator=g,
                )
                target = o2_target
                if orbit_mode == "predicted_orbit":
                    h = model.extract_atom_hidden(
                        z=sample["z"],
                        frac=noisy.frac_coords_t,
                        cell=noisy.lattice_t,
                        t=noisy.t,
                        atomic_numbers=sample["z"],
                    )
                    bar, logits, labels = model.resolve_bar_r(
                        h=h, K=int(sample["Z"]), oracle_bar_r=oracle_bar, mode="predicted_orbit"
                    )
                    del bar, logits
                    synth = torch.tensor(
                        [partition.orbits[int(o)][0] for o in labels.tolist()],
                        dtype=torch.long,
                        device=device,
                    )
                    target = build_orbit_aware_target(
                        synth.cpu(), sample["copy"].cpu(), partition=partition, K=int(sample["Z"]), anchor_role=anchor
                    )
                    target = _move_o2_target(target, device)

                out_dec = model.map_decode(
                    o2_target=target,
                    backbone_tree=backbone,
                    z=sample["z"],
                    frac_t=noisy.frac_coords_t,
                    cell_t=noisy.lattice_t,
                    t=noisy.t,
                    role_z=sample["role_z"],
                    role_edge_index=sample["role_edge_index"],
                    role_bond_type=sample["role_bond_type"],
                    oracle_bar_r=oracle_bar,
                    atomic_numbers=sample["z"],
                    orbit_mode=orbit_mode,
                )
                metrics = evaluate_n1_once(
                    output=out_dec, sample=sample, oracle_bar=oracle_bar, partition=partition
                )
                row = {
                    "t": float(noisy.t.detach()),
                    "t_fraction": float(tf),
                    "sigma_x": float(noisy.sigma_x),
                    "sigma_l": float(noisy.sigma_l),
                    "log_snr_x": float(noisy.log_snr_x),
                    "log_snr_l": float(noisy.log_snr_l),
                    "seed": int(seed),
                    "orbit_mode": orbit_mode,
                    "noise_source": "mattergen_native_forward_process",
                    "hidden_source": model_cfg.hidden_source,
                    "soft_c_semantics": SOFT_C_SEMANTICS,
                    **{
                        k: (float(v) if isinstance(v, (int, float)) or hasattr(v, "item") else v)
                        for k, v in metrics.items()
                        if not isinstance(v, dict)
                    },
                    "high_noise_leakage_suspected": bool(
                        float(tf) >= 0.9
                        and metrics.get("exact_C")
                        and (
                            metrics.get("structured_entropy") is not None
                            and float(metrics.get("structured_entropy") or 1) < 0.1
                        )
                    ),
                }
                row["exact_C"] = bool(metrics.get("exact_C"))
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "event": "curve_point",
                            **{
                                k: row[k]
                                for k in ("t_fraction", "seed", "orbit_mode", "exact_C", "copy_pair_f1")
                                if k in row
                            },
                        }
                    ),
                    flush=True,
                )

        with (out / out_name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        from collections import defaultdict

        by_t = defaultdict(list)
        for r in rows:
            by_t[r["t_fraction"]].append(r)
        agg = []
        for tf in sorted(by_t):
            chunk = by_t[tf]

            def mean_key(k, chunk=chunk):
                vals = [float(c[k]) for c in chunk if c.get(k) is not None and c[k] == c[k]]
                return sum(vals) / max(1, len(vals)) if vals else None

            agg.append(
                {
                    "t_fraction": tf,
                    "exact_C_rate": sum(1 for c in chunk if c.get("exact_C")) / len(chunk),
                    "copy_pair_f1": mean_key("copy_pair_f1"),
                    "ARI": mean_key("ARI"),
                    "same_copy_pair_AUC": mean_key("same_copy_pair_AUC"),
                    "orbit_atom_accuracy": mean_key("orbit_atom_accuracy"),
                    "orbit_attachment_MAP_gap": mean_key("orbit_attachment_MAP_gap"),
                    "log_snr_x": mean_key("log_snr_x"),
                    "sigma_x": mean_key("sigma_x"),
                }
            )
        thr_out = {
            "soft_c_semantics": SOFT_C_SEMANTICS,
            "hidden_source": model_cfg.hidden_source,
            "first_soft_signal_auc": crossing(agg, "same_copy_pair_AUC", float(thr.get("soft_auc", 0.75))),
            "useful_grouping_f1": crossing(agg, "copy_pair_f1", float(thr.get("useful_f1", 0.8))),
            "reliable_map": crossing(
                agg,
                "copy_pair_f1",
                float(thr.get("reliable_f1", 0.95)),
                gap_key="orbit_attachment_MAP_gap",
                gap_thr=float(thr.get("map_gap_delta", 1.0)),
            ),
            "exact_recovery_region": next(
                (
                    {"t_fraction": a["t_fraction"], "exact_C_rate": a["exact_C_rate"]}
                    for a in agg
                    if a["exact_C_rate"] is not None and a["exact_C_rate"] >= float(thr.get("exact_c_rate", 0.9))
                ),
                None,
            ),
            "aggregated": agg,
        }
        (out / f"{orbit_mode}_summary.json").write_text(json.dumps(thr_out, indent=2))

    print(json.dumps({"event": "curve_done", "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
