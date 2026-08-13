#!/usr/bin/env python3
"""Train J1 joint A+X+L diffusion (RHODIN01 MVP)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from torch_geometric.data import Batch  # noqa: F401 used via build_cg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mattergen.assignment.global_copy_assembly.orbit_membership import build_orbit_partition
from mattergen.assignment.joint_assignment_diffusion.ctmc import simulate_forward_ctmc
from mattergen.assignment.joint_assignment_diffusion.joint_model import JointAXLModel
from mattergen.assignment.joint_assignment_diffusion.losses import (
    event_conditioned_assignment_ce,
    event_conditioned_g_teacher_ce,
    geometry_step_loss,
)
from mattergen.assignment.soft_c_geometry_feedback_n2.geometry_loss import mattergen_geometry_loss
from mattergen.assignment.joint_assignment_diffusion.reverse_eval import (
    aggregate_event_bins,
    aggregate_g_teacher_bins,
    event_bin_name,
)
from mattergen.assignment.joint_assignment_diffusion.schedule import AsyncJumpSchedule
from mattergen.assignment.joint_assignment_diffusion.state import a_from_role_and_copy
from mattergen.assignment.joint_assignment_diffusion.symmetry import (
    apply_symmetry_to_geometry,
    apply_symmetry_to_state,
    sample_symmetry_augment,
)
from mattergen.assignment.noisy_copy_assignment.gemnet_loader import (
    build_mol_conditioning_from_sample,
    load_molecular_csp_gemnet,
)
from mattergen.assignment.noisy_copy_assignment.mattergen_noise_adapter import (
    MatterGenNativeNoiseAdapter,
    PROVENANCE,
)
from mattergen.common.data.chemgraph import ChemGraph


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def permute_mol_conditioning(extra: dict, atom_perm: torch.Tensor) -> dict:
    """Remap mol_* tensors after atom permutation (inverse map for edge indices)."""
    device = atom_perm.device
    n = int(atom_perm.numel())
    inv = torch.empty(n, dtype=torch.long, device=device)
    inv[atom_perm] = torch.arange(n, device=device)
    out = {}
    for k, v in extra.items():
        if k == "mol_copy_id":
            continue
        if not torch.is_tensor(v):
            out[k] = v
            continue
        v = v.to(device=device)
        if k == "mol_x" or k == "mol_atom_id" or (v.ndim >= 1 and v.shape[0] == n and k != "mol_bond_edge_index"):
            out[k] = v[atom_perm]
        elif k == "mol_bond_edge_index" and v.numel() > 0:
            out[k] = inv[v.long()]
        else:
            out[k] = v
    return out


def build_cg(sample: dict, frac, cell, extra_mol: dict | None = None):
    """Build ChemGraphBatch with molecular-CSP mol_* fields required by GemNet conditioner."""
    n = int(sample["N"])
    lat = cell if cell.ndim == 3 else cell.unsqueeze(0)
    device = frac.device
    kw = dict(
        atomic_numbers=sample["z"].long().to(device),
        pos=frac,
        cell=lat.to(device),
        num_atoms=torch.tensor([n], dtype=torch.long, device=device),
        num_nodes=n,
    )
    if extra_mol is None:
        # Build from role/copy graph if sample lacks native mol_*
        base = {
            "z": sample["z"],
            "role": sample.get("role"),
            "copy": sample.get("copy"),
            "role_edge_index": sample.get("role_edge_index"),
            "role_bond_type": sample.get("role_bond_type"),
        }
        for k in ("mol_x", "mol_bond_edge_index", "mol_bond_attr", "mol_atom_id"):
            if k in sample:
                base[k] = sample[k]
        extra_mol = build_mol_conditioning_from_sample(base)
    for k, v in extra_mol.items():
        if k == "mol_copy_id":
            continue
        if torch.is_tensor(v):
            kw[k] = v.to(device)
        else:
            kw[k] = v
    missing = [k for k in ("mol_x", "mol_bond_edge_index", "mol_bond_attr") if k not in kw]
    if missing:
        raise KeyError(f"ChemGraph missing mol conditioning fields: {missing}")
    return Batch.from_data_list([ChemGraph(**kw)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mattergen-model-path", type=str, default=None)
    p.add_argument("--mattergen-load-epoch", type=int, default=None)
    p.add_argument("--mattergen-checkpoint", type=str, default=None)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit("Refusing without --execute")

    cfg = yaml.safe_load(args.config.read_text())["joint_j1"]
    device = resolve_device(str(cfg.get("device", "auto")))
    out = Path(args.output_dir or cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    sample = torch.load(cfg["fixed_sample_path"], map_location="cpu", weights_only=False)
    if sample["id"] != cfg["fixed_sample_id"]:
        raise ValueError("sample id mismatch")
    orbits = json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    per_role = [v for _, v in sorted(orbits["role_orbits"].items(), key=lambda x: int(x[0]))]
    partition = build_orbit_partition(per_role)

    gem = cfg.get("gemnet") or {}
    bundle = load_molecular_csp_gemnet(
        model_path=args.mattergen_model_path or gem["model_path"],
        load_epoch=args.mattergen_load_epoch if args.mattergen_load_epoch is not None else gem.get("load_epoch", 294),
        checkpoint_path=args.mattergen_checkpoint or gem.get("checkpoint_path"),
        freeze=False,
        strict=True,
    )
    denoiser = bundle.denoiser.to(device)
    pl = bundle.lightning_module.to(device)
    # unfreeze denoiser for joint training
    for p in denoiser.parameters():
        p.requires_grad_(True)

    sch_cfg = cfg.get("schedule") or {}
    schedule = AsyncJumpSchedule.from_config(sch_cfg)
    g_ctx = str(cfg.get("g_copy_context_mode") or "orbit_slot_geometry")
    model = JointAXLModel(
        denoiser, num_orbits=partition.J, schedule=schedule, g_copy_context_mode=g_ctx
    ).to(device)
    sample_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    model.set_orbit_relations(partition, sample_d["role_edge_index"], sample_d["role_bond_type"])

    opt_cfg = cfg.get("optim") or {}
    opt = torch.optim.AdamW(
        [
            {"params": list(model.pretrained_parameters()), "lr": float(opt_cfg.get("lr_pretrained", 1e-5))},
            {"params": list(model.new_module_parameters()), "lr": float(opt_cfg.get("lr_new", 1e-4))},
        ],
        weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
    )
    noise = MatterGenNativeNoiseAdapter(limit_density=float(cfg.get("limit_density", 0.05)))
    loss_fn = pl.diffusion_module.loss_fn
    loss_w = cfg.get("loss") or {}
    steps = int(args.steps or cfg.get("steps", 1000))
    clip = float(opt_cfg.get("gradient_clip_norm", 1.0))
    log_every = int(cfg.get("log_every_steps", 20))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.get("seed", 17)))

    prov = {
        "J1_MODE": "joint_axl_ctmc_j1_1",
        "RATE_MODEL": "fixed_exit_beta_softmax",
        "ASSIGNMENT_LOSS": "event_conditioned_reverse_ce",
        "JOINT_TIME": "single_global_tau_or_t",
        "J1_2": "event_conditioned_assignment",
        "J1_3A": "improvement_weighted_g_teacher",
        "J1_3B2": "candidate_to_copy_pbc_geometry",
        "G_SUPERVISION": str((cfg.get("g_supervision") or "improvement_weighted")),
        "G_TEACHER_TEMPERATURE": float((cfg.get("g_teacher_temperature") or 0.02)),
        "G_COPY_CONTEXT_MODE": str(cfg.get("g_copy_context_mode") or "orbit_slot_geometry"),
        "R_WINDOW": list(schedule.r_window),
        "G_WINDOW": list(schedule.g_window),
        "KAPPA_R": schedule.kappa_r,
        "KAPPA_G": schedule.kappa_g,
        "R_LOCK": schedule.r_lock,
        "G_LOCK": schedule.g_lock,
        "EXPECTED_JUMPS": schedule.expected_jump_budget(),
        "MATTERGEN_LOAD_EPOCH": bundle.load_epoch,
        "MATTERGEN_CHECKPOINT_SHA": bundle.checkpoint_sha256,
        "NOISE_SOURCE": PROVENANCE["noise_source"],
        "LIE_SPLITTING": "A_first_same_t",
        "SAMPLE": sample["id"],
    }
    (out / "runtime_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(json.dumps({"event": "j1_start", **prov}), flush=True)

    clean_state0 = a_from_role_and_copy(
        role=sample_d["role"],
        copy=sample_d["copy"],
        partition=partition,
        atomic_numbers=sample_d["z"],
        role_z=sample_d["role_z"],
        K=int(sample_d["Z"]),
    )

    g_sup = str((cfg.get("g_supervision") or "improvement_weighted")).strip()
    if g_sup not in ("historical", "improvement_weighted"):
        raise ValueError(f"unknown g_supervision={g_sup}")
    g_temp = float(cfg.get("g_teacher_temperature") or 0.02)
    event_records: list[dict] = []
    with (out / "training_trace.jsonl").open("w", buffering=1) as stream:
        for step in range(steps):
            # symmetry augmentation
            aug = sample_symmetry_augment(
                atomic_numbers=sample_d["z"], K=int(sample_d["Z"]), generator=g
            )
            st0 = apply_symmetry_to_state(clean_state0, aug)
            geo = apply_symmetry_to_geometry(
                frac=sample_d["pos"],
                atomic_numbers=sample_d["z"],
                role=sample_d["role"],
                copy=sample_d["copy"],
                aug=aug,
            )
            # forward CTMC on A (uniform π prior, fixed exit β_R/β_G)
            traj = simulate_forward_ctmc(st0, schedule=schedule, t_start=0.0, t_end=1.0, generator=g)
            n_R_all = sum(1 for e in traj.events if e.kind == "R")
            n_G_all = sum(1 for e in traj.events if e.kind == "G")

            # Focus: R/G sample a real forward event; U is geometry-only t~U(0,1)
            focus_u = float(torch.rand((), generator=g).item())
            if focus_u < 0.34:
                t_focus = "R"
            elif focus_u < 0.68:
                t_focus = "G"
            else:
                t_focus = "U"

            picked = None
            if t_focus in ("R", "G"):
                pool = [e for e in traj.events if e.kind == t_focus]
                # Retry a few trajectories so R/G-focus almost always has a target
                retries = 0
                while not pool and retries < 4:
                    traj = simulate_forward_ctmc(st0, schedule=schedule, t_start=0.0, t_end=1.0, generator=g)
                    pool = [e for e in traj.events if e.kind == t_focus]
                    n_R_all = sum(1 for e in traj.events if e.kind == "R")
                    n_G_all = sum(1 for e in traj.events if e.kind == "G")
                    retries += 1
                if pool:
                    idx = int(torch.randint(0, len(pool), (), generator=g).item())
                    picked = pool[idx]

            if picked is not None:
                t_f = float(picked.time)
                state_t = traj.state_at(t_f)  # A_{τ+}
            else:
                t_focus = "U"
                t_ten = noise.sample_t(1, device=device)
                t_f = float(t_ten.reshape(-1)[0].item())
                state_t = traj.state_at(t_f)

            t = torch.tensor([t_f], device=device, dtype=torch.float32)
            noisy = noise.corrupt_fixed_sample(
                frac_coords_0=geo["pos"],
                lattice_0=sample_d["cell"],
                num_atoms=int(sample_d["N"]),
                t=t,
                generator=g,
            )

            samp_aug = dict(sample_d)
            samp_aug["z"] = geo["z"]
            samp_aug["pos"] = geo["pos"]
            if "role" in geo:
                samp_aug["role"] = geo["role"]
            if "copy" in geo:
                samp_aug["copy"] = geo["copy"]
            mol_extra = build_mol_conditioning_from_sample(
                {
                    "z": samp_aug["z"],
                    "role": samp_aug.get("role", sample_d["role"]),
                    "copy": samp_aug.get("copy", sample_d["copy"]),
                    "role_edge_index": sample_d["role_edge_index"],
                    "role_bond_type": sample_d["role_bond_type"],
                }
            )
            clean_cg = build_cg(samp_aug, geo["pos"], sample_d["cell"], extra_mol=mol_extra)
            noisy_cg = build_cg(samp_aug, noisy.frac_coords_t, noisy.lattice_t, extra_mol=mol_extra)

            opt.zero_grad(set_to_none=True)
            lam_r = float(loss_w.get("lambda_r", 1.0))
            lam_g = float(loss_w.get("lambda_g", 1.0))
            ce_r = torch.zeros((), device=device)
            ce_g = torch.zeros((), device=device)
            ev_diag: dict = {}
            if picked is not None:
                if picked.kind == "G" and g_sup == "improvement_weighted":
                    ev_diag = event_conditioned_g_teacher_ce(
                        model=model,
                        chemgraph_t=noisy_cg,
                        t=noisy.t,
                        state_after=state_t,
                        c0=st0.C(),
                        copy0=st0.copy_of(),
                        hist_i=picked.i,
                        hist_j=picked.j,
                        temperature=g_temp,
                    )
                else:
                    ev_diag = event_conditioned_assignment_ce(
                        model=model,
                        chemgraph_t=noisy_cg,
                        t=noisy.t,
                        state_after=state_t,
                        kind=picked.kind,
                        i=picked.i,
                        j=picked.j,
                    )
                L_geom, _gmet = mattergen_geometry_loss(
                    loss_fn=loss_fn,
                    corruption=noise.corruption,
                    clean_batch=clean_cg,
                    noisy_batch=noisy_cg,
                    score_model_output=ev_diag["chemgraph_scores"],
                    t=torch.as_tensor(noisy.t, dtype=torch.float32, device=device).reshape(-1),
                )
                if picked.kind == "R":
                    ce_r = ev_diag["CE"]
                else:
                    ce_g = ev_diag["CE"]
                rec = {
                    "step": step,
                    "kind": picked.kind,
                    "tau": t_f,
                    "bin": event_bin_name(picked.kind, t_f),
                    "g_supervision": g_sup if picked.kind == "G" else "historical",
                    "CE": float(ev_diag["CE"].detach()),
                    "uniform_CE": float(ev_diag.get("uniform_CE", ev_diag.get("uniform_CE_teacher", 0.0))),
                    "delta_CE": float(ev_diag.get("delta_CE", ev_diag.get("delta_CE_teacher", 0.0))),
                }
                # R historical diagnostics
                for k in ("target_probability", "target_rank", "top1", "top5", "entropy", "num_legal"):
                    if k in ev_diag:
                        rec[k] = ev_diag[k] if not torch.is_tensor(ev_diag[k]) else float(ev_diag[k].detach())
                # G teacher / policy diagnostics
                for k in (
                    "num_legal_G",
                    "num_beneficial_G",
                    "num_positive_teacher_support",
                    "teacher_entropy_G",
                    "teacher_effective_support_G",
                    "teacher_max_prob_G",
                    "best_delta_F1",
                    "mean_positive_delta_F1",
                    "historical_delta_F1",
                    "historical_inverse_rank_by_utility",
                    "historical_inverse_is_beneficial",
                    "historical_inverse_is_best",
                    "historical_inverse_teacher_mass",
                    "g_teacher_no_beneficial",
                    "CE_teacher",
                    "delta_CE_teacher",
                    "P_beneficial",
                    "P_best",
                    "top1_is_beneficial",
                    "top1_is_best",
                    "top1_delta_F1",
                    "top1_delta_ARI",
                    "expected_delta_F1",
                    "uniform_expected_delta_F1",
                    "g_copy_context_mode",
                    "slot_embedding_norm_mean",
                    "slot_embedding_norm_std",
                    "slot_pair_feature_norm",
                    "slot_orbit_pairwise_var",
                    "candidate_copy_geom_norm_mean",
                    "candidate_copy_geom_norm_std",
                    "candidate_copy_relation_norm_mean",
                    "candidate_copy_relation_norm_std",
                    "candidate_copy_relation_variance_across_copies",
                    "candidate_copy_relation_variance_across_candidates",
                    "current_vs_cross_relation_distance",
                    "g_logit_std_across_legal_moves",
                ):
                    if k in ev_diag and ev_diag[k] is not None:
                        v = ev_diag[k]
                        if torch.is_tensor(v):
                            rec[k] = float(v.detach())
                        elif isinstance(v, str):
                            rec[k] = v
                        else:
                            rec[k] = float(v)
                event_records.append(rec)
            else:
                geom = geometry_step_loss(
                    model=model,
                    loss_fn=loss_fn,
                    corruption=noise.corruption,
                    clean_cg=clean_cg,
                    noisy_cg=noisy_cg,
                    t=noisy.t,
                    state_at_t=state_t,
                )
                L_geom = geom["L_geom"]
            total = L_geom + lam_r * ce_r + lam_g * ce_g
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            row = {
                "step": step,
                "global_t": t_f,
                "t_focus": t_focus,
                "g_supervision": g_sup,
                "g_copy_context_mode": g_ctx,
                "has_event_target": picked is not None,
                "event_kind": None if picked is None else picked.kind,
                "geometry_loss": float(L_geom.detach()),
                "CE_R": float(ce_r.detach()),
                "CE_G": float(ce_g.detach()),
                "L_R": float(ce_r.detach()),
                "L_G": float(ce_g.detach()),
                "uniform_CE_R": float(ev_diag.get("uniform_CE", 0.0)) if picked is not None and picked.kind == "R" else 0.0,
                "uniform_CE_G": float(ev_diag.get("uniform_CE_teacher", ev_diag.get("uniform_CE", 0.0))) if picked is not None and picked.kind == "G" else 0.0,
                "delta_CE_R": float(ev_diag.get("delta_CE", 0.0)) if picked is not None and picked.kind == "R" else 0.0,
                "delta_CE_G": float(ev_diag.get("delta_CE_teacher", ev_diag.get("delta_CE", 0.0))) if picked is not None and picked.kind == "G" else 0.0,
                "P_beneficial": float(ev_diag.get("P_beneficial", 0.0)) if ev_diag else 0.0,
                "P_best": float(ev_diag.get("P_best", 0.0)) if ev_diag else 0.0,
                "expected_delta_F1": float(ev_diag.get("expected_delta_F1", 0.0)) if ev_diag else 0.0,
                "uniform_expected_delta_F1": float(ev_diag.get("uniform_expected_delta_F1", 0.0)) if ev_diag else 0.0,
                "top1_is_beneficial": float(ev_diag.get("top1_is_beneficial", 0.0)) if ev_diag else 0.0,
                "g_teacher_no_beneficial": float(ev_diag.get("g_teacher_no_beneficial", 0.0)) if ev_diag else 0.0,
                "target_probability": float(ev_diag.get("target_probability", 0.0)) if ev_diag else 0.0,
                "target_rank": int(ev_diag.get("target_rank", ev_diag.get("historical_inverse_rank_by_utility", -1)) or -1) if ev_diag else -1,
                "top1": float(ev_diag.get("top1", ev_diag.get("top1_is_best", 0.0))) if ev_diag else 0.0,
                "top5": float(ev_diag.get("top5", 0.0)) if ev_diag else 0.0,
                "entropy": float(ev_diag.get("entropy", ev_diag.get("teacher_entropy_G", 0.0))) if ev_diag else 0.0,
                "num_legal": int(ev_diag.get("num_legal", ev_diag.get("num_legal_G", 0))) if ev_diag else 0,
                "total_loss": float(total.detach()),
                "num_fwd_events": len(traj.events),
                "n_R_fwd": n_R_all,
                "n_G_fwd": n_G_all,
                "beta_R_t": float(schedule.beta_r(t_f).item()),
                "beta_G_t": float(schedule.beta_g(t_f).item()),
                "state_t_legal": state_t.validate()["legal"],
            }
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == steps:
                print(json.dumps(row), flush=True)

    # Save weights first so a diagnostics dump cannot lose a finished run.
    torch.save(
        {
            "joint_state_dict": model.state_dict(),
            "schedule": sch_cfg,
            "provenance": prov,
            "steps": steps,
        },
        out / "final_checkpoint.pt",
    )
    torch.save(
        {"joint_state_dict": model.state_dict(), "provenance": prov},
        out / "best_checkpoint.pt",
    )
    try:
        (out / "event_bin_summary.json").write_text(json.dumps(aggregate_event_bins(event_records), indent=2))
        (out / "g_teacher_bin_summary.json").write_text(json.dumps(aggregate_g_teacher_bins(event_records), indent=2))
        with (out / "event_bin_trace.jsonl").open("w") as ef:
            for rec in event_records:
                ef.write(json.dumps(rec) + "\n")
    except Exception as exc:
        print(json.dumps({"event": "j1_diag_dump_failed", "error": str(exc)}), flush=True)
    print(json.dumps({"event": "j1_train_done", "output": str(out), "n_event_targets": len(event_records)}), flush=True)


if __name__ == "__main__":
    main()
