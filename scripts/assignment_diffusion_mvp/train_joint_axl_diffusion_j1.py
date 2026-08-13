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
from mattergen.assignment.joint_assignment_diffusion.losses import joint_training_step_losses
from mattergen.assignment.joint_assignment_diffusion.schedule import (
    AsyncJumpSchedule,
    next_reverse_grid_s,
)
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
    model = JointAXLModel(denoiser, num_orbits=partition.J, schedule=schedule).to(device)
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
        "ASSIGNMENT_LOSS": "reverse_categorical_nll_event_mean",
        "JOINT_TIME": "single_global_t",
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

            # ---- single global time t (joint state S_t = (A_t, X_t, L_t)) ----
            # Optional focus sampling still yields ONE t, not independent t_R/t_G/t_X.
            focus_u = float(torch.rand((), generator=g).item())
            if focus_u < 0.34:
                t_focus = "R"
                t_f = schedule.sample_t_proportional_to_beta(kind="R", generator=g)
            elif focus_u < 0.68:
                t_focus = "G"
                t_f = schedule.sample_t_proportional_to_beta(kind="G", generator=g)
            else:
                t_focus = "U"
                t_ten = noise.sample_t(1, device=device)
                t_f = float(t_ten.reshape(-1)[0].item())
            t = torch.tensor([t_f], device=device, dtype=torch.float32)
            # Reverse endpoint aligned with A-first Lie sampler grid (not s~U(0,t))
            eval_grid = (cfg.get("evaluation") or {}).get("timesteps") or None
            t_s = next_reverse_grid_s(t_f, eval_grid)
            state_t = traj.state_at(t_f)
            state_s = traj.state_at(t_s)
            H_R_to_t = schedule.integrated_beta(0.0, t_f, kind="R")
            H_G_to_t = schedule.integrated_beta(0.0, t_f, kind="G")
            H_R_seg = schedule.integrated_beta(t_s, t_f, kind="R")
            H_G_seg = schedule.integrated_beta(t_s, t_f, kind="G")

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
            losses = joint_training_step_losses(
                model=model,
                loss_fn=loss_fn,
                corruption=noise.corruption,
                clean_cg=clean_cg,
                noisy_cg=noisy_cg,
                t=noisy.t,
                t_s=t_s,
                state_at_t=state_t,
                traj=traj,
                lambda_r=float(loss_w.get("lambda_r", 1.0)),
                lambda_g=float(loss_w.get("lambda_g", 1.0)),
                h_r_segment=H_R_seg,
                h_g_segment=H_G_seg,
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            n_R_all = sum(1 for e in traj.events if e.kind == "R")
            n_G_all = sum(1 for e in traj.events if e.kind == "G")
            n_R_seg = sum(1 for e in traj.events_on_segment(t_s, t_f) if e.kind == "R")
            n_G_seg = sum(1 for e in traj.events_on_segment(t_s, t_f) if e.kind == "G")
            H_R_full = schedule.integrated_beta(0.0, 1.0, kind="R")
            H_G_full = schedule.integrated_beta(0.0, 1.0, kind="G")
            row = {
                "step": step,
                "global_t": t_f,
                "global_s": t_s,
                "t_focus": t_focus,
                "geometry_loss": float(losses["L_geom"].detach()),
                "CE_R": float(losses["CE_R"].detach()),
                "CE_G": float(losses["CE_G"].detach()),
                "L_R": float(losses["L_R"].detach()),
                "L_G": float(losses["L_G"].detach()),
                "uniform_CE_R": float(losses["uniform_CE_R"].detach()),
                "uniform_CE_G": float(losses["uniform_CE_G"].detach()),
                "delta_CE_R": float(losses["delta_CE_R"].detach()),
                "delta_CE_G": float(losses["delta_CE_G"].detach()),
                "total_loss": float(losses["loss"].detach()),
                "num_fwd_events": len(traj.events),
                "n_R_fwd": n_R_all,
                "n_G_fwd": n_G_all,
                "n_R_events": float(losses["n_R_events"].detach()),
                "n_G_events": float(losses["n_G_events"].detach()),
                "n_R_events_segment": n_R_seg,
                "n_G_events_segment": n_G_seg,
                "num_legal_R": float(losses["num_legal_R"].detach()),
                "num_legal_G": float(losses["num_legal_G"].detach()),
                "H_R_full": H_R_full,
                "H_G_full": H_G_full,
                "H_R_segment": H_R_seg,
                "H_G_segment": H_G_seg,
                "expected_R_segment": H_R_seg,
                "expected_G_segment": H_G_seg,
                "H_R_to_t": H_R_to_t,
                "H_G_to_t": H_G_to_t,
                "beta_R_t": float(schedule.beta_r(t_f).item()),
                "beta_G_t": float(schedule.beta_g(t_f).item()),
                "state_t_legal": state_t.validate()["legal"],
                "state_s_legal": state_s.validate()["legal"],
            }
            for k in ("entropy_R", "entropy_G", "logit_mean_R", "logit_mean_G"):
                if k in losses:
                    row[k] = float(losses[k].detach())
            stream.write(json.dumps(row) + "\n")
            if step % log_every == 0 or step + 1 == steps:
                print(json.dumps(row), flush=True)

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
    print(json.dumps({"event": "j1_train_done", "output": str(out)}), flush=True)


if __name__ == "__main__":
    main()
