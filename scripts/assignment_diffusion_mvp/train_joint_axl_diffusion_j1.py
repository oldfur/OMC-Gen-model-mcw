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
    schedule = AsyncJumpSchedule(
        r_lock=float(sch_cfg.get("r_lock", 0.72)),
        g_lock=float(sch_cfg.get("g_lock", 0.52)),
        kappa_r=float(sch_cfg.get("kappa_r", 4.0)),
        kappa_g=float(sch_cfg.get("kappa_g", 6.0)),
    )
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
        "J1_MODE": "joint_axl_ctmc",
        "R_LOCK": schedule.r_lock,
        "G_LOCK": schedule.g_lock,
        "MATTERGEN_LOAD_EPOCH": bundle.load_epoch,
        "MATTERGEN_CHECKPOINT_SHA": bundle.checkpoint_sha256,
        "NOISE_SOURCE": PROVENANCE["noise_source"],
        "LIE_SPLITTING": "A_first",
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
            # forward CTMC on A
            traj = simulate_forward_ctmc(st0, schedule=schedule, t_start=0.0, t_end=1.0, generator=g)
            # sample geometry time t and segment [s,t]
            t = noise.sample_t(1, device=device)
            t_t = float(t.reshape(-1)[0].item())
            # s ~ Uniform(0,t) for reverse segment
            u = float(torch.rand((), generator=g).item())
            t_s = u * t_t
            state_t = traj.state_at(t_t)
            state_s = traj.state_at(t_s)
            # geometry noise
            noisy = noise.corrupt_fixed_sample(
                frac_coords_0=geo["pos"],
                lattice_0=sample_d["cell"],
                num_atoms=int(sample_d["N"]),
                t=t,
                generator=g,
            )
            # sample dict for chemgraph uses augmented atoms
            samp_aug = dict(sample_d)
            samp_aug["z"] = geo["z"]
            samp_aug["pos"] = geo["pos"]
            clean_cg = build_cg(samp_aug, geo["pos"], sample_d["cell"])
            noisy_cg = build_cg(samp_aug, noisy.frac_coords_t, noisy.lattice_t)

            opt.zero_grad(set_to_none=True)
            losses = joint_training_step_losses(
                model=model,
                loss_fn=loss_fn,
                corruption=noise.corruption,
                clean_cg=clean_cg,
                noisy_cg=noisy_cg,
                t=noisy.t,
                state_for_geometry=state_s,  # A_s for geometry (A-first Lie)
                traj=traj,
                t_s=t_s,
                t_t=t_t,
                lambda_r=float(loss_w.get("lambda_r", 1.0)),
                lambda_g=float(loss_w.get("lambda_g", 1.0)),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            row = {
                "step": step,
                "t": t_t,
                "t_s": t_s,
                "geometry_loss": float(losses["L_geom"].detach()),
                "L_R": float(losses["L_R"].detach()),
                "L_G": float(losses["L_G"].detach()),
                "total_loss": float(losses["loss"].detach()),
                "num_fwd_events": len(traj.events),
                "state_t_legal": state_t.validate()["legal"],
                "state_s_legal": state_s.validate()["legal"],
            }
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
