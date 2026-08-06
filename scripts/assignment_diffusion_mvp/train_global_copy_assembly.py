"""Future training entry point; intentionally not invoked in the MVP coding turn."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
import yaml
import torch

from mattergen.assignment.global_copy_assembly import AssemblyTarget, GlobalCopyAssemblyConfig, GlobalStructuredCopyAssembly, build_assembly_target


def load_setup(config_path: Path):
    cfg=yaml.safe_load(config_path.read_text())["global_copy_assembly"]
    if not cfg.get("enabled",False): raise ValueError("global_copy_assembly.enabled must be true for this standalone entry point")
    sample=torch.load(cfg["fixed_sample_path"],map_location="cpu",weights_only=False)
    if sample["id"]!=cfg["fixed_sample_id"] or sample["split"]!=cfg["split"]: raise ValueError("fixed sample identity/split mismatch")
    orbit_json=json.loads(Path(cfg["automorphism_orbits_path"]).read_text())
    orbits=[value for _,value in sorted(orbit_json["role_orbits"].items(),key=lambda item:int(item[0]))]
    allowed={field.name for field in fields(GlobalCopyAssemblyConfig)}; model_cfg=GlobalCopyAssemblyConfig(**{key:value for key,value in cfg.items() if key in allowed})
    model=GlobalStructuredCopyAssembly(model_cfg);tree=model.select_tree(orbits,sample["role_edge_index"],M=int(sample["M"]))
    target=build_assembly_target(sample["role"],sample["copy"],M=int(sample["M"]),K=int(sample["Z"]),anchor_role=tree.root)
    return cfg,sample,target,tree,model


def training_step(model, target, tree, sample):
    """One finite-checked optimization objective; caller owns optimizer/loop."""
    output=model.loss(target=target,tree=tree,z=sample["z"],frac=sample["pos"],cell=sample["cell"],role_z=sample["role_z"],role_edge_index=sample["role_edge_index"],role_bond_type=sample["role_bond_type"])
    if not torch.isfinite(output["loss"]): raise FloatingPointError("global copy-assembly loss is non-finite")
    return output


def resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def move_target(target: AssemblyTarget, device: torch.device) -> AssemblyTarget:
    return AssemblyTarget(
        role_sets={role:nodes.to(device) for role,nodes in target.role_sets.items()},
        permutations={role:state.to(device) for role,state in target.permutations.items()},
        anchor_role=target.anchor_role,
        K=target.K,
        M=target.M,
    )


def move_sample_tensors(sample: dict, device: torch.device) -> dict:
    """Move model-visible tensors only; ``copy`` remains target construction data."""
    return {key:(value.to(device) if torch.is_tensor(value) else value) for key,value in sample.items() if key != "copy"}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--steps",type=int,required=True);parser.add_argument("--output-dir",type=Path,default=None,help="optional run directory; avoids overwriting a prior diagnostic trace");parser.add_argument("--execute",action="store_true",help="required safety acknowledgement; this command is not run by the implementation turn")
    args=parser.parse_args()
    if not args.execute: raise SystemExit("Refusing to train without --execute")
    cfg,sample,target,tree,model=load_setup(args.config)
    if args.output_dir is not None:
        cfg["output_dir"]=str(args.output_dir)
    device=resolve_device(str(cfg.get("device","auto")))
    sample=move_sample_tensors(sample,device);target=move_target(target,device);model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg.get("learning_rate",5e-5)),weight_decay=float(cfg.get("weight_decay",1e-4)))
    gradient_clip_norm=float(cfg.get("gradient_clip_norm",1.0));log_every_steps=int(cfg.get("log_every_steps",10))
    if gradient_clip_norm <= 0 or log_every_steps <= 0:
        raise ValueError("gradient_clip_norm and log_every_steps must be positive")
    output_dir=Path(cfg["output_dir"]);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"config_audit.json").write_text(json.dumps({"status":"CLEAN_GEOMETRY_ORACLE_R_ONLY","use_copy_id_as_input":False,"use_oracle_copy_relation":False,"sample":sample["id"],"device":str(device),"learning_rate":optimizer.param_groups[0]["lr"],"gradient_clip_norm":gradient_clip_norm,"pair_score_scale":model.config.pair_score_scale},indent=2))
    (output_dir/"anchor_and_tree.json").write_text(json.dumps({"anchor":tree.root,"tree_edges":tree.tree_edges,"non_tree_edges":tree.non_tree_edges,"preorder":tree.preorder,"postorder":tree.postorder},indent=2))
    (output_dir/"permutation_convention.json").write_text(json.dumps({"convention":"P_r[q]=k: role-r instance q is assigned copy-gauge label k","anchor":"P_anchor[q]=q"},indent=2))
    print(json.dumps({"event":"training_start","device":str(device),"steps":args.steps,"learning_rate":optimizer.param_groups[0]["lr"],"gradient_clip_norm":gradient_clip_norm}),flush=True)
    with (output_dir/"training_trace.jsonl").open("w",buffering=1) as stream:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True);values=training_step(model,target,tree,sample);values["loss"].backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()): raise FloatingPointError("global copy-assembly gradient is non-finite")
            gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=gradient_clip_norm)
            if not torch.isfinite(gradient_norm): raise FloatingPointError("global copy-assembly gradient norm is non-finite")
            optimizer.step();stream.write(json.dumps({"step":step,**{key:float(value.detach()) for key,value in values.items()}})+"\n")
            if step % log_every_steps == 0 or step+1 == args.steps:
                print(json.dumps({"step":step,**{key:float(value.detach()) for key,value in values.items()},"gradient_norm":float(gradient_norm.detach())}),flush=True)
    # Checkpoint selection must monitor val/projected_bond_f1 then val/copy_pair_f1;
    # this minimal fixed-sample entry deliberately never selects best_loss=0.
    (output_dir/"checkpoint_selection.json").write_text(json.dumps({"monitor":"val/projected_bond_f1","tie_break":"val/copy_pair_f1","forbidden_monitor":"best_loss=0"},indent=2))


if __name__=="__main__": main()
