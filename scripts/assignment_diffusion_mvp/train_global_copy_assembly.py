"""Future training entry point; intentionally not invoked in the MVP coding turn."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
import yaml
import torch

from mattergen.assignment.global_copy_assembly import GlobalCopyAssemblyConfig, GlobalStructuredCopyAssembly, build_assembly_target


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


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--steps",type=int,required=True);parser.add_argument("--execute",action="store_true",help="required safety acknowledgement; this command is not run by the implementation turn")
    args=parser.parse_args()
    if not args.execute: raise SystemExit("Refusing to train without --execute")
    cfg,sample,target,tree,model=load_setup(args.config);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    output_dir=Path(cfg["output_dir"]);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"config_audit.json").write_text(json.dumps({"status":"CLEAN_GEOMETRY_ORACLE_R_ONLY","use_copy_id_as_input":False,"use_oracle_copy_relation":False,"sample":sample["id"]},indent=2))
    (output_dir/"anchor_and_tree.json").write_text(json.dumps({"anchor":tree.root,"tree_edges":tree.tree_edges,"non_tree_edges":tree.non_tree_edges,"preorder":tree.preorder,"postorder":tree.postorder},indent=2))
    (output_dir/"permutation_convention.json").write_text(json.dumps({"convention":"P_r[q]=k: role-r instance q is assigned copy-gauge label k","anchor":"P_anchor[q]=q"},indent=2))
    with (output_dir/"training_trace.jsonl").open("w") as stream:
        for step in range(args.steps):
            values=training_step(model,target,tree,sample);optimizer.zero_grad(set_to_none=True);values["loss"].backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()): raise FloatingPointError("global copy-assembly gradient is non-finite")
            optimizer.step();stream.write(json.dumps({"step":step,**{key:float(value.detach()) for key,value in values.items()}})+"\n")
    # Checkpoint selection must monitor val/projected_bond_f1 then val/copy_pair_f1;
    # this minimal fixed-sample entry deliberately never selects best_loss=0.
    (output_dir/"checkpoint_selection.json").write_text(json.dumps({"monitor":"val/projected_bond_f1","tie_break":"val/copy_pair_f1","forbidden_monitor":"best_loss=0"},indent=2))


if __name__=="__main__": main()
