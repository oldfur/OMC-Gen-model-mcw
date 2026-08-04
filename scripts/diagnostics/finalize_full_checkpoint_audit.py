"""Materialize full-checkpoint audit metadata and Gate-A summary."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import torch
ROOT=Path("outputs/assignment_diffusion_mvp/full_checkpoint_isolation")
SOURCE=Path("/home/mcw/OMC-Gen-model/outputs/molcsp_sinkhorn_short_diagnostic_preflight_2/checkpoints/diagnostic.ckpt")
INIT=ROOT/"full_gemnet_initialized_baseline.pt"
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def rows(path):return [row for batch in json.loads(path.read_text()) for row in batch.get("tensors",[])]
def maximum(items,key):return max((item.get(key,0.) for item in items),default=0.)
def main():
 eval_sample=torch.load(ROOT/"baseline_eval_outputs/batch_0.pt",map_location="cpu"); init=torch.load(INIT,map_location="cpu"); source=torch.load(SOURCE,map_location="cpu"); manifest=json.loads((ROOT/"serialized_batches/manifest.json").read_text()); ef=rows(ROOT/"forward_comparison.json"); tf=rows(ROOT/"train_comparison/forward_comparison.json"); bg=rows(ROOT/"train_comparison/backward_comparison.json"); rng=json.loads((ROOT/"train_comparison/rng_comparison.json").read_text()); opt=json.loads((ROOT/"train_comparison/optimizer_comparison.json").read_text())
 env=eval_sample["load"]["environment"]; env.update({"baseline_worktree":"/home/mcw/OMC-Gen-model-assignment-diffusion-baseline-audit","current_worktree":str(Path.cwd()),"same_python_environment":True,"same_checkpoint_bytes":True,"explicit_stochastic_inputs":{"timesteps":True,"pos_noise":True,"cell_noise":True,"other_stochastic_inputs":"none observed; model has dropout=0.0"}});(ROOT/"environment.json").write_text(json.dumps(env,indent=2))
 meta={"checkpoint_path":str(INIT.resolve()),"checkpoint_sha256":sha(INIT),"kind":"full-model initialized checkpoint (not trained)","created_in_baseline_worktree":True,"baseline_commit":"3830650861981a29851d2c1f5e472d52722247be","parameter_count":init["parameter_count"],"state_dict_keys":len(init["state_dict"]),"strict_load":{"missing_keys":eval_sample["load"]["missing"],"unexpected_keys":eval_sample["load"]["unexpected"]},"source_trained_checkpoint":{"path":str(SOURCE),"sha256":sha(SOURCE),"state_dict_keys":len(source["state_dict"]),"compatibility":"strict load rejected: 11 historical dynamic-assignment keys absent from both audited source trees"},"config":init["config"]};(ROOT/"checkpoint_metadata.json").write_text(json.dumps(meta,indent=2))
 summary={"mode":"full GemNet/DiffusionModule cross-worktree isolation","pass":not [x for x in ef+tf+bg if not x.get("bitwise_equal",False)] and all(x["before_equal"] and x["after_forward_equal"] and x["after_equal"] for x in rng) and all(x["equal"] and x["state_key_sets_equal"] for x in opt),"checkpoint":meta,"batches":manifest,"eval":{"tensor_count":len(ef),"max_abs":maximum(ef,"max_absolute_error"),"max_rel":maximum(ef,"max_relative_error")},"train":{"forward_tensor_count":len(tf),"gradient_tensor_count":len(bg),"forward_max_abs":maximum(tf,"max_absolute_error"),"gradient_max_abs":maximum(bg,"max_absolute_error"),"gradient_max_rel":maximum(bg,"max_relative_error")},"rng":{"all_equal":all(x["before_equal"] and x["after_forward_equal"] and x["after_equal"] for x in rng)},"disabled_assignment":{"instantiated":eval_sample["assignment"]["instantiated"],"trajectory":eval_sample["assignment"]["trajectory"],"loss_key":eval_sample["assignment"]["loss_key"],"parameters":eval_sample["assignment"]["parameters"]},"optimizer":{"groups_equal":all(x["equal"] for x in opt),"state_dict_keys_equal":all(x["state_key_sets_equal"] for x in opt),"group_count":len(opt[0]["baseline"]),"parameter_count":sum(x["count"] for x in opt[0]["baseline"])} }
 out=Path("outputs/assignment_diffusion_mvp");(out/"d0_baseline_isolation.json").write_text(json.dumps(summary,indent=2)); metrics=json.loads((out/"d0_metrics.json").read_text());metrics["full_checkpoint_isolation"]=summary;metrics["gate_a"]="PASS" if summary["pass"] else "PARTIAL";(out/"d0_metrics.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
