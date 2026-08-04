"""Tensor-by-tensor comparison for separate full-model audit subprocesses."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def flatten(value, prefix=""):
    output={}
    if isinstance(value,torch.Tensor): output[prefix]=value
    elif isinstance(value,dict):
        for key,item in value.items(): output.update(flatten(item,f"{prefix}.{key}".strip(".")))
    return output
def compare(left,right):
    a,b=flatten(left),flatten(right); rows=[]
    for name in sorted(set(a)|set(b)):
        if name not in a or name not in b: rows.append({"name":name,"present_baseline":name in a,"present_current":name in b}); continue
        x,y=a[name],b[name]; same_shape=tuple(x.shape)==tuple(y.shape); same_dtype=x.dtype==y.dtype
        if same_shape and same_dtype:
            delta=(x-y).abs(); denom=torch.maximum(torch.maximum(x.abs(),y.abs()),torch.tensor(1e-30,dtype=x.dtype)); rows.append({"name":name,"shape":list(x.shape),"dtype":str(x.dtype),"bitwise_equal":bool(torch.equal(x,y)),"max_absolute_error":float(delta.max()) if delta.numel() else 0.,"max_relative_error":float((delta/denom).max()) if delta.numel() else 0.,"allclose":bool(torch.allclose(x,y,rtol=1e-6,atol=1e-7)),"nan_count":int(torch.isnan(x).sum()+torch.isnan(y).sum()),"inf_count":int(torch.isinf(x).sum()+torch.isinf(y).sum())})
        else: rows.append({"name":name,"baseline_shape":list(x.shape),"current_shape":list(y.shape),"baseline_dtype":str(x.dtype),"current_dtype":str(y.dtype),"bitwise_equal":False,"allclose":False})
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline",nargs="+",required=True);p.add_argument("--current",nargs="+",required=True);p.add_argument("--output-dir",required=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);forward=[];backward=[];rng=[];optimizer=[]
    for left_path,right_path in zip(a.baseline,a.current):
        left,right=torch.load(left_path,map_location="cpu"),torch.load(right_path,map_location="cpu"); common={key:left[key] for key in ["t","noisy_pos","noisy_cell","pos_score","cell_score","atomic_numbers_output","loss","loss_components","internals"]}; other={key:right[key] for key in ["t","noisy_pos","noisy_cell","pos_score","cell_score","atomic_numbers_output","loss","loss_components","internals"]}; forward.append({"batch_ids":left["ids"],"tensors":compare(common,other)}); backward.append({"batch_ids":left["ids"],"tensors":compare({"gradients":left["gradients"]},{"gradients":right["gradients"]})}); rng.append({"batch_ids":left["ids"],"before_equal":left["rng_before"]==right["rng_before"],"after_forward_equal":left["rng_after_forward"]==right["rng_after_forward"],"after_equal":left["rng_after"]==right["rng_after"],"baseline":left["rng_after"],"current":right["rng_after"]}); optimizer.append({"batch_ids":left["ids"],"equal":left["optimizer_groups"]==right["optimizer_groups"],"baseline":left["optimizer_groups"],"current":right["optimizer_groups"],"state_key_sets_equal":set(left["state_keys"])==set(right["state_keys"]),"parameter_names_equal":left["parameter_names"]==right["parameter_names"],"assignment_current":right["assignment"]})
    (out/"forward_comparison.json").write_text(json.dumps(forward,indent=2));(out/"backward_comparison.json").write_text(json.dumps(backward,indent=2));(out/"rng_comparison.json").write_text(json.dumps(rng,indent=2));(out/"optimizer_comparison.json").write_text(json.dumps(optimizer,indent=2));
if __name__=="__main__":main()
