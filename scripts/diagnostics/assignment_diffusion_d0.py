"""Small deterministic D0 operator audit; no model or geometry diffusion training."""
from pathlib import Path
import json
import torch
from mattergen.common.assignment_diffusion.assignment_diffusion import gauge_center,sinkhorn
torch.manual_seed(11)
mask=torch.tensor([[1,0,1,0,0],[0,1,0,1,0],[1,0,1,0,0],[0,1,0,1,0],[0,0,0,0,1]],dtype=torch.bool)
y=torch.randn(5,5);p=gauge_center(y,mask);a=sinkhorn(p,mask)
row=float((p.sum(1)).abs().max());col=float((p.sum(0)).abs().max());idem=float((gauge_center(p,mask)-p).abs().max());marg=float((a.sum(1)-1).abs().max().maximum((a.sum(0)-1).abs().max()));forbid=float(a[~mask].abs().max())
shift=torch.randn(5,1)+torch.randn(1,5);inv=float((sinkhorn(y,mask)-sinkhorn(y+shift,mask)).abs().max())
try:gauge_center(torch.tensor([[1.,0.],[0.,0.]]),torch.tensor([[1,0],[0,0]],dtype=torch.bool));bad='NO_ERROR'
except ValueError:bad='ValueError'
from mattergen.common.assignment_diffusion.assignment_diffusion import AssignmentDiffusion
d=AssignmentDiffusion(steps=5);eps=gauge_center(torch.randn_like(p),mask);l0=p;ab=d.alpha_bar[2];lt=gauge_center(ab.sqrt()*l0+(1-ab).sqrt()*eps,mask);rec=(lt-ab.sqrt()*l0)/(1-ab).sqrt();recon=float((rec-eps).abs().max());_,aa,tr=d.reverse_ddpm(mask,lambda yy,tt:torch.zeros_like(yy),seed=7);_,aa2,tr2=d.reverse_ddpm(mask,lambda yy,tt:torch.zeros_like(yy),seed=7);reverse_repro=float((aa-aa2).abs().max())
metrics={'seed':11,'dtype':'float32','gauge_iterations':20,'mask_shape':[5,5],'gauge_row_residual':row,'gauge_column_residual':col,'gauge_idempotence':idem,'sinkhorn_marginal_error':marg,'forbidden_mass':forbid,'gauge_shift_sinkhorn_error':inv,'infeasible_mask':bad,'forward_reconstruction_error':recon,'reverse_reproducibility_error':reverse_repro,'reverse_steps':len(tr),'reverse_final_marginal_error':tr[-1]['marginal_error']};out=Path('outputs/assignment_diffusion_mvp');out.mkdir(parents=True,exist_ok=True);out.joinpath('d0_metrics.json').write_text(json.dumps(metrics,indent=2))
print(out/'d0_metrics.json')
