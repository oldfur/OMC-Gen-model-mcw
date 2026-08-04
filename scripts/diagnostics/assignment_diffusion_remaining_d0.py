"""Remaining no-training D0 checks: live predictor equivariance and DDPM algebra."""
import json
from pathlib import Path
import torch
from mattergen.common.assignment_diffusion import AssignmentDiffusion, gauge_center, sinkhorn

def predict(d, y, z, zr, deg, pos, cell, ta=3, tx=.2):
    m=z[:,None].eq(zr[None,:]); a=sinkhorn(y,m,d.iters,d.tol)
    hi=d.atom(z)+d.geometry(torch.cat([pos,cell.reshape(1,9).expand(len(z),-1)],-1))
    hr=d.role(zr)+d.role_graph(deg[:,None]); g=(a.T@hi)/a.sum(0)[:,None].clamp_min(1e-8)
    n=len(z); extra=torch.stack([y,a,torch.full_like(y,ta/d.steps),torch.full_like(y,tx)],-1)
    return d.net(torch.cat([hi[:,None].expand(-1,n,-1),hr[None].expand(n,-1,-1),g[None].expand(n,-1,-1),extra],-1)).squeeze(-1).masked_fill(~m,0.)

def stats(xs):
    x=torch.tensor(xs); return {'max':float(x.max()),'mean':float(x.mean()),'p95':float(torch.quantile(x,.95))}

def main():
 torch.manual_seed(71); d=AssignmentDiffusion(steps=16).eval(); n=6; z=torch.tensor([6,6,8,8,1,1]); zr=z.clone(); deg=torch.arange(n,dtype=torch.float); pos=torch.randn(n,3); cell=torch.eye(3); mask=z[:,None].eq(zr[None,:]); y=gauge_center(torch.randn(n,n),mask); base=predict(d,y,z,zr,deg,pos,cell)
 row=[]; col=[]; both=[]
 for seed in range(20):
  g=torch.Generator().manual_seed(seed); pr=torch.randperm(n,generator=g); pc=torch.randperm(n,generator=g)
  r=predict(d,y[pr],z[pr],zr,deg,pos[pr],cell); c=predict(d,y[:,pc],z,zr[pc],deg[pc],pos,cell); b=predict(d,y[pr][:,pc],z[pr],zr[pc],deg[pc],pos[pr],cell)
  row.append(float((r-base[pr]).abs().max())); col.append(float((c-base[:,pc]).abs().max())); both.append(float((b-base[pr][:,pc]).abs().max()))
 # Independent closed-form expression intentionally does not call d.posterior.
 oracle=[]
 for dtype in (torch.float32,torch.float64):
  yy=gauge_center(torch.randn(n,n,dtype=dtype),mask); x0=gauge_center(torch.randn(n,n,dtype=dtype),mask)
  for t in (1,8,15):
   # Keep schedule scalars in their stored dtype, exactly as the implementation
   # does; y/x0 may be float64 without silently changing the trained schedule.
   ab=d.alpha_bar[t]; prev=d.alpha_bar[t-1]; alpha=ab/prev; beta=1-alpha
   ref=(prev.sqrt()*beta/(1-ab))*x0+(alpha.sqrt()*(1-prev)/(1-ab))*yy; var=beta*(1-prev)/(1-ab)
   got, gotvar=d.posterior(yy,x0,t); oracle.append({'dtype':str(dtype).split('.')[-1],'t':t,'mean_error':float((got-ref).abs().max()),'variance_error':float((gotvar-var).abs()),'gauge_residual':float(max(gauge_center(got,mask).sub(got).abs().max(),torch.tensor(0.)))})
 out={'seed':71,'predictor_inputs':{'Yt':[n,n],'crystal_pos':[n,3],'cell':[3,3],'slot_atomic_numbers':[n],'role_atomic_numbers':[n],'role_degree':[n],'element_mask':[n,n]},'predictor_has_flatten':False,'predictor_has_copy_index':False,'entrywise_mlp_with_column_aggregate':True,'row_equivariance':stats(row),'column_equivariance':stats(col),'joint_equivariance':stats(both),'reverse_oracle':oracle}
 p=Path('outputs/assignment_diffusion_mvp');p.mkdir(parents=True,exist_ok=True);p.joinpath('d0_predictor_equivariance.json').write_text(json.dumps(out,indent=2));p.joinpath('d0_reverse_oracle.json').write_text(json.dumps(out['reverse_oracle'],indent=2));print(p/'d0_predictor_equivariance.json')
if __name__=='__main__': main()
