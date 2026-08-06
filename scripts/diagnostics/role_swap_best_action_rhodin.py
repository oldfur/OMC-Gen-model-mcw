"""Oracle and best-action-set policy audit for fixed RHODIN01 swap/Gibbs R."""
from __future__ import annotations
import hashlib, json, math, os, time
from pathlib import Path
import torch
from mattergen.common.role_partition_diffusion import build_targets
from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
from mattergen.common.role_partition_diffusion.swap_gibbs import apply_action, assert_legal

OUT=Path('outputs/assignment_diffusion_mvp/role_swap_best_action');SRC=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt');BASE=Path('outputs/assignment_diffusion_mvp/role_swap_gibbs_q_masked')
SEED=17; T=64; MAX_STEPS=int(os.environ.get('SWAP_BEST_MAX_STEPS','5000')); CLEAN_P=.15
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load():
 s=torch.load(SRC,map_location='cpu');return {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in s.items()}
def pairs(state,z):
 same=z[:,None].eq(z[None,:]);different=state[:,None].ne(state[None,:]);return torch.triu(same&different,diagonal=1).nonzero().long()
def legal(state,s):assert_legal(state,s['z'],s['role_z'],s['Z'])
def net():return RolePartitionDiffusion(role_steps=T,role_diffusion_type='swap_gibbs',swap_terminal_randomization_steps=128).cuda()
def terminal_states(s):
 d=net().role_diffusion;truth=s['role'];out=[]
 for n in range(32):out.append(d.terminal_prior(truth,s['z'],torch.Generator(device='cuda').manual_seed(1000+n)))
 return out
def score(model,s,state,t,pos=None):
 pos=s['pos'] if pos is None else pos;hx=model.crystal_encoder(s['z'],pos,s['cell']);hm=model.molecule_encoder(s['role_z'],s['role_edge_index'],s['role_bond_type']);p=pairs(state,s['z']);logits,_=model.swap_head(hx,hm,state,p,pos,s['cell'],t);return p,logits
def gains(state,truth,p):
 if not len(p): return torch.zeros(1,device=state.device)
 i,j=p[:,0],p[:,1]
 before=(state[i]==truth[i]).to(torch.float32)+(state[j]==truth[j]).to(torch.float32)
 after=(state[j]==truth[i]).to(torch.float32)+(state[i]==truth[j]).to(torch.float32)
 return torch.cat([torch.zeros(1,device=state.device),after-before])
def oracle(s,states):
 truth=s['role'];rows=[]
 for n,initial in enumerate(states):
  st=initial.clone();hist=[int((st!=truth).sum())];local=False
  for k in range(T):
   p=pairs(st,s['z']);g=gains(st,truth,p);best=int(g.argmax())
   if g[best]<=0: local=not bool(torch.equal(st,truth));break
   st=apply_action(st,p[best-1]);legal(st,s);hist.append(int((st!=truth).sum()))
  rows.append({'seed':1000+n,'initial_hamming':hist[0],'final_hamming':hist[-1],'exact':bool(torch.equal(st,truth)),'swaps':len(hist)-1,'local_nonclean_optimum':local,'within_64':len(hist)-1<=T,'trajectory':hist})
 return rows
def loss_and_diag(logits,g):
 best=g==g.max();loss=torch.logsumexp(logits,0)-torch.logsumexp(logits[best],0);p=torch.softmax(logits,0);return loss,{'best_mass':float(p[best].sum()),'top1_best':bool(best[logits.argmax()]),'top5_best':bool(best[torch.topk(logits,min(5,len(logits))).indices].any()),'p_plus':float(p[g>0].sum()),'p_zero':float(p[g==0].sum()),'p_minus':float(p[g<0].sum()),'entropy':float(-(p*p.clamp_min(1e-30).log()).sum()),'noop_probability':float(p[0])}
@torch.no_grad()
def policy_eval(model,s,states,clean=False,t=T):
 truth=s['role'];all=[]
 for st0 in states:
  st=truth.clone() if clean else st0; p,l=score(model,s,st,t);g=gains(st,truth,p);_,d=loss_and_diag(l,g);all.append(d)
 keys=all[0].keys();return {k:sum(float(x[k]) for x in all)/len(all) for k in keys}
def train(s,states):
 torch.manual_seed(SEED);m=net();torch.save({'state_dict':{k:v.detach().cpu() for k,v in m.state_dict().items()},'loss_type':'best_action_set','seed':SEED,'step':0},OUT/'checkpoints'/'step_0.pt');opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-2);g=torch.Generator(device='cuda').manual_seed(SEED);truth=s['role'];curve=[];best=((-1.,-1.,-1.),None);nonfinite=0
 for step in range(MAX_STEPS):
  clean=bool(torch.rand((),device='cuda',generator=g)<CLEAN_P)
  if clean: st=truth.clone();t=T
  else:
   t=int(torch.randint(1,T+1,(),device='cuda',generator=g));st=truth.clone();n=int(math.floor(128*t/T))
   for _ in range(n):
    p=pairs(st,s['z']);st=apply_action(st,p[torch.randint(len(p),(),device='cuda',generator=g)])
  p,l=score(m,s,st,t);ga=gains(st,truth,p);loss,d=loss_and_diag(l,ga);opt.zero_grad()
  if not torch.isfinite(loss):raise FloatingPointError('non-finite best-action loss')
  loss.backward()
  if any(q.grad is not None and not torch.isfinite(q.grad).all() for q in m.parameters()):raise FloatingPointError('non-finite gradient')
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
  if step%100==0:
   clean_eval=policy_eval(m,s,states[:8],clean=True);state_eval=policy_eval(m,s,states[:8]);key=(state_eval['best_mass'],clean_eval['top1_best'],state_eval['top1_best'])
   curve.append({'step':step,'loss':float(loss),'clean_draw':clean,**d,'validation':state_eval,'clean_validation':clean_eval})
   if key>best[0]:best=(key,{k:v.detach().cpu() for k,v in m.state_dict().items()})
 final={k:v.detach().cpu() for k,v in m.state_dict().items()};m.load_state_dict(best[1]);return m,curve,best[0],final,nonfinite
@torch.no_grad()
def sampler(model,s,initial,kind,geometry='correct',seed=0):
 truth=s['role'];st=initial.clone();hist=[];ever=False;first=None;left=False;start=time.perf_counter();posn=neg=neu=noops=0
 pos=s['pos'] if geometry=='correct' else (torch.zeros_like(s['pos']) if geometry=='zero' else s['pos'][torch.randperm(s['N'],device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))])
 for n,t in enumerate(range(T,0,-1)):
  p,l=score(model,s,st,t,pos);temp=1. if kind=='stochastic' else (1.-.95*n/(T-1) if n<T-8 else .05)
  choice=int(l.argmax()) if kind=='map' or n>=T-8 and kind=='annealed' else int(torch.multinomial(torch.softmax(l/temp,0),1))
  ga=gains(st,truth,p);gain=float(ga[choice]);posn+=gain>0;neu+=gain==0;neg+=gain<0
  if choice==0:noops+=1;break
  st=apply_action(st,p[choice-1]);legal(st,s);acc=float((st==truth).float().mean());exact=bool(torch.equal(st,truth));
  if exact and not ever:ever=True;first=n+1
  if ever and not exact:left=True
  hist.append({'t':t,'accuracy':acc,'hamming':int((st!=truth).sum()),'gain':gain})
 elapsed=time.perf_counter()-start;denom=max(1,len(hist)+noops);return {'final_accuracy':float((st==truth).float().mean()),'exact':bool(torch.equal(st,truth)),'peak_accuracy':max([float((initial==truth).float().mean())]+[x['accuracy'] for x in hist]),'ever_exact':ever,'first_exact_step':first,'left_after_exact':left,'positive_rate':posn/denom,'neutral_rate':neu/denom,'negative_rate':neg/denom,'no_op_rate':noops/denom,'executed_steps':len(hist),'wall_seconds':elapsed,'trajectory':hist}
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 s=load();target=build_targets(s['role'],s['copy'],s['role_z'],s['Z']);target.validate(s['z']);states=terminal_states(s)
 manifest={'baseline_checkpoint':str(BASE/'checkpoints/swap_role_best.pt'),'checkpoint_sha256':sha(BASE/'checkpoints/swap_role_best.pt'),'metrics_sha256':sha(BASE/'swap_sampling_metrics.json'),'config':{'role_diffusion_type':'swap_gibbs','swap_loss_type':'boltzmann_gain'},'metrics':json.loads((BASE/'swap_sampling_metrics.json').read_text())}
 (OUT/'failure_baseline_manifest.json').write_text(json.dumps(manifest,indent=2))
 o=oracle(s,states);(OUT/'oracle_greedy_metrics.json').write_text(json.dumps(o,indent=2))
 if sum(x['exact'] for x in o)!=32:raise RuntimeError('STOP: oracle greedy did not reach clean from every terminal state')
 m,curve,key,final,nonfinite=train(s,states);torch.save({'state_dict':m.state_dict(),'loss_type':'best_action_set','seed':SEED},OUT/'checkpoints'/'best.pt');torch.save({'state_dict':final,'loss_type':'best_action_set','seed':SEED},OUT/'checkpoints'/'final.pt')
 pol=policy_eval(m,s,states);clean=policy_eval(m,s,states,clean=True);(OUT/'policy_mass_metrics.json').write_text(json.dumps({'terminal_policy':pol,'clean_policy':clean},indent=2));(OUT/'best_action_training_metrics.json').write_text(json.dumps({'max_steps':MAX_STEPS,'clean_probability':CLEAN_P,'checkpoint_key':key,'nonfinite':nonfinite,'curve':curve},indent=2))
 comparison={};traces={}
 for kind in ('map','annealed','stochastic'):
  rows=[sampler(m,s,st,kind,seed=i) for i,st in enumerate(states)];comparison[kind]={'exact_R':sum(x['exact'] for x in rows),'mean_accuracy':sum(x['final_accuracy'] for x in rows)/32,'mean_peak_accuracy':sum(x['peak_accuracy'] for x in rows)/32,'ever_exact':sum(x['ever_exact'] for x in rows),'left_after_exact':sum(x['left_after_exact'] for x in rows),'positive_rate':sum(x['positive_rate'] for x in rows)/32,'neutral_rate':sum(x['neutral_rate'] for x in rows)/32,'negative_rate':sum(x['negative_rate'] for x in rows)/32,'no_op_rate':sum(x['no_op_rate'] for x in rows)/32,'executed_steps':sum(x['executed_steps'] for x in rows)/32,'wall_seconds':sum(x['wall_seconds'] for x in rows)/32};traces[kind]=rows
 (OUT/'sampler_comparison.json').write_text(json.dumps(comparison,indent=2));(OUT/'swap_trajectories.json').write_text(json.dumps(traces,indent=2))
 ab={};
 for mode in ('correct','zero','mismatch'):
  vals=[]
  for i,st in enumerate(states[:8]):
   # Temporarily vary only the geometry condition, retaining the terminal state.
   vals.append(sampler(m,s,st,'map',geometry=mode,seed=i)['final_accuracy'])
  ab[mode]=sum(vals)/len(vals)
 (OUT/'condition_ablation.json').write_text(json.dumps(ab,indent=2))
 passed=clean['noop_probability']>=.99 and comparison['map']['exact_R']>=28 and comparison['map']['mean_accuracy']>=.99
 (OUT/'initial_report.md').write_text(json.dumps({'oracle_exact':sum(x['exact'] for x in o),'policy_mass':pol,'clean':clean,'samplers':comparison,'PASS':passed,'allow_q_c':passed,'allow_three_seed':passed,'allow_D2':False},indent=2));print((OUT/'initial_report.md').read_text())
if __name__=='__main__':main()
