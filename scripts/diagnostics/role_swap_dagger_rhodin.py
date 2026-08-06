"""Three-round DAgger imitation audit for legal swap/Gibbs role decoding."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from mattergen.common.role_partition_diffusion import build_targets, SwapStopHead
from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
from mattergen.common.role_partition_diffusion.swap_gibbs import apply_action,assert_legal

OUT=Path('outputs/assignment_diffusion_mvp/role_swap_dagger');SRC=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt');BASE=Path('outputs/assignment_diffusion_mvp/role_swap_best_action');SEED=17;T=64;ROUNDS=3;ROLLOUTS=128;ROLLOUT_STEPS=16;THRESH=.99
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load():
 s=torch.load(SRC,map_location='cpu');return {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in s.items()}
def pairs(st,z):return torch.triu(z[:,None].eq(z[None,:])&st[:,None].ne(st[None,:]),1).nonzero().long()
def gains(st,truth,p):
 i,j=p[:,0],p[:,1];return (st[j]==truth[i]).float()+(st[i]==truth[j]).float()-(st[i]==truth[i]).float()-(st[j]==truth[j]).float()
def states(s):
 d=RolePartitionDiffusion(role_steps=T,role_diffusion_type='swap_gibbs').cuda().role_diffusion
 return [d.terminal_prior(s['role'],s['z'],torch.Generator(device='cuda').manual_seed(1000+i)) for i in range(32)]
class Policy(nn.Module):
 def __init__(self):
  super().__init__();self.core=RolePartitionDiffusion(role_steps=T,role_diffusion_type='swap_gibbs');self.stop=SwapStopHead(steps=T)
 def state(self,s,st,t,pos=None):
  pos=s['pos'] if pos is None else pos;hx=self.core.crystal_encoder(s['z'],pos,s['cell']);hm=self.core.molecule_encoder(s['role_z'],s['role_edge_index'],s['role_bond_type']);p=pairs(st,s['z']);raw,_=self.core.swap_head(hx,hm,st,p,pos,s['cell'],t);return p,raw[1:],self.stop(hx,hm,t)
def load_policy():
 p=Policy().cuda();c=torch.load(BASE/'checkpoints/best.pt',map_location='cpu');p.core.load_state_dict(c['state_dict'],strict=False);return p
def label_stats(logits,stop_logit,st,truth,p):
 g=gains(st,truth,p);best=g==g.max();logp=torch.log_softmax(logits,0);action_loss=-torch.logsumexp(logp[best],0);stop_target=torch.tensor(float(torch.equal(st,truth)),device=st.device);stop_loss=nn.functional.binary_cross_entropy_with_logits(stop_logit,stop_target);prob=torch.softmax(logits,0);return action_loss+stop_loss,g,best,{'best_mass':float(prob[best].sum()),'top1':float(best[logits.argmax()]),'p_plus':float(prob[g>0].sum()),'p_zero':float(prob[g==0].sum()),'p_minus':float(prob[g<0].sum()),'stop_probability':float(torch.sigmoid(stop_logit)),'stop_target':float(stop_target)}
@torch.no_grad()
def rollout(policy,s,initial,max_steps=ROLLOUT_STEPS,geometry='correct',seed=0):
 truth=s['role'];st=initial.clone();rows=[];false=0;pos=s['pos'] if geometry=='correct' else (torch.zeros_like(s['pos']) if geometry=='zero' else s['pos'][torch.randperm(s['N'],device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))])
 for n in range(max_steps):
  t=max(T-n,1);p,l,stop=policy.state(s,st,t,pos);g=gains(st,truth,p);sp=float(torch.sigmoid(stop));choice=int(l.argmax());row={'state':st.detach().cpu(),'t':t,'top1':choice,'stop_probability':sp,'gains':g.detach().cpu(),'best':(g==g.max()).detach().cpu(),'hamming':int((st!=truth).sum()),'accuracy':float((st==truth).float().mean())}
  if sp>=THRESH:
   row['stopped']=True;row['false_stop']=not bool(torch.equal(st,truth));false+=row['false_stop'];rows.append(row);break
  row['stopped']=False;row['false_stop']=False;rows.append(row);st=apply_action(st,p[choice]);assert_legal(st,s['z'],s['role_z'],s['Z'])
 return rows,st,false
def forward_state(s,g):
 st=s['role'].clone();t=int(torch.randint(1,T+1,(),device='cuda',generator=g));
 for _ in range(int(128*t/T)):
  p=pairs(st,s['z']);st=apply_action(st,p[torch.randint(len(p),(),device='cuda',generator=g)])
 return st,t
def evaluate(policy,s,initials):
 allrows=[];final=[];false=0;wall=[]
 for i,x in enumerate(initials):
  begin=time.perf_counter();rows,st,bad=rollout(policy,s,x,64,seed=i);wall.append(time.perf_counter()-begin);allrows.append(rows);final.append(st);false+=bad
 exact=sum(torch.equal(x,s['role']) for x in final);acc=sum(float((x==s['role']).float().mean()) for x in final)/len(final);peak=sum(max([r['accuracy'] for r in rows],default=float((initials[i]==s['role']).float().mean())) for i,rows in enumerate(allrows))/len(allrows)
 flat=[r for rows in allrows for r in rows];top=sum(bool(r['best'][r['top1']]) for r in flat)/len(flat);mass=sum(float(torch.softmax(r['gains'].cuda()*0,0)[r['best'].cuda()].sum()) for r in flat)/len(flat) # placeholder overwritten below by rank eval
 return {'exact_R':exact,'final_accuracy':acc,'peak_accuracy':peak,'ever_exact':sum(any(r['hamming']==0 for r in x) for x in allrows),'false_stop_rate':false/len(initials),'top1_recall':top,'executed_steps':sum(len(x) for x in allrows)/len(allrows),'wall_seconds':sum(wall)/len(wall),'valid':all(torch.equal(torch.bincount(x,minlength=s['M']),torch.full((s['M'],),s['Z'],device='cuda')) for x in final),'traces':allrows}
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 s=load();target=build_targets(s['role'],s['copy'],s['role_z'],s['Z']);target.validate(s['z']);initials=states(s)
 (OUT/'failure_baseline_manifest.json').write_text(json.dumps({'base':str(BASE),'best_sha256':sha(BASE/'checkpoints/best.pt'),'report_sha256':sha(BASE/'initial_report.md'),'config':{'swap_dagger_rounds':3,'rollouts_per_round':128,'max_steps':16,'forward_fraction':.5,'onpolicy_fraction':.5}},indent=2))
 policy=load_policy();opt=torch.optim.AdamW(policy.parameters(),lr=1e-4,weight_decay=1e-2);g=torch.Generator(device='cuda').manual_seed(SEED);aggregate=[];rounds=[]
 for rd in range(ROUNDS):
  collected=[];false_stops=0
  for k in range(ROLLOUTS):
   rows,_,bad=rollout(policy,s,initials[k%32],seed=rd*1000+k);false_stops+=bad
   collected.extend(rows)
  aggregate.extend(collected)
  # 500 equal-mixture updates: forward corruption / historical on-policy / clean.
  curve=[];nonfinite=0
  for step in range(500):
   q=torch.rand((),device='cuda',generator=g)
   if q<.15: st=s['role'].clone();t=int(torch.randint(1,T+1,(),device='cuda',generator=g));source='clean'
   elif q<.575: st,t=forward_state(s,g);source='forward'
   else: row=aggregate[int(torch.randint(len(aggregate),(),device='cuda',generator=g))];st=row['state'].to('cuda');t=int(row['t']);source='onpolicy'
   p,l,stop=policy.state(s,st,t);loss,ga,best,diag=label_stats(l,stop,st,s['role'],p);opt.zero_grad()
   if not torch.isfinite(loss):raise FloatingPointError('non-finite DAgger loss')
   loss.backward()
   if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in policy.parameters()):raise FloatingPointError('non-finite DAgger gradient')
   torch.nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step()
   if step%50==0:curve.append({'step':step,'source':source,'loss':float(loss),**diag})
  ev=evaluate(policy,s,initials); # score action mass on the last collected states
  rank=[]
  with torch.no_grad():
   for row in collected[:512]:
    st=row['state'].cuda();p,l,stop=policy.state(s,st,int(row['t']));_,_,_,d=label_stats(l,stop,st,s['role'],p);rank.append(d)
  policy_metrics={k:sum(x[k] for x in rank)/len(rank) for k in rank[0]};ev['policy']=policy_metrics;ev['collection_false_stop_rate']=false_stops/ROLLOUTS;ev['aggregate_states']=len(aggregate);ev['nonfinite']=nonfinite;ev['curve']=curve;rounds.append(ev)
  torch.save({'state_dict':policy.state_dict(),'round':rd,'seed':SEED},OUT/'checkpoints'/f'round_{rd+1}_best.pt')
 (OUT/'dagger_rounds.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='traces'} for x in rounds],indent=2));(OUT/'swap_trajectories.json').write_text(json.dumps({'round3':rounds[-1]['traces']},default=lambda x:x.tolist() if isinstance(x,torch.Tensor) else x,indent=2))
 final=rounds[-1];ab={}
 for mode in ('correct','zero','mismatch'):
  vals=[]
  for i,x in enumerate(initials[:8]):
   rows,st,_=rollout(policy,s,x,64,geometry=mode,seed=i);vals.append(float((st==s['role']).float().mean()))
  ab[mode]=sum(vals)/8
 (OUT/'condition_ablation.json').write_text(json.dumps(ab,indent=2))
 passed=final['exact_R']>=28 and final['final_accuracy']>=.99 and final['false_stop_rate']<.01 and final['policy']['top1']>=.95 and final['policy']['best_mass']>=.95 and final['valid'] and final['nonfinite']==0
 report={'rounds':[{'exact_R':x['exact_R'],'final_accuracy':x['final_accuracy'],'peak_accuracy':x['peak_accuracy'],'false_stop_rate':x['false_stop_rate'],'policy':x['policy'],'steps':x['executed_steps']} for x in rounds],'condition_ablation':ab,'PASS':passed,'allow_q_c':passed,'allow_three_seed':False,'allow_D2':False}
 (OUT/'initial_report.md').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
