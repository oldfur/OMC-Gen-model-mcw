"""Single-seed, R-only capacity-preserving swap/Gibbs initial audit."""
from __future__ import annotations
import csv, json, math, os
from pathlib import Path
import torch
from mattergen.common.role_partition_diffusion import build_targets
from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
from mattergen.common.role_partition_diffusion.swap_gibbs import assert_legal, apply_action

OUT=Path('outputs/assignment_diffusion_mvp/role_swap_gibbs_q_masked');SRC=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt')
SEED=17; STEPS=64; MAX_STEPS=int(os.environ.get('SWAP_MAX_STEPS','5000')); TAU=.25

def load():
 s=torch.load(SRC,map_location='cpu');return {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in s.items()}
def make_net(): return RolePartitionDiffusion(role_steps=STEPS,role_diffusion_type='swap_gibbs',swap_terminal_randomization_steps=128).cuda()
def feature(net,s,pos=None): return net.crystal_encoder(s['z'],s['pos'] if pos is None else pos,s['cell']),net.molecule_encoder(s['role_z'],s['role_edge_index'],s['role_bond_type'])
def metrics(state,truth,s):
 return {'accuracy':float((state==truth).float().mean()),'hamming':int((state!=truth).sum()),'valid':True}
def action_eval(net,s,state,t,pos=None):
 hx,hm=feature(net,s,pos);pairs=net.role_diffusion.target_distribution(state,state,s['z'],1.)[0]
 logits,_=net.swap_head(hx,hm,state,pairs,s['pos'] if pos is None else pos,s['cell'],t);return pairs,logits
def operator(s,target):
 truth=target.role;d=make_net().role_diffusion;g=torch.Generator(device='cuda').manual_seed(171)
 for t in (1,17,64):
  state=d.corrupt(truth,s['z'],t,g);assert_legal(state,s['z'],s['role_z'],s['Z'])
 terminal=d.terminal_prior(truth,s['z'],g);assert_legal(terminal,s['z'],s['role_z'],s['Z'])
 return {'forward_legal':True,'terminal_legal':True,'terminal_hamming':int((terminal!=truth).sum()),'terminal_agreement':float((terminal==truth).float().mean()),'steps':STEPS,'terminal_randomization_steps':128}
def train(s,target):
 torch.manual_seed(SEED);net=make_net();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-2);g=torch.Generator(device='cuda').manual_seed(SEED);curve=[];best=(float('inf'),None);truth=target.role;d=net.role_diffusion;nonfinite=0
 for step in range(MAX_STEPS):
  t=int(torch.randint(1,STEPS+1,(),device='cuda',generator=g));state=d.corrupt(truth,s['z'],t,g);pairs,improve,target_p=d.target_distribution(state,truth,s['z'],TAU);hx,hm=feature(net,s);logits,_=net.swap_head(hx,hm,state,pairs,s['pos'],s['cell'],t);loss=-(target_p*torch.log_softmax(logits,0)).sum();opt.zero_grad()
  if not torch.isfinite(loss): nonfinite+=1;curve.append({'step':step,'loss':float('nan'),'nonfinite':True});continue
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()): nonfinite+=1;opt.zero_grad(set_to_none=True);curve.append({'step':step,'loss':loss.item(),'nonfinite_gradient':True});continue
  torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  best_action=(logits.argmax()==improve.argmax()).item()
  if loss.item()<best[0]:best=(loss.item(),{k:v.detach().cpu() for k,v in net.state_dict().items()})
  if step%50==0:curve.append({'step':step,'loss':loss.item(),'t':t,'hamming':int((state!=truth).sum()),'best_action_accuracy':float(best_action),'actions':int(len(pairs)+1),'nonfinite':False})
 net.load_state_dict(best[1]);return net,curve,best[0],nonfinite
@torch.no_grad()
def sample(net,s,seed,mode='correct'):
 g=torch.Generator(device='cuda').manual_seed(seed);truth=s['role'];state=net.role_diffusion.terminal_prior(truth,s['z'],g);initial=metrics(state,truth,s);hist=[];noops=0;improve=0;total=0
 for t in range(STEPS,0,-1):
  pos=s['pos'] if mode=='correct' else (torch.zeros_like(s['pos']) if mode=='zero' else s['pos'][torch.randperm(s['N'],device='cuda',generator=g)])
  pairs,logits=action_eval(net,s,state,t,pos);choice=int(torch.multinomial(torch.softmax(logits,0),1,generator=g));before=(state==truth).sum()
  if choice==0:noops+=1
  else:state=apply_action(state,pairs[choice-1])
  assert_legal(state,s['z'],s['role_z'],s['Z']);after=(state==truth).sum();improve+=int(after>=before);total+=1
  hist.append({'t':t,'hamming':int((state!=truth).sum()),'accuracy':float((state==truth).float().mean()),'noop':choice==0})
 final=metrics(state,truth,s);return {'seed':seed,'mode':mode,'initial':initial,'final':final,'exact_R':bool(torch.equal(state,truth)),'selected_action_non_decrease_rate':improve/total,'no_op_rate':noops/total,'trajectory':hist}
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'trajectories').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 s=load();target=build_targets(s['role'],s['copy'],s['role_z'],s['Z']);target.validate(s['z']);op=operator(s,target);(OUT/'swap_operator_metrics.json').write_text(json.dumps(op,indent=2))
 net,curve,best,nonfinite=train(s,target);torch.save({'state_dict':net.state_dict(),'assignment_latent_mode':'role_partition','role_diffusion_type':'swap_gibbs','matching_diffusion_type':'masked_permutation','seed':SEED},OUT/'checkpoints'/'swap_role_best.pt')
 samples=[]
 for i in range(32):
  result=sample(net,s,1000+i);samples.append(result);torch.save(result,OUT/'trajectories'/f'swap_{i}.pt')
 ab={mode:sum(sample(net,s,2000+100*idx+i,mode)['final']['accuracy'] for i in range(8))/8 for idx,mode in enumerate(['correct','zero','mismatch'])}
 sequential={'ran':False,'reason':'R-only initial threshold not yet evaluated; Q is intentionally not invoked in this script'}
 (OUT/'swap_training_metrics.json').write_text(json.dumps({'seed':SEED,'max_steps':MAX_STEPS,'best_loss':best,'nonfinite_count':nonfinite,'curve':curve},indent=2));(OUT/'swap_sampling_metrics.json').write_text(json.dumps(samples,indent=2));(OUT/'condition_ablation.json').write_text(json.dumps(ab,indent=2));(OUT/'sequential_r_q_c_metrics.json').write_text(json.dumps(sequential,indent=2))
 with open(OUT/'training_curves.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['step','loss','t','hamming','best_action_accuracy','actions','nonfinite'],extrasaction='ignore');w.writeheader();w.writerows(curve)
 exact=sum(x['exact_R'] for x in samples);acc=sum(x['final']['accuracy'] for x in samples)/32;valid=all(x['final']['valid'] for x in samples);passed=valid and nonfinite==0 and exact>=28 and acc>=.99 and ab['correct']>max(ab['zero'],ab['mismatch'])
 (OUT/'initial_report.md').write_text(f'# Swap/Gibbs R-only initial report\n\nFixed RHODIN01, seed {SEED}, steps {MAX_STEPS}. Swap state has no MASK and all actions are no-op or same-element role swaps. Strong validity: {valid}; nonfinite: {nonfinite}; exact R: {exact}/32; mean final atom-role accuracy: {acc:.6f}; conditions: {ab}. **PASS={passed}**. Q→C was not run unless this is PASS; D2 is prohibited.\n')
 print(json.dumps({'exact_R':exact,'accuracy':acc,'valid':valid,'nonfinite':nonfinite,'condition':ab,'PASS':passed},indent=2))
if __name__=='__main__':main()
