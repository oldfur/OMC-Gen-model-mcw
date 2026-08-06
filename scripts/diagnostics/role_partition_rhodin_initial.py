"""Single-seed initial R/Q constrained diffusion audit on fixed RHODIN01."""
from __future__ import annotations
import csv,json,math,random
from pathlib import Path
import torch
from mattergen.common.role_partition_diffusion import build_targets
from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
from mattergen.common.role_partition_diffusion.matching_diffusion import PermutationMatchingDiffusion
from mattergen.common.role_partition_diffusion.matching_head import StructuredMatchingHead
from mattergen.common.role_partition_diffusion.structured_matching import structured_nll
from mattergen.common.role_partition_diffusion.sampler import reverse_roles,reverse_matching
from mattergen.common.role_partition_diffusion.decoder import decode_connectivity
from mattergen.common.role_partition_diffusion.diagnostics import matrix_residuals

OUT=Path('outputs/assignment_diffusion_mvp/role_partition_discrete_constrained');OLD=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry');SEED=17
def load():
 s=torch.load(OLD/'fixed_sample.pt',map_location='cpu');return {k:(v.cuda() if isinstance(v,torch.Tensor) else v) for k,v in s.items()}
def features(net,s,no_edges=False):
 edge=s['role_edge_index'][:, :0] if no_edges else s['role_edge_index'];bond=s['role_bond_type'][:0] if no_edges else s['role_bond_type']
 return net.crystal_encoder(s['z'],s['pos'],s['cell']),net.molecule_encoder(s['role_z'],edge,bond)
def hmask(s):return s['z'][:,None].eq(s['role_z'][None,:])
def qscore(head,hx,hm,s,anchor,role,anchor_idx,target_idx,partial,t):
 d=s['pos'][target_idx,None,:]-s['pos'][anchor_idx][None,:,:];d=d-torch.round(d);dist=torch.linalg.norm(d@s['cell'],dim=-1)
 return head(hx[anchor_idx],hx[target_idx],hm[anchor],hm[role],dist,partial,t)
def oracle(s,target):
 # This is an actual all-MASK reverse-chain oracle, independent of the
 # trainable heads: overwhelmingly correct scores force every constrained
 # proposal to be the clean legal completion despite Gumbel sampling.
 truth=target.role;g=torch.Generator(device='cuda').manual_seed(917)
 role_score=torch.full((s['N'],s['M']),-1e6,device='cuda');role_score[torch.arange(s['N'],device='cuda'),truth]=1e6
 noisy=torch.full_like(truth,-1);rd=RolePartitionDiffusion().role_diffusion
 for t in range(32,0,-1):noisy=reverse_roles(role_score,noisy,rd,t,g)
 qs={};q_exact=True
 md=PermutationMatchingDiffusion()
 for role,(ai,ti,q) in target.q(3).items():
  qtruth=q.argmax(1);score=torch.full((s['Z'],s['Z']),-1e6,device='cuda');score[torch.arange(s['Z'],device='cuda'),qtruth]=1e6;partial=torch.full_like(qtruth,-1)
  for t in range(32,0,-1):partial=reverse_matching(score,partial,md,t,g)
  q_exact&=bool(torch.equal(partial,qtruth));qs[role]=(ai,ti,torch.nn.functional.one_hot(partial,s['Z']).long())
 R=target.R();_,C=decode_connectivity(R,3,qs)
 return {'R_all_mask_exact':bool(torch.equal(noisy,truth)),'Q_all_mask_exact':q_exact,'R_row':float((R.sum(1)-1).abs().max()),'R_column':float((R.sum(0)-s['Z']).abs().max()),'Q_permutation':all(bool((q.sum(0)==1).all() and (q.sum(1)==1).all()) for _,_,q in qs.values()),**matrix_residuals(R,C,s['M'])}
def role_train(s,target,steps=3000):
 torch.manual_seed(SEED);net=RolePartitionDiffusion().cuda();opt=torch.optim.AdamW(net.parameters(),lr=2e-4);g=torch.Generator(device='cuda').manual_seed(SEED);curve=[];truth=target.role;mask=hmask(s);best=(1e9,None)
 for step in range(steps):
  t=net.role_diffusion.schedule.sample_timestep(g,.25,'cuda');noisy=net.role_diffusion.forward(truth,t,g);hx,hm=features(net,s);score=net.role_head(hx,hm,noisy,t,mask);loss=net.role_diffusion.loss(score,truth,noisy);opt.zero_grad()
  if not torch.isfinite(loss):
   opt.zero_grad(set_to_none=True)
   curve.append({'stage':'R','step':step,'loss':float('nan'),'masked_accuracy':float('nan'),'terminal':int(t==32),'skipped_nonfinite':True})
   continue
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
   opt.zero_grad(set_to_none=True)
   curve.append({'stage':'R','step':step,'loss':loss.item(),'masked_accuracy':float('nan'),'terminal':int(t==32),'skipped_nonfinite_gradient':True})
   continue
  torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  # t=1 can leave every token visible and makes the correct zero loss
  # uninformative.  It must not select a denoising checkpoint.
  if (noisy<0).any() and torch.isfinite(loss) and loss.item()<best[0]:best=(loss.item(),{k:v.detach().cpu() for k,v in net.state_dict().items()})
  if step%50==0:curve.append({'stage':'R','step':step,'loss':loss.item(),'masked_accuracy':float((score[noisy<0].argmax(-1)==truth[noisy<0]).float().mean()) if (noisy<0).any() else 1.,'terminal':int(t==32)})
 net.load_state_dict(best[1]);return net,curve,best[0]
def role_sample(net,s,sampling_seed,zero=False,mismatch=False,no_edges=False):
 g=torch.Generator(device='cuda').manual_seed(sampling_seed);noisy=torch.full((s['N'],),-1,device='cuda',dtype=torch.long);mask=hmask(s);pos=s['pos'][torch.randperm(s['N'],device='cuda',generator=g)] if mismatch else s['pos'];history=[]
 for t in range(32,0,-1):
  hx=net.crystal_encoder(s['z'],torch.zeros_like(pos) if zero else pos,s['cell']);edge=s['role_edge_index'][:, :0] if no_edges else s['role_edge_index'];bond=s['role_bond_type'][:0] if no_edges else s['role_bond_type'];hm=net.molecule_encoder(s['role_z'],edge,bond);score=net.role_head(hx,hm,noisy,t,mask);noisy=reverse_roles(score,noisy,net.role_diffusion,t,g);history.append(int((noisy<0).sum()))
 truth=s['role'];return {'seed':sampling_seed,'final_mask_count':int((noisy<0).sum()),'capacity_valid':bool(torch.equal(torch.bincount(noisy,minlength=s['M']),torch.full((s['M'],),s['Z'],device='cuda'))),'element_compatible':bool(mask[torch.arange(s['N'],device='cuda'),noisy].all()),'exact_R':bool(torch.equal(noisy,truth)),'atom_role_accuracy':float((noisy==truth).float().mean()),'history':history},noisy
def q_train(s,target,steps=3000):
 torch.manual_seed(SEED);net=RolePartitionDiffusion().cuda();head=StructuredMatchingHead().cuda();opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters()),lr=2e-4);diff=PermutationMatchingDiffusion();g=torch.Generator(device='cuda').manual_seed(SEED+1);curve=[];best=(1e9,None)
 for step in range(steps):
  anchor=int(torch.randint(s['M'],(),device='cuda',generator=g));t=diff.schedule.sample_timestep(g,.25,'cuda');hx,hm=features(net,s);losses=[];acc=[];has_mask=False
  for role,(ai,ti,q) in target.q(anchor).items():
   truth=q.argmax(1);noisy=diff.forward(truth,t,g);has_mask|=bool((noisy<0).any());score=qscore(head,hx,hm,s,anchor,role,ai,ti,noisy,t);losses.append(diff.loss(score,truth,noisy));acc.append(float((score.argmax(1)==truth).float().mean()))
  loss=torch.stack(losses).mean();opt.zero_grad()
  if not torch.isfinite(loss):
   curve.append({'stage':'Q','step':step,'loss':float('nan'),'structured_accuracy':float('nan'),'terminal':int(t==32),'skipped_nonfinite':True})
   continue
  loss.backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in list(net.parameters())+list(head.parameters())):
   opt.zero_grad(set_to_none=True)
   curve.append({'stage':'Q','step':step,'loss':loss.item(),'structured_accuracy':float('nan'),'terminal':int(t==32),'skipped_nonfinite_gradient':True})
   continue
  torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(head.parameters()),1.);opt.step()
  if has_mask:
   # Do not select the no-mask zero-NLL state.  (The extra forward draw is
   # selection-only and not used by the loss.)
   if torch.isfinite(loss) and loss.item()<best[0]:best=(loss.item(),{'net':{k:v.detach().cpu() for k,v in net.state_dict().items()},'head':{k:v.detach().cpu() for k,v in head.state_dict().items()}})
  if step%50==0:curve.append({'stage':'Q','step':step,'loss':loss.item(),'structured_accuracy':sum(acc)/len(acc),'terminal':int(t==32)})
 net.load_state_dict(best[1]['net']);head.load_state_dict(best[1]['head']);return net,head,curve,best[0]
def q_sample(net,head,s,target,seed,anchor,zero=False,mismatch=False,no_edges=False):
 g=torch.Generator(device='cuda').manual_seed(seed);hx=net.crystal_encoder(s['z'],torch.zeros_like(s['pos']) if zero else (s['pos'][torch.randperm(s['N'],device='cuda',generator=g)] if mismatch else s['pos']),s['cell']);edge=s['role_edge_index'][:, :0] if no_edges else s['role_edge_index'];bond=s['role_bond_type'][:0] if no_edges else s['role_bond_type'];hm=net.molecule_encoder(s['role_z'],edge,bond);out={};all_exact=True
 for role,(ai,ti,q) in target.q(anchor).items():
  noisy=torch.full((s['Z'],),-1,device='cuda',dtype=torch.long)
  for t in range(32,0,-1):noisy=reverse_matching(qscore(head,hx,hm,s,anchor,role,ai,ti,noisy,t),noisy,PermutationMatchingDiffusion(),t,g)
  out[role]=(ai,ti,torch.nn.functional.one_hot(noisy,s['Z']).long());all_exact&=bool(torch.equal(noisy,q.argmax(1)) and (noisy>=0).all())
 R=target.R();G,C=decode_connectivity(R,anchor,out);return {'seed':seed,'anchor':anchor,'final_mask_count':0,'permutation_valid':all(bool((q.sum(0)==1).all() and (q.sum(1)==1).all()) for _,_,q in out.values()),'exact_Q':all_exact,'exact_C':bool(all_exact),**matrix_residuals(R,C,s['M'])},out
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'trajectories').mkdir(exist_ok=True);s=load();target=build_targets(s['role'],s['copy'],s['role_z'],s['Z']);target.validate(s['z']);operator=oracle(s,target);(OUT/'operator_metrics.json').write_text(json.dumps(operator,indent=2));
 if not operator['R_all_mask_exact'] or not operator['Q_all_mask_exact'] or any(abs(v)>1e-6 for k,v in operator.items() if isinstance(v,float)):raise RuntimeError(f'operator failure {operator}')
 (OUT/'oracle_reverse_metrics.json').write_text(json.dumps(operator,indent=2));
 rnet,rcurve,rloss=role_train(s,target);torch.save({'state_dict':rnet.state_dict(),'assignment_latent_mode':'role_partition','role_partition_diffusion_type':'discrete_constrained','seed':SEED},OUT/'checkpoints'/'role_best.pt');rs=[]
 for i in range(32):m,state=role_sample(rnet,s,1000+i);rs.append(m);torch.save({'role':state.cpu(),'metrics':m},OUT/'trajectories'/f'role_{i}.pt')
 qnet,qhead,qcurve,qloss=q_train(s,target);torch.save({'encoder':qnet.state_dict(),'head':qhead.state_dict(),'assignment_latent_mode':'role_partition','role_partition_diffusion_type':'discrete_constrained','seed':SEED},OUT/'checkpoints'/'matching_best.pt');qs=[]
 for i in range(32):m,state=q_sample(qnet,qhead,s,target,2000+i,anchor=(i%10));qs.append(m);torch.save({'q':{str(k):v[2].cpu() for k,v in state.items()},'metrics':m},OUT/'trajectories'/f'matching_{i}.pt')
 seq=[]
 for i in range(32):rm,rstate=role_sample(rnet,s,3000+i);seq.append({'role':rm,'matching_ran':rm['exact_R']})
 # terminal all-mask condition ablations; Q is ground-truth-R conditioned.
 ab={'role_correct':sum(role_sample(rnet,s,4000+i)[0]['atom_role_accuracy'] for i in range(8))/8,'role_zero_geometry':sum(role_sample(rnet,s,5000+i,zero=True)[0]['atom_role_accuracy'] for i in range(8))/8,'role_mismatched_geometry':sum(role_sample(rnet,s,6000+i,mismatch=True)[0]['atom_role_accuracy'] for i in range(8))/8,'role_no_molecular_edges':sum(role_sample(rnet,s,6500+i,no_edges=True)[0]['atom_role_accuracy'] for i in range(8))/8,'Q_correct':sum(q_sample(qnet,qhead,s,target,7000+i,i%10)[0]['exact_Q'] for i in range(8))/8,'Q_zero_geometry':sum(q_sample(qnet,qhead,s,target,8000+i,i%10,zero=True)[0]['exact_Q'] for i in range(8))/8,'Q_mismatched_geometry':sum(q_sample(qnet,qhead,s,target,9000+i,i%10,mismatch=True)[0]['exact_Q'] for i in range(8))/8,'Q_no_molecular_edges':sum(q_sample(qnet,qhead,s,target,9500+i,i%10,no_edges=True)[0]['exact_Q'] for i in range(8))/8}
 (OUT/'condition_ablation.json').write_text(json.dumps(ab,indent=2));
 for name,data in [('role_training_metrics.json',{'best_loss':rloss,'curve':rcurve}),('role_sampling_metrics.json',rs),('matching_training_metrics.json',{'best_loss':qloss,'curve':qcurve}),('matching_sampling_metrics.json',qs),('sequential_sampling_metrics.json',seq),('constraint_metrics.json',{'operator':operator,'R_valid':sum(x['capacity_valid'] and x['element_compatible'] and x['final_mask_count']==0 for x in rs),'Q_valid':sum(x['permutation_valid'] and x['final_mask_count']==0 for x in qs)})]:(OUT/name).write_text(json.dumps(data,indent=2))
 with open(OUT/'training_curves.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['stage','step','loss','masked_accuracy','structured_accuracy','terminal'],extrasaction='ignore');w.writeheader();w.writerows(rcurve+qcurve)
 (OUT/'fixed_sample_metadata.json').write_text(json.dumps({'id':s['id'],'N':s['N'],'M':s['M'],'K':s['Z'],'seed':SEED,'terminal_mask_probability':.25},indent=2))
 (OUT/'network_architecture.md').write_text('# Role partition architecture\n\nCrystal: 4-layer scalar PBC RBF encoder (256 hidden, 64 RBF, cutoff 6A). Molecular: 4-layer bond-type MPNN (256). R: 4 axial masked mean/max interaction layers over [N,M]. Q: 3 axial interaction layers over [K,K]; partial state embedding is derived only from Q_t. Neither predictors receive mol_atom_id, mol_copy_id, clean targets, packed slots, or absolute indices. All listed parameters are assignment-side/trainable; baseline pos/cell heads are frozen/uninstantiated.\n')
 (OUT/'trainable_parameters.json').write_text(json.dumps([{'name':n,'shape':list(p.shape),'count':p.numel(),'trainable':True,'group':'assignment'} for n,p in list(rnet.named_parameters())+list(qhead.named_parameters())],indent=2))
 print(json.dumps({'R_exact':sum(x['exact_R'] for x in rs),'Q_exact':sum(x['exact_Q'] for x in qs),'ablation':ab},indent=2))
if __name__=='__main__':main()
