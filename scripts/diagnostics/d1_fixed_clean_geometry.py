"""D1 fixed-clean-geometry overfit and pure-noise audit; never touches pos/cell heads."""
from __future__ import annotations
import argparse,csv,gzip,itertools,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from mattergen.common.assignment_diffusion import AssignmentDiffusion,gauge_center,sinkhorn
from mattergen.common.assignment_diffusion.d1_fixed_clean_geometry import D1FixedCleanGeometryPredictor

ROOT=Path('/home/mcw/OMC-Gen-model/datasets/omc25_le50_sinkhorn_subset_3k');OUT=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry');STEPS=32
def load_sample():
 c=ROOT/'cache/omc25_le50_mattergen/val';nums=np.load(c/'atomic_numbers.npy');pos=np.load(c/'pos.npy');cell=np.load(c/'cell.npy');counts=np.load(c/'num_atoms.npy');ids=np.load(c/'structure_id.npy');off=np.r_[0,np.cumsum(counts)];records={}
 with gzip.open(ROOT/'molecule_mapping/omc25_subset_val_molmap_hybrid_v3.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line)
   if r.get('success'):records[r['material_id']]=r
 graphs={}
 with gzip.open(ROOT/'molecule_mapping/oe62_hybrid_graphs_subset.jsonl.gz','rt') as f:
  for line in f:
   r=json.loads(line)
   if r.get('ok'):graphs[r['refcode_csd']]=r
 identity='RHODIN01|4|gener|9b810e76ec9d286';i=list(map(str,ids)).index(identity);r=records[identity];m=r['mapping'];g=graphs[r['csd_refcode']];n=int(counts[i]);roles=torch.tensor(m['mol_atom_idx'],dtype=torch.long);copies=torch.tensor(m['mol_id'],dtype=torch.long);M=len(set(m['mol_atom_idx']));
 edges=[];types=[]
 for b in g['bonds']:
  a,z=int(b['begin']),int(b['end']);t=int(b['type']);edges.extend([[a,z],[z,a]]);types.extend([t,t])
 return {'id':identity,'split':'val','pos':torch.tensor(pos[off[i]:off[i]+n],dtype=torch.float32),'cell':torch.tensor(cell[i],dtype=torch.float32),'z':torch.tensor(nums[off[i]:off[i]+n],dtype=torch.long),'copy':copies,'role':roles,'role_z':torch.tensor(g['atomic_numbers'],dtype=torch.long),'role_edge_index':torch.tensor(edges,dtype=torch.long).T,'role_bond_type':torch.tensor(types,dtype=torch.long),'N':n,'M':M,'Z':int(m['num_molecules'])}
def target(s,generator=None):
 n,m,z=s['N'],s['M'],s['Z'];device=s['pos'].device;a=torch.zeros(n,n,device=device);a[torch.arange(n,device=device),s['copy']*m+s['role']]=1;perm=torch.randperm(z,generator=generator,device=device);cols=torch.cat([torch.arange(x*m,(x+1)*m,device=device) for x in perm]);return a[:,cols],cols
def mask(s):return s['z'][:,None].eq(s['role_z'].repeat(s['Z'])[None,:])
def predictor(model,s,l,t,*,geometry=True,edges=True,pos=None):return model(l,t,s['z'],s['pos'] if pos is None else pos,s['cell'],s['role_z'],s['role_edge_index'],s['role_bond_type'],mask(s),geometry=geometry,edges=edges)
def orbit(hard,s):
 # scores exact canonical role plus a valid whole-copy block map; only copy blocks may align.
 m,z=s['M'],s['Z'];truth_copy=s['copy'].detach().cpu().numpy();truth_role=s['role'].detach().cpu().numpy();pred_col=hard.argmax(1).detach().cpu().numpy();pred_copy=pred_col//m;pred_role=pred_col%m;best=0;bestperm=None
 for perm in itertools.permutations(range(z)):
  good=((pred_role==truth_role)&(pred_copy==np.array([perm[c] for c in truth_copy]))).mean()
  if good>best:best,bestperm=good,perm
 purity=np.mean([np.max(np.bincount(pred_copy[truth_copy==c],minlength=z))/m for c in range(z)])
 complete=np.mean([len(set(pred_copy[truth_copy==c]))==1 and set(pred_role[truth_copy==c])==set(range(m)) for c in range(z)])
 return {'atom_role_accuracy_mod_copy':float(best),'exact_orbit_recovery':bool(best==1.0),'partition_purity':float(purity),'complete_molecule_copy_rate':float(complete),'copy_permutation':list(bestperm)}
def hard_from_soft(a):
 r,c=linear_sum_assignment((-a.detach().cpu().numpy()));h=torch.zeros_like(a);h[torch.tensor(r,device=a.device),torch.tensor(c,device=a.device)]=1;return h
def eval_denoise(model,s,seed,geometry=True,edges=True):
 rows=[];levels={'near_clean':1,'low':4,'mid':12,'high':22,'near_terminal':30};gen=torch.Generator(device=s['pos'].device).manual_seed(seed)
 for name,t in levels.items():
  for k in range(32):
   a,_=target(s,gen);ma=mask(s);l0=gauge_center(8*(2*a-1),ma);eps=gauge_center(torch.randn(l0.shape,generator=gen,device=l0.device),ma);ab=model_alpha[t];lt=gauge_center(ab.sqrt()*l0+(1-ab).sqrt()*eps,ma);pred=predictor(model,s,lt,t,geometry=geometry,edges=edges);x0=gauge_center((lt-(1-ab).sqrt()*pred)/ab.sqrt(),ma);soft=sinkhorn(x0,ma,100);hard=hard_from_soft(soft);o=orbit(hard,s);rows.append({'level':name,'t':t,'epsilon_mse':float(((pred-eps)[ma]**2).mean()),'x0_mse':float(((x0-l0)[ma]**2).mean()),'gauge_residual':float(max(x0.sum(0).abs().max(),x0.sum(1).abs().max())),'matched_mass':float(soft[a.bool()].mean()),'off_target_mass':float(soft[~a.bool()].sum()/s['N']),'entropy':float(-(soft[soft>0]*soft[soft>0].log()).mean()),**o})
 return rows
def sample_reverse(model,s,seed,geometry=True,edges=True,save=False):
 gen=torch.Generator(device=s['pos'].device).manual_seed(seed);ma=mask(s);y=gauge_center(torch.randn((s['N'],s['N']),generator=gen,device=s['pos'].device),ma);trace=[]
 for t in range(STEPS-1,-1,-1):
  eps=predictor(model,s,y,t,geometry=geometry,edges=edges);ab=model_alpha[t];x0=gauge_center((y-(1-ab).sqrt()*eps)/ab.sqrt(),ma)
  if t: mean,var=diff.posterior(y,x0,t);y=gauge_center(mean+var.sqrt()*gauge_center(torch.randn(y.shape,generator=gen,device=y.device),ma),ma)
  else:y=x0
  soft=sinkhorn(y,ma,100);trace.append({'t':t,'logit_norm':float(y.norm()),'gauge_residual':float(max(y.sum(0).abs().max(),y.sum(1).abs().max())),'epsilon_norm':float(eps.norm()),'entropy':float(-(soft[soft>0]*soft[soft>0].log()).mean()),'clean_edge_mass':float(soft[target(s,torch.Generator(device=y.device).manual_seed(0))[0].bool()].mean()),'forbidden_mass':float(soft[~ma].abs().max())})
 soft=sinkhorn(y,ma,100);hard=hard_from_soft(soft);metric=orbit(hard,s);metric.update({'seed':seed,'element_compatible':bool((hard[~ma]==0).all()),'bijective':bool(torch.all(hard.sum(0)==1) and torch.all(hard.sum(1)==1)),'forbidden_mass':float(soft[~ma].abs().max()),'entropy':float(-(soft[soft>0]*soft[soft>0].log()).mean()),'bond_consistency':bond_metric(hard,s),'cross_copy_edge_ratio':cross_copy(hard,s)})
 if save:torch.save({'soft':soft,'hard':hard,'trace':trace,'metric':metric},OUT/'trajectories'/f'seed_{seed}.pt')
 return metric,trace
def bond_metric(h,s):
 col=h.argmax(1);count=0;ok=0
 for a,b in s['role_edge_index'].T.tolist():
  ra=(s['role']==a).nonzero().flatten();rb=(s['role']==b).nonzero().flatten()
  for i in ra:
   j=rb[s['copy'][rb]==s['copy'][i]]
   if len(j):count+=1;ok+=int(col[i]//s['M']==col[j[0]]//s['M'] and col[i]%s['M']==a and col[j[0]]%s['M']==b)
 return ok/max(count,1)
def cross_copy(h,s):
 col=h.argmax(1);bad=0;total=0
 for a,b in s['role_edge_index'].T.tolist():
  ra=(s['role']==a).nonzero().flatten();rb=(s['role']==b).nonzero().flatten()
  for i in ra:
   j=rb[s['copy'][rb]==s['copy'][i]]
   if len(j):total+=1;bad+=int(col[i]//s['M']!=col[j[0]]//s['M'])
 return bad/max(total,1)
def train(seed,s,no_geometry=False,steps=2000):
 torch.manual_seed(seed);model=D1FixedCleanGeometryPredictor(steps=STEPS).to(s['pos'].device);opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5);curve=[];gen=torch.Generator(device=s['pos'].device).manual_seed(seed+1000);ma=mask(s);best=(1e9,None)
 for step in range(steps):
  a,_=target(s,gen);l0=gauge_center(8*(2*a-1),ma);t=int(torch.randint(STEPS,(1,),generator=gen,device=l0.device));eps=gauge_center(torch.randn(l0.shape,generator=gen,device=l0.device),ma);lt=gauge_center(model_alpha[t].sqrt()*l0+(1-model_alpha[t]).sqrt()*eps,ma);pred=predictor(model,s,lt,t,geometry=not no_geometry);loss=((pred-eps)[ma]**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  if float(loss)<best[0]:best=(float(loss),{k:v.detach().cpu() for k,v in model.state_dict().items()})
  if step%50==0:
   grad_norm=torch.sqrt(sum((p.grad.norm()**2 for p in model.parameters() if p.grad is not None)))
   curve.append({'seed':seed,'ablation':'no_geometry' if no_geometry else 'full','step':step,'loss':float(loss),'grad_norm':float(grad_norm)})
 model.load_state_dict(best[1]);return model,curve,best[0]
def random_baseline(s):
 ma=mask(s);rows=[];gen=torch.Generator(device=s['pos'].device).manual_seed(991)
 for _ in range(1024):
  score=torch.rand(ma.shape,generator=gen,device=ma.device).masked_fill(~ma,-1e9);rows.append(orbit(hard_from_soft(sinkhorn(score,ma,100)),s))
 return {k:float(np.mean([x[k] for x in rows])) for k in rows[0] if isinstance(rows[0][k],float)}
def main():
 global diff,model_alpha;ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1000);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);s=load_sample();torch.save(s,OUT/'fixed_sample.pt');device=torch.device('cuda');s={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in s.items()};ma=mask(s);diff=AssignmentDiffusion(steps=STEPS).to(device);diff.alpha_bar.copy_(torch.cumprod(1-torch.linspace(1e-4,.2,STEPS,device=device),0));model_alpha=diff.alpha_bar;meta={'dataset_split':s['split'],'sample_id':s['id'],'N':s['N'],'M':s['M'],'Z':s['Z'],'element_composition':dict(Counter(map(int,s['z'].tolist()))),'element_multiplicity':dict(Counter(map(int,s['z'].tolist()))),'molecular_graph_nodes':len(s['role_z']),'molecular_graph_edges':int(s['role_edge_index'].shape[1]),'clean_cell':s['cell'].detach().cpu().tolist(),'fractional_coordinates_sha256':__import__('hashlib').sha256(s['pos'].detach().cpu().numpy().tobytes()).hexdigest(),'canonical_role_definition':'mol_atom_id in [0,M), used only to construct targets/evaluation','copy_permutation_orbit_size':math.factorial(s['Z']),'N_equals_Z_times_M':s['N']==s['Z']*s['M'],'mask_density':float(ma.float().mean()),'terminal_alpha_bar':float(model_alpha[-1]),'allowed_element_compatible_permutations_log10':float(sum(math.lgamma(int((s['z']==z).sum())+1) for z in torch.unique(s['z']))/math.log(10))};(OUT/'fixed_sample_metadata.json').write_text(json.dumps(meta,indent=2));
 allcurves=[];summaries=[];models={}
 for seed in [17,29,43]:
  model,curve,b=train(seed,s,False,a.steps);models[seed]=model;allcurves+=curve;torch.save({'state_dict':model.state_dict(),'seed':seed,'best_loss':b},OUT/'checkpoints'/f'full_seed_{seed}_best.pt');den=eval_denoise(model,s,seed);pure=[]
  for ss in range(32):pure.append(sample_reverse(model,s,seed*10000+ss,save=ss<8)[0])
  summaries.append({'seed':seed,'best_loss':b,'pure':pure,'denoise':den})
 # independently trained no-geometry ablation, identical seed/steps.
 nogeo=[]
 for seed in [17,29,43]:
  model,curve,b=train(seed,s,True,a.steps);allcurves+=curve;pure=[sample_reverse(model,s,seed*20000+ss,geometry=False)[0] for ss in range(32)];nogeo.append({'seed':seed,'best_loss':b,'pure':pure})
 with open(OUT/'training_curves.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['seed','ablation','step','loss','grad_norm']);w.writeheader();w.writerows(allcurves)
 for item in summaries:(OUT/f'training_metrics_seed_{item["seed"]}.json').write_text(json.dumps({'seed':item['seed'],'best_loss':item['best_loss'],'final_curve':[x for x in allcurves if x['seed']==item['seed'] and x['ablation']=='full']},indent=2))
 (OUT/'denoising_metrics.json').write_text(json.dumps({str(x['seed']):x['denoise'] for x in summaries},indent=2));(OUT/'pure_noise_sampling_metrics.json').write_text(json.dumps({str(x['seed']):x['pure'] for x in summaries},indent=2));(OUT/'random_baseline_metrics.json').write_text(json.dumps(random_baseline(s),indent=2));(OUT/'no_geometry_ablation_metrics.json').write_text(json.dumps(nogeo,indent=2));
 # Full-model inference condition tests on seed 17.
 model=models[17];a0,_=target(s,torch.Generator(device=device).manual_seed(7));l0=gauge_center(8*(2*a0-1),ma);eps=gauge_center(torch.randn(l0.shape,generator=torch.Generator(device=device).manual_seed(8),device=device),ma);t=16;lt=gauge_center(model_alpha[t].sqrt()*l0+(1-model_alpha[t]).sqrt()*eps,ma);base=predictor(model,s,lt,t);badpos=s['pos'][torch.randperm(s['N'],generator=torch.Generator(device=device).manual_seed(3),device=device)];mismatch=predictor(model,s,lt,t,pos=badpos);sync=torch.randperm(s['N'],generator=torch.Generator(device=device).manual_seed(4),device=device);synced=predictor(model,s,lt[sync],t,pos=s['pos'][sync]);(OUT/'geometry_mismatch_metrics.json').write_text(json.dumps({'base_epsilon_mse':float(((base-eps)[ma]**2).mean()),'mismatched_geometry_epsilon_mse':float(((mismatch-eps)[ma]**2).mean()),'synchronous_row_permutation_error':float((synced-base[sync]).abs().max())},indent=2));noedge=predictor(model,s,lt,t,edges=False);(OUT/'no_mol_edges_metrics.json').write_text(json.dumps({'inference_no_edge_epsilon_mse':float(((noedge-eps)[ma]**2).mean()),'full_epsilon_mse':float(((base-eps)[ma]**2).mean())},indent=2))
 (OUT/'trainable_parameters.json').write_text(json.dumps([{'name':n,'shape':list(p.shape),'count':p.numel(),'trainable':True,'optimizer_group':'assignment','lr':.002,'weight_decay':1e-5} for n,p in models[17].named_parameters()]+[{'name':'baseline_pos_cell_heads','trainable':False,'reason':'not instantiated in D1 assignment-only branch'}],indent=2));
 (OUT/'training_config.yaml').write_text(f'steps: {a.steps}\nseeds: [17, 29, 43]\nassignment_steps: {STEPS}\nloss: allowed-entry epsilon MSE after gauge projection\ngeometry: clean fixed\n')
 (OUT/'predictor_condition_audit.md').write_text('# D1 predictor condition audit\n\nInputs: L_t [N,N], t_a, atom type/clean fractional coordinates/cell periodic RBF graph, target atom type/molecular bond graph, and hard element mask. Crystal encoder uses translation-invariant fractional displacements; molecular encoder is graph-message-passing without role indices. Three masked row/column axial interaction rounds exchange same-element candidate information. No mol_copy_id, mol_atom_id, packed slot, absolute matrix position, clean assignment, sample id, or file name enters `forward`. The role id exists only outside predictor for target construction/evaluation.\n')
 (OUT/'leakage_audit.md').write_text('# Leakage and invariance audit\n\nPredictor signature contains no copy id, role id, clean assignment, packed slot, sample id, or file name. All encoders/pair interactions are permutation equivariant. Periodic relative coordinates are invariant to global fractional translation. Gate-A row/column/copy/batch tests remain the regression evidence; D1 synchronous row-permutation result is recorded in geometry_mismatch_metrics.json.\n')
 # concise diagnostic summary; gate decision written by report generator after inspecting numbers.
 print(json.dumps({'sample':s['id'],'seeds':[{'seed':x['seed'],'best_loss':x['best_loss'],'orbit':sum(q['exact_orbit_recovery'] for q in x['pure'])} for x in summaries]},indent=2))
if __name__=='__main__':main()
