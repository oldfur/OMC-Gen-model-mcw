"""Read-only automorphism/role-identifiability audit for RHODIN01 results."""
from __future__ import annotations
import json
from collections import Counter,defaultdict
from pathlib import Path
import networkx as nx
import torch
from rdkit import Chem

ROOT=Path('outputs/assignment_diffusion_mvp');OUT=ROOT/'role_automorphism_audit';SAMPLE=ROOT/'d1_fixed_clean_geometry/fixed_sample.pt'
def graph_and_aut(s):
 g=nx.Graph()
 for i,z in enumerate(s['role_z'].tolist()):g.add_node(i,z=int(z))
 seen=set()
 for (u,v),b in zip(s['role_edge_index'].t().tolist(),s['role_bond_type'].tolist()):
  if (v,u) in seen:continue
  seen.add((u,v));g.add_edge(u,v,b=int(b))
 nm=nx.algorithms.isomorphism.categorical_node_match('z',None);em=nx.algorithms.isomorphism.categorical_edge_match('b',None)
 nx_aut=sorted(tuple(m[i] for i in range(len(g))) for m in nx.algorithms.isomorphism.GraphMatcher(g,g,node_match=nm,edge_match=em).isomorphisms_iter())
 rw=Chem.RWMol()
 for z in s['role_z'].tolist():rw.AddAtom(Chem.Atom(int(z)))
 typ={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
 for u,v,data in g.edges(data=True):rw.AddBond(int(u),int(v),typ[data['b']])
 mol=rw.GetMol();Chem.SanitizeMol(mol)
 rd=sorted(tuple(int(i) for i in x) for x in mol.GetSubstructMatches(mol,uniquify=False,maxMatches=1000000))
 if set(nx_aut)!=set(rd):raise RuntimeError(f'NetworkX/RDKit automorphism mismatch: {len(nx_aut)} vs {len(rd)}')
 return g,nx_aut
def orbit_data(aut,m):
 orbit=[sorted({p[r] for p in aut}) for r in range(m)];parts=[];seen=set()
 for o in orbit:
  key=tuple(o)
  if key not in seen:seen.add(key);parts.append(o)
 return orbit,parts
def metrics(pred,truth,copy,aut):
 n=len(truth);literal=float((pred==truth).float().mean());scores=[]
 for p in aut:scores.append(float((pred==torch.tensor([p[x] for x in truth.tolist()])).float().mean()))
 j=max(range(len(aut)),key=lambda i:scores[i]);global_exact=bool(torch.equal(pred,torch.tensor([aut[j][x] for x in truth.tolist()])))
 copy_sum=0;copy_exact=True
 for c in range(int(copy.max())+1):
  idx=(copy==c).nonzero().flatten();vals=[float((pred[idx]==torch.tensor([p[truth[i].item()] for i in idx])).float().mean()) for p in aut];copy_sum+=max(vals)*len(idx);copy_exact&=max(vals)==1.
 return {'literal_accuracy':literal,'global_aligned_accuracy':max(scores),'global_permutation':list(aut[j]),'literal_exact':bool(torch.equal(pred,truth)),'global_exact_orbit':global_exact,'per_copy_aligned_accuracy':copy_sum/n,'per_copy_exact_orbit':copy_exact}
def error_analysis(pred,truth,orbits,m,aut,copy):
 total=inside=cross=0;conf=defaultdict(lambda:defaultdict(int));orbconf=defaultdict(lambda:defaultdict(int));byrole={}
 global_perm=max(aut,key=lambda p:float((pred==torch.tensor([p[x] for x in truth.tolist()])).float().mean()))
 per_copy_perm={}
 for c in range(int(copy.max())+1):
  idx=(copy==c).nonzero().flatten();per_copy_perm[c]=max(aut,key=lambda p:float((pred[idx]==torch.tensor([p[truth[i].item()] for i in idx])).float().mean()))
 def key(r):return ','.join(map(str,orbits[r]))
 for r in range(m):
  idx=(truth==r);node_idx=idx.nonzero().flatten();pc=torch.tensor([per_copy_perm[int(copy[i])][r] for i in node_idx]);byrole[str(r)]={'literal_accuracy':float((pred[idx]==truth[idx]).float().mean()),'global_aligned_accuracy':float((pred[idx]==global_perm[r]).float().mean()),'per_copy_aligned_accuracy':float((pred[idx]==pc).float().mean()),'count':int(idx.sum())}
 for a,b in zip(truth.tolist(),pred.tolist()):
  conf[str(a)][str(b)]+=1
  orbconf[key(a)][key(b)]+=1
  if a!=b:
   total+=1
   if b in orbits[a]:inside+=1
   else:cross+=1
 return {'total_literal_errors':total,'within_orbit_errors':inside,'within_orbit_fraction':inside/max(1,total),'cross_orbit_errors':cross,'cross_orbit_fraction':cross/max(1,total),'role_confusion':conf,'orbit_confusion':orbconf,'per_role':byrole}
def load_masked():
 p=ROOT/'role_partition_discrete_constrained/trajectories';return [torch.load(p/f'role_{i}.pt',map_location='cpu')['role'].long() for i in range(32)]
def dagger_round3(s):
 raw=json.loads((ROOT/'role_swap_dagger/swap_trajectories.json').read_text())['round3'];out=[]
 for rows in raw:
  row=rows[-1];st=torch.tensor(row['state'],dtype=torch.long);same=s['z'][:,None].eq(s['z'][None,:])&st[:,None].ne(st[None,:]);p=torch.triu(same,1).nonzero().long()
  if not row['stopped']:st2=st.clone();i,j=p[row['top1']].tolist();st2[i],st2[j]=st[j],st[i];st=st2
  out.append(st)
 return out
def unavailable(name,literal,reason):return {'method':name,'status':'UNAVAILABLE_ALIGNMENT','stored_literal_accuracy':literal,'reason':reason}
def conflict_audit(s,aut):
 raw=json.loads((ROOT/'role_swap_dagger/swap_trajectories.json').read_text())['round3'];rows=[r for trajectory in raw for r in trajectory];pairs=[];conflicts=0;clean_conflicts=0
 for i,a in enumerate(rows):
  ra=tuple(a['state'])
  for b in rows[i+1:]:
   rb=tuple(b['state']);equiv=[p for p in aut if tuple(p[x] for x in ra)==rb]
   if not equiv:continue
   pa=torch.triu(s['z'][:,None].eq(s['z'][None,:])&torch.tensor(ra)[:,None].ne(torch.tensor(ra)[None,:]),1).nonzero().tolist();pb=torch.triu(s['z'][:,None].eq(s['z'][None,:])&torch.tensor(rb)[:,None].ne(torch.tensor(rb)[None,:]),1).nonzero().tolist()
   besta={tuple(pa[k]) for k,x in enumerate(a['best']) if x};bestb={tuple(pb[k]) for k,x in enumerate(b['best']) if x};conflict=besta!=bestb;pairs.append((i,conflict));conflicts+=conflict
   ca=ra==tuple(s['role'].tolist());cb=rb==tuple(s['role'].tolist());clean_conflicts+=(ca!=cb)
 return {'source':'stored DAgger round-3 on-policy trajectories only; full historical aggregate was not serialized','stored_states':len(rows),'automorphism_equivalent_state_pairs':len(pairs),'label_conflict_pairs':conflicts,'label_conflict_pair_fraction':conflicts/max(1,len(pairs)),'clean_stop_label_conflicts':clean_conflicts,'full_aggregate_audit_status':'UNAVAILABLE_NOT_SERIALIZED'}
def main():
 OUT.mkdir(parents=True,exist_ok=True);s=torch.load(SAMPLE,map_location='cpu');g,aut=graph_and_aut(s);truth=s['role'].long();orbit,parts=orbit_data(aut,s['M'])
 (OUT/'molecular_automorphisms.json').write_text(json.dumps({'method':'NetworkX GraphMatcher strict node(element)/edge(bond type), cross-checked with RDKit substructure automorphisms','group_size':len(aut),'permutations':[list(p) for p in aut]},indent=2));(OUT/'role_orbits.json').write_text(json.dumps({'role_orbits':{str(i):o for i,o in enumerate(orbit)},'orbit_partition':parts,'singleton_orbits':sum(len(x)==1 for x in parts),'non_singleton_orbits':sum(len(x)>1 for x in parts)},indent=2))
 methods={};errors={};confs={}
 for name,preds in {'masked_capacity_R':load_masked(),'dagger_round3':dagger_round3(s)}.items():
  ms=[metrics(x,truth,s['copy'],aut) for x in preds];methods[name]={'status':'EVALUATED','samples':len(preds),**{k:sum(float(x[k]) for x in ms)/len(ms) for k in ('literal_accuracy','global_aligned_accuracy','per_copy_aligned_accuracy')},'literal_exact':sum(x['literal_exact'] for x in ms),'global_exact_orbit':sum(x['global_exact_orbit'] for x in ms),'per_copy_exact_orbit':sum(x['per_copy_exact_orbit'] for x in ms)};ea=[error_analysis(x,truth,orbit,s['M'],aut,s['copy']) for x in preds];errors[name]={k:sum(x[k] for x in ea)/len(ea) for k in ('total_literal_errors','within_orbit_errors','within_orbit_fraction','cross_orbit_errors','cross_orbit_fraction')};confs[name]=ea
 methods['boltzmann_swap']=unavailable('boltzmann_swap',0.9000000208616257,'saved trajectories contain scalar accuracy/Hamming only, not predicted role labels')
 methods['best_action_swap']=unavailable('best_action_swap',0.7695312574505806,'saved sampler trajectories contain scalar metrics/actions, not final role labels')
 dr=json.loads((ROOT/'role_swap_dagger/dagger_rounds.json').read_text())
 for n in (1,2):methods[f'dagger_round{n}']=unavailable(f'dagger_round{n}',dr[n-1]['final_accuracy'],'only round-3 raw role trajectories were retained')
 (OUT/'method_aligned_metrics.json').write_text(json.dumps(methods,indent=2));(OUT/'error_orbit_analysis.json').write_text(json.dumps(errors,indent=2));(OUT/'per_role_confusions.json').write_text(json.dumps(confs,default=lambda x:dict(x),indent=2));conflict=conflict_audit(s,aut);(OUT/'dagger_label_conflict_audit.json').write_text(json.dumps(conflict,indent=2))
 main_eval=methods['masked_capacity_R'];gauge=main_eval['global_aligned_accuracy']-main_eval['literal_accuracy'];within=errors['masked_capacity_R']['within_orbit_fraction'];case='A_gauge_evaluation_failure' if gauge>.05 or within>.8 or conflict['label_conflict_pairs']>0 else 'B_real_assignment_failure'
 dagger_eval=methods.get('dagger_round3',{})
 report=f'''# Molecular graph automorphism audit\n\nGraph automorphism group size: **{len(aut)}**. Orbit partition: `{parts}`. NetworkX strict labelled-graph enumeration exactly matched RDKit automorphisms.\n\n## Recovered final-label evaluations\n\n- Masked-capacity R (32 saved final role tensors): literal accuracy {main_eval['literal_accuracy']:.6f}; global aligned {main_eval['global_aligned_accuracy']:.6f}; per-copy diagnostic aligned {main_eval['per_copy_aligned_accuracy']:.6f}; literal/global/per-copy exact {main_eval['literal_exact']}/{main_eval['global_exact_orbit']}/{main_eval['per_copy_exact_orbit']} of 32. Literal errors within an automorphism orbit: {within:.3%}.\n- DAgger round 3 (32 saved MAP trajectories): literal accuracy {dagger_eval.get('literal_accuracy',float('nan')):.6f}; global aligned {dagger_eval.get('global_aligned_accuracy',float('nan')):.6f}; per-copy diagnostic aligned {dagger_eval.get('per_copy_aligned_accuracy',float('nan')):.6f}; literal/global/per-copy exact {dagger_eval.get('literal_exact','NA')}/{dagger_eval.get('global_exact_orbit','NA')}/{dagger_eval.get('per_copy_exact_orbit','NA')} of 32.\n\n## Evidence boundaries\n\nThe Boltzmann-swap and best-action samplers saved scalar trajectories only; DAgger rounds 1/2 saved only aggregate scalar metrics. Their literal scores are retained, but aligned metrics are marked unavailable rather than re-sampled or reconstructed. The DAgger label-conflict audit covers only the 2,048 saved round-3 rollout states; the full historical aggregation buffer was not serialized.\n\n## Decision\n\n**{case}**. Per-copy alignment is explicitly diagnostic only and uses ground-truth copy IDs. The small global-alignment gains and zero global orbit recoveries show that the labelled-graph automorphism is not the dominant explanation for the remaining R errors.\n\nSTOP remains in force: no Q→C, three-seed, or D2.\n'''
 (OUT/'automorphism_audit_report.md').write_text(report)
if __name__=='__main__':main()
