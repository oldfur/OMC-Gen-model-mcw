import json,torch
from pathlib import Path
from mattergen.common.data.molecule_dataset import MoleculeMappedCrystalDataset
from mattergen.common.data.collate import collate
from mattergen.common.assignment_diffusion import AssignmentDiffusion
root=Path('subsets/omc25_le50_sinkhorn_subset_3k');ds=MoleculeMappedCrystalDataset.from_cache_path(cache_path=str(root/'cache/omc25_le50_mattergen/val'),mapping_jsonl=str(root/'molecule_mapping/omc25_subset_val_molmap_hybrid_v3.jsonl.gz'),smiles_graph_jsonl=str(root/'molecule_mapping/oe62_hybrid_graphs_subset.jsonl.gz'),strict_transfer_mode='rdkit_explicit_h_full_match',properties=[])
sel=[]; representatives=[]; seen=set()
for i in range(len(ds)):
 x=ds[i]; signature=(int(x.num_atoms.item()), int(x.mol_num_molecules[0]))
 if int(x.mol_num_molecules[0]) >= 2:
  if signature not in seen: representatives.append(x); seen.add(signature)
  sel.append(x)
 if len(representatives) >= 20 and len(sel) >= 32: break
if len(sel) < 32: raise RuntimeError(f'need 32 compatible samples, found {len(sel)}')
# Ensure the first batch set covers every available (N,Z) signature before
# filling remaining positions from the same real validation split.
sel=(representatives + sel)[:32]
d=AssignmentDiffusion(steps=10);opt=torch.optim.Adam(d.parameters(),lr=1e-4);rows=[]; losses=[]; errors=[]
for start in range(0,32,2):
 batch_samples=sel[start:start+2]; b=collate(batch_samples); t=torch.tensor([.1,.3]); opt.zero_grad();loss,err=d.loss(b,t);loss.backward();opt.step();losses.append(float(loss));errors.append(float(err));batch=b.get_batch_idx('pos')
 for j,x in enumerate(batch_samples):
  idx=(batch==j).nonzero().flatten();a,z,zr,_=d._target(b,idx);m=z[:,None].eq(zr[None,:]); l0=d.kappa*(2*a-1); from mattergen.common.assignment_diffusion import gauge_center,sinkhorn; clean=sinkhorn(gauge_center(l0,m),m,d.iters,d.tol)
  rows.append({'batch':start//2,'id':str(x.structure_id),'N':int(idx.numel()),'M':int(torch.unique(b.mol_atom_id[idx]).numel()),'Z':int(x.mol_num_molecules[0]),'element_counts':{str(int(q)):int((z==q).sum()) for q in torch.unique(z)},'shape':list(a.shape),'mask_density':float(m.float().mean()),'a0_row_error':float((a.sum(1)-1).abs().max()),'a0_col_error':float((a.sum(0)-1).abs().max()),'clean_sinkhorn_error':float((clean.sum(1)-1).abs().max().maximum((clean.sum(0)-1).abs().max())),'clean_matched_probability':float(clean[a.bool()].mean()),'off_target_mass':float(clean[~a.bool()].sum()/int(idx.numel())),'entropy':float(-(clean[clean>0]*clean[clean>0].log()).mean())})
o=Path('outputs/assignment_diffusion_mvp');o.mkdir(parents=True,exist_ok=True);o.joinpath('d0_real_metrics.json').write_text(json.dumps({'batches':16,'batch_size':2,'loss_mean':sum(losses)/len(losses),'losses':losses,'max_marginal_error':max(errors),'samples':rows,'finite_gradients':all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in d.parameters())},indent=2));print(o/'d0_real_metrics.json')
