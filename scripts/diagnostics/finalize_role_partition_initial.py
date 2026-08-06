from __future__ import annotations
import hashlib,json,csv
from pathlib import Path
import math
import torch
from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
from mattergen.common.role_partition_diffusion.matching_head import StructuredMatchingHead
P=Path('outputs/assignment_diffusion_mvp/role_partition_discrete_constrained');OLD=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry')
def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 role=json.loads((P/'role_sampling_metrics.json').read_text());match=json.loads((P/'matching_sampling_metrics.json').read_text());seq=json.loads((P/'sequential_sampling_metrics.json').read_text());op=json.loads((P/'operator_metrics.json').read_text());
 baseline={'commit':Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/gate_a_commit.txt').read_text().strip(),'method':'full_assignment continuous Gaussian diffusion','gate_b':'FAIL','metrics_source':str(OLD/'gate_b_report.md'),'pure_noise_metrics':json.loads((OLD/'pure_noise_sampling_metrics.json').read_text()),'sha256':sha(OLD/'pure_noise_sampling_metrics.json')}
 Path('outputs/assignment_diffusion_mvp/full_assignment_failed_baseline_manifest.json').write_text(json.dumps(baseline,indent=2))
 (P/'failed_baseline_manifest.json').write_text(json.dumps(baseline,indent=2))
 (P/'checkpoints'/'metadata.json').write_text(json.dumps({'assignment_latent_mode':'role_partition','role_partition_diffusion_type':'discrete_constrained','seed':17,'sample':'RHODIN01|4|gener|9b810e76ec9d286','trained_baseline_modules':False},indent=2))
 sample=torch.load(OLD/'fixed_sample.pt',map_location='cpu')
 torch.save(sample,P/'fixed_sample.pt')
 z=sample['z'].tolist();rz=sample['role_z'].tolist();K=int(sample['Z']);M=int(sample['M'])
 composition={str(e):z.count(e) for e in sorted(set(z))}; role_composition={str(e):rz.count(e) for e in sorted(set(rz))}
 allowed=1
 for e,n in composition.items():allowed*=math.factorial(n)//(math.factorial(K)**role_composition[e])
 frac=sample['pos'].contiguous().numpy().tobytes()
 complete=all(set(sample['role'][sample['copy']==c].tolist())==set(range(M)) for c in range(K))
 metadata={'dataset_split':sample['split'],'sample_crystal_id':sample['id'],'N':int(sample['N']),'M':M,'K':K,'Z':K,'element_composition':composition,'molecular_role_element_composition':role_composition,'molecular_graph_nodes':M,'molecular_graph_edges':int(sample['role_edge_index'].shape[1]),'clean_cell':sample['cell'].tolist(),'fractional_coordinates_sha256':hashlib.sha256(frac).hexdigest(),'canonical_role_definition':'role is mol_atom_id, used only to construct clean R_0/Q_0; never a predictor feature','copy_permutation_orbit_size':math.factorial(K),'element_mask_allowed_R_count':allowed,'random_element_compatible_R_exact_probability':1.0/allowed,'clean_target_copy_role_complete':complete}
 (P/'fixed_sample_metadata.json').write_text(json.dumps(metadata,indent=2))
 role_net=RolePartitionDiffusion();match_net=RolePartitionDiffusion();match_head=StructuredMatchingHead()
 def entries(prefix,net,group):
  return [{'name':f'{prefix}.{n}','shape':list(v.shape),'count':v.numel(),'trainable':True,'optimizer_group':group,'learning_rate':2e-4,'weight_decay':1e-2} for n,v in net.named_parameters()]
 params=entries('role',role_net,'role')+entries('matching_encoder',match_net,'matching')+entries('matching_head',match_head,'matching')
 (P/'trainable_parameters.json').write_text(json.dumps({'parameters':params,'total_assignment_side_parameters':sum(x['count'] for x in params),'frozen_modules':['all original pos score parameters','all original cell score parameters','all original atom-type parameters','all original pos/cell diffusion schedule parameters'],'note':'R-only and Q-only preliminary stages instantiate separate assignment-side encoder copies; no baseline module is instantiated or optimized.'},indent=2))
 role_counts={name:sum(v.numel() for n,v in role_net.named_parameters() if n.startswith(name)) for name in ('crystal_encoder','molecule_encoder','role_head')}
 qhead_count=sum(v.numel() for v in match_head.parameters())
 (P/'network_architecture.md').write_text(f'''# Role-partition discrete-constrained architecture

- Crystal encoder: a 4-layer scalar periodic-radius message-passing network (atom type, wrapped PBC displacement distance/RBF; 256 hidden, 64 RBF, cutoff 6 Å, maximum 64 neighbours), **{role_counts['crystal_encoder']:,}** parameters per stage. It has no packed-node, copy, or role-ID input.
- Molecular encoder: a 4-layer atom-type/bond-type MPNN, **{role_counts['molecule_encoder']:,}** parameters per stage. Molecular node permutation only permutes its output; there is no node-position or canonical-role embedding.
- R compatibility head: noisy visible-role embedding or MASK embedding plus crystal/molecular pair features and timestep, followed by four masked mean/max axial row/column interactions over [N,M], **{role_counts['role_head']:,}** parameters. It produces `S_R[N,M]` with element-forbidden pairs at −∞.
- Q structured head: anchor/target crystal features, PBC distance RBF, molecular role-pair/bond condition, partial-state (`MATCHED/EXCLUDED/UNKNOWN`) and timestep, followed by three axial interactions over [K,K], **{qhead_count:,}** parameters. It produces target-row/anchor-column scores and never flattens the matrix.

The initial R-only and Q-only checks use separate assignment-side encoder copies (total optimized parameters recorded in `trainable_parameters.json`); all original pos/cell/atom-type diffusion modules stay frozen and are not instantiated by this harness. Predictor inputs exclude `mol_atom_id`, `mol_copy_id`, clean R/Q/C, packed slot, and absolute index. `mol_atom_id`/`mol_copy_id` are used only by target construction.
''')
 (P/'mode_router_audit.md').write_text('# Mode router audit\n\n`assignment_latent_mode: none` is the default and instantiates neither assignment module. With no new mode, legacy `assignment_diffusion_enabled: true` maps to `full_assignment`; false/omitted maps to `none`. An explicit new mode takes precedence; an explicitly supplied, contradictory legacy boolean emits a warning. `full_assignment` instantiates only `AssignmentDiffusion`; `role_partition` instantiates only `RolePartitionDiffusion`; neither path mixes optimizer parameters. Checkpoint metadata persists `assignment_latent_mode` and `role_partition_diffusion_type`. The executed constructor audit is in `mode_router_audit.json`.\n')
 r_exact=sum(x['exact_R'] for x in role);q_exact=sum(x['exact_Q'] for x in match)
 role_curve=json.loads((P/'role_training_metrics.json').read_text())['curve'];q_curve=json.loads((P/'matching_training_metrics.json').read_text())['curve']
 nonfinite_role=[x['step'] for x in role_curve if not math.isfinite(x['loss'])];nonfinite_q=[x['step'] for x in q_curve if not math.isfinite(x['loss'])]
 with open(P/'training_curves.csv','w',newline='') as stream:
  fields=['stage','step','loss','masked_accuracy','structured_accuracy','terminal'];writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(role_curve+q_curve)
 report=f'''# Role-partition discrete constrained — initial RHODIN01 report

Commit: `{__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()}`. No new branch was created. This is a one-seed initial test only; no D2, no pos/cell training, and no C diffusion occurred.

## Implemented route

`none | full_assignment | role_partition` is routed through `assignment_latent_mode`; default is `none`. `role_partition + discrete_constrained` uses capacity-constrained canonical role R, then permutation-constrained Q, then deterministic `C=GG^T`. The old failed full-assignment baseline is retained in `full_assignment_failed_baseline_manifest.json`.

## Strong constraints and oracle

Clean R/Q/C oracle passed: {op}. R reverse output was capacity-valid and element-compatible for {sum(x['capacity_valid'] and x['element_compatible'] for x in role)}/32; Q output was a valid permutation for {sum(x['permutation_valid'] for x in match)}/32. Every decoded C has zero reported symmetry, diagonal, row-sum, idempotence, and CR residual. These are support constraints, not learned recovery success.

## Initial result: STOP

R all-MASK stochastic reverse exact recovery: **{r_exact}/32**. Q-only (ground-truth R) all-MASK exact recovery: **{q_exact}/32**. Sequential reverse was therefore not run beyond its explicit stop records (`matching_ran=false`); no result is claimed for R→Q. Terminal condition ablations do not show useful geometry signal. The NaN/Inf guard skipped {len(nonfinite_role)} R losses (first 20 steps: {nonfinite_role[:20]}) and {len(nonfinite_q)} Q losses. This fails the requested R-only and Q-only initial thresholds and the no-NaN stability requirement, so formal three-seed verification and D2 are prohibited.

The matching training trace includes mathematically valid zero NLL states when the partial matching leaves only one legal completion. Checkpoint selection now excludes fully visible no-mask states, but this did not yield generation recovery. Terminal correct/zero/mismatch/no-edge condition results are recorded in `condition_ablation.json`; they do not meet the requested condition-usefulness criterion. Correct follow-up is to stabilize structured logits and demonstrate terminal correct-vs-zero/mismatch separation before retrying the same fixed RHODIN01 sample.
'''
 (P/'initial_report.md').write_text(report);print(json.dumps({'R_exact':r_exact,'Q_exact':q_exact,'sequential_ran':any(x['matching_ran'] for x in seq),'status':'FAIL_STOP'},indent=2))
if __name__=='__main__':main()
