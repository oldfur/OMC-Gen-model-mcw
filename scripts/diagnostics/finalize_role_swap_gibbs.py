"""Write the explicit stop report for the fixed-sample swap/Gibbs audit."""
from __future__ import annotations
import json, subprocess
from pathlib import Path

OUT=Path('outputs/assignment_diffusion_mvp/role_swap_gibbs_q_masked')
OLD=Path('outputs/assignment_diffusion_mvp/role_partition_discrete_constrained')

def main():
 samples=json.loads((OUT/'swap_sampling_metrics.json').read_text());train=json.loads((OUT/'swap_training_metrics.json').read_text());op=json.loads((OUT/'swap_operator_metrics.json').read_text());ab=json.loads((OUT/'condition_ablation.json').read_text())
 exact=sum(x['exact_R'] for x in samples);acc=sum(x['final']['accuracy'] for x in samples)/len(samples);initial=sum(x['initial']['accuracy'] for x in samples)/len(samples);valid=sum(x['final']['valid'] for x in samples);improve=sum(x['selected_action_non_decrease_rate'] for x in samples)/len(samples);noop=sum(x['no_op_rate'] for x in samples)/len(samples)
 baseline=json.loads((OLD/'role_sampling_metrics.json').read_text())
 base_exact=sum(x['exact_R'] for x in baseline)
 sequential={'ran':False,'reason':f'R-only failed: exact_R={exact}/32 (<28) and atom_role_accuracy={acc:.6f} (<0.99). Q and C remain untouched.'}
 (OUT/'sequential_r_q_c_metrics.json').write_text(json.dumps(sequential,indent=2))
 result={'masked_capacity_baseline':{'preserved':True,'output':str(OLD/'role_sampling_metrics.json'),'exact_R':f'{base_exact}/32'},'swap':{'state':'legal R labels only; no MASK','actions':'no-op plus unordered same-element unequal-role swaps','forward':'floor(s_max*t/T) uniform legal swaps; s_max=128','terminal_prior':'per-element role-multiset shuffle','loss':'cross entropy from softmax(agreement-improvement/tau_target), tau_target=0.25','reverse':'stochastic categorical action chain for t=64..1'},'operator':op,'r_only':{'samples':len(samples),'valid':f'{valid}/32','nonfinite':train['nonfinite_count'],'initial_accuracy_mean':initial,'final_accuracy_mean':acc,'exact_R':f'{exact}/32','action_non_decrease_rate':improve,'no_op_rate':noop},'condition_ablation':ab,'pass':False,'allow_q_c':False,'allow_three_seed':False,'allow_d2':False}
 (OUT/'summary.json').write_text(json.dumps(result,indent=2))
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 (OUT/'initial_report.md').write_text(f'''# Swap/Gibbs R-only fixed RHODIN01 report

Commit `{commit}`; branch `feature/assignment-diffusion-mvp`; no branch created. The previous masked-capacity R baseline remains unchanged at `{base_exact}/32` exact-R in `{OLD}`.

## Method

R is always a legal capacity-constrained role vector. The only non-noop action swaps unequal roles on two same-element atoms. Forward corruption applies `floor(128*t/64)` uniform legal swaps; the terminal prior shuffles each element block's role multiset. The head obtains atom-role compatibility from the independent PBC crystal and molecular graph encoders, scores each legal swap from the compatibility gain plus pair/PBC RBF/timestep features, and includes a no-op score. Training uses the Boltzmann clean-agreement-improvement target at temperature 0.25. Reverse sampling is stochastic categorical, never independent role sampling.

## Actual result — FAIL / STOP

All {valid}/32 final states were capacity- and element-valid and non-finite count was {train['nonfinite_count']}. Mean terminal accuracy improved from {initial:.6f} to {acc:.6f}; selected action non-decrease rate was {improve:.6f}; no-op rate was {noop:.6f}. However exact-R was **{exact}/32**, below 28/32, and accuracy was below 0.99. Correct geometry ({ab['correct']:.6f}) outperformed zero ({ab['zero']:.6f}) and mismatch ({ab['mismatch']:.6f}), so the condition is used, but recovery is still insufficient.

Q structured masked-permutation diffusion and `C=GG^T` were not run or changed. Three-seed validation and D2 are prohibited.
''')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
