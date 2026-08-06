"""Recompute post-training sampler and per-corruption diagnostics without training."""
from __future__ import annotations
import importlib.util, json, subprocess
from pathlib import Path
import torch

HERE='scripts/diagnostics/role_swap_best_action_rhodin.py';OUT=Path('outputs/assignment_diffusion_mvp/role_swap_best_action')
def main():
 sp=importlib.util.spec_from_file_location('swap_best',HERE);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 s=m.load();states=m.terminal_states(s);model=m.net();ckpt=torch.load(OUT/'checkpoints/best.pt',map_location='cpu');model.load_state_dict(ckpt['state_dict']);model.eval()
 per={}
 g=torch.Generator(device='cuda').manual_seed(771)
 for t in (1,8,16,24,32,40,48,56,64):
  rows=[]
  for _ in range(8):
   st=s['role'].clone()
   for __ in range(int(128*t/m.T)):
    p=m.pairs(st,s['z']);st=m.apply_action(st,p[torch.randint(len(p),(),device='cuda',generator=g)])
   rows.append(st)
  per[str(t)]=m.policy_eval(model,s,rows,t=t)
 pol={'terminal_policy':m.policy_eval(model,s,states),'clean_policy':m.policy_eval(model,s,states,clean=True),'by_corruption_timestep':per}
 (OUT/'policy_mass_metrics.json').write_text(json.dumps(pol,indent=2))
 comp={};traces={}
 for kind in ('map','annealed','stochastic'):
  rows=[m.sampler(model,s,st,kind,seed=i) for i,st in enumerate(states)]
  comp[kind]={'exact_R':sum(x['exact'] for x in rows),'mean_accuracy':sum(x['final_accuracy'] for x in rows)/32,'mean_peak_accuracy':sum(x['peak_accuracy'] for x in rows)/32,'ever_exact':sum(x['ever_exact'] for x in rows),'left_after_exact':sum(x['left_after_exact'] for x in rows),'positive_rate':sum(x['positive_rate'] for x in rows)/32,'neutral_rate':sum(x['neutral_rate'] for x in rows)/32,'negative_rate':sum(x['negative_rate'] for x in rows)/32,'no_op_rate':sum(x['no_op_rate'] for x in rows)/32,'executed_steps':sum(x['executed_steps'] for x in rows)/32,'wall_seconds':sum(x['wall_seconds'] for x in rows)/32};traces[kind]=rows
 (OUT/'sampler_comparison.json').write_text(json.dumps(comp,indent=2));(OUT/'swap_trajectories.json').write_text(json.dumps(traces,indent=2))
 ab={mode:sum(m.sampler(model,s,st,'map',geometry=mode,seed=i)['final_accuracy'] for i,st in enumerate(states[:8]))/8 for mode in ('correct','zero','mismatch')};(OUT/'condition_ablation.json').write_text(json.dumps(ab,indent=2))
 oracle=json.loads((OUT/'oracle_greedy_metrics.json').read_text());report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'oracle_greedy_exact':sum(x['exact'] for x in oracle),'oracle_max_swaps':max(x['swaps'] for x in oracle),'clean_noop_probability':pol['clean_policy']['noop_probability'],'clean_noop_top1':pol['clean_policy']['top1_best'],'terminal_mass':pol['terminal_policy'],'samplers':comp,'condition_ablation':ab,'diagnosis':'action space is reachable; policy has clean/terminal supervised accuracy but applies premature no-op on generated intermediate states (exposure/state-distribution policy failure). MAP/annealed/stochastic all fail similarly, so sampler oscillation is not the primary cause.','PASS':False,'allow_q_c':False,'allow_three_seed':False,'allow_D2':False}
 (OUT/'initial_report.md').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
