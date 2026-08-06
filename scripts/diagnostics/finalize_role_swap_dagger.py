"""Final DAgger diagnostics; no training and no Q invocation."""
from __future__ import annotations
import importlib.util,json,subprocess
from pathlib import Path
import torch
OUT=Path('outputs/assignment_diffusion_mvp/role_swap_dagger')
def main():
 sp=importlib.util.spec_from_file_location('dagger','scripts/diagnostics/role_swap_dagger_rhodin.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 s=m.load();initials=m.states(s);policy=m.Policy().cuda();state=torch.load(OUT/'checkpoints/round_3_best.pt',map_location='cpu');policy.load_state_dict(state['state_dict']);policy.eval()
 clean=[]
 with torch.no_grad():
  for t in (1,8,16,24,32,40,48,56,64):
   p,l,stop=policy.state(s,s['role'],t);clean.append({'t':t,'stop_probability':float(torch.sigmoid(stop)),'swap_top1_gain':float(m.gains(s['role'],s['role'],p)[l.argmax()])})
 rounds=json.loads((OUT/'dagger_rounds.json').read_text());ab=json.loads((OUT/'condition_ablation.json').read_text())
 policy_mass={'rounds':[{'round':i+1,'best_action_mass':r['policy']['best_mass'],'best_action_top1_recall':r['policy']['top1'],'p_plus':r['policy']['p_plus'],'p_zero':r['policy']['p_zero'],'p_minus':r['policy']['p_minus'],'stop_probability_nonclean':r['policy']['stop_probability'],'onpolicy_false_stop_rate':r['false_stop_rate']} for i,r in enumerate(rounds)],'clean_stop':clean}
 (OUT/'policy_mass_metrics.json').write_text(json.dumps(policy_mass,indent=2))
 r3=rounds[-1]
 report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'rounds':policy_mass['rounds'],'round3':{'exact_R':r3['exact_R'],'final_accuracy':r3['final_accuracy'],'peak_accuracy':r3['peak_accuracy'],'ever_exact':r3['ever_exact'],'top1_recall_onpolicy_rollout':r3['top1_recall'],'mean_steps':r3['executed_steps'],'wall_seconds':r3['wall_seconds'],'valid':r3['valid'],'nonfinite':r3['nonfinite']},'clean_stop':clean,'condition_ablation':ab,'diagnosis':f"DAgger eliminated observed nonclean false stops in round 3, but clean stop probability is far below the 0.99 threshold and self-induced 64-step action top-1 recall is {r3['top1_recall']:.4f}. The policy therefore keeps swapping beyond clean/near-clean states and drifts. This is an on-policy ranking plus stop-calibration failure, not an action-space or sampler-noise issue.",'PASS':False,'allow_q_c':False,'allow_three_seed':False,'allow_D2':False}
 (OUT/'initial_report.md').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
