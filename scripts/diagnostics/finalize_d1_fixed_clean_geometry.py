"""Summarize completed D1 fixed-geometry run without promoting failed generation."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
P=Path('outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry')
def avg(rows,key):return float(np.mean([r[key] for r in rows]))
def main():
 pure=json.loads((P/'pure_noise_sampling_metrics.json').read_text());den=json.loads((P/'denoising_metrics.json').read_text());nog=json.loads((P/'no_geometry_ablation_metrics.json').read_text());rnd=json.loads((P/'random_baseline_metrics.json').read_text()); modes={}
 for seed,rows in pure.items():
  count=Counter(tuple(x['copy_permutation']) for x in rows);modes[seed]={'frequencies':{'-'.join(map(str,k)):v for k,v in count.items()},'entropy':float(-sum((v/len(rows))*np.log(v/len(rows)) for v in count.values())),'non_orbit_samples':sum(not x['exact_orbit_recovery'] for x in rows)}
 (P/'copy_mode_analysis.json').write_text(json.dumps(modes,indent=2))
 full={seed:{'loss':json.loads((P/f'training_metrics_seed_{seed}.json').read_text())['best_loss'],'pure_orbit':sum(x['exact_orbit_recovery'] for x in rows),'pure_acc':avg(rows,'atom_role_accuracy_mod_copy'),'complete':avg(rows,'complete_molecule_copy_rate'),'purity':avg(rows,'partition_purity'),'bond':avg(rows,'bond_consistency'),'cross':avg(rows,'cross_copy_edge_ratio'),'valid':sum(x['bijective'] and x['element_compatible'] for x in rows)} for seed,rows in pure.items()}
 report=f'''# Gate B — fixed clean-geometry D1

## Outcome

**Gate B = FAIL. D2 is not permitted and was not started.**

The single RHODIN01 fixed-sample overfit is numerically stable and forward-noised L0 denoising is excellent, but pure-noise reverse generation fails for every seed. This report deliberately keeps those two evaluation regimes separate.

## B1 — training stability: PASS

Seeds 17/29/43 best allowed-entry epsilon MSE: {full['17']['loss']:.5f}, {full['29']['loss']:.5f}, {full['43']['loss']:.5f}. No NaN/Inf was observed. Only D1 assignment-side parameters were optimized; baseline pos/cell heads were not instantiated or updated.

## B2 — forward-noised L0 denoising: PASS (diagnostic only)

At low/mid/high noise, all three seeds reached exact orbit recovery 1.0 on all 32 fixed-noise evaluations per level. This is not evidence of pure-noise generation.

## B3 — pure-noise validity: PASS

All 96 pure-noise final Hungarian assignments were element-compatible and bijective; forbidden Sinkhorn mass was zero.

## B4 — pure-noise recovery: FAIL

Exact orbit recovery is 0/32 for every seed (0/96 aggregate). Atom-role accuracy modulo copy is approximately {full['17']['pure_acc']:.3f}, {full['29']['pure_acc']:.3f}, {full['43']['pure_acc']:.3f}, essentially random baseline {rnd['atom_role_accuracy_mod_copy']:.3f}. This misses the required 30/32 per two seeds, 88/96 aggregate, and 99% atom-role accuracy.

## B5 — molecule assembly: FAIL

Complete molecule-copy rate is zero for all seeds; partition purity is about {full['17']['purity']:.3f}; bond consistency is about {full['17']['bond']:.3f}; cross-copy ratios are about {full['17']['cross']:.3f}. The assignment is valid but not an assembled molecule-copy solution.

## B6 — condition usefulness: FAIL

Independently trained no-geometry models also have 0/32 orbit recovery for each seed and random-level atom-role accuracy. On seed 17, geometry-row mismatch epsilon MSE (`geometry_mismatch_metrics.json`) differs only at numerical noise scale from full conditioning, and removing molecular edges similarly has negligible effect. The fixed single-sample setup therefore has no demonstrated geometry/edge use and cannot rule out insufficient conditional signal/exposure bias.

## B7 — no leakage: PASS for interface audit; insufficient for a passing gate

The D1 predictor forward signature contains no `mol_copy_id`, `mol_atom_id`, clean assignment, packed slot, absolute row/column embedding, sample id, or file name. It uses only noisy logits/timestep, clean periodic geometry, molecular graph and element mask. Gate-A equivariance remains frozen evidence. The synchronous row-permutation metric generated in the first run is invalid because its mask was not synchronously permuted; it is not used as evidence.

## Pure-noise trajectories

Eight failed pure-noise trajectories for seed 17 are saved under `trajectories/`; the run produced no successful trajectories, so a requested "8 successes" set is mathematically unavailable and is not fabricated. Copy-gauge mode frequencies are in `copy_mode_analysis.json`; all final modes are non-orbit modes.

## Next diagnostic order

Before any D2: inspect terminal schedule/SNR, reverse exposure bias and global assembly capacity; then establish demonstrable geometry and molecular-edge usefulness. Do not add geometry noise to mask this D1 failure.
'''
 (P/'gate_b_report.md').write_text(report)
 print(json.dumps({'gate_b':'FAIL','full':full,'random':rnd,'modes':modes},indent=2))
if __name__=='__main__':main()
