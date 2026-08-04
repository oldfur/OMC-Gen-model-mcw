# Assignment-diffusion MVP — Gate A final audit

Commit: `3830650861981a29851d2c1f5e472d52722247be`, branch `feature/assignment-diffusion-mvp`. The MVP modifications are uncommitted in this worktree; `git status`, `git rev-parse HEAD`, `git diff BASE...HEAD`, and the complete working-tree diff were saved under `full_checkpoint_isolation/`. No D1/D2, assignment training, or position/cell-diffusion change was run.

## Historical reference and current regression reference

The initial pre-modification runtime reference is **unavailable**: the first command used system Python and failed with `ModuleNotFoundError: No module named 'torch'`. It cannot be retrospectively recovered and was not fabricated. Substitute evidence is the git base commit, saved diffs, a complete full-checkpoint cross-worktree comparison, and the current all-Gate-A regression. This current commit/worktree state is the regression reference for later changes; any later code edit must rerun all Gate-A checks and compare with these artifacts.

## Full checkpoint isolation (A9)

- Baseline detached worktree: `/home/mcw/OMC-Gen-model-assignment-diffusion-baseline-audit` at `3830650861981a29851d2c1f5e472d52722247be`; it was removed after the audit.
- Environment on both sides: `/home/mcw/miniconda3/envs/adit/bin/python`, PyTorch `2.3.1`, CUDA `12.1`, CPU float32, deterministic algorithms enabled. CUDA GPU was detected but CPU was selected for exact reproducibility. Complete environment data is in `full_checkpoint_isolation/environment.json`.
- Existing trained checkpoint candidate: `/home/mcw/OMC-Gen-model/outputs/molcsp_sinkhorn_short_diagnostic_preflight_2/checkpoints/diagnostic.ckpt`, SHA-256 recorded in `checkpoint_metadata.json`. Strict load was correctly rejected because its 338-key state_dict contains 11 historical dynamic-assignment keys absent from both audited source trees.
- Therefore the permitted fallback was used: baseline worktree seed `20260804` initialized the **complete** configured `GemNetTDenoiser + GemNetT + MolecularGraphConditioner + DiffusionModule` (46,057,338 parameters), saved as `full_gemnet_initialized_baseline.pt`, and both subprocesses strict-loaded identical bytes. It is explicitly an initialized full-model checkpoint, not a trained checkpoint. Strict load had zero missing/unexpected keys; state_dict contains 327 keys.
- Three real serialized OMC batches, all batch size 2, were saved without tiny/mock substitution: Z=2 (`WIGXIU...` pair), Z=4 (`AKOVOL01...` pair), and mixed Z=2/4 (`GEYMOM`, `KIMNAV`), N values 32, 44, 46 and 48, and M values 11, 12, 16, 23. They include repeated elements. Exact timestep `[0.17, 0.73]`, pos noise, and cell noise were pre-generated and injected into the unmodified SDE sampling formulas; no other stochastic model inputs were observed (dropout is 0.0).
- Eval: 42 captured tensors across the three batches (noisy inputs, pos/cell/atom outputs, losses, conditioning, GemNet internals) were bitwise identical. Pos/cell/loss max absolute and relative differences: `0`.
- Train without optimizer step: the same 42 forward tensors and all 648 original parameter-gradient tensors were bitwise identical. Gradient max absolute/relative differences: `0`; no NaN/Inf.
- Python, NumPy, torch CPU, and CUDA RNG states before forward, after forward, and after backward were identical. Optimizer group, names, counts, hyperparameters, original state_dict key set, and parameter names were identical.
- Current disabled state: assignment module not instantiated, no assignment trajectory, no assignment loss key, no assignment parameters in optimizer, and no assignment gradients.

## Re-run Gate A results

Commands actually run:

- `PYTHONPATH=. /home/mcw/miniconda3/envs/omg/bin/python scripts/diagnostics/assignment_diffusion_d0.py`
- `PYTHONPATH=. /home/mcw/miniconda3/envs/omg/bin/python scripts/diagnostics/assignment_diffusion_forward_d0.py`
- `PYTHONPATH=. /home/mcw/miniconda3/envs/omg/bin/python scripts/diagnostics/assignment_diffusion_gate_a.py`
- Full subprocess commands and logs: `full_checkpoint_isolation/logs/`.

Existing operator invariants, slot-order-independent clean targets, real packed assignment forward/backward, projected noise, 20-seed live predictor equivariance, malformed-input validation, reverse posterior oracle, and no-NaN/Inf all re-passed. The detailed numbers remain in `d0_metrics.json`, `d0_predictor_equivariance.json`, `d0_malformed_inputs.json`, and `d0_reverse_oracle.json`.

| Gate A item | Result |
| --- | --- |
| A1 operator invariants | PASS |
| A2 clean target independent of packed slot order | PASS |
| A3 real OMC packed forward/backward | PASS |
| A4 projected-noise mathematics | PASS |
| A5 live row/column/copy/batch equivariance | PASS |
| A6 batch independence | PASS |
| A7 reverse DDPM posterior oracle | PASS |
| A8 malformed inputs fail loudly | PASS |
| A9 full GemNet checkpoint baseline isolation | PASS |

**Gate A = PASS.** The historical runtime reference remains unavailable/non-recoverable evidence, not a failure. D1 is now permitted by the gate; it was not started in this task.
