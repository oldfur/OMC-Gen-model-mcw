#!/usr/bin/env bash
# N2 remote: soft-C feedback → geometry denoising adapters (frozen GemNet + frozen N1).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true ;;
    *) echo "Unknown: $1" >&2; exit 2 ;;
  esac
  shift
done
if [[ "$EXECUTE" != "true" ]]; then
  echo "Refusing without --execute" >&2
  exit 2
fi

CFG="configs/assignment_diffusion_mvp/soft_c_geometry_feedback_n2.yaml"
OUT="outputs/assignment_diffusion_mvp/soft_c_geometry_feedback_n2"
SAMPLE="outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt"
N1_CKPT="${N1_CKPT:-outputs/assignment_diffusion_mvp/noisy_copy_assignment_n1/best_checkpoint.pt}"

MATTERGEN_RUN="${MATTERGEN_RUN:-/public/home/lmy/mattergen_omc25/outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567}"
MATTERGEN_LOAD_EPOCH="${MATTERGEN_LOAD_EPOCH:-294}"
MATTERGEN_CKPT="${MATTERGEN_CKPT:-${MATTERGEN_RUN}/lightning_logs/version_0/checkpoints/epoch=294-loss_val=0.04.ckpt}"

echo "[1/6] preflight artifacts"
[[ -f "$SAMPLE" ]] || { echo "missing fixed sample"; exit 1; }
[[ -f "$N1_CKPT" ]] || { echo "missing N1 checkpoint: $N1_CKPT"; exit 1; }
[[ -d "$MATTERGEN_RUN" ]] || { echo "FATAL: MATTERGEN_RUN missing: $MATTERGEN_RUN"; exit 1; }
[[ -f "$MATTERGEN_CKPT" ]] || { echo "FATAL: MATTERGEN_CKPT missing: $MATTERGEN_CKPT"; exit 1; }
echo "  MATTERGEN_CKPT_SHA256=$(sha256sum "$MATTERGEN_CKPT" | awk '{print $1}')"
echo "  N1_CKPT_SHA256=$(sha256sum "$N1_CKPT" | awk '{print $1}')"
echo "  N2_MODE=soft_c_feedback  G_DIFFUSION=false"

echo "[2/6] unit tests (N2)"
python -m pytest \
  mattergen/assignment/soft_c_geometry_feedback_n2/tests/test_n2_soft_c_feedback.py \
  -q

echo "[3/6] train N2 adapters (frozen base GemNet + frozen N1)"
python scripts/assignment_diffusion_mvp/train_soft_c_geometry_feedback_n2.py \
  --config "$CFG" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --n1-checkpoint "$N1_CKPT" \
  --n2-mode B2_combined \
  --execute

echo "[4/6] paired geometry eval (B0–B5)"
python scripts/assignment_diffusion_mvp/evaluate_soft_c_geometry_feedback_n2.py \
  --config "$CFG" \
  --adapter-checkpoint "$OUT/final_adapter_checkpoint.pt" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --n1-checkpoint "$N1_CKPT" \
  --execute

echo "[5/6] assemble report"
python scripts/assignment_diffusion_mvp/assemble_soft_c_geometry_feedback_n2_report.py \
  --output-dir "$OUT"

echo "[6/6] DONE N2 → $OUT"
echo "See n2_report.md / paired_geometry_eval.jsonl / ablation_summary.json"
