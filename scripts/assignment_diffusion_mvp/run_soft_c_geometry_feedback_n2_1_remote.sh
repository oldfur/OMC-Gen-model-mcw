#!/usr/bin/env bash
# N2.1 remote: strict C-dependent causal edge feedback test.
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

CFG="configs/assignment_diffusion_mvp/soft_c_geometry_feedback_n2_1.yaml"
OUT="outputs/assignment_diffusion_mvp/soft_c_geometry_feedback_n2_1_causal_edge"
N1_CKPT="${N1_CKPT:-outputs/assignment_diffusion_mvp/noisy_copy_assignment_n1/best_checkpoint.pt}"
MATTERGEN_RUN="${MATTERGEN_RUN:-/public/home/lmy/mattergen_omc25/outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567}"
MATTERGEN_LOAD_EPOCH="${MATTERGEN_LOAD_EPOCH:-294}"
MATTERGEN_CKPT="${MATTERGEN_CKPT:-${MATTERGEN_RUN}/lightning_logs/version_0/checkpoints/epoch=294-loss_val=0.04.ckpt}"

echo "[1/5] preflight"
[[ -f "$N1_CKPT" ]] || { echo "missing N1 ckpt $N1_CKPT"; exit 1; }
[[ -d "$MATTERGEN_RUN" ]] || { echo "missing MATTERGEN_RUN"; exit 1; }
[[ -f "$MATTERGEN_CKPT" ]] || { echo "missing MATTERGEN_CKPT"; exit 1; }
echo "  FORMULA: delta_e = g * q * s * F_psi(e)"
echo "  GROUP_CONTEXT=false"

echo "[2/5] unit tests"
python -m pytest mattergen/assignment/soft_c_geometry_feedback_n2_1/tests -q

echo "[3/5] train N2.1 causal edge adapter"
python scripts/assignment_diffusion_mvp/train_soft_c_geometry_feedback_n2_1.py \
  --config "$CFG" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --n1-checkpoint "$N1_CKPT" \
  --execute

echo "[4/5] paired eval B0/B2/B5/B6 (shared noisy batch; same adapter)"
python scripts/assignment_diffusion_mvp/evaluate_soft_c_geometry_feedback_n2_1.py \
  --config "$CFG" \
  --adapter-checkpoint "$OUT/final_adapter_checkpoint.pt" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --n1-checkpoint "$N1_CKPT" \
  --execute

echo "[5/5] report"
python scripts/assignment_diffusion_mvp/assemble_soft_c_geometry_feedback_n2_1_report.py --output-dir "$OUT"
echo "DONE N2.1 → $OUT"
