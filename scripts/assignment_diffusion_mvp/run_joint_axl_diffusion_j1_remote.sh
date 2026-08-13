#!/usr/bin/env bash
# J1 remote: joint assignment + geometry diffusion (RHODIN01 MVP).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true ;;
    *) echo "Unknown $1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$EXECUTE" == "true" ]] || { echo "Refusing without --execute" >&2; exit 2; }

CFG="configs/assignment_diffusion_mvp/joint_axl_diffusion_j1.yaml"
OUT="outputs/assignment_diffusion_mvp/joint_axl_diffusion_j1"
MATTERGEN_RUN="${MATTERGEN_RUN:-/public/home/lmy/mattergen_omc25/outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567}"
MATTERGEN_LOAD_EPOCH="${MATTERGEN_LOAD_EPOCH:-294}"
MATTERGEN_CKPT="${MATTERGEN_CKPT:-${MATTERGEN_RUN}/lightning_logs/version_0/checkpoints/epoch=294-loss_val=0.04.ckpt}"

echo "[1/4] preflight"
[[ -f outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt ]] || exit 1
[[ -d "$MATTERGEN_RUN" ]] || { echo "missing MATTERGEN_RUN"; exit 1; }
[[ -f "$MATTERGEN_CKPT" ]] || { echo "missing MATTERGEN_CKPT"; exit 1; }

echo "[2/4] train J1"
python scripts/assignment_diffusion_mvp/train_joint_axl_diffusion_j1.py \
  --config "$CFG" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --execute

echo "[3/4] evaluate"
python scripts/assignment_diffusion_mvp/evaluate_joint_axl_diffusion_j1.py \
  --config "$CFG" \
  --checkpoint "$OUT/final_checkpoint.pt" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --execute

echo "[4/4] sample (ctmc_A + static_A)"
python scripts/assignment_diffusion_mvp/sample_joint_axl_diffusion_j1.py \
  --config "$CFG" \
  --checkpoint "$OUT/final_checkpoint.pt" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --assignment-mode all \
  --execute

echo "DONE J1.3-B1 orbit-slot → $OUT"
