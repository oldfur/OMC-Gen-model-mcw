#!/usr/bin/env bash
# Paired Original vs G-conditioned geometry ablation (RHODIN01, 1000 steps, seed 17).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
SMOKE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true ;;
    --smoke) SMOKE=true ;;
    *) echo "Unknown $1" >&2; exit 2 ;;
  esac
  shift
done
[[ "$EXECUTE" == "true" ]] || { echo "Refusing without --execute" >&2; exit 2; }

pick_free_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    return
  fi
  local gpu
  gpu="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if($2+0>=8192) printf "%s %s\n",$2,$1}' \
    | sort -nr | awk 'NR==1{print $2}')"
  if [[ -z "${gpu}" ]]; then
    echo "no GPU with >=8GiB free" >&2
    nvidia-smi
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${gpu}"
  echo "selected GPU ${gpu} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
}
pick_free_gpu

CFG="configs/assignment_diffusion_mvp/joint_axl_diffusion_j1.yaml"
BASE="outputs/assignment_diffusion_mvp/g_geometry_ablation"
MATTERGEN_RUN="${MATTERGEN_RUN:-/public/home/lmy/mattergen_omc25/outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567}"
MATTERGEN_LOAD_EPOCH="${MATTERGEN_LOAD_EPOCH:-294}"
MATTERGEN_CKPT="${MATTERGEN_CKPT:-${MATTERGEN_RUN}/lightning_logs/version_0/checkpoints/epoch=294-loss_val=0.04.ckpt}"
MG=(--mattergen-model-path "$MATTERGEN_RUN" --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" --mattergen-checkpoint "$MATTERGEN_CKPT")

echo "[0] smoke"
python scripts/assignment_diffusion_mvp/smoke_g_geometry_ablation.py --config "$CFG" --execute "${MG[@]}"
if [[ "$SMOKE" == "true" ]]; then
  echo "SMOKE-ONLY done"
  exit 0
fi

run_arm () {
  local arm="$1" cond="$2" mode="$3"
  local out="$BASE/$arm"
  mkdir -p "$out"
  if [[ -f "$out/final_checkpoint.pt" ]]; then
    echo "[train $arm] skip (found $out/final_checkpoint.pt)"
  else
    echo "[train $arm]"
    python scripts/assignment_diffusion_mvp/train_joint_axl_diffusion_j1.py \
      --config "$CFG" --output-dir "$out" \
      --geometry-assignment-conditioning "$cond" \
      "${MG[@]}" --execute
  fi
  echo "[eval $arm]"
  python scripts/assignment_diffusion_mvp/evaluate_joint_axl_diffusion_j1.py \
    --config "$CFG" --checkpoint "$out/final_checkpoint.pt" --output-dir "$out" \
    --geometry-assignment-conditioning "$cond" \
    "${MG[@]}" --execute
  echo "[sample $arm]"
  python scripts/assignment_diffusion_mvp/sample_joint_axl_diffusion_j1.py \
    --config "$CFG" --checkpoint "$out/final_checkpoint.pt" --output-dir "$out" \
    --geometry-assignment-conditioning "$cond" \
    --assignment-mode "$mode" \
    "${MG[@]}" --execute
}

run_arm original false static_A
run_arm g_conditioned true ctmc_A

python scripts/assignment_diffusion_mvp/compare_g_geometry_ablation.py \
  --original "$BASE/original" --g-conditioned "$BASE/g_conditioned" \
  --out "$BASE/comparison.json"

echo "DONE G→geometry ablation → $BASE"
