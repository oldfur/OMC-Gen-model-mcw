#!/usr/bin/env bash
# Scaled Clean-G Oracle (gated t>=0.5) vs Original. Denoising + held-out sampling.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
SMOKE=false
STAGE="all"  # dataset|audit|smoke|train|eval|sample|sample_n150|all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true ;;
    --smoke) SMOKE=true ;;
    --stage) STAGE="$2"; shift ;;
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
  echo "selected GPU ${gpu}"
}
pick_free_gpu

CFG="configs/assignment_diffusion_mvp/scaled_clean_g_geometry.yaml"
DS="outputs/assignment_diffusion_mvp/scaled_clean_g_dataset"
BASE="outputs/assignment_diffusion_mvp/scaled_clean_g_geometry"
MATTERGEN_RUN="${MATTERGEN_RUN:-/public/home/lmy/mattergen_omc25/outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567}"
MATTERGEN_LOAD_EPOCH="${MATTERGEN_LOAD_EPOCH:-294}"
MATTERGEN_CKPT="${MATTERGEN_CKPT:-${MATTERGEN_RUN}/lightning_logs/version_0/checkpoints/epoch=294-loss_val=0.04.ckpt}"
MG=(--mattergen-model-path "$MATTERGEN_RUN" --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" --mattergen-checkpoint "$MATTERGEN_CKPT")

CACHE="/public/home/lmy/mattergen_omc25/datasets/cache/omc25_le50_mattergen"
MAP_TR="/public/home/lmy/mattergen_omc25/datasets/molecule_mapping/omc25_le300_train_molmap_hybrid_v3.jsonl.gz"
MAP_VA="/public/home/lmy/mattergen_omc25/datasets/molecule_mapping/omc25_le300_val_molmap_hybrid_v3.jsonl.gz"

if [[ "$STAGE" == "audit" ]]; then
  echo "[assignment audit only]"
  python scripts/assignment_diffusion_mvp/build_scaled_clean_g_dataset.py \
    --cache-root "$CACHE" --molmap-train "$MAP_TR" --molmap-val "$MAP_VA" \
    --out "$DS" --train-n 3000 --val-n 250 --test-n 150 --seed 17 --execute --audit-only
  echo "[smoke gate + gemnet]"
  python scripts/assignment_diffusion_mvp/smoke_scaled_clean_g_geometry.py --config "$CFG" --execute "${MG[@]}"
  echo "AUDIT STAGE done"
  exit 0
fi

if [[ "$STAGE" == "all" || "$STAGE" == "dataset" ]]; then
  if [[ ! -f "$DS/manifest.json" ]]; then
    echo "[dataset]"
    python scripts/assignment_diffusion_mvp/build_scaled_clean_g_dataset.py \
      --cache-root "$CACHE" --molmap-train "$MAP_TR" --molmap-val "$MAP_VA" \
      --out "$DS" --train-n 3000 --val-n 250 --test-n 150 --seed 17 --execute
  else
    echo "[dataset] skip (found $DS/manifest.json)"
  fi
fi

echo "[smoke gate]"
python scripts/assignment_diffusion_mvp/smoke_scaled_clean_g_geometry.py --config "$CFG"
if [[ "$STAGE" == "all" || "$STAGE" == "smoke" ]]; then
  echo "[smoke execute]"
  python scripts/assignment_diffusion_mvp/smoke_scaled_clean_g_geometry.py --config "$CFG" --execute "${MG[@]}"
fi
if [[ "$SMOKE" == "true" ]]; then
  echo "SMOKE-ONLY done"
  exit 0
fi

run_arm () {
  local arm="$1"
  local out="$BASE/$arm"
  mkdir -p "$out"
  if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    if [[ -f "$out/final_checkpoint.pt" ]]; then
      echo "[train $arm] skip"
    else
      echo "[train $arm]"
      python scripts/assignment_diffusion_mvp/train_scaled_clean_g_geometry.py \
        --config "$CFG" --ablation-arm "$arm" --output-dir "$out" --dataset-dir "$DS" \
        "${MG[@]}" --execute
    fi
  fi
  if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    echo "[eval $arm]"
    python scripts/assignment_diffusion_mvp/eval_scaled_clean_g_denoising.py \
      --config "$CFG" --checkpoint "$out/final_checkpoint.pt" --ablation-arm "$arm" \
      --output-dir "$out" --split val "${MG[@]}" --execute
  fi
  if [[ "$STAGE" == "all" || "$STAGE" == "sample" ]]; then
    echo "[sample $arm]"
    python scripts/assignment_diffusion_mvp/sample_scaled_clean_g_geometry.py \
      --config "$CFG" --checkpoint "$out/final_checkpoint.pt" --ablation-arm "$arm" \
      --output-dir "$out" "${MG[@]}" --execute
  fi
}

if [[ "$STAGE" == "sample_n150" ]]; then
  N150="outputs/assignment_diffusion_mvp/scaled_clean_g_geometry_n150"
  mkdir -p "$N150/original" "$N150/clean_g"
  for arm in original clean_g; do
    ckpt="$BASE/$arm/final_checkpoint.pt"
    [[ -f "$ckpt" ]] || { echo "missing checkpoint $ckpt" >&2; exit 1; }
    echo "[sample_n150 $arm] ckpt=$ckpt"
    python scripts/assignment_diffusion_mvp/sample_scaled_clean_g_geometry.py \
      --config "$CFG" --checkpoint "$ckpt" --ablation-arm "$arm" \
      --output-dir "$N150/$arm" --n-crystals 150 --n-traj-per-crystal 2 \
      "${MG[@]}" --execute
  done
  python scripts/assignment_diffusion_mvp/compare_scaled_clean_g_geometry.py \
    --original "$N150/original" --clean-g "$N150/clean_g" \
    --out "$N150/comparison.json"
  echo "DONE sample_n150 → $N150"
  exit 0
fi

if [[ "$STAGE" != "dataset" && "$STAGE" != "smoke" ]]; then
  run_arm original
  run_arm clean_g
fi

if [[ "$STAGE" == "all" || "$STAGE" == "sample" || "$STAGE" == "eval" ]]; then
  python scripts/assignment_diffusion_mvp/compare_scaled_clean_g_geometry.py \
    --original "$BASE/original" --clean-g "$BASE/clean_g" \
    --out "$BASE/comparison.json"
fi
echo "DONE scaled Clean-G → $BASE"
