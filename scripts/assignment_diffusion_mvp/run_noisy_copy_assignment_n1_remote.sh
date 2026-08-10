#!/usr/bin/env bash
# N1 remote runner: MatterGen-native noise → frozen epoch294 GemNet → orbit-aware assignment.
# NO geometry feedback. NO ContextCrystalEncoder fallback.
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

CFG="configs/assignment_diffusion_mvp/noisy_copy_assignment_n1.yaml"
OUT="outputs/assignment_diffusion_mvp/noisy_copy_assignment_n1"
HARD_R="outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/geometry_only_hard_r.jsonl"
SAMPLE="outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt"

# Primary molecular-CSP full-prior run (NOT mattergen_base / topology / assign-bond / setattn).
MATTERGEN_RUN="${MATTERGEN_RUN:-/public/home/lmy/mattergen_omc25/outputs/singlerun/2026-06-03/le50_molcsp_scratch_mean_cell025_bondhuber_lowt04_posonly_w1e-3_8gpu01234567}"
MATTERGEN_LOAD_EPOCH="${MATTERGEN_LOAD_EPOCH:-294}"
MATTERGEN_CKPT="${MATTERGEN_CKPT:-${MATTERGEN_RUN}/lightning_logs/version_0/checkpoints/epoch=294-loss_val=0.04.ckpt}"

echo "[1/6] preflight artifacts"
[[ -f "$SAMPLE" ]] || { echo "missing fixed sample: $SAMPLE"; exit 1; }
[[ -f "$HARD_R" ]] || { echo "missing geometry hard-R — run O2 export first"; exit 1; }
[[ -f "outputs/assignment_diffusion_mvp/role_automorphism_audit/role_orbits.json" ]] || { echo "missing role orbits"; exit 1; }

echo "[2/6] preflight MatterGen epoch294 checkpoint (no fallback)"
if [[ ! -d "$MATTERGEN_RUN" ]]; then
  echo "FATAL: MATTERGEN_RUN directory missing: $MATTERGEN_RUN" >&2
  echo "N1 will not fall back to ContextCrystalEncoder." >&2
  exit 1
fi
if [[ ! -f "$MATTERGEN_CKPT" ]]; then
  echo "FATAL: MATTERGEN_CKPT missing: $MATTERGEN_CKPT" >&2
  echo "N1 will not fall back to ContextCrystalEncoder." >&2
  exit 1
fi
# Lightweight provenance: config.yaml + ckpt name
if [[ -f "$MATTERGEN_RUN/config.yaml" ]]; then
  echo "  MATTERGEN_RUN config.yaml: present"
else
  echo "  WARN: $MATTERGEN_RUN/config.yaml missing (MatterGenCheckpointInfo may still resolve hydra conf)"
fi
CKPT_SHA="$(sha256sum "$MATTERGEN_CKPT" | awk '{print $1}')"
echo "  MATTERGEN_RUN=$MATTERGEN_RUN"
echo "  MATTERGEN_LOAD_EPOCH=$MATTERGEN_LOAD_EPOCH"
echo "  MATTERGEN_CKPT=$MATTERGEN_CKPT"
echo "  MATTERGEN_CKPT_SHA256=$CKPT_SHA"
echo "  HIDDEN_SOURCE=gemnet"
echo "  CONTEXT_CRYSTAL_ENCODER_USED=false"
echo "  GEOMETRY_FEEDBACK=false"

echo "[3/6] unit tests (N1)"
python -m pytest \
  mattergen/assignment/noisy_copy_assignment/tests/test_noisy_copy_assignment_n1.py \
  -q

echo "[4/6] train N1 (frozen GemNet epoch294, RESTARTED)"
python scripts/assignment_diffusion_mvp/train_noisy_copy_assignment_n1.py \
  --config "$CFG" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --hidden-source gemnet \
  --execute

echo "[5/6] recovery curves (oracle_orbit + predicted_orbit)"
python scripts/assignment_diffusion_mvp/evaluate_noisy_copy_assignment_curve_n1.py \
  --config "$CFG" \
  --checkpoint "$OUT/final_checkpoint.pt" \
  --mattergen-model-path "$MATTERGEN_RUN" \
  --mattergen-load-epoch "$MATTERGEN_LOAD_EPOCH" \
  --mattergen-checkpoint "$MATTERGEN_CKPT" \
  --hidden-source gemnet \
  --execute

echo "[6/6] assemble report"
python scripts/assignment_diffusion_mvp/assemble_noisy_copy_assignment_n1_report.py \
  --output-dir "$OUT"

echo "DONE N1 → $OUT"
echo "PRIMARY_N1=gemnet  ABLATION=context_encoder (not run by default)"
echo "See n1_report.md / oracle_orbit_curve.jsonl / predicted_orbit_curve.jsonl"
