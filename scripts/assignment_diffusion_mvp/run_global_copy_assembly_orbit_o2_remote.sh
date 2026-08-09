#!/usr/bin/env bash
# Remote one-shot O2 orbit-aware assembly workflow (user executes on GPU node).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$EXECUTE" != "true" ]]; then
  echo "Refusing to run remote O2 workflow without --execute" >&2
  exit 2
fi

GEOMETRY_CKPT="${GEOMETRY_ONLY_CHECKPOINT_PATH:-outputs/assignment_diffusion_mvp/role_oracle_partition_diagnostic/checkpoints/geometry_only/best.pt}"
OUT_DIR="outputs/assignment_diffusion_mvp/global_copy_assembly_orbit_aware_o2"
CFG="configs/assignment_diffusion_mvp/global_copy_assembly_orbit_aware_o2.yaml"
R_OUT="outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r"

echo "[1/6] pytest (global assembly + O2 tests)"
python -m pytest \
  mattergen/assignment/global_copy_assembly/tests/test_global_copy_assembly.py \
  mattergen/assignment/global_copy_assembly/tests/test_orbit_aware_o2.py \
  -q

echo "[2/6] geometry-only checkpoint"
if [[ -f "$GEOMETRY_CKPT" ]]; then
  echo "SKIP: reusing $GEOMETRY_CKPT"
else
  python scripts/diagnostics/role_oracle_partition_diagnostic.py
fi

echo "[3/6] export geometry-only hard-R (canonical-shaped artifact)"
python scripts/assignment_diffusion_mvp/export_geometry_only_hard_r.py

echo "[4/6] train O2 orbit-aware assembly (RESTARTED)"
python scripts/assignment_diffusion_mvp/train_global_copy_assembly_orbit_o2.py \
  --config "$CFG" \
  --steps 5000 \
  --output-dir "$OUT_DIR" \
  --execute

echo "[5/6] evaluate best checkpoint (independent MAP)"
python scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly_orbit_o2.py \
  --config "$CFG" \
  --checkpoint "$OUT_DIR/best_checkpoint.pt" \
  --execute

echo "[6/6] evaluate final checkpoint"
python scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly_orbit_o2.py \
  --config "$CFG" \
  --checkpoint "$OUT_DIR/final_checkpoint.pt" \
  --execute

echo "DONE O2"
echo "Artifacts:"
echo "  hard-R: $R_OUT/geometry_only_hard_r.jsonl"
echo "  O2 dir: $OUT_DIR"
echo "  map_evaluation_metrics.json / evaluation_metrics.json / training_trace.jsonl"
