#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=true
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$EXECUTE" != "true" ]]; then
  echo "Refusing to run remote workflow without --execute" >&2
  exit 2
fi

GEOMETRY_CKPT="${GEOMETRY_ONLY_CHECKPOINT_PATH:-outputs/assignment_diffusion_mvp/role_oracle_partition_diagnostic/checkpoints/geometry_only/best.pt}"
OUT_DIR="outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r"

echo "[1/8] Running pytest checks"
python -m pytest mattergen/assignment/global_copy_assembly/tests/test_global_copy_assembly.py

echo "[2/8] Role-oracle partition diagnostic (geometry-only checkpoint source)"
if [[ -f "$GEOMETRY_CKPT" ]]; then
  echo "SKIP: reusing existing geometry-only checkpoint: $GEOMETRY_CKPT"
else
  python scripts/diagnostics/role_oracle_partition_diagnostic.py
fi

echo "[3/8] Exporting geometry-only hard-R artifact"
python scripts/assignment_diffusion_mvp/export_geometry_only_hard_r.py

echo "[4/8] Auditing geometry-only hard-R artifact"
python scripts/assignment_diffusion_mvp/audit_geometry_only_hard_r.py

echo "[5/8] Training global copy assembly on predicted R (RESTARTED clean run)"
# No reliable optimizer/scheduler/RNG resume in this MVP entry point.
# Always start a fresh predicted-R training write into OUT_DIR (overwrites prior traces).
python scripts/assignment_diffusion_mvp/train_global_copy_assembly.py \
  --config configs/assignment_diffusion_mvp/global_copy_assembly_clean_geometry_r.yaml \
  --steps 5000 \
  --output-dir "$OUT_DIR" \
  --execute

echo "[6/8] Evaluating best checkpoint (independent MAP)"
python scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly.py \
  --config configs/assignment_diffusion_mvp/global_copy_assembly_clean_geometry_r.yaml \
  --checkpoint "$OUT_DIR/best_checkpoint.pt" \
  --execute

echo "[7/8] Evaluating final checkpoint (independent MAP)"
python scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly.py \
  --config configs/assignment_diffusion_mvp/global_copy_assembly_clean_geometry_r.yaml \
  --checkpoint "$OUT_DIR/final_checkpoint.pt" \
  --execute

echo "[8/8] Assembling final report"
python scripts/assignment_diffusion_mvp/assemble_global_copy_assembly_report.py

echo "DONE: predicted-R global copy assembly workflow finished"
echo "Key artifacts under $OUT_DIR:"
echo "  - geometry_only_hard_r.jsonl"
echo "  - geometry_only_r_audit.json"
echo "  - training_trace.jsonl / best_checkpoint.pt / final_checkpoint.pt"
echo "  - map_evaluation_metrics.json / evaluation_metrics.json"
echo "  - global_copy_assembly_geometry_r_report.md"
