#!/usr/bin/env bash
# Remote frozen-checkpoint O2 information-source audit (NO TRAINING).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXECUTE=false
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true ;;
    --checkpoint) CHECKPOINT="$2"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$EXECUTE" != "true" ]]; then
  echo "Refusing to run information-source audit without --execute" >&2
  exit 2
fi

CFG="configs/assignment_diffusion_mvp/global_copy_assembly_orbit_aware_o2.yaml"
OUT="outputs/assignment_diffusion_mvp/global_copy_assembly_orbit_aware_o2"
HARD_R="outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/geometry_only_hard_r.jsonl"
FINAL="$OUT/final_checkpoint.pt"
BEST="$OUT/best_checkpoint.pt"

echo "[audit] checking inputs (no training)"
if [[ ! -f "$HARD_R" ]]; then
  echo "FATAL: missing geometry hard-R artifact: $HARD_R" >&2
  exit 1
fi
if [[ -n "$CHECKPOINT" ]]; then
  CKPT="$CHECKPOINT"
elif [[ -f "$FINAL" ]]; then
  CKPT="$FINAL"
elif [[ -f "$BEST" ]]; then
  CKPT="$BEST"
else
  echo "FATAL: missing O2 checkpoint under $OUT (best/final)" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "FATAL: checkpoint not found: $CKPT" >&2
  exit 1
fi
echo "[audit] using checkpoint: $CKPT"
echo "[audit] hard-R: $HARD_R"
echo "[audit] output: $OUT/information_source_audit/ (does not overwrite O2 metrics)"

echo "[audit] running frozen information-source audit gates A–G"
python scripts/assignment_diffusion_mvp/audit_orbit_o2_information_source.py \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --tie-break-mode all \
  --execute

echo "[audit] DONE"
echo "Report: $OUT/information_source_audit/information_source_audit.md"
echo "Summary: $OUT/information_source_audit/audit_summary.json"
