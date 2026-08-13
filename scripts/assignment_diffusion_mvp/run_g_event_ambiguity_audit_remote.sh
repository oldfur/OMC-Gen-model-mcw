#!/usr/bin/env bash
# Small CPU audit: G-event target ambiguity (no GemNet, no training).
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
[[ -f outputs/assignment_diffusion_mvp/d1_fixed_clean_geometry/fixed_sample.pt ]] || exit 1

echo "[audit] G-event target ambiguity"
python scripts/assignment_diffusion_mvp/audit_g_event_target_ambiguity.py \
  --config "$CFG" \
  --output-dir "$OUT" \
  --n-seeds 12 \
  --max-events 300 \
  --seed0 2001 \
  --execute

echo "DONE G-ambiguity audit → $OUT/g_event_ambiguity_summary.json"
