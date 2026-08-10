#!/usr/bin/env bash
# N1 remote runner: MatterGen-native noise → orbit-aware assignment (NO geometry feedback).
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

echo "[1/5] preflight"
[[ -f "$SAMPLE" ]] || { echo "missing fixed sample"; exit 1; }
[[ -f "$HARD_R" ]] || { echo "missing geometry hard-R — run O2 export first"; exit 1; }
[[ -f "outputs/assignment_diffusion_mvp/role_automorphism_audit/role_orbits.json" ]] || { echo "missing role orbits"; exit 1; }

echo "[2/5] unit tests (N1 + O2 subset)"
python -m pytest \
  mattergen/assignment/noisy_copy_assignment/tests/test_noisy_copy_assignment_n1.py \
  -q

echo "[3/5] train N1 (frozen backbone, RESTARTED)"
python scripts/assignment_diffusion_mvp/train_noisy_copy_assignment_n1.py \
  --config "$CFG" --execute

echo "[4/5] recovery curves (oracle_orbit + predicted_orbit)"
python scripts/assignment_diffusion_mvp/evaluate_noisy_copy_assignment_curve_n1.py \
  --config "$CFG" \
  --checkpoint "$OUT/final_checkpoint.pt" \
  --execute

echo "[5/5] assemble report"
python scripts/assignment_diffusion_mvp/assemble_noisy_copy_assignment_n1_report.py \
  --output-dir "$OUT"

echo "DONE N1 → $OUT"
echo "See n1_report.md / oracle_orbit_curve.jsonl / predicted_orbit_curve.jsonl"
