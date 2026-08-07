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

python -m pytest mattergen/assignment/global_copy_assembly/tests/test_global_copy_assembly.py
python scripts/diagnostics/role_oracle_partition_diagnostic.py
python scripts/assignment_diffusion_mvp/export_geometry_only_hard_r.py
python scripts/assignment_diffusion_mvp/audit_geometry_only_hard_r.py
python scripts/assignment_diffusion_mvp/train_global_copy_assembly.py --config configs/assignment_diffusion_mvp/global_copy_assembly_clean_geometry_r.yaml --steps 5000 --output-dir outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r --execute
python scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly.py --config configs/assignment_diffusion_mvp/global_copy_assembly_clean_geometry_r.yaml --checkpoint outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/best_checkpoint.pt --execute
python scripts/assignment_diffusion_mvp/evaluate_global_copy_assembly.py --config configs/assignment_diffusion_mvp/global_copy_assembly_clean_geometry_r.yaml --checkpoint outputs/assignment_diffusion_mvp/global_copy_assembly_geometry_r/final_checkpoint.pt --execute
python scripts/assignment_diffusion_mvp/assemble_global_copy_assembly_report.py
