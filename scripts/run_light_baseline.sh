#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
BASELINE_PYTHON="${BASELINE_PYTHON:-$PROJECT_DIR/.venv-baselines/bin/python}"
if [[ ! -x "$BASELINE_PYTHON" ]]; then
  echo "Missing isolated CPU environment: $BASELINE_PYTHON. See docs/RUN_LIGHT_BASELINE_ZH.md." >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec "$BASELINE_PYTHON" -m decision_study.baselines "$@"
