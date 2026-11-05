#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
CONTROLLER="${PYTHON_BIN:-$PROJECT_DIR/.venv-controller/bin/python}"
if [[ ! -x "$CONTROLLER" ]]; then
  echo "Controller missing: $CONTROLLER. Set PYTHON_BIN to the existing runtime controller." >&2
  exit 2
fi
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/decision-study-size-control-pycache}"
exec "$CONTROLLER" -m decision_study.size_control "$@"
