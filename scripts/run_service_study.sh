#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
CONTROLLER="${PYTHON_BIN:-$REPO_ROOT/.venv-controller/bin/python}"
if [[ ! -x "$CONTROLLER" ]]; then
  echo "Controller missing; set PYTHON_BIN to the existing runtime controller." >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$CONTROLLER" -m decision_study.service_study "$@"
