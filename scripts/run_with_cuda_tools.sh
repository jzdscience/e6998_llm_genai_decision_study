#!/usr/bin/env bash
# Keep the controller interpreter; the Python helper runs probes in the model env.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv-controller/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv-controller/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
controller_python="$(command -v "$PYTHON_BIN")"
exec "$controller_python" scripts/run_with_cuda_tools.py "$@"
