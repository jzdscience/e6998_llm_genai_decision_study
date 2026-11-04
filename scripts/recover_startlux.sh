#!/usr/bin/env bash
# Add the missing reference-only dependencies, then smoke both vLLM consumers.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/recover_startlux.sh" >&2
  exit 2
fi
: "${OUTPUT_DIR:=$PWD/results/solo-v1}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv-controller/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv-controller/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
if [[ -d "$PWD/.venv-controller/bin" ]]; then
  export PATH="$PWD/.venv-controller/bin:$PATH"
fi
"$PYTHON_BIN" scripts/repair_startlux_reference.py --output-dir "$OUTPUT_DIR"
"$PYTHON_BIN" -m decision_study run --job smoke --models startlux qwen \
  --cost-mode provided --output-dir "$OUTPUT_DIR"
echo "Return $OUTPUT_DIR/exports/pilot_bundle.zip."
