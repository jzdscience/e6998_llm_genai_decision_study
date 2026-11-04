#!/usr/bin/env bash
# Repair the pre-prediction Python.h incident, then test Kev alone.
set -euo pipefail
cd "$(dirname "$0")/.."
cleanup_legacy=false
if [[ "${1:-}" == "--cleanup-legacy-cache" ]]; then
  cleanup_legacy=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: bash scripts/recover_kev.sh [--cleanup-legacy-cache]" >&2
  exit 2
fi
: "${OUTPUT_DIR:=$PWD/results/solo-v1}"
: "${SESSION_HOURS_LEFT:=22}"
: "${MANAGED_PYTHON_VERSION:=3.12.8}"
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
if "$cleanup_legacy"; then
  "$PYTHON_BIN" scripts/cleanup_legacy_cache.py --apply
fi
"$PYTHON_BIN" -m decision_study repair-python \
  --models kev --managed-python-version "$MANAGED_PYTHON_VERSION" \
  --cost-mode provided --session-hours-left "$SESSION_HOURS_LEFT" \
  --output-dir "$OUTPUT_DIR"
"$PYTHON_BIN" -m decision_study run --job smoke --models kev \
  --cost-mode provided --session-hours-left "$SESSION_HOURS_LEFT" \
  --output-dir "$OUTPUT_DIR"
echo "Kev-only smoke finished. Return $OUTPUT_DIR/exports/pilot_bundle.zip."
echo "The pilot bundle remains partial until StartLux and Qwen also pass smoke."
