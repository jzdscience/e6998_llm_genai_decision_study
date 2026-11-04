#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OUTPUT_DIR:=$PWD/results/solo-v1}"
: "${COST_MODE:=provided}"
: "${HOURLY_USD:=0}"
: "${SESSION_HOURS_LEFT:=22}"
: "${JOB:=all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m pip install -e . uv
if [[ "$JOB" == "all" ]]; then
  for task in smoke kev_full startlux_full qwen_test; do
    "$PYTHON_BIN" -m decision_study run --job "$task" --output-dir "$OUTPUT_DIR" --hourly-usd "$HOURLY_USD" --cost-mode "$COST_MODE" --session-hours-left "$SESSION_HOURS_LEFT"
  done
  "$PYTHON_BIN" -m decision_study evaluate --output-dir "$OUTPUT_DIR"
  "$PYTHON_BIN" -m decision_study export --kind experiment --output-dir "$OUTPUT_DIR"
  # Timing is a deliberate separate invocation after reviewing the experiment bundle.
  echo "Accuracy run finished. Return experiment_bundle.zip for review; run JOB=benchmark for real timing."
else
  "$PYTHON_BIN" -m decision_study run --job "$JOB" --output-dir "$OUTPUT_DIR" --hourly-usd "$HOURLY_USD" --cost-mode "$COST_MODE" --session-hours-left "$SESSION_HOURS_LEFT"
fi
echo "Stop your cloud VM in its provider console. Python exiting does not stop VM billing."
