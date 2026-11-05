#!/usr/bin/env bash
# Fixed three-classifier CPU experiment in a new checkout and separate result root.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$(cd "${1:?Usage: bash scripts/run_cheap_baseline_suite.sh /path/to/old/decision-study}" && pwd)"
cd "$REPO_ROOT"
if [[ "$RUNTIME_ROOT" == "$REPO_ROOT" ]]; then
  echo "BLOCKED: source checkout must be separate from the old runtime." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "BLOCKED: commit or preserve source changes before running the fixed experiment." >&2
  git status --short
  exit 2
fi
CONTROLLER="$RUNTIME_ROOT/.venv-controller/bin/python"
"$CONTROLLER" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), "Use Python 3.11–3.13"'
"$CONTROLLER" -m venv "$REPO_ROOT/.venv-cheap-baselines"
"$REPO_ROOT/.venv-cheap-baselines/bin/python" -m pip install --only-binary=:all: -r configs/cheap_baseline_requirements.txt
"$REPO_ROOT/.venv-cheap-baselines/bin/python" -m pip check
INPUT_ROOT="$RUNTIME_ROOT/results/aux-clinc-cheap-v1-inputs"
OUTPUT_ROOT="$RUNTIME_ROOT/results/aux-clinc-cheap-v1"
V1_ROOT="$RUNTIME_ROOT/results/solo-v1"
V2_ROOT="$REPO_ROOT/results/solo-v2"
echo "Preparing fixed training inputs"
bash scripts/run_cheap_baselines.sh prepare --download \
  --raw-dir "$INPUT_ROOT/raw" --prepared-dir "$INPUT_ROOT/prepared" \
  --v1-dir "$V1_ROOT" --v2-dir "$V2_ROOT"
echo "Verifying/downloading the fixed CPU encoder"
bash scripts/run_cheap_baselines.sh download-model --model-dir "$INPUT_ROOT/model"
echo "Checking data, model and dependency identities"
bash scripts/run_cheap_baselines.sh preflight \
  --prepared-dir "$INPUT_ROOT/prepared" --model-dir "$INPUT_ROOT/model" \
  --v1-dir "$V1_ROOT" --v2-dir "$V2_ROOT"
echo "Running three classifiers, then freezing and evaluating the complete suite"
bash scripts/run_cheap_baselines.sh run \
  --prepared-dir "$INPUT_ROOT/prepared" --model-dir "$INPUT_ROOT/model" \
  --v1-dir "$V1_ROOT" --v2-dir "$V2_ROOT" --output-dir "$OUTPUT_ROOT"
echo "RESULT BUNDLE: $OUTPUT_ROOT/exports/cheap_baseline_bundle.zip"
