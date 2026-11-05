#!/usr/bin/env bash
# Existing models/environment stay in place. Only new study roots are written.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$(cd -- "${1:?Usage: bash scripts/run_followup_suite.sh OLD_RUNTIME [--new-allocation] [--resume-interrupted]}" && pwd)"
shift
NEW_ALLOCATION=0
RESUME=0
for arg in "$@"; do
  case "$arg" in
    --new-allocation) NEW_ALLOCATION=1 ;;
    --resume-interrupted) RESUME=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
cd "$REPO_ROOT"
if [[ "$REPO_ROOT" == "$RUNTIME_ROOT" || -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "BLOCKED: use the clean independent decision-study-src checkout." >&2
  exit 2
fi
export PYTHON_BIN="$RUNTIME_ROOT/.venv-controller/bin/python"
CPU_PYTHON="$REPO_ROOT/.venv-cheap-baselines/bin/python"
if [[ ! -x "$PYTHON_BIN" || ! -x "$CPU_PYTHON" ]]; then
  echo "BLOCKED: existing controller and accepted cheap-baseline environments are required; see docs/RUN_FOLLOWUPS_ZH.md." >&2
  exit 2
fi
"$CPU_PYTHON" -m pip check
echo "Source commit: $(git rev-parse HEAD)"
echo "Reading actual current Slurm allowance; no hours variable or old session deadline reused."
SERVICE_ARGS=(start \
  --v1-dir "$RUNTIME_ROOT/results/solo-v1" \
  --bundle-dir "$RUNTIME_ROOT/results/aux-clinc-cheap-v1" \
  --model-dir "$RUNTIME_ROOT/results/aux-clinc-cheap-v1-inputs/model" \
  --cpu-python "$CPU_PYTHON" \
  --output-dir "$RUNTIME_ROOT/results/aux-clinc-service-v1")
SIZE_ARGS=(start \
  --v1-dir "$RUNTIME_ROOT/results/solo-v1" \
  --v3-dir "$RUNTIME_ROOT/results/solo-v3" \
  --output-dir "$RUNTIME_ROOT/results/aux-size-control-v1")
if [[ "$NEW_ALLOCATION" == 1 ]]; then
  SERVICE_ARGS+=(--new-allocation)
  SIZE_ARGS+=(--new-allocation)
fi
if [[ "$RESUME" == 1 ]]; then
  SERVICE_ARGS+=(--resume-interrupted)
fi
bash scripts/run_service_study.sh "${SERVICE_ARGS[@]}"
echo "Service experiment complete; starting standalone StartLux9B quality control."
bash scripts/run_size_control.sh "${SIZE_ARGS[@]}"
echo "FOLLOWUPS_COMPLETE — both pipelines finished; return both final bundles for independent audit."
echo "$RUNTIME_ROOT/results/aux-clinc-service-v1/exports/service_study_bundle.zip"
echo "$RUNTIME_ROOT/results/aux-size-control-v1/exports/size_control_bundle.zip"
