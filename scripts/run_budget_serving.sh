#!/usr/bin/env bash
# One resumable command for anli-budget-serving-v1: verify pinned inputs, actual Slurm allocation and
# native process ownership, token and co-residency preflights, all 60 fixed cells, freeze, gold,
# evaluation, audit and export. Successful gates continue; allocation limits checkpoint; real
# failures stop with a diagnostic bundle. Rerun the same command in a new allocation to resume.
set -euo pipefail
USAGE='Usage: bash scripts/run_budget_serving.sh OLD_RUNTIME_ROOT [--retry-failed] [--output-dir DIR]'
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ $# -ge 1 ]] || { echo "$USAGE" >&2; exit 2; }
RUNTIME_ROOT="$(cd -- "$1" && pwd -P)"
shift
if [[ "$REPO_ROOT" == "$RUNTIME_ROOT" || "$REPO_ROOT" == "$RUNTIME_ROOT"/* || "$RUNTIME_ROOT" == "$REPO_ROOT"/* ]]; then
  echo 'BLOCKED: run from the independent source checkout, not inside the old runtime root.' >&2; exit 2
fi
OUTPUT="$RUNTIME_ROOT/results/aux-anli-budget-serving-v1"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --retry-failed) ARGS+=("$1") ;;
    --output-dir) [[ $# -ge 2 ]] || { echo "$USAGE" >&2; exit 2; }; OUTPUT="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; echo "$USAGE" >&2; exit 2 ;;
  esac
  shift
done
CONTROLLER="$RUNTIME_ROOT/.venv-controller/bin/python"
[[ -x "$CONTROLLER" ]] || { echo 'BLOCKED: the existing controller interpreter is missing' >&2; exit 2; }
cd "$REPO_ROOT"
# No ambient import path, user site, startup hook, manual hours or deadline variable reaches the study.
unset PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE PYTHONEXECUTABLE PYTHONSAFEPATH
unset DECISION_STUDY_DEADLINE_EPOCH DECISION_BUDGET_WORKER_OUTPUT DECISION_BUDGET_WORKER_ATTEMPT
unset DECISION_STUDY_RUNTIME_DIR SESSION_HOURS_LEFT HOURS_LEFT
export PYTHONPATH="$REPO_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
RESOLVED="$("$CONTROLLER" -B -s -c 'import decision_study, pathlib; print(pathlib.Path(decision_study.__file__).resolve().parents[1])')"
if [[ "$RESOLVED" != "$REPO_ROOT" ]]; then
  echo "BLOCKED: decision_study resolves to $RESOLVED, not this checkout $REPO_ROOT" >&2; exit 2
fi
exec "$CONTROLLER" -B -s -m decision_study.budget_serving start --runtime-root "$RUNTIME_ROOT" \
  --output-dir "$OUTPUT" ${ARGS[@]+"${ARGS[@]}"}
