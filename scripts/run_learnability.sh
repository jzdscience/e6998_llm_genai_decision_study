#!/usr/bin/env bash
# One command continues after every successful gate and checkpoints at allocation limits.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$(cd -- "${1:?Usage: bash scripts/run_learnability.sh OLD_RUNTIME [--retry-failed]}" && pwd)"
shift
if [[ "$REPO_ROOT" == "$RUNTIME_ROOT" ]]; then
  echo 'BLOCKED: use the independent source checkout.' >&2; exit 2
fi
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --retry-failed) ARGS+=("$arg") ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
cd "$REPO_ROOT"
CONTROLLER="$RUNTIME_ROOT/.venv-controller/bin/python"
[[ -x "$CONTROLLER" ]] || { echo 'BLOCKED: existing controller missing' >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$CONTROLLER" -B -m decision_study.learnability start --runtime-root "$RUNTIME_ROOT" \
  --output-dir "$RUNTIME_ROOT/results/aux-anli-learnability-v1" ${ARGS[@]+"${ARGS[@]}"}
