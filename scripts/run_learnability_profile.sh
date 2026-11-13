#!/usr/bin/env bash
# Only old V3 train-development inputs; no training, holdout scoring or service SLO study.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$(cd -- "${1:?Usage: bash scripts/run_learnability_profile.sh RUNTIME_ROOT [--new-allocation] [--retry-failed]}" && pwd)"
shift
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --new-allocation|--retry-failed) EXTRA+=("$arg") ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
if [[ "$REPO_ROOT" == "$RUNTIME_ROOT" ]]; then
  echo "BLOCKED: use an independent source checkout; the old runtime root stays read-only." >&2
  exit 2
fi
cd "$REPO_ROOT"
CONTROLLER="$RUNTIME_ROOT/.venv-controller/bin/python"
[[ -x "$CONTROLLER" ]] || { echo "BLOCKED: existing controller is missing" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1
"$CONTROLLER" -B -m decision_study.learnability_profile start --runtime-root "$RUNTIME_ROOT" \
  --output-dir "$RUNTIME_ROOT/results/aux-anli-learnability-profile-v1" ${EXTRA[@]+"${EXTRA[@]}"}
