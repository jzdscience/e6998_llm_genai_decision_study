#!/usr/bin/env bash
# Direct-accuracy matrix completion (protocol direct-matrix-v1): 8 cells, 20,400 new test predictions.
# StartLux-9B and Qwen3.5-9B on CLINC / SQuAD Study 1 / SQuAD Study 2, Kev-4B on SQuAD Study 2 / ANLI.
# Existing runtimes, environments and checkpoints are reused read-only; only the new study root is written.
#
#   bash scripts/run_direct_matrix.sh RUNTIME_ROOT [--preflight-only] [--new-allocation] [--resume] [--retry-preflight]
set -euo pipefail
USAGE="Usage: bash scripts/run_direct_matrix.sh RUNTIME_ROOT [--preflight-only] [--new-allocation] [--resume] [--retry-preflight]"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$(cd -- "${1:?$USAGE}" && pwd)"
shift
COMMAND=start
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --preflight-only) COMMAND=preflight-only ;;
    --new-allocation|--resume|--retry-preflight) EXTRA+=("$arg") ;;
    *) echo "Unknown option: $arg" >&2; echo "$USAGE" >&2; exit 2 ;;
  esac
done
cd "$REPO_ROOT"
if [[ "$REPO_ROOT" == "$RUNTIME_ROOT" || -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "BLOCKED: run from the clean, independent decision-study-src checkout (not the old runtime root)." >&2
  exit 2
fi
CONTROLLER="${CONTROLLER_PYTHON:-$RUNTIME_ROOT/.venv-controller/bin/python}"
if [[ ! -x "$CONTROLLER" ]]; then
  echo "BLOCKED: existing controller environment missing at $CONTROLLER" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
echo "Source commit: $(git rev-parse HEAD)"
echo "Reading the actual current Slurm allowance; no hours variable or old session deadline is reused."
OUTPUT="$RUNTIME_ROOT/results/aux-direct-matrix-v1"
INPUTS=(--v1-dir "$RUNTIME_ROOT/results/solo-v1" --v2-dir "$RUNTIME_ROOT/results/solo-v2"
        --v3-dir "$RUNTIME_ROOT/results/solo-v3" --size-control-dir "$RUNTIME_ROOT/results/aux-size-control-v1-r2"
        --output-dir "$OUTPUT")
run() { "$CONTROLLER" -m decision_study.direct_matrix "$@"; }
if [[ "$COMMAND" == preflight-only ]]; then
  run prepare "${INPUTS[@]}"
  run init-runtime --output-dir "$OUTPUT" ${EXTRA[@]+"${EXTRA[@]}"}
  run preflight --output-dir "$OUTPUT" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "PREFLIGHT_COMPLETE — return $OUTPUT/exports/direct_matrix_bundle.zip for review before the formal run."
else
  run start "${INPUTS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "DIRECT_MATRIX_FINISHED — return $OUTPUT/exports/direct_matrix_bundle.zip for independent audit."
fi
