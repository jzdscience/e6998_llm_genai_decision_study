#!/usr/bin/env bash
# ANLI serving study: StartLux-4B, StartLux-9B, and StartLux-4B -> Qwen3.5-9B cascade.
# Existing runtimes, environments and checkpoints are reused read-only; only the new study root is written.
#
#   bash scripts/run_upgrade_study.sh RUNTIME_ROOT [--preflight-only] [--new-allocation] [--resume-interrupted] [--retry-preflight]
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="$(cd -- "${1:?Usage: bash scripts/run_upgrade_study.sh RUNTIME_ROOT [--preflight-only] [--new-allocation] [--resume-interrupted] [--retry-preflight]}" && pwd)"
shift
COMMAND=start
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --preflight-only) COMMAND=preflight-only ;;
    --new-allocation|--resume-interrupted|--retry-preflight) EXTRA+=("$arg") ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
cd "$REPO_ROOT"
if [[ "$REPO_ROOT" == "$RUNTIME_ROOT" || -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "BLOCKED: use the clean independent decision-study-src checkout." >&2
  exit 2
fi
CONTROLLER="$RUNTIME_ROOT/.venv-controller/bin/python"
if [[ ! -x "$CONTROLLER" ]]; then
  echo "BLOCKED: existing controller environment missing at $CONTROLLER" >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
echo "Source commit: $(git rev-parse HEAD)"
echo "Reading the actual current Slurm allowance; no hours variable or old session deadline reused."
OUTPUT="$RUNTIME_ROOT/results/aux-anli-upgrade-service-v1-r2"
INPUTS=(--v1-dir "$RUNTIME_ROOT/results/solo-v1" --v3-dir "$RUNTIME_ROOT/results/solo-v3"
        --size-control-dir "$RUNTIME_ROOT/results/aux-size-control-v1-r2" --output-dir "$OUTPUT")
run() { "$CONTROLLER" -m decision_study.upgrade_study "$@"; }
if [[ "$COMMAND" == preflight-only ]]; then
  run prepare "${INPUTS[@]}"
  run init-runtime --output-dir "$OUTPUT" ${EXTRA[@]+"${EXTRA[@]}"}
  run preflight --output-dir "$OUTPUT" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "PREFLIGHT_COMPLETE — return $OUTPUT/exports/upgrade_study_bundle.zip for review before the formal run."
else
  run start "${INPUTS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
  echo "UPGRADE_STUDY_FINISHED — return $OUTPUT/exports/upgrade_study_bundle.zip for independent audit."
fi
