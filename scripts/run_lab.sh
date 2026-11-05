#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -n "${DECISION_LAB_PYTHON:-}" ]]; then
    LAB_PYTHON="$DECISION_LAB_PYTHON"
elif [[ -x "$PROJECT_DIR/.venv-lab/bin/python" ]]; then
    LAB_PYTHON="$PROJECT_DIR/.venv-lab/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    LAB_PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    cat >&2 <<'EOF'
Create the local CPU environment once, from the repository root:
  python3 -m venv .venv-lab
  .venv-lab/bin/python -m pip install -e '.[lab]'
Then run: bash scripts/run_lab.sh
EOF
    exit 2
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export GRADIO_ANALYTICS_ENABLED=False
exec "$LAB_PYTHON" -m decision_study.lab serve "$@"
