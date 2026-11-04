#!/usr/bin/env bash
# Expose already-installed build tools while keeping the controller interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -gt 1 ]]; then
  echo "Usage: bash scripts/run_with_vllm_tools.sh [smoke|startlux_full|qwen_test|benchmark]" >&2
  exit 2
fi
job="${1:-smoke}"
case "$job" in
  smoke|startlux_full|qwen_test|benchmark) ;;
  *) echo "Unsupported job: $job" >&2; exit 2 ;;
esac
: "${OUTPUT_DIR:=$PWD/results/solo-v1}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PWD/.venv-controller/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv-controller/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
# Resolve this before PATH changes; never let vLLM's Python become the controller.
controller_python="$(command -v "$PYTHON_BIN")"
controller_python="$("$controller_python" -c 'import os,sys; print(os.path.abspath(sys.executable))')"
vllm_tools="$("$controller_python" - "$OUTPUT_DIR" <<'PY'
import json
import os
from pathlib import Path
import sys

out = Path(sys.argv[1]).expanduser().resolve()
lock = json.loads((out / 'runtime/lock.json').read_text())
profiles = out / 'runtime/selected_profiles.json'
selected = json.loads(profiles.read_text()) if profiles.exists() else {}
if any(selected.get(m, 'primary') != 'primary' for m in ('startlux', 'qwen')):
    raise SystemExit('This launcher requires the recorded primary vLLM profiles.')
installed = json.loads((out / 'runtime/installations.json').read_text())
if installed['lock_hash'] != lock['lock_hash']:
    raise SystemExit('Runtime installation lock mismatch.')
name = 'vllm-' + lock['packages']['vllm_primary']
tools = Path(installed['environments'][name]['python']).parent
if not tools.is_absolute() or not os.access(tools / 'ninja', os.X_OK):
    raise SystemExit('Recorded vLLM bin/ninja is missing or not executable; no packages changed.')
print(tools)
PY
)"
if [[ -d "$PWD/.venv-controller/bin" ]]; then
  export PATH="$PWD/.venv-controller/bin:$PATH"
fi
export PATH="$vllm_tools:$PATH"
"$controller_python" - "$vllm_tools" <<'PY'
from pathlib import Path
import shutil
import subprocess
import sys

expected = Path(sys.argv[1]) / 'ninja'
found = shutil.which('ninja')
if found is None or Path(found).resolve() != expected.resolve():
    raise SystemExit('PATH did not select the recorded vLLM Ninja.')
result = subprocess.run(['ninja', '--version'], check=True, capture_output=True,
                        text=True, timeout=10)
print('Controller:', sys.executable, flush=True)
print('Ninja:', found, 'version', result.stdout.strip(), flush=True)
PY
if [[ "$job" == "smoke" ]]; then
  exec "$controller_python" -m decision_study run --job smoke --models qwen \
    --cost-mode provided --output-dir "$OUTPUT_DIR"
else
  exec "$controller_python" -m decision_study run --job "$job" \
    --cost-mode provided --output-dir "$OUTPUT_DIR"
fi
