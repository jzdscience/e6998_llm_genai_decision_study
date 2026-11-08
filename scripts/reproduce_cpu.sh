#!/usr/bin/env bash
# Reproduce everything that does not need a GPU, model weights or the GPU node.
#
#   bash scripts/reproduce_cpu.sh
#
# 1. CPU test suite
# 2. Decision Lab portable evidence check (V1-V3 cases, no model calls)
# 3. Re-run of the 11/6 follow-up audits, compared byte for byte with the tracked receipts
#    (needs the external evidence root; set EVIDENCE_ROOT if it is not the default sibling path)
# 4. Paper figures, compared byte for byte with paper/figures/*.pdf
# 5. Paper PDF, built in a temporary copy (needs pdflatex and bibtex)
#
# Outputs go to a fresh temporary directory; tracked files are never modified.
# Exit status is non-zero if any check that ran failed. Skipped steps are reported.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${DECISION_PYTHON:-}" ]]; then
    PY="$DECISION_PYTHON"
elif [[ -x .venv-lab/bin/python ]]; then
    PY=.venv-lab/bin/python
elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    echo "No environment found. Create one first:" >&2
    echo "  python3 -m venv .venv-lab && .venv-lab/bin/python -m pip install -e '.[lab,test]'" >&2
    exit 2
fi

EVIDENCE_ROOT="${EVIDENCE_ROOT:-$ROOT/../project/solo-experiments-2026-10-30/decision-study}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/decision-study-repro.XXXXXX")"
FAILED=()
SKIPPED=()

step() { printf '\n== %s\n' "$*"; }
fail() { echo "FAIL: $*"; FAILED+=("$*"); }
skip() { echo "SKIP: $*"; SKIPPED+=("$*"); }

step "1/5 CPU test suite"
if "$PY" -m pytest -q -p no:cacheprovider > "$OUT/pytest.log" 2>&1; then
    tail -n 1 "$OUT/pytest.log"
else
    tail -n 15 "$OUT/pytest.log"; fail "pytest (log: $OUT/pytest.log)"
fi
# The CLINC baseline tests need scikit-learn and ONNX Runtime, which live in a separate pinned
# environment (configs/cheap_baseline_requirements.txt). Run them there when it exists.
BASELINE_PY="${BASELINE_PYTHON:-.venv-cheap-baselines/bin/python}"
if [[ -x "$BASELINE_PY" ]]; then
    if "$BASELINE_PY" -m pytest -q -p no:cacheprovider tests/test_cheap_baselines.py tests/test_light_baseline.py \
         > "$OUT/pytest-baselines.log" 2>&1; then
        echo "baseline tests: $(tail -n 1 "$OUT/pytest-baselines.log")"
    else
        tail -n 15 "$OUT/pytest-baselines.log"; fail "baseline tests (log: $OUT/pytest-baselines.log)"
    fi
else
    skip "baseline tests need $BASELINE_PY (python3 -m venv .venv-cheap-baselines && .venv-cheap-baselines/bin/python -m pip install -r configs/cheap_baseline_requirements.txt)"
fi

step "2/5 Decision Lab evidence check"
if "$PY" -m decision_study.lab check > "$OUT/lab-check.json" 2> "$OUT/lab-check.err"; then
    echo "OK, report in $OUT/lab-check.json"
else
    cat "$OUT/lab-check.err"; fail "lab check"
fi

step "3/5 follow-up audits (service study, StartLux-9B size control)"
SERVICE_ZIP="$EVIDENCE_ROOT/records/bundles/service-complete-20261106-d212c130c707.zip"
SIZE_ZIP="$EVIDENCE_ROOT/records/bundles/size-control-complete-20261106-f427f7762bac.zip"
V3_ZIP="$EVIDENCE_ROOT/records/bundles/v3-return-20261103-232529-a8d0dc672d83.zip"
if [[ -f "$SERVICE_ZIP" && -f "$SIZE_ZIP" && -f "$V3_ZIP" ]]; then
    A=reports/followups-2026-11-06/audit
    E=reports/followups-2026-11-06/evidence
    R="$OUT/audits"
    if "$PY" "$A/audit_service.py" --archive "$SERVICE_ZIP" --evidence-root "$EVIDENCE_ROOT" --out "$R/service" > "$OUT/audit-service.log" 2>&1 \
       && "$PY" "$A/audit_supplement.py" --archive "$SERVICE_ZIP" --evidence-root "$EVIDENCE_ROOT" --out "$R/service-supplement" > "$OUT/audit-supplement.log" 2>&1 \
       && "$PY" "$A/audit_service_runtime.py" --repo-root "$ROOT" --evidence-root "$EVIDENCE_ROOT" \
            --result-root "$EVIDENCE_ROOT/results/returned-service-complete-2026-11-06" --out "$R/runtime" > "$OUT/audit-runtime.log" 2>&1 \
       && "$PY" "$A/audit_size_control.py" --bundle "$SIZE_ZIP" --v3-bundle "$V3_ZIP" --repo "$ROOT" --out "$R/size" > "$OUT/audit-size.log" 2>&1 \
       && "$PY" "$A/recheck_from_raw.py" \
            --service-root "$EVIDENCE_ROOT/results/returned-service-complete-2026-11-06" \
            --clinc-gold "$EVIDENCE_ROOT/results/returned-kev-2026-11-01/data/gold.json" \
            --size-root "$EVIDENCE_ROOT/results/returned-size-control-complete-2026-11-06" \
            --out "$R/raw-recheck.json" > "$OUT/audit-raw.log" 2>&1; then
        mismatches=0
        while read -r produced tracked; do
            if cmp -s "$R/$produced" "$E/$tracked"; then
                echo "same  $tracked"
            else
                echo "DIFF  $tracked"; mismatches=$((mismatches + 1))
            fi
        done <<'PAIRS'
service/aggregates.csv service/aggregates.csv
service/cells.csv service/cells.csv
service/prediction-consistency.json service/prediction-consistency.json
service/allocation-sensitivity.json service/allocation-sensitivity.json
service-supplement/supplemental-audit.json service/supplemental-audit.json
runtime/service-runtime-audit.json service/service-runtime-audit.json
runtime/resource-summary.csv service/resource-summary.csv
size/identity-audit.json size/identity-audit.json
size/runtime-audit.json size/runtime-audit.json
raw-recheck.json raw-recheck-2026-11-07.json
PAIRS
        # Two receipts record their run context: the archive path passed in, and the numpy version.
        # Compare them with that one field removed.
        for spec in "service/metrics-audit.json:service/metrics-audit.json:archive" \
                    "size/metrics-audit.json:size/metrics-audit.json:numpy_version"; do
            IFS=: read -r produced tracked field <<< "$spec"
            if "$PY" - "$R/$produced" "$E/$tracked" "$field" <<'PYEOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
a.pop(sys.argv[3], None); b.pop(sys.argv[3], None)
sys.exit(0 if a == b else 1)
PYEOF
            then echo "same  $tracked ($field excluded)"
            else echo "DIFF  $tracked"; mismatches=$((mismatches + 1)); fi
        done
        [[ $mismatches -eq 0 ]] || fail "$mismatches audit outputs differ from the tracked receipts"
    else
        fail "an audit script exited non-zero (logs in $OUT)"
    fi
else
    skip "external evidence root not found at $EVIDENCE_ROOT (set EVIDENCE_ROOT); see docs/ARTIFACTS_ZH.md"
fi

step "4/5 paper figures"
if "$PY" paper/figures/make_figures.py --out "$OUT/figures" > "$OUT/figures.log" 2>&1; then
    for f in fig_rescue_harm fig_budget_transfer fig_serving; do
        if cmp -s "$OUT/figures/$f.pdf" "paper/figures/$f.pdf"; then echo "same  $f.pdf"
        else echo "DIFF  $f.pdf"; fail "figure $f differs from the tracked PDF"; fi
    done
else
    cat "$OUT/figures.log"; fail "figure script"
fi

step "5/5 paper PDF"
if command -v pdflatex > /dev/null && command -v bibtex > /dev/null; then
    cp -R paper "$OUT/paper"
    if (cd "$OUT/paper" && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
          && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
          && pdflatex -interaction=nonstopmode -halt-on-error main.tex) > "$OUT/paper-build.log" 2>&1; then
        if grep -qE "undefined|Overfull" "$OUT/paper/main.log"; then
            grep -E "undefined|Overfull" "$OUT/paper/main.log" | head -5; fail "paper build has warnings"
        else
            echo "OK, built $OUT/paper/main.pdf"
        fi
    else
        tail -n 20 "$OUT/paper-build.log"; fail "paper build (log: $OUT/paper-build.log)"
    fi
else
    skip "pdflatex or bibtex not installed"
fi

printf '\nOutputs: %s\n' "$OUT"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then printf 'Skipped: %s\n' "${SKIPPED[@]}"; fi
if [[ ${#FAILED[@]} -gt 0 ]]; then printf 'Failed: %s\n' "${FAILED[@]}"; exit 1; fi
echo "All checks that ran passed."
