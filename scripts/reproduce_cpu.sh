#!/usr/bin/env bash
# Reproduce everything that does not need a GPU, model weights or the GPU node.
#
#   bash scripts/reproduce_cpu.sh                  # skips steps whose inputs are missing
#   bash scripts/reproduce_cpu.sh --strict         # every step must run; a skip counts as a failure
#   bash scripts/reproduce_cpu.sh --only audits-9b # run only the named steps (repeat or comma-separate)
#
# 1. tests        CPU test suite
# 2. lab          Decision Lab portable evidence check (V1-V3 cases and the follow-up page, no model calls)
# 3. audits-1106  Re-run of the 11/6 follow-up audits, compared byte for byte with the tracked receipts
# 4. audits-9b    Re-run of the 9B serving audits (integrity and raw metrics) and of the Lab follow-up
#                 bundle, compared byte for byte with reports/upgrade-9b-serving-2026-11-09/evidence/
#                 and reports/decision-lab-followups-2026-11-10/
#                 (steps 3 and 4 need the external evidence root; set EVIDENCE_ROOT if it is not the
#                 default sibling path)
# 5. report-9b    9B serving report tables (always compared byte for byte) and figure (compared when the
#                 pinned Matplotlib version is installed)
# 6. figures      Paper figures, compared byte for byte with paper/figures/*.pdf when the pinned Matplotlib
#                 version (paper/requirements-figures.txt) is installed
# 7. paper        Paper PDF, built in a temporary copy (needs pdflatex and bibtex)
#
# Outputs go to a fresh temporary directory; tracked files are never modified.
# Exit status is non-zero if any check that ran failed (or, with --strict, if any step was skipped).
set -uo pipefail

STRICT=0
ONLY=""
STEPS="tests lab audits-1106 audits-9b report-9b figures paper"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strict) STRICT=1 ;;
        --only)
            [[ $# -ge 2 ]] || { echo "--only needs a step name" >&2; exit 2; }
            ONLY="$ONLY ${2//,/ }"; shift ;;
        --only=*) value="${1#--only=}"; ONLY="$ONLY ${value//,/ }" ;;
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
for name in $ONLY; do
    [[ " $STEPS " == *" $name "* ]] || { echo "unknown step: $name (steps: $STEPS)" >&2; exit 2; }
done
want() { [[ -z "${ONLY// /}" || " $ONLY " == *" $1 "* ]]; }

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
# Byte equality of rendered figures only holds for the Matplotlib version that produced them.
PINNED_MPL="$(sed -n 's/^matplotlib==//p' paper/requirements-figures.txt)"
HAVE_MPL="$("$PY" -c 'import matplotlib; print(matplotlib.__version__)' 2>/dev/null)"

step() { printf '\n== %s\n' "$*"; }
fail() { echo "FAIL: $*"; FAILED+=("$*"); }
skip() {
    if [[ $STRICT -eq 1 ]]; then echo "FAIL (strict, would skip): $*"; FAILED+=("strict: $*")
    else echo "SKIP: $*"; SKIPPED+=("$*"); fi
}

if want tests; then
step "1/7 CPU test suite"
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
fi

if want lab; then
step "2/7 Decision Lab evidence check"
if "$PY" -m decision_study.lab check > "$OUT/lab-check.json" 2> "$OUT/lab-check.err"; then
    echo "OK, report in $OUT/lab-check.json"
else
    cat "$OUT/lab-check.err"; fail "lab check"
fi
fi

if want audits-1106; then
step "3/7 follow-up audits (service study, StartLux-9B size control)"
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
        # recheck_from_raw.py records its verdict in the file and exits 0 either way; check it here.
        if "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "PASS" else 1)' "$R/raw-recheck.json"; then
            echo "raw recheck status: PASS"
        else
            echo "raw recheck status: not PASS"; mismatches=$((mismatches + 1))
        fi
        [[ $mismatches -eq 0 ]] || fail "$mismatches audit checks failed"
    else
        fail "an audit script exited non-zero (logs in $OUT)"
    fi
else
    skip "external evidence root not found at $EVIDENCE_ROOT (set EVIDENCE_ROOT); see docs/ARTIFACTS_ZH.md"
fi
fi

if want audits-9b; then
step "4/7 9B serving audits (integrity, raw metrics) and Lab follow-up evidence"
# Inputs relative to the evidence root, as indexed in records/BUNDLE_INDEX.json.
NEEDED_9B="records/bundles/upgrade-formal-20261109-9baeca3eb7f8.zip
records/bundles/upgrade-preflight-r4-20261109-8adafb29ff45.zip
records/bundles/v3-return-20261103-232529-a8d0dc672d83.zip
records/bundles/size-control-complete-20261106-f427f7762bac.zip
results/solo-v3-prepared/sealed/holdout_gold.json"
MISSING_9B=""
while read -r relative; do
    [[ -f "$EVIDENCE_ROOT/$relative" ]] || MISSING_9B="$MISSING_9B $relative"
done <<< "$NEEDED_9B"
if [[ -n "$MISSING_9B" ]]; then
    skip "9B serving study: raw inputs missing under $EVIDENCE_ROOT:$MISSING_9B (set EVIDENCE_ROOT; see docs/ARTIFACTS_ZH.md)"
elif ! git -C "$ROOT" cat-file -e '7c81947d8ed51b18ace9ba8ae0dad0476c9b6b19^{commit}' 2> /dev/null; then
    skip "9B serving study: the integrity audit needs the git history with commit 7c81947 (use a git clone)"
else
    A=reports/upgrade-9b-serving-2026-11-09/audit
    E=reports/upgrade-9b-serving-2026-11-09/evidence
    R="$OUT/audits-9b"
    if "$PY" -I "$A/audit_integrity.py" --repo-root "$ROOT" --evidence-root "$EVIDENCE_ROOT" --output "$R/integrity" > "$OUT/audit-9b-integrity.log" 2>&1 \
       && "$PY" -I "$A/audit_metrics.py" --evidence-root "$EVIDENCE_ROOT" --output "$R/metrics" > "$OUT/audit-9b-metrics.log" 2>&1; then
        mismatches=0
        while read -r produced tracked; do
            if cmp -s "$R/$produced" "$E/$tracked"; then echo "same  $tracked"
            else echo "DIFF  $tracked"; mismatches=$((mismatches + 1)); fi
        done <<'PAIRS'
integrity/integrity-audit.json integrity-audit.json
metrics/audit-receipt.json metrics/audit-receipt.json
metrics/recomputed.json metrics/recomputed.json
metrics/cells.csv metrics/cells.csv
metrics/stall-analysis.json metrics/stall-analysis.json
metrics/stall_late_rows.csv metrics/stall_late_rows.csv
PAIRS
        # Both receipts carry their own verdict; a byte-identical copy of a failing receipt must still fail.
        if "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); sys.exit(0 if a["status"]==b["status"]=="PASS" and not b["discrepancies"] else 1)' \
             "$R/integrity/integrity-audit.json" "$R/metrics/audit-receipt.json"; then
            echo "integrity and metrics status: PASS ($(grep -c '"check":' "$R/integrity/integrity-audit.json") integrity checks)"
        else
            echo "integrity or metrics status: not PASS"; mismatches=$((mismatches + 1))
        fi
        if "$PY" -m decision_study.lab build-followups --evidence-root "$EVIDENCE_ROOT" --bundle "$OUT/lab-followups" --repo-root "$ROOT" \
             > "$OUT/lab-followups.log" 2>&1; then
            for f in followups.json manifest.json; do
                if cmp -s "$OUT/lab-followups/$f" "reports/decision-lab-followups-2026-11-10/$f"; then echo "same  decision-lab-followups-2026-11-10/$f"
                else echo "DIFF  decision-lab-followups-2026-11-10/$f"; mismatches=$((mismatches + 1)); fi
            done
        else
            tail -n 5 "$OUT/lab-followups.log"; mismatches=$((mismatches + 1))
        fi
        [[ $mismatches -eq 0 ]] || fail "$mismatches 9B serving checks failed (logs in $OUT)"
    else
        tail -n 5 "$OUT/audit-9b-integrity.log" "$OUT/audit-9b-metrics.log"
        fail "a 9B serving audit exited non-zero (logs in $OUT)"
    fi
fi
fi

if want report-9b; then
step "5/7 9B serving report tables and figure"
RP=reports/upgrade-9b-serving-2026-11-09
if "$PY" -I "$RP/build_tables.py" --out "$OUT/report-9b" > "$OUT/report-9b-tables.log" 2>&1; then
    tables_ok=1
    for f in summary.csv quality.csv resources.csv; do
        if cmp -s "$OUT/report-9b/$f" "$RP/$f"; then echo "same  $f"
        else echo "DIFF  $f"; tables_ok=0; fi
    done
    [[ $tables_ok -eq 1 ]] || fail "9B serving report tables differ from the tracked files"
    if "$PY" "$RP/render_figures.py" --tables "$OUT/report-9b" --out "$OUT/report-9b" > "$OUT/report-9b-figure.log" 2>&1; then
        if [[ "$HAVE_MPL" == "$PINNED_MPL" ]]; then
            for f in correct_goodput.png correct_goodput.svg; do
                if cmp -s "$OUT/report-9b/figures/$f" "$RP/figures/$f"; then echo "same  figures/$f"
                else echo "DIFF  figures/$f"; fail "report figure $f differs from the tracked file"; fi
            done
        else
            echo "report figure regenerated in $OUT/report-9b/figures"
            skip "byte comparison of the report figure needs matplotlib==$PINNED_MPL (installed: ${HAVE_MPL:-none})"
        fi
    else
        cat "$OUT/report-9b-figure.log"; fail "report figure script"
    fi
else
    cat "$OUT/report-9b-tables.log"; fail "report table script"
fi
fi

if want figures; then
step "6/7 paper figures"
if "$PY" paper/figures/make_figures.py --out "$OUT/figures" > "$OUT/figures.log" 2>&1; then
    tracked_figures="$(cd paper/figures && ls fig_*.pdf | sort)"
    made_figures="$(cd "$OUT/figures" && ls fig_*.pdf | sort)"
    if [[ "$tracked_figures" != "$made_figures" ]]; then
        fail "the figure script made [$(echo $made_figures)] but paper/figures holds [$(echo $tracked_figures)]"
    elif [[ "$HAVE_MPL" == "$PINNED_MPL" ]]; then
        for pdf in $tracked_figures; do
            if cmp -s "$OUT/figures/$pdf" "paper/figures/$pdf"; then echo "same  $pdf"
            else echo "DIFF  $pdf"; fail "figure ${pdf%.pdf} differs from the tracked PDF"; fi
        done
    else
        echo "figures regenerated in $OUT/figures"
        skip "byte comparison of figures needs matplotlib==$PINNED_MPL (installed: ${HAVE_MPL:-none}); see paper/requirements-figures.txt"
    fi
else
    cat "$OUT/figures.log"; fail "figure script"
fi
fi

if want paper; then
step "7/7 paper PDF"
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
fi

printf '\nOutputs: %s\n' "$OUT"
if [[ ${#SKIPPED[@]} -gt 0 ]]; then printf 'Skipped: %s\n' "${SKIPPED[@]}"; fi
if [[ ${#FAILED[@]} -gt 0 ]]; then printf 'Failed: %s\n' "${FAILED[@]}"; exit 1; fi
echo "All checks that ran passed."
