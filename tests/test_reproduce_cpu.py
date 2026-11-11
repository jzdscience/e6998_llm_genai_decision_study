"""scripts/reproduce_cpu.sh must name a missing 9B serving input, and --strict must fail on it.

Runs only the targeted step against an empty evidence root, so it needs no external artifacts and no
GPU, and it never reaches an audit or a tracked file.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(tmp_path, *args):
    env = {**os.environ, "EVIDENCE_ROOT": str(tmp_path / "evidence"), "DECISION_PYTHON": sys.executable,
           "TMPDIR": str(tmp_path)}
    (tmp_path / "evidence/records/bundles").mkdir(parents=True, exist_ok=True)
    return subprocess.run(["bash", str(REPO / "scripts/reproduce_cpu.sh"), *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=180)


def test_missing_9b_inputs_are_named_in_a_visible_skip(tmp_path):
    relaxed = _run(tmp_path, "--only", "audits-9b")
    assert relaxed.returncode == 0, relaxed.stdout + relaxed.stderr
    assert "SKIP: 9B serving study: raw inputs missing" in relaxed.stdout
    assert "upgrade-formal-20261109-9baeca3eb7f8.zip" in relaxed.stdout
    assert "holdout_gold.json" in relaxed.stdout


def test_missing_9b_inputs_fail_strict_mode(tmp_path):
    strict = _run(tmp_path, "--strict", "--only", "audits-9b")
    assert strict.returncode == 1, strict.stdout + strict.stderr
    assert "FAIL (strict, would skip): 9B serving study" in strict.stdout
    assert "Failed: strict: 9B serving study" in strict.stdout


def test_only_rejects_unknown_steps(tmp_path):
    result = _run(tmp_path, "--only", "everything")
    assert result.returncode == 2
    assert "unknown step" in result.stderr
