"""Runtime-repair release uses a fresh output without changing scientific rules."""
import os
from pathlib import Path
import subprocess

import pytest

from decision_study.common import StudyError
from decision_study.learnability import study


def test_repair_wrapper_uses_new_output_and_keeps_failed_run(tmp_path):
    runtime = tmp_path/'runtime'
    old = runtime/'results/aux-anli-learnability-v1/study.json'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'old failed run must remain intact')
    controller = runtime/'.venv-controller/bin/python'
    controller.parent.mkdir(parents=True)
    controller.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$ARG_CAPTURE"\n')
    controller.chmod(0o700)
    capture = tmp_path/'args'
    subprocess.run(['bash', str(study.PROJECT/'scripts/run_learnability.sh'), str(runtime)],
                   check=True, env={**os.environ, 'ARG_CAPTURE': str(capture)}, timeout=10)
    args = capture.read_text().splitlines()
    assert args[args.index('--output-dir')+1] == str(runtime/'results/aux-anli-learnability-v1-r2')
    assert old.read_bytes() == b'old failed run must remain intact'


def test_repair_and_legacy_outputs_validate_but_unrelated_results_do_not(tmp_path):
    runtime = tmp_path/'runtime'
    for name in ('aux-anli-learnability-v1', 'aux-anli-learnability-v1-r2'):
        study.validate_output(runtime/'results'/name, runtime)
    for out in (runtime/'results/solo-v1', runtime/'results/aux-anli-learnability-v1-unrelated',
                tmp_path/'aux-anli-learnability-v1-r2'):
        with pytest.raises(StudyError, match='separate'):
            study.validate_output(out, runtime)
