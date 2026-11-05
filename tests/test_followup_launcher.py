"""Run the actual Bash launchers against inert executables; never start models."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    source, runtime = tmp_path/'source with spaces', tmp_path/'runtime with spaces'
    (source/'scripts').mkdir(parents=True)
    runtime.mkdir()
    for name in ('run_followup_suite.sh', 'run_service_study.sh', 'run_size_control.sh'):
        shutil.copyfile(PROJECT/'scripts'/name, source/'scripts'/name)
    fake = '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['FIXTURE_LOG'],'a') as handle:
    handle.write(json.dumps({'cwd':os.getcwd(),'args':sys.argv[1:]})+'\\n')
if 'decision_study.service_study' in sys.argv and os.environ.get('FIXTURE_FAIL_SERVICE'):
    sys.exit(2)
'''
    for path in (source/'.venv-cheap-baselines/bin/python', runtime/'.venv-controller/bin/python'):
        path.parent.mkdir(parents=True)
        path.write_text(fake)
        path.chmod(0o755)
    (source/'.gitignore').write_text('.venv*/\n')
    subprocess.run(['git','init','-q',source],check=True)
    subprocess.run(['git','add','--all'],cwd=source,check=True)
    subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                    '-c','commit.gpgsign=false','commit','-qm','software fixture'],cwd=source,check=True)
    log=tmp_path/'calls.jsonl'
    def run(*args, failure=False):
        env={**os.environ,'FIXTURE_LOG':str(log)}
        if failure: env['FIXTURE_FAIL_SERVICE']='1'
        result=subprocess.run(['bash',str(source/'scripts/run_followup_suite.sh'),str(runtime),*args],
                              cwd=tmp_path,env=env,text=True,capture_output=True)
        calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result,calls
    return source,runtime,run


def test_first_run_has_no_empty_optional_arguments_and_uses_correct_source(launch):
    source, runtime, run=launch
    result,calls=run()
    assert result.returncode==0,result.stderr
    assert 'FOLLOWUPS_COMPLETE' in result.stdout
    assert len(calls)==3
    assert calls[0]['args']==['-m','pip','check']
    assert calls[1]['args'][:3]==['-m','decision_study.service_study','start']
    assert calls[2]['args'][:3]==['-m','decision_study.size_control','start']
    assert all(c['cwd']==str(source) and '' not in c['args'] for c in calls)
    assert str(runtime/'results/aux-clinc-service-v1-r2') in calls[1]['args']
    assert str(runtime/'results/aux-size-control-v1-r2') in calls[2]['args']


def test_resume_flags_apply_only_to_intended_pipeline(launch):
    _,_,run=launch
    result,calls=run('--new-allocation','--resume-interrupted')
    assert result.returncode==0,result.stderr
    assert calls[1]['args'][-2:]==['--new-allocation','--resume-interrupted']
    assert calls[2]['args'][-1]=='--new-allocation'
    assert '--resume-interrupted' not in calls[2]['args']


def test_service_failure_does_not_launch_size_job_or_claim_completion(launch):
    _,_,run=launch
    result,calls=run(failure=True)
    assert result.returncode==2
    assert len(calls)==2
    assert 'FOLLOWUPS_COMPLETE' not in result.stdout
