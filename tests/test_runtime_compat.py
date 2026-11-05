"""Reproduce cwd-visible pip metadata without changing any real environment."""
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import venv

import pytest

import decision_study
from decision_study import runtime as frozen
from decision_study import runtime_compat as compat
from decision_study.common import StudyError, write_json
from decision_study.service_study import runtime as service
from decision_study.size_control import runtime as size

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def layout(tmp_path):
    old = tmp_path / 'original-project'
    v1 = old / 'results/solo-v1'
    work = v1 / 'runtime/work'
    work.mkdir(parents=True)
    output = tmp_path / 'new-output'
    binding = {'v1_dir': str(v1), 'source_work_dir': str(work),
               'work_dir': str(output / 'runtime-work')}
    return old, v1, work, output, binding


def test_real_frozen_verifier_preserves_exact_hash_and_rejects_real_package_drift(layout, monkeypatch):
    old, _, _, output, binding = layout
    monkeypatch.delenv('DECISION_STUDY_DEADLINE_EPOCH', raising=False)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    prefix = output / 'software-fixture-venv'
    venv.EnvBuilder(with_pip=True).create(prefix)
    python = prefix / 'bin/python'
    site = Path(subprocess.check_output([python, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    dist = site / 'compat_drift_fixture-1.0.dist-info'
    dist.mkdir()
    metadata = dist / 'METADATA'
    metadata.write_text('Metadata-Version: 2.1\nName: compat-drift-fixture\nVersion: 1.0\n')
    egg = old / 'decision_study.egg-info'
    egg.mkdir()
    (egg / 'PKG-INFO').write_text('Metadata-Version: 2.1\nName: decision-study\nVersion: 0.1.0\n')
    # Model the original metadata visibility explicitly, independently of local
    # pip's treatment of cwd. Production must recover it without inherited paths.
    original_env = {**os.environ, 'PYTHONPATH': str(old)}
    freeze = lambda cwd, env=None: subprocess.check_output(
        [python, '-m', 'pip', 'freeze', '--all'], cwd=cwd, env=env, text=True)
    expected = freeze(old, original_env)
    neutral = output / 'neutral'
    neutral.mkdir()
    outside = freeze(neutral)
    assert expected.splitlines() == sorted(expected.splitlines(), key=str.lower)
    assert 'decision-study==0.1.0\n' in expected
    assert 'decision-study' not in outside
    assert outside == expected.replace('decision-study==0.1.0\n', '')
    info = {'python': str(python), 'freeze_sha256': hashlib.sha256(expected.encode()).hexdigest()}
    identity_before = dict(info)
    original_metadata = (egg / 'PKG-INFO').read_bytes()
    monkeypatch.chdir(neutral)
    with pytest.raises(RuntimeError, match='dependencies changed'):
        frozen._verify_environment(output, info)
    with compat.legacy_environment(binding, PROJECT):
        frozen._verify_environment(output, info)
        assert Path.cwd() == PROJECT
        # A real change in the installed distribution's metadata is not ignored.
        metadata.write_text('Metadata-Version: 2.1\nName: compat-drift-fixture\nVersion: 2.0\n')
        dist.rename(site / 'compat_drift_fixture-2.0.dist-info')
        with pytest.raises(RuntimeError, match='dependencies changed'):
            frozen._verify_environment(output, info)
    assert Path.cwd() == neutral
    assert info == identity_before
    assert (egg / 'PKG-INFO').read_bytes() == original_metadata
    assert 'PYTHONPATH' not in os.environ


def test_lazy_imports_stay_in_new_checkout_and_state_is_restored(layout, monkeypatch, tmp_path):
    old, _, _, _, binding = layout
    (old / 'scripts').mkdir()
    (old / 'scripts/prepare_cuda_toolkit.py').write_text('raise AssertionError("OLD SOURCE IMPORTED")\n')
    caller = tmp_path / 'caller'
    relative = caller / 'relative-fixture-path'
    relative.mkdir(parents=True)
    (relative / 'runtime_compat_relative_fixture.py').write_text('ORIGIN = "caller"\n')
    monkeypatch.chdir(caller)
    original = ['', 'relative-fixture-path', *sys.path]
    monkeypatch.setattr(sys, 'path', original.copy())
    modules = {name: sys.modules.pop(name) for name in list(sys.modules)
               if name == 'scripts' or name.startswith('scripts.')}
    try:
        with compat.legacy_environment(binding, PROJECT):
            assert Path.cwd() == PROJECT
            loaded = importlib.import_module('scripts.prepare_cuda_toolkit')
            assert Path(loaded.__file__).resolve() == PROJECT / 'scripts/prepare_cuda_toolkit.py'
            assert importlib.import_module('runtime_compat_relative_fixture').ORIGIN == 'caller'
            assert sys.path[0] == str(PROJECT)
            assert str(relative) in sys.path
            assert Path(decision_study.__file__).resolve().parent == PROJECT / 'decision_study'
        assert Path.cwd() == caller
        assert sys.path == original
    finally:
        for name in list(sys.modules):
            if name == 'scripts' or name.startswith('scripts.'):
                sys.modules.pop(name)
        sys.modules.update(modules)
        sys.modules.pop('runtime_compat_relative_fixture', None)


def test_exception_and_nested_contexts_restore_cwd_and_import_path(layout, tmp_path, monkeypatch):
    old, _, _, _, binding = layout
    second = tmp_path / 'second-project/results/solo-v1'
    second_work = second / 'runtime/work'
    second_work.mkdir(parents=True)
    second_binding = {**binding, 'v1_dir': str(second), 'source_work_dir': str(second_work)}
    caller = tmp_path / 'exception-caller'
    caller.mkdir()
    monkeypatch.chdir(caller)
    previous_cwd, previous_path, previous_env = Path.cwd(), list(sys.path), dict(os.environ)
    with pytest.raises(ValueError, match='intentional fixture error'):
        with compat.legacy_environment(binding, PROJECT):
            outer_path = list(sys.path)
            outer_env = dict(os.environ)
            assert Path.cwd() == PROJECT
            with compat.legacy_environment(second_binding, PROJECT):
                assert Path.cwd() == PROJECT
                assert os.environ['PYTHONPATH'].split(os.pathsep)[:2] == [str(PROJECT), str(second.parent.parent)]
            assert Path.cwd() == PROJECT
            assert sys.path == outer_path
            assert dict(os.environ) == outer_env
            raise ValueError('intentional fixture error')
    assert Path.cwd() == previous_cwd
    assert sys.path == previous_path
    assert dict(os.environ) == previous_env


def test_wrong_import_and_unknown_runtime_layout_block_before_chdir(layout, monkeypatch):
    _, v1, _, _, binding = layout
    before = Path.cwd(), list(sys.path)
    for bad in ({**binding, 'source_work_dir': str(v1 / 'different-work')},
                {**binding, 'v1_dir': str(v1.parent.parent / 'not-results/solo-v1')}):
        with pytest.raises(StudyError, match='runtime layout'):
            with compat.legacy_environment(bad, PROJECT):
                pytest.fail('entered invalid runtime layout')
    for module, filename in ((decision_study, '__init__.py'), (frozen, 'runtime.py')):
        with monkeypatch.context() as patch:
            patch.setattr(module, '__file__', str(v1.parent.parent / 'decision_study' / filename))
            with pytest.raises(StudyError, match='Controller imports'):
                with compat.legacy_environment(binding, PROJECT):
                    pytest.fail('entered old-controller context')
    assert (Path.cwd(), sys.path) == before


def test_cwd_change_is_not_allowed_from_a_worker_thread(layout):
    binding = layout[-1]
    errors = []
    before = Path.cwd(), list(sys.path)
    def worker():
        try:
            with compat.legacy_environment(binding, PROJECT):
                errors.append('context incorrectly entered')
        except StudyError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=worker)
    thread.start(); thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(errors) == 1 and 'main thread' in errors[0]
    assert (Path.cwd(), sys.path) == before


def test_service_cuda_token_and_managed_paths_all_use_legacy_metadata_context(layout, monkeypatch):
    old, _, _, output, binding = layout
    control = output / 'runtime-control'
    session = {'deadline_epoch': time.time() + 120}
    installed = {'environments': {'synthetic': {'python': 'not-executed'}}}
    write_json(control / 'runtime/last_services.json', {'processes': [{'pid': 4242}]})
    monkeypatch.setattr(service, 'validate', lambda _: (control, binding, session, installed))
    monkeypatch.setattr(service.frozen, 'doctor', lambda *a, **k: {'gpu': [{'name': 'NVIDIA L40S', 'memory_mb': 46068}]})
    seen = []
    def check(name, result):
        def call(*args, **kwargs):
            assert Path.cwd() == PROJECT
            assert os.environ['PYTHONPATH'].split(os.pathsep)[:2] == [str(PROJECT), str(old)]
            seen.append(name)
            return result
        return call
    monkeypatch.setattr(service.shared, '_cuda_preflight', check('cuda', {}))
    monkeypatch.setattr(service.frozen, 'token_preflight_examples', check('tokens', {'status': 'software-fixture'}))
    monkeypatch.setattr(service.frozen, 'runtime_config', lambda *args: {})
    @contextmanager
    def managed(*args, **kwargs):
        assert Path.cwd() == PROJECT
        seen.append('service-start')
        try:
            yield {'startlux': 'http://127.0.0.1:1'}
        finally:
            assert Path.cwd() == PROJECT
            assert os.environ['PYTHONPATH'].split(os.pathsep)[:2] == [str(PROJECT), str(old)]
            seen.append('service-stop')
    monkeypatch.setattr(service.frozen, 'managed_services', managed)
    before = Path.cwd(), list(sys.path), dict(os.environ)
    with service.gpu_service(output, 'startlux4b', [{'software_fixture': True}]) as value:
        assert Path.cwd() == PROJECT
        assert value['model'] == 'startlux'
    assert seen == ['cuda', 'tokens', 'service-start', 'service-stop']
    assert (Path.cwd(), sys.path, dict(os.environ)) == before


def test_size_environment_applies_cwd_without_rebinding_explicit_script_source(layout):
    old, _, _, _, binding = layout
    session = {'deadline_epoch': time.time() + 120}
    before = Path.cwd(), list(sys.path), dict(os.environ)
    with size._environment(binding, session):
        assert Path.cwd() == PROJECT
        assert os.environ['PYTHONPATH'].split(os.pathsep)[:2] == [str(PROJECT), str(old)]
        assert sys.path[0] == str(PROJECT)
        assert size.PROJECT == PROJECT
        assert os.environ['DECISION_STUDY_RUNTIME_DIR'] == binding['work_dir']
    assert (Path.cwd(), sys.path, dict(os.environ)) == before


def test_child_imports_new_source_and_keeps_prior_relative_path_meaning(layout, monkeypatch, tmp_path):
    old, _, _, output, binding = layout
    (old / 'decision_study').mkdir()
    (old / 'decision_study/__init__.py').write_text('raise AssertionError("OLD SOURCE IMPORTED")\n')
    caller = tmp_path / 'child-caller'
    relative = caller / 'relative-child-path'
    relative.mkdir(parents=True)
    (relative / 'runtime_compat_child_fixture.py').write_text('ORIGIN = "original caller"\n')
    (caller / 'decision_study').mkdir()
    (caller / 'decision_study/__init__.py').write_text('raise AssertionError("CALLER SHADOW IMPORTED")\n')
    monkeypatch.chdir(caller)
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(['relative-child-path', '', str(old)]))
    before = Path.cwd(), list(sys.path), dict(os.environ)
    script = ('import json, decision_study, runtime_compat_child_fixture as fixture; '
              'print(json.dumps({"source": decision_study.__file__, "relative": fixture.ORIGIN}))')
    with compat.legacy_environment(binding, PROJECT):
        result = frozen._run([sys.executable, '-c', script], output, 'source-fixture.log', capture=True)
        record = json.loads(result)
        assert Path(record['source']).resolve() == PROJECT / 'decision_study/__init__.py'
        assert record['relative'] == 'original caller'
        assert all(Path(entry).is_absolute() for entry in os.environ['PYTHONPATH'].split(os.pathsep))
    assert (Path.cwd(), sys.path, dict(os.environ)) == before


def test_symlink_cannot_disguise_overlapping_source_and_runtime_roots(tmp_path, monkeypatch):
    project = tmp_path / 'same-project'
    source = project / 'results/solo-v1'
    work = source / 'runtime/work'
    work.mkdir(parents=True)
    alias = tmp_path / 'alias'
    alias.symlink_to(project, target_is_directory=True)
    binding = {'v1_dir': str(alias / 'results/solo-v1'),
               'source_work_dir': str(alias / 'results/solo-v1/runtime/work')}
    monkeypatch.setattr(decision_study, '__file__', str(project / 'decision_study/__init__.py'))
    monkeypatch.setattr(frozen, '__file__', str(project / 'decision_study/runtime.py'))
    before = Path.cwd(), list(sys.path), dict(os.environ)
    with pytest.raises(StudyError, match='exist separately'):
        with compat.legacy_environment(binding, project):
            pytest.fail('entered physically overlapping roots')
    assert (Path.cwd(), sys.path, dict(os.environ)) == before
