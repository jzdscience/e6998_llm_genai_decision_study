"""Artificial isolated runtime fixtures; no GPU, download or real process started."""
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from decision_study.common import StudyError, fingerprint, read_json, write_json
from decision_study.v3 import runtime


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob('*') if path.is_file()}


class V3RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.v1, self.out = self.root / 'v1', self.root / 'v3'
        self.env = patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture-allocation-1', 'DECISION_STUDY_RUNTIME_DIR': '/irrelevant-inherited-work'})
        self.env.start()
        self.addCleanup(self.env.stop)
        defaults = read_json(runtime.PROJECT / 'configs/runtime_defaults.json')
        self.lock = {**defaults, 'defaults_hash': fingerprint(defaults), 'lock_hash': fingerprint(defaults),
                     'created_at': 'fixture-original', 'verified_sources': [{'repo': defaults['models']['qwen']['repo'], 'sha': defaults['models']['qwen']['revision']}]}
        self.env_name = 'vllm-' + defaults['packages']['vllm_primary']
        self.work = self.v1 / 'runtime/work'
        self.python = self.work / 'envs' / self.env_name / 'bin/python'
        self.python.parent.mkdir(parents=True)
        self.python.write_text('ARTIFICIAL non-executable environment sentinel')
        models = {}
        for model in ('startlux', 'qwen'):
            path = self.work / 'cache/huggingface/hub' / model / 'snapshots' / defaults['models'][model]['revision']
            path.mkdir(parents=True)
            (path / 'SENTINEL').write_text('ARTIFICIAL checkpoint, no weights')
            models[model] = {'model': model, 'model_path': str(path), 'revision': defaults['models'][model]['revision'],
                             'base_path': None, 'base_revision': None, 'native_temperature': defaults['models'][model].get('native_temperature')}
        self.installed = {'lock_hash': self.lock['lock_hash'], 'environments': {self.env_name: {'python': str(self.python),
                          'freeze_sha256': 'fixture-freeze', 'python_version': '3.12.8', 'managed_python': True}}, 'models': models}
        self.doctor = {'gpu': [{'index': 0, 'name': 'NVIDIA L40S', 'memory_mb': 46068.0, 'driver': 'fixture-driver', 'compute_capability': '8.9'}], 'gpu_ready': True}
        write_json(self.v1 / 'runtime/lock.json', self.lock)
        write_json(self.v1 / 'runtime/installations.json', self.installed)
        write_json(self.v1 / 'runtime/doctor.json', self.doctor)
        self.initial_v1 = snapshot(self.v1)

    def initialize(self, **kwargs):
        return runtime.initialize(self.v1, self.out, session_hours_left=kwargs.pop('session_hours_left', 2), **kwargs)

    def fake_download(self, command, control, log_name, **kwargs):
        self.assertEqual(command[1], runtime.PROJECT / 'scripts/model_probe.py')
        self.assertEqual(command[2], 'download')
        self.assertEqual(os.environ['DECISION_STUDY_RUNTIME_DIR'], str(self.out / 'runtime-work'))
        self.assertEqual(Path(command[0]), self.python)
        model_path = self.out / 'runtime-work/cache/huggingface/hub/qwen9b/snapshots' / runtime.QWEN9B_REVISION
        model_path.mkdir(parents=True, exist_ok=True)
        result = Path(command[command.index('--result') + 1])
        write_json(result, {'model': 'qwen', 'model_path': str(model_path), 'revision': runtime.QWEN9B_REVISION,
                            'base_path': None, 'base_revision': None, 'native_temperature': None})

    def test_initialize_preserves_v1_and_makes_independent_role_identities(self):
        first = self.initialize()
        self.assertEqual(snapshot(self.v1), self.initial_v1)
        self.assertEqual(first['binding']['source_work_dir'], str(self.work))
        self.assertFalse((self.out / 'runtime-work').exists())
        for role in runtime.ROLES:
            control = runtime.control_root(self.out, role)
            installed = read_json(control / 'runtime/installations.json')
            self.assertEqual(installed['environments'][self.env_name]['python'], str(self.python))
            lock = read_json(control / 'runtime/lock.json')
            memory = read_json(control / 'runtime/selected_memory.json')
            if role == 'qwen9b':
                self.assertEqual(lock['models']['qwen']['repo'], runtime.QWEN9B_REPO)
                self.assertEqual(lock['models']['qwen']['revision'], runtime.QWEN9B_REVISION)
                self.assertNotEqual(lock['lock_hash'], self.lock['lock_hash'])
                self.assertEqual(installed['models'], {})
                self.assertEqual(memory, {'qwen': .8})
            else:
                self.assertEqual(lock, self.lock)
                self.assertEqual(memory, {runtime.backend_model(role): .4})
                self.assertEqual(runtime.runtime_identity(self.out, role)['runtime'], runtime.frozen.runtime_config(self.v1, runtime.backend_model(role)))

    def test_same_allocation_cannot_extend_deadline_even_with_more_hours(self):
        first = self.initialize(session_hours_left=1)
        second = self.initialize(session_hours_left=20)
        self.assertEqual(first['session'], second['session'])
        with self.assertRaisesRegex(StudyError, 'Cannot renew the same allocation'):
            self.initialize(new_allocation=True)

    def test_new_allocation_requires_changed_slurm_id_and_explicit_acknowledgment(self):
        first = self.initialize()
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'fixture-allocation-2'}):
            with self.assertRaisesRegex(StudyError, 'new_allocation=True'):
                self.initialize()
            with self.assertRaisesRegex(StudyError, 'differs from v3'):
                runtime._validated(self.out, 'qwen4b')
            changed = self.initialize(session_hours_left=3, new_allocation=True)
            self.assertEqual(changed['session']['allocation']['slurm_job_id'], 'fixture-allocation-2')
            self.assertGreater(changed['session']['deadline_epoch'], first['session']['deadline_epoch'])
        self.assertEqual(snapshot(self.v1), self.initial_v1)

    def test_expired_session_and_unidentified_v2_inheritance_are_rejected(self):
        with self.assertRaisesRegex(StudyError, 'lack allocation identity'):
            self.initialize(session_source=self.root / 'v2/runtime-control/runtime/session.json', confirm_same_allocation=True)
        self.initialize()
        value = read_json(self.out / 'runtime/session.json')
        value['deadline_epoch'] = time.time() - 1
        write_json(self.out / 'runtime/session.json', value)
        self.assertEqual(runtime.runtime_identity(self.out, 'qwen9b')['role'], 'qwen9b')
        with self.assertRaisesRegex(StudyError, 'expired'):
            runtime._validated(self.out, 'qwen9b')
        with self.assertRaisesRegex(StudyError, 'expired'):
            self.initialize(session_hours_left=20)

    def test_first_initialization_needs_real_remaining_time_and_separate_output(self):
        for hours in (None, True, float('nan'), 0, 25):
            with self.subTest(hours=hours), self.assertRaises(StudyError):
                self.initialize(session_hours_left=hours)
        self.assertFalse(self.out.exists())
        with self.assertRaisesRegex(StudyError, 'non-overlapping'):
            runtime.initialize(self.v1, self.v1 / 'nested', session_hours_left=1)
        write_json(self.out / 'runtime-control/binding.json', {'fixture': 'v2'})
        with self.assertRaisesRegex(StudyError, 'v2 output'):
            self.initialize()

    def test_changed_lock_source_or_environment_identity_is_rejected(self):
        self.initialize()
        control = runtime.control_root(self.out, 'qwen9b')
        original = read_json(control / 'runtime/lock.json')
        changed = copy.deepcopy(original)
        changed['models']['qwen']['revision'] = '0' * 40
        write_json(control / 'runtime/lock.json', changed)
        with self.assertRaisesRegex(StudyError, 'identity changed'):
            runtime.runtime_identity(self.out, 'qwen9b')
        write_json(control / 'runtime/lock.json', original)
        altered = copy.deepcopy(self.installed)
        altered['environments'][self.env_name]['freeze_sha256'] = 'unexpected new dependencies'
        write_json(self.v1 / 'runtime/installations.json', altered)
        with self.assertRaisesRegex(StudyError, 'Source runtime metadata changed'):
            runtime._validated(self.out, 'qwen4b')

    def test_download_only_9b_and_reuses_old_interpreter_without_environment_changes(self):
        self.initialize()
        with patch.object(runtime.frozen, '_verify_environment'), patch.object(runtime.frozen, '_run', side_effect=self.fake_download) as run:
            for role in ('startlux4b', 'qwen4b'):
                runtime.ensure_download(self.out, role)
            run.assert_not_called()
            first = runtime.ensure_download(self.out, 'qwen9b')
            second = runtime.ensure_download(self.out, 'qwen9b')
            self.assertEqual(first, second)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(snapshot(self.v1), self.initial_v1)
        self.assertEqual(self.initialize()['session'], read_json(self.out / 'runtime/session.json'))

    def test_mismatched_download_does_not_replace_checkpoint_binding(self):
        self.initialize()
        def bad_download(command, control, log_name, **kwargs):
            result = Path(command[command.index('--result') + 1])
            write_json(result, {'model': 'qwen', 'revision': '0' * 40, 'model_path': self.installed['models']['qwen']['model_path']})
        with patch.object(runtime.frozen, '_verify_environment'), patch.object(runtime.frozen, '_run', side_effect=bad_download):
            with self.assertRaisesRegex(StudyError, 'pinned snapshot inside v3'):
                runtime.ensure_download(self.out, 'qwen9b')
        self.assertEqual(read_json(runtime.control_root(self.out, 'qwen9b') / 'runtime/installations.json')['models'], {})
        self.assertEqual(snapshot(self.v1), self.initial_v1)

    def test_runtime_context_restores_environment_and_closes_budget_on_exception(self):
        self.initialize()
        before = dict(os.environ)
        with patch.object(runtime.frozen, 'doctor', return_value=self.doctor), \
             patch.object(runtime, '_cuda_preflight', return_value={'CUDA_HOME': '/fixture/toolkit'}):
            with self.assertRaisesRegex(RuntimeError, 'artificial interruption'):
                with runtime.runtime_context(self.v1, self.out, 'qwen9b') as (control, budget):
                    self.assertEqual(os.environ['CUDA_HOME'], '/fixture/toolkit')
                    self.assertEqual(os.environ['DECISION_STUDY_RUNTIME_DIR'], str(self.out / 'runtime-work'))
                    budget.check()
                    raise RuntimeError('artificial interruption')
        self.assertEqual(dict(os.environ), before)
        self.assertTrue(read_json(runtime.control_root(self.out, 'qwen9b') / 'budget.json')['sessions'][-1]['closed'])
        self.assertEqual(snapshot(self.v1), self.initial_v1)

    def test_missing_or_changed_allocation_never_launches_gpu_preflight(self):
        with patch.object(runtime.frozen, 'doctor') as doctor:
            with self.assertRaises((StudyError, FileNotFoundError)):
                with runtime.runtime_context(self.v1, self.out, 'qwen4b'):
                    self.fail('Uninitialized runtime entered')
            doctor.assert_not_called()
        self.initialize()
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'another-job'}), patch.object(runtime.frozen, 'doctor') as doctor:
            with self.assertRaises(StudyError):
                with runtime.runtime_context(self.v1, self.out, 'qwen4b'):
                    self.fail('Wrong allocation entered')
            doctor.assert_not_called()

    def test_managed_services_starts_exactly_one_role_and_tokenizer_uses_that_lock(self):
        self.initialize()
        seen = []
        @contextmanager
        def services(project, control, models):
            seen.append((control, models))
            self.assertEqual(os.environ['DECISION_STUDY_RUNTIME_DIR'], str(self.out / 'runtime-work'))
            yield {'startlux': 'http://fixture/startlux', 'qwen': 'http://fixture/qwen'}
        with patch.object(runtime.frozen, 'managed_services', services), \
             patch.object(runtime.frozen, 'token_preflight_examples', return_value={'status': 'ok'}) as tokens:
            with runtime.managed_services(self.out, 'qwen4b') as endpoint:
                self.assertEqual(endpoint, 'http://fixture/qwen')
            result = runtime.token_preflight(self.out, 'startlux4b', [{'sample_id': 'fixture'}])
        self.assertEqual(seen, [(runtime.control_root(self.out, 'qwen4b'), ['qwen'])])
        self.assertEqual(tokens.call_args.args[1:3], (runtime.control_root(self.out, 'startlux4b'), 'startlux'))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(snapshot(self.v1), self.initial_v1)

    def test_wrong_role_control_and_symlinked_work_are_rejected(self):
        self.initialize()
        with self.assertRaisesRegex(StudyError, 'does not belong'):
            runtime.ensure_download(self.out, 'qwen9b', control=runtime.control_root(self.out, 'qwen4b'))
        (self.out / 'runtime-work').symlink_to(self.work, target_is_directory=True)
        with self.assertRaisesRegex(StudyError, 'symlinked runtime path'):
            runtime._validated(self.out, 'qwen9b')

    def test_offline_identity_survives_expiry_other_host_and_missing_source(self):
        self.initialize()
        expected = {role: runtime.runtime_identity(self.out, role) for role in runtime.ROLES}
        session = read_json(self.out / 'runtime/session.json')
        session['deadline_epoch'] = time.time() - 1
        write_json(self.out / 'runtime/session.json', session)
        self.v1.rename(self.root / 'source-not-present-on-this-machine')
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'another-job'}), patch.object(runtime.socket, 'gethostname', return_value='offline-laptop'):
            for role in runtime.ROLES:
                self.assertEqual(runtime.runtime_identity(self.out, role), expected[role])
            with patch.object(runtime.frozen, 'doctor') as doctor:
                with self.assertRaises((StudyError, FileNotFoundError)):
                    with runtime.runtime_context(self.v1, self.out, 'qwen4b'):
                        self.fail('Offline evidence cannot start a GPU runtime')
                doctor.assert_not_called()

    def test_cuda_preflight_reuses_verified_toolkit_and_writes_new_build_directory(self):
        self.initialize()
        control, binding, session, installed = runtime._validated(self.out, 'qwen9b')
        from scripts import prepare_cuda_toolkit
        calls = []
        def probe(command, control, log_name, **kwargs):
            calls.append(command)
            result = command[command.index('--result') + 1]
            write_json(result, {'status': 'complete', 'environment': {key: 'fixture' for key in runtime.ENV_KEYS}})
        with patch.object(prepare_cuda_toolkit, 'validate_target') as validate, \
             patch.object(runtime.frozen, '_verify_environment'), patch.object(runtime.frozen, '_run', side_effect=probe):
            runtime._cuda_preflight(control, binding, installed)
        validate.assert_called_once_with(self.work / 'toolchains' / runtime.TOOLKIT, runtime.PROJECT / 'configs/cuda_build_toolkit_13_0_flashinfer.txt')
        self.assertEqual(calls[0][calls[0].index('--work-dir') + 1], self.out / 'runtime-work/cuda-toolchain-13.0-flashinfer')
        self.assertEqual(calls[0][calls[0].index('--toolkit-target') + 1], self.work / 'toolchains' / runtime.TOOLKIT)
        self.assertEqual(snapshot(self.v1), self.initial_v1)


if __name__ == '__main__':
    unittest.main()
