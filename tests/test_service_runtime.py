"""Isolated allocation/control fixtures; no GPU, real process or download."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from decision_study.common import StudyError, fingerprint, read_json, write_json
from decision_study.service_study import cpu, runtime


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class ServiceRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.v1, self.out = self.root / 'v1', self.root / 'service'
        env = patch.dict(os.environ, {'SLURM_JOB_ID': 'service-fixture-1',
                                     'DECISION_STUDY_RUNTIME_DIR': '/unrelated/inherited'})
        env.start(); self.addCleanup(env.stop)
        defaults = read_json(runtime.PROJECT / 'configs/runtime_defaults.json')
        lock = {**defaults, 'defaults_hash': fingerprint(defaults), 'lock_hash': fingerprint(defaults)}
        self.env_name = 'vllm-' + defaults['packages']['vllm_primary']
        self.work = self.v1 / 'runtime/work'
        self.python = self.work / 'envs' / self.env_name / 'bin/python'
        self.python.parent.mkdir(parents=True)
        self.python.write_text('NONEXECUTABLE SOFTWARE FIXTURE')
        models = {}
        for name in ('startlux', 'qwen'):
            directory = self.work / 'models' / name / defaults['models'][name]['revision']
            directory.mkdir(parents=True)
            (directory / 'SENTINEL').write_text('NO WEIGHTS')
            models[name] = {'model': name, 'model_path': str(directory),
                            'revision': defaults['models'][name]['revision'],
                            'base_path': None, 'base_revision': None}
        installed = {'lock_hash': lock['lock_hash'], 'models': models,
                     'environments': {self.env_name: {'python': str(self.python), 'freeze_sha256': 'fixture'}}}
        for name, value in {'lock.json': lock, 'installations.json': installed,
                            'doctor.json': {'gpu': [], 'gpu_ready': False},
                            'selected_memory.json': {'startlux': 0.4, 'qwen': 0.4},
                            'selected_profiles.json': {'startlux': 'primary', 'qwen': 'primary'}}.items():
            write_json(self.v1 / 'runtime' / name, value)
        self.original = snapshot(self.v1)

    def test_initialize_is_readonly_for_old_runtime_and_idempotent(self):
        first = runtime.initialize(self.v1, self.out, 2)
        again = runtime.initialize(self.v1, self.out, 20)
        self.assertEqual(first, again)
        self.assertEqual(snapshot(self.v1), self.original)
        self.assertEqual(first['binding']['source_work_dir'], str(self.work))
        self.assertEqual(first['binding']['work_dir'], str(self.out / 'runtime-work'))
        self.assertFalse((self.out / 'runtime-work').exists())
        control, _, session, installed = runtime.validate(self.out)
        self.assertEqual(installed['environments'][self.env_name]['python'], str(self.python))
        self.assertEqual(read_json(control / 'runtime/selected_memory.json'), {'startlux': 0.4, 'qwen': 0.4})
        self.assertEqual(session, first['session'])

    def test_same_allocation_may_shorten_but_never_renew(self):
        initial = runtime.initialize(self.v1, self.out, 2)['session']
        shortened = runtime.initialize(self.v1, self.out, 0.5)['session']
        self.assertLess(shortened['deadline_epoch'], initial['deadline_epoch'])
        preserved = read_json(self.out / 'runtime/session-history' / (fingerprint(initial) + '.json'))
        self.assertEqual(preserved, initial)
        with self.assertRaisesRegex(StudyError, 'Cannot renew the same allocation'):
            runtime.initialize(self.v1, self.out, 20, new_allocation=True)
        self.assertEqual(snapshot(self.v1), self.original)

    def test_new_job_requires_explicit_acknowledgment_and_preserves_prior_session(self):
        first = runtime.initialize(self.v1, self.out, 1)['session']
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'service-fixture-2'}):
            with self.assertRaisesRegex(StudyError, 'explicitly acknowledged'):
                runtime.initialize(self.v1, self.out, 2)
            with self.assertRaises(StudyError):
                runtime.validate(self.out)
            second = runtime.initialize(self.v1, self.out, 2, new_allocation=True)['session']
            self.assertEqual(second['allocation']['slurm_job_id'], 'service-fixture-2')
            self.assertGreater(second['deadline_epoch'], first['deadline_epoch'])
        self.assertEqual(snapshot(self.v1), self.original)
        self.assertTrue((self.out / 'runtime/session-history' / (fingerprint(first) + '.json')).exists())

    def test_same_job_on_changed_host_cannot_claim_new_allocation(self):
        runtime.initialize(self.v1, self.out, 1)
        with patch.object(runtime.shared.socket, 'gethostname', return_value='different-fixture-host'):
            with self.assertRaisesRegex(StudyError, 'different Slurm job'):
                runtime.initialize(self.v1, self.out, 2, new_allocation=True)

    def test_missing_job_or_invalid_hours_never_creates_output(self):
        for hours in (None, True, 0, 25, float('nan'), float('inf')):
            with self.subTest(hours=hours), self.assertRaises(StudyError):
                runtime.initialize(self.v1, self.out, hours)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '', 'SLURM_JOBID': ''}):
            with self.assertRaisesRegex(StudyError, 'actual company Slurm'):
                runtime.initialize(self.v1, self.out, 2)
        self.assertFalse(self.out.exists())
        self.assertEqual(snapshot(self.v1), self.original)

    def test_changed_control_is_preflighted_before_any_session_rewrite(self):
        runtime.initialize(self.v1, self.out, 2)
        path = self.out / 'runtime-control/runtime/selected_memory.json'
        write_json(path, {'startlux': .1, 'qwen': .1})
        before = snapshot(self.out)
        with self.assertRaisesRegex(StudyError, 'Changed runtime control'):
            runtime.initialize(self.v1, self.out, 1)
        self.assertEqual(snapshot(self.out), before)
        with self.assertRaisesRegex(StudyError, 'Runtime control changed'):
            runtime.validate(self.out)
        self.assertEqual(snapshot(self.v1), self.original)

    def test_source_change_and_overlapping_output_are_rejected(self):
        with self.assertRaisesRegex(StudyError, 'non-overlapping'):
            runtime.initialize(self.v1, self.v1 / 'nested', 2)
        runtime.initialize(self.v1, self.out, 2)
        write_json(self.v1 / 'runtime/selected_memory.json', {'startlux': .5, 'qwen': .4})
        with self.assertRaisesRegex(StudyError, 'source identity'):
            runtime.validate(self.out)

    def test_expiry_and_changed_allocation_prevent_gpu_entry(self):
        runtime.initialize(self.v1, self.out, 2)
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'different'}), patch.object(runtime.frozen, 'doctor') as doctor:
            with self.assertRaises(StudyError):
                with runtime.gpu_service(self.out, 'startlux4b', []):
                    self.fail('entered wrong allocation')
            doctor.assert_not_called()
        session = read_json(self.out / 'runtime/session.json')
        session['deadline_epoch'] = time.time() - 1
        write_json(self.out / 'runtime/session.json', session)
        with patch.object(runtime.frozen, 'doctor') as doctor:
            with self.assertRaisesRegex(StudyError, 'expired'):
                with runtime.gpu_service(self.out, 'startlux4b', []):
                    self.fail('entered expired allocation')
            doctor.assert_not_called()
        with self.assertRaisesRegex(StudyError, 'expired'):
            runtime.initialize(self.v1, self.out, 20)
        self.assertEqual(snapshot(self.v1), self.original)

    def fake_health(self):
        return {'status': 'ok', 'service': 'frozen_minilm_cpu', 'pid': 4242, 'instance_id': 'fixture-instance',
                'identity': {'recipe': 'minilm400', 'classifier_sha256': cpu.ACCEPTED_MODELS['minilm400'],
                             'baseline_receipts': cpu.ACCEPTED_RECEIPTS,
                             'encoder': {'provider': 'CPUExecutionProvider'}},
                'environment': {'thread_environment': cpu.THREAD_ENV, 'threadpools': [{'num_threads': 1}]},
                'workers': 1, 'batch_size': 1, 'numeric_threads': 1, 'prediction_cache': False,
                'queue_capacity': 64, 'maximum_admitted': 65, 'maximum_http_handlers': 72}

    def cpu_context(self, health):
        class Process:
            pid = 4242
            def poll(self):
                return None
        patches = [patch.object(runtime.frozen, '_check_port'),
                   patch.object(runtime.subprocess, 'Popen', return_value=Process()),
                   patch.object(runtime.urllib.request, 'urlopen', side_effect=lambda *_a, **_k: io.BytesIO(json.dumps(health).encode())),
                   patch.object(runtime, 'stop')]
        from contextlib import ExitStack
        stack = ExitStack()
        mocks = [stack.enter_context(p) for p in patches]
        return stack, mocks

    def test_cpu_health_binds_child_pid_frozen_model_and_single_thread_contract(self):
        runtime.initialize(self.v1, self.out, 2)
        stack, mocks = self.cpu_context(self.fake_health())
        with stack:
            with runtime.cpu_service(self.out, 'minilm400', python=self.python,
                                     bundle_dir=self.root / 'bundle', model_dir=self.root / 'encoder') as live:
                self.assertEqual(live['model'], 'minilm400')
                self.assertEqual(live['pids'], [4242])
                self.assertIn('decision_study.service_study.cpu', live['command'])
            mocks[-1].assert_called_once()
        self.assertEqual(snapshot(self.v1), self.original)

    def test_cpu_wrong_identity_and_resource_contract_are_rejected_and_stopped(self):
        runtime.initialize(self.v1, self.out, 2)
        variants=[]
        for key, value in (('pid', 9999), ('workers', 2), ('batch_size', 32), ('prediction_cache', True), ('queue_capacity', 128)):
            value_health=copy.deepcopy(self.fake_health()); value_health[key]=value; variants.append(value_health)
        value_health=copy.deepcopy(self.fake_health()); value_health['identity']['recipe']='minilm1600'; variants.append(value_health)
        value_health=copy.deepcopy(self.fake_health()); value_health['identity']['classifier_sha256']='changed'; variants.append(value_health)
        value_health=copy.deepcopy(self.fake_health()); value_health['identity']['baseline_receipts']={}; variants.append(value_health)
        value_health=copy.deepcopy(self.fake_health()); value_health['environment']['thread_environment']['OMP_NUM_THREADS']='2'; variants.append(value_health)
        for health in variants:
            with self.subTest(health=health):
                stack,mocks=self.cpu_context(health)
                with stack:
                    with self.assertRaisesRegex(StudyError, 'identity or resident'):
                        with runtime.cpu_service(self.out, 'minilm400', python=self.python,
                                                 bundle_dir=self.root/'bundle', model_dir=self.root/'encoder'):
                            self.fail('invalid CPU service admitted')
                    mocks[-1].assert_called_once()


if __name__ == '__main__':
    unittest.main()
