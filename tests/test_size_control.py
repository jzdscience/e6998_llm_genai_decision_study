"""Artificial boundary and resumability checks; no model weights or GPU used."""
from contextlib import ExitStack, contextmanager
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch
import zipfile
from decision_study.common import StudyError, fingerprint, read_json, write_json, utc_now
from decision_study.size_control import data, runtime, study, cli
from decision_study.size_control.identity import ROLE, protocol
from decision_study.runner import _probability_comparison
from decision_study.storage import PredictionStore

LABELS = ["entailment", "neutral", "contradiction"]


class SizeRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.v1 = self.root / 'v1'; self.out = self.root / 'size'
        env = patch.dict(os.environ, {'SLURM_JOB_ID': '111'}); env.start(); self.addCleanup(env.stop)
        defaults = read_json(runtime.PROJECT / 'configs/runtime_defaults.json')
        self.lock = {**defaults, 'lock_hash': fingerprint(defaults), 'defaults_hash': fingerprint(defaults)}
        self.work = self.v1 / 'runtime/work'; self.env_name = 'vllm-0.31.0'
        self.python = self.work / 'envs' / self.env_name / 'bin/python'
        self.python.parent.mkdir(parents=True); self.python.write_text('synthetic')
        models = {}
        for model in ('startlux', 'qwen'):
            p = self.work / 'models' / model / defaults['models'][model]['revision']; p.mkdir(parents=True)
            models[model] = {'model': model, 'model_path': str(p), 'revision': p.name}
        installed = {'lock_hash': self.lock['lock_hash'], 'environments': {self.env_name: {'python': str(self.python),
                     'freeze_sha256': 'fixture', 'python_version': '3.11'}}, 'models': models}
        write_json(self.v1 / 'runtime/lock.json', self.lock)
        write_json(self.v1 / 'runtime/installations.json', installed)
        self.doctor = {'gpu': [{'name': 'NVIDIA L40S', 'memory_mb': 46068, 'driver': 'fixture', 'compute_capability': '8.9'}]}
        write_json(self.v1 / 'runtime/doctor.json', self.doctor)
        self.before = {str(p.relative_to(self.v1)): p.read_bytes() for p in self.v1.rglob('*') if p.is_file()}

    def init(self, **kwargs):
        return runtime.initialize(self.v1, self.out, session_hours_left=2, **kwargs)

    def test_isolated_model_revision_temperature_and_memory(self):
        self.init()
        control = runtime.control_root(self.out, ROLE)
        lock = read_json(control / 'runtime/lock.json')
        self.assertEqual(lock['models']['startlux']['revision'], protocol()['model']['revision'])
        self.assertEqual(lock['models']['startlux']['native_temperature'], 1.1014)
        self.assertEqual(read_json(control / 'runtime/selected_memory.json'), {'startlux': .8})
        self.assertEqual(read_json(control / 'runtime/installations.json')['models'], {})
        after = {str(p.relative_to(self.v1)): p.read_bytes() for p in self.v1.rglob('*') if p.is_file()}
        self.assertEqual(self.before, after)

    def test_same_job_cannot_extend_or_fake_renewal(self):
        first = self.init()['session']
        self.assertEqual(first, self.init()['session'])
        with self.assertRaises(StudyError): self.init(new_allocation=True)
        with patch.dict(os.environ, {'SLURM_JOB_ID': '222'}):
            with self.assertRaises(StudyError): self.init()
            self.assertEqual(self.init(new_allocation=True)['session']['allocation']['slurm_job_id'], '222')

    def test_expired_allocation_blocks_gpu(self):
        self.init(); path = self.out / 'runtime/session.json'; s = read_json(path)
        s['deadline_epoch'] = time.time() - 1; write_json(path, s)
        with patch.object(runtime.frozen, 'doctor') as doctor:
            with self.assertRaises(StudyError):
                with runtime.runtime_context(self.v1, self.out, ROLE): pass
            doctor.assert_not_called()

    def test_changed_original_metadata_blocks_resume(self):
        self.init(); path = self.v1 / 'runtime/installations.json'; value = read_json(path)
        value['environments'][self.env_name]['freeze_sha256'] = 'changed'; write_json(path, value)
        with self.assertRaises(StudyError): runtime._validated(self.out, ROLE)

    def test_download_fixed_role_and_metadata_checked_before_accept(self):
        self.init(); seen = []
        def download(command, control, log, **kwargs):
            seen.append(command)
            self.assertEqual(command[command.index('--model') + 1], 'startlux')
            p = self.out / 'runtime-work/cache/snapshots' / runtime.STARTLUX9B_REVISION; p.mkdir(parents=True)
            write_json(command[command.index('--result') + 1], {'model': 'startlux', 'revision': runtime.STARTLUX9B_REVISION, 'model_path': str(p)})
        with patch.object(runtime.frozen, '_verify_environment'), patch.object(runtime.frozen, '_run', side_effect=download), \
             patch.object(runtime, '_verify_model_metadata', side_effect=StudyError('metadata differs')):
            with self.assertRaisesRegex(StudyError, 'metadata differs'): runtime.ensure_download(self.out, ROLE)
        self.assertEqual(read_json(runtime.control_root(self.out, ROLE) / 'runtime/installations.json')['models'], {})
        self.assertEqual(len(seen), 1)

    def test_offline_identity_no_source_or_live_session_needed(self):
        self.init(); expected = runtime.runtime_identity(self.out, ROLE)
        self.v1.rename(self.root / 'moved-source')
        with patch.dict(os.environ, {'SLURM_JOB_ID': '333'}):
            self.assertEqual(runtime.runtime_identity(self.out, ROLE), expected)
            with self.assertRaises((StudyError, FileNotFoundError)): runtime._validated(self.out, ROLE)

    def test_missing_slurm_id_blocks_even_with_explicit_hours(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID':'','SLURM_JOBID':''}):
            with self.assertRaisesRegex(StudyError,'numeric current SLURM_JOB_ID'): self.init()

    def test_multiple_gpus_do_not_match_single_l40s_protocol(self):
        self.init(); doctor={'gpu':self.doctor['gpu']*2}
        with patch.object(runtime.frozen,'doctor',return_value=doctor):
            with self.assertRaisesRegex(StudyError,'single L40S'):
                with runtime.runtime_context(self.v1,self.out,ROLE): pass

    def test_existing_gpu_process_blocks_before_cuda_preflight(self):
        self.init()
        result = type('Result', (), {'stdout':'1234\n5678\n'})()
        with patch.object(runtime.frozen,'doctor',return_value=self.doctor), \
             patch.object(runtime.subprocess,'run',return_value=result), \
             patch.object(runtime,'_cuda_preflight') as preflight:
            with self.assertRaisesRegex(StudyError,'existing compute processes'):
                with runtime.runtime_context(self.v1,self.out,ROLE): pass
            preflight.assert_not_called()

    def test_idle_gpu_query_rejects_unknown_output_and_accepts_empty(self):
        result = type('Result', (), {'stdout':''})()
        with patch.object(runtime.subprocess,'run',return_value=result) as query:
            self.assertEqual(runtime.assert_gpu_idle()['compute_pids'],[])
            self.assertIn('--query-compute-apps=pid',query.call_args.args[0])
            result.stdout='[Not Supported]\n'
            with self.assertRaisesRegex(StudyError,'Unexpected nvidia-smi'): runtime.assert_gpu_idle()

    def test_symlink_output_rejected(self):
        self.out.symlink_to(self.v1, target_is_directory=True)
        with self.assertRaises(StudyError): self.init()


class SizeStudyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.out = Path(temp.name)
        self.examples = []; self.gold = {}
        for split, count in [('development', 6), ('holdout', 12)]:
            for i in range(count):
                sid = f'artificial-{split}-{i}'
                e = {'sample_id': sid, 'task': 'anli', 'split': split, 'round': f'R{i%3+1}',
                     'group_id': f'artificial-{split}-premise-{i//2}', 'state': f'Artificial input {i}',
                     'instructions': 'Software test only', 'labels': LABELS, 'criteria': {l:l for l in LABELS},
                     'input_hash': fingerprint(sid)}
                self.examples.append(e); self.gold[sid] = LABELS[i%3]
        self.manifest = {'manifest_hash': 'artificial-manifest', 'count': len(self.examples)}
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(data, 'verify', return_value=self.manifest))
        self.stack.enter_context(patch.object(data, 'examples', side_effect=lambda output, role: [e for e in self.examples if e['split'] == role]))
        self.stack.enter_context(patch.object(runtime, 'runtime_identity', return_value=self.runtime_identity()))
        self.refs = {r: [self.row(e) for e in self.load('holdout')] for r in ('startlux4b', 'qwen4b', 'qwen9b')}
        self.stack.enter_context(patch.object(data, 'references', return_value=self.refs))
        self.reads = 0
        self.stack.enter_context(patch.object(data, 'gold_for_evaluation', side_effect=self.read_gold))
        write_json(self.out / 'references/provenance.json', {'original_evaluation': {'direct': {
            r: {'model': r, 'n': 12, 'correct': 12, 'accuracy': 1.0, 'failures': 0} for r in self.refs}}})

    def runtime_identity(self):
        p = protocol()['model']; effective = {k:p[k] for k in ('native_temperature','gpu_memory_utilization','precision','max_model_len','think_mode','bidirectional','none_check','allow_none','prefix_cache','truncate')}
        effective.update(model_repo=p['repo'], model_revision=p['revision'], backend='vllm', vllm_version='0.31.0')
        return {'role': ROLE, 'runtime': effective}

    def load(self, role): return [e for e in self.examples if e['split'] == role]

    def row(self, e, cfg=None):
        pred = self.gold[e['sample_id']]; probs = {l:.8 if l == pred else .1 for l in LABELS}
        row = {'sample_id':e['sample_id'], 'task':'anli', 'split':e['split'], 'input_hash':e['input_hash'],
               'model':'startlux', 'status':'ok', 'prediction':pred, 'probabilities':probs,
               'raw_response':{'answers':{'decision':{'choice':pred,'probabilities':probs}}},
               'latency_ms':1, 'recorded_at':utc_now(), 'attempts':[{'status':'ok','request_id':str(uuid.uuid4())}]}
        if cfg: row.update(config_hash=fingerprint(cfg), model_revision=cfg['runtime']['model_revision'])
        return row

    def read_gold(self, output, freeze, access):
        self.assertEqual(read_json(output / 'evaluation-access.json')['freeze_hash'], freeze['freeze_hash'])
        self.reads += 1; write_json(output / 'sealed/holdout_gold.json', self.gold)
        return self.gold

    def complete(self):
        freeze = study.make_freeze(self.out); cfg = study.config_for(self.out); directory = study.job_root(self.out)
        store = PredictionStore(directory, 'startlux', cfg, spool_dir=self.out/'spool')
        for e in self.load('holdout'): store.append(self.row(e,cfg))
        store.sync(); rows = list(store.rows.values()); a = [self.row(e) for e in self.load('development')]
        b = [self.row(e) for e in self.load('development')]
        write_json(directory/'smoke.json', {'config_hash':fingerprint(cfg), 'reference':a, 'current':b,
                   'checks':_probability_comparison(self.load('development'),a,b)})
        write_json(directory/'token_preflight.json', {'status':'ok','model':'startlux','model_revision':cfg['runtime']['model_revision'],
                   'max_model_len':8192,'n_examples':len(self.examples),'truncated':False,'overlong':[],
                   'effective_config_hash':fingerprint(cfg['runtime']),
                   'rows':[{'sample_id':e['sample_id'],'input_tokens':10,'reserved_output_tokens':1,'token_ids_sha256':fingerprint(e['sample_id'])} for e in self.examples]})
        done = data.seal({'config_hash':fingerprint(cfg),'freeze_hash':freeze['freeze_hash'],'count':12,'failures':0,
                   'predictions_hash':fingerprint(sorted(rows,key=lambda r:r['sample_id'])),
                   'smoke_sha256':study.verify_smoke(self.out,cfg),'token_preflight_sha256':study.verify_tokens(self.out,cfg)}, 'completion_hash')
        write_json(directory/'completion.json', done)
        return freeze

    def test_freeze_and_status_never_read_gold(self):
        first = study.make_freeze(self.out)
        self.assertEqual(first, study.make_freeze(self.out)); self.assertTrue(study.status(self.out)['frozen'])
        self.assertEqual(self.reads,0)

    def test_evaluate_replays_idempotently_and_exports_no_weights(self):
        self.complete(); result = study.evaluate(self.out); self.assertEqual(result['n'],12)
        before = (self.out/'evaluation.json').read_bytes()
        self.assertEqual(study.evaluate(self.out),result); self.assertEqual(before,(self.out/'evaluation.json').read_bytes())
        write_json(self.out/'runtime-work/models/weights.json', {'fake':'weight'})
        with zipfile.ZipFile(study.export(self.out)) as z:
            self.assertIn('evaluation.json',z.namelist()); self.assertFalse(any(n.startswith('runtime-work/') for n in z.namelist()))

    def test_missing_freeze_blocks_gold(self):
        with self.assertRaises((StudyError,FileNotFoundError)): study.evaluate(self.out)
        self.assertEqual(self.reads,0)

    def test_missing_shard_blocks_gold(self):
        self.complete(); next((study.job_root(self.out)/'predictions/startlux').glob('shard-*.json')).unlink()
        with self.assertRaises(StudyError): study.evaluate(self.out)
        self.assertEqual(self.reads,0)

    def test_missing_token_proof_blocks_gold(self):
        self.complete(); (study.job_root(self.out)/'token_preflight.json').unlink()
        with self.assertRaises(FileNotFoundError): study.evaluate(self.out)
        self.assertEqual(self.reads,0)

    def test_smoke_raw_response_mismatch_blocks_gold(self):
        self.complete(); path=study.job_root(self.out)/'smoke.json'; value=read_json(path)
        value['current'][0]['prediction']='contradiction'; write_json(path,value)
        with self.assertRaises(StudyError): study.evaluate(self.out)
        self.assertEqual(self.reads,0)

    def test_model_change_cannot_reuse_freeze(self):
        self.complete(); wrong=self.runtime_identity(); wrong['runtime']['native_temperature']=1.1103
        with patch.object(runtime,'runtime_identity',return_value=wrong):
            with self.assertRaises(StudyError): study.evaluate(self.out)
        self.assertEqual(self.reads,0)

    def test_partial_access_blocks_silent_retry(self):
        self.complete()
        with patch.object(data,'gold_for_evaluation',side_effect=StudyError('artificial failure')):
            with self.assertRaises(StudyError): study.evaluate(self.out)
        with self.assertRaisesRegex(StudyError,'without a completed result'): study.evaluate(self.out)
        self.assertEqual(self.reads,0)

    def test_completion_skip_verifies_before_any_gpu(self):
        self.complete()
        with patch.object(runtime,'runtime_context') as context:
            self.assertEqual(study.run_job('/unused',self.out)['count'],12); context.assert_not_called()

    def test_actual_collection_path_and_smoke_only_keep_gold_closed(self):
        class Budget:
            def check(self): pass
            def estimate_remaining(self,*args): pass
        @contextmanager
        def context(*args,**kwargs): yield self.out, Budget()
        @contextmanager
        def services(*args,**kwargs): yield 'http://artificial-endpoint'
        cfg=study.config_for(self.out)
        tokens={'status':'ok','model':'startlux','model_revision':cfg['runtime']['model_revision'],
                'max_model_len':8192,'n_examples':len(self.examples),'truncated':False,'overlong':[],
                'effective_config_hash':fingerprint(cfg['runtime']),
                'rows':[{'sample_id':e['sample_id'],'input_tokens':10,'reserved_output_tokens':1,
                         'token_ids_sha256':fingerprint(e['sample_id'])} for e in self.examples]}
        with patch.object(runtime,'runtime_context',context), patch.object(runtime,'ensure_download'), \
             patch.object(runtime,'token_preflight',return_value=tokens), patch.object(runtime,'managed_services',services), \
             patch.object(study.BoundClient,'predict',side_effect=self.row):
            smoke=study.run_job('/unused',self.out,smoke_only=True)
            self.assertEqual(smoke['status'],'smoke_passed')
            self.assertFalse((study.job_root(self.out)/'completion.json').exists())
            done=study.run_job('/unused',self.out)
            self.assertEqual(done['count'],12)
            self.assertEqual(self.reads,0)
            self.assertTrue(study.status(self.out)['collection_complete'])

    def test_cli_completed_start_skips_slurm_init_and_gold_with_new_allocation(self):
        self.complete(); study.evaluate(self.out); reads=self.reads
        with patch.object(cli,'checkout_identity',return_value={'fixture':True}), \
             patch.object(data,'prepare',return_value=self.manifest), \
             patch.object(cli,'slurm_hours_left') as slurm, \
             patch.object(runtime,'initialize') as initialize, \
             patch.object(study,'run_job') as run_job:
            result=cli.main(['start','--v1-dir','/artificial-v1','--v3-dir','/artificial-v3',
                             '--output-dir',str(self.out),'--new-allocation'])
            self.assertEqual(result,0); slurm.assert_not_called(); initialize.assert_not_called(); run_job.assert_not_called()
            self.assertEqual(self.reads,reads)

    def test_cli_complete_skip_validates_chain_before_skipping(self):
        self.complete(); study.evaluate(self.out)
        path=self.out/'evaluation-access.json'; access=read_json(path); access['freeze_hash']='wrong'
        write_json(path,data.seal({k:v for k,v in access.items() if k!='access_hash'},'access_hash'))
        with patch.object(cli,'checkout_identity',return_value={'fixture':True}), \
             patch.object(data,'prepare',return_value=self.manifest), patch.object(runtime,'initialize') as initialize:
            result=cli.main(['start','--v1-dir','/artificial-v1','--v3-dir','/artificial-v3',
                             '--output-dir',str(self.out),'--new-allocation'])
            self.assertEqual(result,2); initialize.assert_not_called()

    def test_prediction_chronology_checked(self):
        self.complete(); path=next((study.job_root(self.out)/'predictions/startlux').glob('shard-*.json'))
        shard=read_json(path); shard['rows'][0]['recorded_at']='2000-01-01T00:00:00+00:00'; shard['sha256']=fingerprint(shard['rows']); write_json(path,shard)
        # Reseal completion to isolate chronology from prediction-hash checking.
        path=study.job_root(self.out)/'completion.json'; done=read_json(path)
        from decision_study.storage import load_predictions
        done['predictions_hash']=fingerprint(sorted(load_predictions(study.job_root(self.out),'startlux'),key=lambda r:r['sample_id']))
        done=data.seal({k:v for k,v in done.items() if k!='completion_hash'},'completion_hash'); write_json(path,done)
        with self.assertRaisesRegex(StudyError,'predates'): study.evaluate(self.out)
        self.assertEqual(self.reads,0)


if __name__ == '__main__': unittest.main()
