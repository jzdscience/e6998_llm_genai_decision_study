"""Actual full-study orchestration with artificial data and hardware boundaries.

The fixture substitutes external data/Slurm/model services and the process launch
boundary. It retains prepare/context, fitting, freeze, evaluation, numeric audit,
report, export and the single-command dispatcher. No real ANLI labels are read.
"""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import pytest

from decision_study.common import StudyError, file_hash, fingerprint, read_json, write_json
from decision_study.service_study.identity import fixed_write
from decision_study.learnability import study as s, modeling as m


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    runtime = tmp_path/'runtime'
    for name in ('solo-v1', 'solo-v3', 'aux-size-control-v1-r2'):
        folder = runtime/'results'/name
        folder.mkdir(parents=True)
        (folder/'keep.json').write_text('{"historical":true}\n')
    output = runtime/'results/aux-anli-learnability-v1'
    template = tmp_path/'templates'
    template.mkdir()
    monkeypatch.setattr(m, 'FIT_SIZES', (18, 36))
    monkeypatch.setattr(m, 'EMBEDDING_DIM', 6)
    monkeypatch.setattr(m, 'BOOTSTRAP_REPLICATES', 20)
    recipe = copy.deepcopy(m.RECIPE); recipe.update(text_dimension=6, bootstrap_replicates=20)
    monkeypatch.setattr(m, 'RECIPE', recipe)
    rows, gold, embeddings = [], {}, []
    predictions = {role: [] for role in s.profile.ROLES}
    rng = np.random.default_rng(6)
    for split, count in (('fitting', 36), ('selection', 18), ('eligible_dev', 30), ('train_holdout', 30)):
        for i in range(count):
            sid = f'artificial:{split}:{i:03}'
            row = {'sample_id': sid, 'task': 'anli', 'split': split, 'labels': list(m.LABELS),
                   'state': 'Artificial input '+str(i), 'round': f'R{i%3+1}', 'premise_hash': f'{split}:{i//2}'}
            rows.append(row); gold[sid] = m.LABELS[i%3]
            vector = rng.normal(size=6); embeddings.append(vector/np.linalg.norm(vector))
            for role, divisor in zip(s.profile.ROLES, (4, 3, 5)):
                predicted = gold[sid] if i%divisor else m.LABELS[(i+1)%3]
                pred = {'sample_id': sid, 'status': 'ok', 'prediction': predicted}
                if role == 'startlux4b': pred['probabilities'] = {label: .8 if label == predicted else .1 for label in m.LABELS}
                predictions[role].append(pred)
    fit_ids = [r['sample_id'] for r in rows if r['split'] == 'fitting']
    train_gold = {r['sample_id']: gold[r['sample_id']] for r in rows if r['split'] in {'fitting','selection'}}
    nested = {'18': fit_ids[:18], '36': fit_ids}
    files = {'data/examples.json': rows, 'data/fit_gold.json': train_gold, 'data/nested_fit_ids.json': nested,
             'data/manifest.json': {'artificial_fixture': True}}
    for name, value in files.items(): fixed_write(template/name, value)
    protocol = {'protocol': 'anli-learnability-full-v1', 'revision': 1,
                'data': {'expected_files': {name: file_hash(template/name) for name in files}}}
    monkeypatch.setattr(s, 'protocol', lambda: copy.deepcopy(protocol))
    monkeypatch.setattr(s.profile, 'source_identity', lambda: {'files': {'artificial_source.py': 'a'*64}})
    events, allocations = [], []
    control = {'job': 'job1', 'checkpoint_role': None, 'checkpoint_cpu': None}
    allocation = lambda: {'hostname': 'artificial-host', 'slurm_job_id': control['job']}
    monkeypatch.setattr(s.profile.v3, '_allocation', allocation)
    monkeypatch.setattr(s, 'cpu_environment', lambda runtime: {'python': sys.executable, 'model_dir': str(tmp_path/'model'), 'packages': {}})
    def acquire(archive, **kwargs):
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists(): archive.write_bytes(b'artificial source, no real labels')
    monkeypatch.setattr(s.data, 'acquire_source', acquire)
    def prepare_data(archive, historical, out):
        for name, value in files.items(): fixed_write(out/name, value)
    monkeypatch.setattr(s.data, 'prepare', prepare_data)
    monkeypatch.setattr(s.data, 'load_prepared', lambda out: {
        'examples': read_json(out/'data/examples.json'), 'fit_gold': read_json(out/'data/fit_gold.json'),
        'fit_ids_by_size': read_json(out/'data/nested_fit_ids.json'), 'manifest': read_json(out/'data/manifest.json')})
    monkeypatch.setattr(s.profile, 'prepare', lambda dirs, out: fixed_write(out/'settings.json', {'artificial_profile': True}))
    def initialize(out, *, new_allocation=False):
        allocations.append(new_allocation)
        value = {'allocation': allocation(), 'deadline_epoch': time.time()+3600,
                 'settings_hash': fingerprint(read_json(out/'settings.json'))}
        write_json(out/'runtime/session.json', value)
        return value
    monkeypatch.setattr(s.profile, 'initialize', initialize)
    monkeypatch.setattr(s.ownership, 'run_selftest', lambda out: events.append('ownership'))
    def preflight(out, **kwargs):
        events.append('preflight')
        fixed_write(out/'accepted.json', {'artificial_preflight': True})
    monkeypatch.setattr(s.profile, 'run', preflight)
    def prepare_collection(out, profile_out, inputs, identity):
        assert inputs == rows
        fixed_write(out/'settings.json', {'study_identity': identity})
    monkeypatch.setattr(s.collection, 'prepare', prepare_collection)
    def collect(out, profile_out, role, **kwargs):
        marker = out/f'{role}.json'
        if marker.exists(): events.append(f'reuse:{role}'); return {'status': 'COMPLETE'}
        if control['checkpoint_role'] == role:
            control['checkpoint_role'] = None; events.append('checkpoint:'+role); return {'status': 'CHECKPOINT', 'role': role}
        events.append('collect:'+role); fixed_write(marker, predictions[role]); return {'status': 'COMPLETE'}
    monkeypatch.setattr(s.collection, 'collect_role', collect)
    monkeypatch.setattr(s.collection, 'predictions', lambda out, role, **kwargs: read_json(out/f'{role}.json'))
    def get_embeddings(out, **kwargs):
        assert (out/'embeddings/complete.json').is_file()
        return [r['sample_id'] for r in rows], np.asarray(embeddings)
    monkeypatch.setattr(s.collection, 'embeddings', get_embeddings)
    real_cpu_stage = s.cpu_stage
    def cpu(out, name, command, **kwargs):
        events.append(name)
        if control['checkpoint_cpu'] == name:
            control['checkpoint_cpu'] = None; events.append('cpu-checkpoint:'+name); return False
        if name == 'embeddings':
            fixed_write(out/'collection/embeddings/identity.json', {'artificial_embeddings': True})
            fixed_write(out/'collection/embeddings/complete.json', {'rows': len(rows)})
            matrix_path = out/'collection/embeddings/chunks/artificial/vectors.npy'
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            if not matrix_path.exists():
                with matrix_path.open('xb') as stream: np.save(stream, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
        elif name == 'fit-and-freeze-scores': s.fit_and_score(out)
        elif name == 'evaluate': s.evaluate(out)
        else: pytest.fail('Unexpected CPU stage '+name)
        return True
    monkeypatch.setattr(s, 'cpu_stage', cpu)
    original_freeze = s.freeze
    def freeze(out):
        assert not (out/'evaluation/access.json').exists()
        events.append('freeze')
        return original_freeze(out)
    monkeypatch.setattr(s, 'freeze', freeze)
    def evaluation_gold(archive, manifest, inputs):
        assert (output/'freeze.json').is_file()
        assert (output/'evaluation/access.json').is_file()
        s.verify_freeze(output)
        events.append('gold')
        return {r['sample_id']: gold[r['sample_id']] for r in inputs if r['split'] in s.EVAL_SPLITS}
    monkeypatch.setattr(s.data, 'evaluation_gold', evaluation_gold)
    return {'runtime': runtime, 'output': output, 'events': events, 'control': control, 'allocations': allocations,
            'args': ['start', '--runtime-root', str(runtime), '--output-dir', str(output)],
            'cpu_stage': real_cpu_stage, 'rows': rows,
            'eval_gold': {r['sample_id']: gold[r['sample_id']] for r in rows if r['split'] in s.EVAL_SPLITS}}


def test_one_command_reaches_freeze_then_gold_then_numeric_audit_and_bundle(pipeline):
    p = pipeline
    assert s.main(p['args']) == 0
    assert 'ownership' in p['events'], 'GPU collection began without the ownership gate'
    assert p['events'].index('ownership') < p['events'].index('collect:startlux4b')
    assert p['events'].index('fit-and-freeze-scores') < p['events'].index('freeze') < p['events'].index('gold')
    assert p['events'].count('gold') == 1
    assert (p['output']/'evaluation/complete.json').is_file()
    assert (p['output']/'REPORT.md').is_file()
    assert s.audit(p['output'])['status'] == 'PASS_PACKAGED_EVIDENCE'
    assert len(list((p['output']/'exports').glob('learnability_bundle-*.zip'))) == 1
    assert read_json(p['output']/'status.json')['status'] == 'COMPLETE'
    # Rerunning a complete study performs read-only audit/report, without any model call.
    before = list(p['events'])
    assert s.main(p['args']) == 0 and p['events'] == before


def test_collection_checkpoint_auto_renews_different_allocation_and_reuses_success(pipeline):
    p = pipeline; p['control']['checkpoint_role'] = 'qwen9b'
    assert s.main(p['args']) == 0
    assert not (p['output']/'freeze.json').exists() and 'gold' not in p['events']
    p['control']['job'] = 'job2'
    assert s.main(p['args']) == 0
    assert p['allocations'] == [False, True]
    assert p['events'].count('collect:startlux4b') == 1 and 'reuse:startlux4b' in p['events']
    assert p['events'].count('gold') == 1


def test_evaluation_checkpoint_after_freeze_survives_session_change_without_refit(pipeline):
    p = pipeline; p['control']['checkpoint_cpu'] = 'evaluate'
    assert s.main(p['args']) == 0
    assert (p['output']/'freeze.json').is_file() and 'gold' not in p['events']
    frozen_hash = file_hash(p['output']/'freeze.json')
    before_fit = p['events'].count('fit-and-freeze-scores')
    p['control']['job'] = 'job2'
    assert s.main(p['args']) == 0
    assert file_hash(p['output']/'freeze.json') == frozen_hash
    assert p['events'].count('fit-and-freeze-scores') == before_fit
    assert p['allocations'] == [False, True]


@pytest.mark.parametrize('name', ['solo-v1', 'solo-v3', 'aux-size-control-v1-r2'])
def test_old_output_roots_rejected_before_any_lock_or_diagnostics(pipeline, name):
    p = pipeline; target = p['runtime']/'results'/name
    before = {str(f.relative_to(target)): f.read_bytes() for f in target.rglob('*') if f.is_file()}
    assert s.main(['start', '--runtime-root', str(p['runtime']), '--output-dir', str(target)]) == 2
    after = {str(f.relative_to(target)): f.read_bytes() for f in target.rglob('*') if f.is_file()}
    assert before == after and not p['events']


def test_audit_rejects_altered_result_even_after_receipt_hash_update(pipeline):
    p = pipeline; assert s.main(p['args']) == 0
    result_path = p['output']/'evaluation/results.json'
    results = read_json(result_path); results['eligible_dev']['baselines']['direct_startlux4b']['correct'] += 1
    write_json(result_path, results)
    receipt_path = p['output']/'evaluation/complete.json'; receipt = read_json(receipt_path)
    receipt['results_sha256'] = file_hash(result_path); write_json(receipt_path, receipt)
    with pytest.raises(StudyError, match='recomputation'):
        s.audit(p['output'])


def test_complete_cache_rejects_changed_frozen_scores(pipeline):
    p = pipeline; p['control']['checkpoint_cpu'] = 'fit-and-freeze-scores'
    assert s.main(p['args']) == 0
    s.fit_and_score(p['output'])
    assert not (p['output']/'freeze.json').exists()
    score_path = p['output']/'fitting/scores-eligible_dev.json'
    scores = read_json(score_path); next(iter(scores['cells'].values()))['scores'][0] += .5
    write_json(score_path, scores)
    with pytest.raises(StudyError, match='changed'):
        s.fit_and_score(p['output'])


def test_unreceipted_partial_models_are_recomputed_and_never_trusted(pipeline, monkeypatch):
    p = pipeline; p['control']['checkpoint_cpu'] = 'fit-and-freeze-scores'
    assert s.main(p['args']) == 0
    fixed_write(p['output']/'fitting/models.json', {'arbitrary_unaccepted_model': True})
    calls = []
    original = m.fit_suite
    def counted(*args, **kwargs): calls.append(True); return original(*args, **kwargs)
    monkeypatch.setattr(m, 'fit_suite', counted)
    with pytest.raises(StudyError): s.fit_and_score(p['output'])
    assert calls == [True]
    assert not (p['output']/'fitting/complete.json').exists() and 'gold' not in p['events']


def test_cpu_worker_failure_recovers_descendants_even_if_direct_stop_errors(pipeline, monkeypatch):
    p = pipeline
    s.prepare(p['output'], p['runtime']); s.initialize_allocation(p['output'])
    monkeypatch.setattr(s.profile, 'subreaper', lambda: nullcontext())
    events = []
    class Child:
        pid = 999999; returncode = 7
        def poll(self): return self.returncode
    monkeypatch.setattr(s.subprocess, 'Popen', lambda *a, **kw: Child())
    def stop(*args): events.append('direct-stop'); raise OSError('artificial stop error')
    def recover(*args, **kwargs): events.append('recover'); return {'gpu_compute_processes': []}
    monkeypatch.setattr(s.profile, 'stop_child', stop)
    monkeypatch.setattr(s.profile, 'recover_stop', recover)
    with pytest.raises(StudyError, match='recovery completed'):
        p['cpu_stage'](p['output'], 'test-worker', ['fake'])
    assert events == ['direct-stop', 'recover']
    attempts = list((p['output']/'cpu-attempts/test-worker').iterdir())
    assert len(attempts) == 1
    assert (attempts[0]/'parent-stop.json').exists() and not (attempts[0]/'finished.json').exists()


def test_cpu_restart_requires_cleanup_or_proven_ended_old_job(pipeline, monkeypatch):
    p = pipeline
    s.prepare(p['output'], p['runtime']); s.initialize_allocation(p['output'])
    folder = p['output']/'cpu-attempts/fit/old'
    fixed_write(folder/'plan.json', {'allocation': {'hostname': 'artificial-host', 'slurm_job_id': 'job1'}})
    with pytest.raises(StudyError, match='cleanup proof'):
        s.previous_cpu_stopped(p['output'], 'fit')
    p['control']['job'] = 'job2'; s.initialize_allocation(p['output'])
    monkeypatch.setattr(s.profile, 'slurm_job_state', lambda job: {'terminal': False})
    with pytest.raises(StudyError, match='ended'):
        s.previous_cpu_stopped(p['output'], 'fit')
    monkeypatch.setattr(s.profile, 'slurm_job_state', lambda job: {'terminal': True, 'job': job})
    monkeypatch.setattr(s.profile.old, 'assert_gpu_owned_by', lambda owned: [])
    s.previous_cpu_stopped(p['output'], 'fit')
    assert read_json(folder/'recovered-stop.json')['no_historical_pid_signalled'] is True


def test_evaluation_checks_frozen_bytes_before_access_receipt_or_gold(pipeline, monkeypatch):
    p = pipeline; p['control']['checkpoint_cpu'] = 'evaluate'
    assert s.main(p['args']) == 0
    model_path = p['output']/'fitting/models.json'
    models = read_json(model_path); models['unexpected_edit_after_freeze'] = True
    write_json(model_path, models)
    calls = []
    def gold(*args): calls.append(True); return p['eval_gold']
    monkeypatch.setattr(s.data, 'evaluation_gold', gold)
    with pytest.raises(StudyError, match='Frozen evidence changed'):
        s.evaluate(p['output'])
    assert not calls and not (p['output']/'evaluation/access.json').exists()


def test_compact_export_rechecks_metrics_without_feature_bytes_and_binds_omissions(pipeline, tmp_path):
    p = pipeline; assert s.main(p['args']) == 0
    bundle = next((p['output']/'exports').glob('learnability_bundle-*.zip'))
    unpacked = tmp_path/'returned-bundle'
    with zipfile.ZipFile(bundle) as archive:
        assert not any(name.startswith('source/') for name in archive.namelist())
        assert not any(name.endswith('.npy') for name in archive.namelist())
        archive.extractall(unpacked)
    assert s.audit(unpacked)['status'] == 'PASS_PACKAGED_EVIDENCE'
    omitted = read_json(unpacked/'OMITTED_FEATURES.json')
    assert len(omitted) == 1
    next(iter(omitted.values()))['sha256'] = '0'*64
    write_json(unpacked/'OMITTED_FEATURES.json', omitted)
    with pytest.raises(StudyError, match='Frozen evidence changed'):
        s.audit(unpacked)


def test_missing_completion_after_freeze_cannot_restart_fitting(pipeline, monkeypatch):
    p = pipeline; assert s.main(p['args']) == 0
    (p['output']/'fitting/complete.json').unlink()
    calls = []
    original = m.fit_suite
    def fit(*args, **kwargs): calls.append(True); return original(*args, **kwargs)
    monkeypatch.setattr(m, 'fit_suite', fit)
    with pytest.raises((StudyError, FileNotFoundError)):
        s.fit_and_score(p['output'])
    assert not calls, 'A missing receipt after gold access must not trigger a new fit'
    assert not (p['output']/'fitting/complete.json').exists()


def test_compact_reexport_preserves_omissions_without_duplicate_members(pipeline, tmp_path):
    p = pipeline; assert s.main(p['args']) == 0
    bundle = next((p['output']/'exports').glob('learnability_bundle-*.zip'))
    unpacked = tmp_path/'first-return'
    with zipfile.ZipFile(bundle) as archive: archive.extractall(unpacked)
    before = read_json(unpacked/'OMITTED_FEATURES.json')
    second_bundle = s.export(unpacked)
    second = tmp_path/'second-return'
    with zipfile.ZipFile(second_bundle) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'A compact re-export must not duplicate generated manifests'
        archive.extractall(second)
    assert read_json(second/'OMITTED_FEATURES.json') == before
    assert s.audit(second)['status'] == 'PASS_PACKAGED_EVIDENCE'
