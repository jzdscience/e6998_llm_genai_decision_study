"""One command, automatic scientific boundaries, resumable real Slurm execution.

Only evaluation_worker opens selected evaluation gold, after immutable fitting,
score and evidence hashes have been frozen. Failed checks stop, successful ones
continue without an external/manual approval round.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile

from ..common import StudyError, canonical, exclusive_run, file_hash, fingerprint, read_json, utc_now, write_json
from ..service_study.identity import fixed_write
from ..learnability_profile import study as profile
from ..learnability_profile import ownership
from . import collection, data, modeling

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / 'configs/anli_learnability_full_protocol.json'
EVAL_SPLITS = ('eligible_dev', 'train_holdout')


def protocol():
    value = read_json(CONFIG)
    if value.get('protocol') != 'anli-learnability-full-v1' or value.get('revision') != 1:
        raise StudyError('Unsupported full-study protocol')
    return value


def path(output, relative):
    return profile.safe(output, relative)


def validate_output(output, runtime=None, *, existing=False):
    output = profile.normal(output)
    marker = path(output, 'study.json')
    if output == PROJECT or output in PROJECT.parents or PROJECT in output.parents:
        raise StudyError('Study output must be outside the source checkout')
    if runtime is not None:
        runtime = profile.normal(runtime)
        if output.parent != runtime/'results' or output.name != 'aux-anli-learnability-v1':
            raise StudyError('Use the separate results/aux-anli-learnability-v1 output')
    if marker.exists():
        if read_json(marker).get('protocol') != protocol():
            raise StudyError('Output belongs to another protocol; preserve it')
    elif existing or (output.exists() and any(output.iterdir())):
        raise StudyError('Output contains unrelated files or lacks its study identity')


def context(output):
    settings = read_json(path(output, 'study.json'))
    if settings['protocol'] != protocol() or settings['source_files'] != profile.source_identity()['files']:
        raise StudyError('Study source/configuration changed; preserve results and use the pinned checkout')
    for name, digest in protocol()['data']['expected_files'].items():
        if file_hash(path(output, name)) != digest:
            raise StudyError(f'Frozen prepared data differs: {name}')
    prepared = data.load_prepared(output)
    return settings, prepared


def cpu_environment(runtime):
    candidates = [runtime.parent/'decision-study-src/.venv-cheap-baselines/bin/python',
                  runtime/'.venv-cheap-baselines/bin/python']
    expected = read_json(PROJECT/'configs/clinc_cheap_protocol.json')['environment']
    code = 'import json,sys,importlib.metadata as m; names=json.loads(sys.argv[1]); print(json.dumps({n:m.version(n) for n in names},sort_keys=True))'
    for python in candidates:
        if not python.is_file():
            continue
        result = subprocess.run([str(python), '-I', '-B', '-c', code, json.dumps(list(expected))],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and json.loads(result.stdout) == expected:
            model_dir = runtime/'results/aux-clinc-cheap-v1-inputs/model'
            collection.encoder.verify(model_dir, {'encoder': collection.encoder.ENCODER_SPEC})
            return {'python': str(python), 'model_dir': str(model_dir), 'packages': expected}
    raise StudyError('Existing pinned cheap-baseline CPU environment is missing/changed; no automatic environment replacement')


def prepare(output, runtime):
    output, runtime = profile.normal(output), profile.normal(runtime)
    settings = {'protocol': protocol(), 'source_files': profile.source_identity()['files'],
                'runtime_root': str(runtime), 'scope': 'Fixed one-command experiment; no manual successful-stage handoff'}
    fixed_write(path(output, 'study.json'), settings)
    archive = path(output, 'source/anli_v1.0.zip')
    candidates = [runtime/'results/solo-v3/source/anli_v1.0.zip',
                  runtime/'results/solo-v3/data/anli_v1.0.zip']
    data.acquire_source(archive, candidates=candidates)
    data.prepare(archive, runtime/'results/solo-v3/data/examples.jsonl', output)
    context(output)
    cpu = cpu_environment(runtime)
    fixed_write(path(output, 'runtime/cpu-environment.json'), cpu)
    dirs = {'v1': runtime/'results/solo-v1', 'v3': runtime/'results/solo-v3',
            'size_control': runtime/'results/aux-size-control-v1-r2'}
    profile.prepare(dirs, path(output, 'profile'))
    return settings


def initialize_allocation(output):
    root = path(output, 'profile')
    previous = path(root, 'runtime/session.json')
    new = False
    if previous.exists():
        old = read_json(previous)['allocation']
        current = profile.v3._allocation()
        new = old != current and old.get('slurm_job_id') != current.get('slurm_job_id')
    # The profile helper verifies an actual different job and terminal old job.
    return profile.initialize(root, new_allocation=new)


def remaining(output):
    return profile.session(path(output, 'profile'))['deadline_epoch']-time.time()


def stage_status(output, stage, status, **details):
    write_json(path(output, 'status.json'), {'stage': stage, 'status': status, 'at': utc_now(), **details})
    print(f'{stage}: {status}', flush=True)


def previous_cpu_stopped(output, name, *, retry_failed=False):
    current = profile.session(path(output, 'profile'))['allocation']
    base = path(output, 'cpu-attempts/'+name)
    for folder in sorted(base.iterdir()) if base.exists() else []:
        if not path(folder, 'plan.json').exists():
            continue
        plan = read_json(path(folder, 'plan.json'))
        if path(folder, 'failure.json').exists() and not retry_failed:
            raise StudyError(f'Earlier {name} failed; inspect diagnostics, then use --retry-failed')
        stopped = next((p for p in (path(folder, 'parent-stop.json'), path(folder, 'recovered-stop.json')) if p.exists()), None)
        if stopped is not None:
            receipt = read_json(stopped)
            if receipt.get('gpu_compute_processes') != [] or receipt.get('plan_hash') != fingerprint(plan):
                raise StudyError('Earlier CPU stage has invalid stop evidence')
            if stopped.name == 'parent-stop.json':
                if receipt.get('gpu_checked') is not True or 'kernel pidfds' not in receipt.get('ownership', ''):
                    raise StudyError('CPU parent stop does not prove owned descendant recovery')
            else:
                proof = receipt.get('old_allocation_proof', {})
                old_job = plan['allocation']['slurm_job_id']
                new_job = receipt.get('new_allocation', {}).get('slurm_job_id')
                if not proof.get('terminal') or proof.get('job_id') != old_job or not new_job or new_job == old_job:
                    raise StudyError('CPU allocation recovery belongs to a different or unfinished job')
            continue
        if plan['allocation'] == current:
            raise StudyError('Earlier CPU worker lacks cleanup proof in this allocation; refusing concurrent restart')
        proof = profile.slurm_job_state(plan['allocation']['slurm_job_id'])
        if not proof.get('terminal'):
            raise StudyError('Cannot prove earlier CPU allocation ended')
        gpu = profile.old.assert_gpu_owned_by(set())
        fixed_write(path(folder, 'recovered-stop.json'), {'old_allocation_proof': proof,
                    'new_allocation': current, 'gpu_compute_processes': gpu, 'no_historical_pid_signalled': True,
                    'plan_hash': fingerprint(plan)})


def cpu_stage(output, name, command, *, retry_failed=False):
    """One bounded, owned CPU process; checkpoints retain immutable feature chunks."""
    if remaining(output) < 300:
        return False
    previous_cpu_stopped(output, name, retry_failed=retry_failed)
    folder = path(output, f'cpu-attempts/{name}/{time.time_ns()}')
    folder.mkdir(parents=True)
    began = time.time()
    deadline = profile.session(path(output, 'profile'))['deadline_epoch']-120
    plan = {'stage': name, 'started_epoch': began, 'deadline_epoch': deadline,
            'allocation': profile.session(path(output, 'profile'))['allocation'], 'last_services_before': None,
            'command': command}
    fixed_write(path(folder, 'plan.json'), plan)
    child = None
    checkpoint = False
    with profile.subreaper():
        try:
            with path(folder, 'worker.log').open('x') as log:
                child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            fixed_write(path(folder, 'worker-process.json'), {'pid': child.pid, 'pgid': child.pid})
            while child.poll() is None:
                if time.time() >= deadline:
                    checkpoint = True
                    break
                profile.session(path(output, 'profile'))
                try: child.wait(timeout=min(.5, max(.001, deadline-time.time())))
                except subprocess.TimeoutExpired: pass
            if checkpoint:
                profile.stop_child(child, began)
            elif child.returncode != 0 or time.time() > deadline:
                raise StudyError(f'{name} worker failed (exit {child.returncode}); see {folder}')
        except BaseException as exc:
            fixed_write(path(folder, 'failure.json'), {'error': str(exc), 'type': type(exc).__name__})
            raise
        finally:
            stop_error = None
            if child is not None:
                try:
                    profile.stop_child(child, began)
                except BaseException as exc:
                    stop_error = exc
            stopped = profile.recover_stop(path(output, 'collection'), 'qwen9b', plan,
                                           worker_pid=child.pid if child is not None else None,
                                           journal=path(folder, 'parent-cleanup.jsonl'))
            stopped['plan_hash'] = fingerprint(plan)
            fixed_write(path(folder, 'parent-stop.json'), stopped)
            if stop_error is not None:
                fixed_write(path(folder, 'worker-stop-error.json'), {'error': str(stop_error)})
                raise StudyError('CPU child stop failed; descendant recovery completed but stage is not accepted') from stop_error
    fixed_write(path(folder, 'finished.json'), {'status': 'CHECKPOINT' if checkpoint else 'COMPLETE',
                                               'plan_hash': fingerprint(plan), 'at': utc_now()})
    return not checkpoint


def prediction_maps(output, *, offline=False):
    return {role: {r['sample_id']: r for r in collection.predictions(path(output, 'collection'), role, offline=offline)}
            for role in profile.ROLES}


def fit_and_score(output):
    """Gold-free evaluation scoring; fitting receives exactly fitting+selection gold."""
    _, prepared = context(output)
    rows = prepared['examples']
    if path(output, 'freeze.json').exists() or path(output, 'evaluation/access.json').exists():
        verify_fitting(output)
        verify_freeze(output)
        return
    if path(output, 'fitting/complete.json').exists():
        verify_fitting(output)
        return
    predictions = prediction_maps(output)
    ids, vectors = collection.embeddings(path(output, 'collection'))
    features = dict(zip(ids, vectors))
    fitting = [r for r in rows if r['split'] == 'fitting']
    selection = [r for r in rows if r['split'] == 'selection']
    model_path = path(output, 'fitting/models.json')
    # Recompute an unreceipted partial result, rather than trusting it as a cache.
    models = modeling.fit_suite(fitting, selection, predictions['startlux4b'],
                    {r: predictions[r] for r in modeling.FALLBACKS}, prepared['fit_gold'], features,
                    prepared['fit_ids_by_size'])
    fixed_write(model_path, models)
    for split in EVAL_SPLITS:
        examples = [r for r in rows if r['split'] == split]
        scores = modeling.score_suite(models, examples, predictions['startlux4b'], features)
        fixed_write(path(output, f'fitting/scores-{split}.json'), scores)
    fixed_write(path(output, 'fitting/complete.json'), {
        'models_sha256': file_hash(model_path),
        'scores': {split: file_hash(path(output, f'fitting/scores-{split}.json')) for split in EVAL_SPLITS},
        'prepared_manifest_sha256': file_hash(path(output, 'data/manifest.json')),
        'feature_identity_sha256': file_hash(path(output, 'collection/embeddings/identity.json')),
        'evaluation_gold_used': False, 'recipe': modeling.RECIPE})


def verify_fitting(output):
    fitting = read_json(path(output, 'fitting/complete.json'))
    if (fitting['recipe'] != modeling.RECIPE or fitting['evaluation_gold_used'] is not False
            or fitting['models_sha256'] != file_hash(path(output, 'fitting/models.json'))
            or fitting['prepared_manifest_sha256'] != file_hash(path(output, 'data/manifest.json'))
            or fitting['feature_identity_sha256'] != file_hash(path(output, 'collection/embeddings/identity.json'))
            or fitting['scores'] != {s: file_hash(path(output, f'fitting/scores-{s}.json')) for s in EVAL_SPLITS}):
        raise StudyError('Fitted models/scores or their prepared inputs/features changed')
    return fitting


def mutable_runtime_file(relative):
    return (relative.endswith('/runtime/session.json') or relative.endswith('/runtime/ownership-selftest.json')
            or '/runtime/session-history/' in relative or '/runtime/allocation-checks/' in relative)


def files_under(output, prefixes):
    output = profile.normal(output)
    files = {}
    for prefix in prefixes:
        base = path(output, prefix)
        if base.is_file():
            files[prefix] = file_hash(base)
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            for name in dirs:
                if (Path(directory)/name).is_symlink(): raise StudyError('Symlinked evidence directory')
            dirs[:] = sorted(n for n in dirs if n not in {'runtime-work', '__pycache__', 'exports'})
            for name in sorted(names):
                relative = str((Path(directory)/name).relative_to(output))
                p = path(output, relative)
                if name == '.active.lock' or mutable_runtime_file(relative): continue
                files[relative] = file_hash(p)
    return files


def freeze(output):
    context(output)
    verify_fitting(output)
    if path(output, 'evaluation/access.json').exists() and not path(output, 'freeze.json').exists():
        raise StudyError('Evaluation access exists without prior freeze')
    prediction_maps(output)
    collection.embeddings(path(output, 'collection'))
    value = {'protocol_hash': fingerprint(protocol()), 'files': files_under(output,
             ('study.json', 'data', 'fitting', 'profile', 'collection', 'cpu-attempts', 'runtime/cpu-environment.json')),
             'models_and_scores_frozen_before_evaluation_gold': True}
    fixed_write(path(output, 'freeze.json'), value)
    return value


def verify_freeze(output, *, allow_omitted=False):
    context(output)
    value = read_json(path(output, 'freeze.json'))
    if value.get('protocol_hash') != fingerprint(protocol()) or value.get('models_and_scores_frozen_before_evaluation_gold') is not True:
        raise StudyError('Freeze identity differs')
    omitted = read_json(path(output, 'OMITTED_FEATURES.json')) if allow_omitted and path(output, 'OMITTED_FEATURES.json').exists() else {}
    for name, digest in value['files'].items():
        p = path(output, name)
        if not p.exists() and name in omitted and name.startswith('collection/embeddings/chunks/') and name.endswith('.npy'):
            if omitted[name]['sha256'] == digest: continue
        if not p.is_file() or file_hash(p) != digest:
            raise StudyError(f'Frozen evidence changed or is missing: {name}')
    return value


def evaluate(output):
    verify_freeze(output)
    freeze_hash = file_hash(path(output, 'freeze.json'))
    access_path = path(output, 'evaluation/access.json')
    if access_path.exists():
        if read_json(access_path)['freeze_sha256'] != freeze_hash:
            raise StudyError('Evaluation access is bound to a different freeze')
    else:
        fixed_write(access_path, {'freeze_sha256': freeze_hash, 'at': utc_now(),
                                  'purpose': 'Fixed post-freeze evaluation only; no model selection'})
    _, prepared = context(output)
    examples = prepared['examples']
    gold = data.evaluation_gold(path(output, 'source/anli_v1.0.zip'), prepared['manifest'], examples)
    fixed_write(path(output, 'sealed/evaluation_gold.json'), gold)
    predictions = prediction_maps(output)
    results = {}
    for split in EVAL_SPLITS:
        rows = [r for r in examples if r['split'] == split]
        results[split] = modeling.evaluate_suite(read_json(path(output, f'fitting/scores-{split}.json')), rows,
                        predictions['startlux4b'], {r: predictions[r] for r in modeling.FALLBACKS},
                        {r['sample_id']: gold[r['sample_id']] for r in rows})
    fixed_write(path(output, 'evaluation/results.json'), results)
    fixed_write(path(output, 'evaluation/complete.json'), {'freeze_sha256': freeze_hash,
        'access_sha256': file_hash(access_path), 'gold_sha256': file_hash(path(output, 'sealed/evaluation_gold.json')),
        'results_sha256': file_hash(path(output, 'evaluation/results.json'))})
    return results


def audit(output):
    """Returned-evidence integrity; omitted feature matrices need separate replay."""
    verify_freeze(output, allow_omitted=True)
    receipt = read_json(path(output, 'evaluation/complete.json'))
    checks = {'freeze_sha256': 'freeze.json', 'access_sha256': 'evaluation/access.json',
              'gold_sha256': 'sealed/evaluation_gold.json', 'results_sha256': 'evaluation/results.json'}
    if any(receipt.get(key) != file_hash(path(output, name)) for key, name in checks.items()):
        raise StudyError('Evaluation receipt changed')
    if read_json(path(output, 'evaluation/access.json'))['freeze_sha256'] != receipt['freeze_sha256']:
        raise StudyError('Gold access does not bind the frozen models/scores')
    predictions = prediction_maps(output, offline=True)
    _, prepared = context(output)
    gold = read_json(path(output, 'sealed/evaluation_gold.json'))
    expected_gold_ids = {r['sample_id'] for r in prepared['examples'] if r['split'] in EVAL_SPLITS}
    if set(gold) != expected_gold_ids:
        raise StudyError('Returned evaluation gold IDs differ')
    recomputed = {}
    for split in EVAL_SPLITS:
        rows = [r for r in prepared['examples'] if r['split'] == split]
        recomputed[split] = modeling.evaluate_suite(read_json(path(output, f'fitting/scores-{split}.json')), rows,
                        predictions['startlux4b'], {r: predictions[r] for r in modeling.FALLBACKS},
                        {r['sample_id']: gold[r['sample_id']] for r in rows})
    if fingerprint(recomputed) != fingerprint(read_json(path(output, 'evaluation/results.json'))):
        raise StudyError('Returned metrics differ from frozen-score/raw-prediction recomputation')
    return {'status': 'PASS_PACKAGED_EVIDENCE', 'results': recomputed,
            'limit': 'Compact bundle omits feature matrix bytes; numerical feature/model replay requires retained matrices or deterministic regeneration. No new live node/GPU observation.'}


def report(output):
    results = read_json(path(output, 'evaluation/results.json'))
    lines = ['# ANLI learnability results', '', 'The primary population is the locally audited eligible dev subset. External usage is unconfirmed.',
             'All other fits, the train-source holdout and deployment thresholds are secondary/descriptive.', '']
    for split, result in results.items():
        lines += [f'## {split} ({result["n"]} inputs)', '', '| Configuration | Accuracy (%) | Deferred |', '|---|---:|---:|']
        for name, cell in {**result['baselines'], **result['cells']}.items():
            lines.append(f'| {name} | {100*cell["accuracy"]:.3f} | {cell.get("deferred", "—")} |')
        main = result['comparisons'][f'qwen9b/n{max(modeling.FIT_SIZES)}']
        lines += ['', f'Text − front gain: {100*main["accuracy_difference"]:.3f} pp. Interval: {main["interval"]}.', '']
    lines += ['The 1 pp practical reference is not an equivalence margin. The offline exact quota is not an online budget guarantee.', '']
    target = path(output, 'REPORT.md')
    payload = '\n'.join(lines)
    if target.exists() and target.read_text() != payload:
        raise StudyError('Existing report differs from fixed results')
    target.write_text(payload)


def export(output):
    output = profile.normal(output)
    files = []
    omitted_path = path(output, 'OMITTED_FEATURES.json')
    omitted = read_json(omitted_path) if omitted_path.exists() else {}
    for directory, dirs, names in os.walk(output, followlinks=False):
        if any((Path(directory)/n).is_symlink() for n in dirs): raise StudyError('Symlinked evidence directory')
        dirs[:] = sorted(n for n in dirs if n not in {'runtime-work', 'exports', '__pycache__', 'source'})
        for name in sorted(names):
            relative = str((Path(directory)/name).relative_to(output))
            p = path(output, relative)
            if name in {'.active.lock', 'OMITTED_FEATURES.json', 'BUNDLE_MANIFEST.json'}: continue
            if relative.startswith('collection/embeddings/chunks/') and name.endswith('.npy'):
                omitted[relative] = {'sha256': file_hash(p), 'bytes': p.stat().st_size}
                continue
            files.append(p)
    target = path(output, f'exports/learnability_bundle-{time.time_ns()}.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(p.relative_to(output)): {'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in files}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files: archive.write(p, str(p.relative_to(output)))
        archive.writestr('OMITTED_FEATURES.json', canonical(omitted)+b'\n')
        archive.writestr('BUNDLE_MANIFEST.json', canonical(manifest)+b'\n')
    return target


def run(output, runtime, *, retry_failed=False):
    if path(output, 'evaluation/complete.json').exists():
        audit(output); report(output)
        return 'COMPLETE'
    prepare(output, runtime)
    initialize_allocation(output)
    if path(output, 'freeze.json').exists():
        # Do not mutate profile/collection evidence after freeze.
        command = [read_json(path(output, 'runtime/cpu-environment.json'))['python'], '-B', '-m',
                   'decision_study.learnability', 'evaluate', '--output-dir', str(output)]
        if not cpu_stage(output, 'evaluate', command, retry_failed=retry_failed): return 'CHECKPOINT'
        audit(output); report(output)
        return 'COMPLETE'
    stage_status(output, 'development-preflight', 'RUNNING')
    ownership.run_selftest(path(output, 'profile'))
    profile.run(path(output, 'profile'), retry_failed=retry_failed)
    settings, prepared = context(output)
    collection.prepare(path(output, 'collection'), path(output, 'profile'), prepared['examples'], fingerprint(settings))
    for role in profile.ROLES:
        stage_status(output, role, 'COLLECTING')
        result = collection.collect_role(path(output, 'collection'), path(output, 'profile'), role, retry_failed=retry_failed)
        if result['status'] == 'CHECKPOINT':
            stage_status(output, role, 'CHECKPOINT', **{k:v for k,v in result.items() if k not in {'role','status'}})
            return 'CHECKPOINT'
    cpu = read_json(path(output, 'runtime/cpu-environment.json'))
    commands = [
        ('embeddings', [cpu['python'], '-B', '-m', 'decision_study.learnability.collection', 'embed',
                        '--output-dir', str(path(output, 'collection')), '--model-dir', cpu['model_dir']]),
        ('fit-and-freeze-scores', [cpu['python'], '-B', '-m', 'decision_study.learnability', 'fit', '--output-dir', str(output)])]
    for name, command in commands:
        stage_status(output, name, 'RUNNING')
        if not cpu_stage(output, name, command, retry_failed=retry_failed): return 'CHECKPOINT'
    freeze(output)
    command = [cpu['python'], '-B', '-m', 'decision_study.learnability', 'evaluate', '--output-dir', str(output)]
    if not cpu_stage(output, 'evaluate', command, retry_failed=retry_failed): return 'CHECKPOINT'
    audit(output); report(output)
    stage_status(output, 'study', 'COMPLETE')
    return 'COMPLETE'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'fit', 'evaluate', 'audit', 'export'))
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args(argv)
    output = profile.normal(args.output_dir)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt(f'signal {sig}')))
    try:
        validate_output(output, args.runtime_root, existing=args.command != 'start')
    except Exception as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr); return 2
    if args.command == 'audit':
        print(json.dumps(audit(output), ensure_ascii=False, indent=2)); return 0
    if args.command in {'fit', 'evaluate'}:
        (fit_and_score if args.command == 'fit' else evaluate)(output)
        return 0
    with exclusive_run(output):
        try:
            if args.command == 'start':
                if args.runtime_root is None: raise StudyError('start requires --runtime-root')
                result = run(output, args.runtime_root, retry_failed=args.retry_failed)
                print(f'STUDY_{result}', flush=True)
            print(f'RESULT BUNDLE: {export(output)}', flush=True)
            return 0
        except BaseException as exc:
            write_json(path(output, f'diagnostics/failure-{time.time_ns()}.json'), {'error': str(exc), 'type': type(exc).__name__, 'traceback': traceback.format_exc()})
            try: print(f'DIAGNOSTIC BUNDLE: {export(output)}', flush=True)
            except Exception as exporting: print(f'Export failed: {exporting}', file=sys.stderr)
            print(f'BLOCKED: {exc}', file=sys.stderr)
            return 2
