"""Independent, read-only audit of saved HTTP measurement evidence.

Uses stdlib + numpy only, never imports the experiment implementation. Rebuilds
counts from each raw response and arrival/worker timestamps. Writes new audit
artifacts here, never into the archived study. No fitting or service requests.
"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import zipfile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--archive', type=Path)
parser.add_argument('--evidence-root', type=Path)
parser.add_argument('--out', type=Path, required=True, help='New audit output directory; existing directories are preserved')
args = parser.parse_args()
REPO = args.repo_root.expanduser().resolve()
ARCHIVE = (args.archive or REPO.parent / 'service_study_bundle.zip').expanduser().resolve()
EVIDENCE = (args.evidence_root or REPO.parent / 'project/solo-experiments-2026-10-30/decision-study').expanduser().resolve()
OUT = args.out.expanduser().resolve()
if OUT.exists():
    parser.error('Output already exists; choose a new audit directory')
OUT.mkdir(parents=True, exist_ok=False)
COMMIT = '8b5370d8aea7ed57dc42c92f4fe07bcf023a8e07'
GOLD_SHA = 'bc5ec4891caed470970115a41e2f6debe2a2967b3f5ff3496622a3d29311fc35'
CONDITIONS = ['minilm400', 'minilm1600', 'startlux4b', 'qwen4b']
LABELS = ['account_blocked', 'balance', 'bill_balance', 'bill_due', 'freeze_account', 'interest_rate',
          'min_payment', 'order_checks', 'pay_bill', 'pin_change', 'report_fraud', 'routing',
          'spending_history', 'transactions', 'transfer', 'oos']
MODELS = {'minilm400': '103247d32c6bd123cdfe37e9cb2fe78ed0e44c43523e3f08ca6ec960f91a5663',
          'minilm1600': '4e8f16765aa0e196d768d8c7f8fd42e089cbadaf2e9261a92d6909b3efb84dc6'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return sha(canon(value))


def unseal(value):
    assert value['self_hash'] == identity({k: v for k, v in value.items() if k != 'self_hash'})


def assert_equal(actual, saved, where):
    if isinstance(actual, dict):
        assert isinstance(saved, dict) and set(actual) == set(saved), (where, set(actual), set(saved))
        for key in actual:
            assert_equal(actual[key], saved[key], f'{where}/{key}')
    elif isinstance(actual, list):
        assert isinstance(saved, list) and len(actual) == len(saved), where
        for i, (a, b) in enumerate(zip(actual, saved)):
            assert_equal(a, b, f'{where}/{i}')
    elif isinstance(actual, float):
        assert math.isfinite(actual) and math.isclose(actual, saved, rel_tol=1e-11, abs_tol=1e-8), (where, actual, saved)
    else:
        assert actual == saved, (where, actual, saved)


def q(values, p):
    return float(np.quantile(values, p)) if values else None


def stats(values):
    return {'mean': statistics.fmean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'values': values}


def source(name):
    assert not name.startswith('/') and '..' not in Path(name).parts
    return subprocess.check_output(['git', 'show', f'{COMMIT}:{name}'], cwd=REPO)


def expected_schedule(examples, rate, repeat, p):
    # Separate arrival and sample RNG streams, rebuilt from the prespecified seed.
    seed = p['seed'] + 1000 * repeat + rate
    times, samples = random.Random(seed), random.Random(seed + 1000000)
    ids = [e['sample_id'] for e in examples]
    result, cycle, elapsed = [], [], 0.0
    while True:
        elapsed += times.expovariate(rate)
        if elapsed >= p['duration_seconds']:
            return result
        if not cycle:
            cycle = ids[:]
            samples.shuffle(cycle)
        result.append({'offer_id': len(result), 'sample_id': cycle.pop(), 'arrival_seconds': elapsed})


def parse_saved_response(condition, row, service, request_id):
    result = row['result']
    assert result['sample_id'] == row['sample_id'] and result['status'] == row['status']
    assert row['status'] == 'ok', (condition, row['status'], result.get('error'))
    raw = result['raw_response']
    if condition.startswith('minilm'):
        assert result['attempts'] == 1
        assert raw['recipe'] == condition and raw['classifier_sha256'] == MODELS[condition]
        assert raw['instance_id'] == service['service']['health']['instance_id']
        assert raw['labels'] == LABELS and raw['request_id'] == result['request_id'] == request_id
        assert raw['truncated'] is False and raw['token_count'] > 0
        pred, probabilities = raw['label'], dict(zip(LABELS, raw['probabilities']))
        assert len(raw['probabilities']) == 16 and raw['predicted_label'] == pred
        assert_equal(raw['encoder_ms'] + raw['classifier_ms'], raw['service_ms'], 'CPU component time')
    else:
        assert len(result['attempts']) == 1 and result['attempts'][0]['attempt'] == 1
        assert result['attempts'][0]['status'] == 'ok' and result['error'] is None
        assert result['model'] == {'startlux4b': 'startlux', 'qwen4b': 'qwen'}[condition]
        if condition == 'qwen4b':
            choice = raw['choices'][0]
            assert choice['finish_reason'] == 'stop'
            content = json.loads(choice['message']['content'])
            assert set(content) == {'label'}
            pred, probabilities = content['label'], None
        else:
            answer = raw['answers']['decision']
            pred, probabilities = answer['choice'], answer['probabilities']
    assert pred in LABELS and result['prediction'] == pred
    if probabilities is not None:
        assert set(probabilities) == set(LABELS)
        assert all(not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in probabilities.values())
        assert abs(math.fsum(probabilities.values()) - 1) <= 1e-5
        assert max(probabilities.values()) - probabilities[pred] <= 1e-8
    assert result['probabilities'] == probabilities
    return pred


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


index = json.loads((REPO / 'records/BUNDLE_INDEX.json').read_text())['bundles']
original = next(x for x in index if x['id'] == 'formal-experiment-20261102-181054')
baseline = next(x for x in index if x['id'] == 'cheap-baselines-20261105-073350')
for entry in (original, baseline):
    path = EVIDENCE / entry['archive']
    assert path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256']
with zipfile.ZipFile(EVIDENCE / original['archive']) as z:
    gold_bytes = z.read('data/gold.json')
    assert sha(gold_bytes) == GOLD_SHA
    gold = json.loads(gold_bytes)
    old_manifest = json.loads(z.read('bundle_manifest.json'))['files']
    old_examples = z.read('data/examples.jsonl')
    assert sha(old_examples) == old_manifest['data/examples.jsonl']
    old_main = sorted([json.loads(x) for x in old_examples.splitlines()
                       if json.loads(x)['task'] == 'clinc' and json.loads(x)['split'] == 'test'], key=lambda x: x['sample_id'])
    old_labels = {m: {} for m in ('startlux4b', 'qwen4b')}
    for m, name in (('startlux4b', 'startlux'), ('qwen4b', 'qwen')):
        for member in z.namelist():
            if member.startswith(f'predictions/{name}/shard-') and member.endswith('.json'):
                raw = z.read(member)
                assert sha(raw) == old_manifest[member]
                shard = json.loads(raw)
                assert identity(shard['rows']) == shard['sha256']
                for row in shard['rows']:
                    if row['task'] == 'clinc' and row['split'] == 'test':
                        assert row['sample_id'] not in old_labels[m]
                        old_labels[m][row['sample_id']] = row['prediction']
with zipfile.ZipFile(EVIDENCE / baseline['archive']) as z:
    baseline_freeze = json.loads(z.read('freeze.json'))
    for m in ('minilm400', 'minilm1600'):
        raw = z.read(f'{m}.predictions.jsonl')
        assert sha(raw) == baseline_freeze['artifacts'][f'{m}.predictions.jsonl']
        old_labels[m] = {row['sample_id']: row['predicted_label'] for row in map(json.loads, raw.splitlines()) if row['split'] == 'test'}
assert all(len(v) == 600 for v in old_labels.values())

cells, raw_metrics, allocation_groups = [], [], defaultdict(list)
prediction_counts = {m: defaultdict(Counter) for m in CONDITIONS}
per_rate_predictions = defaultdict(lambda: defaultdict(Counter))
errors, temporal_findings = [], []
with zipfile.ZipFile(ARCHIVE) as z:
    names = z.namelist()
    assert len(names) == len(set(names))
    get = lambda n: json.loads(z.read(n))
    settings, freeze, access, evaluation = map(get, ['settings.json', 'freeze.json', 'evaluation-access.json', 'evaluation.json'])
    p = settings['protocol']
    assert settings['source']['commit'] == COMMIT and p == json.loads(source('configs/service_study_protocol.json'))
    for name, digest in settings['source']['files'].items():
        assert sha(source(name)) == digest, name
    main, warmup = get('workload.json'), get('warmup.json')
    assert main == old_main and len(main) == 600 and len(warmup) == 64
    assert all(row['sample_id'] in gold for row in main)
    assert sha(z.read('workload.json')) == settings['workload_sha256']
    assert sha(z.read('warmup.json')) == settings['warmup_sha256']
    expected_order = []
    for repeat in range(5):
        conditions = CONDITIONS[:]
        random.Random(6998 + repeat).shuffle(conditions)
        for condition in conditions:
            rates = [5, 25, 100]
            random.Random(f'6998-{repeat}-{condition}').shuffle(rates)
            for rate in rates:
                expected_order.append({'cell_id': f'r{repeat}-{condition}-rate{rate}', 'condition': condition,
                    'rate': rate, 'repeat': repeat, 'schedule_file': f'schedules/r{repeat}-rate{rate}.json'})
    assert expected_order == settings['order'] and len(expected_order) == 60
    schedules = {}
    for name, digest in settings['schedules'].items():
        assert sha(z.read(name)) == digest
        repeat = int(name.split('/r')[1].split('-')[0]); rate = int(name.split('-rate')[1].split('.')[0])
        schedules[name] = get(name)
        assert schedules[name] == expected_schedule(main, rate, repeat, p), name
    assert len(schedules) == 15
    unseal(freeze); unseal(evaluation)
    assert freeze['status'] == 'FROZEN' and freeze['settings_hash'] == identity(settings)
    assert evaluation['status'] == 'COMPLETE'
    assert access['freeze_sha256'] == evaluation['freeze_sha256'] == sha(z.read('freeze.json'))
    assert evaluation['access_sha256'] == sha(z.read('evaluation-access.json'))
    assert access['gold_sha256'] == evaluation['gold_sha256'] == GOLD_SHA
    expected_complete = {f"cells/{c['cell_id']}/attempt-0001/complete.json" for c in expected_order}
    assert set(freeze['cells']) == expected_complete
    assert {n for n in names if n.startswith('cells/') and n.endswith('/complete.json')} == expected_complete
    assert not any('attempt-0002/' in n or n.endswith('/interrupted.json') or n.endswith('/unaccepted-complete.json') for n in names)
    saved_cells = {r['cell_id']: r for r in evaluation['cells']}
    assert set(saved_cells) == {c['cell_id'] for c in expected_order} and len(evaluation['cells']) == 60
    all_pairs = defaultdict(set)
    for cell in expected_order:
        cid, condition = cell['cell_id'], cell['condition']
        folder = f'cells/{cid}/attempt-0001'
        completed = get(folder + '/complete.json')
        unseal(completed)
        assert sha(z.read(folder + '/complete.json')) == freeze['cells'][folder + '/complete.json']
        assert set(completed['files']) == {'plan.json', 'traces.jsonl', 'occupancy.jsonl', 'resources.jsonl'}
        blobs = {name: z.read(folder + '/' + name) for name in completed['files']}
        assert all(sha(blobs[n]) == digest for n, digest in completed['files'].items())
        plan = json.loads(blobs['plan.json'])
        assert identity(plan) == completed['plan_hash']
        for k, v in {**cell, 'settings_hash': identity(settings), 'protocol': p,
                     'schedule_sha256': settings['schedules'][cell['schedule_file']], 'workload_file': 'workload.json',
                     'workload_sha256': settings['workload_sha256']}.items():
            assert plan[k] == v, (cid, k)
        service = get(plan['service_record_file'])
        assert sha(z.read(plan['service_record_file'])) == plan['service_record_sha256']
        assert service['condition'] == condition and service['repeat'] == cell['repeat']
        assert service['allocation'] == plan['allocation']
        assert service['runtime_binding_hash'] == sha(z.read('runtime/binding.json'))
        finished = get(str(Path(plan['service_record_file']).parent / 'finished.json'))
        assert datetime.fromisoformat(finished['at']) <= datetime.fromisoformat(access['access_at'])
        warmed = get(plan['warmup_record_file'])
        assert sha(z.read(plan['warmup_record_file'])) == plan['warmup_record_sha256']
        assert [r['sample_id'] for r in warmed['rows']] == [r['sample_id'] for r in warmup]
        assert len(warmed['rows']) == 64 and all(r['status'] == 'ok' for r in warmed['rows'])
        rows = [json.loads(line) for line in blobs['traces.jsonl'].splitlines()]
        assert len(rows) == len(schedules[cell['schedule_file']])
        rows.sort(key=lambda r: r['offer_id'])
        assert [{k: r[k] for k in ('offer_id', 'sample_id', 'arrival_seconds')} for r in rows] == schedules[cell['schedule_file']]
        counters = Counter(); latencies=[]; http=[]; clientq=[]; worker=[]; dispatch=[]; stamps=Counter()
        correct = timely_correct = timely = success = admitted = 0
        for row in rows:
            counters[row['status']] += 1
            assert row['dispatch_seconds'] >= row['arrival_seconds'] - 1e-7
            dispatch.append(1000 * (row['dispatch_seconds'] - row['arrival_seconds']))
            if row['rejected']:
                assert row['status'] == 'client_rejected' and row['result'] is None and row['reason'] == 'bounded_client_inflight'
                continue
            admitted += 1
            assert row['completion_seconds'] >= row['worker_start_seconds'] >= row['dispatch_seconds']
            total_ms = 1000 * (row['completion_seconds'] - row['arrival_seconds'])
            queue_ms = 1000 * (row['worker_start_seconds'] - row['dispatch_seconds'])
            work_ms = 1000 * (row['completion_seconds'] - row['worker_start_seconds'])
            for field, value in [('arrival_to_completion_ms', total_ms), ('client_queue_ms', queue_ms), ('worker_ms', work_ms)]:
                assert_equal(value, row[field], f'{cid}/{field}')
            latencies.append(total_ms); clientq.append(queue_ms); worker.append(work_ms)
            stamps[row['worker_start_seconds']] += 1; stamps[row['completion_seconds']] += 1
            result = row['result']
            assert result['status'] == row['status']
            if row['status'] != 'ok':
                errors.append({'cell_id': cid, 'offer_id': row['offer_id'], 'status': row['status'], 'error': result.get('error')})
                continue
            pred = parse_saved_response(condition, row, service, f'{cid}-{row["offer_id"]}')
            assert 0 <= result['latency_ms'] <= work_ms + 1e-5
            http.append(result['latency_ms']); success += 1
            timely += total_ms <= 500
            right = pred == gold[row['sample_id']]
            correct += right; timely_correct += right and total_ms <= 500
            prediction_counts[condition][row['sample_id']][pred] += 1
            per_rate_predictions[(condition, cell['rate'])][row['sample_id']][pred] += 1
        assert admitted == success, (cid, errors[-1:] if errors else [])
        metrics = {'offered': len(rows), 'admitted': admitted, 'successful': success,
                   'failed_admitted': admitted-success, 'rejected': len(rows)-admitted, 'status_counts': dict(counters),
                   'duration_seconds': 180, 'offered_rate_actual': len(rows)/180, 'met_slo': timely,
                   'slo_fraction_all_offered': timely/len(rows), 'goodput_per_second': timely/180,
                   'successful_per_offered_window_second': success/180,
                   'latency_ms_all_admitted': {f'p{int(qv*100)}': q(latencies,qv) for qv in (.5,.95,.99)},
                   'max_dispatch_lag_ms': max(dispatch)}
        measured_report = completed['report']
        for key, value in metrics.items():
            assert_equal(value, measured_report[key], f'{cid}/report/{key}')
        metrics.update(correct=correct, correct_met_slo=timely_correct,
                       correct_goodput_per_second=timely_correct/180,
                       correct_fraction_all_offered=correct/len(rows), accuracy_successful=correct/success,
                       note='Repeated requests describe this offered workload, not independent quality samples')
        expected_eval = {**cell, **metrics, 'allocation': plan['allocation'], 'attempt': folder}
        assert_equal(expected_eval, saved_cells[cid], f'{cid}/evaluation')
        events = [json.loads(x) for x in blobs['occupancy.jsonl'].splitlines()]
        assert len(events) == 2*admitted and Counter(e['at_seconds'] for e in events) == stamps
        area = previous_time = level = 0
        for event in events:
            assert event['at_seconds'] >= previous_time
            assert abs(event['active_workers']-level) == 1 and 0 <= event['active_workers'] <= 16
            area += (event['at_seconds']-previous_time)*level
            previous_time, level = event['at_seconds'], event['active_workers']
        assert level == 0 and measured_report['drained_elapsed_seconds'] >= max(180, previous_time)
        assert measured_report['drained_elapsed_seconds'] <= 245
        work_seconds = sum(worker)/1000
        assert_equal(area, measured_report['worker_occupancy_seconds'], f'{cid}/occupancy')
        assert_equal(work_seconds, measured_report['sum_worker_seconds'], f'{cid}/worker sum')
        assert_equal(abs(area-work_seconds), measured_report['occupancy_accounting_error_seconds'], f'{cid}/occupancy discrepancy')
        assert_equal(success/measured_report['drained_elapsed_seconds'], measured_report['successful_per_drained_second'], f'{cid}/drained throughput')
        resource_rows = len(blobs['resources.jsonl'].splitlines())
        assert resource_rows > 0
        allocation = plan['allocation']['slurm_job_id']
        all_pairs[f'r{cell["repeat"]}-rate{cell["rate"]}'].add(identity(plan['allocation']))
        flat = {**cell, 'allocation': allocation, 'hostname': plan['allocation']['hostname'],
                'started_at': measured_report['started_at'], 'finished_at': finished['at'],
                **{k:v for k,v in metrics.items() if k not in ('note','latency_ms_all_admitted')},
                **{f'latency_{k}_ms':v for k,v in metrics['latency_ms_all_admitted'].items()},
                'http_p50_ms': q(http,.5), 'http_p95_ms': q(http,.95), 'http_mean_ms': statistics.fmean(http),
                'client_queue_p50_ms': q(clientq,.5), 'client_queue_p95_ms': q(clientq,.95),
                'client_queue_mean_ms': statistics.fmean(clientq), 'dispatch_lag_p95_ms': q(dispatch,.95),
                'dispatch_lag_mean_ms': statistics.fmean(dispatch), 'worker_mean_ms': statistics.fmean(worker),
                'worker_occupancy_seconds': area, 'mean_active_workers': area/measured_report['drained_elapsed_seconds'],
                'max_active_workers': max(e['active_workers'] for e in events),
                'drained_elapsed_seconds': measured_report['drained_elapsed_seconds'],
                'resource_rows': resource_rows, 'startup_seconds': service['startup_seconds']}
        cells.append(flat); raw_metrics.append(metrics); allocation_groups[allocation].append(flat)
        print(json.dumps({'cell': cid, 'correct_goodput': timely_correct/180, 'statuses': dict(counters)}), flush=True)
    assert evaluation['pairing_same_allocation'] == {k: len(v)==1 for k,v in all_pairs.items()}
    saved_summaries = {(s['condition'],s['rate']):s for s in evaluation['summaries']}
    assert len(saved_summaries) == len(evaluation['summaries']) == 12
    aggregates = []
    for condition in CONDITIONS:
        for rate in (5,25,100):
            selected = [r for r in cells if r['condition']==condition and r['rate']==rate]
            assert len(selected)==5
            s = saved_summaries[condition,rate]
            assert s['repeats'] == 5
            for key, saved in s['metrics'].items():
                values = [r[key] for r in selected]
                assert_equal({'median':statistics.median(values),'min':min(values),'max':max(values),'values':values}, saved, f'{condition}/{rate}/{key}')
            keys = ['goodput_per_second','correct_goodput_per_second','slo_fraction_all_offered',
                    'correct_fraction_all_offered','accuracy_successful','successful_per_offered_window_second',
                    'latency_p50_ms','latency_p95_ms','latency_p99_ms','http_p50_ms','http_p95_ms','http_mean_ms',
                    'client_queue_p50_ms','client_queue_p95_ms','client_queue_mean_ms','worker_mean_ms',
                    'dispatch_lag_mean_ms','dispatch_lag_p95_ms','mean_active_workers','startup_seconds']
            aggregate = {'condition':condition,'rate':rate,'repeats':5,
                         'allocation_counts':dict(Counter(r['allocation'] for r in selected)),
                         'totals':{key:sum(r[key] for r in selected) for key in
                                   ['offered','admitted','successful','failed_admitted','rejected','met_slo','correct','correct_met_slo']},
                         'metrics':{key:stats([r[key] for r in selected]) for key in keys}}
            aggregates.append(aggregate)

prediction_audit = {}
for condition, cases in prediction_counts.items():
    assert set(cases) == set(old_labels[condition])
    varied = {sid: dict(labels) for sid,labels in cases.items() if len(labels)>1}
    disagree = {sid: sum(n for pred,n in labels.items() if pred != old_labels[condition][sid]) for sid,labels in cases.items()}
    prediction_audit[condition] = {
        'unique_inputs':len(cases), 'unique_inputs_with_varying_successful_labels':len(varied),
        'varying_inputs':varied, 'responses_differing_from_historical_label':sum(disagree.values()),
        'inputs_with_any_historical_disagreement':sum(n>0 for n in disagree.values()),
        'historical_unique_correct':sum(pred==gold[sid] for sid,pred in old_labels[condition].items()),
        'unique_inputs_always_correct':sum(set(labels)=={gold[sid]} for sid,labels in cases.items()),
        'unique_inputs_ever_correct':sum(gold[sid] in labels for sid,labels in cases.items()),
        'successful_response_count':sum(sum(labels.values()) for labels in cases.values())}
    if condition.startswith('minilm'):
        assert not varied and sum(disagree.values()) == 0
allocation_summaries=[]
for allocation, allocated in allocation_groups.items():
    for condition in CONDITIONS:
        for rate in (5,25,100):
            selected=[r for r in allocated if r['condition']==condition and r['rate']==rate]
            if not selected: continue
            allocation_summaries.append({'allocation':allocation,'condition':condition,'rate':rate,
                'repeats':[r['repeat'] for r in selected],'count':len(selected),
                **{k:stats([r[k] for r in selected]) for k in
                   ['goodput_per_second','correct_goodput_per_second','slo_fraction_all_offered','latency_p95_ms','client_queue_mean_ms','http_mean_ms']}})
aggregate_flat=[]
for agg in aggregates:
    row={k:v for k,v in agg.items() if k not in ('metrics','totals')}
    row.update({f'{k}_total':v for k,v in agg['totals'].items()})
    for key,value in agg['metrics'].items():
        row.update({f'{key}_{stat}':v for stat,v in value.items() if stat!='values'})
    aggregate_flat.append(row)
receipt = {'status':'PASS','archive':str(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
           'archive_sha256':sha(ARCHIVE.read_bytes()),'source_commit':COMMIT,
           'source_files_verified':len(settings['source']['files']), 'completed_cells':len(cells),
           'formal_offers':sum(r['offered'] for r in cells),'gold_sha256':GOLD_SHA,
           'baseline_archive_sha256':baseline['sha256'],'original_v1_archive_sha256':original['sha256'],
           'chain_verified':True,'all_saved_metrics_recomputed':True,'one_attempt_each':True,
           'schedule_reproduced':15,'raw_successful_responses_reparsed':sum(r['successful'] for r in cells),
           'occupancy_time_and_event_accounting_verified':True,'errors':errors,
           'pairing_same_allocation':evaluation['pairing_same_allocation'],
           'allocation_cells':{k:len(v) for k,v in allocation_groups.items()},
           'latency_summary_note':'Each aggregate quantile is a summary of five per-cell quantiles, never a pooled request quantile.',
           'quality_note':'Repeated successful requests are not independent quality observations; CPU labels compared to accepted historical predictions.',
           'limits':['Archive integrity is not a trusted execution attestation.',
                     'New result evaluation/collection functions never called; no models run.',
                     'Resource payload bytes checked here; detailed resource interpretation belongs to separate audit.']}
write_json('metrics-audit.json',receipt)
write_json('cells.json',cells);write_csv('cells.csv',cells)
write_json('aggregates.json',aggregates);write_csv('aggregates.csv',aggregate_flat)
write_json('allocation-sensitivity.json',allocation_summaries)
write_json('prediction-consistency.json',prediction_audit)
write_json('per-input-prediction-counts.json',{m:{sid:dict(labels) for sid,labels in cases.items()} for m,cases in prediction_counts.items()})
print(json.dumps(receipt,indent=2))
