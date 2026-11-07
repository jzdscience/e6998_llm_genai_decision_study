#!/usr/bin/env python3
"""Independent read-only source, runtime, lifecycle and resource audit."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import zipfile

def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hashed(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return hashed(path.read_bytes())


def fingerprint(value):
    return hashed(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())


def dt(text):
    return datetime.fromisoformat(text)


def summarize(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values),
            'mean': statistics.mean(values)} if values else None


def local_log(root, remote):
    if '/runtime-control/runtime/logs/' in remote:
        return root/'runtime-control/runtime/logs'/Path(remote).name
    return root/'runtime/logs'/Path(remote).name


def audit(repo_root, evidence_root, result_root, out):
    REPO, HIST, ROOT, OUT = repo_root, evidence_root, result_root, out
    config, binding = read(ROOT/'settings.json'), read(ROOT/'runtime/binding.json')
    p, source = config['protocol'], config['source']
    assert source['commit'] == '8b5370d8aea7ed57dc42c92f4fe07bcf023a8e07'
    checked_source = {}
    for name, expected in source['files'].items():
        data = subprocess.check_output(['git', 'show', source['commit']+':'+name], cwd=REPO)
        assert hashed(data) == expected, name
        checked_source[name] = expected
    assert read(REPO/'configs/service_study_protocol.json') == p
    checked_inputs = {}
    for name, expected in {'workload.json': config['workload_sha256'],
                           'warmup.json': config['warmup_sha256'], **config['schedules']}.items():
        assert sha(ROOT/name) == expected, name
        checked_inputs[name] = expected
    controls = ROOT/'runtime-control/runtime'
    for name, expected in binding['controls'].items():
        assert fingerprint(read(controls/name)) == expected, name
    installed, lock = read(controls/'installations.json'), read(controls/'lock.json')
    assert read(controls/'selected_memory.json') == {'qwen': .4, 'startlux': .4}
    assert read(controls/'selected_profiles.json') == {'qwen': 'primary', 'startlux': 'primary'}
    original_runtime = HIST/'results/returned-benchmark-2026-11-02-130735/runtime'
    for name, expected in binding['source_files'].items():
        assert sha(original_runtime/name) == expected, name
    original_installed = read(original_runtime/'installations.json')
    assert installed['environments']['vllm-0.31.0'] == original_installed['environments']['vllm-0.31.0']
    assert installed['models'] == {m: original_installed['models'][m] for m in ('qwen', 'startlux')}
    assert lock == read(original_runtime/'lock.json')
    original_zip = HIST/'records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip'
    with zipfile.ZipFile(original_zip) as zipped:
        for name, expected in config['inputs'].items():
            assert hashed(zipped.read(name)) == expected, name
        original_rows = [json.loads(line) for line in zipped.read('data/examples.jsonl').splitlines()]
    workload, warmup = read(ROOT/'workload.json'), read(ROOT/'warmup.json')
    assert workload == sorted([x for x in original_rows if x['task'] == 'clinc' and x['split'] == 'test'], key=lambda x:x['sample_id'])
    assert warmup == sorted([x for x in original_rows if x['task'] == 'clinc' and x['split'] == 'development'], key=lambda x:x['sample_id'])
    assert len(workload) == 600 and len(warmup) == 64
    all_ids = [x['sample_id'] for x in workload+warmup]
    order = []
    for repeat in range(5):
        conditions = p['conditions'].copy()
        random.Random(p['seed']+repeat).shuffle(conditions)
        for condition in conditions:
            rates = p['rates_per_second'].copy()
            random.Random(f"{p['seed']}-{repeat}-{condition}").shuffle(rates)
            for rate in rates:
                order.append({'cell_id':f'r{repeat}-{condition}-rate{rate}', 'repeat':repeat,
                              'condition':condition, 'rate':rate, 'schedule_file':f'schedules/r{repeat}-rate{rate}.json'})
    assert config['order'] == order
    for repeat in range(5):
        for rate in p['rates_per_second']:
            seed = p['seed']+repeat*1000+rate
            arrivals, samples = random.Random(seed), random.Random(seed+1_000_000)
            expected, cycle, elapsed = [], [], 0.0
            while True:
                elapsed += arrivals.expovariate(rate)
                if elapsed >= p['duration_seconds']:
                    break
                if not cycle:
                    cycle = [x['sample_id'] for x in workload]
                    samples.shuffle(cycle)
                expected.append({'offer_id':len(expected), 'sample_id':cycle.pop(), 'arrival_seconds':elapsed})
            assert read(ROOT/f'schedules/r{repeat}-rate{rate}.json') == expected
    cheap_zip = HIST/'records/bundles/cheap-baselines-20261105-073350-7f00178c7bd1.zip'
    with zipfile.ZipFile(cheap_zip) as zipped:
        cheap_freeze = json.loads(zipped.read('freeze.json'))
        encoder = json.loads(zipped.read('embeddings.json'))['model_identity']
        for condition, saved in config['baseline'].items():
            assert saved['classifier_sha256'] == hashed(zipped.read(condition+'.model.json'))
            assert saved['protocol_sha256'] == hashed(zipped.read('protocol.json'))
            for name, expected in saved['baseline_receipts'].items():
                assert hashed(zipped.read(name)) == expected
    inventory_text = (controls/'logs/environment-check.log').read_text()
    runs = re.split(r'\nCOMMAND [^\n]+\n', inventory_text)
    assert not runs[0].strip()
    expected_inventory = installed['environments']['vllm-0.31.0']['freeze_sha256']
    inventory_counts = Counter(hashed(text.encode()) for text in runs[1:])
    assert set(inventory_counts) == {expected_inventory}
    sessions = [read(x) for x in (ROOT/'runtime/session-history').glob('*.json')]+[read(ROOT/'runtime/session.json')]
    for path in (ROOT/'runtime/session-history').glob('*.json'):
        assert path.stem == fingerprint(read(path))
    assert read(controls/'session.json') == read(ROOT/'runtime/session.json')
    latest_allowed = defaultdict(list)
    for session in sessions:
        latest_allowed[fingerprint(session['allocation'])].append(session)
    cuda_reports = [read(x) for x in controls.glob('cuda-toolchain-*.json')]
    for report in cuda_reports:
        assert report['status'] == 'complete' and report['sampling']['status'] == 'complete'
        assert report['sampling']['observed'] == report['sampling']['expected'] == [3, 7]
        assert report['sampling']['cuda_synchronized'] is True
    assert len(cuda_reports) == 30

    cells, instances, service_dirs = [], [], []
    allocation_groups, pairings = defaultdict(list), defaultdict(set)
    runtime_configs, token_identities = defaultdict(set), defaultdict(set)
    logs_report, all_warnings = [], []
    for cell in order:
        cid, condition = cell['cell_id'], cell['condition']
        attempts = sorted((ROOT/'cells'/cid).glob('attempt-*'))
        assert len(attempts) == 1 and attempts[0].name == 'attempt-0001', cid
        attempt_dir = attempts[0]
        assert not (attempt_dir/'interrupted.json').exists() and not (attempt_dir/'unaccepted-complete.json').exists()
        attempt, plan, completed = [read(attempt_dir/n) for n in ('attempt.json', 'plan.json', 'complete.json')]
        assert attempt['explicit_resume_acknowledgment'] is False and attempt['prior_interrupted_attempts'] == []
        assert completed['self_hash'] == fingerprint({k:v for k,v in completed.items() if k!='self_hash'})
        assert completed['plan_hash'] == fingerprint(plan)
        for name, digest in completed['files'].items():
            assert sha(attempt_dir/name) == digest
        assert plan['settings_hash'] == fingerprint(config)
        assert all(plan[k] == v for k,v in cell.items())
        assert plan['protocol'] == p and plan['workload_sha256'] == sha(ROOT/'workload.json')
        assert plan['schedule_sha256'] == config['schedules'][plan['schedule_file']]
        assert sha(ROOT/plan['service_record_file']) == plan['service_record_sha256']
        assert sha(ROOT/plan['warmup_record_file']) == plan['warmup_record_sha256']
        start_path = ROOT/plan['service_record_file']
        start, warm = read(start_path), read(ROOT/plan['warmup_record_file'])
        finish = read(start_path.parent/'finished.json')
        service_dirs.append(str(start_path.parent.relative_to(ROOT)))
        assert start['condition'] == condition and start['repeat'] == cell['repeat']
        assert start['allocation'] == plan['allocation']
        assert start['runtime_binding_hash'] == sha(ROOT/'runtime/binding.json')
        assert [x['sample_id'] for x in warm['rows']] == [x['sample_id'] for x in warmup]
        assert all(x['status'] == 'ok' and (x['attempts'] == 1 if condition.startswith('minilm')
                   else len(x['attempts']) == 1 and x['attempts'][0]['attempt'] == 1) for x in warm['rows'])
        began, attempt_at, measure_at, ended = map(dt, (start['started_at'], attempt['started_at'], completed['report']['started_at'], finish['at']))
        assert began < attempt_at <= measure_at < ended
        assert ended >= measure_at+timedelta(seconds=completed['report']['drained_elapsed_seconds'])
        allocation_key = fingerprint(start['allocation'])
        matching_sessions = latest_allowed[allocation_key]
        assert any(dt(s['started_at']) <= began and ended.timestamp() <= s['deadline_epoch'] for s in matching_sessions)
        allocation_groups[allocation_key].append(cid)
        pairings[f"r{cell['repeat']}-rate{cell['rate']}"] .add(allocation_key)
        hw, service = start['hardware'], start['service']
        assert hw['hostname'] == start['allocation']['hostname']
        assert hw['slurm']['SLURM_JOB_ID'] == start['allocation']['slurm_job_id']
        assert plan['service_pids'] == service['pids']
        assert plan['endpoint'] == service['endpoint']
        if condition.startswith('minilm'):
            health = service['health']
            assert health['status'] == 'ok' and health['service'] == 'frozen_minilm_cpu'
            assert health['pid'] == service['pids'][0]
            assert health['identity'] == {**config['baseline'][condition], 'encoder': encoder}
            assert health['environment']['packages'] == cheap_freeze['environment']['packages']
            for k in ('workers', 'batch_size', 'numeric_threads'):
                assert health[k] == 1
            assert health['queue_capacity'] == 64 and health['maximum_admitted'] == 65 and health['maximum_http_handlers'] == 72
            assert health['prediction_cache'] is False
            assert all(v == ('false' if k == 'TOKENIZERS_PARALLELISM' else '1') for k,v in health['environment']['thread_environment'].items())
            assert all(x['num_threads'] == 1 for x in health['environment']['threadpools'])
            assert health['resources']['affinity_enforced'] is False
            assert dt(health['started_at_utc']) <= began
            assert service['command'][1:3] == ['-m', 'decision_study.service_study.cpu']
            assert service['command'][0] == config['cpu_python']
            instances.append(health['instance_id'])
            logged = [json.loads(line) for line in local_log(ROOT, service['log']).read_text().splitlines() if line.strip()]
            assert len(logged) == 1 and logged[0]['ready'] is True
            assert logged[0]['health']['instance_id'] == health['instance_id']
        else:
            model = {'startlux4b':'startlux', 'qwen4b':'qwen'}[condition]
            runtime, metadata, tokens = service['runtime'], service['services'], service['tokens']
            assert runtime['model_revision'] == lock['models'][model]['revision']
            for k,v in {'precision':'bfloat16', 'prefix_cache':False, 'truncate':False, 'gpu_memory_utilization':.4,
                        'max_model_len':8192, 'chunked_prefill':False, 'dependency_freeze_sha256':expected_inventory}.items():
                assert runtime[k] == v, (cid,k)
            assert runtime['library_versions']['vllm'] == '0.31.0'
            assert metadata['models'][model]['config'] == runtime
            health = metadata['models'][model]['health']
            assert health['status'] == 'ok'
            if model == 'startlux':
                assert health['prompt_attention'] == 'bidirectional' and health['defaults']['think_mode'] == 'never'
                assert runtime['native_temperature'] == 1.1103 and runtime['allow_none'] is False and runtime['none_check'] is False
                if health['engine']['prefix_cache'] is not False:
                    all_warnings.append({'cell':cid, 'finding':'StartLux health says prefix_cache true; runtime and actual engine command/log disable it'})
            else:
                assert runtime['thinking'] is False and runtime['constrained_json'] is True and runtime['max_output_tokens'] == 64
            assert dt(metadata['started_at']) <= dt(metadata['ready_at']) <= began
            assert tokens['effective_config_hash'] == fingerprint(runtime)
            assert tokens['status'] == 'ok' and tokens['truncated'] is False and tokens['overlong'] == []
            assert tokens['n_examples'] == 664 and tokens['model_revision'] == runtime['model_revision']
            assert [x['sample_id'] for x in tokens['rows']] == all_ids
            assert all(x['input_tokens']+x['reserved_output_tokens'] <= 8192 for x in tokens['rows'])
            assert [x['pid'] for x in metadata['processes']] == service['pids']
            runtime_configs[model].add(fingerprint(runtime))
            token_identities[model].add(fingerprint(tokens))
            for process in metadata['processes']:
                command = process['command']
                log_path = local_log(ROOT, process['log'])
                log_lines = log_path.read_text().splitlines()
                if process['name'].endswith('-engine'):
                    assert '--no-enable-prefix-caching' in command and '--no-enable-chunked-prefill' in command
                    assert command[command.index('--gpu-memory-utilization')+1] == '0.4'
                    assert command[command.index('--model')+1] == installed['models'][model]['model_path']
                    assert any("'enable_prefix_caching': False" in line and "'enable_chunked_prefill': False" in line for line in log_lines)
                    shutdown = [i for i,line in enumerate(log_lines) if '[shutdown]' in line or 'Shutting down' in line]
                    errors = [i for i,line in enumerate(log_lines) if re.search(r'\bERROR\b|CUDA out of memory|OutOfMemoryError',line)]
                    pre_shutdown = [i for i in errors if not shutdown or i < min(shutdown)]
                    logs_report.append({'cell':cid,'log':str(log_path.relative_to(ROOT)), 'error_lines':len(errors),
                                        'pre_shutdown_error_lines':[{ 'line':i+1,'text':log_lines[i]} for i in pre_shutdown],
                                        'shutdown_seen':bool(shutdown), 'application_shutdown_complete':any('Application shutdown complete' in x for x in log_lines)})
                else:
                    assert '--no-none-check' in command and command[command.index('--think-mode')+1] == 'never'

        resources = lines(attempt_dir/'resources.jsonl')
        assert len(resources) > 1
        assert [row['sample_index'] for row in resources] == list(range(len(resources)))
        assert all(a['elapsed_seconds'] <= b['elapsed_seconds'] for a,b in zip(resources,resources[1:]))
        gpu_rows, observed_cpu, rss_all, rss_service = [], {}, [], []
        unavailable, inventory_unreadable, gpu_unavailable, inventory_error = 0, 0, 0, 0
        service_identities = set()
        for sample in resources:
            unavailable += len(sample['unavailable_roots'])
            inventory_unreadable += sample['proc_inventory_unreadable_count']
            inventory_error += bool(sample['proc_inventory_error'])
            rows = sample['processes']
            selection = set(service['pids'])
            selection.update(x['pid'] for x in rows if (x['pid'],x['start_time_ticks']) in service_identities)
            while True:
                more = {x['pid'] for x in rows if x['ppid'] in selection}-selection
                if not more: break
                selection.update(more)
            service_identities.update((x['pid'],x['start_time_ticks']) for x in rows if x['pid'] in selection)
            rss_all.append(sum(x['rss_bytes'] for x in rows))
            rss_service.append(sum(x['rss_bytes'] for x in rows if x['pid'] in selection))
            for row in rows:
                key = (row['pid'],row['start_time_ticks'])
                cpu = (row['user_cpu_ticks']+row['system_cpu_ticks'])/sample['clock_ticks_per_second']
                if key not in observed_cpu: observed_cpu[key] = [cpu,cpu,row['pid'] in selection]
                else:
                    assert cpu >= observed_cpu[key][1]
                    observed_cpu[key][1] = cpu
            if sample['gpu']['status'] == 'ok':
                assert len(sample['gpu']['gpus']) == 1
                gpu = sample['gpu']['gpus'][0]
                assert gpu['name'] == 'NVIDIA L40S' and gpu['total_mib'] == 46068.0
                gpu_rows.append(gpu)
            else: gpu_unavailable += 1
        uuids = sorted({x['uuid'] for x in gpu_rows})
        assert len(uuids) == 1
        observed_seconds = resources[-1]['elapsed_seconds']-resources[0]['elapsed_seconds']
        cpu_all = sum(v[1]-v[0] for v in observed_cpu.values())
        cpu_service = sum(v[1]-v[0] for v in observed_cpu.values() if v[2])
        cells.append({**cell, 'allocation':start['allocation'], 'gpu_uuid':uuids[0],
                      'cpu_models':hw['cpu_models'], 'cpus_allowed_list':hw['cpus_allowed_list'],
                      'slurm_mem_per_node':hw['slurm']['SLURM_MEM_PER_NODE'], 'started_at':start['started_at'],
                      'attempt_started_at':attempt['started_at'], 'measurement_started_at':completed['report']['started_at'],
                      'finished_at':finish['at'], 'startup_seconds':start['startup_seconds'],
                      'resource_samples':len(resources), 'observed_resource_span_seconds':observed_seconds,
                      'owned_observed_cpu_seconds':cpu_all, 'service_observed_cpu_seconds':cpu_service,
                      'service_observed_cpu_core_equivalents':cpu_service/observed_seconds,
                      'peak_owned_summed_rss_bytes':max(rss_all), 'peak_service_summed_rss_bytes':max(rss_service),
                      'gpu_utilization_sample_percent':summarize([x['utilization_percent'] for x in gpu_rows]),
                      'gpu_memory_sample_mib':summarize([x['used_mib'] for x in gpu_rows]),
                      'gpu_power_sample_watts':summarize([x['power_watts'] for x in gpu_rows]),
                      'gpu_unavailable_samples':gpu_unavailable, 'unavailable_root_observations':unavailable,
                      'proc_inventory_unreadable_count_sum':inventory_unreadable, 'proc_inventory_error_samples':inventory_error})
    assert len(instances) == len(set(instances)) == 30
    assert len(service_dirs) == len(set(service_dirs)) == len(list((ROOT/'services').iterdir())) == 60
    assert [x['cell_id'] for x in sorted(cells,key=lambda x:x['started_at'])] == [x['cell_id'] for x in order]
    assert all(dt(a['finished_at']) <= dt(b['started_at']) for a,b in zip(cells,cells[1:]))
    access = read(ROOT/'evaluation-access.json')
    assert all(dt(x['finished_at']) < dt(access['access_at']) for x in cells)
    evaluation = read(ROOT/'evaluation.json')
    assert {k:len(v)==1 for k,v in pairings.items()} == evaluation['pairing_same_allocation']
    groups = {}
    for key, members in allocation_groups.items():
        selected = [x for x in cells if x['cell_id'] in members]
        groups[key] = {'allocation':selected[0]['allocation'], 'cells':members, 'count':len(members),
                       'gpu_uuids':sorted({x['gpu_uuid'] for x in selected}),
                       'cpu_models':sorted({model for x in selected for model in x['cpu_models']}),
                       'cpu_affinity_sets':sorted({x['cpus_allowed_list'] for x in selected})}
    resource_groups = {}
    for condition in p['conditions']:
        for rate in p['rates_per_second']:
            selected = [x for x in cells if x['condition']==condition and x['rate']==rate]
            key = f'{condition}-rate{rate}'
            resource_groups[key] = {name:summarize([x[name] for x in selected]) for name in (
                'resource_samples','service_observed_cpu_seconds','service_observed_cpu_core_equivalents',
                'peak_service_summed_rss_bytes','peak_owned_summed_rss_bytes','startup_seconds')}
            for name in ('gpu_utilization_sample_percent','gpu_memory_sample_mib','gpu_power_sample_watts'):
                resource_groups[key][name] = {'per_cell_means':summarize([x[name]['mean'] for x in selected]),
                                              'sample_min':min(x[name]['min'] for x in selected), 'sample_max':max(x[name]['max'] for x in selected)}
    last_error = read(ROOT/'diagnostics/last_error.json')
    audit = {'source_commit':source['commit'], 'source_files_verified':checked_source,
             'settings_inputs_and_schedules_verified':checked_inputs,
             'original_v1_input_hashes_verified':config['inputs'], 'workload_count':len(workload), 'warmup_count':len(warmup),
             'all_15_schedules_and_60_cell_order_independently_regenerated':True,
             'runtime_controls_verified':binding['controls'], 'original_runtime_metadata_verified':binding['source_files'],
             'dependency_freeze_sha256':expected_inventory,'dependency_verification_log_blocks':dict(inventory_counts),
             'cuda_preflights_complete':len(cuda_reports), 'cpu_health_verified':30,'gpu_runtime_and_tokens_verified':30,
             'cpu_unique_service_instances':len(set(instances)), 'single_complete_attempt_each':60,
             'all60_finished_before_first_quality_access':True, 'all60_service_times_within_recorded_allowance':True,
             'all60_service_intervals_in_fixed_order_without_overlap':True,
             'first_start':cells[0]['started_at'], 'final_finished':cells[-1]['finished_at'], 'quality_access_at':access['access_at'],
             'retained_last_error':{'at':last_error['at'],'error':last_error['error'],
                                    'earlier_than_final_completion':dt(last_error['at'])<dt(cells[-1]['finished_at'])},
             'allocation_groups':groups, 'pairing_same_allocation':{k:len(v)==1 for k,v in pairings.items()},
             'runtime_config_hashes':{k:sorted(v) for k,v in runtime_configs.items()},
             'token_report_hashes':{k:sorted(v) for k,v in token_identities.items()},
             'engine_logs':logs_report,'warnings':all_warnings,'resource_groups':resource_groups,'cells':cells,
             'limits':['Artifact hashes and declared source establish consistency, not cryptographic attestation of remote execution.',
                       'One numeric CPU thread is not a dedicated physical core; affinity was not enforced by the service.',
                       'Resource sampling includes the cell client and service descendants; service-only aggregates exclude the cell client.',
                       'CPU deltas cover observed intervals only; RSS sums can double-count shared mappings and are not peak job RAM.',
                       'GPU memory, utilization and power observations cover the whole device and do not establish per-request energy or dollar cost.',
                       'Start/finished receipts and shutdown logs record past lifecycle; they do not establish present node liveness.',
                       'One repeat/rate comparison spans two allocations and must not be called same-hardware paired.']}
    with (OUT/'service-runtime-audit.json').open('x') as handle:
        json.dump(audit,handle,indent=2);handle.write('\n')
    fields = ['cell_id','repeat','condition','rate','hostname','slurm_job_id','gpu_uuid',
              'resource_samples','service_observed_cpu_seconds','service_observed_cpu_core_equivalents',
              'peak_service_summed_rss_mib','gpu_utilization_sample_mean_percent','gpu_memory_sample_peak_mib',
              'startup_seconds','started_at','finished_at']
    with (OUT/'resource-summary.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            row = {key:cell[key] for key in fields if key in cell}
            row.update(cell['allocation'])
            row['peak_service_summed_rss_mib'] = cell['peak_service_summed_rss_bytes']/1048576
            row['gpu_utilization_sample_mean_percent'] = cell['gpu_utilization_sample_percent']['mean']
            row['gpu_memory_sample_peak_mib'] = cell['gpu_memory_sample_mib']['max']
            writer.writerow(row)
    print(json.dumps({k:audit[k] for k in ('source_commit','dependency_verification_log_blocks','cpu_health_verified','gpu_runtime_and_tokens_verified',
        'single_complete_attempt_each','first_start','final_finished','quality_access_at','pairing_same_allocation')},indent=2))
    print(f'Audit outputs: {OUT}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, help='Git source repository containing the recorded commit')
    parser.add_argument('--evidence-root', type=Path, help='Historical root containing records/bundles and results')
    parser.add_argument('--result-root', type=Path, help='Safely restored service result directory')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing paths are refused')
    args = parser.parse_args(argv)
    inferred_repo = next((p for p in Path(__file__).resolve().parents
                          if (p/'pyproject.toml').is_file() and (p/'decision_study').is_dir()), None)
    if args.repo_root is None and inferred_repo is None:
        parser.error('Cannot infer source repository; pass --repo-root')
    repo = (args.repo_root or inferred_repo).expanduser().absolute()
    evidence = (args.evidence_root or repo.parent/'project/solo-experiments-2026-10-30/decision-study').expanduser().absolute()
    result = (args.result_root or evidence/'results/returned-service-complete-2026-11-06').expanduser().absolute()
    for label, path in (('repo-root', repo), ('evidence-root', evidence), ('result-root', result)):
        if not path.is_dir():
            parser.error(f'{label} does not exist: {path}')
    out = args.out.expanduser().absolute()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f'Output already exists and was preserved: {out}')
    audit(repo, evidence, result, out)


if __name__ == '__main__':
    main()
