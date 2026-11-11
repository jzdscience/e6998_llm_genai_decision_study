"""Independent stdlib-only audit of archived upgrade study; never executes bundle code.

Portable copy of the script accepted on 2026-11-10 (see audit/README.md). Only the location of the
inputs and of the output changed; every check is the accepted one.

  python3 -I audit_integrity.py --repo-root REPO --evidence-root ROOT --output DIR
  (--evidence-root defaults to $EVIDENCE_ROOT; --repo-root to the checkout holding this file)

REPO is a git checkout that contains commit 7c81947 and records/BUNDLE_INDEX.json. ROOT is the
external evidence root: the archive paths in the bundle index are relative to it. Writes
DIR/integrity-audit.json and exits non-zero if any check fails.
"""
from pathlib import Path, PurePosixPath
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import zipfile

if not __debug__:
    raise SystemExit('Run without -O: the audit relies on its checks raising.')
_parser = argparse.ArgumentParser(description='Independent integrity audit of the 9B serving-cost bundles.')
_parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
_parser.add_argument('--evidence-root', type=Path, default=os.environ.get('EVIDENCE_ROOT'))
_parser.add_argument('--output', type=Path, required=True, help='new or existing directory for integrity-audit.json')
_args = _parser.parse_args()
if _args.evidence_root is None:
    _parser.error('pass --evidence-root or set EVIDENCE_ROOT')
REPO = _args.repo_root.resolve()
ROOT = _args.evidence_root.resolve()
if not (REPO / 'records/BUNDLE_INDEX.json').is_file():
    _parser.error(f'{REPO} has no records/BUNDLE_INDEX.json; pass --repo-root')
INDEX = json.loads((REPO / 'records/BUNDLE_INDEX.json').read_text())
FORMAL = next(x for x in INDEX['bundles'] if x['id'] == 'upgrade-formal-20261109-180806')
PRE = next(x for x in INDEX['bundles'] if x['id'] == 'upgrade-preflight-r4-20261109-074915')
V3 = next(x for x in INDEX['bundles'] if x['id'] == 'v3-complete-20261103-232529')
CHECKS = []

def sha(data): return hashlib.sha256(data).hexdigest()
def fp(value): return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
def check(name, value, detail=None):
    CHECKS.append({'check': name, 'pass': bool(value), **({'detail': detail} if detail is not None else {})})
    if not value: raise AssertionError(name)
def dt(value): return datetime.datetime.fromisoformat(value)
def close(a, b): return math.isclose(a, b, abs_tol=1e-8)
def percentile(values, q):
    a = sorted(values); i = (len(a)-1)*q; lo = math.floor(i); hi = math.ceil(i)
    return a[lo] + (a[hi]-a[lo])*(i-lo)
def opens(entry):
    path = ROOT / entry['archive']
    check(entry['id']+':size_sha', path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256'])
    z = zipfile.ZipFile(path); names = z.namelist()
    check(entry['id']+':members', len(names) == entry['verified_file_count'] + (1 if entry['bundle_kind']=='v3' else 0))
    check(entry['id']+':safe_unique_crc', len(names) == len(set(names)) and z.testzip() is None and all(
        not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n and
        not stat.S_ISLNK(z.getinfo(n).external_attr >> 16) for n in names))
    return z

z = opens(FORMAL); pre = opens(PRE); v3 = opens(V3)
names = z.namelist()
def j(n): return json.loads(z.read(n))
def jl(n): return [json.loads(line) for line in z.read(n).splitlines() if line.strip()]
settings = j('settings.json'); sh = fp(settings); protocol = settings['protocol']
check('source_commit_revision', settings['source']['commit'] == '7c81947d8ed51b18ace9ba8ae0dad0476c9b6b19' and protocol['revision'] == 4)
check('protocol_source_content', protocol == json.loads(subprocess.check_output(['git','show',settings['source']['commit']+':configs/upgrade_study_protocol.json'], cwd=REPO)))
for n, digest in settings['source']['files'].items():
    check('source:'+n, sha(subprocess.check_output(['git','show',settings['source']['commit']+':'+n], cwd=REPO)) == digest)
for n, digest in settings['file_sha256'].items(): check('prepared:'+n, sha(z.read(n)) == digest)
check('preflight_snapshot_members_all_retained', set(pre.namelist()) <= set(names))
changed_pre = [n for n in pre.namelist() if pre.read(n) != z.read(n)]
allowed_mutable = {'runtime/session.json'} | {'runtime-controls/'+r+'/runtime/'+n for r in protocol['models'] for n in ['doctor.json','last_services.json','logs/environment-check.log','session.json']}
check('preflight_snapshot_changes_only_mutable_runtime', set(changed_pre) <= allowed_mutable, changed_pre)
check('session_did_not_extend', j('runtime/session.json')['deadline_epoch'] <= json.loads(pre.read('runtime/session.json'))['deadline_epoch'])

work = j('workload.json'); warm = j('warmup.json'); inputs = work + warm
v3input = [json.loads(x) for x in v3.read('data/examples.jsonl').splitlines() if x.strip()]
check('v3_examples_sha', sha(v3.read('data/examples.jsonl')) == protocol['workload']['examples_sha256'])
check('workload_warmup_exact_v3', work == sorted([x for x in v3input if x['task']=='anli' and x['split']=='holdout'],key=lambda r:r['sample_id']) and warm == sorted([x for x in v3input if x['task']=='anli' and x['split']=='development'],key=lambda r:r['sample_id']))
check('input_partition', len(work)==3200 and len(warm)==90 and len({r['sample_id'] for r in inputs})==3290)
policy = json.loads(v3.read('policy_freeze.json'))
check('v3_policy_sha', sha(v3.read('policy_freeze.json')) == protocol['cascade_policy']['policy_freeze_sha256'])
check('policy_exact_v3', j('policy.json')['policy']==policy['policies']['qwen9b']['class_conditional'] and j('policy.json')['binding']==settings['policy'])

tokens=j('preflight/tokens.json'); token_sha=sha(z.read('preflight/tokens.json'))
check('tokens_settings_roles', tokens['settings_hash']==sh and set(tokens['reports'])==set(protocol['models']))
token_summary={}
for role,b in tokens['reports'].items():
    raw=j(b['file']); reserve=64 if role=='qwen9b' else 1
    check('token_raw_sha:'+role, sha(z.read(b['file']))==b['sha256'])
    check('token_binding:'+role, all(raw[k]==b[k] for k in ['status','effective_config_hash','model_revision','max_model_len','n_examples']) and raw['status']=='ok' and raw['model_revision']==protocol['models'][role]['revision'] and raw['n_examples']==3290 and raw['renderer']=='official' and not raw['truncated'] and raw['overlong']==[])
    check('token_ids_bounds:'+role, [r['sample_id'] for r in raw['rows']]==[r['sample_id'] for r in inputs] and all(r['input_tokens']>0 and r['reserved_output_tokens']==reserve and r['input_tokens']+reserve<=8192 and len(r['token_ids_sha256'])==64 for r in raw['rows']))
    input_name=next(n for n in names if n.startswith('runtime-controls/'+role+'/runtime/interactive-inputs-') and n.endswith('.jsonl'))
    check('token_input_text:'+role,jl(input_name)==inputs)
    token_summary[role]={'count':len(raw['rows']),'max_input_tokens':max(r['input_tokens'] for r in raw['rows']),'reserved_output_tokens':reserve}

allocation=j('runtime/session.json')['allocation']; role_commands={}; launch_info=[]; runtime_values={}; stage_count=0
check('allocation_fixed',allocation=={'hostname':'sc1ns055is14','slurm_job_id':'1224376'})
def command_value(command, key):
    check('one_flag:'+key+':'+sha(str(command).encode())[:12], command.count(key)==1)
    return command[command.index(key)+1]
def deployment(deploy, condition, label, stop=None):
    check(label+':allocation_roles',deploy['allocation']==allocation and deploy['condition']==condition and [x['role'] for x in deploy['roles']]==protocol['condition_services'][condition])
    check(label+':gpu_initially_empty',deploy['gpu_before_start'][0]['processes']==[])
    groups=[]
    for role in deploy['roles']:
        name=role['role']; expected=protocol['models'][name]; rt=role['runtime']
        check(label+':role_runtime:'+name, role['memory_fraction']==expected['gpu_memory_utilization'] and rt['model_repo']==expected['repo'] and rt['model_revision']==expected['revision'] and rt['gpu_memory_utilization']==expected['gpu_memory_utilization'] and rt['vllm_version']=='0.31.0' and rt['max_model_len']==8192 and rt['prefix_cache'] is False and rt['truncate'] is False and fp(rt)==tokens['reports'][name]['effective_config_hash'])
        runtime_values.setdefault(name,rt)
        check(label+':constant_runtime:'+name,runtime_values[name]==rt)
        engines=[p for p in role['processes'] if p['name'].endswith('-engine')]
        check(label+':one_engine:'+name,len(engines)==1)
        for proc in role['processes']:
            groups.append(proc['pid']); cmd=proc['command']; k=(name,proc['name'])
            role_commands.setdefault(k,cmd)
            check(label+':constant_command:'+proc['name'],role_commands[k]==cmd)
            path=proc['log'].split('/aux-anli-upgrade-service-v1-r2/',1)[1]
            check(label+':log_present:'+proc['name'],path in names)
        cmd=engines[0]['command']; log=z.read(engines[0]['log'].split('/aux-anli-upgrade-service-v1-r2/',1)[1]).decode()
        check(label+':engine_flags:'+name,command_value(cmd,'--max-num-seqs')=='64' and float(command_value(cmd,'--gpu-memory-utilization'))==expected['gpu_memory_utilization'] and command_value(cmd,'--max-model-len')=='8192' and command_value(cmd,'--model').endswith('/'+expected['revision']) and '--no-enable-prefix-caching' in cmd and '--no-enable-chunked-prefill' in cmd)
        check(label+':actual_engine_log_cap64:'+name, "'max_num_seqs': 64" in log)
        kv=[float(x) for x in re.findall(r'Available KV cache memory: ([0-9.]+) GiB',log)]
        capacity=[int(x.replace(',','')) for x in re.findall(r'GPU KV cache size: ([0-9,]+) tokens',log)]
        check(label+':kv_positive:'+name,kv and min(kv)>0 and capacity and min(capacity)>0)
        check(label+':kv_matches_record:'+name,kv==role['engine_log'][engines[0]['name']]['available_kv_cache_gib'] and capacity==role['engine_log'][engines[0]['name']]['kv_cache_tokens'])
        launch_info.append({'scope':label,'condition':condition,'role':name,'kv_gib':kv,'kv_tokens':capacity,'startup_seconds':role['startup_seconds'],'engine_log':engines[0]['log'].split('/aux-anli-upgrade-service-v1-r2/',1)[1]})
    stop=stop or deploy.get('stop_verification')
    check(label+':clean_stop',stop and 'error' not in stop and stop['gpu_compute_processes']==[] and stop['owned_groups']==sorted(groups))
    if condition=='cascade':check(label+':co_resident',len(deploy['gpu_after_start'])==2 and len(deploy['gpu_before_start'][1]['processes'])==1)
    return sorted(groups)

def completion(folder, serial=False):
    global stage_count
    rec=j(folder+'/complete.json'); plan=j(folder+'/plan.json'); stage_count+=1
    check(folder+':completion_self_hash',fp({k:v for k,v in rec.items() if k!='self_hash'})==rec['self_hash'])
    check(folder+':completion_plan_hash',rec['plan_hash']==fp(plan))
    expected={'plan.json','rows.json'} if serial else {'plan.json','traces.jsonl','occupancy.jsonl','resources.jsonl'}
    check(folder+':file_inventory',set(rec['files'])==expected)
    for n,h in rec['files'].items():check(folder+':file_sha:'+n,sha(z.read(folder+'/'+n))==h)
    check(folder+':settings_policy_workload',plan['settings_hash']==sh and plan['policy_sha256']==settings['file_sha256']['policy.json'] and plan['workload_sha256']==settings['file_sha256'][plan['workload_file']])
    if serial:
        rows=j(folder+'/rows.json')['rows']
        check(folder+':serial90',len(rows)==90 and [r['sample_id'] for r in rows]==[r['sample_id'] for r in warm] and all(r['status']=='ok' and all(s['status']=='ok' for s in r['stages']) for r in rows))
        check(folder+':serial_roles',all([s['role'] for s in r['stages']]==protocol['condition_services'][plan['condition']] for r in rows))
    else:
        rows=jl(folder+'/traces.jsonl');schedule=j(plan['schedule_file'])
        check(folder+':schedule_sha',plan['schedule_sha256']==sha(z.read(plan['schedule_file'])))
        check(folder+':schedule_rows',sorted([{k:r[k] for k in ['offer_id','sample_id','arrival_seconds']} for r in rows],key=lambda r:r['offer_id'])==schedule)
    return rec,plan,rows

def overlap(rows):
    events=[];pairwise=0;front=[];back=[]
    for r in rows:
        for s in r['result']['stages']:
            a=r['worker_start_seconds']+s['started_ms']/1000;b=r['worker_start_seconds']+s['ended_ms']/1000
            role=0 if s['role']=='startlux4b' else 1
            (front if role==0 else back).append((a,b,r['offer_id']))
            events.extend([(a,role,1),(b,role,-1)])
    for a,b,i in front:
        for c,d,j in back:
            if i!=j:pairwise+=max(0,min(b,d)-max(a,c))
    count=[0,0];last=0;union=0
    for time,role,delta in sorted(events):
        if min(count)>0:union+=time-last
        count[role]+=delta;last=time
    return pairwise,union

preflight_summary={};receipt_hashes={}
for condition in protocol['conditions']:
    n='preflight/'+condition+'.json'; receipt=j(n);folder=receipt['evidence_dir'];receipt_hashes[condition]=sha(z.read(n))
    check(condition+':preflight_bindings',receipt['status']=='PASS' and receipt['settings_hash']==sh and receipt['token_receipt_sha256']==token_sha and z.read(n)==z.read(folder+'/receipt.json') and receipt['prior_failed_attempts']==[])
    for n,h in receipt['evidence'].items():check(condition+':preflight_file:'+n,sha(z.read(folder+'/'+n))==h)
    srec,splan,srows=completion(folder+'/serial',True);crec,cplan,rows=completion(folder+'/concurrent')
    groups=deployment(receipt['deployment'],condition,'preflight-'+condition)
    check(condition+':preflight_same_services',splan['service_pids']==cplan['service_pids']==groups and splan['endpoints']==cplan['endpoints'])
    check(condition+':preflight_successes',len(rows)==310 and all(r['status']=='ok' for r in rows) and all(v['ok'] and v['status']==200 for v in receipt['health'].values()))
    met=sum(r['arrival_to_completion_ms']<=500 for r in rows)
    check(condition+':preflight_summary',crec['report']==receipt['concurrent']['report'] and crec['report']['met_slo']==met and crec['report']['successful']==310)
    check(condition+':preflight_quantiles',all(close(percentile([r['arrival_to_completion_ms'] for r in rows],q),crec['report']['latency_ms_all_admitted'][k]) for k,q in [('p50',.5),('p95',.95),('p99',.99)]))
    detail={'serial_successful':len(srows),'concurrent_successful':len(rows),'timely_500ms':met,'latency_ms':crec['report']['latency_ms_all_admitted'],'stop_seconds':receipt['deployment']['stop_verification']['seconds'],'gpu_after_concurrent_mib':[x['used_memory_mib'] for x in receipt['gpu_processes_after_concurrent']]}
    if condition=='cascade':
        check('cascade_preflight_forced_two_stages',all(r['result']['deferred'] and r['result']['route']['fallback_called'] and r['result']['route']['reason']=='preflight_forced_deferral' and [s['role'] for s in r['result']['stages']]==['startlux4b','qwen9b'] for r in rows))
        pair,union=overlap(rows)
        check('cascade_pairwise_http_overlap_recomputed',close(pair,receipt['concurrent']['engine_overlap_seconds']) and union>0)
        detail.update(pairwise_http_stage_overlap_seconds=pair,wall_clock_http_stage_overlap_union_seconds=union)
    preflight_summary[condition]=detail

order=settings['order']; expected={(r,c,rate) for r in range(3) for c in protocol['conditions'] for rate in (10,20,40,80)}
check('fixed_36_cell_inventory',len(order)==36 and {(c['repeat'],c['condition'],c['rate']) for c in order}==expected)
check('only_one_attempt_per_cell',set(n.split('/')[1] for n in names if n.startswith('cells/'))=={c['cell_id'] for c in order} and {n.split('/')[2] for n in names if n.startswith('cells/')}=={'attempt-0001'})
check('no_interrupted_rejected_attempt_files',not any(any(k in n for k in ['interrupted','unaccepted','failure.json']) for n in names))
cellinfo=[]; all_uuids=set(); all_gpus=set(); starts=[];stops=[];bound_completes={};resources_count=0
for cell in order:
    cid=cell['cell_id'];folder='cells/'+cid+'/attempt-0001';rec,plan,rows=completion(folder)
    attempt=j(folder+'/attempt.json')
    check(cid+':attempt_not_resumed',not attempt['explicit_resume_acknowledgment'] and attempt['prior_interrupted_attempts']==[])
    check(cid+':fixed_plan',all(plan[k]==v for k,v in cell.items()) and plan['protocol']==protocol and plan['allocation']==allocation and plan['preflight_receipt_sha256']==receipt_hashes and plan['watchdog_seconds']==185)
    for k in ['service','warmup']:check(cid+':'+k+'_sha',sha(z.read(plan[k+'_record_file']))==plan[k+'_record_sha256'])
    start=j(plan['service_record_file']); sfolder=str(PurePosixPath(plan['service_record_file']).parent);stop=j(sfolder+'/finished.json')
    check(cid+':service_identity',start['cell_id']==cid and start['condition']==cell['condition'] and start['allocation']==allocation and stop['cell_id']==cid and stop['attempt']=='attempt-0001' and start['runtime_binding_sha256']==sha(z.read('runtime/binding.json')))
    groups=deployment(start['deployment'],cell['condition'],cid,stop['stop_verification'])
    check(cid+':plan_pids',plan['service_pids']==groups)
    wrec,wplan,wrows=completion(sfolder+'/warmup',True)
    check(cid+':same_warmup_services',wplan['service_pids']==groups and wplan['endpoints']==plan['endpoints'] and wplan['purpose']=='warmup')
    check(cid+':hardware_allocation',start['hardware']['hostname']==allocation['hostname'] and start['hardware']['slurm']['SLURM_JOB_ID']==allocation['slurm_job_id'] and len(start['hardware']['controller_affinity'])==16 and start['hardware']['slurm']['SLURM_MEM_PER_NODE']=='102400')
    resources=jl(folder+'/resources.jsonl');resources_count+=len(resources)
    check(cid+':resource_evidence',len(resources)>0 and all(r['gpu']['status']=='ok' and len(r['gpu']['gpus'])==1 for r in resources))
    for r in resources:
        for gpu in r['gpu']['gpus']:all_uuids.add(gpu['uuid']);all_gpus.add((gpu['name'],gpu['total_mib']))
    begin=dt(rec['report']['started_at']);end=dt(stop['at'])
    check(cid+':chronology',dt(start['deployment']['started_at'])<dt(start['deployment']['ready_at'])<=dt(start['started_at'])<=dt(attempt['started_at'])<=begin<end and begin.timestamp()+max(r.get('completion_seconds',r['dispatch_seconds']) for r in rows)<=end.timestamp()+1)
    starts.append(dt(start['deployment']['started_at']));stops.append(end)
    bound_completes[folder+'/complete.json']=sha(z.read(folder+'/complete.json'))
    cellinfo.append({'cell_id':cid,'allocation':allocation,'deployment_started_at':start['deployment']['started_at'],'measured_started_at':rec['report']['started_at'],'finished_at':stop['at'],'offered':len(rows),'resource_samples':len(resources),'warmup_count':len(wrows),'gpu_peak_mib':max(r['gpu']['gpus'][0]['used_mib'] for r in resources),'stop_seconds':stop['stop_verification']['seconds']})
check('sequential_declared_cell_order',[x['cell_id'] for x in cellinfo]==[x['cell_id'] for x in order] and all(a<b for a,b in zip(stops,starts[1:])))
check('one_physical_gpu',len(all_uuids)==1 and all_gpus=={('NVIDIA L40S',46068.0)})
check('formal_preflight_same_gpu',all({g['uuid'] for r in jl(j('preflight/'+c+'.json')['evidence_dir']+'/concurrent/resources.jsonl') for g in r['gpu']['gpus']}==all_uuids for c in protocol['conditions']))
check('exact_36_service_records',len([n for n in names if n.startswith('services/') and n.endswith('/start.json')])==36 and len([n for n in names if n.startswith('services/') and n.endswith('/finished.json')])==36)
binding=j('runtime/binding.json')
for role,files in binding['control_identity'].items():
    for n,h in files.items():check('runtime_binding:'+role+':'+n,fp(j('runtime-controls/'+role+'/runtime/'+n))==h)
freeze=j('freeze.json');access=j('evaluation-access.json');evaluation=j('evaluation.json')
check('freeze_self_identity',freeze['status']=='FROZEN' and freeze['settings_hash']==sh and freeze['cells']==bound_completes and freeze['self_hash']==fp({k:v for k,v in freeze.items() if k!='self_hash'}))
check('access_binds_frozen_gold',access['freeze_sha256']==sha(z.read('freeze.json')) and access['gold_sha256']==protocol['gold']['sha256'])
check('all_stops_before_recorded_gold_access',max(stops)<dt(access['access_at']) and dt(access['access_at']).timestamp()<j('runtime/session.json')['deadline_epoch'])
check('evaluation_identity',evaluation['status']=='COMPLETE' and evaluation['freeze_sha256']==sha(z.read('freeze.json')) and evaluation['access_sha256']==sha(z.read('evaluation-access.json')) and evaluation['gold_sha256']==protocol['gold']['sha256'] and evaluation['self_hash']==fp({k:v for k,v in evaluation.items() if k!='self_hash'}))
check('evaluation_36_cell_inventory',{r['cell_id'] for r in evaluation['cells']}=={r['cell_id'] for r in order} and len(evaluation['cells'])==36)
summary={}
for condition in protocol['conditions']:
    for role in protocol['condition_services'][condition]:
        launches=[x for x in launch_info if x['condition']==condition and x['role']==role and not x['scope'].startswith('preflight')]
        summary[condition+':'+role]={'count':len(launches),'kv_gib_min':min(x['kv_gib'][0] for x in launches),'kv_gib_max':max(x['kv_gib'][0] for x in launches),'kv_tokens_min':min(x['kv_tokens'][0] for x in launches),'kv_tokens_max':max(x['kv_tokens'][0] for x in launches)}
def bundle_identity(entry):
    """Keep archive identity stable when administrative review metadata changes."""
    keys = ('id', 'archive', 'sha256', 'bytes', 'source_commit', 'protocol_revision',
            'verified_file_count', 'measured_utc', 'gold_access_utc', 'preflight_finished_at_utc')
    return {key: entry[key] for key in keys if key in entry}

out={'status':'PASS','scope':'Independent stdlib identity, runtime and raw preflight evidence audit; no production verifier/evaluate imports, model execution or re-tokenization.', 'formal_bundle':bundle_identity(FORMAL),'preflight_bundle':bundle_identity(PRE),'source_files_checked':len(settings['source']['files']),'prepared_files_checked':len(settings['file_sha256']),'completion_receipts_checked':stage_count,'engine_starts_checked':len(launch_info),'settings_hash':sh,'token_reports':token_summary,'preflight_summary':preflight_summary,'allocation':allocation,'physical_gpu_uuids':sorted(all_uuids),'formal_resource_samples':resources_count,'formal_cell_count':len(cellinfo),'formal_offers':sum(c['offered'] for c in cellinfo),'formal_warmup_requests':sum(c['warmup_count'] for c in cellinfo),'formal_startup_kv_ranges':summary,'cell_inventory':cellinfo,'launches':launch_info,'checks':CHECKS,'limits':['Hashes and recorded access prove an internally consistent artifact chain, not independent OS access surveillance. The input test was already inspected in prior studies.','Raw GPU samples show the same device UUID throughout; they are device-level samples, not exclusive process accounting.','preflight engine_overlap_seconds is the sum of pairwise HTTP-stage interval intersections, with duplicate temporal counting; not wall-clock simultaneous GPU execution or GPU kernel overlap.','Preflight memory is not the fixed formal memory allocation: actual KV differs by startup; formal per-role ranges retained.','Health prefix_cache=true conflicts with actual engine launch no-enable-prefix-caching and effective runtime false.','Per-cell finished.json cleanup receipts are checked but not individually SHA-bound by freeze.json; freeze binds complete->plan->start/warmup/raw evidence. Existing verifier rechecks finished.json separately.','Historical prediction mapping and scored goodput are independently audited by sibling metrics task.']}
_args.output.mkdir(parents=True, exist_ok=True)
target=_args.output/'integrity-audit.json';target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['status','source_files_checked','completion_receipts_checked','engine_starts_checked','formal_resource_samples','formal_offers','formal_startup_kv_ranges','preflight_summary']},indent=2));print('checks',len(CHECKS));print(target)
