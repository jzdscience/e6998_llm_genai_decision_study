#!/usr/bin/env python3
"""Independent archive/identity/runtime/statistical audit of a completed size return.

Does not import the production collector, evaluator or fitting code. Reads original
ZIP bytes and git objects; writes only derived audit files beneath --out.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import unicodedata
import zipfile
import numpy as np

COMMIT = '8b5370d8aea7ed57dc42c92f4fe07bcf023a8e07'
NEW_SHA = 'f427f7762bac745d9316cb02195c09f98ab551794de1da047509be7f9fe7d1ed'
OLD_SHA = 'a8d0dc672d83c3edcb85d005e66fa4b3d000132729f98f0beaab6865dfc57a9e'
LABELS = ['entailment','neutral','contradiction']
REVISION = '18a17f734966b213b2f00fa9c915e8b9af2a9b20'

def check(ok, message):
    if not ok: raise AssertionError(message)

def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def sha(payload): return hashlib.sha256(payload).hexdigest()
def digest(value): return sha(canonical(value))
def load(payload): return json.loads(payload, parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
def dt(value):
    parsed = datetime.fromisoformat(value)
    check(parsed.tzinfo is not None, 'naive timestamp')
    return parsed

def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')

class Bundle:
    def __init__(self, path, expected):
        self.path=path
        self.sha=sha(path.read_bytes());check(self.sha==expected,'original archive SHA differs: '+str(path))
        with zipfile.ZipFile(path) as z:
            infos=z.infolist();names=[i.filename for i in infos]
            check(len(names)==len(set(names)),'duplicate members')
            for i in infos:
                name=i.filename;p=PurePosixPath(name)
                check(not p.is_absolute() and '..' not in p.parts and '\\' not in name and not name.endswith('/'),'unsafe archive member')
                check(str(p)==name and not stat.S_ISLNK(i.external_attr>>16),'unsafe normalized name or symlink')
            check(z.testzip() is None,'ZIP CRC error')
            self.raw={n:z.read(n) for n in names}
        self.manifest=self.j('bundle_manifest.json')
        check(set(self.raw)==set(self.manifest['files'])|{'bundle_manifest.json'},'manifest inventory differs')
        for n,e in self.manifest['files'].items():
            check(len(self.raw[n])==e['bytes'] and sha(self.raw[n])==e['sha256'],'manifest byte/hash differs: '+n)
        self.member_count=len(self.raw)
    def j(self,name):return load(self.raw[name])
    def rows(self,name):return [load(line) for line in self.raw[name].splitlines() if line.strip()]
    def sealed(self,name,key):
        value=self.j(name);check(value[key]==digest({k:v for k,v in value.items() if k!=key}),'self hash '+name);return value


def predictions(bundle, prefix):
    config=bundle.j(prefix+'/config.json');rows=[]
    shards=sorted(n for n in bundle.raw if n.startswith(prefix+'/shard-') and n.endswith('.json'))
    for number,n in enumerate(shards,1):
        check(n.endswith(f'shard-{number:06d}.json'),'noncontiguous shard sequence')
        chunk=bundle.j(n)
        check(chunk['count']==len(chunk['rows']) and chunk['sha256']==digest(chunk['rows']),'shard content mismatch')
        check(chunk['config_hash']==digest(config),'shard configuration mismatch')
        rows.extend(chunk['rows'])
    check(len({r['sample_id'] for r in rows})==len(rows),'duplicate prediction IDs')
    index=bundle.j(prefix+'/index.json')
    check(index=={'model':config['model'],'config_hash':digest(config),'terminal':len(rows),
                 'successful':sum(r['status']=='ok' for r in rows),'failed':sum(r['status']=='error' for r in rows),'shards':len(shards)},'index mismatch')
    return config,rows,len(shards)


def validate_rows(rows, examples, interface, *, config=None, revision=None):
    ex={e['sample_id']:e for e in examples}
    check(len(rows)==len(ex) and {r['sample_id'] for r in rows}==set(ex),'prediction coverage mismatch')
    request_ids=[];retry=0;errors=0
    for r in rows:
        e=ex[r['sample_id']]
        check(r['task']==e['task'] and r['split']==e['split'] and r['input_hash']==e['input_hash'],'input identity mismatch')
        check(r['model']==interface,'interface mismatch')
        if config is not None:check(r['config_hash']==digest(config),'row config differs')
        if revision is not None:check(r['model_revision']==revision,'row checkpoint differs')
        check(isinstance(r['latency_ms'],(int,float)) and math.isfinite(r['latency_ms']) and r['latency_ms']>=0,'latency malformed')
        attempts=r['attempts'];check(1<=len(attempts)<=2 and attempts[-1]['status']==r['status'],'attempt terminal mismatch')
        retry+=len(attempts)-1
        for number,a in enumerate(attempts,1):
            check(a['attempt']==number and isinstance(a['request_id'],str) and bool(a['request_id']),'request attempt identity')
            check(math.isfinite(a['latency_ms']) and a['latency_ms']>=0,'attempt latency malformed')
            request_ids.append(a['request_id'])
        if r['status']=='ok':
            check(r['error'] is None,'successful row has error')
            if interface=='startlux':check(attempts[-1].get('server_request_id')==attempts[-1]['request_id'],'StartLux request ID not echoed')
            raw=r['raw_response']
            if interface=='qwen':
                c=raw['choices'][0];check(c['finish_reason']=='stop','truncated Qwen answer')
                answer=load(c['message']['content'])
                check(set(answer)=={'label'} and answer['label'] in LABELS,'malformed Qwen label')
                check(answer['label']==r['prediction'] and r['probabilities'] is None,'Qwen parser mismatch')
            else:
                answer=raw['answers']['decision'];p=answer['probabilities']
                check(set(p)==set(LABELS) and answer['choice'] in LABELS,'decision labels mismatch')
                check(all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in p.values()),'invalid probability')
                check(abs(math.fsum(p.values())-1)<1e-5,'probability mass')
                check(max(p.values())-p[answer['choice']]<=1e-8,'choice is not probability maximum')
                check(r['prediction']==answer['choice'] and r['probabilities']==p,'saved prediction differs from raw answer')
                check(raw['usage']['output_tokens']==0,'decision model generated output')
            check(r['token_counts']==raw.get('usage',{}),'saved usage differs')
        else:
            check(r['status']=='error' and r['prediction'] is None and r['probabilities'] is None,'unknown terminal error');errors+=1
    check(len(request_ids)==len(set(request_ids)),'duplicate request IDs')
    return {'count':len(rows),'errors':errors,'retries':retry,'attempts':len(request_ids)},request_ids


def decision_request(e):
    return {'state':e['state'],'questions':{'decision':{'type':'choice','instructions':e['instructions'],
            'criteria':{label:e['criteria'][label] for label in e['labels']}}},
            'think_mode':'never','none_check':False,'allow_none':False}


def direct(examples, by_id, gold, model):
    values=[r['status']=='ok' and r['prediction']==gold[e['sample_id']] for e in examples for r in [by_id[e['sample_id']]]]
    return {'model':model,'n':len(examples),'correct':sum(values),'accuracy':sum(values)/len(values),
            'failures':sum(by_id[e['sample_id']]['status']!='ok' for e in examples)},values


def bootstrap(examples,left,right):
    groups=defaultdict(list)
    for i,e in enumerate(examples):groups[e['group_id']].append(i)
    keys=sorted(groups);diff=np.array(left,dtype=np.int64)-np.array(right,dtype=np.int64)
    totals=np.array([sum(diff[i] for i in groups[g]) for g in keys],dtype=np.int64)
    sizes=np.array([len(groups[g]) for g in keys],dtype=np.int64)
    # Independent grouped-sufficient-statistic replay; explicit local RNG/seed.
    rng=np.random.default_rng(6998);draws=[]
    for _ in range(2000):
        selection=rng.integers(low=0,high=len(keys),size=len(keys))
        draws.append(float(totals[selection].sum()/sizes[selection].sum()))
    return {'estimate':float(diff.mean()),'lower':float(np.quantile(draws,.025)),
            'upper':float(np.quantile(draws,.975)),'cluster_count':len(keys),'question_count':len(examples),
            'minimum_cluster_size':int(sizes.min()),'maximum_cluster_size':int(sizes.max()),
            'cross_round_clusters':sum(len({examples[i]['round'] for i in members})>1 for members in groups.values()),
            'replicates':2000,'seed':6998,'draws_sha256':sha(np.asarray(draws,dtype='<f8').tobytes())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--v3-bundle',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    new=Bundle(args.bundle,NEW_SHA);old=Bundle(args.v3_bundle,OLD_SHA)
    check(new.member_count==166 and len(new.manifest['files'])==165,'unexpected current inventory')
    check(not any('runtime-work/' in n or n.endswith('.safetensors') for n in new.raw),'unexpected weights in return')
    git_file=lambda name:subprocess.run(['git','show',COMMIT+':'+name],cwd=args.repo,check=True,capture_output=True).stdout
    protocol=load(git_file('configs/size_control_protocol.json'))
    manifest=new.sealed('data/manifest.json','manifest_hash');freeze=new.sealed('freeze.json','freeze_hash')
    done=new.sealed('predictions-job/completion.json','completion_hash');access=new.sealed('evaluation-access.json','access_hash')
    evaluation=new.sealed('evaluation.json','evaluation_hash')
    check(manifest['checkout']['commit']==COMMIT and manifest['checkout']['clean'] is True,'source checkout identity')
    for name,expected in manifest['source_files'].items():
        check(sha(git_file(name))==expected,'returned source differs from original commit: '+name)
        check(sha((args.repo/name).read_bytes())==expected,'current science source differs: '+name)
    check(manifest['checkout']['source_files']==manifest['source_files'],'checkout source inventory mismatch')
    check(manifest['protocol_hash']==digest(protocol),'protocol digest')
    for name,expected in manifest['files'].items():check(sha(new.raw[name])==expected,'prepared hash '+name)
    for name,expected in protocol['data']['source_sha256'].items():check(sha(old.raw[name])==expected,'old source pin '+name)
    check(new.raw['sealed/holdout_gold.json']==old.raw['sealed/holdout_gold.json'],'old gold changed')
    gold=new.j('sealed/holdout_gold.json')
    all_old=old.rows('data/examples.jsonl');expected_examples=[e for e in all_old if e['split'] in {'development','holdout'}]
    examples=new.rows('data/examples.jsonl');check(examples==expected_examples,'public input/schema/order differs from original')
    check(len(examples)==manifest['count']==3290,'public count')
    formal=[e for e in examples if e['split']=='holdout'];dev=[e for e in examples if e['split']=='development']
    check(len(formal)==3200 and len(dev)==90,'role counts')
    check(Counter(e['round'] for e in formal)=={'R1':1000,'R2':1000,'R3':1200},'round counts')
    check(len(set(e['sample_id'] for e in examples))==len(examples),'duplicate inputs')
    for e in examples:
        check(e['task']=='anli' and e['labels']==LABELS,'wrong task/class axis')
        premise,hypothesis=e['state'][len('Premise:\n'):].split('\n\nHypothesis:\n')
        norm=lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
        ph=sha(norm(premise).encode());ih=sha((norm(premise)+'\0'+norm(hypothesis)).encode())
        check(e['premise_hash']==ph and e['group_id']=='anli:premise:'+ph and e['input_hash']==ih,'normalized grouping mismatch')
    check(set(gold)=={e['sample_id'] for e in formal} and set(gold.values())<=set(LABELS),'gold coverage')
    check(not ({e['group_id'] for e in dev}&{e['group_id'] for e in formal}),'development/test premise overlap')
    config,rows,shard_count=predictions(new,'predictions-job/predictions/startlux')
    check(shard_count==128,'formal shard count')
    row_report,request_ids=validate_rows(rows,formal,'startlux',config=config,revision=REVISION)
    check(config==freeze['config'] and config['sources']==manifest['source_files'],'frozen configuration mismatch')
    check(config['request_hash']==digest([{'id':e['sample_id'],'request':decision_request(e)} for e in formal]),'independent request reconstruction differs')
    check(config['manifest_hash']==freeze['manifest_hash']==manifest['manifest_hash'],'freeze/prepared linkage')
    check(config['protocol_hash']==manifest['protocol_hash'],'config protocol linkage')
    check(config['role']=='startlux9b' and config['expected_predictions']==3200 and config['prediction_concurrency']==1,'role/count/concurrency')
    check(config['max_attempts']==2 and config['timeout_seconds']==120,'retry policy')
    check(freeze['evaluation']==protocol['evaluation'] and freeze['scope']==protocol['scope'],'frozen analysis plan')
    check(done['config_hash']==digest(config) and done['freeze_hash']==freeze['freeze_hash'],'completion linkage')
    check(done['predictions_hash']==digest(sorted(rows,key=lambda r:r['sample_id'])) and done['count']==3200 and done['failures']==row_report['errors'],'completion actual predictions')
    check(done['sampling']['new']==3200 and done['sampling']['requested']==3200 and done['sampling']['terminal_errors']==0,'collection sampling')
    check({r['session_id'] for r in rows}=={done['sampling']['session_id']},'collection session identity')
    check(access['freeze_hash']==freeze['freeze_hash'] and access['completion_hash']==done['completion_hash'] and access['manifest_hash']==manifest['manifest_hash'],'access linkage')
    check(evaluation['freeze_hash']==freeze['freeze_hash'] and evaluation['access_hash']==access['access_hash'] and evaluation['completion_hash']==done['completion_hash'],'evaluation linkage')
    check(evaluation['gold_sha256']==sha(new.raw['sealed/holdout_gold.json']) and evaluation['n']==3200 and evaluation['status']=='complete','evaluation identity')
    times=[dt(r['recorded_at']) for r in rows]
    check(dt(freeze['created_at'])<=min(times)<=max(times)<=dt(done['created_at'])<=dt(access['created_at'])<=dt(evaluation['created_at'])<=dt(new.manifest['created_at']),'freeze/collect/access/evaluation chronology')
    token=new.j('predictions-job/token_preflight.json');smoke=new.j('predictions-job/smoke.json')
    check(done['token_preflight_sha256']==sha(new.raw['predictions-job/token_preflight.json']) and done['smoke_sha256']==sha(new.raw['predictions-job/smoke.json']),'evidence hashes')
    token_expected={'effective_config_hash':digest(config['runtime']),'max_model_len':8192,'model':'startlux','model_revision':REVISION,'n_examples':3290,'overlong':[],'renderer':'official','status':'ok','truncated':False}
    check({k:token[k] for k in token_expected}==token_expected,'token identity')
    tr=token['rows'];check(len(tr)==3290 and {r['sample_id'] for r in tr}=={e['sample_id'] for e in examples},'token coverage')
    tokens={r['sample_id']:r for r in tr}
    for r in tr:check(type(r['input_tokens']) is int and 1<=r['input_tokens']<=8191 and r['reserved_output_tokens']==1 and re.fullmatch('[0-9a-f]{64}',r['token_ids_sha256']),'invalid token evidence')
    check(smoke['config_hash']==digest(config),'smoke configuration')
    smoke_reports=[]
    for phase in ('reference','current'):
        report,ids=validate_rows(smoke[phase],dev,'startlux');request_ids+=ids;smoke_reports.append(report)
    check(len(request_ids)==len(set(request_ids))==3380,'formal/smoke request IDs duplicate or count mismatch')
    check(dt(smoke['created_at'])<=min(times),'smoke after formal inference')
    sa={r['sample_id']:r for r in smoke['reference']};sb={r['sample_id']:r for r in smoke['current']}
    deltas=[max(abs(sa[e['sample_id']]['probabilities'][l]-sb[e['sample_id']]['probabilities'][l]) for l in LABELS) for e in dev]
    agree=float(np.mean([sa[e['sample_id']]['prediction']==sb[e['sample_id']]['prediction'] for e in dev]))
    smoke_check={'argmax_agreement':agree,'count':90,'mean_max_probability_delta':float(np.mean(deltas)),
                 'p95_max_probability_delta':float(np.quantile(deltas,.95)),
                 'passed':bool(agree>=.98 and np.mean(deltas)<=.01 and np.quantile(deltas,.95)<=.05)}
    check(smoke['checks']=={'anli':smoke_check} and smoke_check['passed'],'smoke reconstruction')
    smoke_attempts=[n for n in new.raw if n.startswith('predictions-job/smoke-attempts/')]
    check(len(smoke_attempts)==1 and new.raw[smoke_attempts[0]]==new.raw['predictions-job/smoke.json'],'startup smoke evidence')
    for row in rows+smoke['reference']+smoke['current']:
        check(row['token_counts']['input_tokens']==tokens[row['sample_id']]['input_tokens'],'response token count vs preflight')
    refs={};provenance=new.j('references/provenance.json');reference_counts={}
    for key,name in [('original_manifest','data/manifest.json'),('original_freeze','policy_freeze.json'),('original_access','holdout_access.json'),('original_evaluation','evaluation.json')]:
        check(provenance[key]==old.j(name),'historical provenance mismatch')
    for role in ('startlux4b','qwen4b','qwen9b'):
        interface='startlux' if role=='startlux4b' else 'qwen';root=f'predictions-jobs/{role}-holdout'
        rc,rr,rs=predictions(old,root+'/predictions/'+interface)
        saved=new.rows('references/'+role+'.jsonl')
        check(saved==sorted(rr,key=lambda r:r['sample_id']),'reference rows differ from original ZIP')
        check(provenance['jobs'][role]['config']==rc and provenance['jobs'][role]['completion']==old.j(root+'/completion.json'),'reference original job mismatch')
        check(provenance['jobs'][role]['rows_hash']==digest(saved)==access['reference_hashes'][role],'reference hash linkage')
        validate_rows(saved,formal,interface,config=rc,revision=rc['runtime']['model_revision'])
        refs[role]=saved;reference_counts[role]={'count':len(saved),'original_shards':rs,'predictions_hash':digest(saved)}
    # Every historical archive member has already passed raw SHA + bundle manifest.
    runtime=config['runtime'];model=protocol['model'];prefix='runtime-controls/startlux9b/runtime/'
    required={k:model[k] for k in ('native_temperature','gpu_memory_utilization','precision','max_model_len','think_mode','bidirectional','none_check','allow_none','prefix_cache','truncate')}
    required.update(model_repo=model['repo'],model_revision=REVISION,backend='vllm',vllm_version='0.31.0')
    check(all(runtime.get(k)==v for k,v in required.items()),'effective model settings')
    lock=new.j(prefix+'lock.json');installed=new.j(prefix+'installations.json');binding=new.j('runtime/binding.json')
    check(binding['source_files']==old.j('runtime/binding.json')['source_files'],'V1 source binding changed')
    check(binding['source_work_dir']==old.j('runtime/binding.json')['source_work_dir'] and binding['v1_dir']==old.j('runtime/binding.json')['v1_dir'],'old runtime root changed')
    identity=binding['control_identity']['startlux9b']
    for name,expected in identity['files'].items():check(digest(new.j(prefix+name))==expected,'runtime control identity '+name)
    check(digest({k:v for k,v in installed.items() if k!='models'})==identity['installation_environment'],'environment binding')
    check(new.j(prefix+'selected_memory.json')=={'startlux':.8} and new.j(prefix+'selected_profiles.json')=={'startlux':'primary'},'selected runtime')
    defaults=load(git_file('configs/runtime_defaults.json'));expected_lock=load(canonical(defaults))
    expected_lock['models']['startlux'].update(repo=model['repo'],revision=REVISION,native_temperature=1.1014)
    check({k:lock[k] for k in defaults}==expected_lock,'altered lock defaults')
    check(lock['lock_hash']==lock['defaults_hash']==digest(expected_lock)==installed['lock_hash'],'lock canonical identity')
    record=installed['models']['startlux'];check(record==new.j(prefix+'download-startlux9b.json'),'download record mismatch')
    check(record['revision']==REVISION and record['native_temperature']==1.1014 and record['model_path'].endswith('/snapshots/'+REVISION) and record['model_path'].startswith(binding['work_dir']+'/'),'download snapshot identity')
    check(installed['environments']['vllm-0.31.0']==old.j('runtime-controls/startlux4b/runtime/installations.json')['environments']['vllm-0.31.0'],'reused vLLM environment differs')
    doctor=new.j(prefix+'doctor.json');check(doctor['status']=='ok' and doctor['reserved_gpu_count']==1 and len(doctor['gpu'])==1,'GPU count/doctor')
    check(runtime['gpu']==[{k:doctor['gpu'][0][k] for k in ('name','memory_mb','driver','compute_capability')}] and runtime['gpu'][0]['name']=='NVIDIA L40S','effective GPU identity')
    session=new.j('runtime/session.json');check(session==new.j(prefix+'session.json'),'role session differs')
    check(session['allocation']=={'hostname':'sc1ns052is14','slurm_job_id':'66376166'} and dt(session['started_at'])<=dt(freeze['created_at']) and max(t.timestamp() for t in times)<session['deadline_epoch'],'allocation chronology')
    idle_names=[n for n in new.raw if n.startswith(prefix+'gpu-idle-')]
    check(len(idle_names)==1,'unexpected idle preflight inventory');idle=new.j(idle_names[0])
    check(idle['status']=='idle' and idle['compute_pids']==[] and dt(idle['created_at'])<dt(freeze['created_at']),'idle preflight')
    toolkit_names=[n for n in new.raw if n.startswith(prefix+'cuda-toolchain-') and n.endswith('.json')]
    check(len(toolkit_names)==1,'toolkit preflight inventory');toolkit=new.j(toolkit_names[0])
    check(toolkit['status']=='complete' and toolkit['compiler_header_check']['matched'] and toolkit['sampling']['observed']==toolkit['sampling']['expected']==[3,7] and toolkit['sampling']['cuda_synchronized'],'actual CUDA preflight')
    services=new.j(prefix+'last_services.json');check(services['models']['startlux']['config']==runtime,'served config differs')
    health=services['models']['startlux']['health'];check(health['status']=='ok' and health['kind']=='instant' and health['prompt_attention']=='bidirectional','health/model mode')
    check(dt(services['ready_at'])<=min(times)<=max(times)<=dt(services['stopped_at']),'service/prediction chronology')
    proc={r['name']:r for r in services['processes']};check(set(proc)=={'startlux','startlux-engine'},'unexpected services')
    engine=proc['startlux-engine']['command'];frontend=proc['startlux']['command']
    for flag,value in {'--model':record['model_path'],'--dtype':'bfloat16','--tensor-parallel-size':'1','--max-model-len':'8192','--gpu-memory-utilization':'0.8'}.items():check(engine[engine.index(flag)+1]==value,'engine launch '+flag)
    check('--no-enable-prefix-caching' in engine and '--no-enable-chunked-prefill' in engine and '--quantization' not in engine,'engine disabled caches/nonquantized')
    check(frontend[frontend.index('--think-mode')+1]=='never' and '--no-none-check' in frontend,'frontend mode')
    engine_name=prefix+'logs/'+Path(proc['startlux-engine']['log']).name;engine_text=new.raw[engine_name].decode()
    for text in ('prefix-bidirectional prompt attention ON','prefix caching off','dtype=torch.bfloat16','quantization=None','enable_prefix_caching=False','gpu_memory_utilization\': 0.8'):
        check(text in engine_text,'engine log did not confirm '+text)
    budget=new.j('runtime-controls/startlux9b/budget.json');check(len(budget['sessions'])==1 and budget['sessions'][0]['closed'],'runtime budget not closed')
    # Shutdown errors are retained. They occur after explicit shutdown messages,
    # not silently erased or treated as evidence of incomplete accepted rows.
    error_lines=[line for line in engine_text.splitlines() if re.search(r'\b(ERROR|Traceback|FATAL|OutOfMemoryError)\b',line)]
    check('EngineDeadError' in engine_text and engine_text.index('[shutdown]')<engine_text.index('AsyncLLM output_handler failed'),'unexpected runtime error ordering')
    check('OutOfMemoryError' not in engine_text,'OOM in engine log')
    metrics={};outcomes={};rounds={};subgroups={};confusions={}
    for role,pred in {**refs,'startlux9b':rows}.items():
        by={r['sample_id']:r for r in pred};metrics[role],outcomes[role]=direct(formal,by,gold,role)
        rounds[role]={r:direct([e for e in formal if e['round']==r],by,gold,role)[0] for r in ('R1','R2','R3')}
        subgroups[role]={l:direct([e for e in formal if gold[e['sample_id']]==l],by,gold,role)[0] for l in LABELS}
        matrix=[[0]*3 for _ in LABELS]
        for e in formal:
            r=by[e['sample_id']]
            if r['status']=='ok':matrix[LABELS.index(gold[e['sample_id']])][LABELS.index(r['prediction'])]+=1
        confusions[role]={'labels':LABELS,'true_rows_predicted_columns':matrix}
    check(metrics==evaluation['direct'] and rounds==evaluation['rounds_descriptive'],'direct/round metrics differ')
    check({r:metrics[r] for r in refs}==old.j('evaluation.json')['direct'],'original direct replay differs')
    comparisons={}
    for ref in ('qwen9b','startlux4b'):
        pairs=list(zip(outcomes['startlux9b'],outcomes[ref]));key='startlux9b_minus_'+ref
        counts={'both_correct':sum(a and b for a,b in pairs),'both_wrong':sum(not a and not b for a,b in pairs),
                'candidate_only_correct':sum(a and not b for a,b in pairs),'reference_only_correct':sum(not a and b for a,b in pairs)}
        replay=bootstrap(formal,outcomes['startlux9b'],outcomes[ref]);saved=evaluation['comparisons_descriptive'][key]
        check(all(saved[k]==v for k,v in counts.items()),'paired contingency mismatch')
        check(all(saved['difference'][k]==v for k,v in replay.items() if k!='draws_sha256'),'independent bootstrap mismatch')
        round_pairs={}
        for r in ('R1','R2','R3'):
            pair=[pair for e,pair in zip(formal,pairs) if e['round']==r]
            round_pairs[r]={'n':len(pair),'candidate_only_correct':sum(a and not b for a,b in pair),
                            'reference_only_correct':sum(not a and b for a,b in pair),
                            'net_correct_gain':sum(int(a)-int(b) for a,b in pair)}
        comparisons[key]={**counts,'difference':replay,'round_pair_counts':round_pairs}
    identity_report={'status':'PASS','bundle_sha256':new.sha,'archive_members':new.member_count,'manifest_members':len(new.manifest['files']),
        'old_archive_sha256':old.sha,'old_manifest_members':len(old.manifest['files']),'source_commit':COMMIT,
        'source_files_verified':len(manifest['source_files']),'self_hash_artifacts':5,'formal_shards':shard_count,'formal_rows':row_report,
        'smoke_passes':smoke_reports,'smoke_recomputed':smoke_check,'unique_new_request_ids':len(request_ids),
        'tokens':{'count':len(tr),'min':min(r['input_tokens'] for r in tr),'max':max(r['input_tokens'] for r in tr),'truncated':False,'matched_response_usage':3380},
        'reference_roles':reference_counts,'gold_sha256':sha(new.raw['sealed/holdout_gold.json']),
        'time_chain':{'freeze':freeze['created_at'],'first_prediction':min(times).isoformat(),'last_prediction':max(times).isoformat(),
                      'completion':done['created_at'],'access':access['created_at'],'evaluation':evaluation['created_at'],'export':new.manifest['created_at']},
        'source_identity_limit':'Saved source, unsigned artifact and runtime chains verified; not a cryptographic signature or independent OS access trace.'}
    runtime_report={'status':'PASS_WITH_DISCLOSED_METADATA_CONFLICT','runtime':runtime,'allocation':session['allocation'],
        'idle_preflight':idle,'toolkit_preflight_status':toolkit['status'],'old_environment_unchanged':True,'budget_closed':True,
        'services':{k:services[k] for k in ('started_at','ready_at','stopped_at')},'collection_elapsed_seconds':done['sampling']['elapsed_seconds'],
        'monitored_session_elapsed_seconds':budget['sessions'][0]['elapsed_seconds'],
        'prefix_cache':{'health_value':health['engine']['prefix_cache'],'effective_config':False,'launch_flag':'--no-enable-prefix-caching',
            'engine_log':'enable_prefix_caching=False','plugin_log':'prefix caching off','interpretation':'Direct launch and engine evidence disabled caching; health field semantics unresolved, same previously disclosed V3 conflict.'},
        'shutdown_diagnostics':{'error_line_count':len(error_lines),'first_error':error_lines[0],
            'interpretation':'EngineDeadError follows explicit shutdown/forced EngineCore cleanup at local09:11:24. All3200 rows completed successfully; retain teardown errors without claiming they were inference failures.','all_error_lines':error_lines},
        'limits':['Weights/model metadata bytes are not included in the returned ZIP; revision and readout binding verified through fixed protocol, download receipt, launch path and runtime evidence.',
                  'Token IDs are saved hash evidence, not an independent rerun of the tokenizer;3380 reported response token counts match the preflight.',
                  'Serial collection time excludes startup and is not a service goodput/latency comparison. GPU measurements belong to the separate service study.']}
    metrics_report={'status':'PASS','analysis':'Independent raw-response parsing, direct scoring and normalized-premise paired bootstrap',
        'numpy_version':np.__version__,'direct':metrics,'rounds':rounds,'by_gold_label_descriptive':subgroups,'confusion_matrices':confusions,
        'comparisons':comparisons,'scope':'Retrospective fixed model comparison on already viewed ANLI test; no tuning, fitting, new routing or equivalence claim.',
        'statistical_limits':['Percentile intervals condition on these fixed checkpoints/prompts and empirical premise clusters; not architecture causality or future performance guarantees.',
        'Per-round and gold-label subgroups are descriptive; no multiple-comparison-adjusted subgroup inference performed.']}
    for name,value in [('identity-audit.json',identity_report),('runtime-audit.json',runtime_report),('metrics-audit.json',metrics_report)]:write(args.out/name,value)
    write(args.out/'audit-receipt.json',{'status':'PASS','audit_script_sha256':sha(Path(__file__).read_bytes()),'archive_sha256':new.sha,
          'outputs':{name:sha((args.out/name).read_bytes()) for name in ('identity-audit.json','runtime-audit.json','metrics-audit.json')},
          'executed_production_fitting_evaluation_or_gpu':False})
    print(json.dumps({'status':'PASS','direct':metrics,'comparisons':comparisons,'runtime_caveat':runtime_report['prefix_cache']},indent=2))

if __name__=='__main__':main()
