"""Independent retrospective result audit; no project imports, model fitting or model inference."""
from pathlib import Path, PurePosixPath
from collections import Counter
import argparse, json, zipfile, hashlib, math, unicodedata

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', type=Path, required=True)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--artifact-root', type=Path, required=True,
                    help='Historical artifact root with records/bundles and results directories')
parser.add_argument('--output', type=Path, required=True, help='New independent audit JSON path')
args = parser.parse_args()
REPO = args.repo.resolve()
OLD = args.artifact_root.resolve()
V1 = OLD / 'results/returned-experiment-2026-11-02-101054'
V2 = OLD / 'results/solo-v2-prepared'
BUNDLE = args.bundle.resolve()
OUTPUT = args.output.resolve()
V1_ARCHIVE = OLD / 'records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip'
EXPECTED_COMMIT = '797aff1edfe17d7daaf93551e6cfc236035b6945'
LABELS = ['account_blocked','balance','bill_balance','bill_due','freeze_account','interest_rate','min_payment','order_checks','pay_bill','pin_change','report_fraud','routing','spending_history','transactions','transfer','oos']

def sha(raw): return hashlib.sha256(raw).hexdigest()
def canon(d): return (json.dumps(d,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def unseal(d):
    payload={k:v for k,v in d.items() if k!='self_hash'}
    assert sha(canon(payload))==d['self_hash']

def rows(raw): return [json.loads(x) for x in raw.splitlines() if x.strip()]

with zipfile.ZipFile(BUNDLE) as z:
    names=z.namelist()
    assert len(names)==len(set(names))==7
    assert set(names)=={'protocol.json','model.json','predictions.jsonl','fit.json','freeze.json','evaluation-access.json','evaluation.json'}
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
    assert z.testzip() is None
    raw={n:z.read(n) for n in names}
p={n:json.loads(b) for n,b in raw.items() if n.endswith('.json')}
freeze,access,ev,protocol=p['freeze.json'],p['evaluation-access.json'],p['evaluation.json'],p['protocol.json']
for d in (freeze,access,ev): unseal(d)
assert freeze['status']=='FROZEN' and ev['status']=='COMPLETE'
assert set(freeze['artifacts'])=={'protocol.json','model.json','fit.json','predictions.jsonl'}
for n,h in freeze['artifacts'].items(): assert sha(raw[n])==h
assert access['freeze_sha256']==ev['freeze_sha256']==sha(raw['freeze.json'])
assert ev['access_sha256']==sha(raw['evaluation-access.json'])
assert access['gold_sha256']==ev['gold_sha256']==freeze['sources']['v1_gold_sha256']
assert freeze['label_order']==protocol['labels']==p['model.json']['prediction_label_order']==LABELS
assert freeze['source_identity']['git_commit']==EXPECTED_COMMIT
assert freeze['source_identity']['git_available'] is True and freeze['source_identity']['git_dirty'] is False
for name,h in freeze['source_identity']['files'].items(): assert sha((REPO/name).read_bytes())==h
assert sha((REPO/'configs/clinc_tfidf400_protocol.json').read_bytes())==freeze['protocol_sha256']
assert json.loads((REPO/'configs/clinc_tfidf400_protocol.json').read_bytes())==protocol
assert freeze['environment']['packages']==protocol['environment']
paths={'v1_manifest_sha256':V1/'data/manifest.json','v1_examples_sha256':V1/'data/examples.jsonl','v1_gold_sha256':V1/'data/gold.json','v2_examples_sha256':V2/'data/examples.jsonl','v2_fit_gold_sha256':V2/'data/fit_gold.json'}
source_raw={k:v.read_bytes() for k,v in paths.items()}
assert freeze['sources']==protocol['sources']
for k,b in source_raw.items(): assert sha(b)==protocol['sources'][k]
examples=rows(source_raw['v1_examples_sha256'])
gold=json.loads(source_raw['v1_gold_sha256'])
fitgold=json.loads(source_raw['v2_fit_gold_sha256'])
v2=rows(source_raw['v2_examples_sha256'])
roles={role:sorted((r for r in examples if r['task']=='clinc' and r['split']==role),key=lambda r:r['sample_id']) for role in ['calibration','validation','test','stress']}
assert {k:len(v) for k,v in roles.items()}==protocol['counts']
selected=sum(roles.values(),[])
assert len(set(r['sample_id'] for r in selected))==len(selected)
normalized=[' '.join(unicodedata.normalize('NFKC',r['state']).casefold().split()) for r in selected]
assert len(set(normalized))==len(selected) and all(normalized)
assert all(r['labels']==LABELS for r in selected)
assert all(not any(k in r for k in ('gold','label','answers','is_impossible')) for r in selected)
byid={r['sample_id']:r for r in selected}
v2clinc=[r for r in v2 if r['task']=='clinc']
assert Counter(r['split'] for r in v2clinc)==Counter(calibration=400,validation=400)
assert len(set(r['sample_id'] for r in v2clinc))==800
assert all(byid[r['sample_id']]==r for r in v2clinc)
for role, bank, oos in [('calibration',20,100),('validation',20,100),('test',30,150),('stress',0,850)]:
    expected=Counter({l:bank for l in LABELS[:-1]});expected['oos']=oos
    assert Counter(gold[r['sample_id']] for r in roles[role])==expected
    if role in ('calibration','validation'): assert all(fitgold[r['sample_id']]==gold[r['sample_id']] for r in roles[role])
trainids=[r['sample_id'] for r in roles['calibration']]
assert freeze['training_ids']==trainids
pred=rows(raw['predictions.jsonl']);predids=[r['sample_id'] for r in pred]
expectedrows=roles['validation']+roles['test']+roles['stress']
assert predids==[r['sample_id'] for r in expectedrows]
assert len(set(predids))==1850 and not set(predids)&set(trainids)
for r,e in zip(pred,expectedrows):
    vals=r['probabilities']
    assert r['split']==e['split'] and len(vals)==16
    assert all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in vals)
    assert math.isclose(sum(vals),1,abs_tol=1e-12)
    assert r['predicted_label']==LABELS[max(range(16),key=lambda i:vals[i])]
assert p['fit.json']['training_count']==len(trainids)==400
assert p['fit.json']['training_label_counts']==Counter(gold[i] for i in trainids)
assert p['fit.json']['model_bytes']==len(raw['model.json'])
assert p['fit.json']['feature_count']==len(p['model.json']['vocabulary'])

# Derive metrics from counters. No use of project metric functions or sklearn.
def metrics(ids, predictions):
    pairs=Counter((gold[sid],predictions[sid]) for sid in ids)
    actual=Counter(gold[sid] for sid in ids)
    guessed=Counter(predictions[sid] for sid in ids)
    correct=sum(pairs[l,l] for l in LABELS)
    details={}
    for label in LABELS:
        tp=pairs[label,label]; fp=guessed[label]-tp; fn=actual[label]-tp
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        details[label]={'support':actual[label],'predicted':guessed[label],'true_positive':tp,'false_positive':fp,'false_negative':fn,'precision':precision,'recall':recall,'f1':f1}
    bankids=[i for i in ids if gold[i]!='oos']
    m={'count':len(ids),'accuracy':correct/len(ids),'macro_f1':sum(d['f1'] for d in details.values())/16,'macro_f1_label_count':16,
       'banking_accuracy':sum(predictions[i]==gold[i] for i in bankids)/len(bankids) if bankids else None,
       'oos':{k:details['oos'][k] for k in ('precision','recall','f1')},'confusion_matrix':[[pairs[a,b] for b in LABELS] for a in LABELS],'label_order':LABELS}
    errors={'correct':correct,'incorrect':len(ids)-correct,'banking_count':len(bankids),'banking_correct':sum(predictions[i]==gold[i] for i in bankids),
            'banking_to_oos':sum(predictions[i]=='oos' for i in bankids),'banking_to_other_banking':sum(predictions[i] not in (gold[i],'oos') for i in bankids),
            'oos_count':actual['oos'],'oos_correct':pairs['oos','oos'],'predicted_oos':guessed['oos']}
    return m,details,errors

def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a: close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): close(x,y)
    elif isinstance(a,float): assert math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-13),(a,b)
    else: assert a==b,(a,b)

ids_by_role={k:[r['sample_id'] for r in v] for k,v in roles.items()}
mainbank=[i for i in ids_by_role['test'] if gold[i]!='oos']
pools={'validation':ids_by_role['validation'],'main_test':ids_by_role['test'],'oos_only_stress':ids_by_role['stress'],'historical_mixed_stress':mainbank+ids_by_role['stress']}
baseline={r['sample_id']:r['predicted_label'] for r in pred}
computed={}; details={};errors={}
for pool,ids in pools.items():
    computed[pool],details[pool],errors[pool]=metrics(ids,baseline)
    expected=ev['metrics'][pool]
    compare=computed[pool].copy()
    if pool=='oos_only_stress':compare={'count':len(ids),'oos_recall':computed[pool]['oos']['recall']}
    if pool=='historical_mixed_stress':compare['overlapping_main_banking']=len(set(ids)&set(ids_by_role['test']))
    close(compare,expected)

# V1 bytes are checked against both original archive and its bound bundle manifest.
assert sha(V1_ARCHIVE.read_bytes())=='be748917f2f2184e3a7d3468b4fa222e7d1b77c0abbba863fce64f395c1840e7'
llm_metrics={};pairwise={};manifest_raw=(V1/'bundle_manifest.json').read_bytes();manifest=json.loads(manifest_raw)
verified_v1_files={}
with zipfile.ZipFile(V1_ARCHIVE) as z:
    assert z.read('bundle_manifest.json')==manifest_raw
    for model in ['startlux','qwen','kev']:
        llm={};names=[f'predictions/{model}/config.json']+sorted(n for n in z.namelist() if n.startswith(f'predictions/{model}/shard-') and n.endswith('.json'))
        for n in names:
            b=z.read(n);assert sha(b)==manifest['files'][n]
            assert (V1/n).read_bytes()==b
            verified_v1_files[n]=sha(b)
            if '/shard-' not in n:continue
            shard=json.loads(b)
            for r in shard['rows']:
                assert r['sample_id'] not in llm
                assert r['status']=='ok'
                llm[r['sample_id']]=r['prediction']
        llm_metrics[model]={};pairwise[model]={}
        for pool,ids in pools.items():
            missing=set(ids)-llm.keys()
            if missing:
                llm_metrics[model][pool]={'not_available_count':len(missing)};continue
            mm,dd,ee=metrics(ids,llm)
            llm_metrics[model][pool]={'metrics':mm,'counts':ee}
            c=Counter((baseline[i]==gold[i],llm[i]==gold[i]) for i in ids)
            pairwise[model][pool]={'count':len(ids),'both_correct':c[True,True],'tfidf_only_correct':c[True,False], 'llm_only_correct':c[False,True], 'both_wrong':c[False,False], 'tfidf_minus_llm_pp':100*(c[True,False]-c[False,True])/len(ids)}

result={'status':'PASS','kind':'independent_retrospective_audit','source_bundle_sha256':sha(BUNDLE.read_bytes()),'source_commit':EXPECTED_COMMIT,
 'source_files_verified':{k:{'path':str(paths[k]),'sha256':sha(b)} for k,b in source_raw.items()},
 'artifact_hashes':{n:sha(b) for n,b in raw.items()},'checks':{'self_hashes':True,'freeze_access_evaluation_chain':True,'source_commit_and_file_identity':True,'protocol_identity':True,'package_pins':True,'training_exact_calibration_400':True,'eval_coverage_order_and_probabilities':True,'normalized_text_disjointness':True,'v2_input_and_fitgold_alignment':True,'all_reported_metrics_recomputed':True},
 'training_count':400,'prediction_count':1850,'counts_by_split':{k:len(v) for k,v in roles.items()},
 'metrics':computed,'per_class':details,'error_counts':errors,'v1_comparison':llm_metrics,'paired_descriptive_comparison':pairwise,
 'v1_archive':{'path':str(V1_ARCHIVE),'sha256':sha(V1_ARCHIVE.read_bytes()),'manifest_sha256':sha(manifest_raw),'verified_files':verified_v1_files},
 'limitations':['Post-hoc descriptive comparison on previously viewed test/stress; no new confirmation or model selection.','400 supervised calibration labels differ from original prompted LLM decision setting.','Timing evidence is not replayed or validated by this metric audit.','Receipt hash chain documents declared lifecycle; not an independent forensic proof of historical reads.']}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
for pool,m in computed.items(): print(pool,{k:m[k] for k in ('count','accuracy','macro_f1','banking_accuracy','oos')},errors[pool])
for model,x in llm_metrics.items():
    for pool,m in x.items(): print(model,pool,{k:m['metrics'][k] for k in ('accuracy','macro_f1','banking_accuracy','oos')} if 'metrics' in m else m)
print('PAIRED',json.dumps(pairwise))
print('PASS',OUTPUT)
