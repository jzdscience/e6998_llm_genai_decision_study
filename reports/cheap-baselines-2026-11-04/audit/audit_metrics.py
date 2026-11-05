"""Independent returned-classifier audit using only stdlib; no fitting or inference.

Run without python -O: assertions are intentional identity/metric audit guards.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import subprocess
import unicodedata
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle', required=True, type=Path)
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--artifact-root', required=True, type=Path)
parser.add_argument('--prepared-dir', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
REPO, OLD, PREPARED = args.repo.resolve(), args.artifact_root.resolve(), args.prepared_dir.resolve()
V1 = OLD / 'results/returned-experiment-2026-11-02-101054'
V2 = OLD / 'results/solo-v2-prepared'
COMMIT = '1b70899bbcf15ee686df04b36e0c333a2fc21e5c'
EXPECTED_SHA = '7f00178c7bd1d86eeeb121929131447675d0053c10072f687f88d7e3ae77b2be'
RECIPES = ['tfidf1600', 'minilm400', 'minilm1600']
LABELS = ['account_blocked','balance','bill_balance','bill_due','freeze_account','interest_rate','min_payment','order_checks','pay_bill','pin_change','report_fraud','routing','spending_history','transactions','transfer','oos']
FIT_FILES = ['protocol.json', 'embeddings.npy', 'embeddings.json'] + [f'{r}.{s}' for r in RECIPES for s in ['model.json','predictions.jsonl','fit.json']]

def sha(value): return hashlib.sha256(value).hexdigest()
def canon(value): return (json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def rows(value): return [json.loads(line) for line in value.splitlines() if line.strip()]
def norm(value): return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())
def unseal(value): assert value['self_hash'] == sha(canon({k:v for k,v in value.items() if k!='self_hash'}))
def git_file(commit,name): return subprocess.check_output(['git','-C',str(REPO),'show',f'{commit}:{name}'],timeout=10)

def archive(path, expected_sha, fitted_names):
    assert sha(path.read_bytes()) == expected_sha
    names = set(fitted_names) | {'freeze.json','evaluation-access.json','evaluation.json'}
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist()) == len(names) and set(z.namelist()) == names
        assert sum(i.file_size for i in z.infolist()) < 30_000_000
        for info in z.infolist():
            assert not PurePosixPath(info.filename).is_absolute() and '..' not in PurePosixPath(info.filename).parts
            assert not stat.S_ISLNK(info.external_attr>>16)
        assert z.testzip() is None
        raw = {name:z.read(name) for name in names}
    doc = {name:json.loads(value) for name,value in raw.items() if name.endswith('.json')}
    freeze, access, ev = [doc[name] for name in ['freeze.json','evaluation-access.json','evaluation.json']]
    for value in (freeze,access,ev): unseal(value)
    assert freeze['status']=='FROZEN' and ev['status']=='COMPLETE'
    assert set(freeze['artifacts'])==set(fitted_names)
    for name,value in freeze['artifacts'].items(): assert sha(raw[name])==value
    assert access['freeze_sha256']==ev['freeze_sha256']==sha(raw['freeze.json'])
    assert ev['access_sha256']==sha(raw['evaluation-access.json'])
    assert access['gold_sha256']==ev['gold_sha256']==freeze['sources']['v1_gold_sha256']
    return raw,doc

raw,doc = archive(args.bundle,EXPECTED_SHA,FIT_FILES)
freeze,ev,protocol = doc['freeze.json'],doc['evaluation.json'],doc['protocol.json']
assert protocol['experiment_id']=='clinc-cheap-v1' and protocol['recipes']==RECIPES and protocol['labels']==LABELS
assert freeze['source_identity']['git_commit']==COMMIT and freeze['source_identity']['git_available'] is True and freeze['source_identity']['git_dirty'] is False
for name,value in freeze['source_identity']['files'].items(): assert sha(git_file(COMMIT,name))==value
protocol_bytes=git_file(COMMIT,'configs/clinc_cheap_protocol.json')
assert sha(protocol_bytes)==freeze['protocol_sha256'] and json.loads(protocol_bytes)==protocol
assert freeze['environment']['packages']==protocol['environment'] and freeze['environment']['threads']==1
assert freeze['prepared_hashes']==protocol['prepared'] and freeze['sources']==protocol['sources']
assert freeze['model_identity']=={k:protocol['encoder'][k] for k in ['repository','revision','files','provider']}

# Only after COMPLETE and all identity/access checks do we read historical gold.
paths = {'v1_examples_sha256':V1/'data/examples.jsonl', 'v1_manifest_sha256':V1/'data/manifest.json',
         'v1_gold_sha256':V1/'data/gold.json', 'v2_examples_sha256':V2/'data/examples.jsonl', 'v2_fit_gold_sha256':V2/'data/fit_gold.json'}
sources={name:path.read_bytes() for name,path in paths.items()}
for name,value in sources.items(): assert sha(value)==freeze['sources'][name]
manifest=json.loads(sources['v1_manifest_sha256'])
assert manifest['files']['examples.jsonl']==freeze['sources']['v1_examples_sha256'] and manifest['files']['gold.json']==freeze['sources']['v1_gold_sha256']
examples=rows(sources['v1_examples_sha256']); gold=json.loads(sources['v1_gold_sha256']); fit_gold=json.loads(sources['v2_fit_gold_sha256'])
roles={name:sorted((r for r in examples if r['task']=='clinc' and r['split']==name),key=lambda r:r['sample_id']) for name in ['calibration','validation','test','stress']}
assert {k:len(v) for k,v in roles.items()}=={'calibration':400,'validation':400,'test':600,'stress':850}
selected=sum(roles.values(),[])
assert len(set(r['sample_id'] for r in selected))==len(set(norm(r['state']) for r in selected))==2250
assert all(r['labels']==LABELS and not set(r)&{'gold','label','answers','is_impossible'} for r in selected)
byid={r['sample_id']:r for r in selected}
v2=[r for r in rows(sources['v2_examples_sha256']) if r['task']=='clinc']
assert len(v2)==len(set(r['sample_id'] for r in v2))==800
assert Counter(r['split'] for r in v2)==Counter(calibration=400,validation=400)
assert all(r==byid[r['sample_id']] for r in v2)
for split,bank,oos in [('calibration',20,100),('validation',20,100),('test',30,150),('stress',0,850)]:
    assert Counter(gold[r['sample_id']] for r in roles[split])==Counter({**{l:bank for l in LABELS[:-1]},'oos':oos})
    if split in ('calibration','validation'): assert all(gold[r['sample_id']]==fit_gold[r['sample_id']] for r in roles[split])
prepared_raw={name:(PREPARED/name).read_bytes() for name in protocol['prepared']}
for name,value in prepared_raw.items(): assert sha(value)==protocol['prepared'][name]
prepared_manifest=json.loads(prepared_raw['manifest.json'])
assert prepared_manifest['project_sources']==protocol['sources'] and prepared_manifest['raw_sources']==protocol['raw']['files']
full_train=rows(prepared_raw['train1600.jsonl']); train_byid={r['sample_id']:r for r in full_train}
assert len(full_train)==len(train_byid)==len(set(norm(r['state']) for r in full_train))==1600
assert full_train==sorted(full_train,key=lambda r:r['sample_id'])
assert Counter(r['label'] for r in full_train)==Counter({l:100 for l in LABELS})
calibration=[train_byid[r['sample_id']] for r in roles['calibration']]
assert all(r['state']==byid[r['sample_id']]['state'] and r['label']==fit_gold[r['sample_id']] for r in calibration)
expected_rows=roles['validation']+roles['test']+roles['stress']
expected_ids=[r['sample_id'] for r in expected_rows]
assert not set(expected_ids)&train_byid.keys()
assert not set(norm(r['state']) for r in expected_rows)&set(norm(r['state']) for r in full_train)
development=[r for r in examples if r['task']=='clinc' and r['split']=='development']
assert sum(r['sample_id'] in train_byid for r in development)==60
assert len(development)==64
feature_meta=doc['embeddings.json']
assert feature_meta['inputs']==[{'sample_id':r['sample_id'],'text_sha256':sha(r['state'].encode())} for r in full_train+expected_rows]
assert feature_meta['matrix_sha256']==sha(raw['embeddings.npy']) and feature_meta['model_identity']==freeze['model_identity']
assert feature_meta['encoding']['count']==3450 and len(feature_meta['encoding']['token_lengths'])==3450
assert feature_meta['encoding']['truncated_count']==0 and max(feature_meta['encoding']['token_lengths'])==feature_meta['encoding']['max_tokens_including_special']<=256

def verify_predictions(pred):
    assert [r['sample_id'] for r in pred]==expected_ids
    assert len(pred)==len(set(r['sample_id'] for r in pred))==1850
    for r,e in zip(pred,expected_rows):
        p=r['probabilities']
        assert r['split']==e['split'] and len(p)==16 and all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in p)
        assert abs(sum(p)-1)<1e-8 and r['predicted_label']==LABELS[max(range(16),key=p.__getitem__)]
    return {r['sample_id']:r['predicted_label'] for r in pred}

predictions={}; trainings={}
for recipe in RECIPES:
    train=calibration if recipe=='minilm400' else full_train
    assert freeze['training_ids'][recipe]==[r['sample_id'] for r in train]
    fit=doc[f'{recipe}.fit.json']; model=doc[f'{recipe}.model.json']
    assert fit['recipe']==recipe and fit['training_count']==len(train) and fit['prediction_count']==1850
    assert fit['training_label_counts']==Counter(r['label'] for r in train) and fit['model_bytes']==len(raw[f'{recipe}.model.json'])
    assert model['prediction_label_order']==LABELS and len(model['classes'])==16 and set(model['classes'])==set(LABELS)
    assert all(0<x<protocol['classifier']['max_iter'] for x in model['iterations'])
    assert fit['interpretation']==ev['interpretation']==protocol['interpretation']
    if recipe.startswith('minilm'):
        assert model['encoder']==protocol['encoder'] and model['embeddings_sha256']==sha(raw['embeddings.npy'])
    predictions[recipe]=verify_predictions(rows(raw[f'{recipe}.predictions.jsonl']))
    trainings[recipe]={'count':len(train),'label_counts':dict(Counter(r['label'] for r in train))}

# Historical TF-IDF400 return is a separate fixed run, not a new refit.
old_path=OLD/'records/bundles/light-baseline-20261105-064105-7fcd28785a06.zip'
old_raw,old_doc=archive(old_path,'7fcd28785a06bf235ee46ca829e4d839cf88743894c5b965cc3e04fcb215414f',
                        ['protocol.json','model.json','predictions.jsonl','fit.json'])
assert old_doc['freeze.json']['sources']==freeze['sources']
assert old_doc['freeze.json']['training_ids']==[r['sample_id'] for r in calibration]
assert old_doc['freeze.json']['source_identity']['git_commit']=='797aff1edfe17d7daaf93551e6cfc236035b6945'
for name,value in old_doc['freeze.json']['source_identity']['files'].items():
    assert sha(git_file(old_doc['freeze.json']['source_identity']['git_commit'],name))==value
predictions['tfidf400']=verify_predictions(rows(old_raw['predictions.jsonl']))

# Recheck every consumed original LLM prediction against the immutable formal ZIP.
v1_archive=OLD/'records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip'
assert sha(v1_archive.read_bytes())=='be748917f2f2184e3a7d3468b4fa222e7d1b77c0abbba863fce64f395c1840e7'
v1_verified={}
with zipfile.ZipFile(v1_archive) as z:
    manifest_raw=z.read('bundle_manifest.json'); manifest=json.loads(manifest_raw)
    assert (V1/'bundle_manifest.json').read_bytes()==manifest_raw
    for name in ['kev','qwen','startlux']:
        values={}
        members=[f'predictions/{name}/config.json']+sorted(n for n in z.namelist() if n.startswith(f'predictions/{name}/shard-') and n.endswith('.json'))
        for member in members:
            content=z.read(member)
            assert sha(content)==manifest['files'][member] and (V1/member).read_bytes()==content
            v1_verified[member]=sha(content)
            if '/shard-' in member:
                for row in json.loads(content)['rows']:
                    assert row['sample_id'] not in values and row['status']=='ok'
                    values[row['sample_id']]=row['prediction']
        predictions[name]=values

bank_ids=[r['sample_id'] for r in roles['test'] if gold[r['sample_id']]!='oos']
pools={'validation':[r['sample_id'] for r in roles['validation']], 'main_test':[r['sample_id'] for r in roles['test']],
       'oos_only_stress':[r['sample_id'] for r in roles['stress']],
       'historical_mixed_stress':bank_ids+[r['sample_id'] for r in roles['stress']]}

def metric(ids,pred):
    pairs=Counter((gold[sid],pred[sid]) for sid in ids); actual=Counter(gold[sid] for sid in ids); guessed=Counter(pred[sid] for sid in ids)
    per_class={}
    for label in LABELS:
        tp=pairs[label,label]; fp=guessed[label]-tp; fn=actual[label]-tp
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        per_class[label]={'support':actual[label],'predicted':guessed[label],'tp':tp,'fp':fp,'fn':fn,
                          'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    correct=sum(pairs[l,l] for l in LABELS); bank=[i for i in ids if gold[i]!='oos']; bank_correct=sum(pred[i]==gold[i] for i in bank)
    counts={'correct':correct,'incorrect':len(ids)-correct,'banking_correct':bank_correct,'banking_count':len(bank),
            'banking_to_oos':sum(pred[i]=='oos' for i in bank),
            'banking_to_other_banking':sum(pred[i] not in (gold[i],'oos') for i in bank),
            'oos_correct':pairs['oos','oos'],'oos_count':actual['oos'],'oos_to_banking':actual['oos']-pairs['oos','oos'],'predicted_oos':guessed['oos']}
    metrics={'count':len(ids),'accuracy':correct/len(ids),'macro_f1':sum(p['f1'] for p in per_class.values())/16,'macro_f1_label_count':16,
             'banking_accuracy':bank_correct/len(bank) if bank else None,'oos':{k:per_class['oos'][k] for k in ['precision','recall','f1']},
             'confusion_matrix':[[pairs[a,b] for b in LABELS] for a in LABELS],'label_order':LABELS}
    return {'metrics':metrics,'counts':counts,'per_class':per_class}

def close(actual,expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys(),(actual.keys(),expected.keys())
        for key in actual:close(actual[key],expected[key])
    elif isinstance(actual,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):close(a,b)
    elif isinstance(actual,float):assert math.isclose(actual,expected,rel_tol=1e-13,abs_tol=1e-13),(actual,expected)
    else:assert actual==expected,(actual,expected)

all_metrics={}
for name,pred in predictions.items():
    all_metrics[name]={}
    for pool,ids in pools.items():
        if set(ids)-pred.keys():
            all_metrics[name][pool]={'unavailable':len(set(ids)-pred.keys())};continue
        result=metric(ids,pred);all_metrics[name][pool]=result
        if name in RECIPES or name=='tfidf400':
            saved=ev['metrics'][name][pool] if name in RECIPES else old_doc['evaluation.json']['metrics'][pool]
            actual=result['metrics'].copy()
            if pool=='oos_only_stress':actual={'count':len(ids),'oos_recall':result['metrics']['oos']['recall']}
            if pool=='historical_mixed_stress':actual['overlapping_main_banking']=len(set(ids)&set(pools['main_test']))
            close(actual,saved)

def paired(ids,challenger,reference):
    c=Counter((predictions[challenger][i]==gold[i],predictions[reference][i]==gold[i]) for i in ids)
    return {'count':len(ids),'both_correct':c[True,True],'both_wrong':c[False,False],
            'challenger_only_correct':c[True,False],'reference_only_correct':c[False,True],
            'challenger_minus_reference_pp':100*(c[True,False]-c[False,True])/len(ids)}

comparisons={}
for challenger,reference in [('tfidf1600','tfidf400'),('minilm400','tfidf400'),('minilm1600','tfidf1600'),
                              ('minilm1600','minilm400'),('minilm1600','startlux'),('minilm400','startlux'),
                              ('minilm1600','qwen'),('minilm400','qwen')]:
    key=f'{challenger}_minus_{reference}';comparisons[key]={'challenger':challenger,'reference':reference}
    for pool,ids in pools.items():
        if set(ids)-predictions[challenger].keys() or set(ids)-predictions[reference].keys():continue
        comparisons[key][pool]=paired(ids,challenger,reference)
        for stratum,sub in [('banking',[i for i in ids if gold[i]!='oos']),('oos',[i for i in ids if gold[i]=='oos'])]:
            if sub:comparisons[key][pool][stratum]=paired(sub,challenger,reference)

result={'status':'PASS','audit_kind':'Independent retrospective metric and identity audit; no production imports/fits/inference',
        'source_bundle_sha256':EXPECTED_SHA,'source_commit':COMMIT,'artifact_hashes':{k:sha(v) for k,v in raw.items()},
        'checks':{'complete_three_recipe_freeze_access_evaluation_chain':True,'released_source_protocol_and_packages':True,
                  'five_source_input_hashes':True,'prepared1600_identity':True,'calibration400_exact_subset':True,
                  'normalized_train_eval_disjointness':True,'prediction_rows_probabilities_and_label_axis':True,
                  'shared_feature_ids_text_hashes':True,'all_reported_new_and_tfidf400_metrics_match':True},
        'counts':{'new_recipes':3,'predictions_per_recipe':1850,'new_prediction_rows':5550,'distinct_evaluation_inputs':1850,'shared_embedding_inputs':3450},
        'training':trainings,'metrics':all_metrics,'paired_descriptive_comparison':comparisons,
        'historical_references':{'tfidf400_archive_sha256':sha(old_path.read_bytes()),'v1_archive_sha256':sha(v1_archive.read_bytes()),
                                 'v1_manifest_sha256':sha(manifest_raw),'v1_verified_file_count':len(v1_verified),'v1_verified_files':v1_verified},
        'limitations':['Already-viewed data; descriptive comparisons, no model selection or fresh confirmation.',
                       '400-to1600 changes label count and OOS prior; the1600 pool includes60 engineering-development inputs.',
                       'Supervised classifiers differ from the original LLM prompting/calibration setting.',
                       'Main600 and mixed1300 share450banking samples; pureOOS850 is recall, not a multiclass recognition benchmark.',
                       'Hash chains verify stored identities, not independent historical OS read order.',
                       'This audit does not remeasure timings or replay model numeric computations.']}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
for name,p in all_metrics.items():
    m=p['main_test'];print(name,{k:m['metrics'][k] for k in ['accuracy','macro_f1','banking_accuracy','oos']},m['counts'], 'mixed',p['historical_mixed_stress']['metrics']['accuracy'])
print('PASS',args.output)
