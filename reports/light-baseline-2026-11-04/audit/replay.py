"""Independent, read-only replay of returned JSON parameters (never fit a model).

Uses only stdlib + numpy for tokenization, saved TF-IDF transformation, logits,
softmax, and class remapping. Does not import production baseline code or sklearn.
The training text pass validates vocabulary membership/order and IDF provenance;
it is not used to alter the saved model or generate a new classifier.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,re,zipfile,platform,subprocess
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--examples',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
REPO=args.repo
ZIP=args.bundle
EXAMPLES=args.examples
COMMIT='797aff1edfe17d7daaf93551e6cfc236035b6945'
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda d:(json.dumps(d,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    assert len(names)==len(set(names))==7
    assert set(names)=={'protocol.json','model.json','predictions.jsonl','fit.json','freeze.json','evaluation-access.json','evaluation.json'}
    assert all(i.file_size<20_000_000 for i in z.infolist())
    raw={n:z.read(n) for n in names}
doc={n:json.loads(v) for n,v in raw.items() if n!='predictions.jsonl'}
freeze=doc['freeze.json']; protocol=doc['protocol.json']; model=doc['model.json']; fit=doc['fit.json']
for n in ('freeze.json','evaluation-access.json','evaluation.json'):
    d=doc[n]; assert sha(canonical({k:v for k,v in d.items() if k!='self_hash'}))==d['self_hash'],n
for n,s in freeze['artifacts'].items(): assert sha(raw[n])==s,n
assert doc['evaluation-access.json']['freeze_sha256']==sha(raw['freeze.json'])
assert doc['evaluation.json']['freeze_sha256']==sha(raw['freeze.json'])
assert doc['evaluation.json']['access_sha256']==sha(raw['evaluation-access.json'])
assert freeze['source_identity']['git_commit']==COMMIT
assert freeze['source_identity']['git_available'] is True and freeze['source_identity']['git_dirty'] is False
for n,s in freeze['source_identity']['files'].items():
    b=subprocess.check_output(['git','-C',str(REPO),'show',f'{COMMIT}:{n}'])
    assert sha(b)==s,n
pname='configs/clinc_tfidf400_protocol.json'
protocol_raw=subprocess.check_output(['git','-C',str(REPO),'show',f'{COMMIT}:{pname}'])
assert sha(protocol_raw)==freeze['protocol_sha256']
assert json.loads(protocol_raw)==protocol
assert freeze['sources']==protocol['sources']
assert freeze['environment']['packages']==protocol['environment']
assert sha(EXAMPLES.read_bytes())==protocol['sources']['v1_examples_sha256']
examples=[json.loads(x) for x in EXAMPLES.read_text().splitlines() if x]
selected=[r for r in examples if r['task']=='clinc' and r['split'] in protocol['counts']]
by_id={r['sample_id']:r for r in selected}
roles={name:sorted([r for r in selected if r['split']==name],key=lambda r:r['sample_id']) for name in protocol['counts']}
assert {k:len(v) for k,v in roles.items()}==protocol['counts']
train=roles['calibration']; expected=roles['validation']+roles['test']+roles['stress']
assert freeze['training_ids']==[r['sample_id'] for r in train]
assert set(freeze['training_ids']).isdisjoint(r['sample_id'] for r in expected)
preds=[json.loads(x) for x in raw['predictions.jsonl'].splitlines() if x]
assert [r['sample_id'] for r in expected]==[r['sample_id'] for r in preds]
assert [r['split'] for r in expected]==[r['split'] for r in preds]
assert len(set(r['sample_id'] for r in preds))==1850
assert model['kind']=='tfidf_logistic_regression'
assert protocol['vectorizer']=={
 'analyzer':'word','ngram_range':[1,2],'lowercase':True,'strip_accents':None,
 'token_pattern':r'(?u)\b\w\w+\b','stop_words':None,'min_df':1,'max_df':1.0,
 'max_features':None,'binary':False,'norm':'l2','use_idf':True,'smooth_idf':True,'sublinear_tf':False}
tokenizer=re.compile(protocol['vectorizer']['token_pattern'])
def terms(text):
    t=tokenizer.findall(text.lower())
    return t+[' '.join(t[i:i+2]) for i in range(len(t)-1)]
vocab=model['vocabulary']; nfeature=len(vocab)
assert sorted(vocab.values())==list(range(nfeature))
df=Counter()
for r in train:df.update(set(terms(r['state'])))
assert vocab=={t:i for i,t in enumerate(sorted(df))}
idf=np.asarray(model['idf'],dtype=np.float64)
expected_idf=np.log((1.+len(train))/(np.asarray([df[t] for t in sorted(df)])+1.))+1.
idf_error=float(np.max(np.abs(expected_idf-idf)))
assert np.allclose(expected_idf,idf,rtol=0,atol=1e-14)
coef=np.asarray(model['coefficients'],dtype=np.float64)
intercept=np.asarray(model['intercept'],dtype=np.float64)
assert idf.shape==(nfeature,) and coef.shape==(16,nfeature) and intercept.shape==(16,)
assert all(np.isfinite(x).all() for x in (idf,coef,intercept))
assert fit['feature_count']==nfeature and fit['model_bytes']==len(raw['model.json'])
assert model['iterations'] and all(0<n<protocol['classifier']['max_iter'] for n in model['iterations'])
classes=model['classes']; labels=protocol['labels']
assert len(classes)==16 and set(classes)==set(labels) and model['prediction_label_order']==labels
assert freeze['label_order']==labels
X=np.zeros((len(preds),nfeature),dtype=np.float64)
zero=[]
for i,r in enumerate(expected):
    counts=Counter(terms(r['state']))
    for t,n in counts.items():
        if t in vocab: X[i,vocab[t]]=n*idf[vocab[t]]
    norm=float(np.linalg.norm(X[i]))
    if norm: X[i]/=norm
    else: zero.append(r['sample_id'])
logits=X@coef.T+intercept
exp=np.exp(logits-logits.max(axis=1,keepdims=True))
prob=exp/exp.sum(axis=1,keepdims=True)
remap=[classes.index(label) for label in labels]
prob=prob[:,remap]
saved=np.asarray([r['probabilities'] for r in preds],dtype=np.float64)
assert saved.shape==prob.shape and np.isfinite(saved).all() and (saved>=0).all() and (saved<=1).all()
assert np.allclose(saved.sum(axis=1),1.,rtol=0,atol=1e-12)
error=np.abs(saved-prob)
assert np.allclose(saved,prob,rtol=0,atol=1e-12)
replayed=[labels[i] for i in prob.argmax(axis=1)]
match=sum(label==row['predicted_label'] for label,row in zip(replayed,preds))
assert match==len(preds)
result={
 'status':'PASS','audit_kind':'Independent saved-parameter mathematical replay; no fitting or production prediction/evaluation API calls',
 'archive_sha256':sha(ZIP.read_bytes()),'source_commit':COMMIT,
 'source_files_verified':freeze['source_identity']['files'],
 'protocol_source_sha256':sha(protocol_raw),'protocol_export_sha256':sha(raw['protocol.json']),
 'protocol_semantically_equal':True,'protocol_formatting_explains_different_hashes':True,
 'archive_hash_chain_verified':True,
 'examples_sha256':sha(EXAMPLES.read_bytes()),'train_rows':len(train),'prediction_rows':len(preds),
 'prediction_split_counts':dict(Counter(r['split'] for r in preds)),
 'feature_count':nfeature,'coefficient_shape':list(coef.shape),'iterations':model['iterations'],
 'training_vocabulary_exact_match':True,'training_idf_max_absolute_error':idf_error,
 'classes':classes,'prediction_label_order':labels,'class_column_remap':remap,
 'model_parameters_finite':True,'label_matches':match,'probability_max_absolute_error':float(error.max()),
 'probability_mean_absolute_error':float(error.mean()),'replay_probability_atol':1e-12,
 'zero_feature_rows':len(zero),'zero_feature_split_counts':dict(Counter(by_id[x]['split'] for x in zero)),
 'audit_environment':{'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__},
 'reported_timing':fit['timing'],
 'timing_scope_review':{
  'fit_clock':'Includes calibration-only vocabulary/IDF fit_transform and logistic-regression fit; starts inside threadpool/warnings context, after object construction.',
  'prediction_clock':'One combined 1,850-row transform + predict_proba + class-column remap; excludes model/row JSON serialization, process startup, imports, input verification, service and queues.',
  'evidence_limit':'Stored elapsed durations are not independently remeasured by this audit. Single batch, no repetitions and no CPU hardware model recorded; cannot claim request latency, SLO goodput, whole-session cost, energy or a measured speedup over GPU service.'}
}
args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
