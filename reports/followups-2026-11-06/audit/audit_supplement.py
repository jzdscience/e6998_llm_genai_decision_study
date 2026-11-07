"""Additional independent source inventory, warmup and admission identity checks."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--archive', type=Path)
parser.add_argument('--evidence-root', type=Path)
parser.add_argument('--out', type=Path, required=True, help='Audit output directory; an existing supplemental receipt is never overwritten')
args = parser.parse_args()
REPO = args.repo_root.expanduser().resolve()
ARCHIVE = (args.archive or REPO.parent / 'service_study_bundle.zip').expanduser().resolve()
evidence = (args.evidence_root or REPO.parent / 'project/solo-experiments-2026-10-30/decision-study').expanduser().resolve()
OLD = evidence / 'records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip'
HERE = args.out.expanduser().resolve()
HERE.mkdir(parents=True, exist_ok=True)
if (HERE/'supplemental-audit.json').exists():
    parser.error('Supplemental receipt already exists; preserve it and choose a new output')
COMMIT = '8b5370d8aea7ed57dc42c92f4fe07bcf023a8e07'

with zipfile.ZipFile(OLD) as z:
    examples = [json.loads(x) for x in z.read('data/examples.jsonl').splitlines()]
    old_warmup = sorted([r for r in examples if r['task']=='clinc' and r['split']=='development'], key=lambda r:r['sample_id'])
tracked = subprocess.check_output(['git','ls-tree','-r','--name-only',COMMIT],cwd=REPO,text=True).splitlines()
expected = {n for n in tracked if n.endswith('.py') and n.rsplit('/',1)[0] in
            {'decision_study/service_study','decision_study/cheap_baselines'}}
expected.update({'configs/service_study_protocol.json','configs/cheap_baseline_requirements.txt',
    'configs/runtime_defaults.json','configs/cuda_build_toolkit_13_0_flashinfer.txt',
    'decision_study/common.py','decision_study/adapters.py','decision_study/runtime.py',
    'decision_study/budget.py','decision_study/v3/runtime.py','decision_study/baselines/study.py',
    'decision_study/runtime_compat.py','decision_study/v3/cli.py','decision_study/v3/identity.py',
    'scripts/model_probe.py','scripts/prepare_cuda_toolkit.py','scripts/cuda_toolchain_preflight.py',
    'scripts/run_service_study.sh','scripts/run_followup_suite.sh'})
request_ids = set()
rows = []
with zipfile.ZipFile(ARCHIVE) as z:
    settings = json.loads(z.read('settings.json'))
    assert set(settings['source']['files']) == expected
    assert json.loads(z.read('warmup.json')) == old_warmup and len(old_warmup)==64
    for cell in settings['order']:
        folder = f'cells/{cell["cell_id"]}/attempt-0001'
        plan = json.loads(z.read(folder+'/plan.json'))
        traces = [json.loads(x) for x in z.read(folder+'/traces.jsonl').splitlines()]
        events=[]
        newids=0
        for r in traces:
            if r['rejected']:continue
            events.extend([(r['dispatch_seconds'],1),(r['completion_seconds'],-1)])
            if cell['condition'] in {'startlux4b','qwen4b'}:
                result=r['result']; attempt=result['attempts'][0]; ident=attempt['request_id']
                assert ident not in request_ids
                request_ids.add(ident);newids+=1
                assert result['latency_ms']>=attempt['latency_ms']
                if cell['condition']=='startlux4b':
                    assert attempt['server_request_id']==ident
                else:
                    assert result['raw_response']['id']=='chatcmpl-'+ident
        level=peak=0
        for _,delta in sorted(events):
            level+=delta
            assert 0<=level<=64
            peak=max(peak,level)
        assert level==0
        attempt=json.loads(z.read(folder+'/attempt.json'))
        assert attempt['prior_interrupted_attempts']==[] and attempt['explicit_resume_acknowledgment'] is False
        rows.append({'cell_id':cell['cell_id'],'max_observed_client_inflight':peak,'unique_correlated_gpu_request_ids':newids})
out={'status':'PASS','source_inventory_count':len(expected),'original_warmup_exact':True,
     'unique_correlated_gpu_request_ids':len(request_ids),'max_client_inflight_bound':64,
     'all_attempt_receipts_have_no_prior_interruption':True,'cells':rows}
with (HERE/'supplemental-audit.json').open('x') as handle:
    handle.write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='cells'},indent=2))
