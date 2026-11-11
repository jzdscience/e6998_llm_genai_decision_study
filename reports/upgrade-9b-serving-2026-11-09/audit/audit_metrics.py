#!/usr/bin/env python3
"""Independent stdlib-only replay of saved service traces; never invokes a model.

Portable copy of the script accepted on 2026-11-10 (see audit/README.md). Only the location of the
inputs and of the output changed; every check and every computed value is the accepted one.

  python3 -I audit_metrics.py --evidence-root ROOT --output DIR
  (--evidence-root defaults to $EVIDENCE_ROOT; --bundle, --gold, --v3-bundle and --size-bundle
   override the individual inputs, which otherwise sit at their indexed paths under ROOT)

Writes audit-receipt.json, recomputed.json, cells.csv, stall-analysis.json and stall_late_rows.csv
to DIR and exits non-zero on any failed assertion or discrepancy.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import zipfile

if not __debug__:
    raise SystemExit('Run without -O: the replay checks are assert statements.')

# Paths of the inputs relative to the external evidence root (records/BUNDLE_INDEX.json and
# docs/ARTIFACTS_ZH.md); the hashes below are checked inside the replay.
DEFAULT_INPUTS = {
    'bundle': 'records/bundles/upgrade-formal-20261109-9baeca3eb7f8.zip',
    'gold': 'results/solo-v3-prepared/sealed/holdout_gold.json',
    'v3_bundle': 'records/bundles/v3-return-20261103-232529-a8d0dc672d83.zip',
    'size_bundle': 'records/bundles/size-control-complete-20261106-f427f7762bac.zip',
}


def quantile(values, q):
    if not values:
        return None
    s = sorted(values)
    x = (len(s) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return s[lo] + (s[hi] - s[lo]) * (x - lo)


def stats(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values), values=values)


def compare(actual, expected, path, discrepancies):
    if isinstance(actual, dict):
        for key, value in actual.items():
            if key not in expected:
                discrepancies.append((path + '.' + key, 'missing'))
            else:
                compare(value, expected[key], path + '.' + key, discrepancies)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            discrepancies.append((path, 'length mismatch'))
        else:
            for i, (a, b) in enumerate(zip(actual, expected)):
                compare(a, b, f'{path}[{i}]', discrepancies)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
            discrepancies.append((path, actual, expected))
    elif actual != expected:
        discrepancies.append((path, actual, expected))


def hist_label(history, condition, sid):
    if condition == 'cascade':
        role = 'qwen9b' if history['startlux4b'][sid]['deferred'] else 'startlux4b'
        return history[role][sid]['prediction']
    return history[condition][sid]['prediction']


def route_score(stage, policy):
    confidence = min(max(max(stage['probabilities'].values()), 1e-6), 1 - 1e-6)
    labels = ['entailment', 'neutral', 'contradiction']
    x = [math.log(confidence / (1 - confidence))] + [float(stage['prediction'] == x) for x in labels]
    head = policy['local_head']
    logits = head['coefficients'][0] + sum(c * ((v-m) / s) for c, v, m, s in
        zip(head['coefficients'][1:], x, head['feature_mean'], head['feature_scale']))
    return 1 - 1 / (1 + math.exp(-max(-40, min(40, logits))))


def summarize(rows, gold, duration, slo):
    accepted = [x for x in rows if not x['rejected']]
    ok = [x for x in accepted if x['status'] == 'ok']
    timely = [x for x in ok if 1000 * (x['completion_seconds'] - x['arrival_seconds']) <= slo]
    correct = [x for x in ok if x['result']['prediction'] == gold[x['sample_id']]]
    timely_correct = [x for x in correct if 1000 * (x['completion_seconds'] - x['arrival_seconds']) <= slo]
    return {
        'offered': len(rows), 'admitted': len(accepted), 'successful': len(ok),
        'failed_admitted': len(accepted) - len(ok), 'rejected': len(rows) - len(accepted),
        'status_counts': dict(Counter(x['status'] for x in rows)),
        'duration_seconds': duration, 'offered_rate_actual': len(rows) / duration,
        'met_slo': len(timely), 'slo_fraction_all_offered': len(timely) / len(rows),
        'goodput_per_second': len(timely) / duration,
        'successful_per_offered_window_second': len(ok) / duration,
        'latency_ms_all_admitted': {f'p{int(100*q)}': quantile([1000*(x['completion_seconds']-x['arrival_seconds']) for x in accepted], q) for q in (.5, .95, .99)},
        'max_dispatch_lag_ms': max(1000*(x['dispatch_seconds']-x['arrival_seconds']) for x in rows),
        'correct': len(correct), 'correct_met_slo': len(timely_correct),
        'correct_goodput_per_second': len(timely_correct) / duration,
        'correct_fraction_all_offered': len(correct) / len(rows),
        'accuracy_successful': len(correct) / len(ok),
    }


def timings(row):
    stage_ms = sum(s['latency_ms'] for s in row['result']['stages'])
    return {
        'arrival_seconds': row['arrival_seconds'], 'worker_start_seconds': row['worker_start_seconds'],
        'completion_seconds': row['completion_seconds'],
        'dispatch_lag_ms': 1000*(row['dispatch_seconds']-row['arrival_seconds']),
        'client_queue_ms': row['client_queue_ms'], 'worker_ms': row['worker_ms'],
        'stage_latency_sum_ms': stage_ms,
        'worker_outside_stage_latency_ms': row['worker_ms'] - stage_ms,
        'arrival_to_completion_ms': row['arrival_to_completion_ms'],
    }


def verify_history(history, settings, policy, v3_path, size_path):
    checks = {}
    sources = [
        ('startlux4b', v3_path, 'a8d0dc672d83c3edcb85d005e66fa4b3d000132729f98f0beaab6865dfc57a9e', 'predictions-jobs/startlux4b-holdout', 'startlux'),
        ('qwen9b', v3_path, 'a8d0dc672d83c3edcb85d005e66fa4b3d000132729f98f0beaab6865dfc57a9e', 'predictions-jobs/qwen9b-holdout', 'qwen'),
        ('startlux9b', size_path, 'f427f7762bac745d9316cb02195c09f98ab551794de1da047509be7f9fe7d1ed', 'predictions-job', 'startlux'),
    ]
    for role, path, sha, folder, model in sources:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        with zipfile.ZipFile(path) as z:
            prefix = folder + '/predictions/' + model + '/shard-'
            shards = sorted(n for n in z.namelist() if n.startswith(prefix) and n.endswith('.json'))
            rows = []
            for name in shards:
                part = json.loads(z.read(name))
                assert part['count'] == len(part['rows'])
                rows.extend(part['rows'])
            done = json.loads(z.read(folder + '/completion.json'))
            computed = hashlib.sha256(json.dumps(sorted(rows,key=lambda r:r['sample_id']), ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
            assert computed == done['predictions_hash'] == settings['history']['sources'][role]['predictions_hash']
            assert len(rows) == len(history[role]) == 3200
            assert {r['sample_id'] for r in rows} == set(history[role])
            for r in rows:
                actual = history[role][r['sample_id']]
                assert all(actual[k] == r[k] for k in ('prediction','status'))
                if role != 'qwen9b': assert actual['probabilities'] == r['probabilities']
                if role == 'startlux4b':
                    assert actual['deferred'] == (route_score(r,policy) >= policy['decision']['threshold'])
            checks[role] = {'source_zip_sha256':sha, 'source_prediction_hash':computed,
                'shards':len(shards), 'predictions_compared':len(rows)}
    assert sum(r['deferred'] for r in history['startlux4b'].values()) == 766
    return checks


def main():
    parser = argparse.ArgumentParser(description='Independent replay of the 9B serving-cost traces.')
    parser.add_argument('--evidence-root', type=Path, default=os.environ.get('EVIDENCE_ROOT'))
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--gold', type=Path)
    parser.add_argument('--v3-bundle', type=Path)
    parser.add_argument('--size-bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='new or existing directory for the receipts')
    args = parser.parse_args()
    for name, relative in DEFAULT_INPUTS.items():
        if getattr(args, name) is None:
            if args.evidence_root is None:
                parser.error(f'pass --evidence-root (or set EVIDENCE_ROOT) or --{name.replace("_", "-")}')
            setattr(args, name, args.evidence_root / relative)
    args.output.mkdir(parents=True, exist_ok=True)
    gold_bytes = args.gold.read_bytes()
    assert hashlib.sha256(gold_bytes).hexdigest() == 'ed3525b721ed564a4cdf733e94ac381cfebb4ba06eb4a3a6844a1a0b37824f14'
    gold = json.loads(gold_bytes)
    discrepancies, cells, flat, stall = [], [], [], {}
    all_labels = defaultdict(lambda: defaultdict(set))
    totals = Counter()
    raw_checks = Counter()
    cascade_scores_max_diff = 0.
    with zipfile.ZipFile(args.bundle) as z:
        read = lambda name: json.loads(z.read(name))
        ev, settings, history, policy = (read(x) for x in ('evaluation.json','settings.json','history.json','policy.json'))
        p = settings['protocol']
        assert len(ev['cells']) == 36
        assert {x['cell_id'] for x in ev['cells']} == {x['cell_id'] for x in settings['order']}
        policy = policy['policy']
        history_source_checks = verify_history(history, settings, policy, args.v3_bundle, args.size_bundle)
        for condition in p['conditions']:
            assert set(history[condition if condition != 'cascade' else 'qwen9b']) == set(gold)
        historical = {c: sum(hist_label(history,c,sid)==label for sid,label in gold.items()) / len(gold) for c in p['conditions']}
        compare(historical, ev['historical_full_pool_accuracy'], 'historical', discrepancies)
        for original in ev['cells']:
            folder, condition = original['attempt'], original['condition']
            rows = [json.loads(x) for x in z.read(folder+'/traces.jsonl').splitlines()]
            plan = read(folder+'/plan.json')
            schedule = read(plan['schedule_file'])
            assert len(rows) == len(schedule) and len({r['offer_id'] for r in rows}) == len(rows)
            by_offer = {r['offer_id']: r for r in rows}
            for offer in schedule:
                r = by_offer[offer['offer_id']]
                assert all(r[k] == offer[k] for k in offer)
                raw_checks['offered_schedule_rows'] += 1
                if r['rejected']:
                    assert r['status'] == 'client_rejected' and r['result'] is None
                    continue
                assert r['status'] == r['result']['status'] == 'ok'
                assert r['result']['prediction'] in set(gold.values())
                assert math.isclose(r['arrival_to_completion_ms'], 1000*(r['completion_seconds']-r['arrival_seconds']), abs_tol=1e-9)
                assert math.isclose(r['client_queue_ms'], 1000*(r['worker_start_seconds']-r['dispatch_seconds']), abs_tol=1e-9)
                assert math.isclose(r['worker_ms'], 1000*(r['completion_seconds']-r['worker_start_seconds']), abs_tol=1e-9)
                raw_checks['admitted_timing_decompositions'] += 1
                stages = r['result']['stages']
                assert all(s['status']=='ok' and len(s['attempts'])==1 and s['attempts'][0]['status']=='ok' for s in stages)
                for stage in stages:
                    if stage['interface']=='startlux':
                        answer = stage['raw_response']['answers']['decision']
                        assert answer['choice'] == stage['prediction']
                        assert answer['probabilities'] == stage['probabilities']
                        assert stage['probabilities'][stage['prediction']] >= max(stage['probabilities'].values()) - 1e-8
                    else:
                        answer = json.loads(stage['raw_response']['choices'][0]['message']['content'])
                        assert answer['label'] == stage['prediction']
                    raw_checks['stage_predictions_match_raw_HTTP_response'] += 1
                assert r['result']['prediction'] == stages[-1]['prediction']
                if condition == 'cascade':
                    score = route_score(stages[0], policy)
                    cascade_scores_max_diff = max(cascade_scores_max_diff, abs(score-r['result']['route']['score']))
                    deferred = score >= policy['decision']['threshold']
                    assert deferred == r['result']['route']['deferred'] == r['result']['deferred']
                    assert len(stages) == (2 if deferred else 1)
                    assert r['result']['route']['fallback_called'] == deferred
                    assert [s['role'] for s in stages] == (['startlux4b','qwen9b'] if deferred else ['startlux4b'])
                    raw_checks['cascade_routes_replayed_from_probabilities'] += 1
                else:
                    assert len(stages)==1 and stages[0]['role']==condition
                all_labels[condition][r['sample_id']].add(r['result']['prediction'])
            ok = [r for r in rows if r['status']=='ok']
            details = {'offered':len(rows), 'admitted':len(ok), 'successful':len(ok), 'rejected':len(rows)-len(ok),
                       'status_counts':dict(Counter(r['status'] for r in rows)),
                       'late_successful':{str(s):sum(r['arrival_to_completion_ms']>s for r in ok) for s in (500,1000)}}
            details['history'] = {
                'successful_responses':len(ok),
                'label_agreement':sum(r['result']['prediction']==hist_label(history,condition,r['sample_id']) for r in ok)/len(ok),
                'matched_subset_historical_accuracy':sum(hist_label(history,condition,r['sample_id'])==gold[r['sample_id']] for r in ok)/len(ok),
                'live_accuracy':sum(r['result']['prediction']==gold[r['sample_id']] for r in ok)/len(ok),
            }
            if condition=='cascade':
                n_deferred = sum(r['result']['route']['deferred'] for r in ok)
                details['cascade'] = {
                    'route_reasons':dict(Counter(r['result']['route']['reason'] for r in ok)),
                    'routed':len(ok), 'deferred':n_deferred, 'deferral_rate_routed':n_deferred/len(ok),
                    'fallback_calls':n_deferred,
                    'route_agreement_with_history':sum(r['result']['route']['deferred']==history['startlux4b'][r['sample_id']]['deferred'] for r in ok)/len(ok),
                }
            cell = {k:original[k] for k in ('cell_id','condition','rate','repeat')}
            cell['primary'] = summarize(rows,gold,120,500)
            cell['secondary'] = summarize(rows,gold,120,1000)
            cell['details'] = details
            compare(cell,original,cell['cell_id'],discrepancies)
            cells.append(cell)
            totals.update(details['status_counts'])
            flat.append({**{k:cell[k] for k in ('cell_id','condition','rate','repeat')},
                **{k:cell['primary'][k] for k in ('offered','successful','rejected','correct','accuracy_successful')},
                'goodput500':cell['primary']['correct_goodput_per_second'],
                'goodput1000':cell['secondary']['correct_goodput_per_second'],
                'timely_fraction500':cell['primary']['slo_fraction_all_offered'],
                'timely_fraction1000':cell['secondary']['slo_fraction_all_offered'],
                'latency_p95_ms':cell['primary']['latency_ms_all_admitted']['p95'],
                **details['history']})
            if cell['cell_id']=='r0-startlux4b-rate20':
                late = [r for r in ok if r['arrival_to_completion_ms']>500]
                t = [timings(r) for r in late]
                stall = {'cell_id':cell['cell_id'], 'late500_count':len(late),
                    'late_offers_timings':{k:stats([row[k] for row in t]) for k in t[0]},
                    'all_offers_max_dispatch_lag_ms':max(1000*(r['dispatch_seconds']-r['arrival_seconds']) for r in rows),
                    'all_offers_max_worker_outside_stage_ms':max(r['worker_ms']-sum(s['latency_ms'] for s in r['result']['stages']) for r in ok),
                    'late_offer_ids':[r['offer_id'] for r in late],
                    'boundary':'Observed queue, dispatch and client worker timing; underlying stall cause not isolated.'}
                with (args.output/'stall_late_rows.csv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(t[0])); writer.writeheader();writer.writerows(t)
        summaries=[]
        passing={}
        for c in p['conditions']:
            passing[c]={}
            for key in ('primary','secondary'):
                passing[c][key]=[rate for rate in p['rates_per_second'] if all(x[key]['slo_fraction_all_offered']>=.99 for x in cells if x['condition']==c and x['rate']==rate)]
            for rate in p['rates_per_second']:
                selected=[x for x in cells if x['condition']==c and x['rate']==rate]
                metrics={}
                for key in ('primary','secondary'):
                    for name in ('correct_goodput_per_second','goodput_per_second','slo_fraction_all_offered','accuracy_successful'):
                        metrics[key+'.'+name]=stats([x[key][name] for x in selected])
                metrics['successful_per_second']=stats([x['primary']['successful_per_offered_window_second'] for x in selected])
                metrics['latency_p95_ms']=stats([x['primary']['latency_ms_all_admitted']['p95'] for x in selected])
                if c=='cascade':metrics['deferral_rate_routed']=stats([x['details']['cascade']['deferral_rate_routed'] for x in selected])
                summaries.append({'condition':c,'rate':rate,'repeats':len(selected),'metrics':metrics})
        compare(summaries,ev['summaries'],'summaries',discrepancies)
        for c,values in passing.items():
            for key,rates in values.items():
                actual=max(rates) if rates else None
                assert ev['sustained_rates'][key+'_slo'][c]['highest_passing_rate']==actual
        variability={c:{'inputs_seen':len(d),'inputs_with_more_than_one_live_label':sum(len(v)>1 for v in d.values())} for c,d in all_labels.items()}
        compare(variability,ev['label_variability'],'label_variability',discrepancies)
        live_history_comparison = {}
        for c in p['conditions']:
            selected = [x for x in cells if x['condition']==c]
            histories = [x['details']['history'] for x in selected]
            success_count = sum(h['successful_responses'] for h in histories)
            live_history_comparison[c] = {
                'successful_response_count': success_count,
                'live_accuracy_per_cell_range':[min(h['live_accuracy'] for h in histories),max(h['live_accuracy'] for h in histories)],
                'live_minus_matched_historical_accuracy_pp_per_cell_range': [
                    min(100*(h['live_accuracy']-h['matched_subset_historical_accuracy']) for h in histories),
                    max(100*(h['live_accuracy']-h['matched_subset_historical_accuracy']) for h in histories)],
                'pooled_live_accuracy':sum(h['live_accuracy']*h['successful_responses'] for h in histories)/success_count,
                'pooled_matched_historical_accuracy':sum(h['matched_subset_historical_accuracy']*h['successful_responses'] for h in histories)/success_count,
            }
        receipt={'status':'PASS' if not discrepancies else 'FAIL',
            'scope':'Independent stdlib replay; no production scoring, routing or verification imports; no new model calls or fitting.',
            'bundle_sha256':hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
            'gold_sha256':hashlib.sha256(gold_bytes).hexdigest(),
            'cells':len(cells),'offered_total':sum(totals.values()),'status_counts':dict(totals),
            'raw_checks':dict(raw_checks),'route_score_max_abs_difference':cascade_scores_max_diff,
            'history_source_checks':history_source_checks,
            'historical_full_pool_accuracy':historical,'passing_rates_all_three_repeats_99pct':passing,
            'live_history_comparison':live_history_comparison,
            'label_variability':variability,'discrepancies':discrepancies,
            'interpretation_limits':['99% passing points are not continuous capacity boundaries, especially 4B failing20 but passing40.',
                'Successful-response accuracy is offer-weighted and load-selected; compare the matched historical subset, not only whole-pool accuracy.',
                'Three serving repeats are not independent input samples or statistical replication of model quality.',
                'Client/engine settings and unequal resource budgets restrict generalization; no causal or dollar-cost conclusion.']}
    for name,value in [('audit-receipt.json',receipt),('recomputed.json',{'cells':cells,'summaries':summaries}),('stall-analysis.json',stall)]:
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
    raise SystemExit(bool(discrepancies))


if __name__=='__main__':main()
