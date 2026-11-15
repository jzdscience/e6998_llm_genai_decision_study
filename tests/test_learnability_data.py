"""Artificial allocation fixtures; no production holdout labels are read by tests."""
import copy
import json
from pathlib import Path
import zipfile

import pytest

from decision_study.learnability import data as d


def fixture_source(tmp_path, monkeypatch, *, bad_dev_gold=False):
    source = tmp_path/'source.zip'
    records = {}
    history_rows = []
    eligible_dev = []
    for rd in d.ROUNDS:
        rows = []
        for letter, label in d.LABEL_MAP.items():
            for i in range(20):
                rows.append({'uid': f'{rd}-train-{letter}-{i:03d}',
                             'context': f'{rd} {letter} premise group {i//2}',
                             'hypothesis': f'{rd} {letter} hypothesis {i}', 'label': letter})
        records[rd, 'train'] = rows
        first = rows[0]
        ph, ih = d.hashes(first['context'], first['hypothesis'])
        history_rows.append(d.example({'sample_id': 'anli:'+first['uid'], 'premise': first['context'],
                                      'hypothesis': first['hypothesis'], 'round': rd, 'source_split': 'train',
                                      'premise_hash': ph, 'input_hash': ih}, 'fitting'))
        dev = [{'uid': f'{rd}-dev-blocked', 'context': first['context'], 'hypothesis': 'held out variant', 'label': 'not-a-label' if bad_dev_gold else 'n'},
               {'uid': f'{rd}-dev-eligible', 'context': f'{rd} novel dev premise', 'hypothesis': 'dev claim', 'label': 'not-a-label' if bad_dev_gold else 'n'}]
        records[rd, 'dev'] = dev
        fresh = dev[1]
        ph, ih = d.hashes(fresh['context'], fresh['hypothesis'])
        eligible_dev.append({'sample_id': 'anli:'+fresh['uid'], 'round': rd, 'source_split': 'dev', 'premise_hash': ph, 'input_hash': ih})
        records[rd, 'test'] = [{'uid': f'{rd}-test', 'context': f'{rd} n premise group 0', 'hypothesis': 'test claim', 'label': 'ignored'}]
    with zipfile.ZipFile(source, 'w') as archive:
        for (rd, split), rows in records.items():
            archive.writestr(f'anli_v1.0/{rd}/{split}.jsonl', '\n'.join(json.dumps(r) for r in rows)+'\n')
    history = tmp_path/'history.jsonl'
    history.write_bytes(d.jsonl(history_rows))
    monkeypatch.setattr(d, 'SOURCE_SHA256', d.file_hash(source))
    monkeypatch.setattr(d, 'HISTORY_SHA256', d.file_hash(history))
    monkeypatch.setattr(d, 'HISTORY_COUNT', 3)
    monkeypatch.setattr(d, 'DEV_COUNT', 3)
    monkeypatch.setattr(d, 'DEV_MEMBERSHIP_SHA256', d.sha(d.jsonl(d.membership(eligible_dev))))
    monkeypatch.setattr(d, 'SOURCE_COUNTS', {rd: {'train': 60, 'dev': 2, 'test': 1} for rd in d.ROUNDS})
    monkeypatch.setattr(d, 'FIT_SIZES', (9, 18, 27))
    monkeypatch.setattr(d, 'SELECTION_PER_STRATUM', 1)
    monkeypatch.setattr(d, 'HOLDOUT_PER_STRATUM', 1)
    return source, history, tmp_path/'output'


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source, history, output = fixture_source(tmp_path, monkeypatch)
    d.prepare(source, history, output)
    return source, history, output, d.load_prepared(output)


def test_exact_disjoint_nested_allocation_is_deterministic_and_excludes_test_history(prepared):
    source, history, output, package = prepared
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    d.prepare(source, history, output)
    assert before == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    examples = package['examples']
    role_groups = {role: {r['premise_hash'] for r in examples if r['split'] == role} for role in d.SPLITS}
    for a in d.SPLITS:
        for b in d.SPLITS:
            if a != b:
                assert not role_groups[a] & role_groups[b]
    src = d.read_source(source)
    historical = {r['premise_hash'] for r in src.values() if r['sample_id'] in d.history(history, src)}
    official_test = {r['premise_hash'] for r in src.values() if r['source_split'] == 'test'}
    assert not historical & (role_groups['train_holdout'] | role_groups['eligible_dev'])
    assert not official_test & set.union(*role_groups.values())
    assert {r['sample_id'] for r in examples if r['split'] in {'fitting', 'selection'}} == set(package['fit_gold'])
    ids = package['fit_ids_by_size']
    assert len(ids['9']) == 9 and len(ids['18']) == 18 and len(ids['27']) == 27
    assert set(ids['9']) < set(ids['18']) < set(ids['27'])
    assert not (output/'sealed').exists()


def test_preparation_projects_away_invalid_dev_gold_and_only_final_helper_reads_it(tmp_path, monkeypatch):
    source, history, output = fixture_source(tmp_path, monkeypatch, bad_dev_gold=True)
    manifest = d.prepare(source, history, output)
    package = d.load_prepared(output)
    assert all('gold' not in r for r in package['examples'])
    assert not any('not-a-label' in p.read_text() for p in output.rglob('*') if p.is_file())
    with pytest.raises(d.DataError, match='Evaluation gold'):
        d.evaluation_gold(source, manifest, package['examples'])


def test_evaluation_gold_only_covers_the_two_frozen_holdouts(prepared):
    source, _, _, package = prepared
    gold = d.evaluation_gold(source, package['manifest'], package['examples'])
    wanted = {r['sample_id'] for r in package['examples'] if r['split'] in {'eligible_dev', 'train_holdout'}}
    assert set(gold) == wanted
    assert not set(gold) & set(package['fit_gold'])
    changed = copy.deepcopy(package['examples'])
    changed[0]['state'] += ' changed'
    with pytest.raises(d.DataError, match='input identity'):
        d.evaluation_gold(source, package['manifest'], changed)


def test_pinned_source_and_historical_input_identity_are_enforced(prepared, monkeypatch):
    source, history, output, _ = prepared
    monkeypatch.setattr(d, 'SOURCE_SHA256', '0'*64)
    with pytest.raises(d.DataError, match='source SHA'):
        d.prepare(source, history, output.parent/'bad-source-output')
    monkeypatch.setattr(d, 'SOURCE_SHA256', d.file_hash(source))
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    rows[0]['state'] += ' changed content'
    rows[0]['premise_hash'], rows[0]['input_hash'] = d.hashes(*d.text_pair(rows[0]))
    history.write_bytes(d.jsonl(rows))
    monkeypatch.setattr(d, 'HISTORY_SHA256', d.file_hash(history))
    with pytest.raises(d.DataError, match='source input identity'):
        d.prepare(source, history, output.parent/'changed-history-output')


def test_infeasible_capacity_fails_without_changing_recipe_or_writing(prepared, monkeypatch, tmp_path):
    source, history, _, _ = prepared
    monkeypatch.setattr(d, 'FIT_SIZES', (9, 18, 900))
    fresh = tmp_path/'failed-output'
    with pytest.raises(d.DataError, match='infeasible'):
        d.prepare(source, history, fresh)
    assert not fresh.exists()


def test_prepare_never_overwrites_changed_evidence(prepared):
    source, history, output, _ = prepared
    target = output/'data/fit_gold.json'
    target.write_text('preserve this unexpected edit')
    with pytest.raises(d.DataError, match='Existing prepared evidence differs'):
        d.prepare(source, history, output)
    assert target.read_text() == 'preserve this unexpected edit'


@pytest.mark.parametrize('what', ['extra_gold', 'nonnested', 'balanced_but_not_nested', 'premise_leak', 'wrong_quota', 'dev_membership'])
def test_semantic_validation_rejects_rehashed_tampering(prepared, what):
    _, _, _, package = prepared
    p = copy.deepcopy(package)
    rows, gold, nested, manifest = p['examples'], p['fit_gold'], p['fit_ids_by_size'], p['manifest']
    if what == 'extra_gold':
        target = next(r for r in rows if r['split'] == 'eligible_dev')
        gold[target['sample_id']] = 'neutral'
    elif what == 'nonnested':
        nested['9'][0] = nested['9'][1]
    elif what == 'balanced_but_not_nested':
        lookup = {r['sample_id']: r for r in rows}
        original = nested['9'][0]
        wanted = (lookup[original]['round'], gold[original])
        replacement = next(sid for sid in nested['27'] if sid not in nested['18'] and (lookup[sid]['round'], gold[sid]) == wanted)
        nested['9'][0] = replacement
    elif what == 'wrong_quota':
        gold[nested['9'][0]] = next(label for label in d.LABELS if label != gold[nested['9'][0]])
    elif what == 'dev_membership':
        target = next(r for r in rows if r['split'] == 'eligible_dev')
        target['sample_id'] += '-changed'
        manifest['roles']['eligible_dev']['membership_sha256'] = d.sha(d.jsonl(d.membership([r for r in rows if r['split'] == 'eligible_dev'])))
    else:
        a = next(r for r in rows if r['split'] == 'fitting')
        b = next(r for r in rows if r['split'] == 'selection')
        premise, _ = d.text_pair(a)
        _, hypothesis = d.text_pair(b)
        ph, ih = d.hashes(premise, hypothesis)
        b.update(state='Premise:\n'+premise+d.DELIMITER+hypothesis, premise_hash=ph, input_hash=ih, group_id='anli:premise:'+ph)
        selected = [r for r in rows if r['split'] == 'selection']
        manifest['roles']['selection'].update(premise_groups=len({r['premise_hash'] for r in selected}), membership_sha256=d.sha(d.jsonl(d.membership(selected))))
    with pytest.raises(d.DataError):
        d.validate(rows, gold, nested, manifest)


def test_prepared_hashes_and_symlink_boundaries(prepared, tmp_path):
    _, _, output, _ = prepared
    (output/'data/examples.jsonl').write_text('changed')
    with pytest.raises(d.DataError, match='hash/size'):
        d.load_prepared(output)
    outside = tmp_path/'outside'
    outside.mkdir()
    alias = tmp_path/'alias'
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(d.DataError, match='Symlink'):
        d.safe(alias, 'data/manifest.json')


def test_acquire_source_uses_verified_local_archive_and_never_overwrites(prepared, tmp_path, monkeypatch):
    source, _, _, _ = prepared
    monkeypatch.setattr(d, 'download_source', lambda *a, **k: pytest.fail('download despite valid local source'))
    target = tmp_path/'copied/source.zip'
    assert d.acquire_source(target, [source]) == target
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b'changed')
    with pytest.raises(d.DataError, match='Existing ANLI source'):
        d.acquire_source(target, [source])
    assert target.read_bytes() == b'changed'


def test_conflicting_training_input_is_removed_before_allocation(prepared):
    source_path, history_path, _, _ = prepared
    source = d.read_source(source_path)
    used = d.history(history_path, source)
    original = source['anli:R1-train-c-019']
    other = {**original, 'sample_id': 'anli:contradictory-copy', 'gold': 'entailment'}
    source[other['sample_id']] = other
    examples, _, _, audit = d.build(source, used)
    assert audit['training_cleanup']['conflicting_rows_excluded'] == 2
    assert not any(r['input_hash'] == original['input_hash'] for r in examples)


def test_download_has_an_absolute_subprocess_watchdog(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        assert command[:3] == [d.sys.executable, '-I', '-c']
        assert kwargs['timeout'] == 330 and kwargs['check'] is True
        raise d.subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(d.subprocess, 'run', timeout)
    with pytest.raises(d.DataError, match='absolute watchdog'):
        d.download_source(tmp_path/'partial.zip')
