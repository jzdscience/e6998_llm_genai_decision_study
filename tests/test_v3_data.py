"""Artificial software fixtures; no model calls or evaluation of ANLI labels."""
import json
from pathlib import Path

import pytest

from decision_study.common import canonical, fingerprint
from decision_study.v3 import data


def configuration():
    return {"protocol": data.PROTOCOL, "seed": 6998,
            "data": {"task": "anli", "duplicates": data.DUPLICATE_RULE,
                     "source_url": data.SOURCE_URL, "source_sha256": data.SOURCE_SHA256,
                     "rounds": list(data.ROUNDS), "labels": list(data.LABELS),
                     "calibration_per_round_class": 1, "validation_per_round_class": 1,
                     "development_per_round_class": 1,
                     "training_role_proportions": {"calibration": .6, "validation": .3, "development": .1}}}


def artificial_rows():
    rows = []
    for round_name in data.ROUNDS:
        for label in data.LABELS:
            for i in range(100):
                rows.append({"uid": f"train-{round_name}-{label}-{i}", "round": round_name,
                             "source_split": "train", "label": label,
                             "premise": f"ARTIFICIAL training premise {round_name} {label} {i}.",
                             "hypothesis": f"ARTIFICIAL hypothesis {i}."})
            # A premise intentionally shared between test rounds must keep one
            # group_id, while hypotheses and UIDs remain distinct.
            rows.append({"uid": f"test-{round_name}-{label}", "round": round_name,
                         "source_split": "test", "label": label,
                         "premise": f"ARTIFICIAL official test premise {label}.",
                         "hypothesis": f"ARTIFICIAL test hypothesis {round_name} {label}."})
            rows.append({"uid": f"dev-{round_name}-{label}", "round": round_name,
                         "source_split": "dev", "label": label,
                         "premise": f"ARTIFICIAL official dev premise {round_name} {label}.",
                         "hypothesis": "Unused official development row."})
    return rows


def publish_fixture(root, rows=None):
    config = configuration()
    examples, fit_gold, holdout_gold, audit = data._build(rows or artificial_rows(), config)
    payloads = data._payloads(examples, fit_gold, holdout_gold)
    manifest = data._manifest(examples, payloads, config, "fixture-protocol-file", {}, audit,
                              evidence_type="software_test_fixture")
    data._publish(root, payloads, manifest)
    return examples, fit_gold, holdout_gold, manifest


@pytest.fixture
def prepared(tmp_path):
    return tmp_path, publish_fixture(tmp_path)


def reseal(root, examples=None, fit_gold=None):
    manifest = json.loads((root / data.MANIFEST).read_text())
    examples = examples or [json.loads(line) for line in (root / data.INPUTS).read_text().splitlines()]
    fit_gold = fit_gold or json.loads((root / data.FIT_GOLD).read_text())
    payloads = {data.INPUTS: b"".join(canonical(row) + b"\n" for row in examples),
                data.FIT_GOLD: canonical(fit_gold) + b"\n"}
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
        manifest["files"][name] = data._sha(payload)
    for role in manifest["roles"]:
        selected = data._select_role(examples, role)
        manifest["roles"][role] = {"count": len(selected), "sample_ids_hash": fingerprint([x["sample_id"] for x in selected]),
                                   "group_ids_hash": fingerprint(sorted({x["group_id"] for x in selected})),
                                   "inputs_hash": fingerprint(selected)}
    manifest["counts"] = {role: len(data._select_role(examples, role)) for role in data.SPLITS}
    manifest["manifest_hash"] = fingerprint({k: v for k, v in manifest.items() if k != "manifest_hash"})
    (root / data.MANIFEST).write_bytes(canonical(manifest) + b"\n")


def authorize(root, manifest):
    freeze = {"schema_version": 1, "protocol": data.PROTOCOL,
              "data_manifest_hash": fingerprint(manifest),
              "protocol_hash": manifest["protocol_identity"]["canonical_sha256"],
              "created_at": "2026-11-03T01:00:00+00:00"}
    freeze["freeze_hash"] = fingerprint(freeze)
    (root / "policy_freeze.json").write_bytes(canonical(freeze))
    access = {key: freeze[key] for key in ("freeze_hash", "data_manifest_hash", "protocol_hash")}
    access["created_at"] = "2026-11-03T01:01:00+00:00"
    (root / "holdout_access.json").write_bytes(canonical(access))
    return freeze, access


def test_selection_is_reproducible_disjoint_and_keeps_every_official_test():
    rows, config = artificial_rows(), configuration()
    left = data._build(rows, config)
    assert left == data._build(list(reversed(rows)), config)
    examples, fit_gold, gold, audit = left
    assert len(examples) == 36 and len(fit_gold) == 27 and len(gold) == 9
    assert set(gold) == {"anli:" + row["uid"] for row in rows if row["source_split"] == "test"}
    group_sets = {role: {x["group_id"] for x in examples if x["split"] == role} for role in data.SPLITS}
    for a in data.SPLITS:
        for b in data.SPLITS:
            if a != b:
                assert not group_sets[a] & group_sets[b]
    assert len(group_sets["holdout"]) == 3  # shared across R1/R2/R3
    assert audit["official_test_preserved_count"] == 9
    assert all(x["source_split"] != "dev" for x in examples)


def test_training_overlap_and_duplicate_resolution_are_explicit():
    rows = artificial_rows()
    test = next(row for row in rows if row["source_split"] == "test")
    overlap = {**test, "uid": "overlap-with-test", "source_split": "train", "hypothesis": "Other inference"}
    same = {**rows[0], "uid": "duplicate-z"}
    conflict_a = {**rows[1], "uid": "conflicting-a", "premise": "ARTIFICIAL conflicting input."}
    conflict_b = {**conflict_a, "uid": "conflicting-b", "label": "neutral"}
    _, _, _, audit = data._build(rows + [overlap, same, conflict_a, conflict_b], configuration())
    assert audit["training_exclusion_counts"] == {"test_premise_or_input_overlap": 1,
                                                   "duplicate_training_input": 1,
                                                   "conflicting_training_input": 2}
    # "duplicate-z" sorts before the original train UID, so the original is
    # the audited discarded record, independent of input order.
    assert audit["training_excluded_ids_hashes"]["duplicate_training_input"] == fingerprint([rows[0]["uid"]])


def test_duplicate_uid_and_duplicate_official_test_are_rejected():
    rows = artificial_rows()
    with pytest.raises(data.DataError, match="Duplicate UID"):
        data._build(rows + [dict(rows[0])], configuration())
    test = next(row for row in rows if row["source_split"] == "test")
    with pytest.raises(data.DataError, match="never drop test"):
        data._build(rows + [{**test, "uid": "new-test-uid"}], configuration())


def test_insufficient_role_pool_never_replenishes_from_another_role():
    rows = [row for row in artificial_rows() if row["source_split"] != "train" or data._role(6998, data._text_hash(row["premise"])) != "development"]
    with pytest.raises(data.DataError, match="Insufficient development.*no pool replenishment"):
        data._build(rows, configuration())


def test_inputs_whitelist_ignores_adversarial_metadata():
    rows = artificial_rows()
    for row in rows:
        row.update(reason="SECRET GOLD EXPLANATION", model_label="entailment", emturk="author")
    examples, *_ = data._build(rows, configuration())
    assert all(set(row) == data.EXAMPLE_KEYS for row in examples)
    assert "SECRET GOLD" not in json.dumps(examples)


def test_fit_and_default_verification_never_open_sealed_gold(prepared, monkeypatch):
    root, (_, fit_gold, _, manifest) = prepared
    (root / data.HOLDOUT_GOLD).unlink()
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        if str(path).endswith(data.HOLDOUT_GOLD):
            raise AssertionError("Fit access attempted sealed gold")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert data.verify_fit(root) == data.verify(root) == manifest
    assert len(data.load_examples(root, "fit")) == 18
    assert len(data.load_examples(root, "non_holdout")) == 27
    assert len(data.load_examples(root, "development")) == 9
    assert data.load_fit_gold(root) == fit_gold


def test_resealed_gold_leak_into_inputs_is_rejected(prepared):
    root, (examples, *_rest) = prepared
    examples[0]["reason"] = "Gold-bearing explanation"
    reseal(root, examples=examples)
    with pytest.raises(data.DataError, match="gold-leaking"):
        data.verify_fit(root)


def test_resealed_holdout_label_in_fit_gold_is_rejected(prepared):
    root, (examples, fit_gold, holdout_gold, _) = prepared
    fit_gold.update({next(iter(holdout_gold)): "neutral"})
    reseal(root, fit_gold=fit_gold)
    with pytest.raises(data.DataError, match="exactly non-holdout"):
        data.verify_fit(root)


def test_resealed_cross_role_premise_is_rejected(prepared):
    root, (examples, fit_gold, _, _) = prepared
    # Build an alternate heldout fixture premise assigned to calibration so a
    # role-hash check cannot mask deletion of the overlap check.
    i = 0
    while data._role(6998, data._text_hash(f"ARTIFICIAL overlap premise {i}")) != "calibration":
        i += 1
    premise = f"ARTIFICIAL overlap premise {i}"
    for split in ("calibration", "holdout"):
        row = next(x for x in examples if x["split"] == split)
        replacement = data._example({"uid": row["sample_id"].removeprefix("anli:"), "round": row["round"],
                                     "source_split": row["source_split"], "premise": premise,
                                     "hypothesis": "Different artificial hypothesis " + split}, split)
        row.clear()
        row.update(replacement)
    reseal(root, examples=examples)
    with pytest.raises(data.DataError, match="Premise overlap"):
        data.verify_fit(root)


@pytest.mark.parametrize("which", ["bytes", "manifest", "preparation_source"])
def test_data_or_provenance_tampering_is_rejected(prepared, which):
    root, _ = prepared
    if which == "bytes":
        with (root / data.INPUTS).open("ab") as stream:
            stream.write(b"\n")
    else:
        manifest = json.loads((root / data.MANIFEST).read_text())
        manifest["preparation_source_sha256"] = "0" * 64
        if which == "preparation_source":
            manifest["manifest_hash"] = fingerprint({k: v for k, v in manifest.items() if k != "manifest_hash"})
        (root / data.MANIFEST).write_bytes(canonical(manifest))
    with pytest.raises(data.DataError):
        data.verify_fit(root)


def test_prepare_refuses_unpinned_archive_without_publishing(tmp_path):
    source = tmp_path / "fake.zip"
    source.write_bytes(b"not the official source")
    config = tmp_path / "protocol.json"
    config.write_bytes(canonical(configuration()))
    output = tmp_path / "output"
    with pytest.raises(data.DataError, match="archive SHA256"):
        data.prepare(source, output, protocol_path=config)
    assert not output.exists()


def test_publishing_is_idempotent_and_refuses_changed_existing_data(prepared):
    root, (examples, fit_gold, gold, manifest) = prepared
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    data._publish(root, data._payloads(examples, fit_gold, gold), manifest)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
    with (root / data.INPUTS).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(data.DataError, match="Existing ANLI data differs"):
        data._publish(root, data._payloads(examples, fit_gold, gold), manifest)


def test_parent_symlink_is_rejected(prepared):
    root, _ = prepared
    target = root.parent / (root.name + "-actual-data")
    (root / "data").rename(target)
    (root / "data").symlink_to(target, target_is_directory=True)
    with pytest.raises(data.DataError, match="Symlink"):
        data.verify_fit(root)


def test_gold_requires_existing_matching_freeze_and_first_access(prepared, monkeypatch):
    root, (_, _, gold, manifest) = prepared
    original_open = Path.open

    def guarded(path, *args, **kwargs):
        if str(path).endswith(data.HOLDOUT_GOLD):
            raise AssertionError("Sealed gold read before gate")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", guarded)
        with pytest.raises(data.DataError, match="freeze identity"):
            data.verify(root, include_sealed=True)
        with pytest.raises(data.DataError, match="Missing.*policy_freeze"):
            data.load_holdout_gold_for_evaluation(root, expected_freeze_hash="unknown")
        freeze, _ = authorize(root, manifest)
        (root / "holdout_access.json").unlink()
        with pytest.raises(data.DataError, match="Missing.*holdout_access"):
            data.load_holdout_gold_for_evaluation(root, expected_freeze_hash=freeze["freeze_hash"])
    freeze, _ = authorize(root, manifest)
    assert data.load_holdout_gold_for_evaluation(root, expected_freeze_hash=freeze["freeze_hash"]) == gold


@pytest.mark.parametrize("change", ["freeze_bytes", "wrong_data", "access_policy", "access_time", "sealed_bytes"])
def test_evaluation_gate_rejects_mutation(prepared, change):
    root, (_, _, _, manifest) = prepared
    freeze, access = authorize(root, manifest)
    if change == "freeze_bytes":
        freeze["policies"] = {"changed": "without updating the freeze hash"}
        (root / "policy_freeze.json").write_bytes(canonical(freeze))
    elif change == "wrong_data":
        freeze["data_manifest_hash"] = "0" * 64
        freeze["freeze_hash"] = fingerprint({k: v for k, v in freeze.items() if k != "freeze_hash"})
        (root / "policy_freeze.json").write_bytes(canonical(freeze))
        access.update({key: freeze[key] for key in ("freeze_hash", "data_manifest_hash", "protocol_hash")})
        (root / "holdout_access.json").write_bytes(canonical(access))
    elif change == "access_policy":
        access["freeze_hash"] = "another-policy"
        (root / "holdout_access.json").write_bytes(canonical(access))
    elif change == "access_time":
        access["created_at"] = "2026-11-02T01:00:00+00:00"
        (root / "holdout_access.json").write_bytes(canonical(access))
    else:
        with (root / data.HOLDOUT_GOLD).open("ab") as stream:
            stream.write(b"\n")
    with pytest.raises(data.DataError):
        data.load_holdout_gold_for_evaluation(root, expected_freeze_hash=freeze["freeze_hash"])
