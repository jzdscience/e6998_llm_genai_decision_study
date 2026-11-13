"""Independent small fixtures for the input-only exposure boundary."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import warnings
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("audit_anli_exposure", Path(__file__).parents[1] / "scripts/audit_anli_exposure.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

SID = "anli:11111111-1111-1111-1111-111111111111"
SID2 = "anli:22222222-2222-2222-2222-222222222222"
UNKNOWN = "anli:99999999-9999-9999-9999-999999999999"


def source_row(sid=SID, premise="Ａ  premise", hypothesis="A HYPOTHESIS", split="train"):
    ph, ih = audit.hashes(premise, hypothesis)
    return {"sample_id": sid, "round": "R1", "source_split": split, "premise_hash": ph, "input_hash": ih}


def historical_row(sid=SID):
    expected = source_row(sid)
    return {**expected, "task": "anli", "split": "calibration", "state": "Premise:\nA premise\n\nHypothesis:\na hypothesis"}


def payload(row):
    return (json.dumps(row) + "\n").encode()


def test_source_projects_inputs_without_gold_or_auxiliary_fields(tmp_path):
    path = tmp_path / "source.zip"
    row = {"uid": SID[5:], "context": "Ａ  premise", "hypothesis": "A HYPOTHESIS",
           "label": {"not_a_valid_label": True}, "reason": "MUST NOT ENTER OUTPUT", "model_label": "unused"}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("anli_v1.0/R1/dev.jsonl", payload(row))
    result, members = audit.read_source(path, audit.file_hash(path), {"R1": {"dev": 1}})
    assert result == {SID: source_row(split="dev")}
    assert members["anli_v1.0/R1/dev.jsonl"]["count"] == 1
    assert "label" not in json.dumps(result) and "MUST NOT ENTER" not in json.dumps(result)


def test_source_archive_pin_and_expected_counts_are_enforced(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("anli_v1.0/R1/dev.jsonl", payload({"uid": SID[5:], "context": "P", "hypothesis": "H"}))
    with pytest.raises(audit.AuditError, match="SHA"):
        audit.read_source(path, "0" * 64, {"R1": {"dev": 1}})
    with pytest.raises(audit.AuditError, match="count"):
        audit.read_source(path, audit.file_hash(path), {"R1": {"dev": 2}})


@pytest.mark.parametrize("names", [["a.json", "a.json"], ["../a.json"], ["/a.json"], ["a\\b.json"]])
def test_zip_duplicates_and_unsafe_paths_rejected(names):
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(stream, "w") as z:
            for name in names:
                z.writestr(name, "{}")
    with zipfile.ZipFile(stream) as z, pytest.raises(audit.AuditError):
        audit.zip_members(z)


def test_historical_input_is_recomputed_not_trusted():
    source = {SID: source_row()}
    row = historical_row()
    assert audit.verify_input_rows(payload(row), source, "v3") == {SID}
    row["state"] += " changed"
    # An attacker can rewrite the serialized hashes too; source identity must
    # independently reject this internally consistent but substituted input.
    row["premise_hash"], row["input_hash"] = audit.hashes("A premise", "a hypothesis changed")
    with pytest.raises(audit.AuditError, match="differs from pinned source"):
        audit.verify_input_rows(payload(row), source, "v3")


@pytest.mark.parametrize("mutation,match", [
    ({"sample_id": UNKNOWN}, "Unknown historical"),
    ({"input_hash": "0" * 64}, "serialized"),
    ({"state": "missing delimiter"}, "schema"),
    ({"task": "squad"}, "task"),
])
def test_unknown_or_corrupt_historical_rows_fail_closed(mutation, match):
    row = {**historical_row(), **mutation}
    with pytest.raises(audit.AuditError, match=match):
        audit.verify_input_rows(payload(row), {SID: source_row()}, "v3")


def test_unknown_ledger_schema_and_duplicate_ids_rejected():
    with pytest.raises(audit.AuditError, match="ledger input schema"):
        audit.verify_input_rows(payload(historical_row()), {SID: source_row()}, "unexpected")
    with pytest.raises(audit.AuditError, match="Duplicate historical"):
        audit.verify_input_rows(payload(historical_row()) * 2, {SID: source_row()}, "v3")


def test_unknown_scientific_uuid_not_ignored_but_generic_uuid_is_not_a_sample():
    source = {SID: source_row()}
    with pytest.raises(audit.AuditError, match="Unknown scientific"):
        audit.census_ids(payload({"sample_id": UNKNOWN}), source, "predictions.jsonl")
    assert audit.census_ids(payload({"request_id": UNKNOWN[5:]}), source, "log.json") == set()
    assert audit.census_ids(SID[5:].encode(), source, "notes.md") == {SID}


def test_unknown_gold_map_key_is_an_unresolved_scientific_reference():
    with pytest.raises(audit.AuditError, match="Unknown scientific ANLI reference"):
        audit.census_ids(payload({UNKNOWN: "ignored label"}), {SID: source_row()}, "old_gold.json", scientific=True)


def test_bounded_input_read_accepts_exact_limit_but_rejects_one_extra_byte(tmp_path, monkeypatch):
    path = tmp_path / "input.json"; monkeypatch.setattr(audit, "LIMIT", 8)
    path.write_bytes(b"12345678"); assert audit.read_bounded(path) == b"12345678"
    path.write_bytes(b"123456789")
    with pytest.raises(audit.AuditError, match="audit bound"):
        audit.read_bounded(path)


def test_pruned_subtrees_are_never_traversed(tmp_path, monkeypatch):
    forbidden = []
    for name in ("cache", "weights", "envs", ".venv"):
        p = tmp_path / name
        p.mkdir()
        (p / "bad.jsonl").write_bytes(payload({"sample_id": UNKNOWN}))
        forbidden.append(p)
    (tmp_path / "keep.jsonl").write_bytes(payload(historical_row()))
    (tmp_path / "fit_gold.json").write_text("{}")
    original = audit.os.scandir
    def guarded(path):
        assert Path(path) not in forbidden, "pruned subtree was traversed"
        return original(path)
    monkeypatch.setattr(audit.os, "scandir", guarded)
    assert [p.name for p in audit.iter_files(tmp_path)] == ["fit_gold.json", "keep.jsonl"]


def test_historical_gold_keys_and_unresolved_symlinks_are_not_silently_ignored(tmp_path):
    sealed = tmp_path / "sealed"; sealed.mkdir()
    gold = sealed / "holdout_gold.json"; gold.write_text(json.dumps({SID: "never used"}))
    link = tmp_path / "history-link"; link.symlink_to(sealed, target_is_directory=True)
    skipped = []
    paths = list(audit.iter_files(tmp_path, skipped_symlinks=skipped))
    assert gold in paths and str(link) in skipped
    assert audit.census_ids(gold.read_bytes(), {SID: source_row()}, str(gold)) == {SID}


def test_ledger_pin_changed_fails_before_source_read(tmp_path):
    repo, evidence = tmp_path / "repo", tmp_path / "evidence"
    repo.mkdir(); evidence.mkdir()
    ledger = repo / "ledger.json"
    ledger.write_text('{"schema_version": 1}')
    with pytest.raises(audit.AuditError, match="ledger differs"):
        audit.audit(repo, evidence, ledger, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_safe_relative_refuses_escape_and_external_symlink(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    external = tmp_path / "external"; external.mkdir()
    (root / "link").symlink_to(external)
    for value in ("../outside", "/outside", "link/file"):
        with pytest.raises(audit.AuditError):
            audit.safe_relative(root, value)


def test_git_history_census_includes_deleted_inputs(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    p = tmp_path / "inputs.jsonl"; p.write_bytes(payload(historical_row()))
    subprocess.run(["git", "-C", str(tmp_path), "add", "inputs.jsonl"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    p.unlink()
    blobs = list(audit.git_blobs(tmp_path))
    assert len(blobs) == 1 and blobs[0][1] == "inputs.jsonl"
    assert audit.census_ids(blobs[0][2], {SID: source_row()}, blobs[0][1]) == {SID}


def test_repeat_audit_does_not_turn_candidate_manifest_into_historical_use(tmp_path, monkeypatch):
    repo, evidence = tmp_path / "repo", tmp_path / "evidence"
    repo.mkdir(); evidence.mkdir()
    inputs = evidence / "inputs.jsonl"; inputs.write_bytes(payload(historical_row()))
    source = {SID: source_row(), SID2: source_row(SID2, "untouched premise", "untouched hypothesis", "dev")}
    ledger_data = {"schema_version": 1, "source": {"root": "evidence", "path": "source.bin"},
                   "historical_inputs": [{"root": "evidence", "path": "inputs.jsonl", "sha256": audit.file_hash(inputs),
                                          "schema": "v3", "count": 1, "use": "fixture"}], "archives": [],
                   "scan_exclusions": [{"root": "repo", "path": "audit-only", "reason": "fixture audit only"}]}
    ledger = repo / "ledger.json"; ledger.write_text(json.dumps(ledger_data))
    monkeypatch.setattr(audit, "LEDGER_SHA256", audit.file_hash(ledger))
    monkeypatch.setattr(audit, "read_source", lambda path: (source, {}))
    with zipfile.ZipFile(evidence / "extra-history.zip", "w") as archive:
        archive.writestr("history.json", json.dumps({"sample_id": SID}))
    original_read = Path.read_bytes
    def no_whole_zip_read(path):
        assert path.suffix != '.zip', 'whole unknown ZIP was loaded into memory'
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', no_whole_zip_read)
    first = audit.audit(repo, evidence, ledger, repo / "audit-only/run1")
    second = audit.audit(repo, evidence, ledger, repo / "audit-only/run2")
    assert first["historical"]["count"] == second["historical"]["count"] == 1
    assert first["candidate_dev"]["count"] == second["candidate_dev"]["count"] == 1
    assert second['scan_exclusions'][0]['classification'] == 'audit_only_not_historical_model_use'
    with pytest.raises(audit.AuditError, match="reviewed audit-only"):
        audit.audit(repo, evidence, ledger, repo / "unreviewed-output")
