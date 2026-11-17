"""Supplement tests use real small .npy files and frozen receipt layouts."""
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/archive_learnability_embeddings.py"
SPEC = importlib.util.spec_from_file_location("embedding_capture", SCRIPT)
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.canonical(value) + b"\n")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source, output = tmp_path / "history", tmp_path / "new" / "exports"
    source.mkdir()
    monkeypatch.setattr(a, "EXPECTED_ARRAYS", 2)
    monkeypatch.setattr(a, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(a, "CHUNK_ROWS", 2)
    monkeypatch.setattr(a, "PART_BYTES", 1024)
    identity = {"encoding": {"dimension": 1536}, "inputs_hash": "fixed-input-identity"}
    identity_hash = a.digest(a.canonical(identity))
    write_json(source / (a.PREFIX + "identity.json"), identity)
    write_json(source / (a.PREFIX + "complete.json"),
               {"count": 3, "dimension": 1536, "identity_hash": identity_hash})
    for start in (0, 2):
        folder = source / (a.PREFIX + f"chunks/{start:08d}")
        folder.mkdir(parents=True)
        count = min(2, 3 - start)
        array = folder / f"vectors-{start + 100}.npy"
        with array.open("wb") as stream:
            matrix = np.random.default_rng(start).normal(size=(count, 1536)).astype(np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            np.save(stream, matrix, allow_pickle=False)
        write_json(folder / "receipt.json", {"count": count, "identity_hash": identity_hash,
                   "ids": [f"sample-{i}" for i in range(start, start + count)], "sha256": a.file_digest(array),
                   "start": start, "vectors": array.name, "coverage": {"field_tokens": [1] * (2 * count)}})
    files = {str(p.relative_to(source)): a.file_digest(p) for p in source.rglob("*") if p.is_file()}
    write_json(source / "freeze.json", {"files": files, "models_and_scores_frozen_before_evaluation_gold": True})
    monkeypatch.setattr(a, "PINNED_FREEZE_SHA256", a.file_digest(source / "freeze.json"))
    return source, output


def test_archives_verified_arrays_in_bounded_parts_and_reuses_identical_capture(fixture):
    source, output = fixture
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    receipt = a.capture(source, output)
    assert receipt["status"] == "PASS_ARCHIVED_ONLY" and receipt["full_refit_performed"] is False
    manifest = a.verify_capture(output)
    assert len(manifest["parts"]) > 1
    assert all((output / p["name"]).stat().st_size <= 1024 for p in manifest["parts"])
    first = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in output.iterdir()}
    assert a.capture(source, output) == receipt
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in output.iterdir()} == first
    joined = output.parent / "joined.zip"
    joined.write_bytes(b"".join((output / p["name"]).read_bytes() for p in manifest["parts"]))
    with zipfile.ZipFile(joined) as z:
        assert set(z.namelist()) == set(before)
        assert all(z.read(name) == value for name, value in before.items())
    assert {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_interrupted_capture_reuses_existing_equal_parts(fixture):
    source, output = fixture
    a.capture(source, output)
    manifest = a.verify_capture(output)
    first = output / manifest["parts"][0]["name"]
    old = first.stat().st_mtime_ns
    for p in output.iterdir():
        if p != first:
            p.unlink()
    a.capture(source, output)
    assert first.stat().st_mtime_ns == old
    assert a.verify_capture(output)["array_count"] == 2


def test_changed_array_is_incomplete_and_never_claims_refit(fixture, capsys):
    source, output = fixture
    array = next(source.rglob("*.npy"))
    array.write_bytes(array.read_bytes()[:-1] + b"x")
    assert a.main(["--source-dir", str(source), "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "INCOMPLETE" and result["full_refit_performed"] is False
    assert "changed" in result["error"]
    assert len(list(output.glob("incomplete-*.json"))) == 1
    assert not (output / "capture-receipt.json").exists()
    assert not list(output.glob("*.part*"))


def test_missing_array_returns_nonzero_in_strict_mode(fixture, capsys):
    source, output = fixture
    next(source.rglob("*.npy")).unlink()
    assert a.main(["--source-dir", str(source), "--output-dir", str(output), "--strict"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INCOMPLETE"


def test_freeze_cannot_be_rewritten_to_accept_tampering(fixture):
    source, output = fixture
    freeze_path = source / "freeze.json"
    data = json.loads(freeze_path.read_text())
    name = next(n for n in data["files"] if n.endswith(".npy"))
    (source / name).write_bytes(b"changed")
    data["files"][name] = a.file_digest(source / name)
    write_json(freeze_path, data)
    with pytest.raises(a.CaptureError, match="accepted SHA"):
        a.capture(source, output)


@pytest.mark.parametrize("directory", [False, True])
def test_symlinked_source_members_are_refused_even_with_identical_bytes(fixture, directory):
    source, output = fixture
    if directory:
        original = source / "collection/embeddings/chunks/00000000"
    else:
        original = next(source.rglob("*.npy"))
    outside = output.parent / "outside"
    outside.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(original, outside)
    original.symlink_to(outside, target_is_directory=directory)
    with pytest.raises(a.CaptureError, match="Symlinked"):
        a.capture(source, output)


def test_receipt_path_escape_is_refused_even_when_fixture_freeze_rebound(fixture, monkeypatch):
    source, output = fixture
    receipt = next(source.rglob("receipt.json"))
    data = json.loads(receipt.read_text())
    data["vectors"] = "../../outside.npy"
    write_json(receipt, data)
    freeze = json.loads((source / "freeze.json").read_text())
    freeze["files"][str(receipt.relative_to(source))] = a.file_digest(receipt)
    write_json(source / "freeze.json", freeze)
    monkeypatch.setattr(a, "PINNED_FREEZE_SHA256", a.file_digest(source / "freeze.json"))
    with pytest.raises(a.CaptureError, match="Unsafe receipt"):
        a.capture(source, output)


def test_missing_array_in_rebound_inventory_is_refused(fixture, monkeypatch):
    source, output = fixture
    freeze = json.loads((source / "freeze.json").read_text())
    del freeze["files"][next(n for n in freeze["files"] if n.endswith(".npy"))]
    write_json(source / "freeze.json", freeze)
    monkeypatch.setattr(a, "PINNED_FREEZE_SHA256", a.file_digest(source / "freeze.json"))
    with pytest.raises(a.CaptureError, match="count differs"):
        a.capture(source, output)


def test_existing_different_part_is_preserved_not_overwritten(fixture):
    source, output = fixture
    output.mkdir(parents=True)
    part = output / (a.ARCHIVE + ".part0001")
    part.write_bytes(b"unrelated-existing-file")
    with pytest.raises(a.CaptureError, match="Existing capture file differs"):
        a.capture(source, output)
    assert part.read_bytes() == b"unrelated-existing-file"
    assert not (output / "manifest.json").exists()


def test_capture_output_may_not_overlap_history(fixture):
    source, _ = fixture
    with pytest.raises(a.CaptureError, match="overlap"):
        a.capture(source, source / "exports")
    assert not (source / "exports").exists()


def test_corrupt_part_and_extra_part_fail_verification(fixture):
    source, output = fixture
    a.capture(source, output)
    part = output / (a.ARCHIVE + ".part0001")
    original = part.read_bytes()
    part.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(a.CaptureError, match="part changed"):
        a.verify_capture(output)
    part.write_bytes(original)
    (output / (a.ARCHIVE + ".part9999")).write_bytes(b"extra")
    with pytest.raises(a.CaptureError, match="Unlisted"):
        a.verify_capture(output)


def test_rewritten_manifest_cannot_rebind_changed_archive(fixture):
    source, output = fixture
    a.capture(source, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["array_count"] = 999
    write_json(path, manifest)
    with pytest.raises(a.CaptureError, match="manifest identity"):
        a.verify_capture(output)


def test_no_refit_claim_can_be_inserted_into_completion_receipt(fixture):
    source, output = fixture
    a.capture(source, output)
    path = output / "capture-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["full_refit_performed"] = True
    write_json(path, receipt)
    with pytest.raises(a.CaptureError, match="completion receipt"):
        a.verify_capture(output)


def test_source_hash_gate_independently_rejects_changed_array(fixture):
    source, _ = fixture
    array = next(source.rglob("*.npy"))
    array.write_bytes(array.read_bytes()[:-1] + b"x")
    with pytest.raises(a.CaptureError, match="Frozen evidence changed"):
        a.verify_source(source)


def test_archive_copy_gate_independently_rejects_postverification_change(fixture, tmp_path):
    source, _ = fixture
    members, _ = a.verify_source(source)
    array = next(source.rglob("*.npy"))
    array.write_bytes(array.read_bytes()[:-1] + b"x")
    with pytest.raises(a.CaptureError, match="changed while archiving"):
        a.archive_source(source, tmp_path / "unaccepted.zip", members)


def test_archive_members_remain_bound_when_parts_manifest_and_receipt_are_all_rehashed(fixture, tmp_path):
    source, output = fixture
    a.capture(source, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    members, _ = a.verify_source(source)
    name = next(n for n in members if n.endswith(".npy"))
    path = source / name
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    forged_members = {**members, name: a.file_digest(path)}
    joined = tmp_path / "forged.zip"
    a.archive_source(source, joined, forged_members)
    for old in output.glob(a.ARCHIVE + ".part*"):
        old.unlink()
    parts = []
    data = joined.read_bytes()
    for index, start in enumerate(range(0, len(data), a.PART_BYTES), 1):
        name = f"{a.ARCHIVE}.part{index:04d}"
        part = data[start:start + a.PART_BYTES]
        (output / name).write_bytes(part)
        parts.append({"name": name, "bytes": len(part), "sha256": a.digest(part)})
    manifest.update(parts=parts, archive_bytes=len(data), archive_sha256=a.digest(data))
    write_json(manifest_path, manifest)
    receipt_path = output / "capture-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(manifest_sha256=a.file_digest(manifest_path), parts=len(parts))
    write_json(receipt_path, receipt)
    with pytest.raises(a.CaptureError, match="Archived evidence changed"):
        a.verify_capture(output)


def test_freeze_anchor_is_checked_before_member_inventory_is_trusted(fixture):
    source, _ = fixture
    freeze = json.loads((source / "freeze.json").read_text())
    freeze["unexpected_revision"] = 2
    with pytest.raises(a.CaptureError, match="accepted SHA"):
        a.expected_members(a.canonical(freeze) + b"\n")


@pytest.mark.parametrize("value", [[], {"parts": ["invalid"]}])
def test_invalid_manifest_schema_reports_incomplete_without_traceback(fixture, capsys, value):
    source, output = fixture
    a.capture(source, output)
    write_json(output / "manifest.json", value)
    assert a.main(["--verify-only", "--output-dir", str(output), "--strict"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INCOMPLETE"
