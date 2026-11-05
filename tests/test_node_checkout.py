"""CPU-only fixtures for node preparation; no actual study evidence is generated."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_node_checkout", Path(__file__).resolve().parents[1] / "scripts/prepare_node_checkout.py")
node = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(node)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root):
    git(root, "add", "--all")
    git(root, "-c", "commit.gpgsign=false", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "fixture")
    return git(root, "rev-parse", "HEAD")


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(root).parts}


@pytest.fixture
def checkout(tmp_path):
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    source.mkdir()
    runtime.mkdir()
    (source / ".gitignore").write_text("/results/\n/.validation/\n__pycache__/\n")
    (source / "decision_study").mkdir()
    (source / "decision_study/__init__.py").write_text('"""Artificial fixture package."""\n')
    cli = "import sys\nassert sys.argv[1:] == ['--help'], 'Unexpected experiment command'\nprint('fixture help')\n"
    (source / "decision_study/__main__.py").write_text(cli)
    (source / "scripts").mkdir()
    for version in (2, 3):
        package = source / f"decision_study/v{version}"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(cli)
        files = {}
        for name in node.DATA_NAMES:
            relative = f"results/solo-v{version}/{name}"
            path = runtime / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"ARTIFICIAL data fixture {relative}\n".encode()
            path.write_bytes(payload)
            files[relative] = hashlib.sha256(payload).hexdigest()
        (source / f"V{version}_RELEASE_MANIFEST.json").write_text(json.dumps({"files": files}))
        verifier = ("import hashlib,json\nfrom pathlib import Path\n"
                    f"manifest=json.loads(Path('V{version}_RELEASE_MANIFEST.json').read_text())\n"
                    "for name,expected in manifest['files'].items():\n"
                    " assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected\n"
                    "print('verified fixture')\n")
        (source / f"scripts/verify_v{version}_release.py").write_text(verifier)
    (runtime / "model-and-runtime-sentinel").write_bytes(b"DO NOT TOUCH old environment or model")
    git(source, "init", "-q")
    return source, runtime, commit(source)


def prepare(checkout):
    source, runtime, revision = checkout
    return node.prepare(runtime, revision, sys.executable, project=source)


def test_exact_eight_files_idempotent_and_runtime_untouched(checkout, monkeypatch):
    source, runtime, _ = checkout
    original_source, original_runtime = snapshot(source), snapshot(runtime)
    calls, original_run = [], node.run

    def record(argv, project):
        calls.append([str(x) for x in argv])
        return original_run(argv, project)

    monkeypatch.setattr(node, "run", record)
    first = prepare(checkout)
    assert (first["copied"], first["reused"]) == (8, 0)
    second = prepare(checkout)
    assert (second["copied"], second["reused"]) == (0, 8)
    assert snapshot(runtime) == original_runtime
    assert all((source / name).read_bytes() == payload for name, payload in original_source.items())
    assert {str(p.relative_to(source)) for p in (source / "results").rglob("*") if p.is_file()} == set(first["prepared_sha256"])
    assert git(source, "status", "--porcelain") == ""
    python_calls = [args[2:] for args in calls if args[0] == sys.executable]
    assert python_calls[:6] == [["-c", "import decision_study, json; print(json.dumps(decision_study.__file__))"],
                               ["scripts/verify_v2_release.py"], ["scripts/verify_v3_release.py"],
                               ["-m", "decision_study", "--help"], ["-m", "decision_study.v2", "--help"],
                               ["-m", "decision_study.v3", "--help"]]
    assert all(args[1] == "-B" for args in calls if args[0] == sys.executable)
    assert json.loads((source / ".validation/node-checkout/receipt.json").read_text()) == second


@pytest.mark.parametrize("where", ["runtime", "checkout"])
def test_last_file_mismatch_blocks_all_copies(checkout, where):
    source, runtime, _ = checkout
    target = (runtime if where == "runtime" else source) / "results/solo-v3/sealed/holdout_gold.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"WRONG bytes")
    before_source, before_runtime = snapshot(source), snapshot(runtime)
    with pytest.raises(node.PreparationError, match="differs from release"):
        prepare(checkout)
    assert snapshot(source) == before_source
    assert snapshot(runtime) == before_runtime


@pytest.mark.parametrize("problem", ["wrong-head", "dirty", "short-sha"])
def test_identity_failures_do_not_copy(checkout, problem):
    source, runtime, revision = checkout
    if problem == "dirty":
        (source / "decision_study/__init__.py").write_text("changed = True\n")
    else:
        revision = "0" * 40 if problem == "wrong-head" else revision[:8]
    with pytest.raises(node.PreparationError, match="HEAD|dirty|40-character"):
        node.prepare(runtime, revision, sys.executable, project=source)
    assert not (source / "results").exists()


@pytest.mark.parametrize("side", ["runtime", "checkout"])
def test_symlink_data_parent_rejected_without_touching_external_files(checkout, tmp_path, side):
    source, runtime, _ = checkout
    external = tmp_path / "external"
    external.mkdir()
    (external / "sentinel").write_text("untouched")
    root = runtime if side == "runtime" else source
    path = root / "results/solo-v3"
    if path.exists():
        path.rename(root / "saved-v3")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(external, target_is_directory=True)
    before_source, before_runtime, before_external = snapshot(source), snapshot(runtime), snapshot(external)
    with pytest.raises(node.PreparationError, match="Symlink"):
        prepare(checkout)
    assert snapshot(source) == before_source
    assert snapshot(runtime) == before_runtime
    assert snapshot(external) == before_external


def test_manifest_traversal_rejected(checkout):
    source, runtime, _ = checkout
    path = source / "V3_RELEASE_MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside"] = "a" * 64
    path.write_text(json.dumps(manifest))
    revision = commit(source)
    with pytest.raises(node.PreparationError, match="Unsafe relative"):
        node.prepare(runtime, revision, sys.executable, project=source)
    assert not (source / "results").exists()


def test_wrong_import_origin_rejected_before_copy(checkout, monkeypatch):
    source, runtime, _ = checkout
    original = node.run

    def wrong_import(argv, project):
        if "-c" in argv:
            return json.dumps(str(runtime / "decision_study/__init__.py"))
        return original(argv, project)

    monkeypatch.setattr(node, "run", wrong_import)
    with pytest.raises(node.PreparationError, match="wrong source.*PYTHONPATH"):
        prepare(checkout)
    assert not (source / "results").exists()


def test_interrupted_copy_can_retry_without_replacing_completed_files(checkout, monkeypatch):
    source, _, _ = checkout
    original, calls = node.copy_missing, []

    def interrupted(*args):
        calls.append(args[2])
        if len(calls) == 2:
            raise OSError("simulated interruption")
        return original(*args)

    monkeypatch.setattr(node, "copy_missing", interrupted)
    with pytest.raises(OSError, match="simulated"):
        prepare(checkout)
    first = source / calls[0]
    inode = first.stat().st_ino
    monkeypatch.setattr(node, "copy_missing", original)
    receipt = prepare(checkout)
    assert (receipt["copied"], receipt["reused"]) == (7, 1)
    assert first.stat().st_ino == inode


def test_concurrent_destination_creation_is_never_overwritten(checkout, monkeypatch):
    source, runtime, _ = checkout
    relative = "results/solo-v2/data/examples.jsonl"
    expected = node.digest(runtime / relative)
    original = node.os.link

    def race(temporary, target):
        target.write_bytes(b"concurrent owner")
        return original(temporary, target)

    monkeypatch.setattr(node.os, "link", race)
    with pytest.raises(FileExistsError):
        node.copy_missing(runtime, source, relative, expected)
    assert (source / relative).read_bytes() == b"concurrent owner"
    assert list((source / relative).parent.glob(".prepare-*")) == []


def test_overlapping_roots_blocked(checkout):
    source, _, revision = checkout
    with pytest.raises(node.PreparationError, match="non-overlapping"):
        node.prepare(source, revision, sys.executable, project=source)


def test_symlink_parent_cannot_hide_overlapping_runtime_root(checkout, tmp_path):
    source, _, revision = checkout
    (source / "nested-runtime").mkdir()
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(node.PreparationError, match="non-overlapping"):
        node.prepare(alias / "nested-runtime", revision, sys.executable, project=source)
