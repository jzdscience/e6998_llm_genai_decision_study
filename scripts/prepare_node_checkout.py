#!/usr/bin/env python3
"""Prepare a pinned source checkout using eight verified legacy data files only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
DATA_NAMES = ("data/examples.jsonl", "data/fit_gold.json", "data/manifest.json", "sealed/holdout_gold.json")


class PreparationError(RuntimeError):
    pass


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PreparationError(f"Unsafe relative path: {relative}")
    path = root
    if root.is_symlink():
        raise PreparationError(f"Symlink root: {root}")
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise PreparationError(f"Symlink path: {path}")
    return path


def digest(path):
    if not path.is_file():
        raise PreparationError(f"Missing regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(argv, project):
    completed = subprocess.run([str(x) for x in argv], cwd=project, capture_output=True,
                               text=True, timeout=60)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PreparationError(f"Command failed ({completed.returncode}): {' '.join(map(str, argv))}\n{detail}")
    return completed.stdout.strip()


def prepared_members(project):
    members = {}
    for version in (2, 3):
        manifest = json.loads(safe_path(project, f"V{version}_RELEASE_MANIFEST.json").read_text())
        files = manifest["files"]
        for relative in files:
            safe_path(project, relative)
        expected = {f"results/solo-v{version}/{name}" for name in DATA_NAMES}
        actual = {name for name in files if name.startswith("results/")}
        if actual != expected:
            raise PreparationError(f"V{version} manifest must pin exactly its four prepared data files")
        for name in sorted(expected):
            value = files[name]
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise PreparationError(f"Invalid SHA256 for {name}")
            members[name] = value
    return members


def copy_missing(source_root, project, relative, expected):
    source = safe_path(source_root, relative)
    target = safe_path(project, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".prepare-", delete=False) as f:
            temporary = Path(f.name)
            with source.open("rb") as src:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    f.write(block)
            f.flush()
            os.fsync(f.fileno())
        if digest(temporary) != expected:
            raise PreparationError(f"Source changed during copy: {source}")
        safe_path(project, relative)
        os.link(temporary, target)  # Exclusive installation: never replace an existing file.
        if digest(target) != expected:
            raise PreparationError(f"Copied file failed verification: {target}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(runtime_root, expected_commit, python_bin=None, *, project=PROJECT):
    project, runtime_root = Path(project).absolute(), Path(runtime_root).expanduser().absolute()
    for root in (project, runtime_root):
        if root.is_symlink() or ".." in root.parts or not root.is_dir():
            raise PreparationError(f"Expected an existing, non-symlink directory: {root}")
    source_real, runtime_real = project.resolve(), runtime_root.resolve()
    if source_real == runtime_real or source_real in runtime_real.parents or runtime_real in source_real.parents:
        raise PreparationError("Source checkout and runtime root must be distinct, non-overlapping directories")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise PreparationError("--expected-commit requires a full 40-character lowercase commit SHA")
    git = ["git", "--no-optional-locks"]
    if Path(run(git + ["rev-parse", "--show-toplevel"], project)).resolve() != project.resolve():
        raise PreparationError("Source checkout must be the Git repository root")
    if run(git + ["rev-parse", "HEAD"], project) != expected_commit:
        raise PreparationError("Checkout HEAD differs from --expected-commit; no files copied")
    if run(git + ["status", "--porcelain=v1", "--untracked-files=all"], project):
        raise PreparationError("Source checkout is dirty; preserve and review changes before preparation")
    python = Path(python_bin).expanduser().absolute() if python_bin else runtime_root / ".venv-controller/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise PreparationError(f"Controller Python is missing or not executable: {python}")
    members = prepared_members(project)
    missing = []
    for relative, expected in members.items():
        source, target = safe_path(runtime_root, relative), safe_path(project, relative)
        if digest(source) != expected:
            raise PreparationError(f"Runtime prepared file differs from release: {source}")
        if target.exists():
            if digest(target) != expected:
                raise PreparationError(f"Existing checkout data differs from release: {target}")
        else:
            missing.append(relative)
    receipt_path = safe_path(project, ".validation/node-checkout/receipt.json")
    if receipt_path.exists() and not receipt_path.is_file():
        raise PreparationError(f"Receipt is not a regular file: {receipt_path}")
    probe = "import decision_study, json; print(json.dumps(decision_study.__file__))"
    imported = Path(json.loads(run([python, "-B", "-c", probe], project))).resolve()
    if imported != (project / "decision_study/__init__.py").resolve():
        raise PreparationError(f"Controller imports the wrong source: {imported}; inspect PYTHONPATH and editable installs")
    for relative in missing:
        copy_missing(runtime_root, project, relative, members[relative])
    checks = [[f"scripts/verify_v{v}_release.py"] for v in (2, 3)]
    checks += [["-m", module, "--help"] for module in ("decision_study", "decision_study.v2", "decision_study.v3")]
    for args in checks:
        run([python, "-B", *args], project)
    if run(git + ["status", "--porcelain=v1", "--untracked-files=all"], project):
        raise PreparationError("Preparation left a dirty checkout; review ignore rules and changes")
    receipt = {"status": "READY", "commit": expected_commit, "source_root": str(project),
               "runtime_root": str(runtime_root), "controller": str(python),
               "prepared_sha256": members, "copied": len(missing), "reused": len(members) - len(missing),
               "checks": checks, "scope": "Source/data verification only; no GPU, session initialization or experiment run."}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=receipt_path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(receipt, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        safe_path(project, ".validation/node-checkout/receipt.json")
        os.replace(temporary, receipt_path)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--python-bin", type=Path)
    args = parser.parse_args()
    try:
        receipt = prepare(args.runtime_root, args.expected_commit, args.python_bin)
    except (PreparationError, OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(f"READY {receipt['commit']}\nSource: {receipt['source_root']}\nRuntime: {receipt['runtime_root']}")
    print(f"Prepared files: {receipt['copied']} copied, {receipt['reused']} reused")
    print(receipt["scope"])


if __name__ == "__main__":
    main()
