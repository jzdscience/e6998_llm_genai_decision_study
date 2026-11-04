#!/usr/bin/env python3
"""Install a hash-pinned CUDA toolkit into an isolated, verified target directory.

Run with the recorded model interpreter. Its environment is never installed into
or upgraded. Failed staging directories are preserved for diagnosis.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = PROJECT / "configs/cuda_build_toolkit_13_0_flashinfer.txt"
MARKER = ".cuda-toolkit-receipt.json"
PINS = {"nvidia-cuda-nvcc": "13.0.88", "nvidia-cuda-crt": "13.0.88",
        "nvidia-nvvm": "13.0.88", "nvidia-cuda-runtime": "13.0.96",
        "nvidia-curand": "10.4.0.35"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _target_path(target: Path) -> Path:
    target = target.expanduser().absolute()
    for path in (target, *target.parents):
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlinked toolkit path: {path}")
    target = target.resolve()
    prefix = Path(sys.prefix).resolve()
    if target == prefix or prefix in target.parents or target in prefix.parents:
        raise RuntimeError("Toolkit target must be separate from the model Python environment")
    return target


def _requirements(path: Path) -> tuple[str, dict]:
    # Do not allow a requirements file to inject pip options or extra packages.
    raw = path.read_bytes()
    body = raw.decode("utf-8")
    rows = []
    for line in body.replace("\\\n", " ").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line)
    wheels = {}
    for line in rows:
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)\s+@\s+(https://files\.pythonhosted\.org/packages/"
            r"[A-Za-z0-9_./-]+\.whl)\s+--hash=sha256:([0-9a-f]{64})", line)
        if not match:
            raise RuntimeError("Expected only direct, SHA256-pinned PyPI wheel requirements")
        name, url, sha = match.groups()
        name = _name(name)
        wheel = url.rsplit("/", 1)[1].split("-")
        if (name in wheels or name not in PINS or len(wheel) < 5 or
                _name(wheel[0]) != name or wheel[1] != PINS[name] or
                not wheel[-1].endswith("x86_64.whl")):
            raise RuntimeError(f"Unexpected or duplicate CUDA wheel: {name}")
        wheels[name] = {"version": PINS[name], "url": url, "sha256": sha}
    if set(wheels) != set(PINS):
        raise RuntimeError("CUDA manifest must contain exactly the five pinned components")
    return hashlib.sha256(raw).hexdigest(), wheels


def _distributions(target: Path) -> dict:
    found = {}
    for dist in metadata.distributions(path=[str(target)]):
        name = _name(dist.metadata.get("Name", ""))
        if not name or name in found:
            raise RuntimeError("Missing or duplicate distribution metadata in toolkit target")
        found[name] = dist.version
    if found != PINS:
        raise RuntimeError(f"Isolated toolkit distributions differ from expected pins: {found}")
    return found


def _tree(target: Path) -> dict:
    tree = {}
    for directory, folders, files in os.walk(target, followlinks=False):
        for name in sorted(folders + files):
            path = Path(directory) / name
            relative = path.relative_to(target).as_posix()
            if relative == MARKER:
                continue
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                tree[relative] = {"kind": "directory", "mode": mode}
            elif stat.S_ISREG(info.st_mode):
                tree[relative] = {"kind": "file", "mode": mode,
                                  "size": info.st_size, "sha256": file_hash(path)}
            else:
                raise RuntimeError(f"Toolkit contains a symlink or special file: {path}")
    return tree


def validate_target(target: Path, requirements: Path = DEFAULT_REQUIREMENTS) -> dict:
    """Verify a completed toolkit without importing any of its packages."""
    target = _target_path(Path(target))
    manifest_hash, wheels = _requirements(Path(requirements))
    marker = target / MARKER
    if not target.is_dir() or marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"Existing toolkit has no valid completion receipt; preserved: {target}")
    receipt = json.loads(marker.read_text())
    expected = {"schema_version": 1, "status": "complete", "target": str(target),
                "requirements_sha256": manifest_hash, "wheels": wheels,
                "installer_sha256": file_hash(Path(__file__)), "platform": "linux-x86_64"}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Toolkit receipt identity changed; preserved: {target}")
    tree = _tree(target)
    if receipt.get("tree") != tree or receipt.get("tree_sha256") != _digest(tree):
        raise RuntimeError(f"Toolkit file contents or modes changed; preserved: {target}")
    if receipt.get("distributions") != _distributions(target):
        raise RuntimeError(f"Toolkit distribution metadata changed; preserved: {target}")
    return receipt


def _publish(stage: Path, target: Path) -> None:
    # All installers hold the target-specific lock during this check and rename.
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"Toolkit target appeared during preparation; preserved: {target}")
    os.rename(stage, target)


@contextmanager
def _target_lock(target: Path):
    path = target.parent / f".{target.name}.prepare.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError(f"Invalid toolkit lock file: {path}")
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(target: Path, requirements: Path = DEFAULT_REQUIREMENTS) -> dict:
    if sys.platform != "linux" or platform.machine().lower() != "x86_64":
        raise RuntimeError("This pinned toolkit supports Linux x86_64 only")
    target = _target_path(Path(target))
    requirements = Path(requirements).expanduser().resolve(strict=True)
    manifest_hash, wheels = _requirements(requirements)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _target_lock(target):
        return _prepare_locked(target, requirements, manifest_hash, wheels)


def _prepare_locked(target: Path, requirements: Path, manifest_hash: str, wheels: dict) -> dict:
    if target.exists():
        validate_target(target, requirements)
        return {"status": "complete", "reused": True, "target": str(target),
                "receipt": str(target / MARKER), "requirements_sha256": manifest_hash}
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    print(f"CUDA toolkit staging directory (preserved on failure): {stage}", flush=True)
    temporary = stage / ".pip-work"
    temporary.mkdir()
    env = dict(os.environ, TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull)
    command = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
               "install", "--no-input", "--no-index", "--no-cache-dir", "--no-deps",
               "--only-binary=:all:", "--require-hashes", "--no-compile",
               "--target", str(stage), "-r", str(requirements)]
    try:
        timeout = min(1200.0, float(env.get("DECISION_STUDY_DEADLINE_EPOCH", time.time() + 1200)) - time.time())
        if not 0 < timeout <= 1200:
            raise RuntimeError("No valid session time remains to prepare the toolkit")
        subprocess.run(command, check=True, env=env, timeout=timeout)
        if _requirements(requirements) != (manifest_hash, wheels):
            raise RuntimeError("CUDA requirements changed during installation")
        tree = _tree(stage)
        receipt = {"schema_version": 1, "status": "complete", "target": str(target),
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "requirements_sha256": manifest_hash, "wheels": wheels,
                   "installer_sha256": file_hash(Path(__file__)), "platform": "linux-x86_64",
                   "distributions": _distributions(stage), "tree": tree, "tree_sha256": _digest(tree)}
        with (stage / MARKER).open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish(stage, target)
        validate_target(target, requirements)
    except BaseException:
        print(f"Toolkit preparation failed; no cleanup performed. Inspect: {stage}", file=sys.stderr, flush=True)
        raise
    return {"status": "complete", "reused": False, "target": str(target),
            "receipt": str(target / MARKER), "requirements_sha256": manifest_hash}


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.writing-", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to replace an existing toolkit result; temporary preserved: {temporary}")
    os.rename(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    result = {"status": "blocked", "target": str(args.target.expanduser().absolute())}
    result_path = None
    code = 2
    try:
        if args.result:
            args.result = _target_path(args.result)
            target = args.target.expanduser().resolve()
            if (args.result.exists() or args.result.is_symlink() or args.result == target or
                    target in args.result.parents):
                raise RuntimeError("Result must be a new file outside the toolkit target")
            result_path = args.result
        result = prepare(args.target, args.requirements)
        receipt = validate_target(Path(result["target"]), args.requirements)
        result["verification"] = {key: receipt[key] for key in (
            "wheels", "distributions", "tree_sha256", "installer_sha256", "requirements_sha256")}
        result["receipt_sha256"] = file_hash(Path(result["receipt"]))
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        if result_path:
            try:
                _write_result(result_path, result)
            except Exception as exc:
                print(f"Cannot write toolkit result: {exc}", file=sys.stderr, flush=True)
                code = 2
        print(json.dumps(result, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
