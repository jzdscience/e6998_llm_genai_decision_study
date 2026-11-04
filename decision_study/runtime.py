"""Pinned, isolated GPU runtimes. Importing this module never imports Torch.

Installation and paid GPU work are explicit operations. Failure is reported; no
model, precision, revision, or backend is silently substituted.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROFILES = ("primary", "vllm019", "reference", "kev_no_fused", "kev_eager")
MODELS = ("kev", "startlux", "qwen")


class RuntimeCommandError(subprocess.CalledProcessError):
    """Keep the actual subprocess log discoverable from the top-level error."""

    def __init__(self, returncode, cmd, log_path, *, output=None, stderr=None):
        super().__init__(returncode, cmd, output=output, stderr=stderr)
        self.log_path = str(log_path)

    def __str__(self):
        return f"{super().__str__()} Actual subprocess output: {self.log_path}"


def select_profile(output_dir, model, profile):
    """Choose an explicit per-model fallback before formal prediction starts."""
    if model not in MODELS or profile not in PROFILES:
        raise ValueError("Unknown model/profile")
    out = Path(output_dir)
    path = out / "runtime/selected_profiles.json"
    selections = _read(path) if path.exists() else {}
    if selections.get(model, "primary") != profile:
        predicted = out / "predictions" / model
        if (predicted / "config.json").exists() or any(predicted.glob("shard-*.json")):
            raise RuntimeError("Cannot change a model profile after formal prediction started; use a new OUTPUT_DIR")
    selections[model] = profile
    _write(path, selections)


def _resolve_profile(output_dir, model, profile=None):
    if profile is None:
        path = Path(output_dir) / "runtime/selected_profiles.json"
        profile = (_read(path) if path.exists() else {}).get(model, "primary")
    if profile not in PROFILES:
        raise ValueError(f"Unknown runtime profile {profile!r}")
    return profile


def select_memory_fraction(output_dir, model, fraction):
    """Explicit vLLM reservation change; never rewrite an existing prediction run."""
    if model not in ("startlux", "qwen"):
        raise ValueError("Only vLLM models have a GPU-memory-utilization setting; Kev has no equivalent flag")
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0.1 <= fraction <= 0.9:
        raise ValueError("GPU memory fraction must be between 0.1 and 0.9")
    out = Path(output_dir)
    path = out / "runtime/selected_memory.json"
    selections = _read(path) if path.exists() else {}
    if selections.get(model, 0.40) != fraction:
        predicted = out / "predictions" / model
        if (predicted / "config.json").exists() or any(predicted.glob("shard-*.json")):
            raise RuntimeError("Cannot change GPU memory reservation after formal prediction started; use a new OUTPUT_DIR")
    selections[model] = float(fraction)
    _write(path, selections)


def _memory_fraction(output_dir, model, lock):
    path = Path(output_dir) / "runtime/selected_memory.json"
    return float((_read(path) if path.exists() else {}).get(model, lock["gpu_memory_utilization"]))


def _timeout_seconds(default):
    deadline = os.environ.get("DECISION_STUDY_DEADLINE_EPOCH")
    if deadline is None:
        return float(default)
    remaining = float(deadline) - time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise RuntimeError("Monitored GPU budget deadline reached; stop the VM to stop billing")
    return min(float(default), remaining)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value)
    temp.replace(path)


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "COMS6998-decision-study/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def _valid_sha(sha):
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"An immutable full revision is required, got {sha!r}")
    return sha


def _verify_git_revision(repo, sha):
    """Avoid unauthenticated GitHub API quota making a public pinned commit unusable."""
    try:
        result = _get_json(f"https://api.github.com/repos/{repo}/commits/{sha}")
        if result.get("sha") != sha:
            raise RuntimeError(f"Git revision mismatch for {repo}")
        return {"kind": "github", "repo": repo, "sha": sha, "verification": "commit_api"}
    except urllib.error.HTTPError as exc:
        if exc.code not in (403, 429):
            raise
        exc.close()
        # Full SHA is unchanged. A raw file at that exact commit is only the
        # existence check; installation still git-fetches and verifies HEAD.
        url = f"https://raw.githubusercontent.com/{repo}/{sha}/pyproject.toml"
        with urllib.request.urlopen(url, timeout=45) as response:
            body = response.read()
        return {"kind": "github", "repo": repo, "sha": sha, "verification": "pinned_raw_file",
                "file": "pyproject.toml", "file_sha256": hashlib.sha256(body).hexdigest(),
                "note": "GitHub commit API rate limited; immutable checkout verified again at install"}


def freeze_lock(project_dir, output_dir):
    """Validate the shipped immutable revisions; preserve an existing matching lock."""
    defaults = _read(Path(project_dir) / "configs/runtime_defaults.json")
    expected_hash = _hash(defaults)
    path = Path(output_dir) / "runtime/lock.json"
    if path.exists():
        lock = _read(path)
        stored_defaults = {key: lock.get(key) for key in defaults}
        if lock.get("defaults_hash") != expected_hash or _hash(stored_defaults) != expected_hash:
            raise RuntimeError("Runtime defaults differ from this run's lock; use a new OUTPUT_DIR.")
        _validate_lock(lock)
        return lock
    verified = []
    for name, model in defaults["models"].items():
        for repo_key, revision_key in (("repo", "revision"), ("base_repo", "base_revision")):
            if repo_key not in model:
                continue
            sha = _valid_sha(model[revision_key])
            result = _get_json(f"https://huggingface.co/api/models/{model[repo_key]}/revision/{sha}")
            if result.get("sha") != sha:
                raise RuntimeError(f"HF revision mismatch for {model[repo_key]}")
            verified.append({"kind": "huggingface", "repo": model[repo_key], "sha": sha})
        if "code_repo" in model:
            sha = _valid_sha(model["code_revision"])
            verified.append(_verify_git_revision(model["code_repo"], sha))
    lock = {**defaults, "defaults_hash": expected_hash, "created_at": _now(), "verified_sources": verified}
    lock["lock_hash"] = _hash(defaults)
    _write(path, lock)
    return lock


def _validate_lock(lock):
    for m in lock["models"].values():
        for field in ("revision", "base_revision", "code_revision"):
            if field in m:
                _valid_sha(m[field])


def connectivity_report():
    """Bounded public endpoint probes, without downloads, accounts, or credentials.

    A successful metadata probe is not a promise that all weight/CDN URLs or the
    Git protocol are permitted by a company network policy.
    """
    checks = [
        ("huggingface", "https://huggingface.co/api/models/Qwen/Qwen3.5-4B/revision/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
        ("github", "https://github.com/jaredpalmer/kev"),
        ("github_raw", "https://raw.githubusercontent.com/jaredpalmer/kev/2074cc44256200bb37540103fa4572901faa10f3/pyproject.toml"),
        ("pypi", "https://pypi.org/simple/uv/"),
    ]
    results = []
    for name, url in checks:
        started = time.monotonic()
        row = {"name": name, "url": url}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "COMS6998-decision-study/1.0"})
            with urllib.request.urlopen(request, timeout=8) as response:
                response.read(512)
                row.update(reachable=response.status == 200, status_code=response.status)
        except (urllib.error.URLError, OSError) as exc:
            row.update(reachable=False, error=f"{type(exc).__name__}: {exc}")
        row["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
        results.append(row)
    return {"checked": True, "metadata_endpoints_reachable": all(r["reachable"] for r in results), "endpoints": results,
            "limitations": "Metadata reachability only; installation still verifies package downloads, Git checkout and checkpoint access."}


def doctor(output_dir, require_gpu=False, check_network=False):
    """Write a portable preflight report. GPU checks never import CUDA in the controller."""
    report = {"timestamp": _now(), "system": platform.system(), "machine": platform.machine(),
              "python": sys.version, "executable": sys.executable, "gpu": [], "warnings": []}
    try:
        args = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=True)
        for line in result.stdout.strip().splitlines():
            idx, name, memory, driver, cap = [s.strip() for s in line.split(",")]
            report["gpu"].append({"index": int(idx), "name": name, "memory_mb": float(memory), "driver": driver, "compute_capability": cap})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["warnings"].append(f"NVIDIA GPU inspection unavailable: {type(exc).__name__}: {exc}")
    report["gpu_ready"] = bool(report["gpu"]) and platform.system() == "Linux"
    if report["gpu_ready"]:
        try:
            report["gpu_ready"] = float(report["gpu"][0]["compute_capability"]) >= 8.0
        except ValueError:
            report["gpu_ready"] = False
        if not report["gpu_ready"]:
            report["warnings"].append("The BF16 protocol requires an Ampere-or-newer CUDA GPU; do not substitute T4/FP16.")
    report["selected_gpu_index"] = 0
    report["reserved_gpu_count"] = 1 if report["gpu"] else 0
    if len(report["gpu"]) > 1:
        report["warnings"].append("Only physical GPU 0 is used. Other GPUs are not counted in experiment allocated GPU-seconds; VM billing may cover the entire instance.")
    report["status"] = "ok" if not require_gpu or report["gpu_ready"] else "blocked"
    report["network"] = connectivity_report() if check_network else {"checked": False}
    if check_network and not report["network"]["metadata_endpoints_reachable"]:
        report["warnings"].append("Some public download endpoints are unreachable; inspect network diagnostics before initial installation. Cached CPU analysis remains available.")
    _write(Path(output_dir) / "runtime/doctor.json", report)
    if require_gpu and not report["gpu_ready"]:
        raise RuntimeError("GPU preflight blocked; see runtime/doctor.json. CPU analysis remains available.")
    return report


def _env(output_dir):
    # Avoid hidden user settings changing model computation; do not persist secrets.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KEV_", "STARTLUX_", "VLLM_"))}
    work = _work_dir(output_dir)
    env.update({"HF_HOME": str(work / "cache/huggingface"), "UV_CACHE_DIR": str(work / "cache/uv"),
                "UV_PYTHON_INSTALL_DIR": str(work / "python"),
                "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1", "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "0"})
    return env


def _work_dir(output_dir):
    return Path(os.environ.get("DECISION_STUDY_RUNTIME_DIR", str(Path(output_dir).resolve() / "runtime/work"))).resolve()


def _run(args, output_dir, log_name, *, cwd=None, capture=False):
    log_path = Path(output_dir) / "runtime/logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        log.write("\nCOMMAND " + json.dumps([str(x) for x in args]) + "\n")
        log.flush()
        timeout = _timeout_seconds(7200)
        command = [str(x) for x in args]
        process = subprocess.Popen(command, cwd=cwd, env=_env(output_dir),
                                   stdout=subprocess.PIPE if capture else log,
                                   stderr=subprocess.PIPE if capture else subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            raise RuntimeError("Runtime operation exceeded its budget deadline or 2-hour limit, or was interrupted; owned processes stopped") from None
        if capture:
            log.write((stdout or "") + (stderr or ""))
        if process.returncode:
            raise RuntimeCommandError(process.returncode, command, log_path, output=stdout, stderr=stderr)
        if capture:
            return stdout
    return None


def _checkout(output_dir, repo, revision):
    _valid_sha(revision)
    path = _work_dir(output_dir) / "sources" / (repo.replace("/", "--") + "-" + revision[:12])
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", path], output_dir, "install.log")
        _run(["git", "-C", path, "remote", "add", "origin", f"https://github.com/{repo}.git"], output_dir, "install.log")
        _run(["git", "-C", path, "fetch", "--depth", "1", "origin", revision], output_dir, "install.log")
        _run(["git", "-C", path, "checkout", "--detach", revision], output_dir, "install.log")
    actual = _run(["git", "-C", path, "rev-parse", "HEAD"], output_dir, "install.log", capture=True).strip()
    if actual != revision:
        raise RuntimeError(f"Source checkout changed: {path}")
    dirty = _run(["git", "-C", path, "status", "--porcelain=v1", "--untracked-files=all", "-z"],
                 output_dir, "install.log", capture=True)
    if dirty:
        _remove_verified_kev_build_copies(output_dir, path, repo, revision, dirty)
        dirty = _run(["git", "-C", path, "status", "--porcelain=v1", "--untracked-files=all", "-z"],
                     output_dir, "install.log", capture=True)
        if dirty:
            raise RuntimeError(f"Source checkout remains changed after verified build-copy cleanup: {path}")
    return path


def _remove_verified_kev_build_copies(output_dir, path, repo, revision, dirty):
    """Remove only reproduced setuptools copies, after validating every dirty path.

    Kev's pinned setuptools wheel build leaves build/lib/kev/*.py behind. Each
    copy must have the exact raw Git blob hash of a regular tracked source file
    at HEAD. Modified source, unknown files, symlinks and altered copies fail
    before anything is removed. No gitignore rules or tracked files are changed.
    """
    path = Path(path)
    entries = dirty.split("\0")
    if not entries or entries[-1] != "":
        raise RuntimeError(f"Invalid source status; checkout preserved: {path}")
    entries.pop()
    candidates = []
    for entry in entries:
        relative = entry[3:]
        parts = relative.split("/")
        if (repo != "jaredpalmer/kev" or entry[:3] != "?? " or len(parts) < 4
                or parts[:3] != ["build", "lib", "kev"] or not relative.endswith(".py")
                or any(part in {"", ".", ".."} for part in parts)):
            raise RuntimeError(f"Source checkout changed; unexpected path preserved: {relative!r} in {path}")
        candidate = path.joinpath(*parts)
        if any(parent.is_symlink() for parent in (candidate, *candidate.parents) if parent == path or path in parent.parents):
            raise RuntimeError(f"Source checkout contains a symlink; preserved: {candidate}")
        if not stat.S_ISREG(candidate.lstat().st_mode):
            raise RuntimeError(f"Build artifact is not a regular file; preserved: {candidate}")
        source_relative = "/".join(parts[2:])
        tree = _run(["git", "-C", path, "ls-tree", "-z", revision, "--", source_relative],
                    output_dir, "install.log", capture=True)
        fields = tree.split("\t", 1)
        if (len(fields) != 2 or fields[1] != source_relative + "\0"
                or fields[0].split()[:2] not in (["100644", "blob"], ["100755", "blob"])):
            raise RuntimeError(f"Build artifact has no regular tracked source at pinned HEAD; preserved: {candidate}")
        expected = fields[0].split()[2]
        actual = _run(["git", "-C", path, "hash-object", "--no-filters", "--", candidate],
                      output_dir, "install.log", capture=True).strip()
        if actual != expected:
            raise RuntimeError(f"Build artifact differs from pinned HEAD; preserved: {candidate}")
        candidates.append(candidate)
    # All checks precede the first deletion, including unexpected later paths.
    for candidate in candidates:
        candidate.unlink()
    for parent in sorted({p for candidate in candidates for p in candidate.parents if p != path and path in p.parents},
                         key=lambda p: len(p.parts), reverse=True):
        try:
            parent.rmdir()  # Only empty directories; never recursive deletion.
        except OSError:
            pass
    with (Path(output_dir) / "runtime/logs/install.log").open("a") as log:
        log.write(f"Removed {len(candidates)} verified setuptools build copies from {path}; tracked source untouched.\n")


def _vllm_version(lock, profile):
    return lock["packages"]["vllm_fallback" if profile == "vllm019" else "vllm_primary"]


def _verify_environment(output_dir, info):
    if not Path(info["python"]).exists():
        raise RuntimeError("Local model environment disappeared; rerun the installation step")
    frozen = _run([info["python"], "-m", "pip", "freeze", "--all"], output_dir, "environment-check.log", capture=True)
    if hashlib.sha256(frozen.encode()).hexdigest() != info["freeze_sha256"]:
        raise RuntimeError("Model dependencies changed after installation; restore the resolved environment or use a new OUTPUT_DIR")


def _environment_name(model, profile, lock):
    if model == "kev":
        return "kev"
    if model == "startlux" and profile == "reference":
        return "startlux-reference"
    return "vllm-" + _vllm_version(lock, profile)


def _needed_environments(models, profiles, lock):
    needed = {_environment_name(m, profiles[m], lock) for m in models}
    if "startlux" in models:
        needed.add("startlux-reference")
    return needed


def _portable_requirements(frozen):
    """Keep exact package pins; locked local sources are installed separately.

    pip freeze can see the controller's egg-info in its working directory even
    though decision-study is not installed in the isolated model environment.
    It must never become a requirement fetched from PyPI during recovery.
    """
    lines = []
    for line in frozen.splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^decision[-_.]study(?:\s|\[|[=<>!~@]|$)", stripped):
            continue
        if " @ file:" in stripped:
            continue
        if stripped:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _pinned_versions(frozen):
    pins = {}
    for line in _portable_requirements(frozen).splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            pins[re.sub(r"[-_.]+", "-", match[1]).lower()] = match[2]
    return pins


def _check_preserved_pins(previous, current):
    actual = _pinned_versions(current)
    changed = {name: {"expected": version, "actual": actual.get(name)}
               for name, version in _pinned_versions(previous).items() if actual.get(name) != version}
    if changed:
        raise RuntimeError(f"Resolved dependency pins changed during environment recreation: {changed}")


def _repair_marker(output_dir, env_name):
    return Path(output_dir) / "runtime" / f"repair-{env_name}.json"


def _pending_repair(output_dir, env_name):
    path = _repair_marker(output_dir, env_name)
    record = _read(path) if path.exists() else None
    return record if record and record.get("status") == "pending" else None


def _assert_repair_safe(output_dir, names, lock):
    affected = [m for m in MODELS if _environment_name(m, _resolve_profile(output_dir, m), lock) in names
                or (m == "startlux" and "startlux-reference" in names)]
    for model in affected:
        predictions = Path(output_dir) / "predictions" / model
        if ((predictions / "config.json").exists() or any(predictions.glob("shard-*.json"))
                or any((predictions / "shards").glob("*"))):
            raise RuntimeError(f"Cannot repair an environment used by {model} after formal predictions started; use a new OUTPUT_DIR")
    return affected


def _check_python_headers(output_dir, python, env_name):
    # Match both CPython's compile include and sysconfig's include used by Triton.
    script = ("import json, pathlib, sys, sysconfig; "
              "paths = sorted(set(p for p in [sysconfig.get_path('include'), sysconfig.get_config_var('INCLUDEPY')] if p)); "
              "headers = [str(pathlib.Path(p) / 'Python.h') for p in paths]; "
              "missing = [p for p in headers if not pathlib.Path(p).is_file()]; "
              "print(json.dumps({'python_version': sys.version.split()[0], 'executable': sys.executable, "
              "'base_prefix': sys.base_prefix, 'headers': headers, 'missing': missing})); "
              "sys.exit(0 if headers and not missing else 1)")
    try:
        result = _run([python, "-c", script], output_dir, f"python-headers-{env_name}.log", capture=True)
    except RuntimeCommandError as exc:
        error = RuntimeError(f"Python.h preflight failed for {env_name}. Run repair-python for the affected model; "
                             f"no further rebuild was attempted. See {exc.log_path}")
        error.log_path = exc.log_path
        raise error from exc
    report = json.loads(result)
    if not report.get("headers") or report.get("missing"):
        raise RuntimeError(f"Python.h missing for {env_name}; use the explicit repair-python command")
    _write(Path(output_dir) / "runtime" / f"python-headers-{env_name}.json", report)
    return report


def repair_python(project_dir, output_dir, models, managed_python_version="3.12.8"):
    """Explicit, resumable venv-only replacement; preserve checkpoints and data.

    Called under the CLI's exclusive run lock and budget/session deadline. A
    persisted pending marker authorizes only these exact environment paths to
    be cleared on this call or a resumed installation after interruption.
    """
    if not models or set(models) - set(MODELS):
        raise ValueError("Unknown model")
    if not re.fullmatch(r"3\.12\.\d+", managed_python_version):
        raise ValueError("managed-python-version must be an exact CPython 3.12 patch version, e.g. 3.12.8")
    output_dir, project_dir = Path(output_dir).resolve(), Path(project_dir).resolve()
    lock = freeze_lock(project_dir, output_dir)
    profiles = {m: _resolve_profile(output_dir, m) for m in models}
    needed = _needed_environments(models, profiles, lock)
    affected = _assert_repair_safe(output_dir, needed, lock)
    if not shutil.which("uv"):
        raise RuntimeError("uv is required. Run: python -m pip install uv")
    installations_path = output_dir / "runtime/installations.json"
    installations = _read(installations_path) if installations_path.exists() else {"environments": {}, "models": {}, "lock_hash": lock["lock_hash"]}
    if installations["lock_hash"] != lock["lock_hash"]:
        raise RuntimeError("Installed environments do not match immutable lock")
    # Validate every requested replacement before writing any authorization.
    for name in sorted(needed):
        env_dir = _work_dir(output_dir) / "envs" / name
        if env_dir.is_symlink() or env_dir.parent.is_symlink():
            raise RuntimeError(f"Refusing to replace a symlinked environment: {env_dir}")
        info = installations["environments"].get(name, {})
        if info.get("python", str(env_dir / "bin/python")) != str(env_dir / "bin/python"):
            raise RuntimeError("Recorded environment path differs; refusing to replace another runtime")
        pending = _pending_repair(output_dir, name)
        if pending and pending["managed_python_version"] != managed_python_version:
            raise RuntimeError("A repair is already pending with a different Python version; resume that repair")
    for name in sorted(needed):
        if _pending_repair(output_dir, name):
            continue
        env_dir = _work_dir(output_dir) / "envs" / name
        resolved = output_dir / "runtime/resolved" / f"{name}.txt"
        # A partial old installation may predate installations.json. Snapshot its
        # exact resolution before removing anything; never guess old package versions.
        if not resolved.exists() and (env_dir / "bin/python").exists():
            frozen = _run([env_dir / "bin/python", "-m", "pip", "freeze", "--all"],
                          output_dir, "repair-python.log", capture=True)
            _write_text(resolved, frozen)
        elif not resolved.exists() and name in installations["environments"]:
            raise RuntimeError(f"Cannot preserve dependency pins for {name}: resolved requirements and old interpreter are missing")
        archive = output_dir / "runtime/resolved" / f"repair-{name}-{time.time_ns()}"
        archive.mkdir(parents=True)
        for path in (installations_path, output_dir / "runtime/lock.json", resolved,
                     output_dir / "runtime/selected_profiles.json", output_dir / "runtime/selected_memory.json",
                     output_dir / "runtime" / f"gpu-check-{name}.json"):
            if path.is_file():
                shutil.copy2(path, archive / path.name)
        for path in sorted((output_dir / "runtime/logs").glob("*.log")):
            if name in path.name or path.name in {"install.log", "environment-check.log", "repair-python.log"}:
                with path.open("rb") as source:
                    source.seek(max(0, path.stat().st_size - 131072))
                    (archive / (path.name + ".tail.txt")).write_bytes(source.read())
        marker = {"status": "pending", "started_at": _now(), "environment": name,
                  "environment_path": str(env_dir), "managed_python_version": managed_python_version,
                  "affected_models": affected, "archive": str(archive), "lock_hash": lock["lock_hash"],
                  "resolved_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved.exists() else None}
        _write(_repair_marker(output_dir, name), marker)
    result = install_environments(project_dir, output_dir, models, download_models=False)
    return {"status": "complete", "environments": sorted(needed), "affected_models": affected,
            "managed_python_version": managed_python_version, "checkpoints_downloaded": False,
            "next_step": "Run smoke for the affected models before formal predictions.",
            "installations": result}


def install_environments(project_dir, output_dir, models, profile=None, *, download_models=True):
    """Explicitly install isolated Linux CUDA environments and download pinned weights."""
    if (profile is not None and profile not in PROFILES) or not models or set(models) - set(MODELS):
        raise ValueError("Unknown runtime profile or model")
    project_dir, output_dir = Path(project_dir).resolve(), Path(output_dir).resolve()
    doctor(output_dir, require_gpu=True)
    lock = freeze_lock(project_dir, output_dir)
    if profile is not None:
        for model in models:
            select_profile(output_dir, model, profile)
    profiles = {model: _resolve_profile(output_dir, model) for model in models}
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required. Run: python -m pip install uv")
    installations_path = output_dir / "runtime/installations.json"
    installations = _read(installations_path) if installations_path.exists() else {"environments": {}, "models": {}, "lock_hash": lock["lock_hash"]}
    if installations["lock_hash"] != lock["lock_hash"]:
        raise RuntimeError("Installed environments do not match immutable lock")
    needed = _needed_environments(models, profiles, lock)
    for env_name in sorted(needed):
        env_dir = _work_dir(output_dir) / "envs" / env_name
        python = env_dir / "bin/python"
        pending = _pending_repair(output_dir, env_name)
        if pending:
            _assert_repair_safe(output_dir, {env_name}, lock)
            if pending["environment_path"] != str(env_dir) or pending["lock_hash"] != lock["lock_hash"]:
                raise RuntimeError("Pending repair does not match this runtime path and source lock")
        if pending or env_name not in installations["environments"] or not python.exists():
            if env_dir.is_symlink() or env_dir.parent.is_symlink():
                raise RuntimeError(f"Refusing to recreate a symlinked environment: {env_dir}")
            if env_dir.exists() and not pending:
                raise RuntimeError(f"Existing incomplete environment {env_name} was preserved. Use repair-python to authorize rebuilding it.")
            resolved = output_dir / "runtime/resolved" / f"{env_name}.txt"
            old_frozen = resolved.read_text() if resolved.exists() else None
            if pending and pending["resolved_sha256"] is not None and hashlib.sha256((old_frozen or "").encode()).hexdigest() != pending["resolved_sha256"]:
                # Recover a crash between writing the final freeze and marking
                # the repair complete, without accepting any external pin change.
                if old_frozen != pending.get("new_freeze"):
                    raise RuntimeError("Resolved dependency pins changed since repair authorization")
                old_frozen = (Path(pending["archive"]) / resolved.name).read_text()
            python_version = (pending["managed_python_version"] if pending else
                              installations["environments"].get(env_name, {}).get("python_version", lock["python"]))
            command = [uv, "venv", "--managed-python", "--python", python_version, "--seed"]
            if pending:
                command.append("--clear")
            _run([*command, env_dir], output_dir, "install.log")
            headers = _check_python_headers(output_dir, python, env_name)
            if pending and headers["python_version"] != pending["managed_python_version"]:
                raise RuntimeError("Managed interpreter version differs from explicit repair request")
            if env_name == "kev":
                model = lock["models"]["kev"]
                source = _checkout(output_dir, model["code_repo"], model["code_revision"])
                packages = [f"{source}[serve]", "huggingface_hub"]
            else:
                model = lock["models"]["startlux"]
                source = _checkout(output_dir, model["code_repo"], model["code_revision"])
                packages = ([f"{source}[transformers]", "huggingface_hub"] if env_name == "startlux-reference" else
                            [str(source), f"vllm=={env_name[len('vllm-'):]}", "huggingface_hub"])
            # Reuse a prior explicit resolution when re-creating an environment on the same platform.
            if old_frozen is not None:
                # Local source URLs in pip freeze are host-specific. The immutable
                # repository source above is installed separately at its locked SHA.
                portable = resolved.with_name(env_name + "-portable.txt")
                _write_text(portable, _portable_requirements(old_frozen))
                packages += ["-r", str(portable)]
            _run([uv, "pip", "install", "--python", python, *packages], output_dir, "install.log")
            if env_name == "kev" and profiles.get("kev") not in ("kev_no_fused", "kev_eager"):
                _run([uv, "pip", "install", "--python", python, f"flash-linear-attention=={lock['packages']['kev_fla']}"], output_dir, "install.log")
            _run([uv, "pip", "check", "--python", python], output_dir, "install.log")
            frozen = _run([python, "-m", "pip", "freeze", "--all"], output_dir, "install.log", capture=True)
            if old_frozen is not None:
                _check_preserved_pins(old_frozen, frozen)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if pending:
                # Keep original pins intact until the GPU/Triton probe also passes.
                pending["new_freeze"] = frozen
                _write(_repair_marker(output_dir, env_name), pending)
            else:
                _write_text(resolved, frozen)
            installations["environments"][env_name] = {"python": str(python), "freeze_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
                                                       "created_at": _now(), "python_version": headers["python_version"], "managed_python": True}
            _write(installations_path, installations)
        else:
            if installations["environments"][env_name]["python"] != str(python):
                raise RuntimeError("Local runtime path changed while another installation still exists; use a new OUTPUT_DIR")
    for env_name in sorted(needed):
        info = installations["environments"][env_name]
        _check_python_headers(output_dir, info["python"], env_name)
        _verify_environment(output_dir, info)
        check_path = output_dir / "runtime" / f"gpu-check-{env_name}.json"
        probe_model = "kev" if env_name == "kev" else "startlux"
        _run([info["python"], project_dir / "scripts/model_probe.py", "gpu-check", "--model", probe_model,
              "--output-dir", output_dir, "--result", check_path], output_dir, f"gpu-check-{env_name}.log")
        info["gpu_probe"] = _read(check_path)
        _write(installations_path, installations)
        pending = _pending_repair(output_dir, env_name)
        if pending:
            _write_text(output_dir / "runtime/resolved" / f"{env_name}.txt", pending.pop("new_freeze"))
            pending.update(status="complete", completed_at=_now())
            _write(_repair_marker(output_dir, env_name), pending)
    if not download_models:
        return installations
    for model in models:
        env_name = _environment_name(model, profiles[model], lock)
        _verify_environment(output_dir, installations["environments"][env_name])
        python = installations["environments"][env_name]["python"]
        result_path = output_dir / "runtime" / f"download-{model}.json"
        _run([python, project_dir / "scripts/model_probe.py", "download", "--model", model, "--output-dir", output_dir,
              "--result", result_path], output_dir, "downloads.log")
        installations["models"][model] = _read(result_path)
        _write(installations_path, installations)
    return installations


def runtime_config(output_dir, model, profile=None):
    """Canonical *effective* computation identity plus hardware and dependency resolution."""
    profile = _resolve_profile(output_dir, model, profile)
    if model not in MODELS:
        raise ValueError("Unknown model/profile")
    out = Path(output_dir)
    lock = _read(out / "runtime/lock.json")
    _validate_lock(lock)
    installations = _read(out / "runtime/installations.json")
    env_name = _environment_name(model, profile, lock)
    env_info = installations["environments"][env_name]
    gpu = _read(out / "runtime/doctor.json").get("gpu", [])
    m = lock["models"][model]
    config = {"model": model, "model_revision": m["revision"], "model_repo": m["repo"],
              "code_revision": m.get("code_revision"), "base_revision": m.get("base_revision"),
              "base_repo": m.get("base_repo"), "precision": "bfloat16", "max_model_len": lock["max_model_len"],
              "prefix_cache": False, "truncate": False, "native_temperature": m.get("native_temperature"),
              "dependency_freeze_sha256": env_info["freeze_sha256"],
              "python_version": env_info.get("python_version"), "managed_python": env_info.get("managed_python", False),
              "library_versions": env_info.get("gpu_probe", {}).get("library_versions"),
              "cuda_runtime": env_info.get("gpu_probe", {}).get("cuda_runtime"),
              "gpu": [{k: g[k] for k in ("name", "memory_mb", "driver", "compute_capability")} for g in gpu[:1]]}
    if model == "kev":
        config.update(backend="torch", merged_lora=True, fused=profile not in ("kev_no_fused", "kev_eager"),
                      cuda_graphs=profile != "kev_eager", date_facts=False)
    elif model == "startlux":
        config.update(backend="transformers" if profile == "reference" else "vllm", bidirectional=True,
                      vllm_version=None if profile == "reference" else _vllm_version(lock, profile),
                      none_check=False, allow_none=False, think_mode="never", chunked_prefill=False,
                      speculative_decoding=False, gpu_memory_utilization=_memory_fraction(out, model, lock))
    else:
        config.update(backend="vllm", vllm_version=_vllm_version(lock, profile), thinking=False, temperature=0,
                      max_output_tokens=lock["max_output_tokens"], constrained_json=True,
                      gpu_memory_utilization=_memory_fraction(out, model, lock), chunked_prefill=False)
    return config


def _service_specs(project_dir, output_dir, model, profile, lock, installations):
    """Pure command construction, also covered by CPU tests."""
    m = lock["models"][model]
    env_name = _environment_name(model, profile, lock)
    python = installations["environments"][env_name]["python"]
    model_path = installations["models"][model]["model_path"]
    env = _env(output_dir)
    port = m["port"]
    root = f"http://127.0.0.1:{port}"
    if model == "kev":
        env.update({"KEV_BACKEND": "torch", "KEV_DTYPE": "bf16", "KEV_MERGE": "1",
                    "KEV_FUSED": "0" if profile in ("kev_no_fused", "kev_eager") else "1",
                    "KEV_CUDA_GRAPHS": "0" if profile == "kev_eager" else "1", "KEV_PREFIX_CACHE": "0",
                    "KEV_DATE_FACTS": "0", "KEV_TRUNCATE_STATES": "0"})
        return [{"name": model, "port": port, "command": [python, "-m", "kev.serve", "--run", model_path,
                  "--fallback", "/__decision_study_missing_checkpoint__", "--device", "cuda", "--host", "127.0.0.1", "--port", str(port)],
                 "env": env, "url": root, "health": "/v1/models"}]
    frontend = [str(Path(python).parent / "startlux-decide"), "serve", "--model", model_path,
                "--name", "startlux", "--host", "127.0.0.1", "--port", str(port), "--think-mode", "never", "--no-none-check",
                "--no-images", "--max-model-len", str(lock["max_model_len"]), "--max-inflight", "16"]
    if model == "startlux" and profile == "reference":
        return [{"name": model, "port": port, "command": frontend + ["--backend", "transformers", "--device", "cuda", "--dtype", "bfloat16"],
                 "env": env, "url": root, "health": "/health"}]
    engine_port = m["engine_port"] if model == "startlux" else port
    engine_url = f"http://127.0.0.1:{engine_port}"
    cmd = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path, "--served-model-name", model,
           "--host", "127.0.0.1", "--port", str(engine_port), "--dtype", "bfloat16", "--tensor-parallel-size", "1",
           "--max-model-len", str(lock["max_model_len"]), "--gpu-memory-utilization", str(_memory_fraction(output_dir, model, lock)),
           "--generation-config", "vllm", "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
           "--max-num-batched-tokens", str(lock["max_model_len"]), "--limit-mm-per-prompt", '{"image":0,"video":0}',
           "--disable-uvicorn-access-log"]
    if model == "startlux":
        env = {**env, "STARTLUX_PREFIX_BIDIR": "1", "VLLM_PLUGINS": "startlux_prefix_bidir"}
        cmd += ["--logprobs-mode", "processed_logprobs", "--max-logprobs", "26"]
    specs = [{"name": model + "-engine", "port": engine_port, "command": cmd, "env": env, "url": engine_url,
              "health": "/health", "plugin_required": model == "startlux"}]
    if model == "startlux":
        specs.append({"name": model, "port": port, "command": frontend + ["--backend", "vllm", "--vllm-url", engine_url],
                      "env": _env(output_dir), "url": root, "health": "/health"})
    return specs


def _check_port(port):
    # Some BSD-derived kernels allow a reusable loopback bind beside an active
    # wildcard listener. A TCP-only probe catches that without sending a request
    # to, attaching to, or stopping the unidentified service.
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=1)
    except ConnectionRefusedError:
        pass
    except OSError as exc:
        raise RuntimeError(f"Could not verify that port {port} is unused: {exc}") from exc
    else:
        connection.close()
        raise RuntimeError(f"Port {port} already occupied; refusing to attach to an unidentified model service")
    with socket.socket() as sock:
        # Match the server's normal restart semantics: an owned service can leave
        # accepted connections in TIME_WAIT after its listening socket has closed.
        # REUSEADDR permits those stale connections. Never use SO_REUSEPORT or
        # attach to a pre-existing service.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} already occupied; refusing to attach to an unidentified model service") from exc


def _wait_ready(process, spec, log_path, timeout):
    allowed = _timeout_seconds(timeout)
    deadline = time.monotonic() + allowed
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{spec['name']} exited ({process.returncode}); see {log_path}")
        try:
            with urllib.request.urlopen(spec["url"] + spec["health"], timeout=2) as response:
                body = response.read()
                if response.status == 200:
                    if spec.get("plugin_required"):
                        log = log_path.read_text(errors="replace")
                        if "prefix-bidirectional prompt attention ON" not in log:
                            raise RuntimeError(f"StartLux worker plugin did not activate; refusing causal inference. See {log_path}")
                    return json.loads(body) if body.strip() else {"status": "ok"}
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(1)
    raise RuntimeError(f"Service startup exceeded startup/budget deadline after {allowed:.1f}s; see {log_path}")


@contextmanager
def managed_services(project_dir, output_dir, models, profile=None):
    """Start only owned process groups; validate identities; always stop every group."""
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError("Invalid model list")
    if len(models) > 2 or (len(models) == 2 and "qwen" not in models):
        raise ValueError("At most one decision model plus Qwen may co-reside")
    output_dir = Path(output_dir).resolve()
    report = doctor(output_dir, require_gpu=True)
    lock = _read(output_dir / "runtime/lock.json")
    installations = _read(output_dir / "runtime/installations.json")
    if installations["lock_hash"] != lock["lock_hash"]:
        raise RuntimeError("Runtime installation lock mismatch")
    if len(models) == 2 and report["gpu"][0]["memory_mb"] < 44000:
        raise RuntimeError("Live paired deployment requires the planned 48 GB GPU; individual predictions can run separately")
    if len(models) == 2 and any(_memory_fraction(output_dir, m, lock) > 0.40 for m in models if m != "kev"):
        raise RuntimeError("Live paired deployment requires frozen vLLM reservations <=0.40 each. A standalone 0.80 Colab run needs a new matched deployment run; do not reuse its gate silently.")
    processes, endpoints, metadata = [], {}, {"started_at": _now(), "models": {}, "processes": []}
    try:
        # Kev first: it does not pre-reserve a configurable GPU fraction.
        for model in sorted(models, key=lambda m: (m == "qwen", m)):
            selected_profile = _resolve_profile(output_dir, model, profile)
            env_name = _environment_name(model, selected_profile, lock)
            _verify_environment(output_dir, installations["environments"][env_name])
            specs = _service_specs(project_dir, output_dir, model, selected_profile, lock, installations)
            for spec in specs:
                _check_port(spec["port"])
                log_path = output_dir / "runtime/logs" / f"{spec['name']}-{time.time_ns()}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("w")
                try:
                    process = subprocess.Popen(spec["command"], env=spec["env"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    log.close()
                processes.append(process)
                metadata["processes"].append({"name": spec["name"], "pid": process.pid, "command": spec["command"], "log": str(log_path)})
                _write(output_dir / "runtime/last_services.json", metadata)
                health = _wait_ready(process, spec, log_path, lock["startup_timeout_seconds"])
                if model == "kev":
                    card = health.get("models", [{}])[0]
                    expected = installations["models"]["kev"]["model_path"]
                    if card.get("run") != expected or card.get("base") != lock["models"]["kev"]["base_repo"]:
                        raise RuntimeError("Kev served checkpoint differs from immutable lock")
                    if card.get("truncate_states") or card.get("prefix_cache", {}).get("size") != 0:
                        raise RuntimeError("Kev truncation/cache protocol mismatch")
                if model == "startlux" and spec["name"] == model:
                    if health.get("prompt_attention") != "bidirectional" or health.get("defaults", {}).get("think_mode") != "never":
                        raise RuntimeError("StartLux health reports the wrong attention or thinking mode")
                metadata["models"][model] = {"health": health, "config": runtime_config(output_dir, model, selected_profile)}
            endpoints[model] = f"http://127.0.0.1:{lock['models'][model]['port']}"
        metadata["ready_at"] = _now()
        _write(output_dir / "runtime/last_services.json", metadata)
        yield endpoints
    finally:
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in reversed(processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        metadata["stopped_at"] = _now()
        _write(output_dir / "runtime/last_services.json", metadata)


def token_preflight(project_dir, output_dir, models, profile=None):
    out = Path(output_dir).resolve()
    lock = _read(out / "runtime/lock.json")
    installations = _read(out / "runtime/installations.json")
    results = {}
    for model in models:
        selected_profile = _resolve_profile(output_dir, model, profile)
        env_name = _environment_name(model, selected_profile, lock)
        _verify_environment(out, installations["environments"][env_name])
        python = installations["environments"][env_name]["python"]
        result = out / "runtime" / f"tokens-{model}.json"
        _run([python, Path(project_dir).resolve() / "scripts/model_probe.py", "tokens", "--model", model,
              "--output-dir", out, "--result", result], out, f"tokens-{model}.log")
        results[model] = _read(result)
        results[model]["effective_config_hash"] = _hash(runtime_config(out, model, selected_profile))
        results[model]["examples_sha256"] = hashlib.sha256((out / "data/examples.jsonl").read_bytes()).hexdigest()
        _write(result, results[model])
        if results[model]["status"] != "ok":
            raise RuntimeError(f"Exact token preflight failed; see {result}. No truncation or resampling was applied.")
    return results


def token_preflight_examples(project_dir, output_dir, model, examples, profile=None):
    """Exact rendering for new interactive inputs; never overwrite formal token reports."""
    if not examples:
        raise ValueError("No interactive examples supplied")
    out = Path(output_dir).resolve()
    lock = _read(out / "runtime/lock.json")
    installations = _read(out / "runtime/installations.json")
    selected_profile = _resolve_profile(out, model, profile)
    env_name = _environment_name(model, selected_profile, lock)
    _verify_environment(out, installations["environments"][env_name])
    suffix = _hash(examples)[:16]
    inputs = out / "runtime" / f"interactive-inputs-{model}-{suffix}.jsonl"
    inputs.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
    result = out / "runtime" / f"tokens-interactive-{model}-{suffix}.json"
    _run([installations["environments"][env_name]["python"], Path(project_dir).resolve() / "scripts/model_probe.py",
          "tokens", "--model", model, "--output-dir", out, "--examples", inputs, "--result", result],
         out, f"tokens-interactive-{model}.log")
    report = _read(result)
    report["effective_config_hash"] = _hash(runtime_config(out, model, selected_profile))
    _write(result, report)
    if report["status"] != "ok":
        raise RuntimeError(f"Interactive input exceeds the frozen context limit; see {result}. No text was truncated.")
    return report


def reference_predictions(project_dir, output_dir, model, examples):
    if model not in ("kev", "startlux"):
        raise ValueError("Only decision-model references return probability distributions")
    out = Path(output_dir).resolve()
    installations = _read(out / "runtime/installations.json")
    env_name = "kev" if model == "kev" else "startlux-reference"
    _verify_environment(out, installations["environments"][env_name])
    inputs = out / "runtime" / f"reference-inputs-{model}.jsonl"
    inputs.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
    result = out / "runtime" / f"reference-{model}.json"
    _run([installations["environments"][env_name]["python"], Path(project_dir).resolve() / "scripts/model_probe.py",
          "reference", "--model", model, "--output-dir", out, "--examples", inputs, "--result", result], out, f"reference-{model}.log")
    return _read(result)["predictions"]
