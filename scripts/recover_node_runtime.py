#!/usr/bin/env python3
"""Restore the deleted node runtime that direct-matrix-v1 reads, from pinned archived evidence.

Design: .validation/runtime-recovery-2026-11-11/design.md (reviewed). Commands:

  plan     read-only; writes nothing; reports blockers, pending actions, disk, locks and matrix-output state
  execute  the only mutating entry; sequential resumable phases, then the full verify
  verify   read-only READY / NOT READY report (temporary check logs only, outside every runtime root)

The matrix checkout (--matrix-checkout, pinned commit) supplies every frozen helper and verifier;
this script runs from a separate recovery checkout and never edits either checkout. Restored
metadata stays byte-identical to the archive; environments are accepted only when the frozen
legacy-context ``pip freeze --all`` hashes to the archived value. Nothing is ever re-pinned.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timedelta, timezone

TOOL = "runtime-recovery-2026-11-11"
MATRIX_COMMIT = "4b4e38ab9c040fe7eb02ae5075e87c595399d980"
PINS_FILE = Path(__file__).resolve().parents[1] / "configs/runtime_recovery_2026_11_11_pins.json"
PINS_SHA256 = "7f3e27c095ead75991eef9a822695d211bb7fe649506687bead57806810c4e56"

# Layout below the runtime root (all relative; joined only through ``Layout.path``).
V1, V2, V3 = "results/solo-v1", "results/solo-v2", "results/solo-v3"
SIZE = "results/aux-size-control-v1-r2"
MATRIX_OUTPUT = "results/aux-direct-matrix-v1"
STATE = "results/runtime-recovery-2026-11-11"
ARCHIVE = V1 + "/exports/benchmark_bundle.zip"
V1_RUNTIME = V1 + "/runtime"
V1_WORK = V1 + "/runtime/work"
V3_WORK = V3 + "/runtime-work"
OWNED = (V1_RUNTIME, V3_WORK)
LOCKED_OUTPUTS = (V1, V3, SIZE, MATRIX_OUTPUT)
# Historical session/PID records and the never-existing V1 selections stay absent.
ABSENT_MEMBERS = ("runtime/session.json", "runtime/last_services.json",
                  "runtime/selected_profiles.json", "runtime/selected_memory.json")
TOOLKIT_SETUP = "runtime/cuda-toolkit-setup-1793650332085549754.json"
TOOLKIT_REQUIREMENTS = "configs/cuda_build_toolkit_13_0_flashinfer.txt"
TOOLKIT_IDENTITY = ("wheels", "distributions", "tree_sha256", "installer_sha256", "requirements_sha256")
V3_DOWNLOAD = "runtime-controls/qwen9b/runtime/download-qwen9b.json"
SIZE_INSTALLATIONS = "runtime-controls/startlux9b/runtime/installations.json"
MANAGED_PYTHON = "3.12.8"
ENVIRONMENTS = {  # V1 install commands (archived runtime/logs/install.log), exact pins added.
    "vllm-0.31.0": {"phase": "env-vllm", "source": "StartLuxLabs/StartLux-Decision", "extra": None,
                    "packages": ("vllm==0.31.0", "huggingface_hub"), "after": ()},
    "kev": {"phase": "env-kev", "source": "jaredpalmer/kev", "extra": "serve",
            "packages": ("huggingface_hub",), "after": ("flash-linear-attention==0.5.2",)},
}
PHASES = ("metadata", "sources", "env-vllm", "toolkit", "env-kev", "weights-kev", "weights-qwen9b")
REQUIRES = {"metadata": (), "sources": ("metadata",), "env-vllm": ("sources",), "toolkit": ("env-vllm",),
            "env-kev": ("sources",), "weights-kev": ("env-kev",), "weights-qwen9b": ("env-vllm",)}
GIB = 1 << 30
DISK_GIB = {"metadata": 1, "sources": 2, "env-vllm": 30, "toolkit": 6, "env-kev": 20,
            "weights-kev": 14, "weights-qwen9b": 24}  # estimates incl. uv cache; quotas are invisible
TIMEOUT = {"git": 900, "uv-venv": 1800, "uv-install": 4 * 3600, "uv-check": 900, "python": 900,
           "toolkit": 1500, "download-kev": 3 * 3600, "download-qwen9b": 6 * 3600, "query": 120}
# Ambient settings that would change what is resolved, fetched or discovered.
BLOCKED_ENV = {"PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP", "DECISION_STUDY_RUNTIME_DIR",
               "HF_ENDPOINT", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_CACHE"}
BLOCKED_PREFIXES = ("UV_", "PIP_")
ALLOWED_PREFIXED = {"UV_NATIVE_TLS", "UV_HTTP_TIMEOUT", "UV_HTTP_RETRIES", "UV_NO_PROGRESS"}
DROPPED_PREFIXES = ("KEV_", "STARTLUX_", "VLLM_", "GIT_")  # V1 _env() drops the first three
CHILD_SETTINGS = {"CUDA_VISIBLE_DEVICES": "", "UV_CONCURRENT_DOWNLOADS": "4", "UV_CONCURRENT_BUILDS": "1",
                  "UV_CONCURRENT_INSTALLS": "2", "MAX_JOBS": "2", "OMP_NUM_THREADS": "2",
                  "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
                  "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1",
                  "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
ARCHIVE_LIMITS = {"members": 5000, "member_bytes": 64 << 20, "total_bytes": 1 << 30}


class RecoveryError(RuntimeError):
    """A blocker or failed check; the message names what to inspect."""


class TerminalMismatch(RecoveryError):
    """Rebuilt content differs from the archived identity. Recorded, never retried or re-pinned."""


class ChildError(RecoveryError):
    def __init__(self, message, *, log_path=None, returncode=None):
        super().__init__(message + (f"; log: {log_path}" if log_path else ""))
        self.log_path, self.returncode = log_path, returncode


class StopUnverified(ChildError):
    """An owned child group could not be proven stopped and reaped; a blocked child may still write."""


class Interrupted(BaseException):
    """SIGTERM/SIGHUP/SIGINT received by the recovery controller."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_fd(fd, limit=None):
    """Hash an open file from its start; returns (sha256, bytes read)."""
    digest, total = hashlib.sha256(), 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return digest.hexdigest(), total
        total += len(chunk)
        if limit is not None and total > limit:
            raise RecoveryError("File grew beyond its pinned size while hashing")
        digest.update(chunk)


def open_regular(path):
    """Open an existing regular file without following a final symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RecoveryError(f"Not a regular file: {path}")
    return fd


def sha256_file(path):
    fd = open_regular(path)
    try:
        return sha256_fd(fd)[0]
    finally:
        os.close(fd)


def read_json(path):
    fd = open_regular(path)
    with os.fdopen(fd, "rb") as handle:
        return json.loads(handle.read())


def write_json_atomic(path, value):
    """Recovery-owned state/receipts only: a new temporary file renamed over the target."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative_parts(relative):
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise RecoveryError(f"Unsafe relative path: {relative!r}")
    return parts


def lstat_chain(root, relative):
    """lstat every component below ``root``; refuse symlinks. Returns the final stat or None if absent."""
    current, info = Path(root), None
    parts = relative_parts(relative)
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise RecoveryError(f"Refusing symlinked path component: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise RecoveryError(f"Path component is not a directory: {current}")
    return info


# --------------------------------------------------------------------------- supervised children

def group_members(pgid):
    """Live (non-zombie) members of process group ``pgid``: /proc on Linux, ``ps`` elsewhere."""
    members = []
    if sys.platform.startswith("linux") and os.path.isdir("/proc"):
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                with open(os.path.join(entry.path, "stat"), "rb") as handle:
                    data = handle.read()
                fields = data[data.rindex(b")") + 2:].split()
                state, pgrp = fields[0].decode(), int(fields[2])
            except (OSError, ValueError, IndexError):
                continue  # exited while scanning
            if pgrp == pgid and state not in ("Z", "X"):
                members.append(int(entry.name))
        return sorted(members)
    listing = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "pgid=", "-o", "stat="], check=True, timeout=60,
                             stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             env={**os.environ, "LC_ALL": "C"}).stdout
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1].isdigit() and int(fields[1]) == pgid and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return sorted(members)


class ChildResult:
    def __init__(self, returncode, stdout, stderr, log_path, seconds):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr
        self.log_path, self.seconds = log_path, seconds


class Supervisor:
    """Run one child at a time in its own session, bounded by an absolute deadline.

    The leader's exit is observed with waitid(WEXITED|WNOHANG|WNOWAIT), so it stays unreaped and its
    PID -- which is also the process-group ID -- cannot be reused while surviving group members are
    signalled (SIGTERM, then SIGKILL). Only then is the leader reaped. An exit observed after the
    deadline is not accepted, and leftover group members fail the step even when they could be
    stopped. Members that left the group (setsid/setpgid) are outside this scope. No PID is ever
    read from a file or record.
    """

    CAPTURE_LIMIT = 64 << 20

    def __init__(self, *, poll=0.05, term_grace=10.0, kill_grace=10.0, clock=time.monotonic, niceness=10):
        if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
            raise RecoveryError("This platform lacks waitid(WNOWAIT); owned-child supervision is unavailable")
        nice = shutil.which("nice")
        if niceness and not nice:
            raise RecoveryError("The nice utility is missing; children cannot be lowered in priority")
        self.prefix = [nice, "-n", str(niceness)] if niceness else []
        self.poll, self.term_grace, self.kill_grace, self.clock = poll, term_grace, kill_grace, clock

    def run(self, argv, *, env, timeout, log_path=None, cwd=None, capture=False):
        argv = [os.fspath(a) for a in argv]
        if not timeout > 0:
            raise RecoveryError("A child needs a positive time budget")
        log = None
        if log_path is not None:
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "ab", buffering=0)
            log.write(f"\n{utc_now()} COMMAND {json.dumps(argv)}\n".encode())
        sinks = {"out": bytearray(), "err": bytearray()}
        started = self.clock()
        deadline = started + timeout
        try:
            process = subprocess.Popen(self.prefix + argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True, close_fds=True)
        except BaseException:
            if log:
                log.close()
            raise
        selector, outcome, failure = None, None, None
        try:
            selector = selectors.DefaultSelector()
            for stream, name in ((process.stdout, "out"), (process.stderr, "err")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while True:
                exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
                now = self.clock()
                if exited:
                    outcome = "late-exit" if now > deadline else "exited"
                    break
                if now >= deadline:
                    outcome = "timeout"
                    break
                self._pump(selector, sinks, log, capture, min(self.poll, deadline - now))
        except BaseException as exc:  # Interrupted, KeyboardInterrupt or an unexpected controller error
            outcome, failure = "interrupted", exc
        finally:
            # Signals stay queued while cleanup runs. Restoring the mask may run a handler at once
            # (Interrupted); that is recorded and raised only after the cleanup outcome is known.
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP})
            pending = None
            try:
                cleanup = self._cleanup(process, selector, sinks, log, capture, outcome, started)
            except BaseException as exc:  # defensive: cleanup itself is written not to raise
                cleanup = {"returncode": None, "extra": [], "remaining": None, "verified": False, "seconds": 0.0,
                           "errors": [f"cleanup crashed: {type(exc).__name__}: {exc}"]}
            finally:
                try:
                    signal.pthread_sigmask(signal.SIG_SETMASK, previous)
                except BaseException as exc:
                    pending = exc
        where = {"log_path": log_path, "returncode": cleanup["returncode"]}
        if not cleanup["verified"]:
            message = (f"Owned process group {process.pid} could not be proven stopped and reaped "
                       f"(outcome={outcome}, remaining={cleanup['remaining']}, errors={cleanup['errors']})")
            for label, exc in (("original failure", failure), ("signal received during cleanup", pending)):
                if exc is not None:
                    message += f"; {label}: {type(exc).__name__}: {exc}"
            raise StopUnverified(message, **where) from (failure or pending)
        if failure is not None:
            notes = ([f"child cleanup errors: {cleanup['errors']}"] if cleanup["errors"] else []) + (
                [f"signal received during cleanup: {pending}"] if pending is not None else [])
            for note in notes:
                if hasattr(failure, "add_note"):
                    failure.add_note(note)
            raise failure
        if pending is not None:  # delivered only now, after a verified stop
            raise pending
        if cleanup["errors"]:
            raise ChildError(f"Child cleanup failed after a confirmed stop: {'; '.join(cleanup['errors'])}", **where)
        if outcome == "timeout":
            raise ChildError(f"Exceeded its {timeout:.0f}s budget; owned process group stopped", **where)
        if outcome == "late-exit":
            raise ChildError("Exit observed after the deadline; result not accepted", **where)
        if cleanup["extra"]:
            raise ChildError(f"Leader exited with group members still running (stopped): {cleanup['extra']}", **where)
        stdout = bytes(sinks["out"]).decode(errors="replace") if capture else None
        stderr = (bytes(sinks["err"]).decode(errors="replace") if capture
                  else bytes(sinks["err"][-4000:]).decode(errors="replace").strip())
        return ChildResult(cleanup["returncode"], stdout, stderr, log_path, cleanup["seconds"])

    def _cleanup(self, process, selector, sinks, log, capture, outcome, started):
        """Stop, drain, bounded-reap and close; each step runs even when an earlier one failed.

        Signals go only to the owned group and only while its leader is unreaped (its PID, which is
        the group ID, cannot be reused until then). ``verified`` requires an empty live group and a
        reaped leader; anything less is reported as a stop that could not be proven.
        """
        errors, extra, remaining = [], None, []
        try:
            extra = [pid for pid in group_members(process.pid) if pid != process.pid]
        except Exception as exc:
            errors.append(f"group listing failed: {type(exc).__name__}: {exc}")
        if outcome != "exited" or extra is None or extra:
            try:
                remaining = self._stop_group(process.pid).get("remaining")
            except Exception as exc:
                errors.append(f"group stop failed: {type(exc).__name__}: {exc}")
                remaining = None
                with contextlib.suppress(OSError):
                    os.killpg(process.pid, signal.SIGKILL)
        if selector is not None:
            try:
                self._drain(selector, sinks, log, capture)
            except Exception as exc:
                errors.append(f"output drain failed: {type(exc).__name__}: {exc}")
        returncode = None
        try:
            returncode = process.wait(timeout=max(self.kill_grace, 0.1))
        except subprocess.TimeoutExpired:
            errors.append("leader was not reaped within the bound; it is still alive")
        except Exception as exc:
            errors.append(f"reap failed: {type(exc).__name__}: {exc}")
        for close in (getattr(selector, "close", None), process.stdout.close, process.stderr.close):
            try:
                if close is not None:
                    close()
            except Exception as exc:
                errors.append(f"close failed: {type(exc).__name__}: {exc}")
        seconds = self.clock() - started
        if log is not None:
            try:
                log.write(f"{utc_now()} END outcome={outcome} returncode={returncode} seconds={seconds:.1f} "
                          f"leftover={extra} remaining={remaining} errors={errors}\n".encode())
            except Exception as exc:
                errors.append(f"log write failed: {type(exc).__name__}: {exc}")
            try:
                log.close()
            except Exception as exc:
                errors.append(f"log close failed: {type(exc).__name__}: {exc}")
        return {"returncode": returncode, "extra": extra or [], "remaining": remaining, "errors": errors,
                "verified": returncode is not None and remaining == [], "seconds": seconds}

    def _pump(self, selector, sinks, log, capture, timeout):
        for key, _ in selector.select(timeout=max(timeout, 0)):
            try:
                chunk = os.read(key.fileobj.fileno(), 1 << 16)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            if log:
                log.write(chunk)
            sink = sinks[key.data]
            if capture or key.data == "err":
                sink.extend(chunk)
                if not capture and len(sink) > 1 << 16:
                    del sink[:-(1 << 15)]
                if len(sink) > self.CAPTURE_LIMIT:
                    raise RecoveryError("Child output exceeded the capture limit")

    def _drain(self, selector, sinks, log, capture, limit=2.0):
        end = self.clock() + limit
        while selector.get_map() and self.clock() < end:
            self._pump(selector, sinks, log, capture, 0.1)

    def _stop_group(self, pgid):
        """SIGTERM, then SIGKILL, the owned group; ``remaining`` is None when the listing cannot verify it."""
        remaining = None
        for sig, grace in ((signal.SIGTERM, self.term_grace), (signal.SIGKILL, self.kill_grace)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:  # no member at all, not even the unreaped leader
                return {"remaining": []}
            except PermissionError:  # macOS: EPERM for a zombie-only group
                pass
            end = self.clock() + grace
            while True:
                try:
                    remaining = group_members(pgid)
                except Exception:
                    remaining = None
                if remaining == [] or self.clock() >= end:
                    break
                time.sleep(0.05)
            if remaining == []:
                break
        return {"remaining": remaining}


# --------------------------------------------------------------------------- layout, pins, environment

class Layout:
    def __init__(self, root, checkout, pins, pins_sha256=None):
        """``pins_sha256`` is the digest load_pins() verified; recovery state binds itself to it."""
        self.root, self.checkout, self.pins, self.pins_sha256 = Path(root), Path(checkout), pins, pins_sha256

    def path(self, relative):
        return self.root.joinpath(*relative_parts(relative))

    @property
    def state(self):
        return self.path(STATE)

    def env_dir(self, name):
        return self.path(f"{V1_WORK}/envs/{name}")

    def source_dir(self, repo, revision):
        return self.path(f"{V1_WORK}/sources/{repo.replace('/', '--')}-{revision[:12]}")

    def toolkit_dir(self):
        return self.path(f"{V1_WORK}/toolchains/cuda-13.0-flashinfer-curand10.4.0.35")


def load_pins(path=PINS_FILE, expected_sha256=PINS_SHA256):
    data = Path(path).read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise RecoveryError(f"Recovery pins file changed: {path}")
    pins = json.loads(data)
    members = pins.get("v1_runtime_members", {})
    if (not members or set(pins.get("v1_not_restored", {})) != set(ABSENT_MEMBERS[:2])
            or any(not name.startswith("runtime/") or name in ABSENT_MEMBERS for name in members)
            or not pins.get("matrix_files") or set(pins.get("derived_from", {})) != {"v1", "v3", "size"}):
        raise RecoveryError("Recovery pins are malformed")
    for name, entry in members.items():
        relative_parts(name)
        if (not isinstance(entry.get("size"), int) or entry["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))):
            raise RecoveryError(f"Malformed pin for {name}")
    return pins


def check_root(root):
    root = Path(root)
    if not root.is_absolute():
        raise RecoveryError("--runtime-root must be an absolute path")
    real = os.path.realpath(root)
    if real != str(root):
        raise RecoveryError(f"Runtime root {root} is not its own realpath ({real}); a symlinked component "
                            "(for example a /home alias) cannot be matched to the archived absolute paths")
    if not root.is_dir():
        raise RecoveryError(f"Runtime root is not a directory: {root}")
    return root


def archived_root(installations):
    """The runtime root every archived absolute path was recorded under."""
    roots = set()
    for name, record in installations.get("environments", {}).items():
        python = Path(record.get("python", ""))
        if (not python.is_absolute() or python.parts[-8:] != ("results", "solo-v1", "runtime", "work", "envs",
                                                             name, "bin", "python")):
            raise RecoveryError(f"Archived environment path has an unexpected layout: {python}")
        roots.add(python.parents[7])
    if len(roots) != 1:
        raise RecoveryError(f"Archived environments disagree about the runtime root: {sorted(map(str, roots))}")
    return roots.pop()


def ambient_blockers(environ=None):
    environ = os.environ if environ is None else environ
    found = sorted(k for k in environ if k in BLOCKED_ENV
                   or (k.startswith(BLOCKED_PREFIXES) and k not in ALLOWED_PREFIXED))
    return [f"Resolution-changing ambient setting is set: {k}; unset it for the recovery" for k in found]


def child_env(extra=None, *, keep=()):
    env = {k: v for k, v in os.environ.items()
           if k in keep or not (k.startswith(DROPPED_PREFIXES) or k in BLOCKED_ENV
                                or (k.startswith(BLOCKED_PREFIXES) and k not in ALLOWED_PREFIXED))}
    env.pop("DECISION_STUDY_DEADLINE_EPOCH", None)
    env.update(CHILD_SETTINGS)
    env.update(extra or {})
    return env


def v1_env(layout):
    """V1 _env() locations (work dir derived from the layout, never from DECISION_STUDY_RUNTIME_DIR)."""
    work = layout.path(V1_WORK)
    return {"HF_HOME": str(work / "cache/huggingface"), "UV_CACHE_DIR": str(work / "cache/uv"),
            "UV_PYTHON_INSTALL_DIR": str(work / "python")}


# --------------------------------------------------------------------------- matrix checkout

class GitFailed(RecoveryError):
    """git exited non-zero after the supervisor confirmed its process group stopped."""


def git(sup, repo, *args, timeout=TIMEOUT["query"], log_path=None):
    result = sup.run(["git", "--no-optional-locks", "-C", repo, *args], env=child_env(), timeout=timeout,
                     capture=True, log_path=log_path)
    if result.returncode:
        raise GitFailed(f"git {' '.join(map(str, args))} failed in {repo}: {result.stderr.strip()[-800:]}")
    return result.stdout


def overlaps(a, b):
    a, b = Path(os.path.abspath(a)), Path(os.path.abspath(b))
    return a == b or a in b.parents or b in a.parents


def check_matrix_checkout(sup, checkout, pins, root):
    """The matrix checkout must be the pinned, clean commit with the pinned file bytes."""
    checkout = Path(checkout)
    if not checkout.is_absolute() or os.path.realpath(checkout) != str(checkout):
        raise RecoveryError(f"Matrix checkout must be an absolute realpath: {checkout}")
    if overlaps(checkout, root):
        raise RecoveryError("The matrix checkout and the runtime root must not overlap")
    head = git(sup, checkout, "rev-parse", "HEAD").strip()
    if head != MATRIX_COMMIT:
        raise RecoveryError(f"Matrix checkout is at {head}, expected {MATRIX_COMMIT}; do not move it")
    if git(sup, checkout, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RecoveryError("Matrix checkout has local changes or untracked files; it must stay clean")
    flagged = [line for line in git(sup, checkout, "ls-files", "-v").splitlines()
               if line[:1].islower() or line[:1] == "S"]
    if flagged:
        raise RecoveryError(f"Matrix checkout hides changes (assume-unchanged/skip-worktree): {flagged[:5]}")
    for relative, expected in sorted(pins["matrix_files"].items()):
        info = lstat_chain(checkout, relative)
        if info is None or not stat.S_ISREG(info.st_mode) or sha256_file(checkout / relative) != expected:
            raise RecoveryError(f"Matrix checkout file differs from its pin: {relative}")
    return {"commit": head, "files": len(pins["matrix_files"])}


class Matrix:
    """Frozen modules imported from the verified matrix checkout (bytecode neither read nor written)."""

    NAMES = {"frozen": "decision_study.runtime", "compat": "decision_study.runtime_compat",
             "common": "decision_study.common", "runtime": "decision_study.direct_matrix.runtime",
             "study": "decision_study.direct_matrix.study", "size": "decision_study.size_control.runtime"}

    def __init__(self, checkout):
        import importlib
        self.checkout = Path(checkout)
        package = self.checkout / "decision_study"
        loaded = sys.modules.get("decision_study")
        if loaded is not None and Path(loaded.__file__).resolve().parent != package:
            raise RecoveryError(f"decision_study is already imported from {loaded.__file__}")
        sys.dont_write_bytecode = True
        sys.pycache_prefix = "/dev/null/recovery-no-bytecode"  # cached bytecode is never read or written
        if sys.path[:1] != [str(self.checkout)]:
            sys.path.insert(0, str(self.checkout))
        for attribute, name in self.NAMES.items():
            setattr(self, attribute, importlib.import_module(name))
        for name, module in list(sys.modules.items()):
            if (name == "decision_study" or name.startswith("decision_study.")) and getattr(module, "__file__", None):
                if not Path(module.__file__).resolve().is_relative_to(package):
                    raise RecoveryError(f"{name} was not imported from the matrix checkout")
        path = self.checkout / "scripts/prepare_cuda_toolkit.py"
        spec = importlib.util.spec_from_file_location("recovery_pinned_prepare_cuda_toolkit", path)
        self.toolkit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.toolkit)
        if Path(self.toolkit.__file__).resolve() != path:
            raise RecoveryError("Toolkit verifier was not loaded from the matrix checkout")


# --------------------------------------------------------------------------- frozen verifiers, supervised

@contextlib.contextmanager
def frozen_run_adapter(frozen, sup, log_dir, extra_env, captured):
    """Route frozen._run, which _verify_environment and _check_python_headers call, through the supervisor.

    The frozen verifier code and its comparisons run unchanged. Only process ownership (no nested
    unsupervised session), the log location (never the runtime tree) and GPU visibility (none)
    differ from frozen._run. The original function is restored on exit; source files are untouched.
    """
    original = frozen._run
    if getattr(original, "__module__", None) != frozen.__name__ or getattr(original, "__name__", None) != "_run":
        raise RecoveryError("frozen._run is already replaced; refusing to nest adapters")

    def adapter(args, output_dir, log_name, *, cwd=None, capture=False):
        command = [str(x) for x in args]
        result = sup.run(command, env=child_env(extra_env, keep=("PYTHONPATH",)), cwd=cwd,
                         timeout=TIMEOUT["python"], log_path=Path(log_dir) / log_name, capture=capture)
        stdout = result.stdout.replace("\r\n", "\n").replace("\r", "\n") if capture else None  # text=True
        captured.append({"command": command, "returncode": result.returncode, "stdout": stdout})
        if result.returncode:
            raise frozen.RuntimeCommandError(result.returncode, command, result.log_path,
                                             output=stdout, stderr=result.stderr)
        return stdout

    frozen._run = adapter
    try:
        yield adapter
    finally:
        frozen._run = original


def check_headers(matrix, layout, sup, log_dir, scratch, name, info):
    """Frozen _check_python_headers; its report file goes to ``scratch``, never to the restored runtime."""
    with frozen_run_adapter(matrix.frozen, sup, log_dir, v1_env(layout), []):
        report = matrix.frozen._check_python_headers(scratch, info["python"], name)
    if report.get("python_version") != info["python_version"]:
        raise TerminalMismatch(f"{name} interpreter reports {report.get('python_version')}, "
                               f"archived {info['python_version']}")
    return report


def check_freeze(matrix, layout, sup, log_dir, scratch, name, info, evidence_dir=None):
    """Frozen _verify_environment inside the frozen legacy_environment (original egg-info discoverable)."""
    captured, error = [], None
    binding = {"v1_dir": str(layout.path(V1)), "source_work_dir": str(layout.path(V1_WORK))}
    with matrix.compat.legacy_environment(binding, layout.checkout):
        with frozen_run_adapter(matrix.frozen, sup, log_dir, v1_env(layout), captured):
            try:
                matrix.frozen._verify_environment(scratch, info)
            except ChildError:
                raise
            except (RuntimeError, matrix.frozen.RuntimeCommandError) as exc:
                error = exc
    if error is None:
        return {"environment": name, "freeze_sha256": info["freeze_sha256"], "legacy_context": True}
    last = captured[-1] if captured else None
    if (last and last["returncode"] == 0 and last["stdout"] is not None and last["command"][:1] == [info["python"]]
            and sha256_bytes(last["stdout"].encode()) != info["freeze_sha256"]):
        actual = last["stdout"]
        where = None
        if evidence_dir is not None:
            expected = layout.path(f"{V1_RUNTIME}/resolved/{name}.txt").read_text()
            where = Path(evidence_dir) / f"{name}-freeze-{int(time.time())}"
            where.mkdir(parents=True)
            (where / "actual.txt").write_text(actual)
            (where / "diff.txt").write_text("".join(difflib.unified_diff(
                expected.splitlines(True), actual.splitlines(True), "archived", "actual")))
        raise TerminalMismatch(f"{name}: legacy-context pip freeze --all hashes to {sha256_bytes(actual.encode())}, "
                               f"archived {info['freeze_sha256']}; nothing re-pinned; evidence: {where}")
    raise RecoveryError(f"{name}: frozen environment verifier failed: {error}")


# --------------------------------------------------------------------------- pinned archive

def check_archive_structure(archive):
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(infos) > ARCHIVE_LIMITS["members"] or len(set(names)) != len(names):
        raise RecoveryError("Archive has too many or duplicate member names")
    total = 0
    for info in infos:
        name = info.filename
        if "\\" in name or "\x00" in name or name.startswith("/") or re.match(r"[A-Za-z]:", name):
            raise RecoveryError(f"Unsafe archive member name: {name!r}")
        if any(part in ("", ".", "..") for part in name.rstrip("/").split("/")):
            raise RecoveryError(f"Unsafe archive member path: {name!r}")
        kind = stat.S_IFMT(info.external_attr >> 16)
        if info.is_dir():
            if kind not in (0, stat.S_IFDIR):
                raise RecoveryError(f"Archive directory entry has a special type: {name}")
            continue
        if kind not in (0, stat.S_IFREG):
            raise RecoveryError(f"Archive member is a symlink or special file: {name}")
        if info.flag_bits & 0x1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise RecoveryError(f"Archive member is encrypted or uses an unsupported compression: {name}")
        if info.file_size > ARCHIVE_LIMITS["member_bytes"]:
            raise RecoveryError(f"Archive member is too large: {name}")
        total += info.file_size
    if total > ARCHIVE_LIMITS["total_bytes"]:
        raise RecoveryError("Archive expands beyond the allowed total size")


def open_archive(layout):
    """Read the pinned V1 bundle once into memory; parse exactly the verified bytes."""
    pin = layout.pins["derived_from"]["v1"]
    info = lstat_chain(layout.root, ARCHIVE)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise RecoveryError(f"Pinned archive is missing or not a regular file: {layout.path(ARCHIVE)}")
    fd = open_regular(layout.path(ARCHIVE))
    try:
        if os.fstat(fd).st_size != pin["bytes"]:
            raise RecoveryError("Archive size differs from the pin")
        chunks, total = [], 0
        while total <= pin["bytes"]:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) != pin["bytes"] or sha256_bytes(data) != pin["sha256"]:
        raise RecoveryError(f"Archive identity differs from the pin {pin['sha256'][:12]}")
    archive = zipfile.ZipFile(io.BytesIO(data))
    check_archive_structure(archive)
    present = set(archive.namelist())
    if any(name in present for name in ABSENT_MEMBERS[2:]):
        raise RecoveryError("Archive unexpectedly contains V1 selection files")
    return archive


def read_member(archive, name, pin):
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise RecoveryError(f"Pinned member missing from the archive: {name}") from None
    if info.file_size != pin["size"]:
        raise RecoveryError(f"Archive member size differs from its pin: {name}")
    with archive.open(info) as stream:
        data = stream.read(pin["size"] + 1)  # reading to EOF also checks the CRC
    if len(data) != pin["size"] or sha256_bytes(data) != pin["sha256"]:
        raise RecoveryError(f"Archive member bytes differ from their pin: {name}")
    return data, ((info.external_attr >> 16) & 0o777) or 0o640


def archived_json(archive, layout, name):
    return json.loads(read_member(archive, name, layout.pins["v1_runtime_members"][name])[0])


# --------------------------------------------------------------------------- state, ownership, locks, disk

def ensure_dirs(root, relative):
    """Create missing directories below ``root`` one component at a time, refusing symlinks."""
    current = Path(root)
    for part in relative_parts(relative):
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o777)
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise RecoveryError(f"Expected a real directory: {current}")
    return current


class State:
    """Phase records, ownership of the trees this recovery created, and the state-directory lock."""

    def __init__(self, layout):
        self.layout, self.dir = layout, layout.state
        self.file, self.data, self._lock = self.dir / "state.json", None, None

    def open(self):
        """Execute only: create or reopen the state directory and hold its flock until close()."""
        info = lstat_chain(self.layout.root, STATE)
        if info is None:
            for relative in OWNED:
                if lstat_chain(self.layout.root, relative) is not None:
                    raise RecoveryError(f"{self.layout.path(relative)} exists but no recovery state does; it "
                                        "existed before this recovery and is not recovery-owned")
            os.mkdir(self.dir, 0o777)
        elif not stat.S_ISDIR(info.st_mode):
            raise RecoveryError(f"Recovery state path is not a directory: {self.dir}")
        fd = os.open(self.dir / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise RecoveryError("Another recovery execute holds the state lock") from None
        self._lock = fd
        if not self.layout.pins_sha256:
            self.close()
            raise RecoveryError("The verified pins digest is unknown; refusing to bind recovery state")
        for name in ("logs", "receipts", "failures", "staging", "scratch", "inputs", "preserved"):
            ensure_dirs(self.layout.root, f"{STATE}/{name}")
        if self.file.exists():
            self.data = read_json(self.file)
            if (self.data.get("tool") != TOOL or self.data.get("runtime_root") != str(self.layout.root)
                    or self.data.get("pins_sha256") != self.layout.pins_sha256):
                self.close()
                raise RecoveryError("Recovery state belongs to another tool, root or pin set")
            if self.data.get("stop_unverified"):
                record = self.data["stop_unverified"]
                self.close()
                raise StopUnverified(stop_unverified_message(record))
        else:
            for relative in OWNED:
                if lstat_chain(self.layout.root, relative) is not None:
                    raise RecoveryError(f"{self.layout.path(relative)} existed before this recovery; not owned")
            self.data = {"schema_version": 1, "tool": TOOL, "runtime_root": str(self.layout.root),
                         "matrix_commit": MATRIX_COMMIT, "pins_sha256": self.layout.pins_sha256,
                         "created_at": utc_now(), "owned": {}, "intent": {}, "phases": {}}
            self.save()
        return self

    def close(self):
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)
            self._lock = None

    def save(self):
        write_json_atomic(self.file, self.data)

    @staticmethod
    def read_only(layout):
        path = layout.state / "state.json"
        if lstat_chain(layout.root, f"{STATE}/state.json") is None:
            return None
        return read_json(path)

    def phase(self, name):
        return self.data["phases"].get(name, {})

    def complete(self, name):
        return self.phase(name).get("status") == "complete"

    def begin(self, name):
        if self.data.get("stop_unverified"):
            raise StopUnverified(stop_unverified_message(self.data["stop_unverified"]))
        record = self.data["phases"].setdefault(name, {"attempts": 0})
        if record.get("terminal"):
            raise TerminalMismatch(f"Phase {name} ended in a terminal mismatch ({record.get('error')}); "
                                   "it is never retried automatically")
        record.update(status="started", attempts=record["attempts"] + 1, started_at=utc_now(), error=None,
                      finished_at=None)
        self.save()

    def finish(self, name, receipt):
        path = self.dir / "receipts" / f"{name}.json"
        write_json_atomic(path, {"phase": name, "finished_at": utc_now(), **receipt})
        self.data["phases"][name].update(status="complete", finished_at=utc_now(),
                                         receipt=str(path.relative_to(self.dir)))
        self.save()

    def fail(self, name, exc):
        stopped = not isinstance(exc, StopUnverified)
        self.data["phases"][name].update(status="failed" if stopped else "stop-unverified", finished_at=utc_now(),
                                         terminal=isinstance(exc, TerminalMismatch) or not stopped,
                                         error=f"{type(exc).__name__}: {exc}"[:4000])
        if not stopped:
            self.mark_stop_unverified(name, exc)
        self.save()

    def mark_stop_unverified(self, name, exc):
        """Sticky: a blocked child could resume writing, so no later execute or verify may proceed."""
        self.data["stop_unverified"] = {"phase": name, "at": utc_now(), "error": f"{exc}"[:4000],
                                        "log": str(getattr(exc, "log_path", None))}
        self.save()

    def own(self, relative):
        """Create a recovery-owned top directory, or re-verify it by device and inode on resume."""
        layout, path = self.layout, self.layout.path(relative)
        info, record = lstat_chain(layout.root, relative), self.data["owned"].get(relative)
        if record:
            if info is None or not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (record["dev"], record["ino"]):
                raise RecoveryError(f"Recovery-owned {path} disappeared or was replaced; inspect before resuming")
            return path
        if info is not None:
            if not (self.data["intent"].get(relative) and stat.S_ISDIR(info.st_mode)
                    and info.st_uid == os.getuid() and not any(os.scandir(path))):
                raise RecoveryError(f"{path} existed before this recovery; it is not recovery-owned")
        else:
            self.data["intent"][relative] = utc_now()
            self.save()
            os.mkdir(path, 0o777)
            info = os.lstat(path)
        self.data["owned"][relative] = {"dev": info.st_dev, "ino": info.st_ino, "created_at": utc_now()}
        self.data["intent"].pop(relative, None)
        self.save()
        return path

    def check_owned(self, path):
        """``path`` must lie inside a recovery-owned tree whose top directory is unchanged."""
        path = Path(os.path.abspath(path))
        for relative in OWNED:
            top = self.layout.path(relative)
            if (path == top or top in path.parents) and self.data["owned"].get(relative):
                self.own(relative)  # re-verifies the top directory's device and inode
                lstat_chain(self.layout.root, path.relative_to(self.layout.root).as_posix())
                return path
        raise RecoveryError(f"{path} is not inside a recovery-owned tree; refusing to modify it")


def stop_unverified_message(record):
    return (f"Recovery is blocked: during {record.get('phase')} at {record.get('at')} an owned child process group could "
            f"not be proven stopped ({record.get('error')}). A still-blocked child could resume writing. Inspect live "
            "processes (ps -eo pid,pgid,stat,lstart,args) and the owned trees before any further recovery; neither "
            "execute nor verify proceeds while this record exists.")


@contextlib.contextmanager
def study_locks(layout):
    """Hold the existing .active.lock of every study output for the whole execute, opened without create.

    common.exclusive_run creates the lock and never removes it, so each of these used outputs must
    have one. A missing lock is a blocker: holding nothing would not exclude a concurrent command.
    """
    held = []
    try:
        for relative in LOCKED_OUTPUTS:
            lock = f"{relative}/.active.lock"
            info = lstat_chain(layout.root, lock)
            if info is None or not stat.S_ISREG(info.st_mode):
                raise RecoveryError(f"{layout.path(lock)} is missing or not a regular file; exclusive_run never "
                                    "removes it, so mutual exclusion with study commands cannot be established")
            fd = os.open(layout.path(lock), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                raise RecoveryError(f"{layout.path(relative)} is in use: its .active.lock is held") from None
            held.append(fd)
        yield [str(layout.path(f"{r}/.active.lock")) for r in LOCKED_OUTPUTS]
    finally:
        for fd in held:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def probe_locks(layout):
    """Read-only probe (shared, non-blocking) of every study lock and the recovery state lock."""
    report = {}
    for relative, lock in [(r, f"{r}/.active.lock") for r in LOCKED_OUTPUTS] + [(STATE, f"{STATE}/.lock")]:
        try:
            info = lstat_chain(layout.root, lock)
        except RecoveryError as exc:
            report[relative] = f"unsafe: {exc}"
            continue
        if info is None or not stat.S_ISREG(info.st_mode):
            report[relative] = "missing" if info is None else "not-a-regular-file"
            continue
        fd = os.open(layout.path(lock), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            report[relative] = "free"
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            report[relative] = "held"
        finally:
            os.close(fd)
    return report


def free_bytes(path):
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def check_disk(layout, phase):
    free = free_bytes(layout.path("results"))
    if free < DISK_GIB[phase] * GIB:
        raise RecoveryError(f"Phase {phase} needs about {DISK_GIB[phase]} GiB free; {free / GIB:.1f} GiB available "
                            "(statvfs cannot see quotas)")
    return free


# --------------------------------------------------------------------------- restored metadata checks

def exclude_newer(created_at):
    """uv --exclude-newer cutoff: the archived environment creation time, rounded up to a whole second."""
    moment = datetime.fromisoformat(created_at)
    if moment.tzinfo is None:
        raise RecoveryError("Archived environment creation time has no timezone")
    moment = moment.astimezone(timezone.utc)
    if moment.microsecond:
        moment = moment.replace(microsecond=0) + timedelta(seconds=1)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def source_specs(lock):
    """GitHub checkouts pinned in the V1 lock that the two restored environments install: {repo: revision}."""
    specs = {s.get("repo"): s.get("sha") for s in lock.get("verified_sources", []) if s.get("kind") == "github"}
    kev, wanted = lock["models"]["kev"], sorted({e["source"] for e in ENVIRONMENTS.values()})
    if specs.get(kev.get("code_repo")) != kev.get("code_revision") or any(
            not re.fullmatch(r"[0-9a-f]{40}", specs.get(repo) or "") for repo in wanted):
        raise RecoveryError("V1 lock sources differ from the expected kev/StartLux checkouts")
    return {repo: specs[repo] for repo in wanted}


def check_local_sources(layout, name, frozen_text, lock):
    """The archived freeze must reference exactly this environment's checkout at its original path."""
    repo = ENVIRONMENTS[name]["source"]
    expected = ["file://" + str(layout.source_dir(repo, source_specs(lock)[repo]))]
    found = [line.split(" @ ", 1)[1].strip() for line in frozen_text.splitlines() if " @ file:" in line]
    if found != expected:
        raise RecoveryError(f"{name}: archived local sources {found} differ from {expected}")


def check_source(sup, path, revision, log_path=None):
    """Read-only: pinned HEAD, the raw blob of every tracked file at HEAD, and untracked or ignored extras.

    Ignored files are listed too: a stale ignored build/lib copy could otherwise enter a wheel built
    from a checkout whose HEAD and status look clean.
    """
    path = Path(path)
    try:
        head = git(sup, path, "rev-parse", "--verify", "-q", "HEAD", log_path=log_path).strip()
    except GitFailed:
        return {"state": "incomplete", "problems": ["HEAD does not resolve"], "extras": [], "extra_count": 0}
    if head != revision:
        return {"state": "changed", "problems": [f"HEAD is {head}, pinned {revision}"], "extras": [], "extra_count": 0}
    problems = []
    for record in filter(None, git(sup, path, "ls-tree", "-r", "-z", "--full-tree", revision,
                                   log_path=log_path).split("\0")):
        meta, relative = record.split("\t", 1)
        mode, kind, oid = meta.split()
        try:
            parent = Path(relative).parent.as_posix()
            if parent != ".":
                lstat_chain(path, parent)
            info = os.lstat(path / relative)
        except (OSError, RecoveryError) as exc:
            problems.append(f"{relative}: {exc}")
            continue
        if kind == "blob" and mode == "120000" and stat.S_ISLNK(info.st_mode):
            target = os.fsencode(os.readlink(path / relative))
            same = hashlib.sha1(b"blob %d\0" % len(target) + target).hexdigest() == oid
        elif kind == "blob" and mode in ("100644", "100755") and stat.S_ISREG(info.st_mode):
            same = git_blob_sha1(path / relative) == oid and bool(info.st_mode & 0o100) == (mode == "100755")
        else:
            same = False
        if not same:
            problems.append(f"tracked entry differs from {revision[:12]}: {relative}")
    entries = [e for e in git(sup, path, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored",
                              log_path=log_path).split("\0") if e]
    problems += [f"git reports a tracked change: {e}" for e in entries if not e.startswith(("?? ", "!! "))]
    extras = [e[3:] for e in entries if e.startswith(("?? ", "!! "))]
    return {"state": "changed" if problems else ("extras" if extras else "pristine"), "problems": problems[:20],
            "extras": extras[:50], "extra_count": len(extras)}


def verify_members(layout):
    """Restored bytes equal their pins; absent records stay absent; nothing else appears beside them."""
    problems, pins = [], layout.pins["v1_runtime_members"]
    for name, pin in sorted(pins.items()):
        relative = f"{V1}/{name}"
        try:
            info = lstat_chain(layout.root, relative)
            if (info is None or not stat.S_ISREG(info.st_mode) or info.st_size != pin["size"]
                    or sha256_file(layout.path(relative)) != pin["sha256"]):
                problems.append(f"Restored member is missing or differs from its pin: {relative}")
        except (RecoveryError, OSError) as exc:
            problems.append(f"{relative}: {exc}")
    for name in ABSENT_MEMBERS:
        try:
            if lstat_chain(layout.root, f"{V1}/{name}") is not None:
                problems.append(f"Must stay absent (historical session/PIDs or never-existing selection): {V1}/{name}")
        except RecoveryError as exc:
            problems.append(str(exc))
    top = layout.path(V1_RUNTIME)
    if os.path.isdir(top) and not os.path.islink(top):
        for directory, folders, files in os.walk(top):
            if Path(directory) == top and "work" in folders:
                folders.remove("work")
            for entry in files + [f for f in folders if os.path.islink(os.path.join(directory, f))]:
                member = "runtime/" + Path(directory, entry).relative_to(top).as_posix()
                if member not in pins:
                    problems.append(f"Unexpected entry beside the restored metadata: {V1}/{member}")
    return problems


def original_egg_info(root):
    """decision-study metadata of the original project, which the frozen legacy context exposes to pip freeze."""
    found = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if entry.name.endswith(".egg-info") and entry.is_dir(follow_symlinks=False):
            info = Path(entry.path) / "PKG-INFO"
            if info.is_file() and not info.is_symlink():
                text = info.read_text(errors="replace")
                name = re.search(r"^Name:\s*(\S+)", text, re.M)
                version = re.search(r"^Version:\s*(\S+)", text, re.M)
                if name and re.sub(r"[-_.]+", "-", name[1]).lower() == "decision-study":
                    found.append({"path": entry.path, "version": version[1] if version else None})
    return found


# --------------------------------------------------------------------------- execute phases

class Executor:
    def __init__(self, layout, matrix, sup, state, uv):
        self.layout, self.matrix, self.sup, self.state, self.uv = layout, matrix, sup, state, uv

    def rel(self, path):
        return Path(path).relative_to(self.layout.root).as_posix()

    def member(self, name):
        return self.layout.path(f"{V1}/{name}")

    def run(self, phase, log_name, argv, *, timeout, extra=None, capture=False):
        result = self.sup.run(argv, env=child_env(extra), timeout=timeout, cwd=self.state.dir,
                              log_path=self.state.dir / "logs" / phase / log_name, capture=capture)
        if result.returncode:
            raise ChildError(f"{Path(str(argv[0])).name} exited with status {result.returncode}",
                             log_path=result.log_path, returncode=result.returncode)
        return result

    def execute(self):
        receipts = {}
        for phase in PHASES:
            if self.state.complete(phase):
                continue
            missing = [need for need in REQUIRES[phase] if not self.state.complete(need)]
            if missing:
                raise RecoveryError(f"Phase {phase} requires completed {missing}")
            require_pins(self.layout, metadata=phase != "metadata")
            free = check_disk(self.layout, phase)
            self.state.begin(phase)
            try:
                receipt = getattr(self, "phase_" + phase.replace("-", "_"))()
            except BaseException as exc:
                self.state.fail(phase, exc)
                raise
            receipt["free_bytes_before"] = free
            self.state.finish(phase, receipt)
            receipts[phase] = receipt
        return receipts

    def phase_metadata(self):
        layout, pins = self.layout, self.layout.pins
        archive = open_archive(layout)
        if archived_root(archived_json(archive, layout, "runtime/installations.json")) != layout.root:
            raise RecoveryError("The archived absolute paths name another runtime root")
        self.state.own(V1_RUNTIME)
        staging, restored, present = self.state.dir / "staging", 0, 0
        for name, pin in sorted(pins["v1_runtime_members"].items()):
            relative = f"{V1}/{name}"
            target = self.state.check_owned(layout.path(relative))
            info = lstat_chain(layout.root, relative)
            if info is not None:
                if stat.S_ISREG(info.st_mode) and info.st_size == pin["size"] and sha256_file(target) == pin["sha256"]:
                    present += 1
                    continue
                raise RecoveryError(f"{target} exists with other content; restored files are never overwritten")
            data, mode = read_member(archive, name, pin)
            staged = staging / name.replace("/", "__")
            with contextlib.suppress(FileNotFoundError):
                os.unlink(staged)  # leftover of an interrupted attempt; staging is recovery-owned
            fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, mode)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            ensure_dirs(layout.root, str(Path(relative).parent))
            os.link(staged, target)  # link() never replaces an existing name
            os.unlink(staged)
            restored += 1
        problems = verify_members(layout)
        if problems:
            raise RecoveryError("; ".join(problems[:5]))
        return {"archive_sha256": pins["derived_from"]["v1"]["sha256"], "restored": restored,
                "already_present": present, "kept_absent": list(ABSENT_MEMBERS)}

    def preserve(self, phase, path):
        """Move a recovery-created incomplete or unclean attempt aside under the state directory; nothing is deleted."""
        destination = self.state.dir / "preserved" / phase / f"{path.name}-{time.time_ns()}"
        ensure_dirs(self.layout.root, self.rel(destination.parent))
        self.state.check_owned(path)
        os.rename(path, destination)
        self.state.data.setdefault("preserved", []).append({"phase": phase, "from": str(path),
                                                            "to": str(destination), "at": utc_now()})
        self.state.save()
        return destination

    def pristine_source(self, phase, repo, revision):
        """Pinned HEAD, every tracked blob and no untracked or ignored file: checked before every install.

        A changed tracked source fails closed and stays in place for inspection. A recovery-created
        checkout that is incomplete (HEAD does not resolve) or carries untracked/ignored extras, such as
        stale build copies from an earlier attempt, is preserved under the state directory and replaced
        by a fresh clone, so the build always starts from the pinned tree alone.
        """
        path = self.state.check_owned(self.layout.source_dir(repo, revision))
        log, previous = self.state.dir / "logs" / phase / "git.log", None
        if lstat_chain(self.layout.root, self.rel(path)) is not None:
            found = check_source(self.sup, path, revision, log_path=log)
            if found["state"] == "pristine":
                return {"path": str(path), "revision": revision, "fresh_clone": False}
            if found["state"] == "changed":
                raise RecoveryError(f"Pinned source changed; preserved in place for inspection: {path}: "
                                    f"{found['problems'][:5]}")
            previous = str(self.preserve(phase, path))
        ensure_dirs(self.layout.root, self.rel(path.parent))
        command = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false"]
        for args in (["init", path], ["-C", path, "remote", "add", "origin", f"https://github.com/{repo}.git"],
                     ["-C", path, "fetch", "--depth", "1", "origin", revision], ["-C", path, "checkout", "--detach", revision]):
            self.run(phase, "git.log", command + args, timeout=TIMEOUT["git"], capture=True)
        found = check_source(self.sup, path, revision, log_path=log)
        if found["state"] != "pristine":
            raise RecoveryError(f"Fresh checkout of {repo} is not pristine at {revision}: {found['problems'][:5]} "
                                f"extras={found['extras'][:5]}")
        return {"path": str(path), "revision": revision, "fresh_clone": True, "preserved_previous": previous}

    def phase_sources(self):
        """V1 _checkout commands at the original paths the archived freezes reference."""
        lock = read_json(self.member("runtime/lock.json"))
        return {repo: self.pristine_source("sources", repo, revision) for repo, revision in source_specs(lock).items()}

    # ---- environments: original commands plus exact archived pins; accepted only by the frozen freeze check

    def environment(self, name):
        info = read_json(self.member("runtime/installations.json"))["environments"][name]
        python = self.layout.env_dir(name) / "bin/python"
        if info.get("python") != str(python) or info.get("python_version") != MANAGED_PYTHON:
            raise RecoveryError(f"{name}: archived environment record does not match the runtime layout")
        return info, python

    def portable(self, name, info, lock):
        fd = open_regular(self.member(f"runtime/resolved/{name}.txt"))
        with os.fdopen(fd, "rb") as handle:
            data = handle.read()
        if sha256_bytes(data) != info["freeze_sha256"]:
            raise RecoveryError(f"{name}: restored resolved inventory does not hash to the recorded freeze")
        text = data.decode()
        check_local_sources(self.layout, name, text, lock)
        portable = self.matrix.frozen._portable_requirements(text)  # drops only decision-study and @ file: lines
        if any(re.match(r"(?i)decision[-_.]study\b", line.strip()) or " @ " in line for line in portable.splitlines()):
            raise RecoveryError(f"{name}: installer input still names decision-study or a direct reference")
        if name == "kev" and portable.encode() != self.member("runtime/resolved/kev-portable.txt").read_bytes():
            raise RecoveryError("Frozen _portable_requirements(kev) differs from the archived kev-portable.txt")
        path = self.state.dir / "inputs" / f"{name}-portable.txt"
        if lstat_chain(self.layout.root, self.rel(path)) is not None:
            if path.read_bytes() != portable.encode():
                raise RecoveryError(f"Recorded installer input changed: {path}")
        else:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o640)
            with os.fdopen(fd, "w") as handle:
                handle.write(portable)
        return path, portable

    def phase_env(self, name):
        spec, layout = ENVIRONMENTS[name], self.layout
        phase, lock = spec["phase"], read_json(self.member("runtime/lock.json"))
        info, python = self.environment(name)
        requirements, portable = self.portable(name, info, lock)
        env_dir = self.state.check_owned(layout.env_dir(name))
        # The pinned checkout is verified (or a recovery-created unclean attempt preserved and re-cloned)
        # before any uv command of this attempt runs.
        source_check = self.pristine_source(phase, spec["source"], source_specs(lock)[spec["source"]])
        source = layout.source_dir(spec["source"], source_specs(lock)[spec["source"]])
        requirement = f"{source}[{spec['extra']}]" if spec["extra"] else str(source)
        cutoff, extra, uv = exclude_newer(info["created_at"]), v1_env(layout), [self.uv, "--no-config"]
        ensure_dirs(layout.root, self.rel(env_dir.parent))
        retry = lstat_chain(layout.root, self.rel(env_dir)) is not None  # interrupted recovery-owned attempt
        self.run(phase, "uv.log", uv + ["venv", "--managed-python", "--python", MANAGED_PYTHON, "--seed",
                                        *(["--clear"] if retry else []), env_dir], timeout=TIMEOUT["uv-venv"], extra=extra)
        logs, scratch = self.state.dir / "logs" / phase, self.state.dir / "scratch" / phase
        headers = check_headers(self.matrix, layout, self.sup, logs, scratch, name, info)
        self.run(phase, "uv.log", uv + ["pip", "install", "--python", python, "--exclude-newer", cutoff, requirement,
                                        *spec["packages"], "-r", requirements], timeout=TIMEOUT["uv-install"], extra=extra)
        for package in spec["after"]:
            self.run(phase, "uv.log", uv + ["pip", "install", "--python", python, "--exclude-newer", cutoff, package],
                     timeout=TIMEOUT["uv-install"], extra=extra)
        self.run(phase, "uv.log", uv + ["pip", "check", "--python", python], timeout=TIMEOUT["uv-check"], extra=extra)
        freeze = check_freeze(self.matrix, layout, self.sup, logs, scratch, name, info,
                              evidence_dir=self.state.dir / "failures")
        return {"python": str(python), "python_headers": headers, "exclude_newer": cutoff, "rebuilt_with_clear": retry,
                "source": source_check,
                "installer_input_sha256": sha256_bytes(portable.encode()), "uv_version": self.uv_version, **freeze}

    def phase_env_vllm(self):
        return self.phase_env("vllm-0.31.0")

    def phase_env_kev(self):
        return self.phase_env("kev")

    # ---- CUDA toolkit: the pinned installer, then the pinned validator and the archived receipt identity

    def phase_toolkit(self):
        layout = self.layout
        setup = read_json(self.member(TOOLKIT_SETUP))
        target = self.state.check_owned(layout.toolkit_dir())
        if setup.get("status") != "complete" or setup.get("target") != str(target):
            raise RecoveryError("The archived toolkit report does not name the restored toolkit target")
        requirements = layout.checkout / TOOLKIT_REQUIREMENTS
        _, python = self.environment("vllm-0.31.0")
        ensure_dirs(layout.root, self.rel(target.parent))
        result = self.state.dir / "receipts" / f"toolkit-installer-{time.time_ns()}.json"
        # Every supervisor or installer rejection propagates. A later attempt, made only after a confirmed
        # stop, lets the pinned installer reuse a target it validates itself; a rejected attempt never counts.
        self.run("toolkit", "prepare-cuda-toolkit.log",
                 [python, layout.checkout / "scripts/prepare_cuda_toolkit.py", "--target", target,
                  "--requirements", requirements, "--result", result], timeout=TIMEOUT["toolkit"])
        identity = toolkit_identity(self.matrix, target, requirements, setup)
        return {**identity, "installer_result": str(result)}

    # ---- weights: pinned model_probe download; the result must equal the archived record

    def download(self, phase, model, python, output_dir, hf_home, expected):
        result = self.state.dir / "receipts" / f"{phase}-result-{time.time_ns()}.json"
        self.run(phase, f"{phase}.log", [python, self.layout.checkout / "scripts/model_probe.py", "download",
                                         "--model", model, "--output-dir", output_dir, "--result", result],
                 timeout=TIMEOUT["download-" + phase.split("-", 1)[1]],
                 extra={**v1_env(self.layout), "HF_HOME": str(hf_home)})
        actual = read_json(result)
        if actual != expected:
            raise TerminalMismatch(f"{phase}: download record {actual} differs from the archived {expected}")
        return {"record": actual, "result": str(result), "hf_home": str(hf_home)}

    def phase_weights_kev(self):
        expected = read_json(self.member("runtime/installations.json"))["models"]["kev"]
        if read_json(self.member("runtime/download-kev.json")) != expected:
            raise RecoveryError("Archived kev download and installation records disagree")
        hf_home = self.state.check_owned(self.layout.path(V1_WORK + "/cache/huggingface"))
        if not all(within(expected[key], hf_home / "hub") for key in ("model_path", "base_path")):
            raise RecoveryError("Archived kev snapshots are outside the V1 Hugging Face cache")
        ensure_dirs(self.layout.root, self.rel(hf_home))
        _, python = self.environment("kev")
        return self.download("weights-kev", "kev", python, self.layout.path(V1), hf_home, expected)

    def phase_weights_qwen9b(self):
        layout, controls = self.layout, self.layout.pins["v3_controls"]
        for name in (V3_DOWNLOAD, "runtime-controls/qwen9b/runtime/lock.json"):
            if sha256_file(layout.path(f"{V3}/{name}")) != controls[name]:
                raise RecoveryError(f"V3 control file changed: {V3}/{name}")
        expected = read_json(layout.path(f"{V3}/{V3_DOWNLOAD}"))
        hf_home = layout.path(V3_WORK + "/cache/huggingface")
        if not within(expected["model_path"], hf_home / "hub"):
            raise RecoveryError("The V3 Qwen3.5-9B record is outside the V3 Hugging Face cache")
        self.state.own(V3_WORK)
        ensure_dirs(layout.root, self.rel(hf_home))
        _, python = self.environment("vllm-0.31.0")
        return self.download("weights-qwen9b", "qwen", python, layout.path(f"{V3}/runtime-controls/qwen9b"),
                             hf_home, expected)


def within(path, root):
    path, root = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    return root in path.parents


def toolkit_identity(matrix, target, requirements, setup):
    """Pinned validate_target, then equality with the archived setup report's identity."""
    try:
        receipt = matrix.toolkit.validate_target(Path(target), Path(requirements))
    except Exception as exc:
        raise TerminalMismatch(f"Published toolkit fails the pinned validate_target (preserved): {exc}") from exc
    differs = [key for key in TOOLKIT_IDENTITY if receipt.get(key) != setup["verification"].get(key)]
    if receipt.get("target") != setup["target"]:
        differs.append("target")
    if differs:
        raise TerminalMismatch(f"Toolkit identity differs from the archived receipt in {differs}; preserved: {target}")
    return {"target": receipt["target"], "tree_sha256": receipt["tree_sha256"],
            "installer_sha256": receipt["installer_sha256"], "requirements_sha256": receipt["requirements_sha256"]}


# --------------------------------------------------------------------------- READY verification (read-only)

def git_blob_sha1(path):
    fd = open_regular(path)
    try:
        size = os.fstat(fd).st_size
        digest = hashlib.sha1(b"blob %d\0" % size)
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    finally:
        os.close(fd)


# huggingface_hub >= 1.32 (utils/_shared_blobs.py; the same rules in the archived 1.33.0 and 2.4.0) may move a Xet
# download into one store per cache and turn the repo blob into a relative link to it. A regular marker file holding
# "1\n" identifies the store. The payload is named by its Xet hash, not the raw file SHA-256 / Hub ETag.
SHARED_BLOBS_MARKER = ".huggingface-shared-blobs"
HUB_REPO_DIR = re.compile(r"(?:datasets|kernels|models|spaces)--.+")


def hub_blob_payload(blob):
    """The regular file holding the bytes of ``<cache>/<repo>/blobs/<etag>``, located without following links.

    Accepted: the repo blob as a regular file, or exactly the link huggingface_hub writes into the same cache's marked
    shared store, ``../../blobs/<xet[:2]>/<xet>`` with a 64-hex Xet hash and a regular payload. Each directory below
    the cache root on the way must be a real directory. Anything else raises; nothing is written.
    """
    repo = blob.parent.parent
    cache = repo.parent
    info = lstat_chain(cache, f"{repo.name}/blobs")
    if info is None or not stat.S_ISDIR(info.st_mode):
        raise RecoveryError(f"no blobs directory: {repo / 'blobs'}")
    info = os.lstat(blob)
    if stat.S_ISREG(info.st_mode):
        return blob
    if not stat.S_ISLNK(info.st_mode):
        raise RecoveryError(f"neither a regular file nor a link: {blob}")
    target = os.readlink(blob)
    parts = target.split("/")
    if (not HUB_REPO_DIR.fullmatch(repo.name) or len(parts) != 5 or parts[:3] != ["..", "..", "blobs"]
            or not re.fullmatch(r"[0-9a-f]{64}", parts[4]) or parts[3] != parts[4][:2]):
        raise RecoveryError(f"unsupported link {target!r}; only ../../blobs/<xet[:2]>/<xet> from a hub repository "
                            "folder of the same cache is accepted")
    for relative in ("blobs/" + SHARED_BLOBS_MARKER, "/".join(parts[2:])):
        info = lstat_chain(cache, relative)  # refuses a symlinked store, prefix directory, marker or payload
        if info is None or not stat.S_ISREG(info.st_mode):
            raise RecoveryError(f"missing or not a regular file: {cache / relative}")
    fd = open_regular(cache / "blobs" / SHARED_BLOBS_MARKER)
    try:
        if os.read(fd, 8) != b"1\n":
            raise RecoveryError(f"not a version-1 huggingface_hub shared blob store: {cache / 'blobs'}")
    finally:
        os.close(fd)
    return cache.joinpath(*parts[2:])


def verify_snapshot(snapshot, label, problems):
    """Each entry links to a blob of its own repo (a regular file or one shared-store link, see hub_blob_payload)
    whose bytes hash to the blob name, the Hub ETag; shards present."""
    snapshot = Path(snapshot)
    if not snapshot.is_dir() or snapshot.is_symlink():
        problems.append(f"{label}: snapshot directory is missing: {snapshot}")
        return 0
    blobs, names, checked = snapshot.parent.parent / "blobs", set(), 0
    for directory, folders, files in os.walk(snapshot):
        for folder in folders:
            if os.path.islink(os.path.join(directory, folder)):
                problems.append(f"{label}: directory link inside the snapshot: {folder}")
        for name in files:
            entry = Path(directory, name)
            relative = entry.relative_to(snapshot).as_posix()
            names.add(relative)
            if not entry.is_symlink():
                problems.append(f"{label}: snapshot entry is not a hub blob link: {relative}")
                continue
            blob = Path(os.path.normpath(entry.parent / os.readlink(entry)))
            try:
                if blob.parent != blobs:
                    raise RecoveryError("the link leaves the repository's blobs directory")
                payload = hub_blob_payload(blob)
            except (RecoveryError, OSError) as exc:
                problems.append(f"{label}: {relative} does not resolve to a blob of its repository ({exc})")
                continue
            if re.fullmatch(r"[0-9a-f]{64}", blob.name):  # streamed over the payload, compared with the ETag only
                actual = sha256_file(payload)
            elif re.fullmatch(r"[0-9a-f]{40}", blob.name):
                actual = git_blob_sha1(payload)
            else:
                problems.append(f"{label}: unexpected blob name for {relative}: {blob.name}")
                continue
            if actual != blob.name:
                problems.append(f"{label}: blob content does not match its name: {relative}")
            checked += 1
    if not any(name.endswith(".safetensors") for name in names):
        problems.append(f"{label}: snapshot has no safetensors weights")
    for index in sorted(name for name in names if name.endswith(".safetensors.index.json")):
        try:
            shards = set(json.loads((snapshot / index).read_text())["weight_map"].values())
        except (OSError, ValueError, KeyError, AttributeError) as exc:
            problems.append(f"{label}: unreadable shard index {index}: {exc}")
            continue
        parent = Path(index).parent
        missing = sorted(s for s in shards if (parent / s).as_posix() not in names)
        if missing:
            problems.append(f"{label}: shards listed in {index} are missing: {missing[:5]}")
    return checked


def verify_controls(layout, problems):
    for prefix, key in ((V3, "v3_controls"), (SIZE, "size_controls")):
        for name, expected in sorted(layout.pins[key].items()):
            relative = f"{prefix}/{name}"
            try:
                info = lstat_chain(layout.root, relative)
                if info is None or not stat.S_ISREG(info.st_mode) or sha256_file(layout.path(relative)) != expected:
                    problems.append(f"Intact control file is missing or changed: {relative}")
            except (RecoveryError, OSError) as exc:
                problems.append(f"{relative}: {exc}")


def matrix_dirs(layout):
    return {"v1": str(layout.path(V1)), "v2": str(layout.path(V2)), "v3": str(layout.path(V3)),
            "size_control": str(layout.path(SIZE))}


def verify_matrix_output(matrix, layout, problems):
    """Frozen settings(check_source=True) and plans() for the existing matrix output; nothing is written."""
    output, report = layout.path(MATRIX_OUTPUT), {}
    if lstat_chain(layout.root, MATRIX_OUTPUT) is None:
        problems.append(f"Matrix output is missing: {output}")
        return report
    dirs = matrix_dirs(layout)
    if lstat_chain(layout.root, MATRIX_OUTPUT + "/settings.json") is not None:
        previous = os.environ.get("GIT_OPTIONAL_LOCKS")
        os.environ["GIT_OPTIONAL_LOCKS"] = "0"  # the frozen git status must not refresh the checkout index
        try:
            value = matrix.study.settings(output, check_source=True)
            report["settings"] = {"verified": True, "settings_hash": value.get("settings_hash")}
            if value.get("source_dirs") != dirs:
                problems.append("Matrix settings name other source directories than the restored runtime")
        except Exception as exc:
            problems.append(f"Matrix settings(check_source=True) failed: {type(exc).__name__}: {exc}")
        finally:
            if previous is None:
                os.environ.pop("GIT_OPTIONAL_LOCKS", None)
            else:
                os.environ["GIT_OPTIONAL_LOCKS"] = previous
    else:
        report["settings"] = "absent"
    stored = None
    if lstat_chain(layout.root, MATRIX_OUTPUT + "/runtime/binding.json") is not None:
        stored = read_json(output / "runtime/binding.json")
        if stored.get("source_dirs") != dirs:
            problems.append("The stored matrix binding names other source directories than the restored runtime")
    try:
        binding, _, _ = matrix.runtime.plans(dirs, output)
        report["plans"] = "computed"
        if stored is not None:
            if binding != stored:
                problems.append("Recomputed matrix runtime binding differs from the stored binding.json")
            else:
                report["binding"] = "equal to the stored binding"
        else:
            report["binding"] = "absent (init-runtime has not bound it yet)"
    except Exception as exc:
        problems.append(f"Frozen direct-matrix plans() failed: {type(exc).__name__}: {exc}")
    return report


def pin_problems(layout, *, metadata=True):
    problems = []
    verify_controls(layout, problems)
    if metadata:
        problems += verify_members(layout)
    return problems


def require_pins(layout, *, metadata=True):
    """Gate: nothing derived from a pinned file (program, snapshot or source path) is used unless it matches."""
    problems = pin_problems(layout, metadata=metadata)
    if problems:
        raise RecoveryError("Pinned files fail their pins; nothing derived from them is used: " + "; ".join(problems[:5]))


def verify(layout, sup, matrix, workdir, *, own_locks=False):
    """READY means restored files and identities match; GPU, CUDA and model behaviour remain untested.

    A file that fails its pin contributes no program, source or snapshot path: such a failure yields
    NOT_READY without spawning anything it names.
    """
    problems, report = list(ambient_blockers()), {"checked_at": utc_now(), "runtime_root": str(layout.root)}
    workdir, skipped = Path(workdir), "skipped: a pinned file it derives from failed its pin"
    state = State.read_only(layout)
    member_problems, control_problems = verify_members(layout), []
    verify_controls(layout, control_problems)
    problems += member_problems + control_problems
    if state and state.get("stop_unverified"):
        problems.append(stop_unverified_message(state["stop_unverified"]))
        report.update(problems=problems, status="MATRIX_RUNTIME_NOT_READY", next_step="inspect problems")
        return report  # nothing is spawned while a blocked recovery child may still be writing
    targets = []
    if member_problems:
        report.update(environments=skipped, toolkit=skipped, sources=skipped)
    else:
        installations = read_json(layout.path(f"{V1_RUNTIME}/installations.json"))
        lock = read_json(layout.path(f"{V1_RUNTIME}/lock.json"))
        environments = {}
        for name in ENVIRONMENTS:
            info = installations["environments"][name]
            logs, scratch = workdir / "logs" / name, workdir / "scratch" / name
            try:
                headers = check_headers(matrix, layout, sup, logs, scratch, name, info)
                environments[name] = {**check_freeze(matrix, layout, sup, logs, scratch, name, info),
                                      "python_version": headers["python_version"]}
            except StopUnverified:
                raise
            except Exception as exc:
                problems.append(f"Environment {name}: {type(exc).__name__}: {exc}")
        report["environments"] = environments
        try:
            report["toolkit"] = toolkit_identity(matrix, layout.toolkit_dir(), layout.checkout / TOOLKIT_REQUIREMENTS,
                                                 read_json(layout.path(f"{V1}/{TOOLKIT_SETUP}")))
        except Exception as exc:
            problems.append(f"Toolkit: {type(exc).__name__}: {exc}")
        sources = {}
        for repo, revision in source_specs(lock).items():
            path = layout.source_dir(repo, revision)
            try:
                found = (check_source(sup, path, revision, log_path=workdir / "logs" / "sources.log")
                         if lstat_chain(layout.root, path.relative_to(layout.root).as_posix()) is not None
                         else {"state": "missing", "problems": [str(path)]})
            except StopUnverified:
                raise
            except Exception as exc:
                found = {"state": "error", "problems": [f"{type(exc).__name__}: {exc}"]}
            sources[repo] = found
            if found["state"] not in ("pristine", "extras"):  # extras: generated copies after the verified install
                problems.append(f"Source {repo}: {found['state']}: {found.get('problems', [])[:3]}")
        report["sources"] = sources
        kev, hub = installations["models"]["kev"], layout.path(V1_WORK + "/cache/huggingface/hub")
        targets += [("kev4b-adapter", kev["model_path"], hub), ("kev4b-base", kev["base_path"], hub)]
    if not control_problems:
        try:
            targets.append(("qwen9b", read_json(layout.path(f"{V3}/{V3_DOWNLOAD}"))["model_path"],
                            layout.path(V3_WORK + "/cache/huggingface/hub")))
            targets.append(("startlux9b", read_json(layout.path(f"{SIZE}/{SIZE_INSTALLATIONS}"))["models"]["startlux"]
                            ["model_path"], None))
        except (OSError, ValueError, KeyError, RecoveryError) as exc:
            problems.append(f"Snapshot records unreadable: {exc}")
    snapshots = {}
    for label, path, cache in targets:
        if cache is not None and not within(path, cache):
            problems.append(f"{label}: recorded snapshot is outside its cache: {path}")
            continue
        snapshots[label] = {"path": path, "blobs_checked": verify_snapshot(path, label, problems)}
        if label == "startlux9b":
            try:
                matrix.size._verify_model_metadata(path)
                snapshots[label]["pinned_metadata"] = True
            except Exception as exc:
                problems.append(f"startlux9b: {type(exc).__name__}: {exc}")
    report["snapshots"] = snapshots
    if member_problems or control_problems:
        report["matrix_output"] = skipped
    else:
        report["matrix_output"] = verify_matrix_output(matrix, layout, problems)
    if own_locks:  # execute holds every study lock and the state lock itself
        report["locks"] = "held by this execute"
    else:
        locks = probe_locks(layout)
        report["locks"] = locks
        problems += [f"Study lock {relative}: {value}" for relative, value in locks.items()
                     if relative != STATE and value != "free"]
        if locks.get(STATE) == "held":
            problems.append("A recovery execute is running")
    report["not_recovered"] = ["startlux-reference environment", "StartLux-4B weights", "Qwen3.5-4B weights"]
    report["problems"] = problems
    report["status"] = "MATRIX_RUNTIME_NOT_READY" if problems else "MATRIX_RUNTIME_READY"
    report["next_step"] = ("bash scripts/run_direct_matrix.sh RUNTIME_ROOT --preflight-only from the matrix checkout "
                           "(GPU, CUDA and model behaviour are untested here)") if not problems else "inspect problems"
    return report


# --------------------------------------------------------------------------- commands

PHASE_ACTIONS = {
    "metadata": "restore the pinned runtime members from the verified archive (session, last_services and "
                "selected_* stay absent)",
    "sources": "git init / fetch --depth 1 / checkout --detach of the kev and StartLux-Decision commits at their "
               "original paths",
    "env-vllm": "uv venv (managed CPython 3.12.8) + exact archived pins; accepted only by the frozen legacy freeze hash",
    "toolkit": "pinned prepare_cuda_toolkit.py with the vLLM Python; accepted only with the archived receipt identity",
    "env-kev": "uv venv (managed CPython 3.12.8) + exact archived pins + flash-linear-attention; frozen freeze hash",
    "weights-kev": "pinned model_probe download --model kev into the V1 Hugging Face cache",
    "weights-qwen9b": "pinned model_probe download --model qwen into a new solo-v3/runtime-work",
}


def check_uv(uv):
    info = os.stat(uv)
    if not stat.S_ISREG(info.st_mode) or not os.access(uv, os.X_OK):
        raise RecoveryError(f"uv is not an executable file: {uv}")


def uv_version(sup, uv):
    result = sup.run([uv, "--version"], env=child_env(), timeout=TIMEOUT["query"], capture=True)
    if result.returncode:
        raise RecoveryError(f"{uv} --version failed: {result.stderr.strip()[-300:]}")
    return result.stdout.strip()


def matrix_output_state(layout):
    state = {"exists": lstat_chain(layout.root, MATRIX_OUTPUT) is not None}
    if state["exists"]:
        for name in ("settings.json", "runtime/binding.json", "status.json", ".active.lock"):
            state[name] = lstat_chain(layout.root, f"{MATRIX_OUTPUT}/{name}") is not None
        if state["status.json"]:
            try:
                status = read_json(layout.path(MATRIX_OUTPUT + "/status.json"))
                state["status"] = {"state": status.get("state"), "stages": {
                    stage: {k: str(v)[:400] for k, v in record.items() if k in ("state", "updated_at", "error",
                                                                                 "last_error", "message")}
                    for stage, record in status.get("stages", {}).items()}}
            except (OSError, ValueError, RecoveryError, AttributeError) as exc:
                state["status"] = f"unreadable: {exc}"
    return state


def plan(layout, sup, uv):
    """Read-only: nothing is created, modified or locked beyond a non-blocking shared probe."""
    blockers, warnings = list(ambient_blockers()), []
    report = {"command": "plan", "writes": "none", "runtime_root": str(layout.root)}
    try:
        report["matrix_checkout"] = check_matrix_checkout(sup, layout.checkout, layout.pins, layout.root)
    except RecoveryError as exc:
        blockers.append(str(exc))
    try:
        archive = open_archive(layout)
        report["archive"] = {"sha256": layout.pins["derived_from"]["v1"]["sha256"],
                             "members_to_restore": len(layout.pins["v1_runtime_members"])}
        installations = archived_json(archive, layout, "runtime/installations.json")
        lock = archived_json(archive, layout, "runtime/lock.json")
        if archived_root(installations) != layout.root:
            blockers.append(f"Archived paths name {archived_root(installations)}, not {layout.root}")
        else:
            for name in ENVIRONMENTS:
                member = f"runtime/resolved/{name}.txt"
                text = read_member(archive, member, layout.pins["v1_runtime_members"][member])[0]
                if sha256_bytes(text) != installations["environments"][name]["freeze_sha256"]:
                    blockers.append(f"{member} does not hash to the recorded freeze")
                check_local_sources(layout, name, text.decode(), lock)
    except (RecoveryError, KeyError, ValueError, OSError) as exc:
        blockers.append(f"Archive: {exc}")
    eggs = original_egg_info(layout.root)
    report["original_egg_info"] = eggs
    if not any(egg["version"] == "0.1.0" for egg in eggs):
        blockers.append("No decision-study 0.1.0 egg-info at the runtime root; the archived freezes cannot be reproduced")
    try:
        check_uv(uv)
        report["uv"] = {"path": str(uv), "version": uv_version(sup, uv)}
    except (RecoveryError, OSError) as exc:
        blockers.append(f"uv: {exc}")
    state = State.read_only(layout)
    phases = (state or {}).get("phases", {})
    if state and state.get("stop_unverified"):
        blockers.append(stop_unverified_message(state["stop_unverified"]))
    if state is None:
        for relative in OWNED:
            if lstat_chain(layout.root, relative) is not None:
                blockers.append(f"{relative} exists but no recovery state does; it is not recovery-owned")
    pending = [phase for phase in PHASES if phases.get(phase, {}).get("status") != "complete"]
    for phase in pending:
        if phases.get(phase, {}).get("terminal"):
            blockers.append(f"Phase {phase} ended in a terminal mismatch: {phases[phase].get('error')}")
    report["actions"] = [{"phase": phase, "status": phases.get(phase, {}).get("status", "pending"),
                          "action": PHASE_ACTIONS[phase], "disk_estimate_gib": DISK_GIB[phase]} for phase in pending]
    free = free_bytes(layout.path("results"))
    need = sum(DISK_GIB[phase] for phase in pending)
    report["disk"] = {"free_gib": round(free / GIB, 1), "pending_estimate_gib": need,
                      "note": "estimates include the uv cache; statvfs cannot see quotas"}
    if pending and free < DISK_GIB[pending[0]] * GIB:
        blockers.append(f"Only {free / GIB:.1f} GiB free; the next phase needs about {DISK_GIB[pending[0]]} GiB")
    elif free < need * GIB:
        warnings.append(f"{free / GIB:.1f} GiB free is below the {need} GiB estimate for all pending phases")
    locks = probe_locks(layout)
    report["locks"] = locks
    for relative, value in locks.items():
        if relative == STATE:
            if value == "held":
                blockers.append("A recovery execute is running")
        elif value != "free":
            blockers.append(f"Study lock {relative}: {value}")
    verify_controls(layout, blockers)
    report["matrix_output"] = matrix_output_state(layout)
    if not report["matrix_output"]["exists"]:
        blockers.append("Matrix output is missing")
    report.update(blockers=blockers, warnings=warnings, status="BLOCKED" if blockers else "READY_TO_EXECUTE")
    return report


def run_execute(layout, sup, uv):
    blockers = ambient_blockers()
    if blockers:
        raise RecoveryError("; ".join(blockers))
    check_uv(uv)
    os.umask(0o027)  # the node's umask; the toolkit tree records file modes
    check_matrix_checkout(sup, layout.checkout, layout.pins, layout.root)
    require_pins(layout, metadata=False)  # intact V3 and size-control files, before any write
    matrix = Matrix(layout.checkout)
    if archived_root(archived_json(open_archive(layout), layout, "runtime/installations.json")) != layout.root:
        raise RecoveryError("The archived absolute paths name another runtime root")
    if not any(egg["version"] == "0.1.0" for egg in original_egg_info(layout.root)):
        raise RecoveryError("No decision-study 0.1.0 egg-info at the runtime root; the archived freezes cannot be reproduced")
    with study_locks(layout) as held:
        state = State(layout).open()
        try:
            executor = Executor(layout, matrix, sup, state, str(uv))
            executor.uv_version = uv_version(sup, uv)
            executor.execute()
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            try:
                report = verify(layout, sup, matrix, state.dir / "verify" / stamp, own_locks=True)
            except StopUnverified as exc:
                state.mark_stop_unverified("final-verify", exc)
                raise
            write_json_atomic(state.dir / "receipts" / f"verify-{stamp}.json", report)
            report.update(command="execute", locks_held=held, state=str(state.file),
                          phases={phase: state.phase(phase).get("status") for phase in PHASES})
            return report
        finally:
            state.close()


def temp_parent(layout):
    """The OS temporary directory, validated before anything is created in it."""
    candidate = next((os.environ[k] for k in ("TMPDIR", "TEMP", "TMP") if os.environ.get(k)), "/tmp")
    parent = Path(os.path.realpath(candidate))
    if not parent.is_dir():
        raise RecoveryError(f"Temporary directory does not exist: {parent}")
    for protected in (layout.root, layout.checkout, PINS_FILE.parents[1]):
        protected = Path(os.path.realpath(protected))
        if parent == protected or protected in parent.parents:
            raise RecoveryError(f"TMPDIR resolves inside {protected}; point it outside every runtime root and checkout")
    return parent


def run_verify(layout, sup):
    parent = temp_parent(layout)
    try:
        identity = check_matrix_checkout(sup, layout.checkout, layout.pins, layout.root)
    except StopUnverified:
        raise
    except RecoveryError as exc:
        return {"command": "verify", "status": "MATRIX_RUNTIME_NOT_READY", "problems": [str(exc)]}
    matrix = Matrix(layout.checkout)
    workdir = Path(tempfile.mkdtemp(prefix="recover-node-runtime-verify-", dir=parent))
    info = os.lstat(workdir)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RecoveryError(f"Temporary check directory is not a directory owned by this user: {workdir}")
    report = verify(layout, sup, matrix, workdir)
    report.update(command="verify", matrix_checkout=identity)
    if report["status"] == "MATRIX_RUNTIME_READY":
        shutil.rmtree(workdir)
        report["check_logs"] = "removed"
    else:
        report["check_logs"] = str(workdir)
    return report


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Restore the node runtime direct-matrix-v1 reads (plan/execute/verify).")
    parser.add_argument("command", choices=("plan", "execute", "verify"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--matrix-checkout", type=Path, required=True)
    parser.add_argument("--matrix-commit", required=True)
    parser.add_argument("--uv", type=Path, help="installer (default: RUNTIME_ROOT/.venv-controller/bin/uv, the original)")
    return parser.parse_args(argv)


def main(argv=None, *, pins_path=PINS_FILE, pins_sha256=PINS_SHA256, supervisor=None):
    args = parse_args(argv)

    def interrupt(signum, frame):
        raise Interrupted(f"signal {signum}")

    previous, code = {}, 2
    try:
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupt)
        if args.matrix_commit != MATRIX_COMMIT:
            raise RecoveryError(f"--matrix-commit must be {MATRIX_COMMIT}")
        layout = Layout(check_root(args.runtime_root), args.matrix_checkout, load_pins(pins_path, pins_sha256),
                        pins_sha256=pins_sha256)
        sup = supervisor or Supervisor()
        uv = args.uv or layout.root / ".venv-controller/bin/uv"
        if args.command == "plan":
            report = plan(layout, sup, uv)
            code = 0 if report["status"] == "READY_TO_EXECUTE" else 1
        else:
            report = run_verify(layout, sup) if args.command == "verify" else run_execute(layout, sup, uv)
            code = 0 if report["status"] == "MATRIX_RUNTIME_READY" else 1
    except Interrupted as exc:
        report, code = {"status": "INTERRUPTED", "error": str(exc)}, 130
    except StopUnverified as exc:
        report, code = {"status": "STOP_UNVERIFIED", "error": str(exc)}, 4
    except TerminalMismatch as exc:
        report, code = {"status": "TERMINAL_MISMATCH", "error": str(exc)}, 3
    except Exception as exc:  # RecoveryError, OSError, or a frozen helper's own error
        report = {"status": "FAILED" if args.command == "execute" else "BLOCKED",
                  "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(exc, (RecoveryError, OSError)):
            report["traceback"] = traceback.format_exc()[-4000:]
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    report.setdefault("command", args.command)
    report["tool"] = {"name": TOOL, "script_sha256": sha256_file(Path(__file__).resolve()), "pins_sha256": pins_sha256}
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
