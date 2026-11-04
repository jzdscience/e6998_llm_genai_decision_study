#!/usr/bin/env python3
"""Add two missing StartLux reference dependencies without rebuilding any env.

This incident-specific repair is deliberately outside the controller modules
whose hashes are already bound to the completed Kev predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from decision_study import runtime
from decision_study.common import exclusive_run, file_hash, fingerprint, read_json, set_status, utc_now, write_json

ENVIRONMENT = "startlux-reference"
ADDITIONS = {"accelerate": "1.15.0", "psutil": "7.2.2"}
MARKER = "repair-startlux-reference-deps.json"
LOG = "repair-startlux-reference-deps.log"
CORE = ("adapters.py", "runner.py", "runtime.py", "storage.py", "common.py")
IMPORT_PROBE = """import accelerate, psutil
from accelerate import dispatch_model
from accelerate.utils import get_max_memory
from accelerate.utils.modeling import clean_device_map, get_max_layer_size
from transformers.utils import is_accelerate_available
from transformers.integrations.accelerate import check_and_set_device_map
assert is_accelerate_available()
assert accelerate.__version__ == '1.15.0'
assert psutil.__version__ == '7.2.2'
assert check_and_set_device_map({'': 'cpu'}) == {'': 'cpu'}
print('Accelerate imports and Transformers device_map dependency check passed')
"""


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _freeze(project, output, python):
    # Keep the same cwd as the original freeze: pip can see project egg-info.
    return runtime._run([python, "-m", "pip", "freeze", "--all"], output, LOG,
                        cwd=project, capture=True)


def _check_additive(before, after, *, complete=False):
    """Allow only the two exact new pins; retain every other original line."""
    def parts(text):
        preserved, additions = [], {}
        for line in text.splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
            name = re.sub(r"[-_.]+", "-", match[1]).lower() if match else None
            if name in ADDITIONS:
                if name in additions or match[2] != ADDITIONS[name]:
                    raise RuntimeError(f"Unexpected {name} version; environment preserved")
                additions[name] = match[2]
            else:
                preserved.append(line)
        return Counter(preserved), additions
    old, old_added = parts(before)
    new, new_added = parts(after)
    if old != new or not set(old_added) <= set(new_added):
        raise RuntimeError("Existing dependency pins or local package URLs changed; refusing repair")
    if complete and new_added != ADDITIONS:
        raise RuntimeError("Expected exact Accelerate and psutil additions are missing")


def _protected(project, output, installations):
    """Inputs to other model identities and their saved evidence must not change."""
    info = copy.deepcopy(installations)
    info["environments"].pop(ENVIRONMENT, None)
    paths = []
    for directory in (output / "data", output / "predictions/kev", output / "calibration"):
        if directory.exists():
            paths.extend(p for p in directory.rglob("*") if p.is_file())
    for name in ("runtime/lock.json", "runtime/doctor.json", "runtime/selected_profiles.json",
                 "runtime/selected_memory.json", "smoke/kev.json", "smoke/qwen.json"):
        path = output / name
        if path.exists():
            paths.append(path)
    paths.extend(p for p in (output / "runtime/resolved").glob("*.txt")
                 if not p.name.startswith(ENVIRONMENT))
    return {"installations_except_reference": info,
            "files": {str(p.relative_to(output)): file_hash(p) for p in sorted(set(paths))},
            "core_source": {name: file_hash(project / "decision_study" / name) for name in CORE}}


def repair_reference(project_dir, output_dir):
    """Caller holds the run lock/budget. Resume only our own additive mutation."""
    project, output = Path(project_dir).resolve(), Path(output_dir).resolve()
    installations_path = output / "runtime/installations.json"
    resolved = output / "runtime/resolved" / f"{ENVIRONMENT}.txt"
    marker_path = output / "runtime" / MARKER
    if not (output / "runtime/lock.json").is_file():
        raise RuntimeError("Existing pinned runtime is required; this repair never installs models")
    lock = runtime.freeze_lock(project, output)
    runtime._assert_repair_safe(output, {ENVIRONMENT}, lock)
    if runtime._pending_repair(output, ENVIRONMENT):
        raise RuntimeError("A Python environment rebuild is pending; do not combine two repairs")
    installations = read_json(installations_path)
    if installations.get("lock_hash") != lock["lock_hash"]:
        raise RuntimeError("Runtime installation lock mismatch")
    info = installations["environments"].get(ENVIRONMENT)
    env_dir = runtime._work_dir(output) / "envs" / ENVIRONMENT
    python = env_dir / "bin/python"
    if (not info or info.get("python") != str(python) or not python.is_file()
            or env_dir.is_symlink() or env_dir.parent.is_symlink()):
        raise RuntimeError("Reference environment path missing, changed or symlinked; refusing repair")
    if not resolved.is_file() or resolved.is_symlink() or installations_path.is_symlink() or marker_path.is_symlink():
        raise RuntimeError("Missing or symlinked runtime metadata; refusing repair")
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required; activate the existing controller environment")

    if marker_path.exists():
        marker = read_json(marker_path)
        if (marker.get("schema_version") != 1 or marker.get("additions") != ADDITIONS
                or marker.get("environment_path") != str(env_dir)
                or marker.get("lock_hash") != lock["lock_hash"]
                or marker.get("status") not in {"pending", "complete"}):
            raise RuntimeError("Existing dependency repair marker does not match this operation")
        archive = Path(marker["archive"])
        archive_root = output / "runtime/resolved"
        if (archive.parent != archive_root or not archive.name.startswith("startlux-deps-")
                or archive.is_symlink() or not archive.is_dir()):
            raise RuntimeError("Dependency repair archive path changed")
        before = (archive / "reference-before.txt").read_text()
        old_info = read_json(archive / "installations-before.json")["environments"][ENVIRONMENT]
        if (_sha(before) != marker["before_freeze_sha256"]
                or fingerprint(old_info) != marker["before_info_hash"]):
            raise RuntimeError("Dependency repair archive changed")
        if marker["status"] == "pending" and _protected(project, output, installations) != marker["protected"]:
            raise RuntimeError("Unrelated model metadata or frozen results changed during repair")
        expected_infos = [old_info]
        if marker.get("new_info"):
            expected_infos.append(marker["new_info"])
        if info not in expected_infos:
            raise RuntimeError("Reference metadata changed outside this repair")
        if resolved.read_text() not in [before, marker.get("new_freeze")]:
            raise RuntimeError("Resolved reference requirements changed outside this repair")
        current = _freeze(project, output, python)
        _check_additive(before, current, complete=marker["status"] == "complete")
        if marker.get("new_freeze") and current != marker["new_freeze"]:
            raise RuntimeError("Environment differs from the verified repair candidate")
        if marker["status"] == "complete":
            if info != marker["new_info"] or resolved.read_text() != marker["new_freeze"]:
                raise RuntimeError("Completed repair metadata is inconsistent")
            return {"status": "complete", "already_complete": True, "environment": ENVIRONMENT,
                    "added_pins": ADDITIONS, "checkpoints_downloaded": False}
    else:
        before = resolved.read_text()
        current = _freeze(project, output, python)
        if current != before or _sha(before) != info["freeze_sha256"]:
            raise RuntimeError("Reference environment changed before repair authorization")
        _check_additive(before, current)
        archive = output / "runtime/resolved" / f"startlux-deps-{time.time_ns()}"
        archive.mkdir(parents=True)
        runtime._write_text(archive / "reference-before.txt", before)
        write_json(archive / "installations-before.json", installations)
        old_info = copy.deepcopy(info)
        marker = {"schema_version": 1, "status": "pending", "started_at": utc_now(),
                  "environment_path": str(env_dir), "lock_hash": lock["lock_hash"],
                  "additions": ADDITIONS, "archive": str(archive),
                  "before_freeze_sha256": _sha(before), "before_info_hash": fingerprint(info),
                  "protected": _protected(project, output, installations)}
        write_json(marker_path, marker)  # Before the first environment mutation.

    # StartLux primary hashes its serving environment, not the reference venv.
    # Never let a previously cached smoke stand in for the repaired reference.
    smoke_path = output / "smoke/startlux.json"
    if smoke_path.is_symlink():
        raise RuntimeError("Refusing to invalidate a symlinked smoke report")
    if smoke_path.exists():
        archived_smoke = archive / "smoke-startlux-before.json"
        if archived_smoke.exists() and archived_smoke.read_bytes() != smoke_path.read_bytes():
            raise RuntimeError("StartLux smoke changed while dependency repair was pending")
        shutil.copy2(smoke_path, archived_smoke)
        smoke_path.unlink()

    if not marker.get("new_freeze"):
        runtime._run([uv, "pip", "install", "--python", python, "--no-deps", "--only-binary", ":all:",
                      *[f"{name}=={version}" for name, version in ADDITIONS.items()]],
                     output, LOG, cwd=project)
        frozen = _freeze(project, output, python)
        _check_additive(before, frozen, complete=True)
        runtime._run([uv, "pip", "check", "--python", python], output, LOG, cwd=project)
        runtime._run([python, "-c", IMPORT_PROBE], output, LOG, cwd=project)
        probe_path = output / "runtime/gpu-check-startlux-reference-deps.json"
        runtime._run([python, project / "scripts/model_probe.py", "gpu-check", "--model", "startlux",
                      "--output-dir", output, "--result", probe_path], output, LOG, cwd=project)
        probe = read_json(probe_path)
        if probe.get("status") != "ok" or probe.get("triton_build", {}).get("status") != "ok":
            raise RuntimeError("Repaired reference environment failed the GPU/Triton probe")
        if _freeze(project, output, python) != frozen:
            raise RuntimeError("Reference dependencies changed during verification")
        new_info = {**old_info, "freeze_sha256": _sha(frozen), "gpu_probe": probe,
                    "dependency_repair_completed_at": utc_now()}
        marker.update(new_freeze=frozen, new_info=new_info, verified_at=utc_now())
        write_json(marker_path, marker)

    installations = read_json(installations_path)
    if _protected(project, output, installations) != marker["protected"]:
        raise RuntimeError("Unrelated model state changed; reference repair commit stopped")
    # The pending marker contains both commit values, so either interrupted
    # write can be completed on resume without adopting external drift.
    runtime._write_text(resolved, marker["new_freeze"])
    installations["environments"][ENVIRONMENT] = marker["new_info"]
    write_json(installations_path, installations)
    if _protected(project, output, read_json(installations_path)) != marker["protected"]:
        raise RuntimeError("Protected model state changed unexpectedly")
    marker.update(status="complete", completed_at=utc_now())
    write_json(marker_path, marker)
    return {"status": "complete", "already_complete": False, "environment": ENVIRONMENT,
            "added_pins": ADDITIONS, "checkpoints_downloaded": False,
            "next_step": "Run smoke for StartLux and Qwen; existing Kev results remain unchanged."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    from decision_study.artifacts import export_bundle
    from decision_study.budget import Budget, ensure_session
    stage, acquired = "repair-startlux-reference-deps", False
    try:
        with exclusive_run(output):
            acquired = True
            try:
                ensure_session(output)
                with Budget(output, "pilot", cost_mode="provided"):
                    result = repair_reference(PROJECT, output)
                set_status(output, stage, "complete")
                print(json.dumps(result, indent=2), flush=True)
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                set_status(output, stage, "blocked", error=f"{type(exc).__name__}: {exc}")
                diagnostic = {"stage": stage, "error": str(exc), "exception_type": type(exc).__name__,
                              "traceback": traceback.format_exc()}
                if getattr(exc, "log_path", None):
                    diagnostic["subprocess_log"] = exc.log_path
                write_json(output / "diagnostics/last_error.json", diagnostic)
                print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
            finally:
                try:
                    print(f"PILOT BUNDLE: {export_bundle(output, 'pilot')}", flush=True)
                except Exception as exc:
                    print(f"Could not export pilot bundle: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        if not acquired:
            print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
