"""Profile the 90 already-exposed V3 development inputs on three sequential services.

This module never opens gold. Its timing describes these old development inputs,
not a future fitting population, a service SLO or a new quality result.
Existing frozen modules are dependencies, not modified or monkeypatched here.
"""
from contextlib import contextmanager
import argparse
import copy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

from .. import runtime as frozen
from ..adapters import ModelClient, decision_request, parse_response, qwen_request
from ..common import StudyError, canonical, exclusive_run, file_hash, fingerprint, read_json, utc_now, write_json
from ..runtime_compat import legacy_environment
from ..service_study.identity import fixed_write
from ..service_study.resources import ResourceMonitor, hardware
from ..upgrade_study import runtime as old
from ..upgrade_study.identity import V3_EXAMPLES_SHA256
from ..upgrade_study.launcher import managed_service
from ..v3 import runtime as v3
from ..v3.cli import slurm_hours_left
from ..direct_matrix.runtime import slurm_job_state

PROJECT = Path(__file__).resolve().parents[2]
ROLES = ("startlux4b", "qwen9b", "startlux9b")
INTERFACE = {"startlux4b": "startlux", "qwen9b": "qwen", "startlux9b": "startlux"}
REVISIONS = {"startlux4b": "416088f77cfce09633906a3675c7cd2e5dabcf10",
             "qwen9b": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             "startlux9b": "18a17f734966b213b2f00fa9c915e8b9af2a9b20"}
PROTOCOL = {"id": "anli-learnability-development-profile-v1", "revision": 1,
            "roles": list(ROLES), "gpu_memory_utilization": 0.8, "max_num_seqs": 64,
            "max_model_len": 8192, "count": 90, "concurrency": 1,
            "socket_read_timeout_seconds": 120, "request_acceptance_seconds": 120, "max_attempts": 1,
            "role_watchdog_seconds": 3600, "cleanup_margin_seconds": 120,
            "data": "V3 development: already-exposed official train inputs only",
            "scope": "Feasibility, startup, token counts and collection time only. No new evaluation, gold, training, SLO or representative future fitting-population cost claim."}
DEVELOPMENT_SHA256 = "40f2d5fa8a611eb210e19166accf57d86803bc807f4dc363a49f43890503a458"
CONTROL_NAMES = ("lock.json", "installations.json", "selected_profiles.json", "selected_memory.json")
INPUT_FIELDS = {"sample_id", "task", "split", "group_id", "round", "source_split", "state", "instructions",
                "labels", "criteria", "premise_hash", "input_hash"}


def normal(path):
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def validate_output(output, runtime_root=None, *, existing=False):
    """Run before locks, mkdir, diagnostics or exports can write to a supplied path."""
    output = normal(output)
    safe(output, "settings.json")
    if output == PROJECT or output in PROJECT.parents or PROJECT in output.parents:
        raise StudyError("Profile output must be outside the source checkout")
    if runtime_root is not None:
        runtime_root = normal(runtime_root)
        if (output.parent != runtime_root/"results" or not output.name.startswith("aux-anli-learnability-profile-v1")):
            raise StudyError("Use a new aux-anli-learnability-profile-v1 output directly under the runtime results root")
    marker = safe(output, "settings.json")
    if marker.exists():
        if read_json(marker).get("protocol") != PROTOCOL:
            raise StudyError("Output is not this development profile; no files were changed")
    elif existing or (output.exists() and any(output.iterdir())):
        raise StudyError("Output is missing its profile identity or contains unrelated files; no files were changed")


def safe(root, relative):
    root = normal(root)
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise StudyError("Unsafe profile path")
    for p in (*root.parents, root):
        if p.is_symlink():
            raise StudyError(f"Symlinked profile root: {p}")
    p = root
    for part in parts:
        p /= part
        if p.is_symlink():
            raise StudyError(f"Symlinked profile file: {p}")
    return p


def source_identity():
    """Content identity is authoritative; dirty/commit information is truthful metadata."""
    paths = set()
    for relative in ("decision_study", "scripts", "configs"):
        for directory, names, files in os.walk(PROJECT / relative, followlinks=False):
            names[:] = sorted(n for n in names if n != "__pycache__" and not (Path(directory) / n).is_symlink())
            for name in files:
                path = Path(directory) / name
                if path.suffix in {".py", ".json", ".txt", ".sh"}:
                    if path.is_symlink():
                        raise StudyError(f"Symlinked source dependency: {path}")
                    paths.add(path)
    hashes = {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(paths)}
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL,
                                           timeout=20).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"files": hashes, "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "note": "Exact current file hashes, including uncommitted source; not a clean-release claim"}


def inputs(v3_dir):
    path = safe(v3_dir, "data/examples.jsonl")
    if file_hash(path) != V3_EXAMPLES_SHA256:
        raise StudyError("V3 prepared-input SHA differs from the accepted development source")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = sorted((r for r in rows if r.get("split") == "development"), key=lambda r: r["sample_id"])
    if len(result) != 90 or len({r["sample_id"] for r in result}) != 90 or fingerprint(result) != DEVELOPMENT_SHA256:
        raise StudyError("Expected exactly 90 distinct V3 development inputs")
    if any(set(r) != INPUT_FIELDS or r.get("task") != "anli" or r.get("source_split") != "train"
           or r.get("labels") != ["entailment", "neutral", "contradiction"] for r in result):
        raise StudyError("Profile inputs must be the unmodified train-only V3 development rows without gold")
    return result


def control(output, role):
    if role not in ROLES:
        raise StudyError(f"Unknown role: {role}")
    return safe(output, "runtime-controls/" + role)


def plans(dirs, output):
    output = normal(output)
    for p in [PROJECT, *map(normal, dirs.values())]:
        if output == p or output in p.parents or p in output.parents:
            raise StudyError("Profile output must not overlap source or historical study roots")
    # These helpers read accepted source locks/snapshots; returned dictionaries are copies.
    binding, controls = old.plans(dirs["v1"], dirs["v3"], dirs["size_control"], output)
    binding = copy.deepcopy(binding)
    binding.update(protocol=PROTOCOL["id"], profile_revision=PROTOCOL["revision"], roles=list(ROLES))
    for role in ROLES:
        model = INTERFACE[role]
        controls[role]["selected_memory.json"] = {model: PROTOCOL["gpu_memory_utilization"]}
        lock = controls[role]["lock.json"]
        record = controls[role]["installations.json"]["models"][model]
        if lock["max_model_len"] != 8192 or record["revision"] != REVISIONS[role]:
            raise StudyError(f"Unexpected {role} context/checkpoint")
        snapshot = Path(record["model_path"])
        if not snapshot.is_dir() or snapshot.name != REVISIONS[role]:
            raise StudyError(f"Missing pinned snapshot for {role}; no download or substitution")
    binding["control_identity"] = {r: {n: fingerprint(v) for n, v in controls[r].items()} for r in ROLES}
    return binding, controls


def prepare(dirs, output):
    dirs = {k: str(normal(v)) for k, v in dirs.items()}
    binding, controls = plans(dirs, output)
    rows = inputs(dirs["v3"])
    expected = {"protocol": PROTOCOL, "source_dirs": dirs, "input_source_sha256": V3_EXAMPLES_SHA256,
                "inputs_sha256": fingerprint(rows), "binding": binding, "source": source_identity()}
    settings_path = safe(output, "settings.json")
    if settings_path.exists():
        previous = read_json(settings_path)
        # Unrelated doc/commit metadata changes do not impersonate a new source identity.
        expected["source"] = {**expected["source"], **{k: previous["source"].get(k)
                                                   for k in ("git_commit", "git_dirty", "note")}}
        if previous != expected:
            raise StudyError("Profile source, input or runtime identity changed; use a new output and preserve this one")
    fixed_write(settings_path, expected)
    fixed_write(safe(output, "data/development.json"), rows)
    for role in ROLES:
        for name in CONTROL_NAMES:
            fixed_write(safe(control(output, role), "runtime/" + name), controls[role][name])
    return expected


def context(output, *, offline=False):
    settings = read_json(safe(output, "settings.json"))
    if settings["protocol"] != PROTOCOL or settings["source"]["files"] != source_identity()["files"]:
        raise StudyError("Profile source or fixed protocol changed")
    rows = read_json(safe(output, "data/development.json"))
    if fingerprint(rows) != DEVELOPMENT_SHA256 or settings["input_source_sha256"] != V3_EXAMPLES_SHA256:
        raise StudyError("Profile inputs differ from the pinned V3 development selection")
    if offline:
        binding = settings["binding"]
        if binding.get("protocol") != PROTOCOL["id"] or binding.get("roles") != list(ROLES):
            raise StudyError("Returned profile binding identity differs")
        for role in ROLES:
            identity = binding.get("control_identity", {}).get(role, {})
            if set(identity) != set(CONTROL_NAMES):
                raise StudyError("Returned profile control inventory differs")
            for name, digest in identity.items():
                if fingerprint(read_json(safe(control(output, role), "runtime/"+name))) != digest:
                    raise StudyError("Returned profile control differs from the recorded node binding")
        return settings, rows
    binding, controls = plans(settings["source_dirs"], output)
    if (binding != settings["binding"] or fingerprint(rows) != settings["inputs_sha256"]
            or rows != inputs(settings["source_dirs"]["v3"])):
        raise StudyError("Profile runtime or exact development inputs changed")
    for role in ROLES:
        for name in CONTROL_NAMES:
            if read_json(safe(control(output, role), "runtime/" + name)) != controls[role][name]:
                raise StudyError(f"Profile runtime control changed: {role}/{name}")
    return settings, rows


def initialize(output, *, new_allocation=False):
    settings, _ = context(output)
    hours = slurm_hours_left()
    allocation = v3._allocation()
    if not allocation.get("slurm_job_id") or not math.isfinite(hours) or not 0 < hours <= 24:
        raise StudyError("Profile requires an actual current Slurm allocation with at most 24 hours remaining")
    path = safe(output, "runtime/session.json")
    previous = read_json(path) if path.exists() else None
    deadline = time.time() + hours * 3600
    if previous:
        if previous["allocation"] == allocation:
            if new_allocation:
                raise StudyError("Cannot renew the same allocation")
            deadline = min(deadline, previous["deadline_epoch"])
        else:
            old_job = previous["allocation"].get("slurm_job_id")
            if not new_allocation or old_job == allocation["slurm_job_id"]:
                raise StudyError("A genuinely new Slurm job requires --new-allocation")
            proof = slurm_job_state(old_job)
            fixed_write(safe(output, f"runtime/allocation-checks/{time.time_ns()}.json"), proof)
            if not proof.get("terminal"):
                raise StudyError("Cannot verify that the earlier Slurm allocation ended; no services started")
    if deadline <= time.time() + PROTOCOL["cleanup_margin_seconds"]:
        raise StudyError("Allocation allowance has expired or reached cleanup margin")
    session = {"allocation": allocation, "deadline_epoch": deadline, "updated_at": utc_now(),
               "settings_hash": fingerprint(settings), "notice": "Actual squeue allowance, no deadline extension"}
    if previous:
        fixed_write(safe(output, "runtime/session-history/" + fingerprint(previous) + ".json"), previous)
    write_json(path, session)
    for role in ROLES:
        write_json(safe(control(output, role), "runtime/session.json"), session)
    return session


def session(output):
    value = v3._session(output)
    if value["settings_hash"] != fingerprint(read_json(safe(output, "settings.json"))):
        raise StudyError("Session settings changed")
    return value


def token_check(output, role, rows):
    model = INTERFACE[role]
    target = control(output, role)
    report = frozen.token_preflight_examples(PROJECT, target, model, rows)
    path = target / "runtime" / f"tokens-interactive-{model}-{frozen._hash(rows)[:16]}.json"
    receipt = {"file": str(path.relative_to(normal(output))), "sha256": file_hash(path),
               "runtime_hash": fingerprint(frozen.runtime_config(target, model))}
    verify_tokens(output, role, rows, receipt)
    return receipt


def verify_tokens(output, role, rows, receipt):
    path = safe(output, receipt["file"])
    expected_name = f"runtime-controls/{role}/runtime/tokens-interactive-{INTERFACE[role]}-{frozen._hash(rows)[:16]}.json"
    if receipt["file"] != expected_name or file_hash(path) != receipt["sha256"]:
        raise StudyError("Raw token report is missing or changed")
    report = read_json(path)
    tokens = report.get("rows", [])
    if (report.get("status") != "ok" or report.get("model") != INTERFACE[role]
            or report.get("model_revision") != REVISIONS[role] or report.get("max_model_len") != 8192
            or report.get("n_examples") != len(rows) or report.get("truncated") is not False
            or report.get("overlong") != [] or [r.get("sample_id") for r in tokens] != [r["sample_id"] for r in rows]
            or report.get("effective_config_hash") != receipt["runtime_hash"]
            or any(not isinstance(t.get("input_tokens"), int) or isinstance(t.get("input_tokens"), bool)
                   or t["input_tokens"] < 1 or t.get("reserved_output_tokens") != (64 if role == "qwen9b" else 1)
                   or t["input_tokens"] + t["reserved_output_tokens"] > 8192 for t in tokens)):
        raise StudyError("Token report does not cover the exact train-development inputs without truncation")


@contextmanager
def deployment(output, role, *, sink, launch=None, lister=None):
    target, model = control(output, role), INTERFACE[role]
    meta_path = safe(target, "runtime/last_services.json")
    before = file_hash(meta_path) if meta_path.exists() else None
    began = time.time()
    sink.update(role=role, began_epoch=began, processes=[])
    owned = set()
    try:
        sink["gpu_before"] = old.assert_gpu_owned_by(set(), lister=lister)
        start = time.monotonic()
        try:
            with (launch or managed_service)(target, model, max_num_seqs=64) as endpoints:
                meta = old._launch_record(meta_path, before, began)
                if not meta or not meta.get("processes"):
                    raise StudyError("Launch returned without fresh process ownership metadata")
                ready = time.monotonic()
                sink.update(processes=meta["processes"], startup_seconds=ready-start,
                            startup_started_monotonic=start, ready_monotonic=ready,
                            runtime=meta["models"][model]["config"], health=meta["models"][model]["health"],
                            engine_log=old.engine_log_facts(meta["processes"]))
                owned.update(int(p["pid"]) for p in meta["processes"])
                sink["gpu_started"] = old.assert_gpu_owned_by(owned, lister=lister)
                verify_engine(sink, role)
                yield endpoints[model]
        finally:
            # Includes a launch that started the engine then failed before yielding.
            meta = old._launch_record(meta_path, before, began)
            if meta:
                sink["processes"] = meta.get("processes", [])
                owned.update(int(p["pid"]) for p in sink["processes"])
    finally:
        sink["stop"] = observe_stopped(owned, lister=lister)


def observe_stopped(groups, *, lister=None, grace=15):
    """Worker checks historical launch PGIDs read-only; only the parent recovers orphans.

    A numeric PID/PGID from a launch record is not authority to kill a later process.
    The unchanged launcher still stops the subprocesses it directly owns.
    """
    until = time.monotonic()+grace
    while True:
        alive = [r for r in old.process_table() if r["pgid"] in groups]
        if not alive or time.monotonic() >= until:
            break
        time.sleep(.1)
    gpu = (lister or old.gpu_processes)()
    if alive or gpu:
        raise StudyError("Services have not completely stopped; parent ancestry-based recovery is required")
    return {"owned_groups": sorted(groups), "gpu_compute_processes": gpu, "mode": "read_only_observation"}


def verify_engine(record, role):
    engine = [p for p in record.get("processes", []) if p.get("name") == INTERFACE[role] + "-engine"]
    if len(engine) != 1:
        raise StudyError("Missing unique engine launch command")
    command = engine[0].get("command", [])
    def flag(name, value):
        return command.count(name) == 1 and command.index(name)+1 < len(command) and command[command.index(name)+1] == value
    if not flag("--max-num-seqs", "64") or not flag("--gpu-memory-utilization", "0.8"):
        raise StudyError("Engine command lacks the fixed sequence cap or memory share")
    runtime = record.get("runtime", {})
    if (runtime.get("model_revision") != REVISIONS[role] or runtime.get("max_model_len") != 8192
            or runtime.get("gpu_memory_utilization") != .8 or runtime.get("truncate") is not False):
        raise StudyError("Effective computation differs from the profile")


def request_body(row, model):
    return qwen_request(row, model) if model == "qwen" else decision_request(row, model)


def validate_rows(rows, predictions, role):
    if len(predictions) != len(rows):
        raise StudyError("Incomplete profile predictions")
    for e, r in zip(rows, predictions):
        if (r.get("sample_id") != e["sample_id"] or r.get("task") != "anli" or r.get("split") != "development"
                or r.get("model") != INTERFACE[role] or r.get("status") != "ok"
                or r.get("input_sha256") != fingerprint(e)
                or r.get("request_sha256") != fingerprint(request_body(e, INTERFACE[role]))):
            raise StudyError("Profile prediction failed or does not bind the exact input/request")
        if parse_response(r.get("raw_response"), e, INTERFACE[role]) != (r.get("prediction"), r.get("probabilities")):
            raise StudyError("Profile prediction differs from its preserved raw response")
        attempts = r.get("attempts", [])
        if len(attempts) != 1 or attempts[0].get("status") != "ok" or attempts[0].get("attempt") != 1:
            raise StudyError("Unexpected profile request attempts")
        latency = r.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (float, int)) or not math.isfinite(latency) or latency < 0:
            raise StudyError("Invalid profile latency")
        elapsed = r.get("observed_request_seconds")
        start, finish = r.get("request_started_monotonic"), r.get("request_finished_monotonic")
        values = (elapsed, start, finish)
        if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values)
                or not math.isclose(finish-start, elapsed, abs_tol=1e-8)
                or elapsed > PROTOCOL["request_acceptance_seconds"]
                or latency/1000 > elapsed+1e-3 or r.get("profile_acceptance") != "on_time"):
            raise StudyError("Profile response was late or has inconsistent monotonic timing")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def worker(output, folder):
    """Entire token/start/collection phase is bounded by the parent's watchdog."""
    plan = read_json(safe(folder, "plan.json"))
    role = plan["role"]
    settings, rows = context(output)
    sess = session(output)
    if plan["settings_hash"] != fingerprint(settings) or plan["allocation"] != sess["allocation"]:
        raise StudyError("Worker plan identity changed")
    from .ownership import verify_receipt
    verify_receipt(output, plan["ownership_selftest"], settings, plan["allocation"])
    sess = {**sess, "deadline_epoch": min(sess["deadline_epoch"], plan["deadline_epoch"])}
    binding, target = settings["binding"], control(output, role)
    record = {"role": role}
    try:
        with legacy_environment(binding, PROJECT), v3._environment(binding, sess):
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            write_json(safe(target, "runtime/session.json"), sess)
            doctor = frozen.doctor(target, require_gpu=True)
            gpus = doctor.get("gpu", [])
            if len(gpus) != 1 or gpus[0].get("name") != "NVIDIA L40S" or gpus[0].get("memory_mb", 0) < 44000:
                raise StudyError("Profile requires exactly one L40S with at least 44,000 MiB")
            installed = read_json(safe(target, "runtime/installations.json"))
            os.environ.update(v3._cuda_preflight(target, binding, installed))
            receipt = token_check(output, role, rows)
            fixed_write(safe(folder, "tokens.json"), receipt)
            fixed_write(safe(folder, "hardware.json"), hardware())
            with deployment(output, role, sink=record) as endpoint:
                client = ModelClient(INTERFACE[role], endpoint, timeout=PROTOCOL["socket_read_timeout_seconds"], max_attempts=1)
                with ResourceMonitor([os.getpid(), *[p["pid"] for p in record["processes"]]], safe(folder, "resources.jsonl")):
                    started = time.monotonic()
                    with safe(folder, "rows.jsonl").open("x") as stream:
                        for row in rows:
                            if time.time() >= sess["deadline_epoch"]:
                                raise StudyError("Profile stage deadline reached")
                            request_start = time.monotonic()
                            pred = client.predict(row)
                            request_finish = time.monotonic()
                            elapsed_request = request_finish-request_start
                            pred.update(input_sha256=fingerprint(row), request_sha256=fingerprint(request_body(row, INTERFACE[role])))
                            pred.update(request_started_monotonic=request_start, request_finished_monotonic=request_finish,
                                        observed_request_seconds=elapsed_request,
                                        profile_acceptance="on_time" if elapsed_request <= PROTOCOL["request_acceptance_seconds"] else "late")
                            stream.write(json.dumps(pred, ensure_ascii=False, allow_nan=False)+"\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                            if pred["status"] != "ok":
                                raise StudyError(f"Profile request failed for {row['sample_id']}; no silent retry")
                            if pred["profile_acceptance"] == "late":
                                raise StudyError("Response arrived after the monotonic acceptance deadline; raw row preserved")
                    finished = time.monotonic()
                    elapsed = finished-started
            predictions = read_rows(safe(folder, "rows.jsonl"))
            validate_rows(rows, predictions, role)
            record["collection_seconds"] = elapsed
            record.update(collection_started_monotonic=started, collection_finished_monotonic=finished)
            record["latency_ms"] = {"median": statistics.median(r["latency_ms"] for r in predictions),
                                     "max": max(r["latency_ms"] for r in predictions)}
            fixed_write(safe(folder, "deployment.json"), record)
            files = {name: file_hash(safe(folder, name)) for name in
                     ("plan.json", "tokens.json", "hardware.json", "resources.jsonl", "rows.jsonl", "deployment.json")}
            fixed_write(safe(folder, "complete.json"), {"role": role, "files": files, "finished_at": utc_now(),
                                                        "count": len(rows), "scope": PROTOCOL["scope"]})
    finally:
        if not safe(folder, "deployment.json").exists():
            fixed_write(safe(folder, "deployment.json"), record)


def stop_child(process, since):
    """Signal only the still-waitable direct child; descendant recovery is separate."""
    if process.poll() is not None:
        return  # Its numeric PID may now be reused; never signal its old PGID.
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@contextmanager
def subreaper():
    """Linux adopts orphaned probe/engine descendants, including separate sessions.

    The controller runs one role worker and no unrelated asynchronous subprocesses.
    This is not enabled on macOS: actual GPU execution is Linux-only.
    """
    if sys.platform != "linux":
        raise StudyError("Actual profile workers require Linux child-subreaper ownership")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise StudyError("Linux kernel pidfd support and Python os.pidfd_open/signal.pidfd_send_signal APIs are required for PID-reuse-safe orphan recovery")
    try:
        fd = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
    except OSError as exc:
        raise StudyError("Cannot establish kernel pidfd ownership; a supported Linux kernel (pidfd_open: 5.3+) and enabled pidfd APIs are required") from exc
    if any(r["ppid"] == os.getpid() for r in process_rows()):
        raise StudyError("Profile subreaper requires a sole worker; pre-existing controller children found")
    libc = ctypes.CDLL(None, use_errno=True)
    before = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(before), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
        raise StudyError("Cannot establish Linux child-subreaper ownership")
    try:
        yield
    finally:
        if libc.prctl(36, before.value, 0, 0, 0) != 0:
            raise StudyError("Cannot restore child-subreaper state")


def process_rows():
    """Read PID/parent/group/birth identity, excluding the observation command itself."""
    command = ["ps", "-eo", "pid=,ppid=,pgid=,stat=,lstart="]
    observer = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env={**os.environ, "LC_ALL": "C"})
    try:
        stdout, stderr = observer.communicate(timeout=5)
    except BaseException:
        observer.kill()
        observer.wait(timeout=5)
        raise
    if observer.returncode != 0:
        raise StudyError("Cannot inventory process birth identities")
    rows = []
    for line in stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5 or not all(f.isdigit() for f in fields[:3]):
            continue
        try:
            started = time.mktime(time.strptime(" ".join(fields[4].split()), "%a %b %d %H:%M:%S %Y"))
        except ValueError:
            raise StudyError("Cannot establish an adopted process's birth time") from None
        if int(fields[0]) == observer.pid:
            continue
        rows.append({"pid": int(fields[0]), "ppid": int(fields[1]), "pgid": int(fields[2]),
                     "started": started, "zombie": fields[3].startswith("Z")})
    return rows


def owned_descendants(since):
    """Current ancestry, never launch-record PGIDs, establishes cleanup ownership."""
    rows = []
    for record in process_rows():
        if record["zombie"]:
            if record["ppid"] == os.getpid() and record["started"] >= since-2:
                try:
                    os.waitpid(record["pid"], os.WNOHANG)
                except ChildProcessError:
                    pass
            continue
        rows.append(record)
    selected = {r["pid"] for r in rows if r["ppid"] == os.getpid() and r["started"] >= since-2}
    while True:
        following = selected | {r["pid"] for r in rows if r["ppid"] in selected and r["started"] >= since-2}
        if following == selected:
            break
        selected = following
    return {r["pid"]: r for r in rows if r["pid"] in selected}


def adopted_groups(since):
    return {r["pgid"] for r in owned_descendants(since).values()}


def signal_descendant(pid, since, sig):
    """Bind a kernel pidfd, then recheck current ancestry before sending a signal."""
    if pid not in owned_descendants(since):
        return False
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        if pid not in owned_descendants(since):
            return False
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


def recover_stop(output, role, plan, *, worker_pid=None, journal=None, max_seconds=60, term_seconds=10,
                 check_gpu=True):
    path = safe(control(output, role), "runtime/last_services.json")
    meta = old._launch_record(path, plan["last_services_before"], plan["started_epoch"])
    recorded = {int(p["pid"]) for p in meta.get("processes", [])} if meta else set()
    began = time.monotonic()
    deadline = began+max_seconds
    observed, signals, empty = {}, [], 0
    while time.monotonic() < deadline:
        owned = owned_descendants(plan["started_epoch"])
        observed.update(owned)
        if journal is not None:
            with Path(journal).open("a") as stream:
                stream.write(json.dumps({"at": utc_now(), "owned": list(owned.values())})+"\n")
        if not owned:
            empty += 1
            if empty == 2:
                gpu = old.assert_gpu_owned_by(set()) if check_gpu else None
                return {"recorded_engine_groups": sorted(recorded), "recorded_worker_pid": worker_pid,
                        "observed_owned_processes": list(observed.values()), "signals": signals,
                        "gpu_compute_processes": gpu, "gpu_checked": check_gpu, "seconds": time.monotonic()-began,
                        "ownership": "current descendants of sole-worker Linux subreaper; kernel pidfds; historical PIDs never authorize signals"}
        else:
            empty = 0
            sig = signal.SIGTERM if time.monotonic()-began < term_seconds else signal.SIGKILL
            for pid in sorted(owned, reverse=True):
                if time.monotonic() >= deadline:
                    break
                if signal_descendant(pid, plan["started_epoch"], sig):
                    signals.append({"pid": pid, "signal": sig.name})
        time.sleep(.1)
    raise StudyError("Parent cleanup deadline reached while owned descendants remain; completion refused")


def worker_command(output, folder):
    return [sys.executable, "-B", "-m", "decision_study.learnability_profile", "worker",
            "--output-dir", str(output), "--attempt-dir", str(folder)]


def run_attempt(output, role):
    with subreaper():
        return _run_attempt(output, role)


def _run_attempt(output, role):
    settings, _ = context(output)
    sess = session(output)
    from .ownership import require_receipt
    ownership_proof = require_receipt(output)
    base = safe(output, "attempts/"+role)
    base.mkdir(parents=True, exist_ok=True)
    folder = safe(base, str(time.time_ns()))
    folder.mkdir()
    until = min(sess["deadline_epoch"]-PROTOCOL["cleanup_margin_seconds"], time.time()+PROTOCOL["role_watchdog_seconds"])
    if until-time.time() < 60:
        raise StudyError("Insufficient allowance to start a profile role")
    meta = safe(control(output, role), "runtime/last_services.json")
    plan = {"role": role, "settings_hash": fingerprint(settings), "allocation": sess["allocation"],
            "started_epoch": time.time(), "deadline_epoch": until,
            "ownership_selftest": ownership_proof,
            "last_services_before": file_hash(meta) if meta.exists() else None}
    fixed_write(safe(folder, "plan.json"), plan)
    proc = None
    try:
        with safe(folder, "worker.log").open("x") as log:
            proc = subprocess.Popen(worker_command(output, folder), cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        fixed_write(safe(folder, "worker-process.json"), {"pid": proc.pid, "pgid": proc.pid,
                                                        "started_epoch": plan["started_epoch"],
                                                        "ownership": "direct child; separate session; Linux subreaper active"})
        while proc.poll() is None:
            if time.time() >= until:
                raise StudyError("Profile worker exceeded its absolute watchdog")
            session(output)
            try:
                proc.wait(timeout=min(.25, max(.001, until-time.time())))
            except subprocess.TimeoutExpired:
                pass
        exited = time.time()
        if exited > until or proc.returncode != 0:
            raise StudyError(f"Profile worker did not exit successfully on time (exit {proc.returncode}); see {folder}")
        session(output)
        completed = verify_attempt(output, folder, role, accepted=False)
        stop = recover_stop(output, role, plan, worker_pid=proc.pid, journal=safe(folder, "parent-cleanup.jsonl"))
        fixed_write(safe(folder, "parent-stop.json"), stop)
        fixed_write(safe(folder, "accepted.json"), {"role": role, "complete_sha256": file_hash(safe(folder, "complete.json")),
                                                    "parent_stop_sha256": file_hash(safe(folder, "parent-stop.json")),
                                                    "accepted_epoch": time.time(), "exited_epoch": exited, "deadline_epoch": until,
                                                    "plan_hash": fingerprint(plan), "returncode": proc.returncode})
        return completed
    except BaseException as exc:
        stop_error = None
        if proc is not None:
            try:
                stop_child(proc, plan["started_epoch"])
            except BaseException as stopping:
                stop_error = str(stopping)
        failure = {"at": utc_now(), "error": str(exc), "type": type(exc).__name__}
        if stop_error:
            failure["worker_stop_error"] = stop_error
        try:
            failure["parent_stop"] = recover_stop(output, role, plan, worker_pid=proc.pid if proc else None,
                                                   journal=safe(folder, "parent-cleanup.jsonl"))
        except BaseException as cleanup:
            failure["cleanup_error"] = str(cleanup)
        fixed_write(safe(folder, "failure.json"), failure)
        raise


def verify_attempt(output, folder, role, *, accepted=True, offline=False):
    settings, rows = context(output, offline=offline)
    plan = read_json(safe(folder, "plan.json"))
    complete = read_json(safe(folder, "complete.json"))
    if (plan["role"] != role or plan["settings_hash"] != fingerprint(settings)
            or complete["role"] != role or complete["count"] != len(rows)):
        raise StudyError("Completed profile attempt identity differs")
    from .ownership import verify_receipt
    verify_receipt(output, plan["ownership_selftest"], settings, plan["allocation"])
    names = {"plan.json", "tokens.json", "hardware.json", "resources.jsonl", "rows.jsonl", "deployment.json"}
    if set(complete.get("files", {})) != names or any(file_hash(safe(folder, name)) != digest
                                                    for name, digest in complete["files"].items()):
        raise StudyError("Profile attempt evidence changed")
    receipt = read_json(safe(folder, "tokens.json"))
    verify_tokens(output, role, rows, receipt)
    validate_rows(rows, read_rows(safe(folder, "rows.jsonl")), role)
    deployment_record = read_json(safe(folder, "deployment.json"))
    verify_engine(deployment_record, role)
    if receipt["runtime_hash"] != fingerprint(deployment_record["runtime"]):
        raise StudyError("Token report and actual deployment have different runtime identities")
    observed = read_rows(safe(folder, "rows.jsonl"))
    timing = {name: deployment_record.get(name) for name in
              ("startup_seconds", "collection_seconds", "startup_started_monotonic", "ready_monotonic",
               "collection_started_monotonic", "collection_finished_monotonic")}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in timing.values()):
        raise StudyError("Profile aggregate timing must be finite and nonnegative")
    if (not math.isclose(timing["ready_monotonic"]-timing["startup_started_monotonic"], timing["startup_seconds"], abs_tol=1e-8)
            or not math.isclose(timing["collection_finished_monotonic"]-timing["collection_started_monotonic"],
                                timing["collection_seconds"], abs_tol=1e-8)
            or timing["collection_seconds"]+1e-6 < sum(r["observed_request_seconds"] for r in observed)):
        raise StudyError("Profile aggregate timing differs from recorded monotonic observations")
    recomputed = {"median": statistics.median(r["latency_ms"] for r in observed),
                  "max": max(r["latency_ms"] for r in observed)}
    if deployment_record.get("latency_ms") != recomputed:
        raise StudyError("Profile latency aggregates differ from preserved request rows")
    stop = deployment_record.get("stop", {})
    if stop.get("gpu_compute_processes") != [] or stop.get("owned_groups") != sorted(p["pid"] for p in deployment_record["processes"]):
        raise StudyError("Completed role lacks verified service stop")
    if accepted:
        accept = read_json(safe(folder, "accepted.json"))
        parent_stop = read_json(safe(folder, "parent-stop.json"))
        if (accept.get("role") != role or accept.get("returncode") != 0 or accept.get("plan_hash") != fingerprint(plan)
                or accept.get("complete_sha256") != file_hash(safe(folder, "complete.json"))
                or accept.get("parent_stop_sha256") != file_hash(safe(folder, "parent-stop.json"))
                or not plan["started_epoch"] <= accept.get("exited_epoch", float("inf")) <= plan["deadline_epoch"]
                or accept.get("accepted_epoch", 0) < accept.get("exited_epoch", float("inf"))
                or parent_stop.get("gpu_compute_processes") != []):
            raise StudyError("Role is not accepted after an on-time exit and confirmed stop")
    return {"role": role, "attempt": str(Path(folder).relative_to(normal(output))),
            "count": len(rows), "startup_seconds": deployment_record["startup_seconds"],
            "collection_seconds": deployment_record["collection_seconds"], "latency_ms": deployment_record["latency_ms"],
            "token_report": receipt, "scope": PROTOCOL["scope"]}


def audit(output):
    """Read-only returned-evidence check; it does not re-inspect the node's model bytes."""
    context(output, offline=True)
    results = []
    for role in ROLES:
        accepted = [p for p in attempts(output, role) if safe(p, "accepted.json").exists()]
        if len(accepted) != 1:
            raise StudyError(f"Expected one accepted attempt for {role}")
        results.append(verify_attempt(output, accepted[0], role, offline=True))
    return {"status": "PASS", "roles": results, "scope": PROTOCOL["scope"],
            "limit": "Returned evidence only; node model bytes and live processes are not independently observed"}


def attempts(output, role):
    path = safe(output, "attempts/"+role)
    return sorted((p for p in path.iterdir() if p.is_dir()), key=lambda p: p.name) if path.exists() else []


def previous_attempt_stopped(output, folder, role, plan):
    """Do not signal PIDs from an older controller invocation: they may be reused."""
    old.assert_gpu_owned_by(set())
    worker_path = safe(folder, "worker-process.json")
    groups = {read_json(worker_path)["pgid"]} if worker_path.exists() else set()
    path = safe(control(output, role), "runtime/last_services.json")
    metadata = old._launch_record(path, plan["last_services_before"], plan["started_epoch"])
    if metadata:
        groups.update(int(p["pid"]) for p in metadata.get("processes", []))
    if any(old.group_alive(group) for group in groups):
        raise StudyError("Earlier attempt still has live process groups; refusing to signal historical PIDs")
    # Without a parent cleanup receipt an exited leader cannot prove separately
    # sessioned CPU probes ended. A new, verified-ended allocation is required.
    failure_path = safe(folder, "failure.json")
    stop = read_json(failure_path).get("parent_stop") if failure_path.exists() else None
    if worker_path.exists() and (not isinstance(stop, dict) or stop.get("gpu_compute_processes") != []):
        raise StudyError("Earlier worker has no verified parent cleanup; resume in a genuinely new allocation after it ends")


def run(output, *, retry_failed=False):
    from .ownership import require_receipt
    require_receipt(output)
    summaries = []
    for role in ROLES:
        existing = attempts(output, role)
        accepted = [p for p in existing if safe(p, "accepted.json").exists()]
        if len(accepted) > 1:
            raise StudyError("Multiple accepted attempts for one role")
        if accepted:
            summaries.append(verify_attempt(output, accepted[0], role))
            continue
        if existing and not retry_failed:
            raise StudyError(f"{role} has an interrupted or failed attempt; inspect evidence, then use --retry-failed")
        # Recover only a same-host current-job attempt; cross-job transitions were checked by initialize.
        for folder in existing:
            plan_path = safe(folder, "plan.json")
            if plan_path.exists():
                plan = read_json(plan_path)
                if plan["allocation"] == session(output)["allocation"]:
                    previous_attempt_stopped(output, folder, role, plan)
        run_attempt(output, role)
        summaries.append(verify_attempt(output, attempts(output, role)[-1], role))
    value = {"protocol": PROTOCOL, "roles": summaries, "completed_at": utc_now(), "new_evaluation_or_training": False}
    write_json(safe(output, "summary.json"), value)
    return value


def export(output):
    """Prune build/model caches before descent; preserve bounded diagnostic evidence."""
    output = normal(output)
    files = []
    for directory, dirs, names in os.walk(output, followlinks=False):
        if any((Path(directory)/name).is_symlink() for name in dirs):
            raise StudyError("Symlinked directory in profile evidence")
        dirs[:] = sorted(n for n in dirs if n not in {"runtime-work", "exports", "__pycache__"})
        for name in sorted(names):
            path = Path(directory)/name
            relative = str(path.relative_to(output))
            safe(output, relative)
            if name == ".active.lock":
                continue
            if path.stat().st_size > 64*1024*1024:
                raise StudyError("Profile diagnostic member exceeds 64 MiB; preserve it locally for review")
            files.append(path)
    manifest = {str(p.relative_to(output)): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in files}
    target = safe(output, f"exports/learnability_profile_bundle-{time.time_ns()}.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(output)))
        archive.writestr("BUNDLE_MANIFEST.json", canonical(manifest)+b"\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "start", "export", "audit", "worker"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--new-allocation", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    output = normal(args.output_dir)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {signum}")))
    if args.command == "worker":
        if args.attempt_dir is None or output not in normal(args.attempt_dir).parents:
            parser.error("worker requires an attempt folder inside the output")
        try:
            validate_output(output, existing=True)
        except Exception as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
        try:
            worker(output, normal(args.attempt_dir))
            return 0
        except BaseException as exc:
            write_json(safe(args.attempt_dir, "worker-error.json"), {"error": str(exc), "type": type(exc).__name__,
                                                                    "traceback": traceback.format_exc()})
            return 2
    try:
        validate_output(output, args.runtime_root, existing=args.command not in {"prepare", "start"})
        safe(output, ".active.lock")
        if args.command == "audit":
            print(json.dumps(audit(output), ensure_ascii=False, indent=2))
            return 0
        with exclusive_run(output):
            try:
                if args.command in {"prepare", "start"}:
                    if args.runtime_root is None:
                        parser.error("prepare/start require --runtime-root")
                    runtime = normal(args.runtime_root)
                    dirs = {"v1": runtime/"results/solo-v1", "v3": runtime/"results/solo-v3",
                            "size_control": runtime/"results/aux-size-control-v1-r2"}
                    prepare(dirs, output)
                if args.command in {"run", "start"}:
                    initialize(output, new_allocation=args.new_allocation)
                    from .ownership import run_selftest
                    run_selftest(output)
                    result = run(output, retry_failed=args.retry_failed)
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                if args.command != "prepare":
                    print(f"PROFILE BUNDLE: {export(output)}")
                return 0
            except BaseException as exc:
                write_json(safe(output, f"diagnostics/failure-{time.time_ns()}.json"),
                           {"at": utc_now(), "error": str(exc), "type": type(exc).__name__})
                try:
                    print(f"DIAGNOSTIC BUNDLE: {export(output)}")
                except Exception as exported:
                    print(f"Diagnostic export failed: {exported}", file=sys.stderr)
                raise
    except BaseException as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
