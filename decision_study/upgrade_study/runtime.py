"""Runtime control for the upgrade study, including the two-service cascade.

The frozen V1 launcher (``decision_study.runtime.managed_services``) is used once
per role, each with its own runtime-control root, inside one outer ExitStack.
This controller therefore carries the checks the V1 paired launcher would have
made: a single L40S, an idle GPU before the first start, only this deployment's
process groups on the GPU before the second start, cleanup of every owned group
on any failure, and a confirmed stop after each deployment. Existing V1, V3 and
size-control environments, locks and installation records are read, never written.
"""
from contextlib import ExitStack, contextmanager
import copy
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

from .. import runtime as frozen
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..runtime_compat import legacy_environment
from ..size_control import runtime as size_runtime
from ..v3 import runtime as v3_runtime
from .identity import INTERFACE, PROJECT, PROTOCOL_ID, ROLES, SERVICES, fixed_write, protocol, safe

CONTROL_FILES = ("lock.json", "installations.json", "selected_profiles.json", "selected_memory.json")
LOG_PATTERNS = {
    "model_loading_gib": r"Model loading took ([0-9.]+) GiB",
    "available_kv_cache_gib": r"Available KV cache memory: ([0-9.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([0-9,]+) tokens",
    "max_concurrency": r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
}
FAILURE_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "EngineDeadError", "No available memory for the cache blocks")


def _within(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    return path == root or root in path.parents


def _old_role_lock(role, v1_dir, other_dir):
    """The exact runtime lock V3 or size control froze for this 9B role."""
    if role == "qwen9b":
        _, plans = v3_runtime._plans(v1_dir, other_dir)
    elif role == "startlux9b":
        _, plans = size_runtime._plans(v1_dir, other_dir)
    else:
        raise StudyError(f"No earlier 9B lock for {role}")
    lock = plans[role]["lock.json"]
    stored = read_json(safe(other_dir, f"runtime-controls/{role}/runtime/lock.json"))
    if stored != lock:
        raise StudyError(f"The earlier {role} runtime lock no longer matches its frozen plan")
    return copy.deepcopy(lock)


def _snapshot(role, other_dir, *, check_exists=True):
    installed = read_json(safe(other_dir, f"runtime-controls/{role}/runtime/installations.json"))
    record = installed.get("models", {}).get(INTERFACE[role])
    revision = protocol()["models"][role]["revision"]
    path = Path((record or {}).get("model_path", ""))
    work = Path(read_json(safe(other_dir, "runtime/binding.json"))["work_dir"])
    if (not record or record.get("revision") != revision or not path.is_absolute()
            or path.name != revision or not _within(path, work)):
        raise StudyError(f"Missing or changed {role} snapshot record")
    if check_exists:
        if not path.is_dir():
            raise StudyError(f"Recorded {role} snapshot is missing; nothing is downloaded or substituted")
        if role == "startlux9b":
            size_runtime._verify_model_metadata(path)
    return copy.deepcopy(record)


def plans(v1_dir, v3_dir, size_control_dir, output_dir, *, check_snapshots=True):
    source, lock, installed, _, _, env_name, source_work, hashes = v3_runtime._source(v1_dir)
    output = Path(output_dir).absolute()
    for root in (source, Path(v3_dir).absolute(), Path(size_control_dir).absolute(), PROJECT):
        if output == root or _within(output, root) or _within(root, output):
            raise StudyError("Use an independent output directory that overlaps no input or source root")
    p = protocol()
    locks = {"startlux4b": copy.deepcopy(lock),
             "startlux9b": _old_role_lock("startlux9b", v1_dir, size_control_dir),
             "qwen9b": _old_role_lock("qwen9b", v1_dir, v3_dir)}
    records = {"startlux4b": copy.deepcopy(installed["models"]["startlux"]),
               "startlux9b": _snapshot("startlux9b", size_control_dir, check_exists=check_snapshots),
               "qwen9b": _snapshot("qwen9b", v3_dir, check_exists=check_snapshots)}
    result, ports = {}, {}
    for role in ROLES:
        model, spec = INTERFACE[role], p["models"][role]
        entry = locks[role]["models"][model]
        if entry["repo"] != spec["repo"] or entry["revision"] != spec["revision"] or records[role]["revision"] != spec["revision"]:
            raise StudyError(f"{role} checkpoint differs from the protocol")
        result[role] = {"lock.json": locks[role],
                        "installations.json": {"lock_hash": locks[role]["lock_hash"],
                                               "environments": {env_name: copy.deepcopy(installed["environments"][env_name])},
                                               "models": {model: records[role]}},
                        "selected_profiles.json": {model: "primary"},
                        "selected_memory.json": {model: float(spec["gpu_memory_utilization"])}}
        ports[role] = sorted({entry["port"], *([entry["engine_port"]] if model == "startlux" else [])})
    for condition, roles in SERVICES.items():
        used = [port for role in roles for port in ports[role]]
        if len(used) != len(set(used)):
            raise StudyError(f"Co-resident roles share a port in {condition}")
    binding = {"schema_version": 1, "protocol": PROTOCOL_ID, "v1_dir": str(source), "v3_dir": str(Path(v3_dir).absolute()),
               "size_control_dir": str(Path(size_control_dir).absolute()), "source_work_dir": str(source_work),
               "work_dir": str(safe(output, "runtime-work")), "source_files": hashes, "roles": list(ROLES), "ports": ports,
               "control_identity": {role: {name: fingerprint(value) for name, value in plan.items()} for role, plan in result.items()}}
    return binding, result


def control_root(output_dir, role):
    if role not in ROLES:
        raise StudyError(f"Unknown upgrade-study role: {role}")
    return safe(output_dir, f"runtime-controls/{role}")


def initialize(v1_dir, v3_dir, size_control_dir, output_dir, hours_left, *, new_allocation=False):
    if isinstance(hours_left, bool) or not isinstance(hours_left, (float, int)) or not math.isfinite(hours_left) or not 0 < hours_left <= 24:
        raise StudyError("Actual remaining hours must be finite and in (0,24]")
    binding, role_plans = plans(v1_dir, v3_dir, size_control_dir, output_dir)
    allocation = v3_runtime._allocation()
    if not allocation["slurm_job_id"]:
        raise StudyError("Initialize inside the actual Slurm allocation")
    binding_path = safe(output_dir, "runtime/binding.json")
    if binding_path.exists() and read_json(binding_path) != binding:
        raise StudyError("Runtime binding changed; preserve this output and use a new one")
    session_path = safe(output_dir, "runtime/session.json")
    previous = read_json(session_path) if session_path.exists() else None
    if previous and previous["allocation"] == allocation:
        if new_allocation:
            raise StudyError("Cannot renew the same allocation")
        session = v3_runtime._session(output_dir)
        session = {**session, "deadline_epoch": min(session["deadline_epoch"], time.time() + 3600 * hours_left)}
    else:
        if previous and (not new_allocation or previous["allocation"]["slurm_job_id"] == allocation["slurm_job_id"]):
            raise StudyError("Renewal requires an explicitly acknowledged different Slurm job (--new-allocation)")
        session = {"started_at": utc_now(), "deadline_epoch": time.time() + 3600 * hours_left,
                   "hours_left_at_initialization": hours_left, "allocation": allocation}
    for role, plan in role_plans.items():
        for name, value in plan.items():
            path = safe(control_root(output_dir, role), "runtime/" + name)
            if path.exists() and read_json(path) != value:
                raise StudyError(f"Changed runtime control: {role}/{name}")
    fixed_write(binding_path, binding)
    for role, plan in role_plans.items():
        control = control_root(output_dir, role)
        for name, value in plan.items():
            fixed_write(safe(control, "runtime/" + name), value)
        doctor = safe(control, "runtime/doctor.json")
        if not doctor.exists():
            fixed_write(doctor, read_json(safe(v1_dir, "runtime/doctor.json")))
        write_json(safe(control, "runtime/session.json"), session)
    if previous and previous != session:
        fixed_write(safe(output_dir, f"runtime/session-history/{fingerprint(previous)}.json"), previous)
    write_json(session_path, session)
    return {"binding": binding, "session": session, "notice": "Controls bound; no GPU start or download"}


def validate(output_dir):
    binding = read_json(safe(output_dir, "runtime/binding.json"))
    expected, role_plans = plans(binding["v1_dir"], binding["v3_dir"], binding["size_control_dir"], output_dir)
    if binding != expected:
        raise StudyError("Runtime source identity changed since initialization")
    controls = {}
    for role, plan in role_plans.items():
        control = control_root(output_dir, role)
        for name, value in plan.items():
            if read_json(safe(control, "runtime/" + name)) != value:
                raise StudyError(f"Runtime control changed: {role}/{name}")
        controls[role] = control
    return controls, binding, v3_runtime._session(output_dir), role_plans


def check_time(output_dir, required_seconds=0):
    session = v3_runtime._session(output_dir)
    if session["deadline_epoch"] - time.time() < required_seconds:
        raise StudyError(f"Insufficient allocation allowance for the next unit ({required_seconds:.0f}s). Export and resume in a new allocation")
    return session


def gpu_processes():
    command = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StudyError("Cannot list GPU compute processes") from exc
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            raise StudyError(f"Unexpected nvidia-smi compute-process line: {line!r}")
        rows.append({"pid": int(parts[0]), "used_memory_mib": float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else None})
    return rows


def process_group(pid):
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def assert_gpu_owned_by(owned_groups, *, lister=None):
    """Allow only processes in this deployment's own process groups; never stop others."""
    rows = (lister or gpu_processes)()
    foreign = [row["pid"] for row in rows if process_group(row["pid"]) not in owned_groups]
    if foreign:
        raise StudyError(f"GPU has compute processes outside this deployment {foreign}; stop their owner normally. No process was stopped.")
    return rows


def group_alive(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def verify_stopped(groups, *, timeout=60, lister=None, poll=0.5):
    """Confirm owned groups have exited and the GPU has no compute processes."""
    started = time.monotonic()
    while True:
        alive = [g for g in groups if group_alive(g)]
        rows = (lister or gpu_processes)() if not alive else None
        if not alive and not rows:
            return {"at": utc_now(), "owned_groups": sorted(groups), "seconds": time.monotonic() - started, "gpu_compute_processes": []}
        if time.monotonic() - started >= timeout:
            for group in alive:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise StudyError(f"Owned services did not stop within {timeout}s (alive groups {alive}, GPU processes {rows}); "
                             "only this deployment's own groups were signalled")
        time.sleep(poll)


def engine_log_facts(processes):
    facts = {}
    for process in processes:
        path = Path(process.get("log", ""))
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        values = {}
        for key, pattern in LOG_PATTERNS.items():
            found = re.findall(pattern, text)
            if found:
                values[key] = [float(v.replace(",", "")) for v in found]
        values["preemption_mentions"] = len(re.findall(r"preempt", text, flags=re.IGNORECASE))
        values["failure_mentions"] = {p: text.count(p) for p in FAILURE_PATTERNS if p in text}
        facts[process["name"]] = values
    return facts


def health(endpoint, path="/health", timeout=10):
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=timeout) as response:
            return {"status": response.status, "ok": 200 <= response.status < 300}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


@contextmanager
def deployment(output_dir, condition, *, launcher=None, lister=None, preflight=True, sink=None):
    """Start this condition's services in order and always stop every owned group.

    ``launcher`` and ``lister`` exist for tests; production uses the frozen V1
    single-model launcher and nvidia-smi.
    """
    if condition not in SERVICES:
        raise StudyError(f"Unknown condition: {condition}")
    controls, binding, session, _ = validate(output_dir)
    roles = SERVICES[condition]
    launch = launcher or (lambda control, model: frozen.managed_services(PROJECT, control, [model]))
    record = sink if sink is not None else {}
    record.update(condition=condition, started_at=utc_now(), allocation=session["allocation"], roles=[])
    owned = set()
    with legacy_environment(binding, PROJECT), v3_runtime._environment(binding, session):
        if preflight:
            report = frozen.doctor(controls[roles[0]], require_gpu=True)
            gpus = report.get("gpu", [])
            if len(gpus) != 1 or gpus[0]["name"] != "NVIDIA L40S" or gpus[0]["memory_mb"] < 44000:
                raise StudyError("This study requires one L40S with at least 44,000 MiB")
            installed = read_json(safe(controls[roles[0]], "runtime/installations.json"))
            os.environ.update(v3_runtime._cuda_preflight(controls[roles[0]], binding, installed))
        failure = None
        try:
            with ExitStack() as stack:
                for role in roles:
                    record.setdefault("gpu_before_start", []).append(
                        {"role": role, "processes": assert_gpu_owned_by(owned, lister=lister)})
                    control, model = controls[role], INTERFACE[role]
                    write_json(safe(control, "runtime/session.json"), session)
                    began = time.perf_counter()
                    endpoints = stack.enter_context(launch(control, model))
                    meta = read_json(safe(control, "runtime/last_services.json"))
                    groups = {process["pid"] for process in meta["processes"]}
                    owned |= groups
                    record["roles"].append({
                        "role": role, "model": model, "endpoint": endpoints[model],
                        "startup_seconds": time.perf_counter() - began, "processes": meta["processes"],
                        "health": meta.get("models", {}).get(model, {}).get("health"),
                        "runtime": meta.get("models", {}).get(model, {}).get("config"),
                        "memory_fraction": read_json(safe(control, "runtime/selected_memory.json"))[model],
                        "engine_log": engine_log_facts(meta["processes"])})
                record["gpu_after_start"] = assert_gpu_owned_by(owned, lister=lister)
                record["ready_at"] = utc_now()
                yield {"condition": condition, "endpoints": {r["role"]: r["endpoint"] for r in record["roles"]},
                       "pids": sorted(owned), "record": record}
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # The ExitStack has already run every launcher's own SIGTERM/SIGKILL cleanup.
            if owned:
                try:
                    record["stop_verification"] = verify_stopped(owned, lister=lister)
                except StudyError as stop_error:
                    record["stop_verification"] = {"error": str(stop_error)}
                    if failure is None:
                        raise


def token_check(output_dir, examples):
    """Exact tokenizer lengths for every role on all formal and development inputs; no truncation."""
    controls, binding, session, _ = validate(output_dir)
    reports = {}
    with legacy_environment(binding, PROJECT), v3_runtime._environment(binding, session):
        for role in ROLES:
            report = frozen.token_preflight_examples(PROJECT, controls[role], INTERFACE[role], examples)
            reports[role] = {"status": report.get("status"),
                             "summary": {k: v for k, v in report.items() if not isinstance(v, (list, dict))}}
    return reports
