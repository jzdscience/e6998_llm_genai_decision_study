"""Independent service-study control; original environments/checkpoints read-only."""
from contextlib import contextmanager
import copy
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from .. import runtime as frozen
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..v3 import runtime as shared
from ..runtime_compat import legacy_environment
from .identity import PROJECT, fixed_write, protocol, safe


def plans(v1_dir, output_dir):
    source, lock, installed, _, _, env_name, source_work, hashes = shared._source(v1_dir)
    output = Path(output_dir).absolute()
    if (source == output or source in output.parents or output in source.parents
            or output == PROJECT or PROJECT in output.parents or output in PROJECT.parents):
        raise StudyError("Use independent non-overlapping service-study output")
    if any(safe(output, name).exists() for name in ("runtime-controls", "runtime-control/binding.json", "data/manifest.json")):
        raise StudyError("Output contains another experiment; choose a new independent service directory")
    p = protocol()
    plan = {"lock.json": lock,
            "installations.json": {"lock_hash": lock["lock_hash"],
                "environments": {env_name: copy.deepcopy(installed["environments"][env_name])},
                "models": {m: copy.deepcopy(installed["models"][m]) for m in ("startlux", "qwen")}},
            "selected_profiles.json": {m: "primary" for m in ("startlux", "qwen")},
            "selected_memory.json": {m: p["gpu_memory_utilization"] for m in ("startlux", "qwen")}}
    binding = {"protocol": p["protocol"], "v1_dir": str(source), "source_work_dir": str(source_work),
               "work_dir": str(safe(output, "runtime-work")), "source_files": hashes,
               "controls": {name: fingerprint(value) for name, value in plan.items()}}
    return binding, plan


def initialize(v1_dir, output_dir, hours_left, *, new_allocation=False):
    if isinstance(hours_left, bool) or not isinstance(hours_left, (float, int)) or not math.isfinite(hours_left) or not 0 < hours_left <= 24:
        raise StudyError("Actual remaining hours must be finite and in (0,24]")
    binding, plan = plans(v1_dir, output_dir)
    allocation = shared._allocation()
    if not allocation["slurm_job_id"]:
        raise StudyError("Initialize in the actual company Slurm allocation")
    existing = safe(output_dir, "runtime/binding.json")
    if existing.exists() and read_json(existing) != binding:
        raise StudyError("Runtime source binding changed")
    session_path = safe(output_dir, "runtime/session.json")
    previous = read_json(session_path) if session_path.exists() else None
    if previous and previous["allocation"] == allocation:
        if new_allocation:
            raise StudyError("Cannot renew the same allocation")
        session = shared._session(output_dir)
        # Never extend a deadline from another operator estimate.
        session = {**session, "deadline_epoch": min(session["deadline_epoch"], time.time()+3600*hours_left)}
    else:
        if previous and (not new_allocation or previous["allocation"]["slurm_job_id"] == allocation["slurm_job_id"]):
            raise StudyError("Renewal requires an explicitly acknowledged different Slurm job")
        session = {"started_at": utc_now(), "deadline_epoch": time.time()+3600*hours_left,
                   "hours_left_at_initialization": hours_left, "allocation": allocation}
    control = safe(output_dir, "runtime-control")
    # Preflight all existing files before creating anything.
    for name, value in plan.items():
        path = safe(control, "runtime/"+name)
        if path.exists() and read_json(path) != value:
            raise StudyError(f"Changed runtime control: {name}")
    fixed_write(existing, binding)
    for name, value in plan.items():
        fixed_write(safe(control, "runtime/"+name), value)
    doctor = safe(control, "runtime/doctor.json")
    if not doctor.exists():
        fixed_write(doctor, read_json(safe(v1_dir, "runtime/doctor.json")))
    if previous and previous != session:
        fixed_write(safe(output_dir, f"runtime/session-history/{fingerprint(previous)}.json"), previous)
    write_json(session_path, session)
    write_json(safe(control, "runtime/session.json"), session)
    return {"binding": binding, "session": session, "notice": "No GPU start or model download"}


def validate(output_dir):
    binding = read_json(safe(output_dir, "runtime/binding.json"))
    expected, plan = plans(binding["v1_dir"], output_dir)
    if binding != expected:
        raise StudyError("Runtime source identity changed")
    control = safe(output_dir, "runtime-control")
    for name, value in plan.items():
        if read_json(safe(control, "runtime/"+name)) != value:
            raise StudyError(f"Runtime control changed: {name}")
    session = shared._session(output_dir)
    return control, binding, session, plan["installations.json"]


def check_time(output_dir, required_seconds=0):
    session = shared._session(output_dir)
    if session["deadline_epoch"]-time.time() < required_seconds:
        raise StudyError(f"Insufficient allocation allowance for next unit ({required_seconds:.0f}s needed). Export and resume in a new allocation")
    return session


def stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def idle_device(output_dir):
    """Require the fixed device and no pre-existing compute clients; never kill them."""
    control, _, _, _ = validate(output_dir)
    doctor = frozen.doctor(control, require_gpu=True)
    gpus = doctor.get("gpu", [])
    if len(gpus) != 1 or gpus[0]["name"] != "NVIDIA L40S" or gpus[0]["memory_mb"] < 44000:
        raise StudyError("Service protocol requires one L40S with >=44000MiB")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=15, check=True)
    if result.stdout.strip():
        raise StudyError("GPU has existing compute clients. Stop the other experiment before this study; no unowned processes were killed")
    return doctor


@contextmanager
def gpu_service(output_dir, condition, examples):
    model = {"startlux4b": "startlux", "qwen4b": "qwen"}[condition]
    control, binding, session, installed = validate(output_dir)
    write_json(safe(control, "runtime/session.json"), session)
    with legacy_environment(binding, PROJECT), shared._environment(binding, session):
        doctor = frozen.doctor(control, require_gpu=True)
        if len(doctor.get("gpu", [])) != 1 or doctor["gpu"][0]["name"] != "NVIDIA L40S" or doctor["gpu"][0]["memory_mb"] < 44000:
            raise StudyError("Service protocol requires one L40S with >=44000MiB")
        os.environ.update(shared._cuda_preflight(control, binding, installed))
        tokens = frozen.token_preflight_examples(PROJECT, control, model, examples)
        with frozen.managed_services(PROJECT, control, [model]) as endpoints:
            metadata = read_json(safe(control, "runtime/last_services.json"))
            yield {"endpoint": endpoints[model], "model": model, "tokens": tokens,
                   "runtime": frozen.runtime_config(control, model), "services": metadata,
                   "pids": [p["pid"] for p in metadata["processes"]]}


@contextmanager
def cpu_service(output_dir, condition, *, python, bundle_dir, model_dir):
    p = protocol()
    session = check_time(output_dir, 120)
    if not Path(python).is_file():
        raise StudyError("Missing separate CPU baseline Python")
    frozen._check_port(p["cpu_port"])
    log_path = safe(output_dir, f"runtime/logs/{condition}-{time.time_ns()}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    command = [str(python), "-m", "decision_study.service_study.cpu", "--bundle-dir", str(bundle_dir),
               "--model-dir", str(model_dir), "--recipe", condition, "--port", str(p["cpu_port"]),
               "--queue-capacity", str(p["cpu_queue_capacity"])]
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    endpoint = f"http://127.0.0.1:{p['cpu_port']}"
    try:
        until = min(time.time()+120, session["deadline_epoch"])
        health = None
        while time.time() < until:
            if process.poll() is not None:
                raise StudyError(f"CPU service exited; see {log_path}")
            try:
                import json
                with urllib.request.urlopen(endpoint+"/health", timeout=2) as response:
                    health = json.load(response)
                break
            except (OSError, ValueError):
                time.sleep(.25)
        if health is None:
            raise StudyError(f"CPU startup timeout; see {log_path}")
        from .cpu import ACCEPTED_MODELS, ACCEPTED_RECEIPTS, THREAD_ENV
        identity = health.get("identity", {})
        environment = health.get("environment", {})
        if (health.get("status") != "ok" or health.get("service") != "frozen_minilm_cpu"
                or health.get("pid") != process.pid or not health.get("instance_id")
                or identity.get("recipe") != condition
                or identity.get("classifier_sha256") != ACCEPTED_MODELS.get(condition)
                or identity.get("baseline_receipts") != ACCEPTED_RECEIPTS
                or identity.get("encoder", {}).get("provider") != "CPUExecutionProvider"
                or health.get("workers") != 1 or health.get("batch_size") != 1
                or health.get("numeric_threads") != 1 or health.get("prediction_cache") is not False
                or health.get("queue_capacity") != p["cpu_queue_capacity"]
                or health.get("maximum_admitted") != p["cpu_queue_capacity"] + 1
                or health.get("maximum_http_handlers") != p["cpu_queue_capacity"] + 8
                or environment.get("thread_environment") != THREAD_ENV
                or any(pool.get("num_threads") != 1 for pool in environment.get("threadpools", []))):
            raise StudyError("Wrong CPU service identity or resident single-worker/thread contract")
        yield {"endpoint": endpoint, "model": condition, "health": health,
               "pids": [process.pid], "command": command, "log": str(log_path)}
    finally:
        stop(process)
