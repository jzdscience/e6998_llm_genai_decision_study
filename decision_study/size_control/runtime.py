"""Isolated sequential size-control runtimes using the frozen v1 HTTP/engine interfaces.

Old environments, 4B snapshots and the verified CUDA toolkit are read-only
dependencies. New metadata, downloaded 9B weights and build caches live under size-control.
Importing this module does not inspect a GPU, download a model or renew a session.
"""
from contextlib import contextmanager
import copy
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import warnings

from .. import runtime as frozen
from ..budget import Budget
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..runtime_compat import legacy_environment


PROJECT = Path(__file__).resolve().parents[2]
ROLES = {"startlux9b": "startlux"}
STARTLUX9B_REPO = "startlux-models/StartLux-Decision-v1.1-9B-Instant"
STARTLUX9B_REVISION = "18a17f734966b213b2f00fa9c915e8b9af2a9b20"
TOOLKIT = "cuda-13.0-flashinfer-curand10.4.0.35"
ENV_KEYS = {"CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC", "PATH", "MAX_JOBS",
            "FLASHINFER_NVCC_THREADS", "FLASHINFER_EXTRA_LDFLAGS"}
SOURCE_FILES = ("lock.json", "installations.json", "selected_profiles.json", "selected_memory.json")


def backend_model(role):
    try:
        return ROLES[role]
    except KeyError:
        raise StudyError(f"Unknown size-control role: {role}") from None


def _safe(root, relative):
    root = Path(root).absolute()
    if root.is_symlink():
        raise StudyError(f"Refusing symlinked runtime directory: {root}")
    path = root
    for part in Path(relative).parts:
        if part in {"..", "."} or Path(part).is_absolute():
            raise StudyError("Unsafe runtime path")
        path /= part
        if path.is_symlink():
            raise StudyError(f"Refusing symlinked runtime path: {path}")
    return path


def control_root(output_dir, role):
    backend_model(role)
    return _safe(output_dir, f"runtime-controls/{role}")


def _allocation():
    return {"hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")}


def _source(v1_dir):
    source = Path(v1_dir).absolute()
    lock = read_json(_safe(source, "runtime/lock.json"))
    defaults = read_json(PROJECT / "configs/runtime_defaults.json")
    if (lock.get("defaults_hash") != fingerprint(defaults)
            or {k: lock.get(k) for k in defaults} != defaults
            or lock.get("lock_hash") != fingerprint(defaults)):
        raise StudyError("The source runtime is not the unchanged pinned v1 configuration")
    installations = read_json(_safe(source, "runtime/installations.json"))
    if installations.get("lock_hash") != lock["lock_hash"]:
        raise StudyError("Source runtime installation lock mismatch")
    profiles_path = _safe(source, "runtime/selected_profiles.json")
    memory_path = _safe(source, "runtime/selected_memory.json")
    profiles = read_json(profiles_path) if profiles_path.exists() else {}
    memory = read_json(memory_path) if memory_path.exists() else {}
    if any(profiles.get(model, "primary") != "primary" for model in ("startlux", "qwen")):
        raise StudyError("size-control reuses the verified primary vLLM environment only")
    env_name = "vllm-" + lock["packages"]["vllm_primary"]
    if env_name not in installations.get("environments", {}):
        raise StudyError("Source vLLM environment metadata is missing")
    env = installations["environments"][env_name]
    python = Path(env["python"])
    if not python.is_absolute():
        raise StudyError("Source environment Python must be an absolute path")
    # Derive the shared work location from the recorded installed environment,
    # not a possibly inherited DECISION_STUDY_RUNTIME_DIR variable.
    if python.parts[-4:] != ("envs", env_name, "bin", "python"):
        raise StudyError("Unexpected source environment layout")
    source_work = python.parents[3]
    hashes = {name: file_hash(_safe(source, "runtime/" + name))
              for name in SOURCE_FILES if _safe(source, "runtime/" + name).is_file()}
    for model in ("startlux", "qwen"):
        record = installations.get("models", {}).get(model)
        if (not record or record.get("revision") != lock["models"][model]["revision"]
                or not Path(record.get("model_path", "")).is_absolute()):
            raise StudyError(f"Missing or mismatched existing {model} checkpoint metadata")
    return source, lock, installations, profiles, memory, env_name, source_work, hashes


def _plans(v1_dir, output_dir):
    source, lock, installed, profiles, memory, env_name, source_work, hashes = _source(v1_dir)
    output = Path(output_dir).absolute()
    if output == source or source in output.parents or output in source.parents:
        raise StudyError("size-control and v1 outputs must be separate, non-overlapping directories")
    if _safe(output, "runtime-control/binding.json").exists():
        raise StudyError("This is a v2 output directory; select a separate size-control output")
    binding = {"schema_version": 1, "protocol": "size-control-anli-v1", "v1_dir": str(source),
               "source_work_dir": str(source_work), "work_dir": str(_safe(output, "runtime-work")),
               "source_files": hashes, "roles": list(ROLES)}
    plans = {}
    for role, model in ROLES.items():
        role_lock = copy.deepcopy(lock)
        if role == "startlux9b":
            role_lock["models"]["startlux"].update(repo=STARTLUX9B_REPO, revision=STARTLUX9B_REVISION, native_temperature=1.1014)
            role_defaults = {key: role_lock[key] for key in read_json(PROJECT / "configs/runtime_defaults.json")}
            role_lock.update(defaults_hash=fingerprint(role_defaults), lock_hash=fingerprint(role_defaults))
            role_lock["verified_sources"] = [entry for entry in role_lock.get("verified_sources", [])
                                              if entry.get("repo") != lock["models"]["startlux"]["repo"]]
            role_lock["size-control_source"] = {"role": role, "parent_lock_hash": lock["lock_hash"],
                                     "revision_verification": "Pinned revision; snapshot resolution recorded by explicit download"}
        role_installed = {"lock_hash": role_lock["lock_hash"],
                          "environments": {env_name: copy.deepcopy(installed["environments"][env_name])},
                          "models": {} if role == "startlux9b" else {model: copy.deepcopy(installed["models"][model])}}
        selected_memory = {model: .8 if role == "startlux9b" else float(memory.get(model, lock["gpu_memory_utilization"]))}
        plans[role] = {"lock.json": role_lock, "installations.json": role_installed,
                       "selected_profiles.json": {model: "primary"}, "selected_memory.json": selected_memory}
    binding["control_identity"] = {
        role: {"files": {name: fingerprint(value) for name, value in plan.items() if name != "installations.json"},
               "installation_environment": fingerprint({key: value for key, value in plan["installations.json"].items() if key != "models"}),
               "original_models": fingerprint(plan["installations.json"]["models"]) if role != "startlux9b" else None}
        for role, plan in plans.items()
    }
    return binding, plans


def _session(output_dir, *, require_current=True):
    path = _safe(output_dir, "runtime/session.json")
    if not path.is_file():
        raise StudyError("Initialize size-control with the actual remaining allocation time first")
    session = read_json(path)
    deadline = session.get("deadline_epoch")
    if not isinstance(deadline, (float, int)) or not math.isfinite(deadline) or deadline <= time.time():
        raise StudyError("Recorded size-control allocation is expired; obtain and explicitly initialize a new allocation")
    if require_current and session.get("allocation") != _allocation():
        raise StudyError("Current allocation/host differs from size-control; explicitly initialize the new allocation")
    return session


def initialize(v1_dir, output_dir, *, session_hours_left=None, session_source=None,
               confirm_same_allocation=False, new_allocation=False):
    """Bind existing environments to size-control without starting or downloading anything."""
    if session_source is not None or confirm_same_allocation:
        raise StudyError("v2 sessions lack allocation identity; supply actual remaining hours for size-control")
    output = Path(output_dir).absolute()
    binding, plans = _plans(v1_dir, output)
    binding_path = _safe(output, "runtime/binding.json")
    if binding_path.exists() and read_json(binding_path) != binding:
        raise StudyError("size-control source binding changed; preserve this run and choose a new output")
    if binding_path.exists():
        for role in ROLES:
            _offline_control(output, role)
    session_path = _safe(output, "runtime/session.json")
    prior = read_json(session_path) if session_path.exists() else None
    allocation = _allocation()
    if not str(allocation.get("slurm_job_id") or "").isdigit():
        raise StudyError("A numeric current SLURM_JOB_ID is required for this company-node study")
    if prior is not None and prior.get("allocation") == allocation:
        session = _session(output)
        if new_allocation:
            raise StudyError("Cannot renew the same allocation by setting new_allocation")
    else:
        if prior is not None:
            old_id = prior.get("allocation", {}).get("slurm_job_id")
            if not new_allocation or not allocation["slurm_job_id"] or allocation["slurm_job_id"] == old_id:
                raise StudyError("Renewal requires a different Slurm job ID and new_allocation=True")
        if (isinstance(session_hours_left, bool) or not isinstance(session_hours_left, (float, int))
                or not math.isfinite(session_hours_left) or not 0 < session_hours_left <= 24):
            raise StudyError("Supply actual session_hours_left, finite and in (0, 24]")
        now = time.time()
        session = {"started_at": utc_now(), "deadline_epoch": now + 3600 * session_hours_left,
                   "hours_left_at_initialization": float(session_hours_left), "allocation": allocation,
                   "notice": "Operator or scheduler supplied remaining allowance; not an automatic provider renewal"}
    # Preflight every existing control before changing any metadata.
    for role, plan in plans.items():
        control = control_root(output, role)
        for name, expected in plan.items():
            path = _safe(control, "runtime/" + name)
            if path.exists():
                actual = read_json(path)
                if name == "installations.json" and role == "startlux9b":
                    actual = {**actual, "models": {}}
                if actual != expected:
                    raise StudyError(f"Existing size-control control changed: {role}/{name}")
    for role, plan in plans.items():
        control = control_root(output, role)
        for name, expected in plan.items():
            path = _safe(control, "runtime/" + name)
            if not path.exists():
                write_json(path, expected)
        doctor = _safe(control, "runtime/doctor.json")
        if not doctor.exists():
            write_json(doctor, read_json(_safe(Path(binding["v1_dir"]), "runtime/doctor.json")))
        write_json(_safe(control, "runtime/session.json"), session)
    write_json(binding_path, binding)
    if prior != session:
        write_json(session_path, session)
    return {"runtime_root": str(output / "runtime-controls"), "roles": list(ROLES),
            "session": session, "binding": binding,
            "notice": "Existing environment reused read-only; StartLux9B remains separate; no model or GPU operation performed"}


def _offline_control(output_dir, role):
    """Check copied evidence without touching the original machine or session."""
    output = Path(output_dir).absolute()
    control = control_root(output, role)
    binding_path = _safe(output, "runtime/binding.json")
    if not binding_path.is_file():
        raise StudyError("Initialize the independent size-control runtime before collecting predictions")
    binding = read_json(binding_path)
    if binding.get("protocol") != "size-control-anli-v1" or role not in binding.get("control_identity", {}):
        raise StudyError("Missing frozen size-control control identity")
    identity = binding["control_identity"][role]
    for name in ("lock.json", "selected_profiles.json", "selected_memory.json"):
        if fingerprint(read_json(_safe(control, "runtime/" + name))) != identity["files"].get(name):
            raise StudyError(f"size-control runtime identity changed: {role}/{name}")
    installed = read_json(_safe(control, "runtime/installations.json"))
    if fingerprint({key: value for key, value in installed.items() if key != "models"}) != identity["installation_environment"]:
        raise StudyError("size-control installed environment identity changed")
    if role != "startlux9b" and fingerprint(installed["models"]) != identity["original_models"]:
        raise StudyError("Existing 4B checkpoint binding changed")
    lock = read_json(_safe(control, "runtime/lock.json"))
    model = backend_model(role)
    defaults = read_json(PROJECT / "configs/runtime_defaults.json")
    expected_repo = STARTLUX9B_REPO if role == "startlux9b" else defaults["models"][model]["repo"]
    expected_revision = STARTLUX9B_REVISION if role == "startlux9b" else defaults["models"][model]["revision"]
    if lock["models"][model]["repo"] != expected_repo or lock["models"][model]["revision"] != expected_revision:
        raise StudyError("Role checkpoint differs from the registered model/revision")
    if role == "startlux9b" and installed["models"]:
        record = installed["models"].get("startlux", {})
        path = Path(record.get("model_path", ""))
        if (set(installed["models"]) != {"startlux"} or record.get("revision") != STARTLUX9B_REVISION
                or not path.is_absolute() or path.name != STARTLUX9B_REVISION
                or Path(binding["work_dir"]) not in path.parents):
            raise StudyError("Downloaded 9B checkpoint binding changed")
    return control, binding, installed


def _validated(output_dir, role, control=None):
    output = Path(output_dir).absolute()
    expected_control, binding, installed = _offline_control(output, role)
    if control is not None and Path(control).absolute() != expected_control:
        raise StudyError("Control directory does not belong to the requested size-control role")
    expected_binding, plans = _plans(binding["v1_dir"], output)
    if binding != expected_binding:
        raise StudyError("Source runtime metadata changed after size-control initialization")
    plan = plans[role]
    for name in ("lock.json", "selected_profiles.json", "selected_memory.json"):
        if read_json(_safe(expected_control, "runtime/" + name)) != plan[name]:
            raise StudyError(f"size-control runtime identity changed: {role}/{name}")
    expected_installed = plan["installations.json"]
    if ({key: value for key, value in installed.items() if key != "models"}
            != {key: value for key, value in expected_installed.items() if key != "models"}):
        raise StudyError("size-control installed environment identity changed")
    if role != "startlux9b" and installed["models"] != expected_installed["models"]:
        raise StudyError("Existing 4B checkpoint binding changed")
    if role == "startlux9b" and installed["models"]:
        model = installed["models"].get("startlux", {})
        if (set(installed["models"]) != {"startlux"} or model.get("revision") != STARTLUX9B_REVISION
                or not Path(model.get("model_path", "")).is_absolute()
                or Path(model.get("model_path", "")).name != STARTLUX9B_REVISION
                or Path(binding["work_dir"]) not in Path(model["model_path"]).parents):
            raise StudyError("Downloaded 9B checkpoint binding changed")
    return expected_control, binding, _session(output), installed


@contextmanager
def _environment(binding, session):
    previous = dict(os.environ)
    work = Path(binding["work_dir"])
    deadline = session["deadline_epoch"]
    if previous.get("DECISION_STUDY_DEADLINE_EPOCH"):
        deadline = min(deadline, float(previous["DECISION_STUDY_DEADLINE_EPOCH"]))
    try:
        os.environ.update({"DECISION_STUDY_RUNTIME_DIR": str(work), "DECISION_STUDY_DEADLINE_EPOCH": str(deadline),
                           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(work / "pycache"),
                           "XDG_CACHE_HOME": str(work / "cache"), "TRITON_CACHE_DIR": str(work / "cache/triton"),
                           "TORCHINDUCTOR_CACHE_DIR": str(work / "cache/torchinductor"), "CUDA_CACHE_PATH": str(work / "cache/cuda")})
        with legacy_environment(binding, PROJECT):
            yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _cuda_preflight(control, binding, installed):
    from scripts.prepare_cuda_toolkit import validate_target
    toolkit = Path(binding["source_work_dir"]) / "toolchains" / TOOLKIT
    validate_target(toolkit, PROJECT / "configs/cuda_build_toolkit_13_0_flashinfer.txt")
    info = next(iter(installed["environments"].values()))
    frozen._verify_environment(control, info)
    stamp = time.time_ns()
    report = control / "runtime" / f"cuda-toolchain-{stamp}.json"
    frozen._run([info["python"], PROJECT / "scripts/cuda_toolchain_preflight.py", "--result", report,
                 "--work-dir", Path(binding["work_dir"]) / "cuda-toolchain-13.0-flashinfer", "--toolkit-target", toolkit],
                control, f"cuda-toolchain-{stamp}.log", cwd=PROJECT)
    value = read_json(report)
    environment = value.get("environment", {})
    if value.get("status") != "complete" or set(environment) != ENV_KEYS or not all(isinstance(v, str) for v in environment.values()):
        raise StudyError("CUDA preflight did not produce the expected verified environment")
    return environment


def assert_gpu_idle():
    """Inspect compute PIDs without terminating any process or changing GPU state."""
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise StudyError("Cannot verify that the GPU is idle before starting size control") from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdigit() or int(line) <= 0 for line in lines):
        raise StudyError("Unexpected nvidia-smi compute-process output; no GPU process state assumed")
    pids = sorted({int(line) for line in lines})
    if pids:
        raise StudyError(f"GPU has existing compute processes {pids}; stop the owning job normally before retrying. No processes were killed.")
    return {"created_at": utc_now(), "command": command, "compute_pids": [], "status": "idle"}


@contextmanager
def runtime_context(v1_dir, output_dir, role, *, tool_preflight=True):
    control, binding, session, installed = _validated(output_dir, role)
    if Path(v1_dir).absolute() != Path(binding["v1_dir"]):
        raise StudyError("v1 source differs from the initialized size-control runtime")
    write_json(_safe(control, "runtime/session.json"), session)
    with _environment(binding, session):
        with Budget(control, "prediction", cost_mode="provided") as budget:
            report = frozen.doctor(control, require_gpu=True)
            if len(report.get("gpu", [])) != 1 or report["gpu"][0]["name"] != "NVIDIA L40S" or report["gpu"][0]["memory_mb"] < 44000:
                raise StudyError("This size-control protocol requires the planned single L40S with at least 44,000 MiB")
            idle = assert_gpu_idle()
            write_json(control / "runtime" / f"gpu-idle-{time.time_ns()}.json", idle)
            if tool_preflight:
                os.environ.update(_cuda_preflight(control, binding, installed))
            budget.check()
            yield control, budget


def ensure_download(output_dir, role, *, control=None):
    control, binding, session, installed = _validated(output_dir, role, control)
    model = backend_model(role)
    if model in installed["models"]:
        record = installed["models"][model]
        if not Path(record["model_path"]).is_dir():
            raise StudyError("Recorded checkpoint disappeared; do not silently redownload or substitute it")
        _verify_model_metadata(record["model_path"])
        return record
    if role != "startlux9b":
        raise StudyError("The original 4B checkpoint must already exist")
    work = _safe(output_dir, "runtime-work")
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < 40 * 1024 ** 3:
        warnings.warn("Less than 40 GiB is free for the new 9B snapshot and temporary files; this is a conservative disk check, not a RAM/VRAM guarantee", RuntimeWarning)
    info = next(iter(installed["environments"].values()))
    result = control / "runtime/download-startlux9b.json"
    with _environment(binding, session):
        frozen._verify_environment(control, info)
        frozen._run([info["python"], PROJECT / "scripts/model_probe.py", "download", "--model", "startlux",
                     "--output-dir", control, "--result", result], control, "download-startlux9b.log", cwd=PROJECT)
    record = read_json(result)
    model_path = Path(record.get("model_path", ""))
    if (record.get("model") != "startlux" or record.get("revision") != STARTLUX9B_REVISION
            or not model_path.is_dir() or model_path.name != STARTLUX9B_REVISION or work not in model_path.parents):
        raise StudyError("9B download did not resolve the pinned snapshot inside size-control work")
    _verify_model_metadata(model_path)
    installed["models"] = {"startlux": record}
    write_json(_safe(control, "runtime/installations.json"), installed)
    _verify_model_metadata(model_path)
    return record


def _verify_model_metadata(model_path):
    from .identity import protocol
    for name, expected in protocol()["model"]["metadata_sha256"].items():
        if file_hash(Path(model_path) / name) != expected:
            raise StudyError(f"Pinned StartLux9B model metadata differs: {name}")


def token_preflight(output_dir, role, examples, *, control=None):
    control, binding, session, _ = _validated(output_dir, role, control)
    ensure_download(output_dir, role, control=control)
    with _environment(binding, session):
        return frozen.token_preflight_examples(PROJECT, control, backend_model(role), examples)


def runtime_identity(output_dir, role):
    control, _, _ = _offline_control(output_dir, role)
    return {"role": role, "backend_model": backend_model(role), "runtime": frozen.runtime_config(control, backend_model(role))}


@contextmanager
def managed_services(output_dir, role, *, control=None):
    control, binding, session, _ = _validated(output_dir, role, control)
    ensure_download(output_dir, role, control=control)
    with _environment(binding, session):
        with frozen.managed_services(PROJECT, control, [backend_model(role)]) as endpoints:
            yield endpoints[backend_model(role)]
