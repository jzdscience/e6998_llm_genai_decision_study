"""Runtime bindings and the one-model deployment for the direct-accuracy matrix.

Roles and their accepted assets, all reused read-only:

* ``startlux9b`` and ``qwen9b``: the runtime locks size control and V3 froze for
  these checkpoints, their downloaded snapshots, and the V1 vLLM 0.31 environment.
  They start through the serving study's launcher, which is V1's single-model
  launcher plus ``--max-num-seqs``.
* ``kev4b``: the unchanged V1 lock, ``kev`` environment, LoRA run and base
  snapshots and the V1 primary profile. It starts through V1's own
  ``managed_services`` (Kev is a PyTorch server, not a vLLM engine). Its base model
  is read from the V1 Hugging Face hub cache with ``HF_HUB_OFFLINE=1``.

Every deployment registers the processes it started, including from a launch that
failed half way, and ends with the serving study's ownership-safe stop verification.
"""
from contextlib import ExitStack, contextmanager
import copy
import math
import os
from pathlib import Path
import subprocess
import time

from .. import runtime as frozen
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..runtime_compat import legacy_environment
from ..size_control import runtime as size_runtime
from ..upgrade_study.launcher import managed_service
from ..upgrade_study.runtime import (_launch_record, _old_role_lock, assert_gpu_owned_by, engine_log_facts,
                                     gpu_processes, verify_stopped)
from ..v3 import runtime as v3_runtime
from .identity import (INTERFACE, MAX_MODEL_LEN, MAX_NUM_SEQS, PROJECT, PROTOCOL_ID, REVISIONS, ROLES, TOKEN_RESERVE, normal,
                       VLLM_ROLES, fixed_write, protocol, safe)

CONTROL_FILES = {"startlux9b": ("lock.json", "installations.json", "selected_profiles.json", "selected_memory.json"),
                 "qwen9b": ("lock.json", "installations.json", "selected_profiles.json", "selected_memory.json"),
                 "kev4b": ("lock.json", "installations.json", "selected_profiles.json")}
TOKEN_BINDING_KEYS = {"status", "file", "sha256", "n_examples", "max_model_len", "model_revision", "effective_config_hash"}
# Accepted earlier runs of the same checkpoints: their effective runtime must match apart from the GPU entry.
ACCEPTED_RUNTIME = {"kev4b": ("v1", "predictions/kev/config.json"),
                    "qwen9b": ("v3", "predictions-jobs/qwen9b-holdout/predictions/qwen/config.json"),
                    "startlux9b": ("size_control", "predictions-job/predictions/startlux/config.json")}


def _within(path, root):
    path, root = normal(path), normal(root)
    return path == root or root in path.parents


def _metadata(path):
    """SHA-256 of the top-level JSON files of a snapshot; following the hub's blob links."""
    path = Path(path)
    if not path.is_dir():
        raise StudyError(f"Recorded snapshot is missing: {path}; nothing is downloaded or substituted")
    files = {entry.name: file_hash(path / entry.name) for entry in sorted(os.scandir(path), key=lambda e: e.name)
             if entry.name.endswith(".json") and (path / entry.name).is_file()}
    if not files:
        raise StudyError(f"Snapshot has no metadata files to verify: {path}")
    return files


def _snapshot(role, other_dir, model):
    installed = read_json(safe(other_dir, f"runtime-controls/{role}/runtime/installations.json"))
    record = installed.get("models", {}).get(model)
    work = Path(read_json(safe(other_dir, "runtime/binding.json"))["work_dir"])
    path = Path((record or {}).get("model_path", ""))
    if (not record or record.get("revision") != REVISIONS[role] or not path.is_absolute()
            or path.name != REVISIONS[role] or not _within(path, work)):
        raise StudyError(f"Missing or changed {role} snapshot record")
    return copy.deepcopy(record)


def _kev(source, lock, installed, profiles, source_work):
    if profiles.get("kev", "primary") != "primary":
        raise StudyError("The accepted Kev run used the V1 primary profile; the source runtime now selects another")
    env, record, pinned = installed.get("environments", {}).get("kev"), installed.get("models", {}).get("kev"), lock["models"]["kev"]
    spec = protocol()["roles"]["kev4b"]
    python = Path((env or {}).get("python", ""))
    if not env or not python.is_absolute() or python.parts[-4:] != ("envs", "kev", "bin", "python") or python.parents[3] != source_work:
        raise StudyError("The V1 kev environment record is missing or outside the V1 work directory")
    model_path, base_path = Path((record or {}).get("model_path", "")), Path((record or {}).get("base_path") or "")
    hub = source_work / "cache/huggingface/hub"
    if (not record or record.get("revision") != pinned["revision"]
            or pinned["revision"] != spec["revision"] or record.get("base_revision") != pinned["base_revision"]
            or pinned["base_revision"] != spec["base_revision"] or pinned["base_repo"] != spec["base_repo"]
            or pinned["code_revision"] != spec["code_revision"] or record.get("native_temperature") != spec["native_temperature"]
            or not model_path.is_absolute() or model_path.name != pinned["revision"] or not _within(model_path, hub)
            or not base_path.is_absolute() or base_path.name != pinned["base_revision"]
            or base_path.parent.parent != hub / ("models--" + pinned["base_repo"].replace("/", "--"))):
        raise StudyError("The V1 Kev checkpoint record differs from the protocol")
    return {"lock.json": copy.deepcopy(lock),
            "installations.json": {"lock_hash": lock["lock_hash"], "environments": {"kev": copy.deepcopy(env)},
                                   "models": {"kev": copy.deepcopy(record)}},
            "selected_profiles.json": {"kev": "primary"}}, str(hub)


def plans(dirs, output_dir):
    """Role controls derived from the accepted runtimes; pure apart from reading them and snapshot metadata."""
    v1_dir, v3_dir, size_dir = dirs["v1"], dirs["v3"], dirs["size_control"]
    source, lock, installed, profiles, _, env_name, source_work, hashes = v3_runtime._source(v1_dir)
    output = normal(output_dir)
    for root in (*map(Path, dirs.values()), PROJECT):
        if _within(output, root) or _within(root, output):
            raise StudyError("Use an independent output directory that overlaps no input or source root")
    roles = protocol()["roles"]
    result = {}
    for role, other, sname in (("startlux9b", size_dir, "size_control"), ("qwen9b", v3_dir, "v3")):
        model = INTERFACE[role]
        role_lock = _old_role_lock(role, v1_dir, other)
        record = _snapshot(role, other, model)
        entry = role_lock["models"][model]
        if entry["repo"] != roles[role]["repo"] or entry["revision"] != roles[role]["revision"]:
            raise StudyError(f"{role} lock differs from the protocol checkpoint")
        if role == "startlux9b" and entry.get("native_temperature") != roles[role]["native_temperature"]:
            raise StudyError("StartLux-9B native temperature differs from the protocol")
        result[role] = {"lock.json": role_lock,
                        "installations.json": {"lock_hash": role_lock["lock_hash"],
                                               "environments": {env_name: copy.deepcopy(installed["environments"][env_name])},
                                               "models": {model: record}},
                        "selected_profiles.json": {model: "primary"},
                        "selected_memory.json": {model: float(roles[role]["gpu_memory_utilization"])}}
    result["kev4b"], kev_hub = _kev(source, lock, installed, profiles, source_work)
    accepted = {}
    for role, (key, relative) in ACCEPTED_RUNTIME.items():
        config = read_json(safe(dirs[key], relative))
        if fingerprint(config) != _accepted_config_hash(role):
            raise StudyError(f"Accepted {role} configuration differs from its pinned hash")
        accepted[role] = config
    # Every snapshot is checked on every binding; tests fake only this filesystem boundary (_metadata).
    snapshots = {}
    for role in ROLES:
        record = result[role]["installations.json"]["models"][INTERFACE[role]]
        snapshots[role] = {"model": _metadata(record["model_path"])}
        if role == "startlux9b":
            size_runtime._verify_model_metadata(record["model_path"])
        if role == "kev4b":
            snapshots[role]["base"] = _metadata(record["base_path"])
    binding = {"schema_version": 1, "protocol": PROTOCOL_ID, "v1_dir": str(source), "source_dirs": dict(dirs),
               "source_work_dir": str(source_work), "work_dir": str(safe(output, "runtime-work")),
               "source_files": hashes, "roles": list(ROLES), "kev_hub_cache": kev_hub,
               "accepted_configs": accepted,
               "snapshot_metadata": snapshots, "snapshots_checked": True,
               "control_identity": {role: {name: fingerprint(value) for name, value in plan.items()}
                                    for role, plan in result.items()}}
    return binding, result, accepted


def _accepted_config_hash(role):
    refs = protocol()["references"]
    return {"kev4b": refs["kev4b-clinc"]["config_hash"], "qwen9b": refs["qwen9b-anli"]["config_hash"],
            "startlux9b": refs["startlux9b-anli"]["config_hash"]}[role]


def control_root(output_dir, role):
    if role not in ROLES:
        raise StudyError(f"Unknown direct-matrix role: {role}")
    return safe(output_dir, f"runtime-controls/{role}")


def initialize(dirs, output_dir, hours_left, *, new_allocation=False):
    """Bind controls and the actual allocation; no download, start or GPU call."""
    if isinstance(hours_left, bool) or not isinstance(hours_left, (float, int)) or not math.isfinite(hours_left) or not 0 < hours_left <= 24:
        raise StudyError("Actual remaining hours must be finite and in (0,24]")
    binding, role_plans, _ = plans(dirs, output_dir)
    allocation = v3_runtime._allocation()
    if not allocation["slurm_job_id"]:
        raise StudyError("Initialize inside the actual Slurm allocation (SLURM_JOB_ID is missing)")
    binding_path = safe(output_dir, "runtime/binding.json")
    if binding_path.exists() and read_json(binding_path) != binding:
        raise StudyError("Runtime binding changed since initialization; preserve this output and use a new one")
    session_path = safe(output_dir, "runtime/session.json")
    previous = read_json(session_path) if session_path.exists() else None
    if previous and previous["allocation"] == allocation:
        if new_allocation:
            raise StudyError("--new-allocation was given, but this is the same Slurm job; the deadline is never extended")
        session = v3_runtime._session(output_dir)
        session = {**session, "deadline_epoch": min(session["deadline_epoch"], time.time() + 3600 * hours_left)}
    else:
        if previous and (not new_allocation or previous["allocation"]["slurm_job_id"] == allocation["slurm_job_id"]):
            raise StudyError("This output belongs to another allocation; acknowledge a genuinely different Slurm job with --new-allocation")
        session = {"started_at": utc_now(), "deadline_epoch": time.time() + 3600 * hours_left,
                   "hours_left_at_initialization": hours_left, "allocation": allocation,
                   "notice": "Remaining allowance read from the current allocation; not a provider renewal"}
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
            write_json(doctor, read_json(safe(binding["v1_dir"], "runtime/doctor.json")))
        write_json(safe(control, "runtime/session.json"), session)
    if previous and previous != session:
        fixed_write(safe(output_dir, f"runtime/session-history/{fingerprint(previous)}.json"), previous)
    write_json(session_path, session)
    return {"binding": binding, "session": session, "notice": "Controls bound; no model start, download or GPU call"}


def verify_binding(output_dir):
    """Recompute every binding from the source runtimes and compare with the stored controls."""
    path = safe(output_dir, "runtime/binding.json")
    if not path.is_file():
        raise StudyError("Initialize the runtime binding first (init-runtime)")
    binding = read_json(path)
    if binding.get("snapshots_checked") is not True:
        raise StudyError("The runtime binding does not record verified model snapshots; it cannot drive a node run")
    expected, role_plans, accepted = plans(binding["source_dirs"], output_dir)
    if binding != expected:
        raise StudyError("Runtime source identity or snapshot metadata changed since initialization")
    controls = {}
    for role, plan in role_plans.items():
        control = control_root(output_dir, role)
        for name, value in plan.items():
            if read_json(safe(control, "runtime/" + name)) != value:
                raise StudyError(f"Runtime control changed: {role}/{name}")
        controls[role] = control
    return controls, binding, accepted


def verify_binding_offline(output_dir, settings_value):
    """A returned binding checked from the bundle alone: no source roots, snapshots or allocation needed.

    The stored controls must match the binding's fingerprints, the accepted earlier
    configurations must match their protocol-pinned hashes, and the binding must record
    that snapshot metadata was verified on the node. Production commands use
    ``verify_binding`` instead, which recomputes everything from the source runtimes.
    """
    path = safe(output_dir, "runtime/binding.json")
    if not path.is_file():
        raise StudyError("The bundle has no runtime binding")
    binding = read_json(path)
    if (binding.get("protocol") != PROTOCOL_ID or binding.get("snapshots_checked") is not True
            or binding.get("roles") != list(ROLES) or binding.get("source_dirs") != settings_value["source_dirs"]
            or set(binding.get("snapshot_metadata", {})) != set(ROLES)):
        raise StudyError("The returned runtime binding is not a verified node binding for these settings")
    for role in ROLES:
        if fingerprint(binding["accepted_configs"][role]) != _accepted_config_hash(role):
            raise StudyError(f"Accepted {role} configuration in the binding differs from its pinned hash")
        identity = binding["control_identity"][role]
        if set(identity) != set(CONTROL_FILES[role]):
            raise StudyError(f"Control inventory differs for {role}")
        for name, digest in identity.items():
            if fingerprint(read_json(safe(control_root(output_dir, role), "runtime/" + name))) != digest:
                raise StudyError(f"Runtime control changed: {role}/{name}")
    return {role: control_root(output_dir, role) for role in ROLES}, binding


TERMINAL_JOB_STATES = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY",
                       "PREEMPTED", "REVOKED", "TIMEOUT"}


def slurm_job_state(job_id, *, runner=None):
    """Accounting record of one earlier allocation. Only a listed terminal state counts as ended.

    Unknown, absent, unreadable or still-active records are returned as not terminal; the
    caller fails closed. Nothing is cancelled or signalled.
    """
    command = ["sacct", "--jobs", str(job_id), "--allocations", "--noheader", "--parsable2",
               "--format=JobID,State,End"]
    record = {"query": command, "job_id": str(job_id), "state": None, "terminal": False}
    if not str(job_id).isdigit():
        record["error"] = "recorded job ID is not numeric"
        return record
    try:
        result = (runner or subprocess.run)(command, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    lines = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    matches = [fields for fields in lines if fields and fields[0] == str(job_id)]
    if result.returncode != 0 or len(matches) != 1 or len(matches[0]) < 2:
        record["error"] = "no unique accounting record for the job"
        return record
    state = matches[0][1].strip()
    record["state"] = state
    record["terminal"] = state.split()[0] in TERMINAL_JOB_STATES if state else False
    return record


def validate(output_dir):
    """The verified binding plus the current, unexpired session of this exact allocation."""
    controls, binding, accepted = verify_binding(output_dir)
    return controls, binding, v3_runtime._session(output_dir), accepted


def check_time(output_dir, required_seconds=0):
    session = v3_runtime._session(output_dir)
    if session["deadline_epoch"] - time.time() < required_seconds:
        raise StudyError(f"Insufficient allocation allowance for the next step ({required_seconds:.0f}s); "
                         "export, then resume in a new allocation with --new-allocation")
    return session


@contextmanager
def role_environment(binding, session, role):
    """The V3 runtime environment, plus offline read-only access to the V1 hub cache for Kev."""
    with v3_runtime._environment(binding, session):
        if role == "kev4b":
            os.environ.update({"HF_HUB_CACHE": binding["kev_hub_cache"], "HF_HUB_OFFLINE": "1",
                               "TRANSFORMERS_OFFLINE": "1"})
        yield


def effective_runtime(control, role, accepted):
    """V1's effective runtime identity, which must equal the accepted earlier run except the GPU entry.

    ``accepted`` maps each role to the full accepted earlier configuration (its hash is pinned).
    """
    return check_runtime_value(frozen.runtime_config(control, INTERFACE[role]), role, accepted)


def check_runtime_value(value, role, accepted):
    if fingerprint(accepted[role]) != _accepted_config_hash(role):
        raise StudyError(f"Accepted {role} configuration differs from its pinned hash")
    reference = accepted[role]["runtime"]
    strip = lambda config: {k: v for k, v in config.items() if k != "gpu"}
    if strip(value) != strip(reference):
        changed = sorted(k for k in set(value) | set(reference) if k != "gpu" and value.get(k) != reference.get(k))
        raise StudyError(f"{role} effective runtime differs from its accepted earlier run: {changed}")
    gpus = value.get("gpu", [])
    if len(gpus) != 1 or gpus[0].get("name") != "NVIDIA L40S" or gpus[0].get("memory_mb", 0) < 44000:
        raise StudyError("The recorded GPU is not one L40S with at least 44,000 MiB")
    return value


def _default_launcher(role):
    if role in VLLM_ROLES:
        return lambda control, model: managed_service(control, model, max_num_seqs=MAX_NUM_SEQS)
    return lambda control, model: frozen.managed_services(PROJECT, control, [model])


def refresh_doctor(control):
    report = frozen.doctor(control, require_gpu=True)
    gpus = report.get("gpu", [])
    if len(gpus) != 1 or gpus[0]["name"] != "NVIDIA L40S" or gpus[0]["memory_mb"] < 44000:
        raise StudyError("This study requires exactly one L40S with at least 44,000 MiB")
    return report


@contextmanager
def deployment(output_dir, role, *, launcher=None, lister=None, preflight=True, sink=None):
    """Start one role's services and always stop and verify every process group it started."""
    controls, binding, session, _ = validate(output_dir)
    control, model = controls[role], INTERFACE[role]
    launch = launcher or _default_launcher(role)
    record = sink if sink is not None else {}
    record.update(role=role, model=model, started_at=utc_now(), allocation=session["allocation"],
                  max_num_seqs=MAX_NUM_SEQS if role in VLLM_ROLES else None)
    owned, began = set(), time.time()
    with legacy_environment(binding, PROJECT), role_environment(binding, session, role):
        if preflight:
            refresh_doctor(control)
            if role in VLLM_ROLES:
                installed = read_json(safe(control, "runtime/installations.json"))
                os.environ.update(v3_runtime._cuda_preflight(control, binding, installed))
        failure = None
        try:
            with ExitStack() as stack:
                record["gpu_before_start"] = assert_gpu_owned_by(set(), lister=lister)
                write_json(safe(control, "runtime/session.json"), session)
                services = safe(control, "runtime/last_services.json")
                before = file_hash(services) if services.is_file() else None
                launched_at, t0 = time.time(), time.perf_counter()
                try:
                    endpoints = stack.enter_context(launch(control, model))
                except BaseException:
                    # The launcher recorded each process as it started it; a start that failed half way
                    # still owns those groups, which are verified and cleaned below.
                    partial = _launch_record(services, before, launched_at)
                    if partial is not None:
                        owned |= {int(p["pid"]) for p in partial.get("processes", [])}
                        record.update(startup_failed=True, processes=partial.get("processes", []))
                    raise
                meta = _launch_record(services, before, launched_at)
                if meta is None:
                    raise StudyError(f"{role} started without a record of this launch's processes")
                owned |= {int(p["pid"]) for p in meta["processes"]}
                record.update(endpoint=endpoints[model], startup_seconds=time.perf_counter() - t0,
                              processes=meta["processes"], launcher_max_num_seqs=meta.get("max_num_seqs"),
                              health=meta.get("models", {}).get(model, {}).get("health"),
                              runtime=meta.get("models", {}).get(model, {}).get("config"),
                              engine_log=engine_log_facts(meta["processes"]))
                record["gpu_after_start"] = assert_gpu_owned_by(owned, lister=lister)
                record["ready_at"] = utc_now()
                yield {"role": role, "endpoint": endpoints[model], "pids": sorted(owned), "record": record}
        except BaseException as exc:
            failure = exc
            raise
        finally:
            # The ExitStack has already run the launcher's own SIGTERM/SIGKILL cleanup.
            if owned:
                try:
                    record["stop_verification"] = verify_stopped(owned, since=began, lister=lister)
                except StudyError as stop_error:
                    record["stop_verification"] = {"error": str(stop_error)}
                    if failure is None:
                        raise


def engine_command_limited(record):
    """True when this deployment's vLLM engine command carries exactly the protocol limit."""
    engines = [p for p in record.get("processes", []) if p.get("name", "").endswith("-engine")]
    if record.get("role") not in VLLM_ROLES:
        return not engines and all("--max-num-seqs" not in p.get("command", []) for p in record.get("processes", []))
    if len(engines) != 1:
        return False
    command = engines[0].get("command", [])
    if command.count("--max-num-seqs") != 1:
        return False
    position = command.index("--max-num-seqs") + 1
    return position < len(command) and command[position] == str(MAX_NUM_SEQS)


def token_report_path(output_dir, role, examples):
    return safe(control_root(output_dir, role), f"runtime/tokens-interactive-{INTERFACE[role]}-{frozen._hash(examples)[:16]}.json")


def verify_token_report(output_dir, role, binding, examples):
    """The raw per-input tokenizer report: coverage, model-specific reserve, limits and configuration."""
    output = normal(output_dir)
    path = token_report_path(output, role, examples)
    if not isinstance(binding, dict) or set(binding) != TOKEN_BINDING_KEYS or binding["file"] != str(path.relative_to(output)):
        raise StudyError(f"Token binding for {role} has an unexpected shape or file")
    if path.is_symlink() or not path.is_file() or file_hash(path) != binding["sha256"]:
        raise StudyError(f"Raw token report for {role} is missing or changed")
    report = read_json(path)
    control, model = control_root(output, role), INTERFACE[role]
    rows, reserve = report.get("rows"), TOKEN_RESERVE[model]
    fields = ("n_examples", "max_model_len", "model_revision", "effective_config_hash")
    bad_row = lambda row: (isinstance(row.get("input_tokens"), bool) or not isinstance(row.get("input_tokens"), int)
                           or row["input_tokens"] < 1 or row.get("reserved_output_tokens") != reserve
                           or row["input_tokens"] + reserve > MAX_MODEL_LEN
                           or not isinstance(row.get("token_ids_sha256"), str) or len(row["token_ids_sha256"]) != 64)
    if (report.get("status") != "ok" or binding["status"] != "ok" or report.get("model") != model
            or report.get("model_revision") != REVISIONS[role] or report.get("max_model_len") != MAX_MODEL_LEN
            or report.get("n_examples") != len(examples) or not isinstance(rows, list)
            or [row.get("sample_id") for row in rows] != [e["sample_id"] for e in examples]
            or report.get("overlong") != [] or report.get("truncated") is not False or any(bad_row(r) for r in rows)
            or report.get("effective_config_hash") != frozen._hash(frozen.runtime_config(control, model))
            or {k: binding[k] for k in fields} != {k: report.get(k) for k in fields}):
        raise StudyError(f"Raw token report for {role} does not cover every input within the frozen limits")
    return report


def bind_token_report(output_dir, role, examples):
    output = normal(output_dir)
    path = token_report_path(output, role, examples)
    report = read_json(path)
    binding = {"status": report.get("status"), "file": str(path.relative_to(output)), "sha256": file_hash(path),
               **{k: report.get(k) for k in ("n_examples", "max_model_len", "model_revision", "effective_config_hash")}}
    verify_token_report(output, role, binding, examples)
    return binding


def token_check(output_dir, examples_by_role):
    """Exact official-renderer token counts for every formal and development input of each role."""
    controls, binding, session, accepted = validate(output_dir)
    reports = {}
    with legacy_environment(binding, PROJECT):
        for role in ROLES:
            with role_environment(binding, session, role):
                refresh_doctor(controls[role])
                effective_runtime(controls[role], role, accepted)
                frozen.token_preflight_examples(PROJECT, controls[role], INTERFACE[role], examples_by_role[role])
                reports[role] = bind_token_report(output_dir, role, examples_by_role[role])
    return reports


__all__ = ["plans", "initialize", "validate", "check_time", "deployment", "token_check", "verify_token_report",
           "bind_token_report", "effective_runtime", "check_runtime_value", "engine_command_limited", "gpu_processes",
           "control_root", "role_environment", "refresh_doctor"]
