"""Prepare, preflight, collect, freeze, evaluate, status and export for the direct-accuracy matrix.

Public functions expect the CLI's exclusive output lock (``audit`` is read-only and
takes none). Collection is resumable: accepted rows live in each cell's V1
PredictionStore and are never recollected; interrupted attempts, failed
preflights and deployment records are preserved. An attempt counts toward a
completed cell only when the parent accepted its on-time exit (``accepted.json``)
or an operator-acknowledged ``--resume`` recorded a successful reconciliation of
it (``recovery.json``). Test gold is opened only by ``evaluate`` after the freeze
and access receipts.
"""
from datetime import datetime
from itertools import combinations
import json
import math
import os
from pathlib import Path
import hashlib
import signal
import subprocess
import sys
import time
import zipfile

from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..runner import _probability_comparison
from ..storage import PredictionStore, load_predictions
from ..upgrade_study import runtime as upgrade_runtime
from ..v3 import runtime as v3_runtime
from . import collect, data, runtime, stats
from .data import checked, seal
from .identity import (CELL_ORDER, DATASETS, INTERFACE, MODELS, PROJECT, PROTOCOL_ID, REFERENCE_CELLS, ROLES,
                       TOTAL_NEW, dev_set, fixed_write, normal, protocol, protocol_hash, safe, source_files,
                       source_identity, split_cell)

WORKER = "decision_study.direct_matrix.worker"
HEALTH_PATH = {"kev": "/v1/models", "startlux": "/health", "qwen": "/health"}
EXIT_MARGIN = 3
STAGE_FILES = ("plan.json", "rows.json", "complete.json", "accepted.json")
CLOCK_TOLERANCE_SECONDS = 1.0
EXPORT_SKIP_DIRS = {"runtime-work", "exports", "__pycache__", ".git", "envs", "cache", "models", "weights"}


def _epoch(stamp):
    value = datetime.fromisoformat(stamp)
    if value.tzinfo is None:
        raise StudyError("Timestamp without a time zone")
    return value.timestamp()


# ---------------------------------------------------------------- settings and configuration

def prepare(dirs, output, *, source=None):
    """Exact inputs, verified references and the frozen settings; CPU only, no gold."""
    output = normal(output)
    source = source if source is not None else source_identity(require_clean=True)
    manifest = data.prepare(dirs, output, source=source)
    value = {"schema_version": 1, "protocol": PROTOCOL_ID, "protocol_hash": protocol_hash(), "source": source,
             "source_dirs": dirs, "manifest_hash": manifest["manifest_hash"]}
    path = safe(output, "settings.json")
    if path.exists():
        saved = checked(path, "settings_hash")
        if {k: v for k, v in saved.items() if k not in ("created_at", "settings_hash")} != value:
            raise StudyError("Prepared settings differ (protocol, source checkout or input roots); preserve this "
                             "output and use a new one")
    else:
        write_json(path, seal({**value, "created_at": utc_now()}, "settings_hash"))
    saved = checked(path, "settings_hash")
    return {"cells": len(CELL_ORDER), "new_predictions": TOTAL_NEW, "counts": manifest["counts"],
            "manifest_hash": manifest["manifest_hash"], "settings_hash": saved["settings_hash"],
            "source_commit": source["commit"], "test_gold_opened": False}


def settings(output, *, check_source=False):
    value = checked(safe(output, "settings.json"), "settings_hash")
    if value.get("protocol") != PROTOCOL_ID or value.get("protocol_hash") != protocol_hash():
        raise StudyError("The protocol changed after preparation; preserve this output")
    if value["source"]["files"] != source_files():
        raise StudyError("Source files differ from the prepared checkout; use that commit or a new output")
    if data.verify(output)["manifest_hash"] != value["manifest_hash"]:
        raise StudyError("Prepared inputs differ from the settings")
    if check_source and source_identity(require_clean=True) != value["source"]:
        raise StudyError("The source checkout differs from the prepared commit; do not change it during the study")
    return value


def context(output, *, check_source=False, binding=None):
    value = settings(output, check_source=check_source)
    if binding is None:
        binding = read_json(safe(output, "runtime/binding.json"))
    return {"settings": value, "settings_hash": value["settings_hash"], "binding": binding}


def cell_config_for(output, ctx, cell_id):
    role, _ = split_cell(cell_id)
    value = runtime.effective_runtime(runtime.control_root(output, role), role, ctx["binding"]["accepted_configs"])
    return collect.cell_config(output, cell_id, ctx["settings"], value)


def verify_stored_config(output, ctx, cell_id):
    """A stored cell configuration must equal the one these inputs, sources and protocol imply."""
    role, _ = split_cell(cell_id)
    stored = read_json(collect.cell_dir(output, cell_id) / f"predictions/{INTERFACE[role]}/config.json")
    runtime.check_runtime_value(stored.get("runtime", {}), role, ctx["binding"]["accepted_configs"])
    if stored != collect.cell_config(output, cell_id, ctx["settings"], stored["runtime"]):
        raise StudyError(f"Stored configuration of {cell_id} differs from the prepared inputs, source or protocol")
    return stored


def role_cells(role):
    return [c for c in CELL_ORDER if split_cell(c)[0] == role]


def role_dev_sets(role):
    return sorted({dev_set(split_cell(c)[1]) for c in role_cells(role)}, key=["clinc", "squad", "anli"].index)


def role_inputs(output):
    """Every formal and development input a role receives, once each, in a fixed order."""
    result = {}
    for role in ROLES:
        seen, rows = set(), []
        for cell in role_cells(role):
            dataset = split_cell(cell)[1]
            for example in data.cell_examples(output, cell) + data.dev_examples(output, dev_set(dataset)):
                if example["sample_id"] not in seen:
                    seen.add(example["sample_id"])
                    rows.append(example)
        result[role] = rows
    return result


def token_receipt(output, ctx):
    """Preflight only: run the tokenizer check once if absent, then verify it."""
    path = safe(output, "preflight/tokens.json")
    if not path.exists():
        fixed_write(path, {"settings_hash": ctx["settings_hash"], "at": utc_now(),
                           "reports": runtime.token_check(output, role_inputs(output))})
    return verify_token_receipt(output, ctx)


def verify_token_receipt(output, ctx, roles=ROLES):
    """Read-only: the receipt and every raw tokenizer report it binds. Never tokenizes anything."""
    path = safe(output, "preflight/tokens.json")
    if not path.is_file():
        raise StudyError("Token receipt is missing; it is created only by preflight")
    value = read_json(path)
    if value.get("settings_hash") != ctx["settings_hash"] or set(value.get("reports", {})) != set(ROLES):
        raise StudyError("Token receipt belongs to other settings")
    inputs = role_inputs(output)
    k = protocol()["collection"]["preflight_longest_inputs"]
    longest, maxima = {}, {}
    for role in roles:
        report = runtime.verify_token_report(output, role, value["reports"][role], inputs[role])
        tokens = {row["sample_id"]: row["input_tokens"] for row in report["rows"]}
        for cell in role_cells(role):
            ids = [e["sample_id"] for e in data.cell_examples(output, cell)]
            ranked = sorted(ids, key=lambda sid: (-tokens[sid], sid))
            longest[cell] = ranked[:k]
            maxima[cell] = tokens[ranked[0]]
    return file_hash(path), longest, maxima


# ---------------------------------------------------------------- child processes

def _stop_process(process):
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


def _worker_command(output, folder):
    return [sys.executable, "-m", WORKER, "--root", str(output), "--attempt", str(folder)]


def _run_process(output, folder, plan, watchdog_seconds):
    """Run one owned child under an absolute watchdog and accept only an on-time, successful exit.

    The parent writes ``accepted.json`` after it has seen the child exit 0 within the
    watchdog with a completion receipt. A late or failed exit renames that receipt to
    ``unaccepted-complete.json``; renaming it back does not create the parent's acceptance.
    """
    limit = watchdog_seconds
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise StudyError("A positive finite watchdog bound is required; not enough allocation time remains")
    folder.mkdir(parents=True, exist_ok=True)
    plan = {**plan, "watchdog_deadline_epoch": time.time() + limit}
    fixed_write(folder / "plan.json", plan)
    with (folder / "controller.log").open("x") as log:
        process = subprocess.Popen(_worker_command(output, folder), cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    until = time.monotonic() + limit
    try:
        while True:
            runtime.check_time(output)
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise StudyError("Child process exceeded the absolute parent watchdog; its evidence is preserved")
            try:
                process.wait(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if time.monotonic() > until:
            raise StudyError("Child exit observed after the absolute parent watchdog; its completion is not accepted")
        runtime.check_time(output)
        if process.returncode == EXIT_MARGIN:
            raise StudyError("Allocation stop margin reached; accepted rows are kept. Export, then resume in a new "
                             "allocation with --new-allocation --resume")
        if process.returncode != 0:
            raise StudyError(f"Child process exited {process.returncode}; see {folder / 'controller.log'} and error.json")
        complete = folder / "complete.json"
        if not complete.is_file():
            raise StudyError("Child exited 0 without a completion receipt")
        fixed_write(folder / "accepted.json", {"plan_hash": fingerprint(plan), "returncode": 0, "exited_at": utc_now(),
                                               "exited_epoch": time.time(),
                                               "watchdog_deadline_epoch": plan["watchdog_deadline_epoch"],
                                               "complete_sha256": file_hash(complete),
                                               "accepted_by": "parent after an on-time exit"})
        return plan
    except BaseException:
        _stop_process(process)
        if (folder / "complete.json").exists():
            (folder / "complete.json").rename(folder / "unaccepted-complete.json")
        raise


def attempt_accepted(folder):
    """The parent's acceptance of an on-time, successful child exit, checked against the plan and receipt."""
    path = folder / "accepted.json"
    if not path.is_file() or not (folder / "plan.json").is_file() or not (folder / "complete.json").is_file():
        return None
    accepted, plan = read_json(path), read_json(folder / "plan.json")
    deadline = plan.get("watchdog_deadline_epoch")
    if (accepted.get("plan_hash") != fingerprint(plan) or accepted.get("returncode") != 0
            or accepted.get("complete_sha256") != file_hash(folder / "complete.json")
            or accepted.get("watchdog_deadline_epoch") != deadline or not isinstance(deadline, (int, float))
            or not isinstance(accepted.get("exited_epoch"), (int, float))
            or accepted["exited_epoch"] > deadline + CLOCK_TOLERANCE_SECONDS
            or read_json(folder / "complete.json").get("plan_hash") != fingerprint(plan)
            or any((folder / name).exists() for name in ("interrupted.json", "unaccepted-complete.json", "error.json"))):
        raise StudyError(f"Attempt acceptance in {folder} does not match its plan, receipt or exit")
    return plan


def _serial_watchdog(output, count):
    session = runtime.check_time(output)
    budget = session["deadline_epoch"] - time.time() - protocol()["collection"]["watchdog_margin_seconds"]
    return min(60 + 30 * count, budget)


def verify_serial_stage(output, folder, plan):
    """A preflight stage from its raw rows: parent acceptance, exact IDs, bindings, re-parsed responses."""
    if attempt_accepted(folder) != plan:
        raise StudyError(f"Preflight stage {folder.name} was not accepted by the parent")
    done = read_json(folder / "complete.json")
    pool = {"development": data.dev_examples, "longest": data.formal_examples}[plan["purpose"]](output, plan["set"])
    by_id = {e["sample_id"]: e for e in pool}
    if any(sid not in by_id for sid in plan["sample_ids"]):
        raise StudyError(f"Preflight stage {folder.name} lists unknown inputs")
    examples = [by_id[sid] for sid in plan["sample_ids"]]
    rows = read_json(folder / "rows.json")
    collect.check_rows(examples, rows, cell_id=plan["label"], model=INTERFACE[plan["role"]],
                       purpose="preflight_" + plan["purpose"], deployments={plan["deployment_id"]})
    if [r["sample_id"] for r in rows] != plan["sample_ids"]:
        raise StudyError(f"Preflight stage {folder.name} rows are not in plan order")
    failures = sum(r["status"] != "ok" for r in rows)
    if failures or done.get("failures") != failures or done.get("count") != len(rows):
        raise StudyError(f"{failures} preflight requests failed in {folder.name}")
    return rows


# ---------------------------------------------------------------- preflight

def _prior_failures(output, role):
    return sorted(p.parent.name for p in safe(output, "preflight").glob(f"{role}-*/receipt.json")
                  if read_json(p).get("status") != "PASS")


def required_stages(output, role, longest):
    """The exact preflight stages a role must pass, from the frozen inputs and the token receipt."""
    stages = [{"kind": "serial_check", "role": role, "purpose": "development", "set": name,
               "label": f"development-{name}", "sample_ids": [e["sample_id"] for e in data.dev_examples(output, name)]}
              for name in role_dev_sets(role)]
    stages += [{"kind": "serial_check", "role": role, "purpose": "longest", "set": split_cell(cell)[1],
                "label": f"longest-{cell}", "sample_ids": list(longest[cell])} for cell in role_cells(role)]
    return stages


def _evidence(folder):
    return {str(p.relative_to(folder)): file_hash(p) for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name in STAGE_FILES}


def preflight(output_dir, *, retry_failed=False, launcher=None, lister=None):
    """Per role: tokens, health/identity, development pass and longest-input schema checks, verified stop."""
    output = normal(output_dir)
    runtime.validate(output)
    ctx = context(output, check_source=True)
    token_sha, longest, maxima = token_receipt(output, ctx)
    p = protocol()["collection"]
    results = {}
    for role in ROLES:
        receipt_path = safe(output, f"preflight/{role}.json")
        if receipt_path.exists():
            verify_preflight(output, ctx, role, token_sha=token_sha, longest=longest)
            results[role] = "PASS"
            continue
        prior = _prior_failures(output, role)
        if prior and not retry_failed:
            raise StudyError(f"An earlier failed preflight for {role} is preserved ({prior}). Inspect it; use "
                             "--retry-preflight only for a transient cause, otherwise revise the protocol")
        stamp = time.time_ns()
        folder = safe(output, f"preflight/{role}-{stamp}")
        folder.mkdir(parents=True)
        deployment_id = f"preflight-{role}-{stamp}"
        receipt = {"role": role, "settings_hash": ctx["settings_hash"], "token_receipt_sha256": token_sha,
                   "deployment_id": deployment_id, "started_at": utc_now(), "status": "FAIL",
                   "prior_failed_attempts": prior, "max_formal_input_tokens": {c: maxima[c] for c in role_cells(role)}}
        record = {}
        try:
            session = runtime.check_time(output, p["minimum_seconds_to_start_deployment"])
            with runtime.deployment(output, role, launcher=launcher, lister=lister, sink=record) as service:
                if not runtime.engine_command_limited(record):
                    raise StudyError(f"{role} services do not carry exactly the protocol engine limit")
                summaries = {}
                for stage in required_stages(output, role, longest):
                    plan = {**stage, "deployment_id": deployment_id, "endpoint": service["endpoint"],
                            "until_epoch": session["deadline_epoch"] - p["stop_margin_seconds"]}
                    stage_folder = folder / plan["label"]
                    plan = _run_process(output, stage_folder, plan, _serial_watchdog(output, len(plan["sample_ids"])))
                    rows = verify_serial_stage(output, stage_folder, plan)
                    latencies = sorted(r["latency_ms"] for r in rows)
                    summaries[plan["label"]] = {"count": len(rows), "failures": 0, "max_latency_ms": latencies[-1],
                                                "median_latency_ms": latencies[len(latencies) // 2]}
                receipt["health_after_checks"] = upgrade_runtime.health(service["endpoint"], HEALTH_PATH[INTERFACE[role]])
                if not receipt["health_after_checks"].get("ok"):
                    raise StudyError(f"{role} failed its health check after the preflight stages")
                receipt["stages"] = summaries
        except BaseException as exc:
            receipt.update(error=str(exc), error_type=type(exc).__name__, deployment=record, finished_at=utc_now())
            fixed_write(folder / "receipt.json", receipt)
            raise
        receipt.update(status="PASS", deployment=record, finished_at=utc_now(),
                       evidence_dir=str(folder.relative_to(output)), evidence=_evidence(folder))
        fixed_write(folder / "receipt.json", receipt)
        fixed_write(receipt_path, receipt)
        verify_preflight(output, ctx, role, token_sha=token_sha, longest=longest)
        results[role] = "PASS"
    return results


def verify_preflight(output, ctx, role, *, token_sha, longest):
    """A PASS only with the exact required stage inventory, every raw file and a verified stop."""
    path = safe(output, f"preflight/{role}.json")
    if not path.is_file():
        raise StudyError(f"No preflight PASS for {role}; run preflight first")
    receipt = read_json(path)
    folder = safe(output, receipt.get("evidence_dir") or "missing")
    record = receipt.get("deployment", {})
    if (receipt.get("status") != "PASS" or receipt.get("role") != role or receipt.get("settings_hash") != ctx["settings_hash"]
            or receipt.get("token_receipt_sha256") != token_sha or not (folder / "receipt.json").is_file()
            or read_json(folder / "receipt.json") != receipt or record.get("role") != role
            or not _stop_verified(record) or not runtime.engine_command_limited(record)
            or receipt.get("deployment_id") != f"preflight-{role}-{folder.name.rsplit('-', 1)[-1]}"):
        raise StudyError(f"Preflight receipt for {role} is not a verified PASS for these settings")
    expected = required_stages(output, role, longest)
    labels = [stage["label"] for stage in expected]
    present = sorted(p.name for p in folder.iterdir() if p.is_dir())
    inventory = {f"{label}/{name}" for label in labels for name in STAGE_FILES}
    if present != sorted(labels) or set(receipt.get("evidence", {})) != inventory or set(receipt.get("stages", {})) != set(labels):
        raise StudyError(f"Preflight evidence for {role} does not have exactly the required stages {labels}")
    if _evidence(folder) != receipt["evidence"]:
        raise StudyError(f"Preflight evidence for {role} changed after its receipt")
    for stage in expected:
        plan = read_json(folder / stage["label"] / "plan.json")
        if ({k: plan.get(k) for k in stage} != stage or plan.get("deployment_id") != receipt["deployment_id"]
                or plan.get("endpoint") != record.get("endpoint")
                or set(plan) != set(stage) | {"deployment_id", "endpoint", "until_epoch", "watchdog_deadline_epoch"}):
            raise StudyError(f"Preflight stage {stage['label']} of {role} differs from the required inputs")
        verify_serial_stage(output, folder / stage["label"], plan)
    return receipt


def verify_role_evidence(output, ctx, role):
    """Read-only chain for one role: raw tokenizer report, then the preflight receipt and its raw stages."""
    token_sha, longest, _ = verify_token_receipt(output, ctx, roles=(role,))
    verify_preflight(output, ctx, role, token_sha=token_sha, longest=longest)
    return token_sha, file_hash(safe(output, f"preflight/{role}.json"))


# ---------------------------------------------------------------- stop acceptance

def _stop_verified(record):
    stop = record.get("stop_verification")
    groups = sorted(int(p["pid"]) for p in record.get("processes", []))
    return (isinstance(stop, dict) and "error" not in stop and stop.get("owned_groups") == groups and bool(groups)
            and stop.get("gpu_compute_processes") == [])


def _reverification_valid(value, record, deployment_id):
    if (value.get("passed") is not True or value.get("signals_sent") is not False or value.get("deployment_id") != deployment_id
            or value.get("gpu_compute_processes") != [] or value.get("old_allocation") != record.get("allocation")):
        return False
    if value.get("basis") == "recorded_host_process_table":
        return (value.get("alive") == [] and value.get("unverified_group_members") == []
                and value["current_allocation"].get("hostname") == record["allocation"].get("hostname"))
    if value.get("basis") == "scheduler_accounting_of_earlier_job":
        job = value.get("scheduler", {})
        state = (job.get("state") or "").split()
        return (job.get("terminal") is True and job.get("job_id") == str(record["allocation"].get("slurm_job_id"))
                and bool(state) and state[0] in runtime.TERMINAL_JOB_STATES)
    return False


def deployment_record(output, deployment_id):
    folder = safe(output, f"services/{deployment_id}")
    source = folder / "finished.json" if (folder / "finished.json").is_file() else folder / "start.json"
    return read_json(source).get("deployment", {}) if source.is_file() else {}


def deployment_stop(output, deployment_id):
    """('verified'|'reverified', evidence hash) for a validated stop, else (None, None)."""
    folder = safe(output, f"services/{deployment_id}")
    finished = folder / "finished.json"
    if finished.is_file() and _stop_verified(read_json(finished).get("deployment", {})):
        return "verified", file_hash(finished)
    again = folder / "stop-reverified.json"
    if again.is_file() and _reverification_valid(read_json(again), deployment_record(output, deployment_id), deployment_id):
        return "reverified", file_hash(again)
    return None, None


def reconcile_stops(output, *, lister=None, scheduler=None):
    """Passive re-check of formal deployments whose stop was not confirmed. Sends no signal, cancels no job.

    On the recorded host the process table must show no member of the recorded groups
    started at or after the deployment. On another host only affirmative accounting
    evidence counts: the earlier Slurm job must be in a terminal state. The current GPU
    must show no compute process either way. Anything else fails closed.
    """
    services = safe(output, "services")
    current = v3_runtime._allocation()
    done = []
    for folder in sorted(services.glob("*")) if services.is_dir() else []:
        if deployment_stop(output, folder.name)[0] is not None:
            continue
        record = deployment_record(output, folder.name)
        processes = record.get("processes", [])
        if not processes:
            continue
        groups = {int(p["pid"]) for p in processes}
        since = _epoch(record["started_at"])
        old = record.get("allocation", {})
        value = {"at": utc_now(), "deployment_id": folder.name, "signals_sent": False,
                 "old_allocation": old, "current_allocation": current}
        if old.get("hostname") == current["hostname"]:
            alive, older = upgrade_runtime.owned_members(groups, since)
            value.update(basis="recorded_host_process_table", alive=alive, unverified_group_members=older)
            ended = not alive and not older
        else:
            job = (scheduler or runtime.slurm_job_state)(old.get("slurm_job_id"))
            value.update(basis="scheduler_accounting_of_earlier_job", scheduler=job)
            ended = job.get("terminal") is True and job.get("job_id") == str(old.get("slurm_job_id"))
        gpu = (lister or upgrade_runtime.gpu_processes)()
        value.update(gpu_compute_processes=gpu, passed=bool(ended) and not gpu)
        if not value["passed"] or not _reverification_valid(value, record, folder.name):
            write_json(folder / f"stop-reverify-failed-{time.time_ns()}.json", value)
            raise StudyError(f"Deployment {folder.name} is not proven stopped ({value['basis']}: "
                             f"{value.get('alive', value.get('scheduler'))}; GPU {gpu}). Stop its owner normally or "
                             "wait for the earlier job to end, then rerun. Nothing was signalled or cancelled")
        fixed_write(folder / "stop-reverified.json", value)
        done.append(folder.name)
    return done


# ---------------------------------------------------------------- attempts, smoke and completion

def attempt_folders(output, cell_id):
    return sorted(collect.cell_dir(output, cell_id).glob("attempt-*"))


def _attempt_rows(rows, deployment_id):
    return [r for r in rows if r.get("deployment_id") == deployment_id]


def recovery_payload(output, ctx, cell_id, folder, rows):
    """What a successful reconciliation of an unaccepted attempt records; refuses late or unbound work."""
    deployment_id = folder.name[len("attempt-"):]
    own = _attempt_rows(rows, deployment_id)
    plan = read_json(folder / "plan.json") if (folder / "plan.json").is_file() else None
    if plan is None:
        if own:
            raise StudyError(f"{folder} contributed rows but has no plan")
        state = "no_plan"
    elif plan.get("kind") != "collect" or plan.get("cell_id") != cell_id or plan.get("deployment_id") != deployment_id:
        raise StudyError(f"{folder} plan belongs to another cell or deployment")
    elif (folder / "unaccepted-complete.json").exists():
        state = "late_or_failed_exit_after_completion_receipt"
    elif (folder / "partial.json").exists():
        state = "stopped_at_allocation_margin"
    else:
        state = "interrupted"
    deadline = plan.get("watchdog_deadline_epoch") if plan else None
    late = [r["sample_id"] for r in own if deadline is None or _epoch(r["recorded_at"]) > deadline]
    if late:
        raise StudyError(f"{len(late)} rows of {folder.name} were recorded after its watchdog deadline; that work "
                         "cannot be accepted. Preserve the output and report it for a protocol decision")
    stop = None
    if own:
        stop, _ = deployment_stop(output, deployment_id)
        if stop is None:
            raise StudyError(f"The deployment of {folder.name} has no verified stop; it cannot be reconciled yet")
        verify_smoke(output, folder / "smoke.json", cell_id, plan["config_hash"], deployment_id)
    names = ("plan.json", "smoke.json", "complete.json", "unaccepted-complete.json", "partial.json", "interrupted.json",
             "error.json", "controller.log")
    return {"cell_id": cell_id, "deployment_id": deployment_id, "state": state,
            "plan_hash": fingerprint(plan) if plan else None, "watchdog_deadline_epoch": deadline,
            "files": {name: file_hash(folder / name) for name in names if (folder / name).is_file()},
            "rows": len(own), "rows_ids_fingerprint": fingerprint(sorted(r["sample_id"] for r in own)),
            "latest_recorded_at": max((r["recorded_at"] for r in own), default=None), "stop": stop,
            "acknowledged_by": "--resume", "settings_hash": ctx["settings_hash"]}


def attempt_status(output, ctx, cell_id, folder, rows):
    """'accepted', 'recovered' or None (needs an acknowledged reconciliation)."""
    if attempt_accepted(folder) is not None:
        return "accepted"
    path = folder / "recovery.json"
    if path.is_file():
        saved = checked(path, "recovery_hash")
        payload = recovery_payload(output, ctx, cell_id, folder, rows)
        if {k: saved.get(k) for k in payload} != payload:
            raise StudyError(f"Recovery record of {folder} differs from the current evidence")
        return "recovered"
    return None


def unreconciled_attempts(output, ctx, cell_id):
    rows = _rows_or_empty(output, cell_id)
    return [f.name for f in attempt_folders(output, cell_id) if attempt_status(output, ctx, cell_id, f, rows) is None]


def journal_path(directory, model, config):
    """Where the frozen PredictionStore keeps a cell's journal (its layout is read here, never changed)."""
    return Path(directory) / "spool" / fingerprint(str(Path(directory).resolve()))[:16] / fingerprint(config) / f"{model}.jsonl"


def pending_journal_bytes(output, cell_id):
    """Size of a cell's not-yet-sharded journal; nonzero only after a child was killed mid-cell."""
    model = INTERFACE[split_cell(cell_id)[0]]
    directory = collect.cell_dir(output, cell_id)
    config_path = directory / f"predictions/{model}/config.json"
    if not config_path.is_file():
        return 0
    journal = journal_path(directory, model, read_json(config_path))
    return journal.stat().st_size if journal.is_file() else 0


def recover_journal(output, ctx, cell_id):
    """Explicit --resume only: fold fsynced journal rows into shards through the frozen PredictionStore.

    A child killed between periodic shard writes leaves its last rows only in the
    fsynced journal. The stored configuration is verified first; a byte copy of the
    journal is kept as evidence; the frozen constructor then validates every row,
    refuses conflicts, drops only an incomplete final line (recording that in its own
    journal_recovery.json) and writes the rows to shards. Accepted rows are never
    requested again.
    """
    model = INTERFACE[split_cell(cell_id)[0]]
    directory = collect.cell_dir(output, cell_id)
    if not (directory / f"predictions/{model}/config.json").is_file():
        return None
    stored = verify_stored_config(output, ctx, cell_id)
    journal = journal_path(directory, model, stored)
    payload = journal.read_bytes() if journal.is_file() else b""
    digest = hashlib.sha256(payload).hexdigest()
    copy = directory / "journal-recovery" / f"{digest}.jsonl"
    if payload and not copy.exists():
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(payload)
    before = len(load_predictions(directory, model))
    store = PredictionStore(directory, model, stored, spool_dir=directory / "spool")
    if store.journal != journal:
        raise StudyError("The frozen PredictionStore journal layout differs from the expected path; nothing recovered")
    after = len(load_predictions(directory, model))
    if after != len(store.rows) or (journal.is_file() and journal.stat().st_size):
        raise StudyError(f"Journal recovery for {cell_id} did not leave every durable row in shards")
    return {"journal_sha256": digest, "journal_bytes": len(payload),
            "journal_copy": str(copy.relative_to(directory)) if payload else None,
            "sharded_rows_before": before, "sharded_rows_after": after}


def reconcile_attempts(output, ctx, cell_id):
    """Operator-acknowledged (--resume): record each unaccepted attempt that is safe to keep; refuse otherwise.

    Durable journal rows are folded into shards first, so a recovery record binds every
    row its attempt produced and stays valid after collection continues.
    """
    journal = recover_journal(output, ctx, cell_id)
    rows = _rows_or_empty(output, cell_id)
    recorded = []
    for folder in attempt_folders(output, cell_id):
        if attempt_status(output, ctx, cell_id, folder, rows) is None:
            payload = recovery_payload(output, ctx, cell_id, folder, rows)
            fixed_write(folder / "recovery.json", seal({**payload, "reconciled_at": utc_now(), "journal_recovery": journal},
                                                       "recovery_hash"))
            recorded.append(folder.name)
    return recorded


def _rows_or_empty(output, cell_id):
    model = INTERFACE[split_cell(cell_id)[0]]
    directory = collect.cell_dir(output, cell_id)
    return load_predictions(directory, model) if (directory / f"predictions/{model}/config.json").is_file() else []


def verify_smoke(output, path, cell_id, config_hash, deployment_id):
    if not path.is_file():
        raise StudyError(f"Development smoke for {cell_id} in {deployment_id} is missing")
    smoke = read_json(path)
    dev = data.dev_examples(output, dev_set(split_cell(cell_id)[1]))
    if (smoke.get("cell_id") != cell_id or smoke.get("deployment_id") != deployment_id
            or smoke.get("config_hash") != config_hash or smoke.get("passed") is not True):
        raise StudyError(f"Development smoke for {cell_id} in {deployment_id} is not a matching PASS")
    model = INTERFACE[split_cell(cell_id)[0]]
    for key in ("reference", "current"):
        collect.check_rows(dev, smoke[key], cell_id=cell_id, model=model, purpose="smoke", deployments={deployment_id})
    checks = _probability_comparison(dev, smoke["reference"], smoke["current"])
    if checks != smoke.get("checks") or not all(c["passed"] for c in checks.values()):
        raise StudyError(f"Development smoke for {cell_id} fails recomputation")


def completion_payload(output, ctx, cell_id):
    """What a completion receipt must say; None while rows, attempt acceptance or stop confirmation are missing."""
    role, _ = split_cell(cell_id)
    model = INTERFACE[role]
    directory = collect.cell_dir(output, cell_id)
    if not (directory / f"predictions/{model}/config.json").is_file():
        return None
    stored = verify_stored_config(output, ctx, cell_id)
    config_hash = fingerprint(stored)
    rows = load_predictions(directory, model)
    examples = data.cell_examples(output, cell_id)
    if len(rows) < len(examples) or pending_journal_bytes(output, cell_id):
        return None
    token_sha, preflight_sha = verify_role_evidence(output, ctx, role)
    deployments = sorted({r.get("deployment_id") for r in rows})
    evidence = {}
    for deployment_id in deployments:
        start_path = safe(output, f"services/{deployment_id}/start.json")
        start = read_json(start_path) if start_path.is_file() else {}
        if start.get("role") != role or cell_id not in start.get("cells", []) or start.get("deployment_id") != deployment_id:
            raise StudyError(f"Rows of {cell_id} come from an unrecorded or foreign deployment {deployment_id}")
        folder = directory / f"attempt-{deployment_id}"
        state = attempt_status(output, ctx, cell_id, folder, rows) if folder.is_dir() else None
        if state is None:
            return None
        basis, digest = deployment_stop(output, deployment_id)
        if basis is None:
            return None
        plan = read_json(folder / "plan.json")
        if plan.get("config_hash") != config_hash:
            raise StudyError(f"Attempt {folder.name} used another configuration")
        late = [r for r in _attempt_rows(rows, deployment_id) if _epoch(r["recorded_at"]) > plan["watchdog_deadline_epoch"]]
        if late:
            raise StudyError(f"{len(late)} rows of {folder.name} were recorded after its watchdog deadline")
        verify_smoke(output, folder / "smoke.json", cell_id, config_hash, deployment_id)
        evidence[deployment_id] = {"stop": basis, "stop_evidence_sha256": digest, "start_sha256": file_hash(start_path),
                                   "attempt": state, "plan_sha256": file_hash(folder / "plan.json"),
                                   "acceptance_sha256": file_hash(folder / ("accepted.json" if state == "accepted"
                                                                            else "recovery.json")),
                                   "smoke_sha256": file_hash(folder / "smoke.json")}
    collect.check_rows(examples, rows, cell_id=cell_id, model=model, purpose="formal", config_hash=config_hash,
                       revision=stored["runtime"]["model_revision"], deployments=set(deployments))
    return {"cell_id": cell_id, "role": role, "dataset": split_cell(cell_id)[1], "config_hash": config_hash,
            "count": len(rows), "predictions_hash": fingerprint(sorted(rows, key=lambda r: r["sample_id"])),
            "failures": sum(r["status"] != "ok" for r in rows), "deployments": evidence,
            "token_receipt_sha256": token_sha, "preflight_receipt_sha256": preflight_sha, "scored": False}


def completed(output, ctx, cell_id):
    """The verified completion receipt, None when absent; any disagreement with evidence is an error."""
    path = collect.cell_dir(output, cell_id) / "completion.json"
    if not path.is_file():
        return None
    saved = checked(path, "completion_hash")
    payload = completion_payload(output, ctx, cell_id)
    if (payload is None or {k: saved.get(k) for k in payload} != payload
            or set(saved) != set(payload) | {"created_at", "completion_hash"}):
        raise StudyError(f"Completion receipt of {cell_id} differs from its evidence")
    return saved


def finalize(output, ctx, cell_id):
    done = completed(output, ctx, cell_id)
    if done is not None:
        return done
    payload = completion_payload(output, ctx, cell_id)
    if payload is None:
        return None
    value = seal({**payload, "created_at": utc_now()}, "completion_hash")
    fixed_write(collect.cell_dir(output, cell_id) / "completion.json", value)
    return completed(output, ctx, cell_id)


# ---------------------------------------------------------------- formal collection

def _groups(cells):
    groups = []
    for cell in cells:
        role = split_cell(cell)[0]
        if groups and groups[-1][0] == role:
            groups[-1][1].append(cell)
        else:
            groups.append((role, [cell]))
    return groups


def run(output_dir, *, resume=False, launcher=None, lister=None, scheduler=None):
    """Collect every pending cell in protocol order, one model at a time; then freeze."""
    output = normal(output_dir)
    runtime.validate(output)
    ctx = context(output, check_source=True)
    for role in ROLES:
        verify_role_evidence(output, ctx, role)
    p = protocol()["collection"]
    if any(safe(output, name).exists() for name in ("freeze.json", "evaluation-access.json", "evaluation.json")):
        return make_freeze(output, require_existing=True)
    reconcile_stops(output, lister=lister, scheduler=scheduler)
    stuck = {cell: names for cell in CELL_ORDER if (names := unreconciled_attempts(output, ctx, cell))}
    if stuck and not resume:
        raise StudyError(f"Unaccepted attempts are preserved {stuck}. Inspect the diagnostic bundle, then rerun with "
                         "--resume to record their reconciliation; accepted rows are kept and never recollected")
    for cell in stuck:
        print(f"{cell}: reconciled {reconcile_attempts(output, ctx, cell)}", flush=True)
    for cell in CELL_ORDER:
        finalize(output, ctx, cell)
    pending = [cell for cell in CELL_ORDER if completed(output, ctx, cell) is None]
    for role, cells in _groups(pending):
        runtime.check_time(output, p["minimum_seconds_to_start_deployment"])
        deployment_id = f"{role}-{time.time_ns()}"
        folder = safe(output, f"services/{deployment_id}")
        folder.mkdir(parents=True)
        record = {}
        try:
            with runtime.deployment(output, role, launcher=launcher, lister=lister, sink=record) as service:
                fixed_write(folder / "start.json", {"deployment_id": deployment_id, "role": role, "cells": cells,
                                                    "at": utc_now(), "deployment": record,
                                                    "binding_sha256": file_hash(safe(output, "runtime/binding.json"))})
                if not runtime.engine_command_limited(record):
                    raise StudyError(f"{role} services do not carry exactly the protocol engine limit")
                for cell in cells:
                    config = cell_config_for(output, ctx, cell)
                    session = runtime.check_time(output, p["stop_margin_seconds"] + 60)
                    attempt = collect.cell_dir(output, cell) / f"attempt-{deployment_id}"
                    plan = {"kind": "collect", "cell_id": cell, "endpoint": service["endpoint"],
                            "deployment_id": deployment_id, "config_hash": fingerprint(config),
                            "until_epoch": session["deadline_epoch"] - p["stop_margin_seconds"]}
                    try:
                        _run_process(output, attempt, plan,
                                     session["deadline_epoch"] - time.time() - p["watchdog_margin_seconds"])
                    except BaseException as exc:
                        fixed_write(attempt / "interrupted.json", {"at": utc_now(), "error": str(exc),
                                                                   "type": type(exc).__name__})
                        raise
                    done = read_json(attempt / "complete.json")
                    rows = collect.stored_rows(output, cell, INTERFACE[role])
                    if not done.get("complete") or len(rows) != len(data.cell_examples(output, cell)):
                        raise StudyError(f"{cell} child exited without every input stored")
                    failures = sum(r["status"] != "ok" for r in rows)
                    print(f"{cell}: {len(rows)} terminal predictions ({done['persisted'] - done['before']} new), "
                          f"{failures} terminal errors", flush=True)
        finally:
            fixed_write(folder / "finished.json", {"at": utc_now(), "deployment_id": deployment_id, "role": role,
                                                   "cells": cells, "stop_verification": record.get("stop_verification"),
                                                   "deployment": record})
        for cell in cells:
            if finalize(output, ctx, cell) is None:
                raise StudyError(f"{cell} is collected, but its deployment stop or attempt acceptance is not "
                                 "confirmed. Inspect the node, then rerun (stops are re-checked passively)")
    return make_freeze(output)


# ---------------------------------------------------------------- freeze and evaluation

def _formal_rows(output, cell_id):
    return collect.stored_rows(output, cell_id, INTERFACE[split_cell(cell_id)[0]])


def freeze_payload(output, ctx, completions):
    return {"schema_version": 1, "protocol": PROTOCOL_ID, "protocol_hash": protocol_hash(),
            "settings_hash": ctx["settings_hash"], "manifest_hash": ctx["settings"]["manifest_hash"],
            "completions": {cell: done["completion_hash"] for cell, done in completions.items()},
            "references_hash": fingerprint(read_json(safe(output, "references/provenance.json"))),
            "scope": protocol()["scope"]}


def all_completions(output, ctx):
    completions = {}
    for cell in CELL_ORDER:
        done = completed(output, ctx, cell)
        if done is None:
            raise StudyError("Every one of the eight cells must be complete and stop-verified before the freeze; "
                             "test gold stays unread")
        completions[cell] = done
    return completions


def make_freeze(output, *, require_existing=False, offline=False):
    """Bind all eight verified completions and the twelve references before any gold read."""
    output = normal(output)
    if offline:
        _, binding = runtime.verify_binding_offline(output, settings(output))
    else:
        _, binding, _ = runtime.verify_binding(output)
    ctx = context(output, binding=binding)
    payload = freeze_payload(output, ctx, all_completions(output, ctx))
    path = safe(output, "freeze.json")
    if path.exists():
        saved = checked(path, "freeze_hash")
        if {k: v for k, v in saved.items() if k not in ("created_at", "freeze_hash")} != payload:
            raise StudyError("The freeze differs from the completed evidence; preserve this run")
        frozen_at = _epoch(saved["created_at"])
        for cell in CELL_ORDER:
            if any(_epoch(r["recorded_at"]) > frozen_at for r in _formal_rows(output, cell)):
                raise StudyError(f"A {cell} prediction was recorded after the freeze")
        return saved
    if require_existing or safe(output, "evaluation-access.json").exists():
        raise StudyError("An existing freeze is required")
    value = seal({**payload, "created_at": utc_now()}, "freeze_hash")
    write_json(path, value)
    return value


def access_payload(output, freeze, completions):
    return {"freeze_hash": freeze["freeze_hash"], "manifest_hash": freeze["manifest_hash"],
            "completion_hashes": {cell: done["completion_hash"] for cell, done in completions.items()},
            "reference_hashes": {cell: fingerprint(rows) for cell, rows in data.references(output).items()}}


def score(output, completions, gold):
    """Pure scoring of all 20 cells and the 40 descriptive paired differences from verified inputs."""
    p = protocol()
    references = data.references(output)
    provenance = read_json(safe(output, "references/provenance.json"))
    matrix, outcomes = {}, {}
    for dataset in DATASETS:
        examples = data.formal_examples(output, dataset)
        labels = p["datasets"][dataset]["labels"]
        for model in MODELS:
            cell = f"{model}-{dataset}"
            if cell in REFERENCE_CELLS:
                rows = {r["sample_id"]: r for r in references[cell]}
                source = {"kind": "historical_reference", **provenance[cell]}
            else:
                rows = {r["sample_id"]: r for r in _formal_rows(output, cell)}
                source = {"kind": "new_retrospective", "completion_hash": completions[cell]["completion_hash"],
                          "predictions_hash": completions[cell]["predictions_hash"]}
            metric, outcome = stats.direct(examples, rows, gold[dataset], labels)
            if cell in REFERENCE_CELLS and metric["correct"] != p["references"][cell]["accepted_correct"]:
                raise StudyError(f"Historical {cell} scores {metric['correct']}, not its accepted "
                                 f"{p['references'][cell]['accepted_correct']}")
            matrix[cell] = {"model": model, "dataset": dataset, **metric, "source": source}
            outcomes[cell] = outcome
    comparisons, e = {}, p["evaluation"]
    for dataset in DATASETS:
        examples = data.formal_examples(output, dataset)
        for a, b in combinations(MODELS, 2):
            comparisons[f"{dataset}:{a}-minus-{b}"] = {
                "dataset": dataset, "candidate": a, "reference": b,
                **stats.paired_difference(examples, outcomes[f"{a}-{dataset}"], outcomes[f"{b}-{dataset}"],
                                          cluster=p["datasets"][dataset]["cluster"],
                                          repetitions=e["bootstrap_repetitions"], seed=e["bootstrap_seed"])}
    return matrix, comparisons


def result_body(output, freeze, access, completions, gold, gold_files):
    p = protocol()
    matrix, comparisons = score(output, completions, gold)
    return {"status": "complete", "protocol": PROTOCOL_ID, "freeze_hash": freeze["freeze_hash"],
            "access_hash": access["access_hash"], "matrix": matrix, "comparisons_descriptive": comparisons,
            "gold_files": gold_files, "new_predictions": sum(c["count"] for c in completions.values()),
            "scope": p["scope"], "retrospective_notice": p["retrospective_notice"],
            "multiplicity": p["evaluation"]["multiplicity"], "claim_limit": p["evaluation"]["claim_limit"]}


def evaluate(output_dir):
    """First gold read: full-denominator accuracy for all 20 cells and 40 paired descriptive intervals."""
    output = normal(output_dir)
    freeze = make_freeze(output, require_existing=True)
    _, binding, _ = runtime.verify_binding(output)
    ctx = context(output, binding=binding)
    completions = all_completions(output, ctx)
    payload = access_payload(output, freeze, completions)
    access_path, result_path = safe(output, "evaluation-access.json"), safe(output, "evaluation.json")
    if access_path.exists():
        access = checked(access_path, "access_hash")
        if {k: access.get(k) for k in payload} != payload:
            raise StudyError("Evaluation access is bound to different evidence")
        if not result_path.exists():
            raise StudyError("Evaluation access exists without a result; preserve the partial evidence for review")
    else:
        if result_path.exists():
            raise StudyError("An evaluation exists without its access receipt")
        access = seal({**payload, "created_at": utc_now()}, "access_hash")
        write_json(access_path, access)
    gold, gold_files = data.gold_for_evaluation(output, freeze, access)
    result = result_body(output, freeze, access, completions, gold, gold_files)
    if result_path.exists():
        old = checked(result_path, "evaluation_hash")
        result = seal({**result, "created_at": old["created_at"]}, "evaluation_hash")
        if result != old:
            raise StudyError("Recomputed evaluation differs from the recorded one; refusing to overwrite")
        return old
    result = seal({**result, "created_at": utc_now()}, "evaluation_hash")
    write_json(result_path, result)
    return result


# ---------------------------------------------------------------- status and export

def _receipt_state(path, key, check=None):
    if not path.is_file():
        return "absent"
    try:
        value = checked(path, key)
        if check is not None:
            check(value)
        return "valid"
    except (StudyError, OSError, KeyError, ValueError, TypeError) as exc:
        return f"invalid: {exc}"


def status(output_dir):
    """Read-only progress with validated receipts; opens no gold and starts nothing."""
    output = normal(output_dir)
    result = {"protocol": PROTOCOL_ID, "output": str(output), "problems": [],
              "notice": "Read-only; no gold, model, GPU or scheduler access"}
    try:
        ctx = context(output, binding=read_json(safe(output, "runtime/binding.json"))
                      if safe(output, "runtime/binding.json").is_file() else {})
    except (StudyError, OSError, KeyError, ValueError) as exc:
        result.update(prepared=False)
        result["problems"].append(str(exc))
        return result
    result.update(prepared=True, source_commit=ctx["settings"]["source"]["commit"],
                  runtime_bound=bool(ctx["binding"]))
    session_path = safe(output, "runtime/session.json")
    if session_path.is_file():
        session = read_json(session_path)
        result["session"] = {"allocation": session.get("allocation"),
                             "seconds_left": round(session.get("deadline_epoch", 0) - time.time())}
    result["preflight"] = {}
    for role in ROLES:
        path = safe(output, f"preflight/{role}.json")
        if not path.is_file():
            result["preflight"][role] = "not run"
            continue
        try:
            verify_role_evidence(output, ctx, role)
            result["preflight"][role] = "PASS (verified)"
        except (StudyError, OSError, KeyError, ValueError) as exc:
            result["preflight"][role] = f"invalid: {exc}"
            result["problems"].append(f"preflight {role}: {exc}")
    cells, completions = {}, {}
    for cell in CELL_ORDER:
        entry = {"expected": len(data.cell_examples(output, cell)), "persisted": 0, "terminal_errors": 0, "complete": False}
        try:
            rows = _rows_or_empty(output, cell)
            entry.update(persisted=len(rows), terminal_errors=sum(r["status"] != "ok" for r in rows),
                         unrecovered_journal_bytes=pending_journal_bytes(output, cell))
            if ctx["binding"]:
                entry["unaccepted_attempts"] = unreconciled_attempts(output, ctx, cell)
                done = completed(output, ctx, cell)
                entry["complete"] = done is not None
                if done is not None:
                    completions[cell] = done
        except (StudyError, OSError, KeyError, ValueError) as exc:
            entry["problem"] = str(exc)
            result["problems"].append(f"{cell}: {exc}")
        cells[cell] = entry
    result["cells"] = cells
    result["pending_cells"] = [c for c, v in cells.items() if not v["complete"]]

    def freeze_matches(value):
        if len(completions) != len(CELL_ORDER) or {k: v for k, v in value.items()
                                                   if k not in ("created_at", "freeze_hash")} != freeze_payload(output, ctx, completions):
            raise StudyError("freeze does not bind the current completions")

    result["freeze"] = _receipt_state(safe(output, "freeze.json"), "freeze_hash", freeze_matches)
    freeze_hash = read_json(safe(output, "freeze.json")).get("freeze_hash") if result["freeze"] == "valid" else None

    def bound_to_freeze(value):
        if value.get("freeze_hash") != freeze_hash or freeze_hash is None:
            raise StudyError("not bound to the valid freeze")

    result["evaluation_access"] = _receipt_state(safe(output, "evaluation-access.json"), "access_hash", bound_to_freeze)
    result["evaluation"] = _receipt_state(safe(output, "evaluation.json"), "evaluation_hash", bound_to_freeze)
    for key in ("freeze", "evaluation_access", "evaluation"):
        if result[key].startswith("invalid"):
            result["problems"].append(f"{key}: {result[key]}")
    return result


def _include(path):
    name = path.name
    return (path.is_file() and not path.is_symlink() and name not in {".active.lock", "bundle_manifest.json"}
            and not name.endswith(".tmp") and not name.startswith("interactive-inputs-"))


def export(output_dir):
    """Diagnostic/result bundle: inputs, manifests, raw responses, receipts, logs; no weights, envs or caches."""
    output = normal(output_dir)
    destination = safe(output, "exports/direct_matrix_bundle.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files, skipped = [], {}
    for parent, dirs, names in os.walk(output, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXPORT_SKIP_DIRS and not d.startswith((".venv", "venv"))
                         and not (Path(parent) / d).is_symlink())
        for name in sorted(names):
            path = Path(parent) / name
            if _include(path):
                files.append(path)
            elif name.startswith("interactive-inputs-") and path.is_file():
                skipped[str(path.relative_to(output))] = {"sha256": file_hash(path), "bytes": path.stat().st_size,
                                                          "reason": "copy of data/formal and data/dev inputs"}
    manifest = {"protocol": PROTOCOL_ID, "created_at": utc_now(),
                "files": {str(p.relative_to(output)): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in files},
                "excluded_duplicates": skipped}
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(output))
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    temporary.replace(destination)
    return destination
