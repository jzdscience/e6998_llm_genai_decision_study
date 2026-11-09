"""Prepare, preflight, measure, freeze and score the upgrade-versus-fallback study.

The sealed ANLI gold is read only in ``evaluate`` and only after every one of the
36 fixed cells has a verified completion receipt and the freeze is written.
"""
from collections import Counter, defaultdict
import hashlib
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np

from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl, utc_now
from ..service_study.measurement import make_schedule, summarize, verify_cell
from ..service_study.resources import hardware
from . import runtime
from .cell import SERIAL_FILES
from .identity import (CONDITIONS, GOLD_SHA256, PROJECT, ROLES, SERVICES, fixed_write, load_history, load_inputs,
                       load_policy, protocol, safe, source_identity)

FAILURE_STATUSES = ("error", "timeout", "deadline", "http_rejected")
PREFLIGHT_EVIDENCE = ("serial/plan.json", "serial/rows.json", "serial/complete.json", "concurrent/plan.json",
                      "concurrent/complete.json", "concurrent/traces.jsonl", "concurrent/occupancy.jsonl",
                      "concurrent/resources.jsonl")
RECEIPT_KEYS = {"condition", "settings_hash", "started_at", "status", "prior_failed_attempts", "token_receipt_sha256",
                "serial", "concurrent", "health", "gpu_processes_after_concurrent", "engine_log_before_stop",
                "deployment", "finished_at", "evidence_dir", "evidence"}


def _bytehash(value):
    return hashlib.sha256(canonical(value) + b"\n").hexdigest()


def planned_order(p):
    """Within each repeat: rates in a seeded order; the three conditions of one rate run back to back."""
    order = []
    for repeat in range(p["repeats"]):
        rates = list(p["rates_per_second"])
        random.Random(f"{p['seed']}-rates-{repeat}").shuffle(rates)
        for rate in rates:
            conditions = list(p["conditions"])
            random.Random(f"{p['seed']}-conditions-{repeat}-{rate}").shuffle(conditions)
            order.extend({"cell_id": f"r{repeat}-{condition}-rate{rate}", "repeat": repeat, "condition": condition,
                          "rate": rate, "schedule_file": f"schedules/r{repeat}-rate{rate}.json"} for condition in conditions)
    return order


def schedules_for(p, workload):
    return {f"schedules/r{repeat}-rate{rate}.json": make_schedule(workload, rate, p["duration_seconds"], p["seed"] + repeat * 1000 + rate)
            for repeat in range(p["repeats"]) for rate in p["rates_per_second"]}


def prepare(v1_dir, v3_dir, size_control_dir, output_dir):
    output = Path(output_dir).absolute()
    for value in (v1_dir, v3_dir, size_control_dir, PROJECT):
        root = Path(value).absolute()
        if output == root or output in root.parents or root in output.parents:
            raise StudyError("Study output must not overlap an input or source root")
    p, source = protocol(), source_identity()
    workload, warmup = load_inputs(v3_dir)
    policy, policy_binding = load_policy(v3_dir)
    history, history_binding = load_history(v3_dir, size_control_dir, workload)
    concurrent = p["feasibility_preflight"]["concurrent_check"]
    contents = {"workload.json": workload, "warmup.json": warmup,
                "policy.json": {"policy": policy, "binding": policy_binding}, "history.json": history,
                "preflight-schedule.json": make_schedule(warmup, concurrent["rate_per_second"], concurrent["duration_seconds"], p["seed"] + 424242),
                **schedules_for(p, workload)}
    settings_value = {"protocol": p, "source": source, "v1_dir": str(Path(v1_dir).absolute()),
                      "v3_dir": str(Path(v3_dir).absolute()), "size_control_dir": str(Path(size_control_dir).absolute()),
                      "policy": policy_binding, "history": history_binding,
                      "file_sha256": {name: _bytehash(value) for name, value in contents.items()},
                      "order": planned_order(p)}
    contents["settings.json"] = settings_value
    for name, value in contents.items():
        path = safe(output, name)
        if path.exists() and read_json(path) != value:
            raise StudyError(f"Preparation conflicts with existing evidence: {name}")
    for name, value in contents.items():
        fixed_write(safe(output, name), value)
    order = settings_value["order"]
    return {"status": "PREPARED", "settings_hash": fingerprint(settings_value), "cells": len(order),
            "offers_planned": sum(len(contents[c["schedule_file"]]) for c in order),
            "offer_window_minutes": len(order) * p["duration_seconds"] / 60,
            "historical_deferrals": history_binding["historical_deferrals"],
            "notice": "Inputs, policy and verified historical predictions only; no GPU call or gold read"}


def settings(output_dir, *, check_source=False):
    value = read_json(safe(output_dir, "settings.json"))
    if check_source and value["source"] != source_identity():
        raise StudyError("Study source differs from the prepared identity")
    if value["protocol"] != protocol():
        raise StudyError("Study protocol changed after preparation")
    if value["order"] != planned_order(value["protocol"]):
        raise StudyError("Fixed cell order changed")
    expected = {"workload.json", "warmup.json", "policy.json", "history.json", "preflight-schedule.json",
                *{c["schedule_file"] for c in value["order"]}}
    if set(value["file_sha256"]) != expected:
        raise StudyError("Prepared file inventory changed")
    for name, digest in value["file_sha256"].items():
        if file_hash(safe(output_dir, name)) != digest:
            raise StudyError(f"Changed prepared input: {name}")
    return value


def _mode(condition, purpose):
    if condition != "cascade":
        return "formal"
    return {"serial": "both_models", "concurrent": "forced_fallback", "formal": "formal"}[purpose]


def _overlap_seconds(rows):
    """Total time during which a fallback stage ran while a front stage of another offer also ran."""
    fronts, backs = [], []
    for row in rows:
        result = row.get("result") or {}
        base = row.get("worker_start_seconds")
        if base is None:
            continue
        for stage in result.get("stages") or []:
            span = (base + stage["started_ms"] / 1000, base + stage["ended_ms"] / 1000, row["offer_id"])
            (backs if stage["role"] == "qwen9b" else fronts).append(span)
    total = 0.0
    for b0, b1, b_id in backs:
        for f0, f1, f_id in fronts:
            if f_id != b_id:
                total += max(0.0, min(b1, f1) - max(b0, f0))
    return total


def _prior_failures(output, condition):
    return sorted(str(p.parent.relative_to(output)) for p in Path(output).glob(f"preflight/{condition}-*/receipt.json")
                  if read_json(p).get("status") != "PASS")


def _watchdog_seconds(p, duration):
    return duration + p["drain_seconds"] + p["watchdog_grace_seconds"]


def _serial_plan(config, warmup_ids, condition, purpose, endpoints, pids):
    """A serial stage (preflight serial check or cell warmup) in the owned child process."""
    p, files = config["protocol"], config["file_sha256"]
    return {"kind": "serial", "purpose": purpose, "condition": condition, "mode": _mode(condition, "serial"),
            "settings_hash": fingerprint(config), "workload_file": "warmup.json", "workload_sha256": files["warmup.json"],
            "sample_ids": warmup_ids, "policy_file": "policy.json", "policy_sha256": files["policy.json"],
            "request_timeout_seconds": p["request_timeout_seconds"],
            "watchdog_seconds": len(warmup_ids) * p["request_timeout_seconds"] + p["watchdog_grace_seconds"],
            "endpoints": endpoints, "service_pids": pids}


def _concurrent_plan(config, condition, endpoints, pids):
    p, files = config["protocol"], config["file_sha256"]
    concurrent = p["feasibility_preflight"]["concurrent_check"]
    return {"cell_id": f"preflight-{condition}", "condition": condition, "mode": _mode(condition, "concurrent"),
            "settings_hash": fingerprint(config),
            "protocol": {**p, "duration_seconds": concurrent["duration_seconds"], "workers": concurrent["workers"],
                         "max_inflight": concurrent["max_inflight"]},
            "schedule_file": "preflight-schedule.json", "schedule_sha256": files["preflight-schedule.json"],
            "workload_file": "warmup.json", "workload_sha256": files["warmup.json"],
            "policy_file": "policy.json", "policy_sha256": files["policy.json"],
            "watchdog_seconds": _watchdog_seconds(p, concurrent["duration_seconds"]),
            "endpoints": endpoints, "service_pids": pids}


def verify_stage(folder, plan_hash):
    """Completion receipt of a serial stage; None when the stage never completed."""
    path = folder / "complete.json"
    if not path.is_file():
        return None
    record = read_json(path)
    if (record.get("plan_hash") != plan_hash
            or record.get("self_hash") != fingerprint({k: v for k, v in record.items() if k != "self_hash"})):
        raise StudyError("Serial-stage completion identity mismatch")
    if set(record.get("files", {})) != set(SERIAL_FILES):
        raise StudyError("Serial-stage evidence inventory mismatch")
    for name, digest in record["files"].items():
        source = folder / name
        if source.is_symlink() or not source.is_file() or file_hash(source) != digest:
            raise StudyError(f"Serial-stage evidence changed or missing: {name}")
    if fingerprint(read_json(folder / "plan.json")) != plan_hash:
        raise StudyError("Saved serial-stage plan mismatch")
    return record


def _serial_stage(output, folder, plan):
    folder.mkdir(parents=True)
    _run_process(output, folder, plan)
    if verify_stage(folder, fingerprint(plan)) is None:
        raise StudyError("A serial stage exited without a completion receipt")
    return read_json(folder / "rows.json")["rows"]


def _verified_serial(output, config, folder, condition, purpose):
    """Rows of a completed serial stage whose plan, receipt and inputs all match the fixed settings."""
    warmup_ids = [e["sample_id"] for e in read_json(output / "warmup.json")]
    plan = read_json(folder / "plan.json")
    if plan != _serial_plan(config, warmup_ids, condition, purpose, plan.get("endpoints"), plan.get("service_pids")):
        raise StudyError(f"{purpose} plan differs from the fixed settings")
    if verify_stage(folder, fingerprint(plan)) is None:
        raise StudyError(f"{purpose} has no completion receipt")
    rows = read_json(folder / "rows.json")["rows"]
    if [r.get("sample_id") for r in rows] != warmup_ids:
        raise StudyError(f"{purpose} did not cover the development inputs in order")
    return plan, rows


def _schedule_rows(folder, schedule, what):
    rows = sorted(read_jsonl(folder / "traces.jsonl"), key=lambda r: r["offer_id"])
    if [{k: row.get(k) for k in ("offer_id", "sample_id", "arrival_seconds")} for row in rows] != schedule:
        raise StudyError(f"{what} do not cover the exact offered schedule")
    return rows


def _serial_summary(rows):
    bad = [r for r in rows if r["status"] != "ok" or any(s["status"] != "ok" for s in r["stages"])]
    return {"count": len(rows), "failures": len(bad),
            "stage_calls": dict(Counter(s["role"] for r in rows for s in r["stages"]))}


def _concurrent_summary(condition, rows, report):
    statuses = Counter(r["status"] for r in rows)
    summary = {"report": report, "statuses": dict(statuses), "engine_overlap_seconds": _overlap_seconds(rows),
               "label": "preflight_forced_deferral" if condition == "cascade" else "preflight_concurrent"}
    if condition == "cascade":
        routes = [(r.get("result") or {}).get("route") or {} for r in rows if r["status"] == "ok"]
        summary["route_reasons"] = dict(Counter(str(route.get("reason")) for route in routes))
        summary["fallback_called"] = sum(bool(route.get("fallback_called")) for route in routes)
    return summary


def _concurrent_problems(condition, summary):
    """Why a concurrent check cannot pass, judged from its own rows."""
    problems = []
    failures = {k: v for k, v in summary["statuses"].items() if k in FAILURE_STATUSES}
    if failures:
        problems.append(f"Concurrent development check had failures {failures}")
    if condition == "cascade":
        ok = summary["statuses"].get("ok", 0)
        if summary["route_reasons"] != {"preflight_forced_deferral": ok} or summary["fallback_called"] != ok:
            problems.append("The cascade concurrent check did not send every request through both engines")
        if summary["engine_overlap_seconds"] <= 0:
            problems.append("The two engines never worked at the same time during the concurrent check")
    return problems


def _sequence_limit(command):
    values = [command[i + 1] for i, part in enumerate(command[:-1]) if part == "--max-num-seqs"]
    return int(values[0]) if len(values) == 1 and str(values[0]).isdigit() else None


def engines_limited(deployment, condition, limit):
    """Whether a deployment record shows this condition's roles, each vLLM engine started with the sequence limit."""
    roles = (deployment or {}).get("roles") or []
    if [r.get("role") for r in roles] != list(SERVICES[condition]):
        return False
    for role in roles:
        engines = [p for p in role.get("processes") or [] if str(p.get("name", "")).endswith("-engine")]
        if not engines or any(_sequence_limit(p.get("command") or []) != limit for p in engines):
            return False
    return True


def verify_tokens(output, config):
    """The token receipt and, behind it, each role's raw tokenizer report on all 3,290 inputs."""
    path = safe(output, "preflight/tokens.json")
    if not path.is_file():
        raise StudyError("Run the development preflight first; the token receipt is missing")
    tokens = read_json(path)
    if set(tokens) != {"settings_hash", "at", "reports"} or tokens["settings_hash"] != fingerprint(config):
        raise StudyError("Token receipt belongs to other settings or has an unexpected shape")
    if set(tokens["reports"]) != set(ROLES):
        raise StudyError("Token receipt does not cover every role")
    examples = read_json(output / "workload.json") + read_json(output / "warmup.json")
    for role, binding in tokens["reports"].items():
        runtime.verify_token_report(output, role, binding, examples)
    return file_hash(path)


def verify_preflight(output, config, condition, *, token_sha256=None):
    """Check a PASS receipt against every file it summarizes; return the receipt's SHA-256."""
    output = Path(output)
    path = safe(output, f"preflight/{condition}.json")
    if not path.is_file():
        raise StudyError(f"Run the development preflight first; missing PASS receipt for {condition}")
    receipt = read_json(path)
    if (set(receipt) != RECEIPT_KEYS or receipt["condition"] != condition or receipt["status"] != "PASS"
            or receipt["settings_hash"] != fingerprint(config)):
        raise StudyError(f"Preflight for {condition} is not a complete PASS for these settings")
    if receipt["token_receipt_sha256"] != (token_sha256 or verify_tokens(output, config)):
        raise StudyError(f"Preflight for {condition} is bound to another token receipt")
    if not re.fullmatch(rf"preflight/{re.escape(condition)}-[0-9]+", receipt["evidence_dir"]):
        raise StudyError(f"Preflight for {condition} names an unexpected evidence folder")
    folder = safe(output, receipt["evidence_dir"])
    if folder.is_symlink() or not folder.is_dir() or read_json(folder / "receipt.json") != receipt:
        raise StudyError(f"Preflight folder for {condition} does not hold this receipt")
    if set(receipt["evidence"]) != set(PREFLIGHT_EVIDENCE):
        raise StudyError(f"Preflight evidence inventory for {condition} changed")
    for name, digest in receipt["evidence"].items():
        source = safe(folder, name)
        if source.is_symlink() or not source.is_file() or file_hash(source) != digest:
            raise StudyError(f"Preflight evidence for {condition} changed or missing: {name}")
    serial_plan, serial = _verified_serial(output, config, folder / "serial", condition, "preflight_serial")
    if _serial_summary(serial) != receipt["serial"] or receipt["serial"]["failures"]:
        raise StudyError(f"Preflight serial summary for {condition} differs from its rows")
    plan = read_json(folder / "concurrent/plan.json")
    if (plan != _concurrent_plan(config, condition, plan.get("endpoints"), plan.get("service_pids"))
            or (plan["endpoints"], plan["service_pids"]) != (serial_plan["endpoints"], serial_plan["service_pids"])):
        raise StudyError(f"Preflight concurrent plan for {condition} differs from the fixed settings")
    complete = verify_cell(folder / "concurrent", fingerprint(plan))
    if complete is None:
        raise StudyError(f"Preflight concurrent check for {condition} has no completion receipt")
    rows = _schedule_rows(folder / "concurrent", read_json(output / "preflight-schedule.json"),
                          f"Preflight concurrent traces for {condition}")
    p = plan["protocol"]
    if any(complete["report"].get(k) != v for k, v in summarize(rows, p["duration_seconds"], p["slo_ms"]).items()):
        raise StudyError(f"Preflight concurrent summary for {condition} differs from raw responses")
    summary = _concurrent_summary(condition, rows, complete["report"])
    if summary != receipt["concurrent"] or _concurrent_problems(condition, summary):
        raise StudyError(f"Preflight concurrent receipt for {condition} differs from its rows or does not pass")
    deployment = receipt["deployment"]
    stop = deployment.get("stop_verification") or {}
    if ([r.get("role") for r in deployment.get("roles", [])] != list(SERVICES[condition])
            or "error" in stop or stop.get("gpu_compute_processes") != []
            or stop.get("owned_groups") != sorted(plan["service_pids"])):
        raise StudyError(f"Preflight deployment record for {condition} is incomplete or its cleanup was not confirmed")
    if not engines_limited(deployment, condition, config["protocol"]["engine"]["max_num_seqs"]):
        raise StudyError(f"Preflight deployment record for {condition} does not show the engine sequence limit")
    return file_hash(path)


def preflight(output_dir, *, retry_failed=False, launcher=None, lister=None):
    output = Path(output_dir).absolute()
    config = settings(output, check_source=True)
    p = config["protocol"]
    workload, warmup = read_json(output / "workload.json"), read_json(output / "warmup.json")
    warmup_ids = [e["sample_id"] for e in warmup]
    schedule = read_json(output / "preflight-schedule.json")
    token_path = safe(output, "preflight/tokens.json")
    if not token_path.exists():
        fixed_write(token_path, {"settings_hash": fingerprint(config), "at": utc_now(),
                                 "reports": runtime.token_check(output, workload + warmup)})
    token_sha256 = verify_tokens(output, config)
    results = {}
    for condition in p["conditions"]:
        receipt_path = safe(output, f"preflight/{condition}.json")
        if receipt_path.exists():
            verify_preflight(output, config, condition, token_sha256=token_sha256)
            results[condition] = read_json(receipt_path)
            continue
        prior = _prior_failures(output, condition)
        if prior and not retry_failed:
            raise StudyError(f"Earlier failed preflight for {condition} is preserved ({prior}). Inspect it; rerun with "
                             "--retry-preflight only for a transient cause, otherwise revise the protocol")
        folder = safe(output, f"preflight/{condition}-{time.time_ns()}")
        folder.mkdir(parents=True)
        receipt = {"condition": condition, "settings_hash": fingerprint(config), "started_at": utc_now(),
                   "status": "FAIL", "prior_failed_attempts": prior, "token_receipt_sha256": token_sha256}
        record = {}
        try:
            runtime.check_time(output, 1800)
            with runtime.deployment(output, condition, launcher=launcher, lister=lister, sink=record) as service:
                serial = _serial_stage(output, folder / "serial", _serial_plan(
                    config, warmup_ids, condition, "preflight_serial", service["endpoints"], service["pids"]))
                receipt["serial"] = _serial_summary(serial)
                if receipt["serial"]["failures"]:
                    raise StudyError(f"{receipt['serial']['failures']} serial development checks failed")
                # Same owned child process and absolute parent watchdog as a formal cell;
                # a socket timeout alone does not bound the concurrent run.
                plan = _concurrent_plan(config, condition, service["endpoints"], service["pids"])
                (folder / "concurrent").mkdir()
                _run_process(output, folder / "concurrent", plan)
                complete = verify_cell(folder / "concurrent", fingerprint(plan))
                if complete is None:
                    raise StudyError("The concurrent check exited without a completion receipt")
                rows = _schedule_rows(folder / "concurrent", schedule, "The concurrent check's traces")
                receipt["concurrent"] = _concurrent_summary(condition, rows, complete["report"])
                receipt["health"] = {role: runtime.health(endpoint) for role, endpoint in service["endpoints"].items()}
                receipt["gpu_processes_after_concurrent"] = (lister or runtime.gpu_processes)()
                receipt["engine_log_before_stop"] = runtime.engine_log_facts(
                    [proc for role in record["roles"] for proc in role["processes"]])
                problems = _concurrent_problems(condition, receipt["concurrent"])
                if not all(h.get("ok") for h in receipt["health"].values()):
                    problems.append("A service failed its health check after the concurrent check")
                if any(facts.get("failure_mentions") for facts in receipt["engine_log_before_stop"].values()):
                    problems.append("Engine logs report out-of-memory or engine failure before shutdown")
                if problems:
                    raise StudyError("; ".join(problems))
            receipt.update(status="PASS", deployment=record, finished_at=utc_now(),
                           evidence_dir=str(folder.relative_to(output)),
                           evidence={name: file_hash(folder / name) for name in PREFLIGHT_EVIDENCE})
        except BaseException as exc:
            receipt.update(error=str(exc), error_type=type(exc).__name__, deployment=record, finished_at=utc_now())
            fixed_write(folder / "receipt.json", receipt)
            raise
        fixed_write(folder / "receipt.json", receipt)
        fixed_write(receipt_path, receipt)
        verify_preflight(output, config, condition, token_sha256=token_sha256)
        results[condition] = receipt
    return {condition: receipt["status"] for condition, receipt in results.items()}


def preflight_hashes(output, config):
    """Verified SHA-256 of each condition's PASS receipt; refuses missing or changed evidence."""
    token_sha256 = verify_tokens(output, config)
    return {condition: verify_preflight(output, config, condition, token_sha256=token_sha256)
            for condition in config["protocol"]["conditions"]}


def _formal_plan(config, cell, receipts, service, start_path, warm_path, allocation, output):
    p, files = config["protocol"], config["file_sha256"]
    return {**cell, "settings_hash": fingerprint(config), "protocol": p,
            "schedule_sha256": files[cell["schedule_file"]], "workload_file": "workload.json",
            "workload_sha256": files["workload.json"], "policy_file": "policy.json", "policy_sha256": files["policy.json"],
            "preflight_receipt_sha256": receipts, "watchdog_seconds": _watchdog_seconds(p, p["duration_seconds"]),
            "endpoints": service["endpoints"], "service_pids": service["pids"],
            "service_record_file": str(start_path.relative_to(output)), "service_record_sha256": file_hash(start_path),
            "warmup_record_file": str(warm_path.relative_to(output)), "warmup_record_sha256": file_hash(warm_path),
            "allocation": allocation}


def measured(output, config, cell, receipts=None):
    """The attempt whose measurement verifies from raw evidence, without the cleanup check."""
    parent = safe(output, "cells/" + cell["cell_id"])
    candidates = sorted(parent.glob("attempt-*")) if parent.exists() else []
    accepted = []
    for folder in candidates:
        if folder.is_symlink() or not folder.is_dir():
            raise StudyError("Unsafe cell attempt path")
        if not (folder / "complete.json").exists():
            continue
        if receipts is None:
            receipts = preflight_hashes(output, config)
        plan = read_json(folder / "plan.json")
        p, files = config["protocol"], config["file_sha256"]
        expected = {**cell, "settings_hash": fingerprint(config), "protocol": p,
                    "schedule_sha256": files[cell["schedule_file"]], "workload_file": "workload.json",
                    "workload_sha256": files["workload.json"], "policy_file": "policy.json", "policy_sha256": files["policy.json"],
                    "preflight_receipt_sha256": receipts, "watchdog_seconds": _watchdog_seconds(p, p["duration_seconds"])}
        if any(plan.get(k) != v for k, v in expected.items()):
            raise StudyError(f"Completed cell {cell['cell_id']} differs from the fixed plan")
        for kind in ("service", "warmup"):
            if file_hash(safe(output, plan[f"{kind}_record_file"])) != plan[f"{kind}_record_sha256"]:
                raise StudyError(f"Completed cell {kind} evidence changed")
        start = read_json(safe(output, plan["service_record_file"]))
        if not engines_limited(start.get("deployment"), cell["condition"], p["engine"]["max_num_seqs"]):
            raise StudyError("Completed cell service record does not show the engine sequence limit")
        warm_plan, warmed = _verified_serial(output, config, safe(output, plan["warmup_record_file"]).parent,
                                             cell["condition"], "warmup")
        if any(r.get("status") != "ok" for r in warmed):
            raise StudyError("Completed cell warmup failed")
        if (warm_plan["endpoints"], warm_plan["service_pids"]) != (plan["endpoints"], plan["service_pids"]):
            raise StudyError("Completed cell warmup ran on another deployment")
        record = verify_cell(folder, fingerprint(plan))
        rows = _schedule_rows(folder, read_json(safe(output, cell["schedule_file"])), "Completed traces")
        summary = summarize(rows, p["duration_seconds"], p["slo_ms"])
        if any(record["report"].get(k) != v for k, v in summary.items()):
            raise StudyError("Completed cell summary differs from raw responses")
        if not read_jsonl(folder / "resources.jsonl"):
            raise StudyError("Completed cell has no resource evidence")
        accepted.append((folder, record))
    if len(accepted) > 1:
        raise StudyError("Multiple completed attempts for one cell; no result selection permitted")
    return accepted[0] if accepted else None


def cleanup_confirmed(output, plan, folder):
    """Whether the deployment that served this attempt was verified stopped afterwards."""
    path = safe(output, plan["service_record_file"]).parent / "finished.json"
    if not path.is_file():
        return False
    finished = read_json(path)
    stop = finished.get("stop_verification") or {}
    return (finished.get("cell_id") == plan["cell_id"] and finished.get("attempt") == folder.name
            and "error" not in stop and stop.get("gpu_compute_processes") == []
            and stop.get("owned_groups") == sorted(plan["service_pids"]))


def completed(output, config, cell, receipts=None):
    """An accepted cell: a verified measurement whose deployment was confirmed stopped."""
    accepted = measured(output, config, cell, receipts)
    if accepted is None or not cleanup_confirmed(output, read_json(accepted[0] / "plan.json"), accepted[0]):
        return None
    return accepted


def _attempt(output, cell, *, resume_interrupted):
    parent = safe(output, "cells/" + cell["cell_id"])
    previous = sorted(parent.glob("attempt-*")) if parent.exists() else []
    if previous and not resume_interrupted:
        raise StudyError(f"Interrupted attempt preserved for {cell['cell_id']}; inspect/export, then use --resume-interrupted")
    for path in previous:
        if path.is_symlink():
            raise StudyError("Cannot rerun an unsafe attempt")
        if (path / "complete.json").exists():
            if cleanup_confirmed(output, read_json(path / "plan.json"), path):
                raise StudyError("Cannot rerun a completed attempt")
            # Measured, but its deployment was never confirmed stopped: keep the receipt unaccepted.
            (path / "complete.json").rename(path / "unaccepted-complete.json")
            fixed_write(path / "unaccepted.json", {"at": utc_now(), "reason": "deployment stop not confirmed"})
    folder = parent / f"attempt-{len(previous) + 1:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    fixed_write(folder / "attempt.json", {"started_at": utc_now(), "prior_interrupted_attempts": [p.name for p in previous],
                                          "explicit_resume_acknowledgment": bool(previous and resume_interrupted)})
    return folder


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


def _run_process(output, folder, plan, *, check_time=True):
    """Run one owned child under an absolute watchdog.

    The exit is awaited with a timeout bounded by the deadline, and the clock and
    allocation are checked again after it, so a completion observed after the
    watchdog is preserved as unaccepted rather than accepted.
    """
    limit = plan["watchdog_seconds"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise StudyError("A positive finite watchdog bound is required")
    fixed_write(folder / "plan.json", plan)
    command = [sys.executable, "-m", "decision_study.upgrade_study.cell", "--root", str(output), "--attempt", str(folder)]
    with (folder / "controller.log").open("x") as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    until = time.monotonic() + limit
    try:
        while True:
            if check_time:
                runtime.check_time(output)
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise StudyError("Child process exceeded the absolute parent watchdog; interrupted evidence preserved")
            try:
                process.wait(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if time.monotonic() > until:
            raise StudyError("Child exit observed after the absolute parent watchdog; its completion is not accepted")
        if check_time:
            runtime.check_time(output)
        if process.returncode != 0:
            raise StudyError(f"Child process exited {process.returncode}; see {folder / 'controller.log'}")
    except BaseException:
        _stop_process(process)
        if (folder / "complete.json").exists():
            (folder / "complete.json").rename(folder / "unaccepted-complete.json")
        raise


def run(output_dir, *, resume_interrupted=False, launcher=None, lister=None):
    output = Path(output_dir).absolute()
    config = settings(output, check_source=True)
    p = config["protocol"]
    receipts = preflight_hashes(output, config)
    pending = [cell for cell in config["order"] if not completed(output, config, cell, receipts)]
    if not pending:
        return freeze(output)
    if any((output / name).exists() for name in ("freeze.json", "evaluation-access.json", "evaluation.json")):
        raise StudyError("Frozen or scored evidence is incomplete; do not resume collection or replace results")
    if not resume_interrupted:
        for cell in pending:
            if list(safe(output, "cells/" + cell["cell_id"]).glob("attempt-*")):
                raise StudyError("An interrupted attempt exists; export/inspect, then use --resume-interrupted")
    warmup_ids = [e["sample_id"] for e in read_json(output / "warmup.json")]
    for cell in pending:
        condition = cell["condition"]
        runtime.check_time(output, _watchdog_seconds(p, p["duration_seconds"]) + 900)
        began = time.perf_counter()
        record, record_dir, folder = {}, None, None
        try:
            with runtime.deployment(output, condition, launcher=launcher, lister=lister, sink=record) as service:
                record_dir = safe(output, f"services/{cell['cell_id']}-{time.time_ns()}")
                record_dir.mkdir(parents=True)
                start = {"cell_id": cell["cell_id"], "condition": condition, "repeat": cell["repeat"], "started_at": utc_now(),
                         "startup_seconds": time.perf_counter() - began, "deployment": service["record"],
                         "hardware": hardware(), "allocation": runtime.check_time(output)["allocation"],
                         "runtime_binding_sha256": file_hash(output / "runtime/binding.json")}
                start_path = record_dir / "start.json"
                fixed_write(start_path, start)
                warmed = _serial_stage(output, record_dir / "warmup", _serial_plan(
                    config, warmup_ids, condition, "warmup", service["endpoints"], service["pids"]))
                if any(row["status"] != "ok" for row in warmed):
                    raise StudyError(f"Warmup failed for {condition}; no formal cell started with this service")
                runtime.check_time(output, _watchdog_seconds(p, p["duration_seconds"]) + 10)
                folder = _attempt(output, cell, resume_interrupted=resume_interrupted)
                plan = _formal_plan(config, cell, receipts, service, start_path, record_dir / "warmup/rows.json",
                                    start["allocation"], output)
                try:
                    _run_process(output, folder, plan)
                except BaseException as exc:
                    fixed_write(folder / "interrupted.json", {"at": utc_now(), "error": str(exc), "type": type(exc).__name__})
                    raise
                measurement = measured(output, config, cell, receipts)
                if measurement is None or measurement[0] != folder:
                    raise StudyError("Cell process exited without a valid completion receipt")
                report = measurement[1]["report"]
                print(f"{cell['cell_id']}: {report['offered']} offered, {report['met_slo']} within {p['slo_ms']} ms; "
                      f"goodput={report['goodput_per_second']:.2f}/s", flush=True)
        finally:
            if record_dir is not None:
                fixed_write(record_dir / "finished.json", {
                    "at": utc_now(), "cell_id": cell["cell_id"], "attempt": folder.name if folder is not None else None,
                    "stop_verification": record.get("stop_verification")})
        if completed(output, config, cell, receipts) is None:
            raise StudyError(f"{cell['cell_id']} was measured, but its deployment was not confirmed stopped; inspect the "
                             "node, then resume with --resume-interrupted to measure it again")
    return freeze(output)


def freeze(output_dir):
    output = Path(output_dir)
    if not (output / "freeze.json").exists() and any((output / n).exists() for n in ("evaluation-access.json", "evaluation.json")):
        raise StudyError("Scoring evidence exists without its freeze")
    config = settings(output)
    receipts = preflight_hashes(output, config)
    cells = {}
    for cell in config["order"]:
        accepted = completed(output, config, cell, receipts)
        if accepted is None:
            raise StudyError("All 36 fixed cells must complete, with confirmed stops, before the gold is read")
        folder, _ = accepted
        cells[str((folder / "complete.json").relative_to(output))] = file_hash(folder / "complete.json")
    value = {"status": "FROZEN", "settings_hash": fingerprint(config), "cells": cells}
    value["self_hash"] = fingerprint(value)
    fixed_write(safe(output, "freeze.json"), value)
    return value


def pairing_same_allocation(scored):
    """For each repeat and rate, whether its paired cells all ran in one Slurm allocation."""
    groups = defaultdict(set)
    for cell in scored:
        groups[f"r{cell['repeat']}-rate{cell['rate']}"].add(fingerprint(cell["allocation"]))
    return {key: len(value) == 1 for key, value in groups.items()}


def historical_label(history, condition, sid):
    if condition == "cascade":
        local = history["startlux4b"][sid]
        return history["qwen9b"][sid]["prediction"] if local["deferred"] else local["prediction"]
    return history[condition][sid]["prediction"]


def cell_details(rows, condition, gold, history, slo_ms, secondary_slo_ms):
    """Counts, cascade routing outcomes and history comparison weighted by actual offers."""
    admitted = [r for r in rows if not r["rejected"]]
    success = [r for r in admitted if r["status"] == "ok"]
    timely = lambda limit: [r for r in success if r["arrival_to_completion_ms"] <= limit]
    details = {"offered": len(rows), "admitted": len(admitted), "successful": len(success), "rejected": len(rows) - len(admitted),
               "status_counts": dict(Counter(r["status"] for r in rows)),
               "late_successful": {str(slo_ms): len(success) - len(timely(slo_ms)),
                                   str(secondary_slo_ms): len(success) - len(timely(secondary_slo_ms))}}
    agree = hist_correct = live_correct = 0
    for row in success:
        sid, label = row["sample_id"], row["result"]["prediction"]
        past = historical_label(history, condition, sid)
        agree += label == past
        hist_correct += past == gold[sid]
        live_correct += label == gold[sid]
    n = len(success)
    details["history"] = {"successful_responses": n, "label_agreement": agree / n if n else None,
                          "matched_subset_historical_accuracy": hist_correct / n if n else None,
                          "live_accuracy": live_correct / n if n else None,
                          "weighting": "Each successful offer counts once; repeated inputs count as often as they occur"}
    if condition == "cascade":
        reasons = Counter(((r.get("result") or {}).get("route") or {}).get("reason") for r in admitted)
        routed = [r for r in admitted if ((r.get("result") or {}).get("route") or {}).get("reason") in
                  ("score_at_or_above_threshold", "score_below_threshold")]
        deferred = [r for r in routed if r["result"]["route"]["deferred"]]
        agree_route = sum(r["result"]["route"]["deferred"] == history["startlux4b"][r["sample_id"]]["deferred"] for r in routed)
        details["cascade"] = {"route_reasons": dict(reasons), "routed": len(routed), "deferred": len(deferred),
                              "deferral_rate_routed": len(deferred) / len(routed) if routed else None,
                              "fallback_calls": sum(bool(((r.get("result") or {}).get("route") or {}).get("fallback_called")) for r in admitted),
                              "route_agreement_with_history": agree_route / len(routed) if routed else None}
    return details


def resource_summary(path):
    used, util = [], []
    for sample in read_jsonl(path):
        for gpu in (sample.get("gpu") or {}).get("gpus") or []:
            if gpu.get("used_mib") is not None:
                used.append(gpu["used_mib"])
            if gpu.get("utilization_percent") is not None:
                util.append(gpu["utilization_percent"])
    return {"gpu_memory_peak_mib": max(used) if used else None, "gpu_utilization_mean_percent": float(np.mean(util)) if util else None,
            "samples": len(used)}


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": float(np.median(values)), "min": min(values), "max": max(values), "values": values}


def summaries(scored, p):
    out = []
    for condition in p["conditions"]:
        for rate in p["rates_per_second"]:
            selected = [c for c in scored if c["condition"] == condition and c["rate"] == rate]
            metrics = {}
            for slo_key in ("primary", "secondary"):
                for name in ("correct_goodput_per_second", "goodput_per_second", "slo_fraction_all_offered", "accuracy_successful"):
                    metrics[f"{slo_key}.{name}"] = _stats([c[slo_key].get(name) for c in selected])
            metrics["successful_per_second"] = _stats([c["primary"]["successful_per_offered_window_second"] for c in selected])
            metrics["latency_p95_ms"] = _stats([c["primary"]["latency_ms_all_admitted"]["p95"] for c in selected])
            if condition == "cascade":
                metrics["deferral_rate_routed"] = _stats([c["details"]["cascade"]["deferral_rate_routed"] for c in selected])
            out.append({"condition": condition, "rate": rate, "repeats": len(selected), "metrics": metrics})
    return out


def sustained_rates(scored, p, key="primary", threshold=0.99):
    """Highest tested rate at which every repeat has at least 99% of offers timely and successful."""
    result = {}
    for condition in p["conditions"]:
        passing = [rate for rate in p["rates_per_second"]
                   if all(c[key]["slo_fraction_all_offered"] >= threshold
                          for c in scored if c["condition"] == condition and c["rate"] == rate)]
        best = max(passing) if passing else None
        result[condition] = {"highest_passing_rate": best,
                             "statement": (f"met at every tested rate up to {max(p['rates_per_second'])}" if best == max(p["rates_per_second"]) and len(passing) == len(p["rates_per_second"])
                                           else "no tested rate met the criterion" if best is None else f"highest passing tested rate {best}")}
    return result


def evaluate(output_dir):
    output = Path(output_dir)
    config = settings(output)
    p = config["protocol"]
    freeze(output)
    freeze_hash = file_hash(output / "freeze.json")
    access_path, evaluation_path = output / "evaluation-access.json", output / "evaluation.json"
    if evaluation_path.exists():
        value, access = read_json(evaluation_path), read_json(access_path)
        if (value.get("freeze_sha256") != freeze_hash or value.get("access_sha256") != file_hash(access_path)
                or value.get("gold_sha256") != GOLD_SHA256 or access.get("freeze_sha256") != freeze_hash
                or value.get("self_hash") != fingerprint({k: v for k, v in value.items() if k != "self_hash"})):
            raise StudyError("Completed evaluation identity mismatch")
        return value
    if access_path.exists():
        raise StudyError("Scoring started earlier without completing; preserve and inspect, do not rescore silently")
    fixed_write(access_path, {"access_at": utc_now(), "freeze_sha256": freeze_hash, "gold_sha256": GOLD_SHA256,
                              "purpose": "Retrospective quality of actual responses in all 36 frozen cells"})
    gold_path = safe(config["v3_dir"], "sealed/holdout_gold.json")
    if file_hash(gold_path) != GOLD_SHA256:
        raise StudyError("ANLI gold identity mismatch")
    gold = read_json(gold_path)
    history = read_json(output / "history.json")
    scored, labels = [], defaultdict(lambda: defaultdict(set))
    receipts = preflight_hashes(output, config)
    for cell in config["order"]:
        folder, _ = completed(output, config, cell, receipts)
        rows = read_jsonl(folder / "traces.jsonl")
        plan = read_json(folder / "plan.json")
        start = read_json(safe(output, plan["service_record_file"]))
        primary = summarize(rows, p["duration_seconds"], p["slo_ms"], gold)
        secondary = summarize(rows, p["duration_seconds"], p["secondary_slo_ms"], gold)
        for row in rows:
            if row["status"] == "ok":
                labels[cell["condition"]][row["sample_id"]].add(row["result"]["prediction"])
        scored.append({**cell, "primary": primary, "secondary": secondary,
                       "details": cell_details(rows, cell["condition"], gold, history, p["slo_ms"], p["secondary_slo_ms"]),
                       "resources": resource_summary(folder / "resources.jsonl"),
                       "startup_seconds": start["startup_seconds"],
                       "service_roles": [{k: r.get(k) for k in ("role", "memory_fraction", "startup_seconds", "engine_log")}
                                         for r in start["deployment"]["roles"]],
                       "allocation": plan["allocation"], "attempt": str(folder.relative_to(output))})
    full_pool = {}
    ids = list(gold)
    for condition in p["conditions"]:
        full_pool[condition] = sum(historical_label(history, condition, sid) == gold[sid] for sid in ids) / len(ids)
    value = {"status": "COMPLETE", "freeze_sha256": freeze_hash, "access_sha256": file_hash(access_path), "gold_sha256": GOLD_SHA256,
             "cells": scored, "summaries": summaries(scored, p),
             "sustained_rates": {"primary_slo": sustained_rates(scored, p, "primary"),
                                 "secondary_slo": sustained_rates(scored, p, "secondary")},
             "historical_full_pool_accuracy": full_pool,
             "label_variability": {condition: {"inputs_with_more_than_one_live_label": sum(len(v) > 1 for v in by_id.values()),
                                               "inputs_seen": len(by_id)} for condition, by_id in labels.items()},
             "pairing_same_allocation": pairing_same_allocation(scored),
             "interpretation": ["Quality is retrospective on the ANLI test already evaluated in V3",
                                "The three deployments differ in total GPU budget and compute; no causal attribution",
                                "Three repeats describe service variability; no significance test",
                                "Capacity statements are limited to the tested rates and this client and vLLM configuration",
                                "Historical replay counts are regression checks; live deferral and accuracy are measurements"]}
    value["self_hash"] = fingerprint(value)
    fixed_write(evaluation_path, value)
    return value


def status(output_dir):
    config = settings(output_dir)
    try:
        receipts, preflight_problem = preflight_hashes(output_dir, config), None
    except StudyError as exc:
        receipts, preflight_problem = None, str(exc)
    # Every cell binds the verified preflight receipts, so none counts while they do not verify.
    done = [cell["cell_id"] for cell in config["order"] if receipts is not None and completed(output_dir, config, cell, receipts)]
    preflight_state = {c: (read_json(safe(output_dir, f"preflight/{c}.json")).get("status")
                           if safe(output_dir, f"preflight/{c}.json").exists() else None) for c in CONDITIONS}
    return {"completed_cells": len(done), "required_cells": len(config["order"]), "preflight": preflight_state,
            "preflight_verified": receipts is not None, "preflight_problem": preflight_problem,
            "evaluation_present": safe(output_dir, "evaluation.json").exists(), "notice": "No process liveness claim"}


def export(output_dir):
    output = Path(output_dir)
    target = safe(output, "exports/upgrade_study_bundle.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root in ("settings.json", "workload.json", "warmup.json", "policy.json", "history.json", "preflight-schedule.json",
                     "freeze.json", "evaluation-access.json", "evaluation.json", "schedules", "preflight", "cells", "services",
                     "runtime", "runtime-controls", "diagnostics"):
            path = safe(output, root)
            files = [path] if path.is_file() else sorted(path.rglob("*")) if path.exists() else []
            for source in files:
                if source.is_symlink():
                    raise StudyError("Refusing symlinked export evidence")
                if source.is_file() and source.name != ".active.lock" and "work" not in source.parts:
                    archive.write(source, str(source.relative_to(output)))
    os.replace(temporary, target)
    return target
