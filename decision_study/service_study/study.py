"""Freeze plans, execute actual services, preserve attempts, then score responses."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import numpy as np

from ..common import StudyError, file_hash, fingerprint, read_json, read_jsonl, utc_now, write_json
from . import runtime
from .client import call_http
from .cpu import verify_bundle
from .identity import (GOLD_SHA256, PROJECT, fixed_write, load_inputs, protocol, safe,
                       source_identity)
from .measurement import make_schedule, summarize, verify_cell
from .resources import hardware


def planned_order(p):
    order = []
    for repeat in range(p["repeats"]):
        conditions = p["conditions"].copy()
        random.Random(p["seed"]+repeat).shuffle(conditions)
        for condition in conditions:
            rates = p["rates_per_second"].copy()
            random.Random(f"{p['seed']}-{repeat}-{condition}").shuffle(rates)
            order.extend({"cell_id": f"r{repeat}-{condition}-rate{rate}", "repeat": repeat,
                          "condition": condition, "rate": rate,
                          "schedule_file": f"schedules/r{repeat}-rate{rate}.json"} for rate in rates)
    return order


def prepare(v1_dir, bundle_dir, model_dir, cpu_python, output_dir):
    output = Path(output_dir).absolute()
    for value in (v1_dir, bundle_dir, model_dir, PROJECT):
        root = Path(value).absolute()
        if output == root or output in root.parents or root in output.parents:
            raise StudyError("Service output must not overlap an input or source root")
    main, warmup, inputs = load_inputs(v1_dir)
    p, source = protocol(), source_identity()
    baseline = {r: verify_bundle(bundle_dir, r)[2] for r in ("minilm400", "minilm1600")}
    schedules, order = {}, []
    for repeat in range(p["repeats"]):
        conditions = p["conditions"].copy()
        random.Random(p["seed"]+repeat).shuffle(conditions)
        for rate in p["rates_per_second"]:
            schedules[f"schedules/r{repeat}-rate{rate}.json"] = make_schedule(main, rate, p["duration_seconds"], p["seed"]+repeat*1000+rate)
        for condition in conditions:
            rates = p["rates_per_second"].copy()
            random.Random(f"{p['seed']}-{repeat}-{condition}").shuffle(rates)
            order.extend({"cell_id": f"r{repeat}-{condition}-rate{rate}", "repeat": repeat,
                          "condition": condition, "rate": rate,
                          "schedule_file": f"schedules/r{repeat}-rate{rate}.json"} for rate in rates)
    from ..common import canonical
    bytehash = lambda value: __import__("hashlib").sha256(canonical(value)+b"\n").hexdigest()
    settings = {"protocol": p, "source": source, "inputs": inputs, "baseline": baseline,
                "v1_dir": str(Path(v1_dir).absolute()), "bundle_dir": str(Path(bundle_dir).absolute()),
                "model_dir": str(Path(model_dir).absolute()), "cpu_python": str(Path(cpu_python).absolute()),
                "workload_sha256": bytehash(main), "warmup_sha256": bytehash(warmup),
                "schedules": {name: bytehash(value) for name, value in schedules.items()}, "order": order}
    # Verify all existing prepared files before adding any missing file.
    contents = {"settings.json": settings, "workload.json": main, "warmup.json": warmup, **schedules}
    for name, value in contents.items():
        path = safe(output, name)
        if path.exists() and read_json(path) != value:
            raise StudyError(f"Preparation conflicts with existing evidence: {name}")
    for name, value in contents.items():
        fixed_write(safe(output, name), value)
    return {"status": "PREPARED", "settings_hash": fingerprint(settings), "cells": len(order),
            "total_offers_planned": sum(len(schedules[c["schedule_file"]]) for c in order),
            "offer_window_hours": len(order)*p["duration_seconds"]/3600,
            "notice": "Public input and historical classifier verification only; no GPU calls or V1 gold read"}


def settings(output_dir, *, check_source=False):
    value = read_json(safe(output_dir, "settings.json"))
    if check_source and value["source"] != source_identity():
        raise StudyError("Service-study source differs from prepared identity")
    if value["protocol"] != protocol():
        raise StudyError("Service protocol changed")
    expected_order = planned_order(value["protocol"])
    if value["order"] != expected_order or set(value["schedules"]) != {c["schedule_file"] for c in expected_order}:
        raise StudyError("Incomplete or changed fixed cell/schedule inventory")
    for name, digest in {"workload.json": value["workload_sha256"], "warmup.json": value["warmup_sha256"], **value["schedules"]}.items():
        if file_hash(safe(output_dir, name)) != digest:
            raise StudyError(f"Changed prepared service input: {name}")
    return value


def completed(output, config, cell):
    parent = safe(output, "cells/"+cell["cell_id"])
    candidates = sorted(parent.glob("attempt-*")) if parent.exists() else []
    accepted = []
    for folder in candidates:
        if folder.is_symlink() or not folder.is_dir():
            raise StudyError("Unsafe cell attempt path")
        if not (folder/"complete.json").exists():
            continue
        plan = read_json(folder/"plan.json")
        expected = {**cell, "settings_hash": fingerprint(config), "protocol": config["protocol"],
                    "schedule_sha256": config["schedules"][cell["schedule_file"]],
                    "workload_file": "workload.json", "workload_sha256": config["workload_sha256"]}
        if any(plan.get(k) != v for k, v in expected.items()):
            raise StudyError("Completed cell differs from fixed workload/condition")
        service_file = safe(output, plan["service_record_file"])
        if file_hash(service_file) != plan["service_record_sha256"]:
            raise StudyError("Completed cell service identity changed")
        warm_path = safe(output, plan["warmup_record_file"])
        if file_hash(warm_path) != plan["warmup_record_sha256"]:
            raise StudyError("Completed cell warmup evidence changed")
        warmed = read_json(warm_path)["rows"]
        expected_warmup = read_json(safe(output, "warmup.json"))
        if ([r.get("sample_id") for r in warmed] != [r["sample_id"] for r in expected_warmup]
                or any(r.get("status") != "ok" for r in warmed)):
            raise StudyError("Completed cell warmup is incomplete or failed")
        record = verify_cell(folder, fingerprint(plan))
        schedule = read_json(safe(output, cell["schedule_file"]))
        rows = sorted(read_jsonl(folder/"traces.jsonl"), key=lambda r: r["offer_id"])
        if [{k: row.get(k) for k in ("offer_id", "sample_id", "arrival_seconds")} for row in rows] != schedule:
            raise StudyError("Completed traces do not cover the exact offered schedule")
        summary = summarize(rows, config["protocol"]["duration_seconds"], config["protocol"]["slo_ms"])
        if any(record["report"].get(k) != v for k, v in summary.items()):
            raise StudyError("Completed cell summary differs from raw responses")
        if not read_jsonl(folder/"resources.jsonl"):
            raise StudyError("Completed cell has no resource evidence")
        accepted.append((folder, record))
    if len(accepted) > 1:
        raise StudyError("Multiple completed attempts for a cell; no result selection permitted")
    return accepted[0] if accepted else None


def _attempt(output, cell, *, resume_interrupted):
    parent = safe(output, "cells/"+cell["cell_id"])
    previous = sorted(parent.glob("attempt-*")) if parent.exists() else []
    if previous and not resume_interrupted:
        raise StudyError(f"Interrupted attempt preserved for {cell['cell_id']}; inspect/export, then explicitly use --resume-interrupted")
    for path in previous:
        if path.is_symlink() or (path/"complete.json").exists():
            raise StudyError("Cannot rerun a completed/unsafe attempt")
    folder = parent/f"attempt-{len(previous)+1:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    fixed_write(folder/"attempt.json", {"started_at": utc_now(), "prior_interrupted_attempts": [p.name for p in previous],
                "explicit_resume_acknowledgment": bool(previous and resume_interrupted)})
    return folder


def _run_process(output, folder, plan):
    p = plan["protocol"]
    fixed_write(folder/"plan.json", plan)
    command = [sys.executable, "-m", "decision_study.service_study.cell", "--root", str(output), "--attempt", str(folder)]
    with (folder/"controller.log").open("x") as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    until = time.monotonic()+p["duration_seconds"]+p["drain_seconds"]+p["watchdog_grace_seconds"]
    try:
        while process.poll() is None:
            runtime.check_time(output)
            if time.monotonic() >= until:
                raise StudyError("Cell exceeded absolute parent watchdog; preserve interrupted evidence")
            time.sleep(.25)
        if time.monotonic() >= until:
            raise StudyError("Cell exit observed after absolute parent watchdog; preserve interrupted evidence")
        if process.returncode != 0:
            raise StudyError(f"Cell process exited {process.returncode}; see {folder/'controller.log'}")
    except BaseException:
        runtime.stop(process)
        # A racing child completion is not accepted after a watchdog/interruption.
        # Preserve it under a distinct name rather than reusing it as a success.
        if (folder/"complete.json").exists():
            (folder/"complete.json").rename(folder/"unaccepted-complete.json")
        raise


def run(output_dir, *, resume_interrupted=False):
    output = Path(output_dir).absolute()
    config = settings(output, check_source=True)
    p = config["protocol"]
    main, warmup = read_json(output/"workload.json"), read_json(output/"warmup.json")
    pending = [cell for cell in config["order"] if not completed(output, config, cell)]
    if not pending:
        return freeze(output)
    if any((output/name).exists() for name in ("freeze.json", "evaluation-access.json", "evaluation.json")):
        raise StudyError("Frozen/scored service evidence is incomplete; do not resume collection or replace results")
    # Detect interrupted work before starting an expensive model.
    if not resume_interrupted:
        for cell in pending:
            if list(safe(output, "cells/"+cell["cell_id"]).glob("attempt-*")):
                raise StudyError("An interrupted attempt exists; export/inspect, then use --resume-interrupted")
    runtime.validate(output)
    for cell in pending:
        repeat, condition = cell["repeat"], cell["condition"]
        runtime.check_time(output, p["duration_seconds"]+p["drain_seconds"]+120)
        runtime.idle_device(output)
        context = (runtime.cpu_service(output, condition, python=config["cpu_python"], bundle_dir=config["bundle_dir"], model_dir=config["model_dir"])
                   if condition.startswith("minilm") else runtime.gpu_service(output, condition, main+warmup))
        began = time.perf_counter()
        # Fresh services for every cell prevent timed-out backend work leaking
        # into the next rate. Client drain alone cannot establish server idle.
        with context as service:
            record_dir = safe(output, f"services/{cell['cell_id']}-{time.time_ns()}")
            record_dir.mkdir(parents=True)
            record = {"condition": condition, "repeat": repeat, "started_at": utc_now(),
                      "startup_seconds": time.perf_counter()-began, "service": service,
                      "hardware": hardware(), "allocation": runtime.check_time(output)["allocation"],
                      "runtime_binding_hash": file_hash(output/"runtime/binding.json")}
            start_path = record_dir/"start.json"
            fixed_write(start_path, record)
            warm_started = time.perf_counter()
            warmed = []
            for i, example in enumerate(warmup):
                runtime.check_time(output, p["request_timeout_seconds"]+10)
                warmed.append(call_http(condition, service["endpoint"], example, f"warmup-{i}", p["request_timeout_seconds"]))
            warm_path = record_dir/"warmup.json"
            fixed_write(warm_path, {"rows": warmed, "elapsed_seconds": time.perf_counter()-warm_started})
            if any(row["status"] != "ok" for row in warmed):
                raise StudyError(f"Warmup failed for {condition}; no formal cell started for this service")
            runtime.check_time(output, p["duration_seconds"]+p["drain_seconds"]+p["watchdog_grace_seconds"]+10)
            folder = _attempt(output, cell, resume_interrupted=resume_interrupted)
            plan = {**cell, "settings_hash": fingerprint(config), "protocol": p,
                    "schedule_sha256": config["schedules"][cell["schedule_file"]],
                    "workload_file": "workload.json", "workload_sha256": config["workload_sha256"],
                    "endpoint": service["endpoint"], "service_pids": service["pids"],
                    "service_record_file": str(start_path.relative_to(output)), "service_record_sha256": file_hash(start_path),
                    "warmup_record_file": str(warm_path.relative_to(output)), "warmup_record_sha256": file_hash(warm_path),
                    "allocation": record["allocation"]}
            try:
                _run_process(output, folder, plan)
            except BaseException as exc:
                fixed_write(folder/"interrupted.json", {"at": utc_now(), "error": str(exc), "type": type(exc).__name__})
                raise
            accepted = completed(output, config, cell)
            if accepted is None:
                raise StudyError("Child exited without valid completed cell")
            report = accepted[1]["report"]
            print(f"{cell['cell_id']}: {report['offered']} offered, {report['met_slo']} timely; goodput={report['goodput_per_second']:.2f}/s", flush=True)
        fixed_write(record_dir/"finished.json", {"at": utc_now(), "notice": "Owned service context exited after this cell"})
    return freeze(output)


def freeze(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir/"freeze.json").exists() and any((output_dir/name).exists() for name in ("evaluation-access.json", "evaluation.json")):
        raise StudyError("Scoring evidence exists without its original freeze")
    config = settings(output_dir)
    cells = {}
    for cell in config["order"]:
        accepted = completed(output_dir, config, cell)
        if accepted is None:
            raise StudyError("All60 fixed cells must complete before quality label access")
        folder, record = accepted
        cells[str((folder/"complete.json").relative_to(output_dir))] = file_hash(folder/"complete.json")
    value = {"status": "FROZEN", "settings_hash": fingerprint(config), "cells": cells}
    value["self_hash"] = fingerprint(value)
    fixed_write(safe(output_dir, "freeze.json"), value)
    return value


def evaluate(output_dir):
    output = Path(output_dir)
    config = settings(output)
    frozen = freeze(output)
    freeze_hash = file_hash(output/"freeze.json")
    access_path, evaluation_path = output/"evaluation-access.json", output/"evaluation.json"
    if evaluation_path.exists():
        value = read_json(evaluation_path)
        access = read_json(access_path)
        if (value.get("freeze_sha256") != freeze_hash or value.get("access_sha256") != file_hash(access_path)
                or value.get("status") != "COMPLETE" or value.get("gold_sha256") != GOLD_SHA256
                or access.get("freeze_sha256") != freeze_hash or access.get("gold_sha256") != GOLD_SHA256
                or value.get("self_hash") != fingerprint({k: v for k, v in value.items() if k != "self_hash"})):
            raise StudyError("Completed evaluation identity mismatch")
        return value
    if access_path.exists():
        raise StudyError("Scoring previously started without completion; preserve and inspect, do not silently rescore")
    fixed_write(access_path, {"access_at": utc_now(), "freeze_sha256": freeze_hash, "gold_sha256": GOLD_SHA256,
                             "purpose": "Retrospective quality of actual HTTP responses in all60 frozen cells"})
    gold_path = safe(config["v1_dir"], "data/gold.json")
    if file_hash(gold_path) != GOLD_SHA256:
        raise StudyError("V1 gold identity mismatch")
    gold = read_json(gold_path)
    scored, allocations = [], {}
    for cell in config["order"]:
        folder, _ = completed(output, config, cell)
        rows = read_jsonl(folder/"traces.jsonl")
        result = summarize(rows, config["protocol"]["duration_seconds"], config["protocol"]["slo_ms"], gold)
        plan = read_json(folder/"plan.json")
        scored.append({**cell, **result, "allocation": plan["allocation"], "attempt": str(folder.relative_to(output))})
        allocations.setdefault(f"r{cell['repeat']}-rate{cell['rate']}", set()).add(fingerprint(plan["allocation"]))
    summaries = []
    for condition in config["protocol"]["conditions"]:
        for rate in config["protocol"]["rates_per_second"]:
            selected = [r for r in scored if r["condition"] == condition and r["rate"] == rate]
            metrics = {}
            for name in ("goodput_per_second", "correct_goodput_per_second", "slo_fraction_all_offered", "correct_fraction_all_offered"):
                values = [r[name] for r in selected]
                metrics[name] = {"median": float(np.median(values)), "min": min(values), "max": max(values), "values": values}
            summaries.append({"condition": condition, "rate": rate, "repeats": len(selected), "metrics": metrics})
    value = {"status": "COMPLETE", "freeze_sha256": freeze_hash, "access_sha256": file_hash(access_path),
             "gold_sha256": GOLD_SHA256, "cells": scored, "summaries": summaries,
             "pairing_same_allocation": {key: len(value) == 1 for key, value in allocations.items()},
             "interpretation": ["Five repetitions measure service variability, not independent quality observations",
                "Prespecified rates do not guarantee observing every service saturation knee",
                "CPU conditions still use the allocated node; no dollar estimate or zero GPU allocation cost",
                "Different allocation groups must not be described as hardware-matched paired measurements",
                "CPU supervised labels and GPU prompting/training conditions differ; quality is retrospective"]}
    value["self_hash"] = fingerprint(value)
    fixed_write(evaluation_path, value)
    return value


def status(output_dir):
    config = settings(output_dir)
    done = [cell["cell_id"] for cell in config["order"] if completed(output_dir, config, cell)]
    evaluation = safe(output_dir, "evaluation.json")
    if evaluation.exists():
        if not safe(output_dir, "freeze.json").exists() or not safe(output_dir, "evaluation-access.json").exists():
            raise StudyError("Evaluation lacks original freeze/access evidence")
        evaluate(output_dir)  # Completed-only branch; never reads gold or creates access.
    return {"completed_cells": len(done), "required_cells": len(config["order"]),
            "state": "COMPLETE" if evaluation.exists() and len(done) == len(config["order"]) else "PARTIAL",
            "evaluation_present": evaluation.exists(), "notice": "No process liveness claim"}


def export(output_dir):
    output = Path(output_dir)
    target = safe(output, "exports/service_study_bundle.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root in ("settings.json", "workload.json", "warmup.json", "freeze.json", "evaluation-access.json", "evaluation.json",
                     "schedules", "cells", "services", "runtime", "runtime-control", "diagnostics"):
            path = safe(output, root)
            files = [path] if path.is_file() else sorted(path.rglob("*")) if path.exists() else []
            for source in files:
                if source.is_symlink():
                    raise StudyError("Refusing symlinked export evidence")
                if source.is_file() and source.name != ".active.lock" and "work" not in source.parts:
                    archive.write(source, str(source.relative_to(output)))
    os.replace(temporary, target)
    return target
