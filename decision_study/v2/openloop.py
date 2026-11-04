"""Bounded open-arrival measurement of the actual frozen online policy."""
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import random
import threading
import time

import numpy as np

from .. import runtime
from ..adapters import ModelClient
from ..benchmark import GPUMonitor
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json, write_jsonl
from ..data import load_examples
from .identity import PROJECT, protocol, source_identity
from .policies import should_defer
from .runner_v2 import verify_runtime_model


def call_policy(example, local, fallback=None, policy=None):
    first = local.predict(example)
    deferred = fallback is not None and (first["status"] != "ok" or should_defer(policy, example, first))
    second = fallback.predict(example) if deferred else None
    final = second if deferred else first
    return {"sample_id": example["sample_id"], "prediction": final["prediction"],
            "status": final["status"], "deferred": deferred, "local": first, "fallback": second}


def arrival_schedule(rate, duration, seed):
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Positive finite arrival rate and duration required")
    rng, offsets, t = random.Random(seed), [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= duration:
            return offsets
        offsets.append(t)


def run_cell(examples, call, *, rate, duration, seed, workers=8, max_inflight=64, slo_ms=500, budget=None):
    """Always account for offered requests, including overload and final failures.

    A semaphore bounds active+queued work. Arrival generation does not wait for a
    queue slot; rejected requests are explicit outcomes. Worker occupancy has its
    own timestamped events, independently of reported latency aggregates.
    """
    if not examples or workers < 1 or max_inflight < workers or slo_ms <= 0:
        raise ValueError("Invalid workload/concurrency/SLO")
    offsets = arrival_schedule(rate, duration, seed)
    slots = threading.BoundedSemaphore(max_inflight)
    mutex = threading.Lock()
    events, rows, pending = [], [], []
    active = 0
    started = time.perf_counter()
    started_at = utc_now()

    def worker(index, example, arrival, dispatched):
        nonlocal active
        with mutex:
            began = time.perf_counter() - started
            active += 1
            events.append({"at_seconds": began, "active_workers": active})
        try:
            result = call(example)
        except Exception as exc:
            result = {"sample_id": example["sample_id"], "status": "error", "prediction": None,
                      "error": {"type": type(exc).__name__, "message": str(exc)}, "deferred": False}
        finally:
            with mutex:
                ended = time.perf_counter() - started
                active -= 1
                events.append({"at_seconds": ended, "active_workers": active})
            slots.release()
        latency = 1000 * (ended-arrival)
        return {"offer_id": index, "sample_id": example["sample_id"], "scheduled_arrival_seconds": arrival,
                "dispatch_seconds": dispatched, "worker_start_seconds": began, "completion_seconds": ended,
                "arrival_to_completion_ms": latency, "worker_ms": 1000*(ended-began),
                "status": result["status"], "rejected": False,
                "meets_slo": result["status"] == "ok" and latency <= slo_ms, "result": result}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        last_budget_check = -1.0
        for i, arrival in enumerate(offsets):
            delay = started + arrival-time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            if budget and arrival - last_budget_check >= 1.0:
                budget.check()
                last_budget_check = arrival
            dispatched = time.perf_counter()-started
            example = examples[i % len(examples)]
            if not slots.acquire(blocking=False):
                rows.append({"offer_id": i, "sample_id": example["sample_id"],
                             "scheduled_arrival_seconds": arrival, "dispatch_seconds": dispatched,
                             "status": "rejected", "rejected": True, "meets_slo": False,
                             "reason": "bounded_inflight_limit"})
            else:
                pending.append(pool.submit(worker, i, example, arrival, dispatched))
        # End of arrival window is independent of when the final request finishes.
        remaining = started + duration-time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        rows.extend(f.result() for f in pending)
    elapsed = time.perf_counter()-started
    rows.sort(key=lambda r: r["offer_id"])
    events.sort(key=lambda r: r["at_seconds"])
    completed = [r for r in rows if not r["rejected"]]
    success = [r for r in completed if r["status"] == "ok"]
    timely = sum(r["meets_slo"] for r in rows)
    latency = [r["arrival_to_completion_ms"] for r in completed]
    area, last, prior = 0.0, 0.0, 0
    for event in events:
        area += (event["at_seconds"]-last)*prior
        last, prior = event["at_seconds"], event["active_workers"]
    worker_time = sum(r["worker_ms"] for r in completed)/1000
    report = {"started_at": started_at, "rate_requested": rate, "duration_seconds": duration,
              "drained_elapsed_seconds": elapsed, "seed": seed, "workers": workers,
              "max_inflight": max_inflight, "slo_ms": slo_ms, "offered": len(rows),
              "offered_rate_actual": len(rows)/duration, "successful": len(success),
              "failed": len(completed)-len(success), "rejected": len(rows)-len(completed),
              "met_slo": timely, "slo_fraction_all_offered": timely/len(rows) if rows else None,
              "goodput_per_second": timely/duration,
              "successful_per_offered_window_second": len(success)/duration,
              "successful_per_drained_second": len(success)/elapsed,
              "latency_ms_all_admitted": {f"p{int(q*100)}": float(np.quantile(latency,q)) if latency else None for q in (.5,.95,.99)},
              "max_dispatch_lag_ms": max((1000*(r["dispatch_seconds"]-r["scheduled_arrival_seconds"]) for r in rows), default=0),
              "worker_occupancy_seconds": area, "sum_worker_seconds": worker_time,
              "occupancy_accounting_error_seconds": abs(area-worker_time),
              "note": "Latency includes offered-arrival and client queue; failed admitted requests remain in latency distribution. Rejections count against SLO."}
    return report, rows, events


def _completed_cell(folder, cell, settings_hash):
    """Only resume completed cells whose raw evidence still matches the receipt."""
    path = Path(folder) / f"{cell}.json"
    if not path.exists():
        return False
    report = read_json(path)
    if (report.get("settings_hash") != settings_hash or report.get("report_hash") !=
            fingerprint({k: v for k, v in report.items() if k != "report_hash"})):
        raise StudyError("Completed openloop cell identity mismatch")
    for kind in ("traces", "occupancy"):
        evidence = Path(folder) / f"{cell}-{kind}.jsonl"
        if not evidence.is_file() or file_hash(evidence) != report.get("evidence_hashes", {}).get(kind):
            raise StudyError(f"Completed openloop {kind} evidence changed or missing")
    return True


def run(v1_dir, output_dir, *, control, budget):
    from .study import load_freeze
    output = Path(output_dir)
    frozen = load_freeze(output, v1_dir=Path(v1_dir))
    config = protocol()["openloop"]
    policy = frozen["selected_policy"]
    examples = [e for e in load_examples(Path(v1_dir), split="validation") if e["task"] == config["task"]]
    examples.sort(key=lambda e: fingerprint({"seed": 6998, "role": "v2-performance", "id": e["sample_id"]}))
    examples = examples[:config["unique_inputs"]]
    if len(examples) != config["unique_inputs"]:
        raise StudyError("Incomplete validation workload for openloop measurement")
    warmup = [e for e in load_examples(Path(v1_dir), split="development") if e["task"] == config["task"]]
    effective = {m: verify_runtime_model(v1_dir, control, m) for m in ("startlux", "qwen")}
    settings = {"protocol": "solo-v2", "config": config, "policy_freeze_hash": frozen["freeze_hash"],
                "sources": source_identity(), "runtime": effective,
                "workload_ids": [e["sample_id"] for e in examples], "workload_hash": fingerprint(examples)}
    folder = output / "openloop"
    existing = folder / "settings.json"
    if existing.exists() and read_json(existing) != settings:
        raise StudyError("Open-arrival settings changed; preserve previous results and use a new run")
    write_json(existing, settings)
    for model in ("startlux", "qwen"):
        runtime.token_preflight_examples(PROJECT, control, model, examples + warmup)
    for repeat in range(config["repeats"]):
        conditions = list(config["conditions"])
        random.Random(6998+repeat).shuffle(conditions)
        for condition in conditions:
            cells = [f"r{repeat}-{condition}-rate{rate:g}" for rate in config["arrival_rates_per_second"]]
            completed = {cell: _completed_cell(folder, cell, fingerprint(settings)) for cell in cells}
            if all(completed.values()):
                continue
            models = ["qwen"] if condition == "qwen_only" else (["startlux"] if condition == "local_only" else ["startlux", "qwen"])
            budget.check()
            startup_begin = time.perf_counter()
            with runtime.managed_services(PROJECT, control, models) as endpoints:
                service_record = {"settings_hash": fingerprint(settings), "condition": condition,
                                  "repeat": repeat, "startup_seconds": time.perf_counter()-startup_begin,
                                  "timing": "Includes runtime checks and service readiness; excludes warmup and offered window"}
                service_metadata = Path(control) / "runtime/last_services.json"
                if service_metadata.exists():
                    service_record["services"] = read_json(service_metadata)
                write_json(folder/f"r{repeat}-{condition}-services.json", service_record)
                clients = {m: ModelClient(m, endpoints[m]) for m in models}
                local = clients["qwen"] if condition == "qwen_only" else clients["startlux"]
                fallback = clients.get("qwen") if condition == "selected_cascade" else None
                invoke = lambda e: call_policy(e, local, fallback, policy)
                began = time.perf_counter()
                with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
                    warm = list(pool.map(invoke, warmup))
                write_json(folder/f"r{repeat}-{condition}-warmup.json", {"rows": warm, "elapsed_seconds": time.perf_counter()-began})
                if any(r["status"] != "ok" for r in warm):
                    raise StudyError("Warmup failed; no performance claims from this condition")
                for index, rate in enumerate(config["arrival_rates_per_second"]):
                    cell = cells[index]
                    path = folder/f"{cell}.json"
                    if completed[cell]:
                        continue
                    budget.check()
                    old_deadline = os.environ.get("DECISION_STUDY_DEADLINE_EPOCH")
                    # Bound draining as well as arrivals. Adapters observe this on
                    # every attempt; no silent cancellation of already sent work.
                    deadline = time.time()+config["duration_seconds"]+120
                    if old_deadline:
                        deadline = min(deadline,float(old_deadline))
                    os.environ["DECISION_STUDY_DEADLINE_EPOCH"] = str(deadline)
                    try:
                        with GPUMonitor() as monitor:
                            report, rows, occupancy = run_cell(examples, invoke, rate=rate,
                                duration=config["duration_seconds"], seed=6998+repeat*100+index,
                                workers=config["workers"], max_inflight=config["max_inflight"],
                                slo_ms=config["slo_ms"], budget=budget)
                    finally:
                        if old_deadline is None:
                            os.environ.pop("DECISION_STUDY_DEADLINE_EPOCH",None)
                        else:
                            os.environ["DECISION_STUDY_DEADLINE_EPOCH"] = old_deadline
                    write_jsonl(folder/f"{cell}-traces.jsonl", rows)
                    write_jsonl(folder/f"{cell}-occupancy.jsonl", occupancy)
                    report.update(settings_hash=fingerprint(settings), condition=condition, repeat=repeat,
                                  memory=monitor.report(), scored_holdout=False)
                    report["evidence_hashes"] = {kind: file_hash(folder/f"{cell}-{kind}.jsonl")
                                                 for kind in ("traces", "occupancy")}
                    report["report_hash"] = fingerprint(report)
                    write_json(path,report)
                    print(f"{cell}: {report['offered']} offered, {report['met_slo']} within SLO; goodput={report['goodput_per_second']:.2f}/s",flush=True)
            if service_metadata.exists():
                service_record["services"] = read_json(service_metadata)
                write_json(folder/f"r{repeat}-{condition}-services.json", service_record)
    return {"cells": len(config["conditions"])*len(config["arrival_rates_per_second"])*config["repeats"],
            "settings_hash": fingerprint(settings), "path": str(folder)}
