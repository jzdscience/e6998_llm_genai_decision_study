"""Gold-free open arrivals, raw evidence, and retrospective service scoring."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
from pathlib import Path
import random
import threading
import time

import numpy as np

from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl, utc_now


def make_schedule(examples, rate, duration, seed):
    if not examples or not math.isfinite(rate) or rate <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Nonempty inputs and positive finite rate/duration required")
    ids = [e["sample_id"] for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Workload IDs must be unique")
    arrivals, samples = random.Random(seed), random.Random(seed + 1_000_000)
    rows, cycle, offset = [], [], 0.0
    while True:
        offset += arrivals.expovariate(rate)
        if offset >= duration:
            return rows
        if not cycle:
            cycle = ids.copy()
            samples.shuffle(cycle)
        rows.append({"offer_id": len(rows), "sample_id": cycle.pop(), "arrival_seconds": offset})


def validate_schedule(schedule, duration):
    if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Positive finite duration required")
    last = -1.0
    for i, row in enumerate(schedule):
        offset = row.get("arrival_seconds")
        if (row.get("offer_id") != i or not isinstance(row.get("sample_id"), str)
                or isinstance(offset, bool) or not isinstance(offset, (float, int))
                or not math.isfinite(offset) or not last < offset < duration):
            raise StudyError("Invalid offered-arrival schedule")
        last = offset


def summarize(rows, duration, slo_ms, gold=None):
    if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in (duration, slo_ms))
            or len({r["offer_id"] for r in rows}) != len(rows)):
        raise StudyError("Invalid service summary inputs")
    admitted = [r for r in rows if not r["rejected"]]
    success = [r for r in admitted if r["status"] == "ok"]
    timely = [r for r in success if r["arrival_to_completion_ms"] <= slo_ms]
    latencies = [r["arrival_to_completion_ms"] for r in admitted]
    total = len(rows)
    summary = {
        "offered": total, "admitted": len(admitted), "successful": len(success),
        "failed_admitted": len(admitted)-len(success), "rejected": total-len(admitted),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "duration_seconds": duration, "offered_rate_actual": total/duration,
        "met_slo": len(timely), "slo_fraction_all_offered": len(timely)/total if total else None,
        "goodput_per_second": len(timely)/duration,
        "successful_per_offered_window_second": len(success)/duration,
        "latency_ms_all_admitted": {f"p{int(q*100)}": float(np.quantile(latencies, q)) if latencies else None
                                     for q in (.5, .95, .99)},
        "max_dispatch_lag_ms": max((1000*(r["dispatch_seconds"]-r["arrival_seconds"]) for r in rows), default=0),
    }
    if gold is not None:
        if any(r["sample_id"] not in gold for r in rows):
            raise StudyError("Missing labels for an offered sample")
        correct = [r for r in success if r["result"]["prediction"] == gold[r["sample_id"]]]
        timely_correct = [r for r in correct if r["arrival_to_completion_ms"] <= slo_ms]
        summary.update(correct=len(correct), correct_met_slo=len(timely_correct),
                       correct_goodput_per_second=len(timely_correct)/duration,
                       correct_fraction_all_offered=len(correct)/total if total else None,
                       accuracy_successful=len(correct)/len(success) if success else None,
                       note="Repeated requests describe this offered workload, not independent quality samples")
    return summary


def run_cell(schedule, call, *, duration, workers=16, max_inflight=64, slo_ms=500,
             drain_seconds=60, request_timeout=30, trace_path=None):
    """No retry; preserve every offered request. Parent process enforces watchdog.

    call receives one scheduled row and a timeout clamped to the remaining drain
    allowance. A socket timeout alone cannot impose an absolute execution bound;
    the controller therefore runs this function in a separate owned process.
    """
    validate_schedule(schedule, duration)
    if (not isinstance(workers, int) or isinstance(workers, bool) or not isinstance(max_inflight, int)
            or isinstance(max_inflight, bool) or workers < 1 or max_inflight < workers
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
                   for v in (slo_ms, drain_seconds, request_timeout))):
        raise ValueError("Invalid concurrency/timing bounds")
    slots, mutex = threading.BoundedSemaphore(max_inflight), threading.Lock()
    rows, events, pending, active = [], [], [], 0
    handle = Path(trace_path).open("xb", buffering=0) if trace_path else None
    started, started_at = time.perf_counter(), utc_now()
    cutoff = started + duration + drain_seconds

    def append(row):
        with mutex:
            rows.append(row)
            if handle:
                handle.write(canonical(row) + b"\n")

    def work(offer, dispatched):
        nonlocal active
        with mutex:
            begin = time.perf_counter()-started
            active += 1
            events.append({"at_seconds": begin, "active_workers": active})
        try:
            remaining = cutoff-time.perf_counter()
            if remaining <= 0:
                result = {"status": "deadline", "prediction": None, "error": "client_queue_reached_drain_cutoff"}
            else:
                result = call(offer, min(request_timeout, remaining))
            if not isinstance(result, dict) or result.get("status") not in {"ok", "error", "timeout", "http_rejected", "deadline"}:
                raise ValueError("Malformed call result")
            if result["status"] == "ok" and not isinstance(result.get("prediction"), str):
                raise ValueError("Successful response has no prediction")
        except Exception as exc:
            result = {"status": "error", "prediction": None,
                      "error": {"type": type(exc).__name__, "message": str(exc)}}
        with mutex:
            end = time.perf_counter()-started
            active -= 1
            events.append({"at_seconds": end, "active_workers": active})
        slots.release()
        append({**offer, "dispatch_seconds": dispatched, "worker_start_seconds": begin,
                "completion_seconds": end, "arrival_to_completion_ms": 1000*(end-offer["arrival_seconds"]),
                "client_queue_ms": 1000*(begin-dispatched), "worker_ms": 1000*(end-begin),
                "rejected": False, "status": result["status"], "result": result})

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offer in schedule:
                delay = started+offer["arrival_seconds"]-time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                dispatched = time.perf_counter()-started
                if not slots.acquire(blocking=False):
                    append({**offer, "dispatch_seconds": dispatched, "status": "client_rejected", "rejected": True,
                            "reason": "bounded_client_inflight", "result": None})
                else:
                    pending.append(pool.submit(work, offer, dispatched))
            remaining = started+duration-time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            for future in pending:
                future.result()
    finally:
        if handle:
            handle.close()
    elapsed = time.perf_counter()-started
    rows.sort(key=lambda r: r["offer_id"])
    events.sort(key=lambda e: e["at_seconds"])
    if [r["offer_id"] for r in rows] != list(range(len(schedule))):
        raise StudyError("Incomplete offer accounting")
    report = summarize(rows, duration, slo_ms)
    area, prior, last = 0.0, 0, 0.0
    for event in events:
        area += (event["at_seconds"]-last)*prior
        last, prior = event["at_seconds"], event["active_workers"]
    worker_time = sum(r.get("worker_ms", 0) for r in rows)/1000
    report.update(started_at=started_at, drained_elapsed_seconds=elapsed,
                  successful_per_drained_second=report["successful"]/elapsed,
                  worker_occupancy_seconds=area, sum_worker_seconds=worker_time,
                  occupancy_accounting_error_seconds=abs(area-worker_time))
    return report, rows, events


def verify_cell(folder, expected_plan_hash):
    folder = Path(folder)
    path = folder / "complete.json"
    if not path.exists():
        return None
    record = read_json(path)
    if (record.get("plan_hash") != expected_plan_hash or record.get("self_hash") !=
            fingerprint({k: v for k, v in record.items() if k != "self_hash"})):
        raise StudyError("Completed service cell identity mismatch")
    required = {"plan.json", "traces.jsonl", "occupancy.jsonl", "resources.jsonl"}
    if set(record.get("files", {})) != required:
        raise StudyError("Completed service cell evidence inventory mismatch")
    for name, digest in record["files"].items():
        source = folder/name
        if source.is_symlink() or not source.is_file() or file_hash(source) != digest:
            raise StudyError(f"Completed service evidence changed or missing: {name}")
    if fingerprint(read_json(folder/"plan.json")) != expected_plan_hash:
        raise StudyError("Saved cell plan mismatch")
    return record
