"""Open-loop offered load, bounded client concurrency, serial stages and gold-free summaries.

The scheduler sleeps until each scheduled arrival, registers the offer in the stage's
ledger, then admits it only if fewer than ``max_inflight`` admitted offers are pending.
A worker that starts an offer gives both model stages min(request timeout, time left
until the end of the drain). Client queueing counts toward the SLO. Every offer keeps
exactly one trace row, including client rejections and failures.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading
import time

import numpy as np

from ..common import StudyError, utc_now
from ..service_study.measurement import validate_schedule
from .ledger import InvariantViolation
from .protocol import ROLES

RESULT_STATUSES = ("ok", "error", "deadline")
ROW_STATUSES = (*RESULT_STATUSES, "client_rejected")


def trace_line(row):
    """One trace row. A malformed model answer may hold NaN/Infinity or a lone surrogate, so rows use
    Python's JSON tokens for non-finite numbers and ASCII escapes; read_jsonl/json.loads read both back."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=True).encode("ascii") + b"\n"


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def check_result(result):
    if (not isinstance(result, dict) or result.get("status") not in RESULT_STATUSES or not isinstance(result.get("stages"), list)
            or (result["status"] == "ok") != isinstance(result.get("prediction"), str)):
        raise InvariantViolation("Malformed offer result")
    return result


def _empty(kind):
    return {"status": "deadline", "prediction": None, "final_source": None, "stages": [], "route": None, "error": {"type": kind}}


def complete_by(result, completed, deadline):
    """The shared deadline covers the whole offer: a success that returns after it is a deadline.

    The raw stages and route stay as evidence; the would-be answer is kept inside the error
    so the audit can replay the route and confirm that only the final boundary was missed.
    """
    if result["status"] != "ok" or completed <= deadline:
        return result
    return {**result, "status": "deadline", "prediction": None, "final_source": None,
            "error": {"type": "offer_after_deadline", "prediction": result["prediction"], "final_source": result["final_source"],
                      "late_by_ms": 1000 * (completed - deadline)}}


def run_open_loop(schedule, call, *, duration, workers, max_inflight, drain_seconds, request_timeout, ledger, trace,
                  journal=None, clock=time.monotonic, sleep=time.sleep):
    """Return (rows, occupancy events, run facts); raise on any invariant break."""
    validate_schedule(schedule, duration)
    if (not isinstance(workers, int) or isinstance(workers, bool) or not isinstance(max_inflight, int)
            or isinstance(max_inflight, bool) or workers < 1 or max_inflight < workers
            or not all(_positive(v) for v in (duration, drain_seconds, request_timeout))):
        raise StudyError("Invalid concurrency or timing bounds")
    slots, mutex, abort = threading.BoundedSemaphore(max_inflight), threading.Lock(), threading.Event()
    rows, events, futures, failures = [], [], [], []
    active = 0
    started_at = utc_now()
    origin = clock()
    ledger.start(origin)  # one time base for offers, permits, fallback steps and traces
    if journal is not None:
        journal.origin = origin
    cutoff = origin + duration + drain_seconds

    def append(row):
        with mutex:
            rows.append(row)
            trace.write(trace_line(row))
            trace.flush()

    def work(offer, dispatched):
        nonlocal active
        with mutex:  # the occupancy time and the counter change are one atomic transition
            begin = clock()
            active += 1
            events.append({"at_seconds": begin - origin, "active_workers": active})
        deadline = min(begin + request_timeout, cutoff)
        try:
            if abort.is_set():
                result = _empty("aborted_after_invariant_violation")
            elif deadline <= begin:
                result = _empty("client_queue_reached_drain_cutoff")
            else:
                result = check_result(call(offer, deadline, begin))
        except Exception as exc:  # request failures are data inside call(); this is an invariant break
            failures.append(exc)
            abort.set()
            result = {**_empty("invariant_violation"), "status": "error",
                      "error": {"type": "invariant_violation", "message": f"{type(exc).__name__}: {exc}"}}
        with mutex:
            end = clock()
            active -= 1
            events.append({"at_seconds": end - origin, "active_workers": active})
        slots.release()
        result = complete_by(result, end, deadline)
        append({**offer, "dispatch_seconds": dispatched, "rejected": False, "worker_start_seconds": begin - origin,
                "deadline_seconds": deadline - origin, "completion_seconds": end - origin,
                "arrival_to_completion_ms": 1000 * (end - origin - offer["arrival_seconds"]),
                "client_queue_ms": 1000 * (begin - origin - dispatched), "worker_ms": 1000 * (end - begin),
                "status": result["status"], "result": result})

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="budget-client") as pool:
            try:
                for offer in schedule:
                    if abort.is_set():
                        break
                    delay = origin + offer["arrival_seconds"] - clock()
                    if delay > 0:
                        sleep(delay)
                    dispatched = clock() - origin
                    ledger.offer(offer["offer_id"])  # before admission, shed offers included
                    if not slots.acquire(blocking=False):
                        append({**offer, "dispatch_seconds": dispatched, "rejected": True, "status": "client_rejected",
                                "reason": "bounded_client_inflight", "result": None})
                    else:
                        futures.append(pool.submit(work, offer, dispatched))
                remaining = origin + duration - clock()
                if remaining > 0 and not abort.is_set():
                    sleep(remaining)
            except BaseException:
                abort.set()
                for future in futures:
                    future.cancel()
                raise
    finally:
        trace.flush()
    if failures:
        raise failures[0]
    for future in futures:  # evidence writes happen after the call; their failure is a failed stage
        future.result()
    elapsed = clock() - origin
    rows.sort(key=lambda r: r["offer_id"])
    if [r["offer_id"] for r in rows] != list(range(len(schedule))):
        raise InvariantViolation("Incomplete offer accounting")
    events.sort(key=lambda e: e["at_seconds"])
    area, prior, last = 0.0, 0, 0.0
    for event in events:
        area += (event["at_seconds"] - last) * prior
        last, prior = event["at_seconds"], event["active_workers"]
    worker_seconds = sum(r.get("worker_ms", 0) for r in rows) / 1000
    facts = {"started_at": started_at, "drained_elapsed_seconds": elapsed, "worker_occupancy_seconds": area,
             "sum_worker_seconds": worker_seconds, "occupancy_accounting_error_seconds": abs(area - worker_seconds),
             "ledger": ledger.state()}
    return rows, events, facts


def run_serial(examples, call, *, request_timeout, trace, clock=time.monotonic):
    """One input at a time, each with its own shared two-stage deadline."""
    if not _positive(request_timeout):
        raise StudyError("Invalid serial request timeout")
    rows, origin = [], clock()
    for index, example in enumerate(examples):
        begin = clock()
        deadline = begin + request_timeout
        result = check_result(call({"offer_id": index, "sample_id": example["sample_id"]}, deadline, begin))
        end = clock()
        result = complete_by(result, end, deadline)
        row = {"index": index, "sample_id": example["sample_id"], "start_seconds": begin - origin,
               "deadline_seconds": deadline - origin, "end_seconds": end - origin, "elapsed_ms": 1000 * (end - begin),
               "status": result["status"], "result": result}
        trace.write(trace_line(row))
        trace.flush()
        rows.append(row)
    return rows


def _quantiles(values):
    return {f"p{q}": (float(np.quantile(values, q / 100)) if values else None) for q in (50, 95, 99)}


def _counts(values):
    return {str(k): v for k, v in sorted(Counter(str(v) for v in values).items())}


def summarize(rows, *, duration, slo_ms, secondary_slo_ms):
    """Gold-free service summary; every offered row stays in the denominators."""
    offered = len(rows)
    admitted = [r for r in rows if not r["rejected"]]
    success = [r for r in admitted if r["status"] == "ok"]

    def timely(limit):
        return sum(1 for r in success if r["arrival_to_completion_ms"] <= limit)
    stages = [s for r in admitted for s in r["result"]["stages"]]
    attempts = {role: sum(1 for s in stages if s["role"] == role and s["attempts"] == 1) for role in ROLES}
    report = {"offered": offered, "admitted": len(admitted), "rejected": offered - len(admitted), "successful": len(success),
              "failed_admitted": len(admitted) - len(success), "status_counts": _counts(r["status"] for r in rows),
              "failure_types": _counts((r["result"].get("error") or {}).get("type") for r in admitted if r["status"] != "ok"),
              "met_slo": {str(slo_ms): timely(slo_ms), str(secondary_slo_ms): timely(secondary_slo_ms)},
              "late_successful": {str(slo_ms): len(success) - timely(slo_ms),
                                  str(secondary_slo_ms): len(success) - timely(secondary_slo_ms)},
              "offered_rate_actual": offered / duration,
              "latency_ms_admitted": _quantiles([r["arrival_to_completion_ms"] for r in admitted]),
              "latency_ms_successful": _quantiles([r["arrival_to_completion_ms"] for r in success]),
              "client_queue_ms": _quantiles([r["client_queue_ms"] for r in admitted]),
              "dispatch_lag_ms": _quantiles([1000 * (r["dispatch_seconds"] - r["arrival_seconds"]) for r in rows]),
              "model_attempts": attempts, "model_attempts_per_offer": {k: (v / offered if offered else None) for k, v in attempts.items()},
              "final_source": _counts(r["result"]["final_source"] for r in success)}
    routes = [r["result"]["route"] for r in admitted if (r["result"].get("route") or {}).get("mode") == "policy"]
    if routes:
        report["routing"] = {
            "routed_offers": len(routes), "candidate_reasons": _counts(x["candidate_reason"] for x in routes),
            "candidates": sum(bool(x["candidate"]) for x in routes),
            "permit_requests": sum(bool(x["permit_requested"]) for x in routes),
            "budget_granted": sum(x["budget_granted"] is True for x in routes),
            "budget_denied": sum(x["budget_denied"] is True for x in routes),
            "fallback_steps": sum(bool(x["fallback_step"]) for x in routes),
            "fallback_attempted": sum(bool(x["fallback_attempted"]) for x in routes),
            "unused_reservations": sum(x["unused_reservation_reason"] is not None for x in routes),
            "router_ms": _quantiles([x["router_ms"] for x in routes if x["router_ms"] is not None])}
    return report
