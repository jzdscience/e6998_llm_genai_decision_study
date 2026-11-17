"""Gold-free re-verification of raw stage evidence, shared by acceptance, resume, freeze and audit.

Nothing here trusts a summary or a hash alone. Every row is checked against the fixed
schedule, every stage against its exact request, its role's deployed endpoint and its raw
response (on failure paths too), every route against a recomputed live gain or preassigned
flag, every permit against an independent replay of the serial ledger, every fallback
against its journal and causal order (front answer, then permit, then actual 9B start),
and every success against the whole-offer deadline.
"""
from __future__ import annotations

import json
import math

from ..adapters import decision_request, parse_response
from ..common import StudyError, fingerprint
from . import router, runtime
from .client import MAX_RESPONSE_BYTES, TEXT_BYTES
from .ledger import InvariantViolation, replay
from .protocol import BACK, CAPPED, FRONT, PAIR_CONDITIONS, SERVICES
from .serve import FORCED_LABELS, empty_route

TOL_MS = 1e-6
TOL_S = 1e-9
SCORE_TOL = 1e-12
SLEEP_TOL_S = 1e-3
PINNED_ENDPOINTS = {role: f"http://127.0.0.1:{ports[0]}" for role, ports in runtime.PORTS.items()}
STAGE_KEYS = {"role", "endpoint", "interface", "request_id", "request_sha256", "started_ms", "deadline_ms", "http_started",
              "attempts", "http_status", "server_request_id", "raw_response", "raw_text", "response_bytes", "token_counts",
              "prediction", "probabilities", "late", "error", "status", "ended_ms", "latency_ms"}
RESULT_KEYS = {"status", "prediction", "final_source", "stages", "route", "error"}
ROW_KEYS = {"offer_id", "sample_id", "arrival_seconds", "dispatch_seconds", "rejected", "worker_start_seconds",
            "deadline_seconds", "completion_seconds", "arrival_to_completion_ms", "client_queue_ms", "worker_ms", "status", "result"}
REJECTED_KEYS = {"offer_id", "sample_id", "arrival_seconds", "dispatch_seconds", "rejected", "status", "reason", "result"}
SERIAL_KEYS = {"index", "sample_id", "start_seconds", "deadline_seconds", "end_seconds", "elapsed_ms", "status", "result"}
FORCED_ROUTE_KEYS = {"mode", "label", "fallback_step", "fallback_attempted", "fallback_not_attempted_reason"}
JOURNAL_KEYS = {"seq", "offer_id", "reservation_sequence", "fallback_attempted", "unused_reason", "at_seconds"}
LATE_KEYS = {"type", "prediction", "final_source", "late_by_ms"}
NOT_STARTED = "deadline_before_fallback"


def check(condition, message):
    if not condition:
        raise InvariantViolation(message)


def _real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(a, b, tolerance):
    return _real(a) and _real(b) and abs(a - b) <= tolerance


def request_sha256(example, cache=None):
    if cache is None:
        return fingerprint(decision_request(example, "startlux"))
    key = example["sample_id"]
    if key not in cache:
        cache[key] = fingerprint(decision_request(example, "startlux"))
    return cache[key]


def verify_stage(stage, example, role, *, deadline_ms, span_ms, cache=None, endpoint=None):
    check(isinstance(stage, dict) and set(stage) == STAGE_KEYS, "Stage record fields differ")
    check(stage["role"] == role and stage["interface"] == "startlux", "Stage role differs from the condition's order")
    check(endpoint is None or stage["endpoint"] == endpoint, "Stage endpoint is not its role's deployed endpoint")
    check(isinstance(stage["request_id"], str) and stage["request_id"], "Stage has no request identifier")
    check(stage["request_sha256"] == request_sha256(example, cache), "Stage request differs from the exact input")
    status = stage["status"]
    check(status in {"ok", "error", "deadline"} and stage["attempts"] in (0, 1) and isinstance(stage["late"], bool)
          and stage["http_started"] is (stage["attempts"] == 1), "Stage status/attempt record is invalid")
    for key in ("started_ms", "ended_ms", "latency_ms", "deadline_ms"):
        check(_real(stage[key]), "Stage timing is not finite")
    check(-TOL_MS <= stage["started_ms"] <= stage["ended_ms"] <= span_ms + TOL_MS
          and _close(stage["latency_ms"], stage["ended_ms"] - stage["started_ms"], TOL_MS)
          and _close(stage["deadline_ms"], deadline_ms, TOL_MS), "Stage timing is inconsistent with its offer")
    raw, parsed = stage["raw_response"], None
    if raw is not None:
        try:
            parsed = parse_response(raw, example, "startlux")
        except StudyError:
            parsed = None
    if not stage["http_started"]:
        check(status == "deadline" and stage["error"] == {"type": "deadline_before_request"} and raw is None
              and stage["ended_ms"] == stage["started_ms"], "Unstarted stage is not a pre-request deadline")
    if status == "ok" or stage["late"]:
        check(parsed is not None, "Accepted stage raw response does not parse")
        check(parsed == (stage["prediction"], stage["probabilities"]), "Stage answer differs from its raw response")
        check(stage["http_status"] == 200, "Answer without HTTP 200")
    if status == "ok":
        check(not stage["late"] and stage["error"] is None and stage["ended_ms"] <= stage["deadline_ms"] + TOL_MS,
              "Successful stage is late or carries an error")
        return
    if stage["late"]:
        check(status == "deadline" and stage["ended_ms"] > stage["deadline_ms"] - TOL_MS
              and stage["error"] == {"type": "response_after_deadline"}, "Late stage is not a deadline failure")
        return
    check(stage["prediction"] is None and stage["probabilities"] is None, "Failed stage keeps an answer")
    check(isinstance(stage["error"], dict) and stage["error"].get("type"), "Failed stage has no error type")
    check(parsed is None, "A valid raw answer is recorded as a failure")
    kind = stage["error"]["type"]
    if status == "deadline":
        check(kind in {"deadline_before_request", "deadline_during_http", "timeout_at_deadline"} and raw is None,
              "Deadline stage reason differs from the client boundary")
    elif kind == "http":
        check(stage["http_status"] not in (None, 200) and raw is None, "HTTP failure without a failing status")
    elif kind == "transport":
        check(raw is None and stage["http_started"], "Transport failure record differs")
    elif kind == "invalid_response":
        _verify_invalid_answer(stage, raw)
    else:
        raise InvariantViolation("Unknown stage failure type")


def _decoded(text):
    try:
        return True, json.loads(text)
    except (ValueError, RecursionError):
        return False, None


def _verify_invalid_answer(stage, raw):
    """Replay where decoding of an unusable HTTP 200 body stopped (raw answers that parse were rejected above)."""
    check(stage["http_status"] == 200, "Invalid answer without an HTTP 200 response")
    phase, text, size = stage["error"].get("phase"), stage["raw_text"], stage["response_bytes"]
    check(isinstance(size, int) and not isinstance(size, bool) and size >= 0, "Invalid answer without its byte count")
    whole = size <= TEXT_BYTES  # raw_text then holds the entire body
    if phase == "size":
        check(raw is None and text is None and size == MAX_RESPONSE_BYTES + 1, "Oversized-answer record differs from the client bound")
    elif phase == "utf8":
        check(raw is None and isinstance(text, str) and size <= MAX_RESPONSE_BYTES and (not whole or "\ufffd" in text),
              "Undecodable-answer record differs from its body text")
    elif phase == "json":
        check(raw is None and isinstance(text, str) and size <= MAX_RESPONSE_BYTES, "Unparseable-answer record differs")
        if whole:
            check(len(text.encode("utf-8")) == size, "Recorded body text differs from the received byte count")
            check(not _decoded(text)[0], "Raw text recorded as unparseable is valid JSON")
    elif phase == "answer":
        if raw is None:  # a JSON null body; only its kept text distinguishes it from an undecoded body
            check(isinstance(text, str), "JSON-null answer keeps no body text")
            if whole:
                ok, value = _decoded(text)
                check(len(text.encode("utf-8")) == size and ok and value is None, "JSON-null answer record differs from its body")
        else:
            check(text is None, "Decoded invalid answer carries a second body copy")
    else:
        raise InvariantViolation("Invalid-answer record lacks its decode phase")


def _expect_front_answer(result, front):
    if result["status"] == "ok":
        check(result["prediction"] == front["prediction"] and result["final_source"] == FRONT and result["error"] is None,
              "Returned front answer differs from the front stage")
    else:
        check(result["status"] == "deadline" and result["error"] == {"type": "deadline_after_routing"},
              "Front answer path failed without the deadline reason")


def verify_result(result, *, condition, mode, label, example, offer_id, policy, flags, decision, journal_entry,
                  journal_expected, deadline_ms, span_ms, cache=None, endpoints=None):
    """Return per-row facts; raise if the route differs from the fixed rules."""
    endpoints = PINNED_ENDPOINTS if endpoints is None else endpoints
    check(isinstance(result, dict) and set(result) == RESULT_KEYS and result["status"] in {"ok", "error", "deadline"},
          "Offer result fields differ")
    stages = result["stages"]
    check(isinstance(stages, list) and len(stages) <= 2, "Offer stage list is invalid")
    facts = {"permit_requested": False, "fallback_step": False, "fallback_attempted": False, "score": None}
    if not stages:
        check(result["status"] == "deadline" and result["route"] is None and result["error"] == {"type": "client_queue_reached_drain_cutoff"}
              and result["prediction"] is None and decision is None and journal_entry is None,
              "Offer without stages is not a drain-cutoff deadline")
        return facts
    order = [FRONT, BACK] if condition in PAIR_CONDITIONS else list(SERVICES[condition])
    for index, stage in enumerate(stages):
        check(index < len(order), "Too many stages for the condition")
        verify_stage(stage, example, order[index], deadline_ms=deadline_ms, span_ms=span_ms, cache=cache,
                     endpoint=endpoints.get(order[index]))
    if len(stages) == 2:
        check(stages[1]["started_ms"] >= stages[0]["ended_ms"] - TOL_MS, "9B stage started before its own front answer")
    if result["status"] == "ok":
        check(isinstance(result["prediction"], str) and result["error"] is None, "Successful offer without a label")
    else:
        check(result["prediction"] is None and result["final_source"] is None and isinstance(result["error"], dict),
              "Failed offer keeps a label or lacks its reason")
    if mode == "direct":
        check(condition not in PAIR_CONDITIONS and result["route"] is None and len(stages) == 1
              and decision is None and journal_entry is None, "Direct offer record differs")
        only = stages[0]
        if only["status"] == "ok":
            check(result["status"] == "ok" and result["prediction"] == only["prediction"] and result["final_source"] == only["role"],
                  "Direct answer differs from its stage")
        else:
            check(result["status"] == only["status"] and result["error"].get("type") ==
                  ("model_call" if only["status"] == "error" else "model_deadline"), "Direct failure differs from its stage")
        return facts
    check(condition in PAIR_CONDITIONS, "Two-role mode on a direct condition")
    route = result["route"]
    front = stages[0]
    if mode == "forced_both":
        check(isinstance(route, dict) and set(route) == FORCED_ROUTE_KEYS and route["mode"] == "forced_both"
              and route["label"] == label and label in FORCED_LABELS and decision is None, "Forced two-role route differs")
        if front["status"] == "deadline":
            check(len(stages) == 1 and result["status"] == "deadline" and not route["fallback_step"] and journal_entry is None
                  and result["error"].get("type") == "front_deadline", "Late forced front still continued")
            return facts
        facts["fallback_step"] = True
        facts["fallback_attempted"] = _verify_fallback(result, route, stages, journal_entry, journal_expected,
                                                       reservation=None, forced=True)
        return facts
    check(mode == "policy", "Unknown call mode")
    expected_route = empty_route(condition, policy, flags, offer_id)
    check(isinstance(route, dict) and set(route) == set(expected_route)
          and all(route[k] == expected_route[k] for k in ("mode", "condition", "threshold", "random_flag")),
          "Policy route identity differs")
    if front["status"] == "deadline":
        check(route["candidate_reason"] == "front_deadline" and route["candidate"] is False and route["score"] is None
              and len(stages) == 1 and result["status"] == "deadline" and result["error"].get("type") == "front_deadline"
              and not route["permit_requested"] and not route["fallback_step"] and decision is None and journal_entry is None,
              "A late or timed-out front continued routing")
        return facts
    if front["status"] == "ok":
        if route["candidate_reason"] == "router_input":
            check(route["score"] is None and result["status"] == "error" and result["error"].get("type") == "router_input"
                  and len(stages) == 1 and not route["candidate"] and not route["permit_requested"]
                  and decision is None and journal_entry is None, "Router-input failure continued routing")
            try:
                router.live_gain(policy, example, front["prediction"], front["probabilities"])
            except StudyError:
                return facts
            raise InvariantViolation("Router-input failure is not reproducible from the front answer")
        check(_real(route["score"]) and _real(route["router_ms"]) and route["router_ms"] >= 0, "Live score is missing")
        facts["score"] = route["score"]
        if condition == "random":
            candidate = route["random_flag"]
            reason = "random_flag" if candidate else "random_no_flag"
        else:
            candidate = route["score"] >= policy["threshold"]
            reason = "gain_at_or_above_threshold" if candidate else "gain_below_threshold"
    else:
        check(route["score"] is None and route["router_ms"] is None, "Failed front was scored")
        candidate, reason = True, "front_failure"
    check(route["candidate"] is candidate and route["candidate_reason"] == reason, "Candidate differs from the fixed rule")
    if not candidate:
        check(len(stages) == 1 and not route["permit_requested"] and not route["fallback_step"]
              and decision is None and journal_entry is None, "Non-candidate used the fallback or the ledger")
        _expect_front_answer(result, front)
        return facts
    budget_fields = ("budget_granted", "budget_denied", "reservation_sequence", "offered_at_reservation",
                     "slots_at_reservation", "reserved_before", "reserved_after")
    if condition in CAPPED:
        check(route["permit_requested"] is True and decision is not None, "Capped candidate without a ledger decision")
        facts["permit_requested"] = True
        expected = {"budget_granted": decision["granted"], "budget_denied": not decision["granted"],
                    "reservation_sequence": decision["seq"], "offered_at_reservation": decision["offered_at_reservation"],
                    "slots_at_reservation": decision["slots"], "reserved_before": decision["reserved_before"],
                    "reserved_after": decision["reserved_after"]}
        check({k: route[k] for k in budget_fields} == expected, "Route permit differs from the ledger replay")
        if not decision["granted"]:
            check(len(stages) == 1 and not route["fallback_step"] and journal_entry is None, "Denied permit still called 9B")
            if front["status"] == "ok":
                _expect_front_answer(result, front)
            else:
                check(result["status"] == "error" and result["error"].get("type") == "front_failure_permit_denied",
                      "Denied failed front did not stay a failure")
            return facts
    else:
        check(route["permit_requested"] is False and decision is None and all(route[k] is None for k in budget_fields),
              "Uncapped threshold condition used the ledger")
    facts["fallback_step"] = True
    facts["fallback_attempted"] = _verify_fallback(result, route, stages, journal_entry, journal_expected,
                                                   reservation=route["reservation_sequence"] if condition in CAPPED else None,
                                                   forced=False)
    return facts


def _verify_fallback(result, route, stages, entry, journal_expected, *, reservation, forced):
    """One 9B invocation; an attempt counts only if the client actually started HTTP."""
    check(route["fallback_step"] is True and len(stages) == 2, "Fallback step without its 9B stage record")
    front, back = stages
    attempted = back["http_started"]
    if journal_expected:
        check(entry is not None and entry["reservation_sequence"] == reservation and entry["fallback_attempted"] is attempted
              and entry["unused_reason"] == (None if attempted else NOT_STARTED), "Fallback journal differs from the 9B stage")
    else:
        check(entry is None, "Unexpected fallback journal entry")
    if not attempted:
        check(route["fallback_attempted"] is False and route["fallback_not_attempted_reason"] == NOT_STARTED
              and result["status"] == "deadline" and result["error"] == {"type": NOT_STARTED},
              "Unattempted fallback lacks its deadline reason")
        if not forced:
            check(route["unused_reservation_reason"] == (NOT_STARTED if reservation is not None else None),
                  "Charged but unused permit is not recorded")
        return False
    check(route["fallback_attempted"] is True and route["fallback_not_attempted_reason"] is None, "Fallback attempt record differs")
    if not forced:
        check(route["unused_reservation_reason"] is None, "Used permit recorded as unused")
    if forced:
        if front["status"] == "ok" and back["status"] == "ok":
            check(result["status"] == "ok" and result["prediction"] == back["prediction"] and result["final_source"] == BACK,
                  "Forced two-role answer differs from the 9B stage")
        else:
            expected = "error" if "error" in (front["status"], back["status"]) else "deadline"
            check(result["status"] == expected and result["error"].get("type") == "forced_both_failure",
                  "Forced two-role failure differs from its stages")
        return True
    if back["status"] == "ok":
        check(result["status"] == "ok" and result["prediction"] == back["prediction"] and result["final_source"] == BACK,
              "Fallback answer differs from the 9B stage")
    else:
        check(result["status"] == back["status"] and result["error"].get("type") ==
              ("fallback_call" if back["status"] == "error" else "fallback_deadline"), "Fallback failure differs from its stage")
    return True


def replay_result(result, completed, deadline):
    """Undo only the whole-offer deadline normalization, after checking that it was due."""
    error = result.get("error") or {}
    if error.get("type") == "offer_after_deadline":
        check(set(error) == LATE_KEYS and result["status"] == "deadline" and result["prediction"] is None
              and result["final_source"] is None and isinstance(error["prediction"], str)
              and completed > deadline - TOL_S and _close(error["late_by_ms"], 1000 * (completed - deadline), 1e-3),
              "Offer-deadline normalization differs from its completion time")
        return {**result, "status": "ok", "prediction": error["prediction"], "final_source": error["final_source"], "error": None}
    if result.get("status") == "ok":
        check(completed <= deadline + TOL_S, "Successful offer completed after its shared deadline")
    return result


def _journal(events):
    entries, last = {}, 0.0
    for index, event in enumerate(events, 1):
        check(isinstance(event, dict) and set(event) == JOURNAL_KEYS and event["seq"] == index and _real(event["at_seconds"])
              and event["at_seconds"] >= last and event["offer_id"] not in entries, "Fallback journal is not serial")
        last = event["at_seconds"]
        entries[event["offer_id"]] = event
    return entries


def verify_open_loop(rows, schedule, *, condition, mode, label, examples, policy, flags, ledger_events, journal_events,
                     duration, drain_seconds, request_timeout, endpoints=None):
    """Verify one open-loop stage or cell from its raw rows, ledger and fallback journal."""
    endpoints = PINNED_ENDPOINTS if endpoints is None else endpoints
    check(len(rows) == len(schedule), "Trace does not cover the offered schedule")
    enforced = mode == "policy" and condition in CAPPED
    replayed = replay(ledger_events, numerator=1, denominator=10, enforced=enforced, offer_count=len(schedule))
    offer_events = {e["offer_id"]: e for e in ledger_events if e["type"] == "offer"}
    decision_events = {e["offer_id"]: e for e in ledger_events if e["type"] != "offer"}
    journal_expected = mode in {"policy", "forced_both"}
    entries = _journal(journal_events) if journal_expected else {}
    check(journal_expected or not journal_events, "Direct stage has a fallback journal")
    if mode == "policy" and condition == "random":
        check(isinstance(flags, list) and len(flags) == len(schedule) and all(f in (0, 1) for f in flags), "Random flags differ")
    permits, steps, attempted, request_ids = set(), set(), set(), set()
    scored, cache = [], {}
    cutoff = duration + drain_seconds
    for offer, row in zip(schedule, sorted(rows, key=lambda r: r.get("offer_id", -1))):
        check({k: row.get(k) for k in ("offer_id", "sample_id", "arrival_seconds")} == offer, "Trace row differs from its scheduled offer")
        offer_id = offer["offer_id"]
        event = offer_events[offer_id]
        check(_real(row.get("dispatch_seconds")) and row["dispatch_seconds"] >= offer["arrival_seconds"] - SLEEP_TOL_S
              and event["at_seconds"] >= row["dispatch_seconds"] - TOL_S, "Offer was registered before its arrival/dispatch")
        if row.get("rejected") is True:
            check(set(row) == REJECTED_KEYS and row["status"] == "client_rejected" and row["reason"] == "bounded_client_inflight"
                  and row["result"] is None and offer_id not in decision_events and offer_id not in entries,
                  "Client rejection record differs")
            continue
        check(set(row) == ROW_KEYS and row["rejected"] is False and all(_real(row[k]) for k in
              ("worker_start_seconds", "deadline_seconds", "completion_seconds", "arrival_to_completion_ms", "client_queue_ms", "worker_ms")),
              "Admitted trace row fields differ")
        start, end = row["worker_start_seconds"], row["completion_seconds"]
        check(event["at_seconds"] <= start + TOL_S and start >= row["dispatch_seconds"] - TOL_S and end >= start
              and _close(row["deadline_seconds"], min(start + request_timeout, cutoff), TOL_S)
              and _close(row["arrival_to_completion_ms"], 1000 * (end - offer["arrival_seconds"]), TOL_MS)
              and _close(row["client_queue_ms"], 1000 * (start - row["dispatch_seconds"]), TOL_MS)
              and _close(row["worker_ms"], 1000 * (end - start), TOL_MS), "Admitted row timing is inconsistent")
        check(isinstance(row["result"], dict) and row["status"] == row["result"].get("status"), "Row status differs from its result")
        result = replay_result(row["result"], end, row["deadline_seconds"])
        decision = replayed["decisions"].get(offer_id)
        facts = verify_result(result, condition=condition, mode=mode, label=label, example=examples[offer["sample_id"]],
                              offer_id=offer_id, policy=policy, flags=flags, decision=decision, journal_entry=entries.get(offer_id),
                              journal_expected=journal_expected, deadline_ms=1000 * (row["deadline_seconds"] - start),
                              span_ms=row["worker_ms"], cache=cache, endpoints=endpoints)
        stages = result["stages"]
        for stage in stages:
            check(stage["request_id"] not in request_ids, "Duplicate request identifier across stages")
            request_ids.add(stage["request_id"])
        front_end = start + stages[0]["ended_ms"] / 1000 if stages else start
        when = None
        if facts["permit_requested"]:
            permits.add(offer_id)
            when = decision_events[offer_id]["at_seconds"]
            check(front_end - TOL_S <= when <= end + TOL_S, "Permit decision was not made after this offer's live front answer")
        if facts["fallback_step"]:
            steps.add(offer_id)
            back = stages[1]
            back_start, back_end = start + back["started_ms"] / 1000, start + back["ended_ms"] / 1000
            check(when is None or back_start >= when - TOL_S, "9B stage started before its budget permit")
            if offer_id in entries:
                at = entries[offer_id]["at_seconds"]
                check(at >= (front_end if when is None else when) - TOL_S, "Fallback journal precedes its front answer or permit")
                if back["http_started"]:
                    check(at <= back_start + TOL_S, "9B HTTP started before its journaled attempt")
                else:
                    check(back_end - TOL_S <= at <= end + TOL_S, "Unused permit journaled before the client's deadline check")
        if facts["fallback_attempted"]:
            attempted.add(offer_id)
        if facts["score"] is not None:
            scored.append((examples[offer["sample_id"]], (stages[0]["prediction"], stages[0]["probabilities"]), facts["score"]))
    check(set(replayed["decisions"]) == permits, "Ledger decisions differ from requesting candidates")
    check(set(entries) == (steps if journal_expected else set()), "Fallback journal differs from fallback steps")
    if scored:
        recomputed = router.batch_gains(policy, [s[0] for s in scored], [s[1] for s in scored])
        worst = max(abs(float(a) - s[2]) for a, s in zip(recomputed, scored))
        check(worst <= SCORE_TOL, f"Recorded live gains differ from recomputation (max {worst:.3g})")
    return {key: value for key, value in replayed.items() if key != "decisions"} | {
        "permit_requests": len(permits), "fallback_steps": len(steps), "fallback_attempted": len(attempted), "scored": len(scored)}


def verify_serial(rows, examples, *, condition, mode, label, request_timeout, policy=None, endpoints=None):
    check([r.get("sample_id") for r in rows] == [e["sample_id"] for e in examples], "Serial stage did not cover inputs in order")
    cache, request_ids = {}, set()
    for index, (row, example) in enumerate(zip(rows, examples)):
        check(isinstance(row, dict) and set(row) == SERIAL_KEYS and row["index"] == index
              and all(_real(row[k]) for k in ("start_seconds", "deadline_seconds", "end_seconds", "elapsed_ms"))
              and _close(row["deadline_seconds"], row["start_seconds"] + request_timeout, TOL_S)
              and row["end_seconds"] >= row["start_seconds"] and _close(row["elapsed_ms"], 1000 * (row["end_seconds"] - row["start_seconds"]), TOL_MS)
              and isinstance(row["result"], dict) and row["status"] == row["result"].get("status"), "Serial row timing or status differs")
        result = replay_result(row["result"], row["end_seconds"], row["deadline_seconds"])
        verify_result(result, condition=condition, mode=mode, label=label, example=example, offer_id=index,
                      policy=policy, flags=None, decision=None, journal_entry=None, journal_expected=False,
                      deadline_ms=1000 * request_timeout, span_ms=row["elapsed_ms"], cache=cache, endpoints=endpoints)
        for stage in result["stages"]:
            check(stage["request_id"] not in request_ids, "Duplicate request identifier across stages")
            request_ids.add(stage["request_id"])
    return serial_summary(rows)


def serial_summary(rows):
    stages = [s for r in rows for s in r["result"]["stages"]]
    return {"count": len(rows), "ok": sum(r["status"] == "ok" for r in rows), "failures": sum(r["status"] != "ok" for r in rows),
            "stage_attempts": {role: sum(1 for s in stages if s["role"] == role and s["attempts"] == 1) for role in (FRONT, BACK)},
            "stage_failures": sum(1 for s in stages if s["status"] != "ok")}


def role_overlap_seconds(rows):
    """Client-observed in-flight overlap of 4B and 9B HTTP calls of distinct offers (not GPU-kernel concurrency)."""
    spans = {FRONT: [], BACK: []}
    for row in rows:
        if row.get("rejected"):
            continue
        base = row["worker_start_seconds"]
        for stage in row["result"]["stages"]:
            if stage["attempts"] == 1:
                spans[stage["role"]].append((base + stage["started_ms"] / 1000, base + stage["ended_ms"] / 1000, row["offer_id"]))
    total = 0.0
    for b0, b1, b_id in spans[BACK]:
        for f0, f1, f_id in spans[FRONT]:
            if f_id != b_id:
                total += max(0.0, min(b1, f1) - max(b0, f0))
    return total
