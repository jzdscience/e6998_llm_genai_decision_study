"""What one admitted offer does in each condition.

``direct``: one call to the condition's only role.
``policy``: StartLux-4B first; the frozen gain (threshold, budget) or the preassigned
Bernoulli flag (random) makes a valid front answer a candidate, and a failed front is
always a candidate. ``threshold`` calls the 9B for every candidate within the shared
deadline. ``budget`` and ``random`` ask the causal ledger for one permit: a denied
valid front returns the front answer, a denied failed front stays a failure, and a
granted permit is charged even if the deadline expires before the HTTP call.
``forced_both``: preflight/warmup only; both roles for every input, no ledger.

Both stages share one absolute monotonic deadline. A front that is late or timed out
never starts the fallback. HTTP, transport, parse and router-input failures are
recorded as data; anything else escaping this module is an invariant break.
"""
from __future__ import annotations

import threading
import time

from ..common import StudyError, canonical
from . import client, router
from .protocol import BACK, CAPPED, FRONT, PAIR_CONDITIONS, SERVICES

MODES = ("direct", "policy", "forced_both")
FORCED_LABELS = ("preflight_serial_forced_both", "preflight_forced_both", "warmup_forced_both")


class FallbackJournal:
    """Serial record of every fallback step, attempted or charged-but-unused."""

    def __init__(self, handle, *, clock=time.monotonic, origin=None):
        self._handle, self._clock = handle, clock
        self.origin = clock() if origin is None else origin
        self._lock = threading.Lock()
        self._seen = set()
        self.sequence = 0

    def record(self, offer_id, reservation_sequence, attempted, reason):
        with self._lock:
            if offer_id in self._seen:
                raise StudyError("Two fallback steps for one offer")
            self._seen.add(offer_id)
            self.sequence += 1
            event = {"seq": self.sequence, "offer_id": offer_id, "reservation_sequence": reservation_sequence,
                     "fallback_attempted": bool(attempted), "unused_reason": reason, "at_seconds": self._clock() - self.origin}
            self._handle.write(canonical(event) + b"\n")
            self._handle.flush()


def _done(status, stages, *, prediction=None, source=None, route=None, error=None):
    if status == "ok" and not isinstance(prediction, str):
        raise StudyError("Successful offer without a returned label")
    return {"status": status, "prediction": prediction if status == "ok" else None,
            "final_source": source if status == "ok" else None, "stages": stages, "route": route, "error": error}


def empty_route(condition, policy, flags, offer_id):
    return {"mode": "policy", "condition": condition,
            "threshold": policy["threshold"] if condition in ("threshold", "budget") else None,
            "random_flag": bool(flags[offer_id]) if condition == "random" else None,
            "score": None, "router_ms": None, "candidate": False, "candidate_reason": None,
            "permit_requested": False, "budget_granted": None, "budget_denied": None, "reservation_sequence": None,
            "offered_at_reservation": None, "slots_at_reservation": None, "reserved_before": None, "reserved_after": None,
            "fallback_step": False, "fallback_attempted": False, "fallback_not_attempted_reason": None,
            "unused_reservation_reason": None}


def make_call(condition, *, mode, endpoints, examples, policy=None, flags=None, ledger=None, journal=None,
              label=None, clock=time.monotonic):
    """Return call(offer, deadline, begin) for one stage or cell."""
    if mode not in MODES or condition not in SERVICES:
        raise StudyError("Unknown condition or call mode")
    roles = SERVICES[condition]
    if set(endpoints) != set(roles):
        raise StudyError("Endpoints differ from the condition's services")
    if (mode == "direct") == (condition in PAIR_CONDITIONS):
        raise StudyError("Call mode does not belong to this condition")
    if mode == "forced_both" and label not in FORCED_LABELS:
        raise StudyError("Forced two-role calls need a preflight/warmup label")
    if mode == "policy":
        if policy is None or (condition == "random") != (flags is not None) or journal is None:
            raise StudyError("Policy calls need the frozen policy, a fallback journal and random flags only for random")
        if (condition in CAPPED) != (ledger is not None and ledger.enforced):
            raise StudyError("Only budget and random use an enforced ledger")

    def call(offer, deadline, begin):
        example = examples[offer["sample_id"]]
        offer_id = offer["offer_id"]

        def stage(role, on_attempt=None):
            return client.call_model(role, endpoints[role], example, deadline, base=begin, clock=clock, on_attempt=on_attempt)

        if mode == "direct":
            role = roles[0]
            only = stage(role)
            if only["status"] == "ok":
                return _done("ok", [only], prediction=only["prediction"], source=role)
            kind = "model_call" if only["status"] == "error" else "model_deadline"
            return _done(only["status"], [only], error={"type": kind, "role": role, "late_response": only["late"],
                                                        "detail": only["error"]})

        front = stage(FRONT)
        stages = [front]
        if mode == "forced_both":
            route = {"mode": "forced_both", "label": label, "fallback_step": False, "fallback_attempted": False,
                     "fallback_not_attempted_reason": None}
            if front["status"] == "deadline":
                return _done("deadline", stages, route=route, error={"type": "front_deadline", "late_response": front["late"]})
            return _fallback(offer_id, stages, route, front, deadline, None, stage, clock, journal, forced=True)

        route = empty_route(condition, policy, flags, offer_id)
        if front["status"] == "deadline":
            route["candidate_reason"] = "front_deadline"
            return _done("deadline", stages, route=route, error={"type": "front_deadline", "late_response": front["late"],
                                                                 "detail": front["error"]})
        if front["status"] == "ok":
            started = time.perf_counter()
            try:
                score = router.live_gain(policy, example, front["prediction"], front["probabilities"])
            except StudyError as exc:
                route.update(candidate_reason="router_input", router_ms=1000 * (time.perf_counter() - started))
                return _done("error", stages, route=route, error={"type": "router_input", "message": str(exc)})
            route.update(score=score, router_ms=1000 * (time.perf_counter() - started))
            if condition == "random":
                candidate = route["random_flag"]
                reason = "random_flag" if candidate else "random_no_flag"
            else:
                candidate = score >= policy["threshold"]
                reason = "gain_at_or_above_threshold" if candidate else "gain_below_threshold"
        else:
            candidate, reason = True, "front_failure"
        route.update(candidate=candidate, candidate_reason=reason)
        if not candidate:
            return _front_answer(stages, route, front, deadline, clock)
        if condition in CAPPED:
            route["permit_requested"] = True
            route.update(ledger.request(offer_id))
            if route["budget_denied"]:
                if front["status"] == "ok":
                    return _front_answer(stages, route, front, deadline, clock)
                return _done("error", stages, route=route, error={"type": "front_failure_permit_denied", "detail": front["error"]})
        return _fallback(offer_id, stages, route, front, deadline, route["reservation_sequence"], stage, clock, journal)

    return call


def _front_answer(stages, route, front, deadline, clock):
    if clock() > deadline:
        return _done("deadline", stages, route=route, error={"type": "deadline_after_routing"})
    return _done("ok", stages, prediction=front["prediction"], source=FRONT, route=route)


def _fallback(offer_id, stages, route, front, deadline, reservation, stage, clock, journal, *, forced=False):
    """Invoke the 9B stage once. The journal and route record an attempt only at the client's
    actual HTTP start; a charged permit whose deadline expires first stays charged and unused."""
    route["fallback_step"] = True

    def started():
        if journal is not None:
            journal.record(offer_id, reservation, True, None)
        route["fallback_attempted"] = True
    back = stage(BACK, on_attempt=started)
    stages.append(back)
    if not back["http_started"]:
        reason = "deadline_before_fallback"
        route["fallback_not_attempted_reason"] = reason
        if reservation is not None:
            route["unused_reservation_reason"] = reason
        if journal is not None:
            journal.record(offer_id, reservation, False, reason)
        return _done("deadline", stages, route=route, error={"type": "deadline_before_fallback"})
    if forced:
        if front["status"] == "ok" and back["status"] == "ok":
            return _done("ok", stages, prediction=back["prediction"], source=BACK, route=route)
        status = "error" if "error" in (front["status"], back["status"]) else "deadline"
        return _done(status, stages, route=route, error={"type": "forced_both_failure", "front": front["error"],
                                                         "back": back["error"], "late_response": back["late"]})
    if back["status"] == "ok":
        return _done("ok", stages, prediction=back["prediction"], source=BACK, route=route)
    kind = "fallback_call" if back["status"] == "error" else "fallback_deadline"
    return _done(back["status"], stages, route=route, error={"type": kind, "late_response": back["late"], "detail": back["error"]})
