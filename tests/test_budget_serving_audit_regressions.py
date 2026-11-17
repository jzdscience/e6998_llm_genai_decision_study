"""Independent audit regressions, with captured payloads and synthetic call times.

The payloads and frozen policy are real archived Study 4 evidence. All timings and
the one changed probability vector below are deliberate test fixtures, not new
model observations. No model, socket, GPU or historical result file is modified.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from decision_study.adapters import decision_request, parse_response
from decision_study.budget_serving import ledger, protocol, verify
from decision_study.common import StudyError, fingerprint


@pytest.mark.parametrize("at", [float("nan"), float("inf"), float("-inf"), -1.0],
                         ids=["nan", "positive-infinity", "negative-infinity", "negative"])
def test_ledger_replay_rejects_nonfinite_or_negative_time(at):
    # Parsing external JSON can produce nonfinite numbers even though the real
    # canonical writer refuses them. Audit must enforce the boundary itself.
    event = {"seq": 1, "type": "offer", "offer_id": 0, "offered_received": 1,
             "slots": 0, "reserved": 0, "denied": 0, "at_seconds": at}
    with pytest.raises(StudyError):
        ledger.replay([event], numerator=1, denominator=10, enforced=True, offer_count=1)


@pytest.mark.parametrize("at", [0.0, 0.125])
def test_ledger_replay_accepts_finite_nonnegative_time(at):
    event = {"seq": 1, "type": "offer", "offer_id": 0, "offered_received": 1,
             "slots": 0, "reserved": 0, "denied": 0, "at_seconds": at}
    result = ledger.replay([event], numerator=1, denominator=10, enforced=True, offer_count=1)
    assert result["offered_received"] == 1 and result["reserved"] == 0


def test_dependency_identity_covers_effective_size_control_configuration(tmp_path, monkeypatch):
    """The runtime invokes size_control._verify_model_metadata, which reads this JSON."""
    checkout = tmp_path / "checkout"
    for directory in ("decision_study", "configs", "scripts"):
        shutil.copytree(protocol.PROJECT / directory, checkout / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # During initial implementation this wrapper does not yet exist. Supply it
    # only in the isolated source copy so an unrelated missing file cannot mask
    # the actual effective-config omission. Once present, its real bytes copy.
    wrapper = checkout / "scripts/run_budget_serving.sh"
    if not wrapper.exists():
        wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    monkeypatch.setattr(protocol, "PROJECT", checkout)
    expected = "configs/size_control_protocol.json"
    inventory = {str(path.relative_to(checkout)) for path in protocol.dependency_files()}
    assert expected in inventory, "Source identity omits a configuration actually read by the runtime"
    before = protocol.source_identity()["files"]
    config = checkout / expected
    config.write_text(config.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = protocol.source_identity()["files"]
    assert before[expected] != after[expected], "Effective configuration bytes must change execution identity"


def _stage(example, raw, role, *, start_ms, end_ms):
    prediction, probabilities = parse_response(raw, example, "startlux")
    return {
        "role": role, "endpoint": "http://127.0.0.1:" + ("18380" if role == "startlux4b" else "18390"),
        "interface": "startlux", "request_id": role + ":" + str(example["sample_id"]),
        "request_sha256": fingerprint(decision_request(example, "startlux")),
        "started_ms": start_ms, "deadline_ms": 30000.0, "http_started": True, "attempts": 1,
        "http_status": 200, "server_request_id": None, "raw_response": raw, "raw_text": None,
        "response_bytes": len(json.dumps(raw).encode("utf-8")), "token_counts": raw.get("usage"),
        "prediction": prediction, "probabilities": probabilities, "late": False, "error": None,
        "status": "ok", "ended_ms": end_ms, "latency_ms": end_ms - start_ms,
    }


def _route(policy, *, candidate):
    # Scores were independently computed from the captured frozen coefficients;
    # this fixture does not ask the function under test to generate its oracle.
    return {
        "mode": "policy", "condition": "budget", "threshold": policy["threshold"], "random_flag": None,
        "score": 0.05556939468565547 if candidate else 0.005444424813571658, "router_ms": 0.1,
        "candidate": candidate,
        "candidate_reason": "gain_at_or_above_threshold" if candidate else "gain_below_threshold",
        "permit_requested": candidate, "budget_granted": True if candidate else None,
        "budget_denied": False if candidate else None, "reservation_sequence": 11 if candidate else None,
        "offered_at_reservation": 10 if candidate else None, "slots_at_reservation": 1 if candidate else None,
        "reserved_before": 0 if candidate else None, "reserved_after": 1 if candidate else None,
        "fallback_step": candidate, "fallback_attempted": candidate,
        "fallback_not_attempted_reason": None, "unused_reservation_reason": None,
    }


@pytest.fixture
def valid_budget_window():
    captured = json.loads((Path(__file__).parent / "fixtures/budget_startlux_responses.json").read_text())
    example, policy = captured["example"], captured["policy"]
    raw4 = captured["responses"]["startlux4b"]["raw_response"]
    raw9 = captured["responses"]["startlux9b"]["raw_response"]
    candidate_raw = copy.deepcopy(raw4)
    candidate_raw["answers"]["decision"].update(
        choice="neutral", probabilities={"entailment": 0.2, "neutral": 0.6, "contradiction": 0.2})
    schedule = [{"offer_id": i, "sample_id": example["sample_id"], "arrival_seconds": (i + 1) / 100}
                for i in range(10)]
    events, rows = [], []
    for offer in schedule:
        i, arrival = offer["offer_id"], offer["arrival_seconds"]
        candidate = i == 9
        events.append({"seq": i + 1, "type": "offer", "offer_id": i, "offered_received": i + 1,
                       "slots": (i + 1) // 10, "reserved": 0, "denied": 0, "at_seconds": arrival + 0.0005})
        start = arrival + 0.001
        front = _stage(example, candidate_raw if candidate else raw4, "startlux4b",
                       start_ms=0.0, end_ms=20.0 if candidate else 2.0)
        front["request_id"] += f":offer-{i}"
        stages = [front]
        end = 0.160 if candidate else start + 0.003
        if candidate:
            back = _stage(example, raw9, "startlux9b", start_ms=40.0, end_ms=50.0)
            back["request_id"] += f":offer-{i}"
            stages.append(back)
        result = {"status": "ok", "prediction": stages[-1]["prediction"],
                  "final_source": stages[-1]["role"], "stages": stages,
                  "route": _route(policy, candidate=candidate), "error": None}
        rows.append({**offer, "dispatch_seconds": arrival, "rejected": False, "worker_start_seconds": start,
                     "deadline_seconds": start + 30.0, "completion_seconds": end,
                     "arrival_to_completion_ms": 1000 * (end - arrival),
                     "client_queue_ms": 1000 * (start - arrival), "worker_ms": 1000 * (end - start),
                     "status": "ok", "result": result})
    events.append({"seq": 11, "type": "reserve", "offer_id": 9, "offered_at_reservation": 10,
                   "slots": 1, "reserved_before": 0, "reserved_after": 1, "denied_after": 0, "at_seconds": 0.130})
    journal = [{"seq": 1, "offer_id": 9, "reservation_sequence": 11, "fallback_attempted": True,
                "unused_reason": None, "at_seconds": 0.140}]
    return {
        "rows": rows, "schedule": schedule, "condition": "budget", "mode": "policy", "label": None,
        "examples": {example["sample_id"]: example}, "policy": policy, "flags": None,
        "ledger_events": events, "journal_events": journal, "duration": 1.0,
        "drain_seconds": 60.0, "request_timeout": 30.0,
    }


def test_full_audit_accepts_causal_captured_payload_fixture(valid_budget_window):
    result = verify.verify_open_loop(**valid_budget_window)
    assert result["offered_received"] == 10
    assert result["reserved"] == result["fallback_attempted"] == 1
    assert result["scored"] == 10


@pytest.mark.parametrize("start_ms", [5.0, 25.0], ids=["before-front-finishes", "after-front-before-permit"])
def test_full_audit_rejects_9b_started_before_front_or_permit(valid_budget_window, start_ms):
    # First prove this complete fixture passes; an unrelated schema failure must
    # not make the regression appear protected.
    verify.verify_open_loop(**valid_budget_window)
    changed = copy.deepcopy(valid_budget_window)
    back = changed["rows"][-1]["result"]["stages"][1]
    back["started_ms"] = start_ms
    back["latency_ms"] = back["ended_ms"] - start_ms
    # Front ends at .121, permit at .130, genuine fallback starts at .141.
    # Both mutations preserve answers, totals, ledger budget and all summaries.
    with pytest.raises(StudyError):
        verify.verify_open_loop(**changed)
