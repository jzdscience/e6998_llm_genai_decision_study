"""Budget-serving units against actual loopback HTTP; no model, GPU, Slurm or gold.

Fake frontends return the captured StartLux response structure
(answers.decision.choice/confidence/probabilities/type, model, usage).
"""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import random
import threading
import time
import zipfile

import pytest

from decision_study.budget_serving import client, export, inputs, measurement, protocol, router, serve, study, verify
from decision_study.budget_serving.ledger import InvariantViolation, Ledger, replay
from decision_study.common import StudyError, canonical, fingerprint, read_json, write_json
from decision_study.learnability import modeling
from decision_study.service_study.measurement import make_schedule

LABELS = list(protocol.LABELS)
CRITERIA = {"entailment": "The premise supports the hypothesis as necessarily true.",
            "neutral": "The premise neither entails nor contradicts the hypothesis.",
            "contradiction": "The premise supports the hypothesis as necessarily false."}
EXTRACTED = Path(__file__).resolve().parents[1] / ".validation/learnability-integration-2026-11-15/claude/scratch/pinned-audit-feasibility/extracted"


def example(i, split="eligible_dev"):
    return {"sample_id": f"artificial:{split}:{i:03d}", "task": "anli", "split": split, "group_id": f"g{i}", "round": f"R{i % 3 + 1}",
            "source_split": "dev", "state": f"Premise:\nAuthored premise {i}.\n\nHypothesis:\nAuthored hypothesis {i}.",
            "instructions": "Classify the relationship.", "labels": LABELS, "criteria": CRITERIA,
            "premise_hash": f"p{i // 2}", "input_hash": f"h{i}"}


def head(intercept, slopes):
    return {"kind": "logistic", "mean": [0.0] * 7, "scale": [1.0] * 7, "coefficients": [intercept, *slopes], "l2": 0.01}


# Local head: front correctness rises with P(entailment); fallback head flat. A confident
# entailment front has negative gain; neutral/contradiction-leaning fronts are candidates.
POLICY = {"cell": "startlux9b/n18000/front", "selected_index": 2, "l2": 0.01, "threshold": 0.05,
          "local_head": head(-1.0, [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "fallback_head": head(0.6, [0.0] * 7)}


def answer(probabilities):
    choice = max(probabilities, key=probabilities.get)
    return {"answers": {"decision": {"choice": choice, "confidence": probabilities[choice], "probabilities": probabilities,
                                     "type": "choice"}}, "model": "startlux", "usage": {"input_tokens": 37, "output_tokens": 0}}


def confident(label="entailment", top=0.9):
    rest = (1 - top) / 2
    return {x: (top if x == label else rest) for x in LABELS}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        data = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.calls.append(body)
            count = len(self.server.calls)
        action = self.server.behavior(body, count)
        try:
            time.sleep(action.get("delay", 0))
            payload = action.get("payload")
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(action.get("status", 200))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-request-id", self.headers.get("X-Request-ID", ""))
            self.end_headers()
            drip = action.get("drip")
            if drip:
                for index in range(len(data)):
                    self.wfile.write(data[index:index + 1])
                    self.wfile.flush()
                    time.sleep(drip)
            else:
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Fake(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, behavior):
        super().__init__(("127.0.0.1", 0), Handler)
        self.behavior, self.calls, self.lock = behavior, [], threading.Lock()
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def endpoint(self):
        return f"http://127.0.0.1:{self.server_address[1]}"

    def stop(self):
        self.shutdown()
        self.server_close()


@pytest.fixture
def servers():
    made = []

    def make(behavior):
        server = Fake(behavior)
        made.append(server)
        return server
    yield make
    for server in made:
        server.stop()


def ok(probabilities=None, delay=0.0):
    return lambda body, count: {"payload": answer(probabilities or confident()), "delay": delay}


def run_stage(condition, schedule, examples, endpoints, *, policy=POLICY, flags=None, duration=1.0, max_inflight=64,
              request_timeout=2.0, drain=2.0, workers=16, clock=time.monotonic):
    pair = condition in protocol.PAIR_CONDITIONS
    trace, ledger_file, journal_file = io.BytesIO(), io.BytesIO(), io.BytesIO()
    ledger = Ledger(enforced=condition in protocol.CAPPED, journal=ledger_file, clock=clock)
    journal = serve.FallbackJournal(journal_file, clock=clock) if pair else None
    call = serve.make_call(condition, mode="policy" if pair else "direct", endpoints=endpoints, examples=examples,
                           policy=policy if pair else None, flags=flags, ledger=ledger if condition in protocol.CAPPED else None,
                           journal=journal, clock=clock)
    rows, events, facts = measurement.run_open_loop(schedule, call, duration=duration, workers=workers, max_inflight=max_inflight,
                                                    drain_seconds=drain, request_timeout=request_timeout, ledger=ledger, trace=trace,
                                                    journal=journal, clock=clock)
    parse = lambda raw: [json.loads(line) for line in raw.getvalue().splitlines()]
    return {"rows": rows, "traces": parse(trace), "ledger": parse(ledger_file), "journal": parse(journal_file), "facts": facts,
            "endpoints": endpoints}


def check(evidence, condition, schedule, examples, *, policy=POLICY, flags=None, duration=1.0, request_timeout=2.0, drain=2.0,
          endpoints=None):
    pair = condition in protocol.PAIR_CONDITIONS
    return verify.verify_open_loop(evidence["traces"], schedule, condition=condition, mode="policy" if pair else "direct",
                                   label=None, examples=examples, policy=policy if pair else None, flags=flags,
                                   ledger_events=evidence["ledger"], journal_events=evidence["journal"], duration=duration,
                                   drain_seconds=drain, request_timeout=request_timeout,
                                   endpoints=endpoints or evidence["endpoints"])


def schedule_of(ids, gaps):
    rows, at = [], 0.0
    for offer_id, (sid, gap) in enumerate(zip(ids, gaps)):
        at += gap
        rows.append({"offer_id": offer_id, "sample_id": sid, "arrival_seconds": at})
    return rows


# ------------------------------------------------------------------------------------ protocol
def test_protocol_file_is_the_implemented_design_with_formula_watchdogs():
    p = protocol.load()
    assert p["conditions"] == list(protocol.CONDITIONS) and p["seed"] == 20261116
    assert protocol.watchdog_seconds(p, "cell", "pair") == 2 * 1200 + 30 * 30 + 120 + 60 + 5 + 180
    assert protocol.watchdog_seconds(p, "cell", "direct9b") == 1200 + 30 * 30 + 120 + 60 + 5 + 180
    assert protocol.watchdog_seconds(p, "pair", "pair") == 2 * 1200 + 90 * 30 + (30 + 60) + 3 * (10 + 60) + 5 + 180
    assert protocol.watchdog_seconds(p, "direct4b", "direct4b") == 1200 + 90 * 30 + (30 + 60) + (10 + 60) + 5 + 180
    for mutate in (lambda v: v["budget"].update(denominator=20), lambda v: v.update(duration_seconds=60),
                   lambda v: v["policy"].update(threshold=0.03), lambda v: v["memory_by_condition"]["budget"].update(startlux9b=0.45),
                   lambda v: v["preflight"]["policy_smoke"].update(conditions=["budget"]), lambda v: v.update(seed="20261116"),
                   lambda v: v["engine"].update(max_num_seqs=128), lambda v: v["source_files"].pop("freeze.json")):
        value = copy.deepcopy(p)
        mutate(value)
        with pytest.raises(StudyError):
            protocol.validate(value)


def test_source_identity_is_the_executing_closure_and_changes_with_a_dependency(tmp_path, monkeypatch):
    files = {str(p.relative_to(protocol.PROJECT)) for p in protocol.dependency_files()}
    for name in ("decision_study/budget_serving/runtime.py", "decision_study/budget_serving/worker.py",
                 "decision_study/learnability/modeling.py", "decision_study/adapters.py", "decision_study/runtime.py",
                 "decision_study/learnability_profile/pidfd.py", "decision_study/service_study/measurement.py",
                 "scripts/model_probe.py", "scripts/prepare_cuda_toolkit.py", "scripts/run_budget_serving.sh",
                 "configs/budget_serving_protocol.json", "configs/runtime_defaults.json"):
        assert name in files
    assert "decision_study/lab/app.py" not in files
    before = protocol.source_identity()["files"]
    real = protocol.file_hash
    monkeypatch.setattr(protocol, "file_hash", lambda p: "0" * 64 if p.name == "modeling.py" else real(p))
    assert protocol.source_identity()["files"] != before


# ------------------------------------------------------------------------------------ ledger
def test_ledger_grants_no_initial_or_future_credit_and_replays_exactly():
    journal = io.BytesIO()
    ledger = Ledger(enforced=True, journal=journal)
    for offer_id in range(9):
        ledger.offer(offer_id)
    assert ledger.request(3)["budget_denied"]  # 9 offers: floor(9/10) = 0 slots, whatever the full schedule holds
    ledger.offer(9)
    assert ledger.request(9)["budget_granted"] and ledger.request(8)["budget_denied"]
    with pytest.raises(InvariantViolation):
        ledger.request(9)  # one decision per offer: no retry, no refund
    with pytest.raises(InvariantViolation):
        ledger.request(10)  # an offer that has not arrived has no credit
    with pytest.raises(InvariantViolation):
        ledger.offer(11)  # offers are registered once each, in order
    events = [json.loads(x) for x in journal.getvalue().splitlines()]
    state = replay(events, numerator=1, denominator=10, enforced=True, offer_count=10)
    assert (state["reserved"], state["denied"], state["slots"]) == (1, 2, 1)
    for tamper in (lambda e: e[9].update(type="reserve", reserved_after=1, denied_after=0),   # grant beyond cap
                   lambda e: e[11].update(type="deny"),                                          # deny with a free slot
                   lambda e: e.pop(5),                                                           # missing offer
                   lambda e: e[0].update(at_seconds=1e9)):                                       # non-serial clock
        changed = copy.deepcopy(events)
        tamper(changed)
        with pytest.raises(InvariantViolation):
            replay(changed, numerator=1, denominator=10, enforced=True, offer_count=10)
    with pytest.raises(InvariantViolation):
        replay(events, numerator=1, denominator=10, enforced=False, offer_count=10)
    with pytest.raises(InvariantViolation):
        Ledger(enforced=False).request(0)


def test_concurrent_permits_never_exceed_the_wall_time_prefix_cap():
    journal = io.BytesIO()
    ledger = Ledger(enforced=True, journal=journal)
    total, registered, stop = 3000, [0], threading.Event()
    rng = random.Random(4)

    def schedule():
        for offer_id in range(total):
            ledger.offer(offer_id)
            registered[0] = offer_id + 1
            if offer_id % 50 == 0:
                time.sleep(0.001)
        stop.set()
    asked, lock = set(), threading.Lock()

    def worker(seed):
        local = random.Random(seed)
        while not stop.is_set() or len(asked) < registered[0] // 3:
            limit = registered[0]
            if not limit:
                continue
            offer_id = local.randrange(limit)
            with lock:
                if offer_id in asked:
                    if stop.is_set() and len(asked) >= limit // 3:
                        return
                    continue
                asked.add(offer_id)
            decision = ledger.request(offer_id)
            assert decision["reserved_after"] <= decision["offered_at_reservation"] // 10
    threads = [threading.Thread(target=schedule)] + [threading.Thread(target=worker, args=(rng.random(),)) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    events = [json.loads(x) for x in journal.getvalue().splitlines()]
    state = replay(events, numerator=1, denominator=10, enforced=True, offer_count=total)
    assert state["reserved"] <= total // 10 and state["reserved"] + state["denied"] == len(asked)
    offered = reserved = 0
    for event in events:  # the cap holds at every instant of the serial journal
        offered += event["type"] == "offer"
        reserved += event["type"] == "reserve"
        assert reserved <= offered // 10


# ------------------------------------------------------------------------------------ client
EXAMPLES = {e["sample_id"]: e for e in (example(i) for i in range(40))}


def test_slow_drip_body_is_cut_at_the_shared_deadline(servers):
    server = servers(lambda body, count: {"payload": answer(confident()), "drip": 0.05})
    began = time.monotonic()
    stage = client.call_model("startlux4b", server.endpoint, EXAMPLES["artificial:eligible_dev:000"], began + 0.4, base=began)
    elapsed = time.monotonic() - began
    assert stage["status"] == "deadline" and stage["error"]["type"] in {"deadline_during_http", "timeout_at_deadline"}
    assert stage["prediction"] is None and stage["attempts"] == 1 and elapsed < 1.5  # whole drip would take > 7 s


def test_complete_but_late_answer_is_a_deadline_with_raw_evidence(servers, monkeypatch):
    server = servers(ok())
    jump = threading.Event()
    real = client.parse_response

    def parse(*args):
        jump.set()  # the clock passes the deadline after the full body arrived
        return real(*args)
    monkeypatch.setattr(client, "parse_response", parse)
    clock = lambda: time.monotonic() + (100 if jump.is_set() else 0)
    began = clock()
    stage = client.call_model("startlux4b", server.endpoint, EXAMPLES["artificial:eligible_dev:001"], began + 5, base=began, clock=clock)
    assert stage["status"] == "deadline" and stage["late"] is True and stage["prediction"] == "entailment"
    assert stage["raw_response"]["answers"]["decision"]["type"] == "choice"
    verify.verify_stage(stage, EXAMPLES["artificial:eligible_dev:001"], "startlux4b", deadline_ms=5000, span_ms=stage["ended_ms"])


def test_http_transport_and_invalid_answers_are_one_attempt_data(servers):
    bad_choice = answer(confident())
    bad_choice["answers"]["decision"]["choice"] = "neutral"
    cases = {"http": {"status": 500, "payload": b'{"detail":"engine"}'}, "invalid_response": {"payload": b"not json"},
             "invalid_choice": {"payload": bad_choice}}
    for name, action in cases.items():
        server = servers(lambda body, count, action=action: action)
        began = time.monotonic()
        stage = client.call_model("startlux9b", server.endpoint, EXAMPLES["artificial:eligible_dev:002"], began + 5, base=began)
        assert stage["status"] == "error" and stage["attempts"] == 1 and len(server.calls) == 1
        assert stage["error"]["type"] == ("http" if name == "http" else "invalid_response")
    closed = Fake(ok())
    endpoint = closed.endpoint
    closed.stop()
    began = time.monotonic()
    stage = client.call_model("startlux9b", endpoint, EXAMPLES["artificial:eligible_dev:002"], began + 5, base=began)
    assert stage["status"] == "error" and stage["error"]["type"] == "transport"
    with pytest.raises(StudyError):
        client.endpoint_address("http://10.0.0.1:18380")


# ------------------------------------------------------------------------------------ routing
def test_live_front_features_are_scored_per_offer_never_cached(servers):
    sequence = [confident("neutral", 0.95), confident("neutral", 0.40), confident("contradiction", 0.60)]
    front = servers(lambda body, count: {"payload": answer(sequence[(count - 1) % 3])})
    back = servers(ok(confident("neutral")))
    sid = "artificial:eligible_dev:005"
    schedule = schedule_of([sid] * 6, [0.01] * 6)
    evidence = run_stage("threshold", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint})
    routes = [r["result"]["route"] for r in sorted(evidence["traces"], key=lambda r: r["offer_id"])]
    scores = {round(r["score"], 12) for r in routes}
    assert len(scores) == 3  # one input, three live answers, three gains
    for row in evidence["traces"]:
        stage = row["result"]["stages"][0]
        expected = router.live_gain(POLICY, EXAMPLES[sid], stage["prediction"], stage["probabilities"])
        assert row["result"]["route"]["score"] == expected
        assert row["result"]["route"]["candidate"] is (expected >= POLICY["threshold"])
    check(evidence, "threshold", schedule, EXAMPLES)


def test_late_front_never_starts_fallback_or_requests_a_permit(servers):
    front = servers(ok(delay=0.8))
    back = servers(ok())
    ids = list(EXAMPLES)[:3]
    schedule = schedule_of(ids, [0.05] * 3)
    evidence = run_stage("budget", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint},
                         request_timeout=0.3, duration=0.5, drain=1.0)
    assert all(r["status"] == "deadline" and len(r["result"]["stages"]) == 1 for r in evidence["traces"])
    assert all(r["result"]["route"]["candidate_reason"] == "front_deadline" for r in evidence["traces"])
    assert back.calls == [] and evidence["journal"] == [] and all(e["type"] == "offer" for e in evidence["ledger"])
    check(evidence, "budget", schedule, EXAMPLES, request_timeout=0.3, duration=0.5, drain=1.0)


def test_cap_counts_front_failures_and_a_denied_failed_front_stays_a_failure(servers):
    failing = {f"artificial:eligible_dev:{i:03d}" for i in range(0, 40, 3)}
    state_to_id = {e["state"]: e["sample_id"] for e in EXAMPLES.values()}

    def front_behavior(body, count):
        if state_to_id[body["state"]] in failing:
            return {"status": 503, "payload": b'{"detail":"overloaded"}'}
        return {"payload": answer(confident("entailment", 0.97))}  # confident: below threshold
    front, back = servers(front_behavior), servers(ok(confident("contradiction")))
    ids = list(EXAMPLES)
    schedule = schedule_of(ids, [0.005] * len(ids))
    evidence = run_stage("budget", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint})
    rows = sorted(evidence["traces"], key=lambda r: r["offer_id"])
    failed = [r for r in rows if r["sample_id"] in failing]
    assert failed and all(r["result"]["route"]["candidate_reason"] == "front_failure" for r in failed)
    assert all(r["result"]["route"]["permit_requested"] for r in failed)
    granted = [r for r in failed if r["result"]["route"]["budget_granted"]]
    denied = [r for r in failed if r["result"]["route"]["budget_denied"]]
    assert denied and all(r["status"] == "error" and r["result"]["error"]["type"] == "front_failure_permit_denied" for r in denied)
    assert granted and all(r["result"]["final_source"] == "startlux9b" for r in granted)
    state = check(evidence, "budget", schedule, EXAMPLES)
    assert state["reserved"] == len(granted) <= len(schedule) // 10 and state["denied"] == len(denied)
    assert len(back.calls) == len(granted)
    # The first failing offer (offer 0) arrives before any slot exists.
    assert rows[0]["result"]["route"]["budget_denied"] is True


def test_threshold_has_no_cap_and_random_uses_preassigned_flags_under_the_same_cap(servers):
    front, back = servers(ok(confident("neutral", 0.40))), servers(ok(confident("contradiction")))
    endpoints = {"startlux4b": front.endpoint, "startlux9b": back.endpoint}
    ids = list(EXAMPLES)
    schedule = schedule_of(ids, [0.004] * len(ids))
    threshold = run_stage("threshold", schedule, EXAMPLES, endpoints)
    assert all(r["result"]["route"]["candidate"] and r["result"]["final_source"] == "startlux9b" for r in threshold["traces"])
    assert check(threshold, "threshold", schedule, EXAMPLES)["reserved"] == 0
    flags = [int(i % 4 == 1) for i in range(len(schedule))]
    for policy in (POLICY, {**POLICY, "threshold": 0.9}):  # the learned threshold never changes random candidates
        evidence = run_stage("random", schedule, EXAMPLES, endpoints, policy=policy, flags=flags)
        rows = sorted(evidence["traces"], key=lambda r: r["offer_id"])
        assert [r["result"]["route"]["candidate"] for r in rows] == [bool(f) for f in flags]
        state = check(evidence, "random", schedule, EXAMPLES, policy=policy, flags=flags)
        assert state["reserved"] <= len(schedule) // 10 and state["permit_requests"] == sum(flags)


def test_expired_permit_is_charged_and_recorded_without_any_9b_request(servers, monkeypatch):
    sid = "artificial:eligible_dev:007"
    state_to_id = {e["state"]: e["sample_id"] for e in EXAMPLES.values()}
    unsure = lambda body: state_to_id[body["state"]] == sid
    front = servers(lambda body, count: {"payload": answer(confident("neutral", 0.40) if unsure(body) else confident("entailment", 0.99))})
    back = servers(ok())
    shift = [0.0]
    clock = lambda: time.monotonic() + shift[0]
    real = serve.router.live_gain

    def gain(policy, ex, *args):
        value = real(policy, ex, *args)
        if ex["sample_id"] == sid:
            shift[0] = 10.0  # the shared deadline passes after routing, before the 9B request
        return value
    monkeypatch.setattr(serve.router, "live_gain", gain)
    schedule = schedule_of([f"artificial:eligible_dev:{i:03d}" for i in range(10, 20)] + [sid], [0.01] * 10 + [0.05])
    evidence = run_stage("budget", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint}, clock=clock)
    row = next(r for r in evidence["traces"] if r["sample_id"] == sid)
    route = row["result"]["route"]
    assert route["budget_granted"] is True and route["fallback_attempted"] is False
    assert row["status"] == "deadline" and route["unused_reservation_reason"] == "deadline_before_fallback"
    assert [e["fallback_attempted"] for e in evidence["journal"]] == [False] and back.calls == []
    monkeypatch.setattr(serve.router, "live_gain", real)
    state = check(evidence, "budget", schedule, EXAMPLES)
    assert state["reserved"] == 1 and state["fallback_attempted"] == 0


def test_every_offer_is_registered_before_admission_including_shed_offers(servers):
    front, back = servers(ok(confident("neutral", 0.40), delay=0.25)), servers(ok())
    ids = list(EXAMPLES)
    schedule = schedule_of(ids, [0.002] * len(ids))
    evidence = run_stage("budget", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint},
                         workers=2, max_inflight=4)
    rejected = [r for r in evidence["traces"] if r["rejected"]]
    assert rejected and all(r["status"] == "client_rejected" for r in rejected)
    state = check(evidence, "budget", schedule, EXAMPLES)
    assert state["offered_received"] == len(schedule) and state["slots"] == len(schedule) // 10
    report = measurement.summarize(evidence["traces"], duration=1.0, slo_ms=500, secondary_slo_ms=1000)
    assert report["offered"] == len(schedule) and report["rejected"] == len(rejected)


def test_an_invariant_break_fails_the_stage_instead_of_becoming_data(servers):
    front, back = servers(ok(confident("neutral", 0.40))), servers(ok())
    schedule = schedule_of(list(EXAMPLES)[:5], [0.01] * 5)
    ledger = Ledger(enforced=True)
    call = serve.make_call("budget", mode="policy", endpoints={"startlux4b": front.endpoint, "startlux9b": back.endpoint},
                           examples=EXAMPLES, policy=POLICY, ledger=ledger, journal=serve.FallbackJournal(io.BytesIO()))

    def double(offer, deadline, begin):
        ledger.request(offer["offer_id"])  # a retry of the same permit is an invariant break
        return call(offer, deadline, begin)
    with pytest.raises(InvariantViolation):
        measurement.run_open_loop(schedule, double, duration=0.2, workers=2, max_inflight=4, drain_seconds=1, request_timeout=1,
                                  ledger=ledger, trace=io.BytesIO())
    used = Ledger(enforced=False)
    used.offer(0)
    with pytest.raises(InvariantViolation):
        measurement.run_open_loop(schedule, call, duration=0.2, workers=2, max_inflight=4, drain_seconds=1, request_timeout=1,
                                  ledger=used, trace=io.BytesIO())


def test_warmup_forces_both_roles_without_any_ledger(servers):
    front, back = servers(ok(confident("entailment", 0.99))), servers(ok(confident("neutral")))
    rows_examples = [example(i, "development") for i in range(5)]
    call = serve.make_call("budget", mode="forced_both", endpoints={"startlux4b": front.endpoint, "startlux9b": back.endpoint},
                           examples={e["sample_id"]: e for e in rows_examples}, label="warmup_forced_both")
    trace = io.BytesIO()
    rows = measurement.run_serial(rows_examples, call, request_timeout=2, trace=trace)
    assert len(front.calls) == len(back.calls) == 5 and all(r["result"]["final_source"] == "startlux9b" for r in rows)
    summary = verify.verify_serial(rows, rows_examples, condition="budget", mode="forced_both", label="warmup_forced_both",
                                   request_timeout=2, endpoints={"startlux4b": front.endpoint, "startlux9b": back.endpoint})
    assert summary["failures"] == 0 and summary["stage_attempts"] == {"startlux4b": 5, "startlux9b": 5}
    with pytest.raises(StudyError):
        serve.make_call("budget", mode="policy", endpoints={"startlux4b": "x", "startlux9b": "y"}, examples={}, policy=POLICY,
                        journal=serve.FallbackJournal(io.BytesIO()))  # a capped condition without its own ledger


def test_verifier_rejects_semantic_tampering_of_routes_scores_and_ledgers(servers):
    front, back = servers(ok(confident("neutral", 0.40))), servers(ok(confident("contradiction")))
    ids = list(EXAMPLES)
    schedule = schedule_of(ids, [0.004] * len(ids))
    evidence = run_stage("budget", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint})
    check(evidence, "budget", schedule, EXAMPLES)
    denied = next(i for i, r in enumerate(evidence["traces"]) if r["result"] and r["result"]["route"]["budget_denied"])
    granted = next(i for i, r in enumerate(evidence["traces"]) if r["result"] and r["result"]["route"]["budget_granted"])

    def score(e):
        e["traces"][granted]["result"]["route"]["score"] += 1e-6

    def flip(e):
        e["traces"][denied]["result"]["route"].update(budget_granted=True, budget_denied=False)

    def drop_event(e):
        index = next(i for i, x in enumerate(e["ledger"]) if x["type"] == "deny")
        del e["ledger"][index]
        for i, x in enumerate(e["ledger"], 1):
            x["seq"] = i

    def answer_swap(e):
        e["traces"][granted]["result"]["prediction"] = "entailment"

    def raw_swap(e):
        e["traces"][granted]["result"]["stages"][1]["raw_response"]["answers"]["decision"]["choice"] = "neutral"
    for tamper in (score, flip, drop_event, answer_swap, raw_swap):
        changed = copy.deepcopy(evidence)
        tamper(changed)
        with pytest.raises(InvariantViolation):
            check(changed, "budget", schedule, EXAMPLES)


# ------------------------------------------------------------------------------------ inputs
@pytest.mark.skipif(not (EXTRACTED / "fitting/models.json").is_file(), reason="accepted Study 4 extraction is not present")
def test_pinned_study4_population_and_policy_bind_by_sha_without_gold(tmp_path):
    p = protocol.load()
    assert set(inputs.verify_source_files(EXTRACTED, p)) == set(protocol.SOURCE_KEYS)
    workload = inputs.load_workload(EXTRACTED, p)
    document = inputs.load_policy(EXTRACTED, p, workload)
    assert len(workload) == 1737 and document["policy"]["threshold"] == 0.027109315183045934
    assert document["binding"]["historical_threshold_deferred_eligible_dev"] == 438
    assert inputs.verify_policy_document(document, p) == document["policy"]
    changed = copy.deepcopy(document)
    changed["cell"]["candidates"][1]["selection"]["correct"] += 1000  # violates the frozen selection rule
    with pytest.raises(StudyError):
        inputs.verify_policy_document(changed, p)
    changed = copy.deepcopy(document)
    changed["policy"]["threshold"] = 0.02
    with pytest.raises(StudyError):
        inputs.verify_policy_document(changed, p)
    # Jointly changing the stored cell and the copied policy keeps them self-consistent but not pinned.
    changed = copy.deepcopy(document)
    for target in (changed["cell"]["candidates"][2]["local_head"], changed["policy"]["local_head"]):
        target["coefficients"][1] += 0.5
    with pytest.raises(StudyError, match="accepted Study 4 models"):
        inputs.verify_policy_document(changed, p)
    # A changed rendered input keeps the five-field membership but not the pinned workload fingerprint.
    edited = copy.deepcopy(workload)
    edited[0]["state"] += " "
    with pytest.raises(StudyError, match="accepted eligible-dev"):
        inputs.validate_workload(edited, p)


def test_only_the_post_freeze_paths_read_gold():
    package = protocol.PROJECT / "decision_study/budget_serving"
    users = {path.name for path in package.glob("*.py") if "load_gold(" in path.read_text()}
    assert users == {"inputs.py", "evaluation.py"}
    source = (package / "evaluation.py").read_text()
    evaluate = source[source.index("def evaluate(ctx)"):source.index("def verify_evaluation")]
    assert evaluate.index("study.verify_freeze(ctx)") < evaluate.index("fixed_write(access") < evaluate.index("inputs.load_gold(")


def test_prepared_order_schedules_and_random_flags_are_seeded_and_label_free():
    p = protocol.load()
    order = study.planned_order(p)
    assert len(order) == 60 and order == study.planned_order(copy.deepcopy(p))
    for repeat in range(3):
        for rate in (10, 20, 40, 80):
            group = [c["condition"] for c in order if c["repeat"] == repeat and c["rate"] == rate]
            assert sorted(group) == sorted(protocol.CONDITIONS)
            assert len({c["schedule_file"] for c in order if c["repeat"] == repeat and c["rate"] == rate}) == 1
    first = study.random_flags(p, "r0-rate10", 5000)
    assert first == study.random_flags(p, "r0-rate10", 5000) and first != study.random_flags(p, "r1-rate10", 5000)
    assert 0.08 < sum(first["flags"]) / 5000 < 0.12
    workload = [example(i) for i in range(30)]
    assert make_schedule(workload, 40, 5, p["seed"] + 40) == make_schedule(workload, 40, 5, p["seed"] + 40)


# ------------------------------------------------------------------------------------ export
def test_large_archive_becomes_hashed_parts_that_rejoin_exactly(tmp_path, monkeypatch):
    output = tmp_path / "out"
    (output / "cells/x").mkdir(parents=True)
    rng = random.Random(1)
    (output / "cells/x/traces.jsonl").write_bytes(bytes(rng.getrandbits(8) for _ in range(9000)))
    (output / "runtime-work/cache").mkdir(parents=True)
    (output / "runtime-work/cache/weights.bin").write_bytes(b"w" * 100)
    (output / "cells/x/engine.log").write_bytes(b"l" * 3000)
    monkeypatch.setattr(export, "PART_BYTES", 4096)
    manifest_path = export.export(output, "result")
    manifest = read_json(manifest_path)
    assert manifest_path.name.endswith(".parts.json") and len(manifest["parts"]) >= 2
    joined = export.join(manifest_path, tmp_path / "joined.zip")
    with zipfile.ZipFile(joined) as archive:
        names = set(archive.namelist())
        bundle = json.loads(archive.read("BUNDLE_MANIFEST.json"))
    assert "cells/x/traces.jsonl" in names and not any(n.startswith("runtime-work") for n in names)
    assert "cells/x/engine.log" in names and bundle["files"]["cells/x/engine.log"]["bytes"] == 3000  # frozen logs are never dropped
    part = manifest_path.parent / manifest["parts"][0]["name"]
    original = part.read_bytes()
    part.write_bytes(b"x" + original[1:])
    with pytest.raises(StudyError):
        export.join(manifest_path, tmp_path / "again.zip")
    assert not (tmp_path / "again.zip").exists() and not list(tmp_path.glob(".again.zip.*"))  # no partial target blocks a retry
    part.write_bytes(original)
    assert export.join(manifest_path, tmp_path / "again.zip").read_bytes() == joined.read_bytes()
    for name, change in {"renamed": lambda m: m["parts"][0].update(name="../outside.part0001"),
                         "reordered": lambda m: m["parts"].reverse(),
                         "oversized": lambda m: m["parts"][0].update(bytes=export.PART_BYTES + 1),
                         "short total": lambda m: m.update(archive_bytes=m["archive_bytes"] - 1)}.items():
        forged = copy.deepcopy(manifest)
        change(forged)
        forged_path = manifest_path.with_name(manifest_path.name)
        write_json(forged_path, forged)
        with pytest.raises(StudyError):
            export.join(forged_path, tmp_path / f"{name.replace(' ', '-')}.zip")
        write_json(forged_path, manifest)
    alias = manifest_path.parent / "renamed-part.bin"  # identical bytes, sizes and hashes; only the name is not canonical
    alias.write_bytes((manifest_path.parent / manifest["parts"][0]["name"]).read_bytes())
    forged = copy.deepcopy(manifest)
    forged["parts"][0]["name"] = alias.name
    write_json(manifest_path, forged)
    with pytest.raises(StudyError, match="canonical"):
        export.join(manifest_path, tmp_path / "alias.zip")
    write_json(manifest_path, manifest)
    linked = manifest_path.parent / manifest["parts"][-1]["name"]
    data = linked.read_bytes()
    linked.unlink()
    (tmp_path / "elsewhere.bin").write_bytes(data)
    linked.symlink_to(tmp_path / "elsewhere.bin")
    with pytest.raises(StudyError):
        export.join(manifest_path, tmp_path / "symlinked.zip")


# ------------------------------------------------------------------------------------ runtime handoff
def _lifecycle(tmp_path):
    output = tmp_path / "out"
    folder = output / "cells/r0-rate10-budget/attempt-0001"
    folder.mkdir(parents=True)
    write_json(output / "runtime/binding.json", {"protocol": "anli-budget-serving-runtime-v1"})
    command = study.worker_command(output, folder)
    allocation = {"hostname": "node", "slurm_job_id": "123"}
    plan = {"kind": "cell", "worker_command": command, "allocation": allocation, "watchdog_seconds": 100.0, "attempt": folder.name}
    sup = {"command": command, "cwd": "x", "started_epoch": 1000.0, "deadline_epoch": 1100.0, "allocation": allocation,
           "binding_hash": fingerprint(read_json(output / "runtime/binding.json")), "ownership_selftest": {"file": "s", "sha256": "d"}}
    return output, folder, plan, sup


def _write_lifecycle(folder, sup, *, stop=None, done=None):
    write_json(folder / "supervision.json", sup)
    write_json(folder / "parent-stop.json", stop or {"plan_hash": fingerprint(sup), "gpu_checked": True, "gpu_compute_processes": [],
                                                     "ownership": "current descendants of sole-worker Linux subreaper"})
    (folder / "worker.log").write_text("worker log\n")
    write_json(folder / "worker-process.json", {"pid": 4321})
    value = {"status": "COMPLETE", "returncode": 0, "exited_epoch": 1050.0, "deadline_epoch": sup["deadline_epoch"],
             "plan_hash": fingerprint(sup), "allocation": sup["allocation"],
             "parent_stop_sha256": study.file_hash(folder / "parent-stop.json"), "supervision_sha256": study.file_hash(folder / "supervision.json")}
    write_json(folder / "supervised.json", {**value, **(done or {})})


def test_runtime_completion_receipts_are_accepted_only_when_bound_on_time_and_stopped(tmp_path, monkeypatch):
    from types import SimpleNamespace
    output, folder, plan, sup = _lifecycle(tmp_path)
    selftests = []
    monkeypatch.setattr(study.runtime, "verify_native_selftest", lambda *a, **k: selftests.append(k["allocation"]))
    runtime_checks = []
    monkeypatch.setattr(study.runtime, "verify_supervised", lambda out, attempt, offline: runtime_checks.append((attempt, offline)))
    ctx = SimpleNamespace(output=output)
    _write_lifecycle(folder, sup)
    assert study.verify_supervised(ctx, folder, plan, offline=False)["supervised_sha256"]
    assert selftests == [plan["allocation"]] and runtime_checks == [(folder, False)]  # the runtime verifier always runs first
    stale = {"plan_hash": "0" * 64}
    cases = {
        "late exit": dict(done={"exited_epoch": 1100.5}),
        "nonzero": dict(done={"returncode": 1}),
        "checkpoint receipt": dict(done={"status": "CHECKPOINT"}),
        "stale plan binding": dict(done=stale),
        "gpu left behind": dict(stop={"plan_hash": fingerprint(sup), "gpu_checked": True, "gpu_compute_processes": [{"pid": 9}],
                                      "ownership": "current descendants of sole-worker Linux subreaper"}),
        "unchecked gpu": dict(stop={"plan_hash": fingerprint(sup), "gpu_checked": False, "gpu_compute_processes": [],
                                    "ownership": "current descendants of sole-worker Linux subreaper"}),
    }
    for name, change in cases.items():
        _write_lifecycle(folder, sup, **change)  # every receipt rehashed consistently
        with pytest.raises(StudyError):
            study.verify_supervised(ctx, folder, plan, offline=False)
    for changed in ({**sup, "command": [*sup["command"][:3], "decision_study.other", *sup["command"][4:]]},
                    {**sup, "deadline_epoch": 1200.0}, {**sup, "allocation": {"hostname": "node", "slurm_job_id": "124"}}):
        _write_lifecycle(folder, changed)
        with pytest.raises(StudyError):
            study.verify_supervised(ctx, folder, plan, offline=False)
    _write_lifecycle(folder, sup)
    (folder / "worker.log").unlink()
    with pytest.raises(StudyError):
        study.verify_supervised(ctx, folder, plan, offline=False)


def test_wrapper_uses_this_checkout_explicit_path_and_no_ambient_deadline(tmp_path):
    import os
    import subprocess
    import sys
    repo = protocol.PROJECT
    root = tmp_path / "old-runtime"
    fake = root / ".venv-controller/bin/python"
    fake.parent.mkdir(parents=True)
    record = tmp_path / "record.json"
    fake.write_text(f"""#!/usr/bin/env bash
if [[ "$3" == "-c" ]]; then exec "{sys.executable}" "$@"; fi
exec "{sys.executable}" -c 'import json,os,sys; json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "env": {{k: os.environ.get(k) for k in ("PYTHONPATH","PYTHONNOUSERSITE","PYTHONHOME","PYTHONSTARTUP","DECISION_STUDY_DEADLINE_EPOCH","DECISION_BUDGET_WORKER_OUTPUT","HOURS_LEFT","SESSION_HOURS_LEFT","PYTHONDONTWRITEBYTECODE")}}}}, open(os.environ["WRAPPER_RECORD"], "w"))' "$@"
""")
    fake.chmod(0o755)
    env = {**os.environ, "WRAPPER_RECORD": str(record), "PYTHONPATH": "/somewhere/else", "PYTHONHOME": "/bad",
           "DECISION_STUDY_DEADLINE_EPOCH": "99999999999", "HOURS_LEFT": "24", "SESSION_HOURS_LEFT": "24",
           "DECISION_BUDGET_WORKER_OUTPUT": "/tmp/x"}
    script = repo / "scripts/run_budget_serving.sh"
    result = subprocess.run(["bash", str(script), str(root), "--retry-failed"], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    value = json.loads(record.read_text())
    resolved = str(root.resolve())
    assert value["argv"] == ["-B", "-s", "-m", "decision_study.budget_serving", "start", "--runtime-root", resolved,
                             "--output-dir", resolved + "/results/aux-anli-budget-serving-v1", "--retry-failed"]
    assert value["cwd"] == str(repo) and value["env"]["PYTHONPATH"] == str(repo) and value["env"]["PYTHONNOUSERSITE"] == "1"
    assert all(value["env"][k] is None for k in ("PYTHONHOME", "PYTHONSTARTUP", "DECISION_STUDY_DEADLINE_EPOCH",
                                                 "DECISION_BUDGET_WORKER_OUTPUT", "HOURS_LEFT", "SESSION_HOURS_LEFT"))
    for args in ([str(root), "--hours", "3"], [str(repo)]):
        assert subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=60).returncode == 2


def test_stage_endpoints_are_bound_to_the_deployed_role_map(servers):
    front, back = servers(ok(confident("neutral", 0.40))), servers(ok(confident("contradiction")))
    ids = list(EXAMPLES)[:12]
    schedule = schedule_of(ids, [0.004] * len(ids))
    evidence = run_stage("threshold", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint})
    check(evidence, "threshold", schedule, EXAMPLES)
    swapped = {"startlux4b": back.endpoint, "startlux9b": front.endpoint}
    with pytest.raises(InvariantViolation, match="endpoint"):
        check(evidence, "threshold", schedule, EXAMPLES, endpoints=swapped)
    both_front = copy.deepcopy(evidence)
    for row in both_front["traces"]:
        for stage in row["result"]["stages"]:
            stage["endpoint"] = front.endpoint  # a 4B answer relabelled as the 9B role
    with pytest.raises(InvariantViolation, match="endpoint"):
        check(both_front, "threshold", schedule, EXAMPLES)
    duplicate = copy.deepcopy(evidence)
    rows = [r for r in duplicate["traces"] if len(r["result"]["stages"]) == 2]
    rows[0]["result"]["stages"][1]["request_id"] = rows[0]["result"]["stages"][0]["request_id"]
    with pytest.raises(InvariantViolation, match="request identifier"):
        check(duplicate, "threshold", schedule, EXAMPLES)


# ------------------------------------------------------------------------------------ unusable HTTP 200 bodies
NAN_ANSWER = json.dumps({"answers": {"decision": {"choice": "entailment", "type": "choice",
                                                  "probabilities": {"entailment": float("nan"), "neutral": 0.5, "contradiction": 0.5}}},
                         "model": "startlux"}).encode()


@pytest.mark.parametrize("body,phase", [(b"null", "answer"), (b" null\n", "answer"), (b"{}", "answer"), (b"[1, 2]", "answer"),
                                        (NAN_ANSWER, "answer"), (b"nul", "json"), (b"", "json"), (b"\xff\xfe{}", "utf8")],
                         ids=["null", "padded-null", "object", "array", "nan", "truncated", "empty", "bad-utf8"])
def test_unusable_http_200_bodies_are_failed_calls_the_audit_accepts(servers, body, phase):
    server = servers(lambda request, count, body=body: {"payload": body})
    example_row = EXAMPLES["artificial:eligible_dev:003"]
    began = time.monotonic()
    stage = client.call_model("startlux4b", server.endpoint, example_row, began + 5, base=began)
    assert stage["status"] == "error" and stage["attempts"] == 1 and len(server.calls) == 1
    assert stage["error"]["type"] == "invalid_response" and stage["error"]["phase"] == phase
    assert stage["prediction"] is None and stage["probabilities"] is None and stage["response_bytes"] == len(body)
    if body.strip() == b"null":
        assert stage["raw_response"] is None and stage["raw_text"] == body.decode()  # JSON null keeps its text
    verify.verify_stage(stage, example_row, "startlux4b", deadline_ms=5000, span_ms=stage["ended_ms"], endpoint=server.endpoint)
    # The evidence row survives the trace writer (NaN, surrogates) and still verifies after reading it back.
    restored = json.loads(measurement.trace_line({"stage": stage}))["stage"]
    verify.verify_stage(restored, example_row, "startlux4b", deadline_ms=5000, span_ms=stage["ended_ms"], endpoint=server.endpoint)


def test_invalid_answer_records_cannot_relabel_a_valid_answer(servers):
    server = servers(ok(confident("neutral", 0.40)))
    example_row = EXAMPLES["artificial:eligible_dev:004"]
    began = time.monotonic()
    good = client.call_model("startlux4b", server.endpoint, example_row, began + 5, base=began)
    body = json.dumps(good["raw_response"])
    failed = dict(good, status="error", prediction=None, probabilities=None)
    forgeries = {
        "valid json called unparseable": dict(failed, raw_response=None, raw_text=body, response_bytes=len(body),
                                              error={"type": "invalid_response", "phase": "json"}),
        "valid object called null": dict(failed, raw_response=None, raw_text=body, response_bytes=len(body),
                                         error={"type": "invalid_response", "phase": "answer"}),
        "null claim with the real byte count": dict(failed, raw_response=None, raw_text="null", response_bytes=len(body),
                                                    error={"type": "invalid_response", "phase": "answer"}),
        "clean text called bad utf8": dict(failed, raw_response=None, raw_text=body, response_bytes=len(body),
                                           error={"type": "invalid_response", "phase": "utf8"}),
        "small body called oversized": dict(failed, raw_response=None, raw_text=None, error={"type": "invalid_response", "phase": "size"}),
        "no decode phase": dict(failed, raw_response=None, raw_text="nul", response_bytes=3, error={"type": "invalid_response"}),
        "valid raw kept": dict(failed, error={"type": "invalid_response", "phase": "answer"}),
    }
    for name, stage in forgeries.items():
        with pytest.raises(InvariantViolation):
            verify.verify_stage(stage, example_row, "startlux4b", deadline_ms=5000, span_ms=good["ended_ms"])


def test_null_and_nan_answers_stay_counted_offer_failures_and_capped_candidates(servers):
    state_to_id = {e["state"]: e["sample_id"] for e in EXAMPLES.values()}
    bodies = {0: b"null", 1: NAN_ANSWER}

    def front_behavior(request, count):
        index = int(state_to_id[request["state"]].rsplit(":", 1)[1])
        return {"payload": bodies.get(index % 4, answer(confident("entailment", 0.97)))}
    front, back = servers(front_behavior), servers(ok(confident("contradiction")))
    ids = list(EXAMPLES)
    schedule = schedule_of(ids, [0.004] * len(ids))
    direct = run_stage("direct4b", schedule, EXAMPLES, {"startlux4b": front.endpoint})
    check(direct, "direct4b", schedule, EXAMPLES)
    report = measurement.summarize(direct["traces"], duration=1.0, slo_ms=500, secondary_slo_ms=1000)
    assert report["status_counts"]["error"] == 20 and report["failure_types"] == {"model_call": 20}
    budget = run_stage("budget", schedule, EXAMPLES, {"startlux4b": front.endpoint, "startlux9b": back.endpoint})
    state = check(budget, "budget", schedule, EXAMPLES)
    reasons = [r["result"]["route"]["candidate_reason"] for r in budget["traces"]]
    assert reasons.count("front_failure") == 20 and state["permit_requests"] == 20 and state["reserved"] <= len(schedule) // 10
