"""Real captured StartLux payloads through the serving HTTP/parser/router boundary.

Only the model hardware is replaced by a local HTTP server. Payloads and frozen
heads come from the accepted Study 4 bundle; historical latency is not an expected
latency for these tests. The changed live output is an explicit perturbation,
never presented as new scientific evidence.
"""
from contextlib import contextmanager
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time

import pytest

from decision_study.budget_serving import client, router, serve, verify
from decision_study.budget_serving.ledger import InvariantViolation, Ledger
from decision_study.common import fingerprint


@pytest.fixture
def captured():
    return json.loads((Path(__file__).parent / "fixtures/budget_startlux_responses.json").read_text())


@contextmanager
def captured_server(raws, *, close_response=False):
    """Serve the exact payloads through the actual HTTP transport, once each."""
    requests = []
    payloads = list(raws)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append({"path": self.path, "body": body, "request_id": self.headers.get("X-Request-ID")})
            if index >= len(payloads):
                self.send_error(500, "Unexpected retry")
                return
            data = json.dumps(payloads[index], ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Typesafe-Request-ID", f"captured-{index}")
            if close_response:
                self.send_header("Connection", "close")
            self.end_headers()
            if close_response:
                # Prevent a buffered full body from masking the HTTPConnection
                # close-before-read behavior when the server will close.
                self.wfile.flush()
                time.sleep(.03)
            self.wfile.write(data)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("Local socket bind is not allowed by this execution sandbox")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive(), "Captured HTTP server did not stop"


def call(role, endpoint, example):
    base = time.monotonic()
    return client.call_model(role, endpoint, example, base + 5, base=base)


@pytest.mark.parametrize("role", ["startlux4b", "startlux9b"])
def test_captured_real_payload_passes_new_transport_and_preserves_optional_fields(captured, role):
    old = captured["responses"][role]
    raw = old["raw_response"]
    # Detect accidental fixture simplification: optional confidence and usage were
    # present in the real model output, and the raw payload SHA binds all fields.
    assert fingerprint(raw) == captured["sources"][role]["raw_response_canonical_sha256"]
    assert raw["model"] == "startlux"
    assert raw["answers"]["decision"]["confidence"] != max(old["probabilities"].values())
    with captured_server([raw]) as (endpoint, requests):
        stage = call(role, endpoint, captured["example"])
    assert stage["status"] == "ok", stage
    assert stage["role"] == role
    assert stage["interface"] == "startlux"
    assert stage["raw_response"] == raw
    assert stage["prediction"] == old["prediction"]
    assert stage["probabilities"] == old["probabilities"]
    assert stage["token_counts"] == old["token_counts"]
    assert stage["server_request_id"] == "captured-0"
    assert stage["request_id"] == requests[0]["request_id"]
    assert stage["request_sha256"] == fingerprint(requests[0]["body"])
    assert stage["http_started"] is True and stage["attempts"] == 1
    assert len(requests) == 1 and requests[0]["path"] == "/v1/systemone"
    assert requests[0]["body"]["think_mode"] == "never"
    assert requests[0]["body"]["none_check"] is False
    assert requests[0]["body"]["allow_none"] is False


def test_same_sample_live_probabilities_cross_frozen_threshold_without_cached_scores(captured):
    raw = captured["responses"]["startlux4b"]["raw_response"]
    changed = copy.deepcopy(raw)
    changed["answers"]["decision"].update(choice="neutral", probabilities={"entailment": .2, "neutral": .6, "contradiction": .2})
    # Keep the separate confidence field unchanged. Correct routing uses the live
    # three probabilities and chosen label, not this optional scalar or sample ID.
    raw9b = captured["responses"]["startlux9b"]["raw_response"]
    with captured_server([raw, changed]) as (endpoint, requests), captured_server([raw9b]) as (back, back_requests):
        policy_call = serve.make_call("threshold", mode="policy", endpoints={"startlux4b": endpoint, "startlux9b": back},
                                      examples={captured["example"]["sample_id"]: captured["example"]},
                                      policy=captured["policy"], journal=serve.FallbackJournal(io.BytesIO()))
        results = []
        for i in range(2):
            base = time.monotonic()
            results.append(policy_call({"offer_id": i, "sample_id": captured["example"]["sample_id"]}, base + 5, base))
    assert [x["final_source"] for x in results] == ["startlux4b", "startlux9b"]
    assert [x["route"]["fallback_attempted"] for x in results] == [False, True]
    assert len(back_requests) == 1
    assert results[1]["stages"][1]["raw_response"] == raw9b
    fronts = [x["stages"][0] for x in results]
    assert all(x["status"] == "ok" for x in fronts)
    gains = [router.live_gain(captured["policy"], captured["example"], x["prediction"], x["probabilities"]) for x in fronts]
    # Independent scalar logistic calculation from the published coefficients.
    # Hard-coded values prevent a production score helper from serving as its own oracle.
    assert gains == pytest.approx([0.005444424813571658, 0.05556939468565547], abs=1e-13)
    assert gains[0] < captured["policy"]["threshold"] < gains[1]
    assert requests[0]["body"] == requests[1]["body"]
    batch = router.batch_gains(captured["policy"], [captured["example"]] * 2,
                               [(x["prediction"], x["probabilities"]) for x in fronts])
    assert list(batch) == pytest.approx(gains, abs=1e-13)


def test_valid_captured_body_survives_connection_close_response(captured):
    raw = captured["responses"]["startlux4b"]["raw_response"]
    with captured_server([raw], close_response=True) as (endpoint, requests):
        stage = call("startlux4b", endpoint, captured["example"])
    assert stage["status"] == "ok", stage.get("error")
    assert stage["raw_response"] == raw
    assert stage["prediction"] == "entailment" and stage["attempts"] == 1
    assert len(requests) == 1


@pytest.mark.parametrize("malformation", ["boolean", "mass", "wrong_argmax", "nonfinite"])
def test_real_payload_with_malformed_probabilities_is_error_and_never_retried(captured, malformation):
    raw = copy.deepcopy(captured["responses"]["startlux4b"]["raw_response"])
    answer = raw["answers"]["decision"]
    if malformation == "boolean":
        answer["probabilities"]["entailment"] = True
    elif malformation == "mass":
        answer["probabilities"]["entailment"] = .4
    elif malformation == "wrong_argmax":
        answer["choice"] = "neutral"
    else:
        answer["probabilities"]["entailment"] = float("nan")
    with captured_server([raw]) as (endpoint, requests):
        stage = call("startlux4b", endpoint, captured["example"])
    assert stage["status"] == "error", stage
    assert stage["error"]["type"] == "invalid_response"
    assert stage["prediction"] is None and stage["probabilities"] is None
    assert stage["raw_response"] is not None
    assert stage["attempts"] == 1 and len(requests) == 1


def test_gold_changes_neither_http_request_nor_live_gain(captured):
    raw = copy.deepcopy(captured["responses"]["startlux4b"]["raw_response"])
    raw["answers"]["decision"].update(choice="neutral", probabilities={"entailment": .2, "neutral": .6, "contradiction": .2})
    raw9b = captured["responses"]["startlux9b"]["raw_response"]
    examples = [{**captured["example"], "gold": label, "correct": flag}
                for label, flag in [("entailment", True), ("contradiction", False)]]
    with captured_server([raw, raw]) as (endpoint, requests), captured_server([raw9b, raw9b]) as (back, back_requests):
        results = []
        for row in examples:
            policy_call = serve.make_call("threshold", mode="policy", endpoints={"startlux4b": endpoint, "startlux9b": back},
                                          examples={row["sample_id"]: row}, policy=captured["policy"],
                                          journal=serve.FallbackJournal(io.BytesIO()))
            base = time.monotonic()
            results.append(policy_call({"offer_id": 0, "sample_id": row["sample_id"]}, base + 5, base))
    stages = [x["stages"][0] for x in results]
    gains = [router.live_gain(captured["policy"], example, stage["prediction"], stage["probabilities"])
             for example, stage in zip(examples, stages)]
    assert all(stage["status"] == "ok" for stage in stages)
    assert requests[0]["body"] == requests[1]["body"]
    assert not {"gold", "correct"}.intersection(requests[0]["body"])
    assert gains == pytest.approx([0.05556939468565547] * 2, abs=1e-13)
    assert [x["final_source"] for x in results] == ["startlux9b"] * 2
    assert len(back_requests) == 2
    assert back_requests[0]["body"] == back_requests[1]["body"]


def test_reserved_fallback_that_expires_before_http_is_charged_but_not_attempted(captured, monkeypatch):
    """Dispatch delay must not turn an unused reservation into a model call."""
    raw = copy.deepcopy(captured["responses"]["startlux4b"]["raw_response"])
    raw["answers"]["decision"].update(choice="neutral", probabilities={"entailment": .2, "neutral": .6, "contradiction": .2})
    now = [0.]
    clock = lambda: now[0]
    budget = Ledger(enforced=True, clock=clock, origin=0.)
    for i in range(10):
        budget.offer(i)
    log = io.BytesIO()

    real_call = client.call_model

    def delayed_dispatch(role, *args, **kwargs):
        if role == "startlux9b":
            now[0] = 2.  # time passes after the route check, before actual HTTP
        return real_call(role, *args, **kwargs)

    monkeypatch.setattr(client, "call_model", delayed_dispatch)

    with captured_server([raw]) as (front, front_requests), captured_server([]) as (back, back_requests):
        policy_call = serve.make_call("budget", mode="policy", endpoints={"startlux4b": front, "startlux9b": back},
                                      examples={captured["example"]["sample_id"]: captured["example"]},
                                      policy=captured["policy"], ledger=budget,
                                      journal=serve.FallbackJournal(log, clock=clock, origin=0.), clock=clock)
        result = policy_call({"offer_id": 9, "sample_id": captured["example"]["sample_id"]}, 1., 0.)
    assert result["status"] == "deadline"
    assert result["route"]["budget_granted"] is True and budget.state()["reserved"] == 1
    assert len(front_requests) == 1 and not back_requests
    assert sum(s["attempts"] for s in result["stages"] if s["role"] == "startlux9b") == 0
    assert result["route"]["fallback_attempted"] is False
    assert result["route"]["unused_reservation_reason"] is not None
    events = [json.loads(line) for line in log.getvalue().splitlines()]
    assert events[-1]["fallback_attempted"] is False


@pytest.mark.parametrize("forgery", ["prediction", "probabilities", "raw_mass"])
def test_audit_reparses_raw_answer_instead_of_trusting_saved_prediction(captured, forgery):
    raw = captured["responses"]["startlux4b"]["raw_response"]
    with captured_server([raw]) as (endpoint, _):
        stage = call("startlux4b", endpoint, captured["example"])
    verify.verify_stage(stage, captured["example"], "startlux4b", deadline_ms=5000, span_ms=stage["ended_ms"])
    altered = copy.deepcopy(stage)
    if forgery == "prediction":
        altered["prediction"] = "neutral"
    elif forgery == "probabilities":
        altered["probabilities"] = {"entailment": .9, "neutral": .05, "contradiction": .05}
    else:
        altered["raw_response"]["answers"]["decision"]["probabilities"]["entailment"] = .4
    with pytest.raises(InvariantViolation, match="raw response"):
        verify.verify_stage(altered, captured["example"], "startlux4b", deadline_ms=5000, span_ms=altered["ended_ms"])


def test_audit_refuses_valid_raw_response_relabelled_as_parse_failure(captured):
    raw = captured["responses"]["startlux4b"]["raw_response"]
    with captured_server([raw]) as (endpoint, _):
        stage = call("startlux4b", endpoint, captured["example"])
    verify.verify_stage(stage, captured["example"], "startlux4b", deadline_ms=5000, span_ms=stage["ended_ms"])
    stage.update(status="error", prediction=None, probabilities=None,
                 error={"type": "invalid_response", "message": "forged parse failure"})
    with pytest.raises(InvariantViolation):
        verify.verify_stage(stage, captured["example"], "startlux4b", deadline_ms=5000, span_ms=stage["ended_ms"])
