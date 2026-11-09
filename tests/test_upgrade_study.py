"""Software acceptance for the ANLI upgrade-versus-fallback serving study.

No GPU or model is used. Real 0.50 single-model and 0.40 + 0.50 co-residency are
checked only by the node development preflight.
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from decision_study.common import StudyError, canonical, file_hash, fingerprint, write_jsonl
from decision_study.service_study.measurement import summarize
from decision_study.upgrade_study import client as up_client
from decision_study.upgrade_study import cli, identity, runtime, study

LABELS = ["entailment", "neutral", "contradiction"]
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get("UPGRADE_EVIDENCE_ROOT", ROOT.parent / "project/solo-experiments-2026-10-30/decision-study"))
V3_DIR = EVIDENCE / "results/returned-v3-2026-11-03-152529"
SIZE_DIR = EVIDENCE / "results/returned-size-control-complete-2026-11-06"
# The returned V1 benchmark output; on the node the same runtime lives at results/solo-v1.
V1_DIR = EVIDENCE / "results/returned-benchmark-2026-11-02-130735"


def example(sid="anli:x", state="Premise:\nA.\nHypothesis:\nB."):
    return {"sample_id": sid, "task": "anli", "split": "holdout", "labels": LABELS, "state": state,
            "instructions": "Classify.", "criteria": {label: label for label in LABELS}}


POLICY = {"task": "anli", "labels": LABELS, "family": "class_conditional", "include_length": False,
          "local_model": "startlux4b", "fallback_model": "qwen9b",
          # P(local correct) = confidence, so the score is 1 - confidence; defer when confidence <= 0.73.
          "local_head": {"kind": "logistic", "feature_mean": [0, 0, 0, 0], "feature_scale": [1, 1, 1, 1],
                         "coefficients": [0, 1, 0, 0, 0]},
          "decision": {"mode": "threshold", "threshold": 0.27, "comparison": ">="}}


# ---------------------------------------------------------------- protocol and plan

def test_protocol_and_fixed_plan():
    p = identity.protocol()
    order = study.planned_order(p)
    assert len(order) == 36 and order == study.planned_order(p)
    assert len({c["cell_id"] for c in order}) == 36
    for i in range(0, 36, 3):
        triple = order[i:i + 3]
        assert {c["condition"] for c in triple} == set(identity.CONDITIONS)
        assert len({(c["repeat"], c["rate"], c["schedule_file"]) for c in triple}) == 1
    workload = [example(f"anli:{i:04d}") for i in range(50)]
    schedules = study.schedules_for(p, workload)
    assert len(schedules) == 12 and schedules == study.schedules_for(p, workload)
    assert set(schedules) == {c["schedule_file"] for c in order}


# ---------------------------------------------------------------- client with a fake clock

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fake_client(clock, script, calls):
    class FakeClient:
        def __init__(self, model, endpoint, *, timeout, max_attempts):
            assert max_attempts == 1
            self.model, self.timeout = model, timeout

        def predict(self, ex):
            spec = script[self.model]
            calls.append((self.model, self.timeout))
            clock.t += spec.get("latency", 0.01)
            attempt = {"attempt": 1, "status": "error" if spec.get("error") else "ok", "request_id": f"{self.model}-{len(calls)}"}
            base = {"sample_id": ex["sample_id"], "task": ex["task"], "split": ex["split"], "model": spec.get("model", self.model),
                    "raw_response": {}, "attempts": [attempt], "latency_ms": 1000 * spec.get("latency", 0.01), "token_counts": {}}
            if spec.get("error"):
                return {**base, "status": "error", "prediction": None, "probabilities": None, "error": {"type": "http", "code": 500}}
            return {**base, "status": "ok", "prediction": spec["label"], "probabilities": spec.get("probabilities"), "error": None}
    return FakeClient


def probs(top, label="entailment"):
    rest = (1 - top) / 2
    return {name: (top if name == label else rest) for name in LABELS}


def run_call(script, timeout=30.0, condition="cascade", mode="formal", policy=POLICY):
    clock, calls = Clock(), []
    result = up_client.call(condition, {"startlux4b": "http://127.0.0.1:1", "startlux9b": "http://127.0.0.1:2",
                                        "qwen9b": "http://127.0.0.1:3"}, example(), timeout, policy,
                            mode=mode, client=fake_client(clock, script, calls), clock=clock)
    return result, calls


def test_direct_call_records_one_stage():
    result, calls = run_call({"startlux": {"label": "neutral", "probabilities": probs(0.9, "neutral")}}, condition="startlux9b", policy=None)
    assert result["status"] == "ok" and result["prediction"] == "neutral" and result["final_source"] == "startlux9b"
    assert [c[0] for c in calls] == ["startlux"] and len(result["stages"]) == 1
    assert result["stages"][0]["request_ids"] == ["startlux-1"]


def test_cascade_keeps_confident_front_answer():
    result, calls = run_call({"startlux": {"label": "entailment", "probabilities": probs(0.95)}, "qwen": {"label": "neutral"}})
    assert result["status"] == "ok" and result["prediction"] == "entailment" and not result["deferred"]
    assert result["route"]["reason"] == "score_below_threshold" and [c[0] for c in calls] == ["startlux"]


def test_cascade_defers_with_remaining_budget_only():
    result, calls = run_call({"startlux": {"label": "entailment", "probabilities": probs(0.5), "latency": 4.0},
                              "qwen": {"label": "neutral", "latency": 0.2}}, timeout=10.0)
    assert result["status"] == "ok" and result["prediction"] == "neutral" and result["final_source"] == "qwen9b"
    assert result["route"]["reason"] == "score_at_or_above_threshold" and result["route"]["fallback_called"]
    assert calls[0] == ("startlux", 10.0) and calls[1][0] == "qwen" and calls[1][1] == pytest.approx(6.0)
    assert [s["role"] for s in result["stages"]] == ["startlux4b", "qwen9b"]


def test_front_failure_is_forced_to_fallback():
    result, calls = run_call({"startlux": {"error": True}, "qwen": {"label": "contradiction"}})
    assert result["status"] == "ok" and result["prediction"] == "contradiction"
    assert result["route"]["reason"] == "front_failure" and [c[0] for c in calls] == ["startlux", "qwen"]


def test_exhausted_budget_records_deadline_without_fallback():
    result, calls = run_call({"startlux": {"error": True, "latency": 31.0}, "qwen": {"label": "neutral"}}, timeout=30.0)
    assert result["status"] == "deadline" and result["prediction"] is None
    assert result["route"]["fallback_called"] is False and [c[0] for c in calls] == ["startlux"]


def test_fallback_failure_is_the_request_failure():
    result, calls = run_call({"startlux": {"label": "entailment", "probabilities": probs(0.5)}, "qwen": {"error": True}})
    assert result["status"] == "error" and result["prediction"] is None and result["final_source"] is None
    assert result["error"]["type"] == "fallback_call"


def test_router_input_error_is_not_routed():
    bad = {"yes": 0.5, "no": 0.5}
    result, calls = run_call({"startlux": {"label": "entailment", "probabilities": bad}, "qwen": {"label": "neutral"}})
    assert result["status"] == "error" and result["error"]["type"] == "router_input"
    assert result["route"]["reason"] == "router_input" and [c[0] for c in calls] == ["startlux"]


def test_wrong_model_identity_is_rejected():
    with pytest.raises(StudyError):
        run_call({"startlux": {"label": "entailment", "probabilities": probs(0.9), "model": "kev"}})


def test_forced_fallback_mode_calls_both_and_needs_no_policy():
    result, calls = run_call({"startlux": {"label": "entailment", "probabilities": probs(0.99)}, "qwen": {"label": "neutral"}},
                             mode="forced_fallback", policy=None)
    assert result["route"]["reason"] == "preflight_forced_deferral" and [c[0] for c in calls] == ["startlux", "qwen"]


# ---------------------------------------------------------------- real HTTP through ModelClient

class FakeModels(BaseHTTPRequestHandler):
    hits = {"startlux": 0, "qwen": 0}

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state = body.get("state") or body["messages"][-1]["content"]
        if self.path == "/v1/systemone":
            FakeModels.hits["startlux"] += 1
            if "FAIL500" in state:
                self.send_response(500); self.end_headers(); return
            top = 0.5 if "LOW" in state else 0.95
            payload = ({"answers": {}} if "BADSCHEMA" in state else
                       {"answers": {"decision": {"choice": "entailment", "probabilities": probs(top)}}})
            slow = "SLOWFRONT" in state
        elif self.path == "/v1/chat/completions":
            FakeModels.hits["qwen"] += 1
            payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": "neutral"})}}], "usage": {}}
            slow = "SLOWBACK" in state
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not slow:
            self.wfile.write(data)
            return
        # Valid JSON streamed in chunks 45 ms apart: no single socket read waits long
        # enough to time out, yet the whole response takes about 0.3 s.
        pieces = [data[i:i + max(1, len(data) // 8)] for i in range(0, len(data), max(1, len(data) // 8))]
        for i, piece in enumerate(pieces):
            if i:
                time.sleep(0.045)
            self.wfile.write(piece)


@pytest.fixture
def fake_server():
    FakeModels.hits = {"startlux": 0, "qwen": 0}
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModels)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.mark.parametrize("state,expected,fallback_hits,reason", [
    ("plain", "entailment", 0, "score_below_threshold"),
    ("LOW confidence", "neutral", 1, "score_at_or_above_threshold"),
    ("FAIL500 front", "neutral", 1, "front_failure")])
def test_cascade_over_http(fake_server, state, expected, fallback_hits, reason):
    endpoints = {"startlux4b": fake_server, "qwen9b": fake_server}
    result = up_client.call("cascade", endpoints, example(state=state), 5.0, POLICY)
    assert result["status"] == "ok" and result["prediction"] == expected and result["route"]["reason"] == reason
    assert FakeModels.hits == {"startlux": 1, "qwen": fallback_hits}
    assert all(len(stage["attempts"]) == 1 for stage in result["stages"])


@pytest.mark.parametrize("condition,script,reason,called", [
    ("startlux4b", {"startlux": {"label": "entailment", "latency": 31.0}}, None, ["startlux"]),
    ("cascade", {"startlux": {"label": "entailment", "probabilities": probs(0.95), "latency": 31.0},
                 "qwen": {"label": "neutral"}}, "front_response_after_deadline", ["startlux"]),
    ("cascade", {"startlux": {"label": "entailment", "probabilities": probs(0.5), "latency": 1.0},
                 "qwen": {"label": "neutral", "latency": 30.0}}, "score_at_or_above_threshold", ["startlux", "qwen"])])
def test_a_success_returned_after_the_budget_is_a_deadline(condition, script, reason, called):
    result, calls = run_call(script, timeout=30.0, condition=condition, policy=POLICY if condition == "cascade" else None)
    assert result["status"] == "deadline" and result["prediction"] is None and result["final_source"] is None
    assert result["error"]["type"] == "response_after_deadline" and [c[0] for c in calls] == called
    assert result["stages"][-1]["status"] == "ok" and result["stages"][-1]["prediction"] is not None  # evidence kept
    if reason:
        assert result["route"]["reason"] == reason


def test_routing_that_ends_after_the_budget_is_a_deadline(monkeypatch):
    # The front answers inside the budget; routing finishes after it.
    clock, calls = Clock(), []
    real_route = up_client.route

    def slow_route(policy, example, row):
        clock.t += 2.0
        return real_route(policy, example, row)

    monkeypatch.setattr(up_client, "route", slow_route)
    script = {"startlux": {"label": "entailment", "probabilities": probs(0.95), "latency": 29.0}, "qwen": {"label": "neutral"}}
    result = up_client.call("cascade", {"startlux4b": "http://127.0.0.1:1", "qwen9b": "http://127.0.0.1:3"}, example(), 30.0,
                            POLICY, client=fake_client(clock, script, calls), clock=clock)
    assert result["status"] == "deadline" and result["route"]["reason"] == "score_below_threshold"
    assert result["route"]["fallback_called"] is False and [c[0] for c in calls] == ["startlux"]


@pytest.mark.parametrize("condition,state,reason,hits", [
    ("startlux4b", "SLOWFRONT plain", None, {"startlux": 1, "qwen": 0}),
    ("cascade", "SLOWFRONT plain", "front_response_after_deadline", {"startlux": 1, "qwen": 0}),
    ("cascade", "LOW SLOWBACK", "score_at_or_above_threshold", {"startlux": 1, "qwen": 1})])
def test_slowly_streamed_success_over_http_is_a_deadline(fake_server, condition, state, reason, hits):
    endpoints = {"startlux4b": fake_server, "qwen9b": fake_server}
    began = time.monotonic()
    result = up_client.call(condition, endpoints, example(state=state), 0.12, POLICY if condition == "cascade" else None)
    assert time.monotonic() - began > 0.2  # the streamed body really arrived after the 0.12 s budget
    assert result["status"] == "deadline" and result["prediction"] is None
    assert result["error"]["type"] == "response_after_deadline" and FakeModels.hits == hits
    assert result["stages"][-1]["status"] == "ok" and result["stages"][-1]["prediction"] is not None
    if reason:
        assert result["route"]["reason"] == reason


def test_invalid_front_schema_over_http_is_a_front_failure(fake_server):
    endpoints = {"startlux4b": fake_server, "qwen9b": fake_server}
    result = up_client.call("cascade", endpoints, example(state="BADSCHEMA front"), 5.0, POLICY)
    front, back = result["stages"]
    assert front["status"] not in ("ok",) and front["error"] is not None and front["prediction"] is None
    assert result["route"]["reason"] == "front_failure" and result["route"]["fallback_called"] is True
    assert result["status"] == "ok" and result["final_source"] == "qwen9b" and back["status"] == "ok"


# ---------------------------------------------------------------- historical regression on real evidence

@pytest.mark.skipif(not (V3_DIR.is_dir() and SIZE_DIR.is_dir()), reason="Needs the audited V3 and size-control returns")
def test_full_pool_routing_regression_and_policy_binding():
    workload, warmup = identity.load_inputs(V3_DIR)
    policy, binding = identity.load_policy(V3_DIR)
    assert binding["sha256"] == identity.POLICY_FREEZE_SHA256 and policy["decision"]["threshold"] == identity.POLICY_THRESHOLD
    history, info = identity.load_history(V3_DIR, SIZE_DIR, workload)
    assert info["historical_deferrals"] == {"class_conditional": 766, "confidence": 743}
    assert sum(v["deferred"] for v in history["startlux4b"].values()) == 766
    assert len(workload) == 3200 and len(warmup) == 90


# ---------------------------------------------------------------- service lifecycle

def _dummy_group():
    return subprocess.Popen(["sleep", "60"], start_new_session=True)


def _fake_services():
    """Fake launcher and nvidia-smi. Each start is a real process group, listed until it exits."""
    state = {"processes": {}, "launched": [], "fail": set(), "endpoint": None}

    @contextmanager
    def launcher(control, model):
        role = control.name
        if role in state["fail"]:
            raise RuntimeError(f"{role} failed to start")
        process = _dummy_group()
        state["processes"][role] = process
        state["launched"].append(process)
        (control / "runtime/last_services.json").write_text(json.dumps(
            {"started_at": datetime.now(timezone.utc).isoformat(), "processes": [{"name": role, "pid": process.pid, "log": ""}],
             "models": {model: {"health": {"ok": True}, "config": {}}}}))
        try:
            yield {model: state["endpoint"] or f"http://127.0.0.1/{role}"}
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)

    def lister():
        # Like nvidia-smi, list every launched group that is still alive, whichever deployment started it.
        return [{"pid": p.pid, "used_memory_mib": 1.0} for p in state["launched"] if p.poll() is None]

    def cleanup():
        for process in state["launched"]:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)

    return SimpleNamespace(launcher=launcher, lister=lister, state=state, cleanup=cleanup)


def _stop_quietly(process):
    """Test cleanup that tolerates a process a faulty controller has already killed."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    process.wait(timeout=10)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    fake = _fake_services()
    controls = {}
    for role, model in identity.INTERFACE.items():
        control = tmp_path / role
        (control / "runtime").mkdir(parents=True)
        (control / "runtime/selected_memory.json").write_text(json.dumps({model: 0.5}))
        controls[role] = control
    session = {"allocation": {"hostname": "test", "slurm_job_id": "1"}}
    monkeypatch.setattr(runtime, "validate", lambda output: (controls, {}, session, {}))
    monkeypatch.setattr(runtime, "legacy_environment", lambda binding, project: nullcontext())
    monkeypatch.setattr(runtime, "v3_runtime", SimpleNamespace(_environment=lambda b, s: nullcontext()))
    fake.output = tmp_path / "out"
    yield fake
    fake.cleanup()


def test_cascade_starts_front_then_fallback_and_stops_both(lifecycle):
    with runtime.deployment(lifecycle.output, "cascade", launcher=lifecycle.launcher, lister=lifecycle.lister, preflight=False) as service:
        assert set(service["endpoints"]) == {"startlux4b", "qwen9b"}
        groups = list(service["pids"])
        assert all(runtime.group_alive(g) for g in groups)
    assert not any(runtime.group_alive(g) for g in groups)
    assert service["record"]["stop_verification"]["gpu_compute_processes"] == []


def test_fallback_start_failure_stops_the_front(lifecycle):
    lifecycle.state["fail"].add("qwen9b")
    sink = {}
    with pytest.raises(RuntimeError):
        with runtime.deployment(lifecycle.output, "cascade", launcher=lifecycle.launcher, lister=lifecycle.lister,
                                preflight=False, sink=sink):
            pass
    front = lifecycle.state["processes"]["startlux4b"]
    # The front's own launcher cleanup (SIGTERM) must run; the stop verifier's SIGKILL fallback is not enough.
    assert front.returncode == -signal.SIGTERM and not runtime.group_alive(front.pid)
    assert "error" not in sink["stop_verification"] and sink["stop_verification"]["seconds"] < 5
    assert sink["stop_verification"]["residual_terminated"] == [] == sink["stop_verification"]["residual_killed"]


@pytest.mark.parametrize("error", [ValueError("measurement failed"), cli.Terminated("received signal 15")])
def test_measurement_error_or_interrupt_stops_both_services(lifecycle, error):
    sink = {}
    with pytest.raises(type(error)):
        with runtime.deployment(lifecycle.output, "cascade", launcher=lifecycle.launcher, lister=lifecycle.lister,
                                preflight=False, sink=sink):
            raise error
    assert len(lifecycle.state["launched"]) == 2 and all(p.poll() is not None for p in lifecycle.state["launched"])
    assert sink["stop_verification"]["gpu_compute_processes"] == []


def test_watchdog_stops_a_hung_cell_process(tmp_path, monkeypatch):
    real_popen, started = subprocess.Popen, []

    def hung(command, **kwargs):
        process = real_popen(["sleep", "60"], stdout=kwargs["stdout"], stderr=kwargs["stderr"], start_new_session=True)
        started.append(process)
        return process

    monkeypatch.setattr(study.subprocess, "Popen", hung)
    folder = tmp_path / "attempt"
    folder.mkdir()
    began = time.monotonic()
    with pytest.raises(StudyError, match="watchdog"):
        study._run_process(tmp_path, folder, {"watchdog_seconds": 0.6}, check_time=False)
    assert time.monotonic() - began < 15
    assert started[0].poll() is not None and not runtime.group_alive(started[0].pid)


def _child_writing_receipt(monkeypatch, write_after, exit_after):
    """Replace the cell child with a real sh process that writes complete.json, then exits 0."""
    real_popen, started = subprocess.Popen, []

    def child(command, **kwargs):
        folder = command[command.index("--attempt") + 1]
        script = f"sleep {write_after}; echo '{{}}' > '{folder}/complete.json'; sleep {exit_after}"
        process = real_popen(["sh", "-c", script], stdout=kwargs["stdout"], stderr=kwargs["stderr"], start_new_session=True)
        started.append(process)
        return process

    monkeypatch.setattr(study.subprocess, "Popen", child)
    return started


def test_receipt_written_before_the_watchdog_but_exit_after_it_is_not_accepted(tmp_path, monkeypatch):
    # Receipt at about 0.2 s, exit at about 0.65 s, watchdog 0.6 s: the exit lands between
    # the deadline and the next 0.25 s poll of the earlier loop, which then accepted it.
    _child_writing_receipt(monkeypatch, write_after=0.2, exit_after=0.45)
    folder = tmp_path / "attempt"
    folder.mkdir()
    with pytest.raises(StudyError, match="watchdog"):
        study._run_process(tmp_path, folder, {"watchdog_seconds": 0.6}, check_time=False)
    assert not (folder / "complete.json").exists() and (folder / "unaccepted-complete.json").exists()


def test_exit_observed_after_the_watchdog_is_not_accepted(tmp_path, monkeypatch):
    # The child writes its receipt and exits at once; once it has exited the parent's clock
    # reads past the deadline, as when the controller is descheduled before its check.
    started = _child_writing_receipt(monkeypatch, write_after=0.05, exit_after=0)
    folder = tmp_path / "attempt"
    folder.mkdir()
    real_monotonic = time.monotonic
    clock = SimpleNamespace(**{name: getattr(time, name) for name in ("time", "time_ns", "sleep", "perf_counter")})
    clock.monotonic = lambda: real_monotonic() + (5.0 if started and started[0].poll() is not None else 0.0)
    monkeypatch.setattr(study, "time", clock)
    with pytest.raises(StudyError, match="after the absolute parent watchdog"):
        study._run_process(tmp_path, folder, {"watchdog_seconds": 2.0}, check_time=False)
    assert not (folder / "complete.json").exists() and (folder / "unaccepted-complete.json").exists()


class Drip(BaseHTTPRequestHandler):
    """Valid headers, then one body byte every 0.1 s: no socket read times out and the body never completes."""
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "100000")
        self.end_headers()
        try:
            for _ in range(200):
                self.wfile.write(b" ")
                time.sleep(0.1)
        except OSError:
            pass


def test_serial_stage_is_bounded_by_the_watchdog_and_its_process_stopped(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Drip)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "out"
    out.mkdir()
    warmup = [example(f"anli:w{i}") for i in range(2)]
    (out / "warmup.json").write_bytes(_bytes(warmup))
    (out / "policy.json").write_bytes(_bytes({"policy": POLICY}))
    config = {"protocol": {**identity.protocol(), "request_timeout_seconds": 0.3, "watchdog_grace_seconds": 0.3},
              "file_sha256": {name: file_hash(out / name) for name in ("warmup.json", "policy.json")}}
    real_popen, started = subprocess.Popen, []

    def spy(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(study.subprocess, "Popen", spy)
    monkeypatch.setattr(runtime, "check_time", lambda output, required_seconds=0: {})
    endpoints = {"startlux4b": f"http://127.0.0.1:{server.server_address[1]}"}
    plan = study._serial_plan(config, [w["sample_id"] for w in warmup], "startlux4b", "warmup", endpoints, [])
    assert plan["watchdog_seconds"] == pytest.approx(0.9)  # 2 inputs x 0.3 s + 0.3 s grace
    began = time.monotonic()
    try:
        with pytest.raises(StudyError, match="watchdog"):
            study._serial_stage(out, out / "warmup", plan)
    finally:
        server.shutdown()
    assert time.monotonic() - began < 10
    assert started[0].poll() is not None and not runtime.group_alive(started[0].pid)
    assert not (out / "warmup/complete.json").exists() and not (out / "warmup/rows.json").exists()


def test_foreign_gpu_process_blocks_without_stopping_it(lifecycle):
    stranger = _dummy_group()
    try:
        foreign = lambda: [{"pid": stranger.pid, "used_memory_mib": 1.0}]
        with pytest.raises(StudyError, match="outside this deployment"):
            with runtime.deployment(lifecycle.output, "startlux9b", launcher=lifecycle.launcher, lister=foreign, preflight=False):
                pass
        assert lifecycle.state["processes"] == {} and stranger.poll() is None
    finally:
        _stop_quietly(stranger)


def _v1_like_failing_launcher(state):
    """Like the frozen V1 launcher: record each process as it starts, then fail readiness.

    The group leader starts a worker that ignores SIGTERM. The launcher's own cleanup
    signals the group and waits for the leader only, so the worker outlives it.
    """
    @contextmanager
    def launcher(control, model):
        if control.name != "qwen9b":
            with state["base"](control, model) as endpoints:
                yield endpoints
            return
        started = datetime.now(timezone.utc).isoformat()
        leader = subprocess.Popen(["sh", "-c", "(trap '' TERM; exec sleep 60) & wait"], start_new_session=True)
        state["leader"] = leader
        (control / "runtime/last_services.json").write_text(json.dumps(
            {"started_at": started, "models": {}, "processes": [{"name": "qwen-engine", "pid": leader.pid, "log": ""}]}))
        try:
            time.sleep(0.3)  # the worker is running with SIGTERM ignored
            state["worker"] = [pid for pid in runtime.owned_members({leader.pid}, since=0)[0] if pid != leader.pid]
            raise RuntimeError("qwen-engine readiness failed")
            yield  # pragma: no cover
        finally:
            os.killpg(leader.pid, signal.SIGTERM)
            leader.wait(timeout=10)
    return launcher


def test_partially_started_fallback_is_registered_and_its_residual_worker_killed(lifecycle, monkeypatch):
    monkeypatch.setattr(runtime, "STOP_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(runtime, "STOP_TERM_SECONDS", 0.5)
    state = {"base": lifecycle.launcher}
    stranger, sink = _dummy_group(), {}
    try:
        with pytest.raises(RuntimeError, match="readiness failed"):
            with runtime.deployment(lifecycle.output, "cascade", launcher=_v1_like_failing_launcher(state),
                                    lister=lambda: [], preflight=False, sink=sink):
                pass
        leader, worker = state["leader"], state["worker"]
        assert len(worker) == 1 and leader.returncode == -signal.SIGTERM
        stop = sink["stop_verification"]
        assert "error" not in stop, stop
        assert leader.pid in stop["owned_groups"] and stop["residual_terminated"] == worker == stop["residual_killed"]
        assert runtime.owned_members({leader.pid}, since=0) == ([], [])
        assert sink["roles"][-1]["role"] == "qwen9b" and sink["roles"][-1]["startup_failed"] is True
        assert stranger.poll() is None and lifecycle.state["processes"]["startlux4b"].returncode == -signal.SIGTERM
    finally:
        _stop_quietly(stranger)


def test_stop_verification_never_signals_group_members_older_than_the_deployment():
    # A group member that started before this deployment cannot be one of its processes.
    stranger = _dummy_group()
    try:
        with pytest.raises(StudyError, match="unverified group members"):
            runtime.verify_stopped({stranger.pid}, since=time.time() + 60, lister=lambda: [], grace=0, term_wait=0, kill_wait=0)
        assert stranger.poll() is None
    finally:
        _stop_quietly(stranger)


def test_stale_service_record_is_never_signalled(lifecycle):
    stranger = _dummy_group()
    control = lifecycle.output.parent / "qwen9b"
    try:
        # Left over from an earlier launch: its PID now belongs to an unrelated process.
        (control / "runtime/last_services.json").write_text(json.dumps(
            {"started_at": datetime.now(timezone.utc).isoformat(), "processes": [{"name": "qwen-engine", "pid": stranger.pid}]}))
        lifecycle.state["fail"].add("qwen9b")  # fails before starting anything or writing a record
        sink = {}
        with pytest.raises(RuntimeError):
            with runtime.deployment(lifecycle.output, "cascade", launcher=lifecycle.launcher, lister=lifecycle.lister,
                                    preflight=False, sink=sink):
                pass
        assert stranger.poll() is None and stranger.pid not in sink["stop_verification"]["owned_groups"]
    finally:
        _stop_quietly(stranger)


# ---------------------------------------------------------------- metrics and history comparison

def trace(offer, sid, status="ok", latency_ms=100.0, label="entailment", deferred=None, reason=None):
    result = None if status == "client_rejected" else {"status": status, "prediction": label if status == "ok" else None,
              "route": None if deferred is None else {"deferred": deferred, "reason": reason, "fallback_called": deferred}}
    return {"offer_id": offer, "sample_id": sid, "arrival_seconds": offer * 0.1, "dispatch_seconds": offer * 0.1,
            "rejected": status == "client_rejected", "status": status, "arrival_to_completion_ms": latency_ms, "result": result}


def test_metrics_use_actual_labels_and_both_slos():
    gold = {"a": "entailment", "b": "neutral", "c": "contradiction"}
    history = {"startlux4b": {"a": {"prediction": "entailment", "deferred": False}, "b": {"prediction": "entailment", "deferred": True},
                              "c": {"prediction": "contradiction", "deferred": False}},
               "qwen9b": {"a": {"prediction": "neutral"}, "b": {"prediction": "neutral"}, "c": {"prediction": "neutral"}}}
    rows = [trace(0, "a", latency_ms=100), trace(1, "b", latency_ms=700, label="neutral", deferred=True, reason="score_at_or_above_threshold"),
            trace(2, "c", latency_ms=1200, label="entailment"), trace(3, "a", status="client_rejected"),
            trace(4, "b", status="error")]
    rows[0]["result"]["route"] = {"deferred": False, "reason": "score_below_threshold", "fallback_called": False}
    rows[2]["result"]["route"] = {"deferred": False, "reason": "score_below_threshold", "fallback_called": False}
    primary = summarize(rows, 120, 500, gold)
    secondary = summarize(rows, 120, 1000, gold)
    assert (primary["offered"], primary["admitted"], primary["successful"], primary["met_slo"]) == (5, 4, 3, 1)
    assert primary["correct"] == 2 and primary["correct_met_slo"] == 1 and primary["correct_goodput_per_second"] == pytest.approx(1 / 120)
    assert secondary["met_slo"] == 2 and secondary["correct_met_slo"] == 2
    details = study.cell_details(rows, "cascade", gold, history, 500, 1000)
    assert details["late_successful"] == {"500": 2, "1000": 1} and details["rejected"] == 1
    assert details["history"]["matched_subset_historical_accuracy"] == pytest.approx(3 / 3)
    assert details["history"]["live_accuracy"] == pytest.approx(2 / 3)
    assert details["cascade"]["deferred"] == 1 and details["cascade"]["route_agreement_with_history"] == pytest.approx(1.0)


def test_sustained_rate_statement():
    p = {"conditions": ["startlux4b"], "rates_per_second": [10, 20, 40, 80]}
    cells = [{"condition": "startlux4b", "rate": r, "primary": {"slo_fraction_all_offered": f}}
             for r, f in ((10, 1.0), (20, 0.995), (40, 0.98), (80, 0.2))]
    assert study.sustained_rates(cells, p)["startlux4b"]["highest_passing_rate"] == 20


# ---------------------------------------------------------------- resume validation and gold gating

def _bytes(value):
    return canonical(value) + b"\n"


def _receipt_hashes(out):
    """Stand-in for preflight_hashes in cell-level tests: receipt bytes only.

    The real verifier checks every preflight file behind a receipt; the rehearsal's
    tamper test exercises it on real preflight evidence.
    """
    def hashes(output, config):
        result = {}
        for condition in identity.CONDITIONS:
            path = out / f"preflight/{condition}.json"
            if not path.exists():
                raise StudyError(f"Run the development preflight first; missing PASS receipt for {condition}")
            result[condition] = file_hash(path)
        return result
    return hashes


@pytest.fixture
def completed_cell(tmp_path, monkeypatch):
    out = tmp_path / "out"
    p = identity.protocol()
    warmup = [example(f"anli:w{i}") for i in range(2)]
    schedule = [{"offer_id": 0, "sample_id": "anli:x", "arrival_seconds": 0.5}, {"offer_id": 1, "sample_id": "anli:x", "arrival_seconds": 1.0}]
    files = {"workload.json": [example()], "warmup.json": warmup, "policy.json": {"policy": POLICY},
             "schedules/r0-rate10.json": schedule}
    for name, value in files.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(_bytes(value))
    cell = {"cell_id": "r0-cascade-rate10", "repeat": 0, "condition": "cascade", "rate": 10, "schedule_file": "schedules/r0-rate10.json"}
    config = {"protocol": p, "order": [cell], "file_sha256": {name: file_hash(out / name) for name in files}}
    for condition in identity.CONDITIONS:
        (out / "preflight").mkdir(exist_ok=True)
        (out / f"preflight/{condition}.json").write_bytes(_bytes({"status": "PASS", "settings_hash": fingerprint(config)}))
    monkeypatch.setattr(study, "preflight_hashes", _receipt_hashes(out))
    endpoints, pids = {"startlux4b": "http://127.0.0.1:1", "qwen9b": "http://127.0.0.1:2"}, [101, 102]
    records = out / "services/r0"
    records.mkdir(parents=True)
    (records / "start.json").write_bytes(_bytes({"x": 1}))
    warm = records / "warmup"
    warm.mkdir()
    warm_plan = study._serial_plan(config, [w["sample_id"] for w in warmup], "cascade", "warmup", endpoints, pids)
    (warm / "plan.json").write_bytes(_bytes(warm_plan))
    (warm / "rows.json").write_bytes(_bytes({"rows": [{"sample_id": w["sample_id"], "status": "ok", "stages": []} for w in warmup],
                                             "elapsed_seconds": 0.1}))
    warm_record = {"plan_hash": fingerprint(warm_plan), "files": {n: file_hash(warm / n) for n in ("plan.json", "rows.json")}}
    warm_record["self_hash"] = fingerprint(warm_record)
    (warm / "complete.json").write_bytes(_bytes(warm_record))
    folder = out / "cells" / cell["cell_id"] / "attempt-0001"
    folder.mkdir(parents=True)
    plan = {**cell, "settings_hash": fingerprint(config), "protocol": p, "schedule_sha256": config["file_sha256"][cell["schedule_file"]],
            "workload_file": "workload.json", "workload_sha256": config["file_sha256"]["workload.json"],
            "policy_file": "policy.json", "policy_sha256": config["file_sha256"]["policy.json"],
            "preflight_receipt_sha256": {c: file_hash(out / f"preflight/{c}.json") for c in identity.CONDITIONS},
            "watchdog_seconds": p["duration_seconds"] + p["drain_seconds"] + p["watchdog_grace_seconds"],
            "endpoints": endpoints, "service_pids": pids,
            "service_record_file": "services/r0/start.json", "service_record_sha256": file_hash(records / "start.json"),
            "warmup_record_file": "services/r0/warmup/rows.json", "warmup_record_sha256": file_hash(warm / "rows.json")}
    (folder / "plan.json").write_bytes(_bytes(plan))
    rows = [{**s, "dispatch_seconds": s["arrival_seconds"], "rejected": False, "status": "ok", "arrival_to_completion_ms": 50.0,
             "result": {"status": "ok", "prediction": "entailment"}} for s in schedule]
    write_jsonl(folder / "traces.jsonl", rows)
    write_jsonl(folder / "occupancy.jsonl", [])
    write_jsonl(folder / "resources.jsonl", [{"gpu": {"gpus": [{"used_mib": 1.0, "utilization_percent": 5.0}]}}])
    record = {"plan_hash": fingerprint(plan), "report": summarize(rows, p["duration_seconds"], p["slo_ms"]),
              "files": {n: file_hash(folder / n) for n in ("plan.json", "traces.jsonl", "occupancy.jsonl", "resources.jsonl")}}
    record["self_hash"] = fingerprint(record)
    (folder / "complete.json").write_bytes(_bytes(record))
    (records / "finished.json").write_bytes(_bytes({"at": "t", "cell_id": cell["cell_id"], "attempt": "attempt-0001",
                                                    "stop_verification": {"owned_groups": pids, "gpu_compute_processes": []}}))
    return SimpleNamespace(out=out, config=config, cell=cell, folder=folder, records=records)


def test_completed_cell_is_verified_from_raw_evidence(completed_cell):
    c = completed_cell
    assert study.completed(c.out, c.config, c.cell) is not None
    with (c.folder / "traces.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(StudyError, match="changed or missing"):
        study.completed(c.out, c.config, c.cell)


def test_changed_preflight_receipt_invalidates_completed_cell(completed_cell):
    c = completed_cell
    (c.out / "preflight/cascade.json").write_bytes(_bytes({"status": "PASS", "settings_hash": fingerprint(c.config), "note": "edited"}))
    with pytest.raises(StudyError, match="differs from the fixed plan"):
        study.completed(c.out, c.config, c.cell)


def test_gold_is_not_read_before_every_cell_completes(tmp_path, monkeypatch):
    p = identity.protocol()
    config = {"protocol": p, "order": study.planned_order(p), "file_sha256": {}, "v3_dir": str(tmp_path)}
    monkeypatch.setattr(study, "settings", lambda output, check_source=False: config)
    monkeypatch.setattr(study, "preflight_hashes", lambda output, config: {})
    monkeypatch.setattr(study, "completed", lambda output, config, cell, receipts=None: None)
    with pytest.raises(StudyError, match="All 36 fixed cells"):
        study.evaluate(tmp_path)
    assert not (tmp_path / "evaluation-access.json").exists()


def test_interrupted_attempt_is_preserved_and_needs_explicit_resume(completed_cell, monkeypatch):
    c = completed_cell
    monkeypatch.setattr(study, "settings", lambda output, check_source=False: c.config)
    with pytest.raises(StudyError, match="Cannot rerun a completed attempt"):
        study._attempt(c.out, c.cell, resume_interrupted=True)
    (c.folder / "complete.json").unlink()  # the cell process died before writing its receipt
    kept = {path.name: path.read_bytes() for path in c.folder.iterdir()}
    with pytest.raises(StudyError, match="interrupted attempt exists"):
        study.run(c.out)
    with pytest.raises(StudyError, match="Interrupted attempt preserved"):
        study._attempt(c.out, c.cell, resume_interrupted=False)
    folder = study._attempt(c.out, c.cell, resume_interrupted=True)
    assert folder.name == "attempt-0002"
    assert {path.name: path.read_bytes() for path in c.folder.iterdir()} == kept
    record = json.loads((folder / "attempt.json").read_text())
    assert record["prior_interrupted_attempts"] == ["attempt-0001"] and record["explicit_resume_acknowledgment"] is True


@pytest.mark.parametrize("finished", [None, {"error": "Owned services did not stop"}])
def test_measured_cell_without_a_confirmed_stop_is_not_accepted(completed_cell, monkeypatch, finished):
    c = completed_cell
    monkeypatch.setattr(study, "settings", lambda output, check_source=False: c.config)
    if finished is None:
        (c.records / "finished.json").unlink()  # the controller died before verifying the stop
    else:
        (c.records / "finished.json").write_bytes(_bytes({"at": "t", "cell_id": c.cell["cell_id"], "attempt": "attempt-0001",
                                                          "stop_verification": finished}))
    assert study.measured(c.out, c.config, c.cell) is not None and study.completed(c.out, c.config, c.cell) is None
    with pytest.raises(StudyError, match="All 36 fixed cells"):
        study.freeze(c.out)
    with pytest.raises(StudyError, match="interrupted attempt exists"):
        study.run(c.out)
    receipt = (c.folder / "complete.json").read_bytes()
    folder = study._attempt(c.out, c.cell, resume_interrupted=True)  # measured again in a new attempt
    assert folder.name == "attempt-0002" and not (c.folder / "complete.json").exists()
    assert (c.folder / "unaccepted-complete.json").read_bytes() == receipt and (c.folder / "unaccepted.json").exists()


def test_pairing_flags_a_repeat_and_rate_split_across_allocations():
    first, second = {"hostname": "n1", "slurm_job_id": "101"}, {"hostname": "n1", "slurm_job_id": "102"}
    cells = [{"repeat": 0, "rate": 10, "allocation": first}, {"repeat": 0, "rate": 10, "allocation": first},
             {"repeat": 0, "rate": 10, "allocation": second}, {"repeat": 1, "rate": 10, "allocation": second},
             {"repeat": 1, "rate": 10, "allocation": second}, {"repeat": 1, "rate": 10, "allocation": second}]
    assert study.pairing_same_allocation(cells) == {"r0-rate10": False, "r1-rate10": True}


# ---------------------------------------------------------------- end-to-end rehearsal with fake model services

@pytest.mark.skipif(not all(d.is_dir() for d in (V1_DIR, V3_DIR, SIZE_DIR)),
                    reason="Needs the audited V1, V3 and size-control returns")
def test_allocation_renewal_needs_a_different_job_and_never_extends(tmp_path, monkeypatch):
    real_plans = runtime.plans
    monkeypatch.setattr(runtime, "plans", lambda *a, **k: real_plans(*a, **{**k, "check_snapshots": False}))
    out = tmp_path / "study"

    def initialize(job, hours, **kwargs):
        monkeypatch.setenv("SLURM_JOB_ID", job)
        began = time.time()
        session = runtime.initialize(V1_DIR, V3_DIR, SIZE_DIR, out, hours, **kwargs)["session"]
        return session, began

    first, began = initialize("job-1", 2)
    assert abs(first["deadline_epoch"] - (began + 2 * 3600)) < 60
    again, _ = initialize("job-1", 5)  # the same allocation cannot be extended
    assert again["deadline_epoch"] == first["deadline_epoch"]
    with pytest.raises(StudyError, match="Cannot renew the same allocation"):
        initialize("job-1", 1, new_allocation=True)
    with pytest.raises(StudyError, match="different Slurm job"):
        initialize("job-2", 3)
    renewed, began = initialize("job-2", 3, new_allocation=True)
    assert renewed["allocation"]["slurm_job_id"] == "job-2" and abs(renewed["deadline_epoch"] - (began + 3 * 3600)) < 60
    assert [json.loads(path.read_text()) for path in (out / "runtime/session-history").glob("*.json")] == [first]
    for role in identity.ROLES:
        assert json.loads((runtime.control_root(out, role) / "runtime/session.json").read_text()) == renewed
    assert runtime.validate(out)[2] == renewed
    monkeypatch.setenv("SLURM_JOB_ID", "job-1")
    with pytest.raises(StudyError, match="differs"):
        runtime.check_time(out)


class AnswerEverything(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        # StartLux frontends and vLLM both answer GET /health at the service root.
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state = body.get("state") or body["messages"][-1]["content"]
        h = int.from_bytes(__import__("hashlib").sha256(state.encode()).digest()[:4], "big")
        # Short service times so the cascade's two engines overlap the way real ones do.
        time.sleep(0.03 if self.path == "/v1/systemone" else 0.12)
        if self.path == "/v1/systemone":
            label = LABELS[h % 3]
            payload = {"answers": {"decision": {"choice": label, "probabilities": probs(0.55 if h % 4 == 0 else 0.95, label)}}}
        else:
            payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": LABELS[(h + 1) % 3]})}}], "usage": {}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _fake_token_report(project, control, model, examples, profile=None):
    """The frozen V1 tokenizer check's report, written where V1 writes it."""
    control = Path(control)
    frozen = runtime.frozen
    report = {"status": "ok", "model": model, "model_revision": identity.protocol()["models"][control.name]["revision"],
              "max_model_len": json.loads((control / "runtime/lock.json").read_text())["max_model_len"],
              "n_examples": len(examples), "overlong": [], "renderer": "official", "truncated": False,
              "rows": [{"sample_id": e["sample_id"], "input_tokens": 100, "reserved_output_tokens": 32,
                        "token_ids_sha256": "0" * 64} for e in examples],
              "effective_config_hash": frozen._hash(frozen.runtime_config(control, model))}
    (control / "runtime" / f"tokens-interactive-{model}-{frozen._hash(examples)[:16]}.json").write_text(json.dumps(report))
    return report


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    """The node's command sequence on the real evidence, with a shortened schedule.

    Runtime binding, session, controls, ownership checks, token binding and every
    child process are the production code. Faked: the GPU launcher, nvidia-smi, the
    hardware doctor, the CUDA toolkit check, the tokenizer run (its V1-format report
    is written in place), the model servers, and the legacy-environment scope (the
    returned V1 copy records the node's absolute work path).
    """
    if not all(d.is_dir() for d in (V1_DIR, V3_DIR, SIZE_DIR)):
        pytest.skip("Needs the audited V1, V3 and size-control returns")
    server = ThreadingHTTPServer(("127.0.0.1", 0), AnswerEverything)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake = _fake_services()
    fake.state["endpoint"] = f"http://127.0.0.1:{server.server_address[1]}"
    short = json.loads(json.dumps(identity.protocol()))
    short.update(rates_per_second=[5], repeats=1, duration_seconds=2, drain_seconds=5, watchdog_grace_seconds=5)
    short["feasibility_preflight"]["concurrent_check"].update(duration_seconds=2)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    out = tmp_path_factory.mktemp("rehearsal") / "study"
    inputs = ["--v1-dir", str(V1_DIR), "--v3-dir", str(V3_DIR), "--size-control-dir", str(SIZE_DIR)]
    seen = SimpleNamespace(out=out, fake=fake, inputs=inputs)

    def command(*args):
        code = cli.main([*args, "--output-dir", str(out)])
        if code != 0:
            error = out / "diagnostics/last_error.json"
            pytest.fail(f"{args[0]} exited {code}: " + (json.loads(error.read_text())["traceback"] if error.exists() else ""))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(study, "protocol", lambda: short)
        mp.setattr(study, "source_identity", lambda require_clean=True: identity.source_identity(require_clean=False))
        mp.setenv("SLURM_JOB_ID", "rehearsal-1")
        real_plans = runtime.plans
        # The recorded snapshot paths exist only on the node.
        mp.setattr(runtime, "plans", lambda *a, **k: real_plans(*a, **{**k, "check_snapshots": False}))
        mp.setattr(runtime.frozen, "doctor", lambda control, require_gpu: {"gpu": [{"name": "NVIDIA L40S", "memory_mb": 46068.0}]})
        mp.setattr(runtime.v3_runtime, "_cuda_preflight", lambda control, binding, installed: {})
        mp.setattr(runtime, "legacy_environment", lambda binding, project: nullcontext())
        mp.setattr(runtime.frozen, "managed_services", lambda project, control, models: fake.launcher(control, models[0]))
        mp.setattr(runtime.frozen, "token_preflight_examples", _fake_token_report)
        mp.setattr(runtime, "gpu_processes", fake.lister)
        try:
            command("prepare", *inputs)
            command("init-runtime", "--session-hours-left", "1")
            seen.session = json.loads((out / "runtime/session.json").read_text())
            command("preflight")
            seen.lister_after_preflight = fake.lister()
            command("start", *inputs, "--session-hours-left", "1")
            seen.evaluation_bytes = (out / "evaluation.json").read_bytes()
            seen.launched_after_start = len(fake.state["launched"])
            seen.lister_after_start = fake.lister()
            command("start", *inputs, "--session-hours-left", "1")  # a finished study is verified, never rescored
            yield seen
        finally:
            server.shutdown()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            fake.cleanup()


def test_end_to_end_rehearsal_through_the_cli(rehearsal):
    out = rehearsal.out
    assert rehearsal.session["allocation"]["slurm_job_id"] == "rehearsal-1"
    tokens = json.loads((out / "preflight/tokens.json").read_text())
    assert {role: binding["n_examples"] for role, binding in tokens["reports"].items()} == {role: 3290 for role in identity.ROLES}
    for condition in identity.CONDITIONS:
        receipt = json.loads((out / f"preflight/{condition}.json").read_text())
        assert receipt["status"] == "PASS" and receipt["concurrent"]["statuses"] == {"ok": receipt["concurrent"]["report"]["offered"]}
    cascade = json.loads((out / "preflight/cascade.json").read_text())
    offered = cascade["concurrent"]["report"]["offered"]
    assert cascade["concurrent"]["engine_overlap_seconds"] > 0
    assert cascade["concurrent"]["route_reasons"] == {"preflight_forced_deferral": offered} == {"preflight_forced_deferral": cascade["concurrent"]["fallback_called"]}
    assert cascade["serial"]["failures"] == 0 and cascade["serial"]["stage_calls"] == {"startlux4b": 90, "qwen9b": 90}
    evidence = out / cascade["evidence_dir"]  # both preflight stages ran in owned child processes
    assert json.loads((evidence / "concurrent/plan.json").read_text())["mode"] == "forced_fallback"
    assert json.loads((evidence / "serial/plan.json").read_text())["mode"] == "both_models"
    assert (evidence / "concurrent/complete.json").is_file() and (evidence / "serial/complete.json").is_file()
    assert not rehearsal.lister_after_preflight

    evaluation = json.loads(rehearsal.evaluation_bytes)
    assert evaluation["status"] == "COMPLETE" and len(evaluation["cells"]) == 3
    by_condition = {c["condition"]: c for c in evaluation["cells"]}
    assert by_condition["cascade"]["details"]["cascade"]["routed"] > 0
    for cell in evaluation["cells"]:
        assert cell["primary"]["offered"] == cell["secondary"]["offered"] > 0
        assert cell["details"]["history"]["successful_responses"] == cell["primary"]["successful"]
    assert set(evaluation["historical_full_pool_accuracy"]) == set(identity.CONDITIONS)
    assert evaluation["pairing_same_allocation"] == {"r0-rate5": True}
    # One preflight deployment per condition plus one fresh deployment per cell; all stopped.
    assert rehearsal.launched_after_start == 2 * (2 + 2) and not rehearsal.lister_after_start
    finished = sorted(out.glob("services/*/finished.json"))
    assert len(finished) == 3
    for path in finished:
        record = json.loads(path.read_text())
        assert record["attempt"] == "attempt-0001" and record["stop_verification"]["gpu_compute_processes"] == []
        assert (path.parent / "warmup/complete.json").is_file()  # warmup ran in an owned child process
    assert (out / "evaluation.json").read_bytes() == rehearsal.evaluation_bytes
    assert (out / "exports/upgrade_study_bundle.zip").stat().st_size > 0
    assert not (out / "diagnostics/last_error.json").exists()


def test_changed_preflight_evidence_is_refused_on_reuse_start_and_completion(rehearsal):
    out = rehearsal.out
    config = study.settings(out)
    cell = config["order"][0]
    receipt_path = out / "preflight/cascade.json"
    receipt_bytes = receipt_path.read_bytes()
    evidence = out / json.loads(receipt_bytes)["evidence_dir"]
    token_file = out / json.loads((out / "preflight/tokens.json").read_text())["reports"]["qwen9b"]["file"]
    touched = [evidence / "concurrent/traces.jsonl", evidence / "serial/rows.json", token_file, receipt_path]
    tampering = {
        "deleted concurrent traces": lambda: (evidence / "concurrent/traces.jsonl").unlink(),
        "edited serial rows": lambda: (evidence / "serial/rows.json").write_text("{}"),
        "edited raw token report": lambda: token_file.write_text(token_file.read_text().replace('"input_tokens": 100', '"input_tokens": 99', 1)),
        "missing cascade receipt": lambda: receipt_path.unlink(),
    }
    for name, tamper in tampering.items():
        saved = {path: path.read_bytes() for path in touched}
        tamper()
        try:
            if receipt_path.exists():
                assert receipt_path.read_bytes() == receipt_bytes  # the top-level PASS itself is unchanged
                with pytest.raises(StudyError):
                    study.preflight(out)  # reuse of the cached PASS
            with pytest.raises(StudyError):
                study.run(out)  # formal start
            with pytest.raises(StudyError):
                study.completed(out, config, cell)  # completion
            with pytest.raises(StudyError):
                study.freeze(out)
            assert study.status(out)["preflight_verified"] is False and study.status(out)["completed_cells"] == 0, name
        finally:
            for path, data in saved.items():
                path.write_bytes(data)
    assert set(study.preflight(out).values()) == {"PASS"} and study.completed(out, config, cell) is not None
    assert study.status(out)["completed_cells"] == 3
