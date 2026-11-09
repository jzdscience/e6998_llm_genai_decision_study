"""Software acceptance for the ANLI upgrade-versus-fallback serving study.

No GPU or model is used. Real 0.50 single-model and 0.40 + 0.50 co-residency are
checked only by the node development preflight.
"""
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
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
        if self.path == "/v1/systemone":
            FakeModels.hits["startlux"] += 1
            if "FAIL500" in body["state"]:
                self.send_response(500); self.end_headers(); return
            top = 0.5 if "LOW" in body["state"] else 0.95
            payload = {"answers": {"decision": {"choice": "entailment", "probabilities": probs(top)}}}
        elif self.path == "/v1/chat/completions":
            FakeModels.hits["qwen"] += 1
            payload = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": "neutral"})}}], "usage": {}}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
            {"processes": [{"name": role, "pid": process.pid, "log": ""}], "models": {model: {"health": {"ok": True}, "config": {}}}}))
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
    plan = {"protocol": {"duration_seconds": 0.2, "drain_seconds": 0.2, "watchdog_grace_seconds": 0.2}}
    began = time.monotonic()
    with pytest.raises(StudyError, match="watchdog"):
        study._run_process(tmp_path, folder, plan, check_time=False)
    assert time.monotonic() - began < 15
    assert started[0].poll() is not None and not runtime.group_alive(started[0].pid)


def test_foreign_gpu_process_blocks_without_stopping_it(lifecycle):
    stranger = _dummy_group()
    try:
        foreign = lambda: [{"pid": stranger.pid, "used_memory_mib": 1.0}]
        with pytest.raises(StudyError, match="outside this deployment"):
            with runtime.deployment(lifecycle.output, "startlux9b", launcher=lifecycle.launcher, lister=foreign, preflight=False):
                pass
        assert lifecycle.state["processes"] == {} and stranger.poll() is None
    finally:
        os.killpg(stranger.pid, signal.SIGTERM)
        stranger.wait(timeout=10)


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


@pytest.fixture
def completed_cell(tmp_path):
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
    records = out / "services/r0"
    records.mkdir(parents=True)
    (records / "start.json").write_bytes(_bytes({"x": 1}))
    (records / "warmup.json").write_bytes(_bytes({"rows": [{"sample_id": w["sample_id"], "status": "ok", "stages": []} for w in warmup]}))
    folder = out / "cells" / cell["cell_id"] / "attempt-0001"
    folder.mkdir(parents=True)
    plan = {**cell, "settings_hash": fingerprint(config), "protocol": p, "schedule_sha256": config["file_sha256"][cell["schedule_file"]],
            "workload_file": "workload.json", "workload_sha256": config["file_sha256"]["workload.json"],
            "policy_file": "policy.json", "policy_sha256": config["file_sha256"]["policy.json"],
            "preflight_receipt_sha256": {c: file_hash(out / f"preflight/{c}.json") for c in identity.CONDITIONS},
            "service_record_file": "services/r0/start.json", "service_record_sha256": file_hash(records / "start.json"),
            "warmup_record_file": "services/r0/warmup.json", "warmup_record_sha256": file_hash(records / "warmup.json")}
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
    return SimpleNamespace(out=out, config=config, cell=cell, folder=folder)


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
    monkeypatch.setattr(study, "completed", lambda output, config, cell: None)
    with pytest.raises(StudyError, match="All 36 fixed cells"):
        study.evaluate(tmp_path)
    assert not (tmp_path / "evaluation-access.json").exists()


def test_run_requires_passing_preflight_for_every_condition(completed_cell, monkeypatch):
    c = completed_cell
    (c.out / "preflight/startlux9b.json").unlink()
    monkeypatch.setattr(study, "settings", lambda output, check_source=False: c.config)
    with pytest.raises(StudyError, match="missing PASS receipt for startlux9b"):
        study.run(c.out)


# ---------------------------------------------------------------- end-to-end rehearsal with fake model services

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


@pytest.mark.skipif(not all(d.is_dir() for d in (V1_DIR, V3_DIR, SIZE_DIR)),
                    reason="Needs the audited V1, V3 and size-control returns")
def test_end_to_end_rehearsal_through_the_cli(tmp_path, monkeypatch):
    """The node's command sequence on the real evidence, with a shortened schedule.

    Runtime binding, session, controls, ownership checks and the cell subprocesses
    are the production code. Faked: the GPU launcher, nvidia-smi, the hardware doctor,
    the CUDA toolkit check, tokenizers, the model servers, and the legacy-environment
    scope (the returned V1 copy records the node's absolute work path).
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), AnswerEverything)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake = _fake_services()
    fake.state["endpoint"] = f"http://127.0.0.1:{server.server_address[1]}"
    short = json.loads(json.dumps(identity.protocol()))
    short.update(rates_per_second=[5], repeats=1, duration_seconds=2, drain_seconds=5, watchdog_grace_seconds=5)
    short["feasibility_preflight"]["concurrent_check"].update(duration_seconds=2)
    monkeypatch.setattr(study, "protocol", lambda: short)
    monkeypatch.setattr(study, "source_identity", lambda require_clean=True: identity.source_identity(require_clean=False))
    monkeypatch.setenv("SLURM_JOB_ID", "rehearsal-1")
    real_plans = runtime.plans
    # The recorded snapshot paths exist only on the node.
    monkeypatch.setattr(runtime, "plans", lambda *a, **k: real_plans(*a, **{**k, "check_snapshots": False}))
    monkeypatch.setattr(runtime.frozen, "doctor",
                        lambda control, require_gpu: {"gpu": [{"name": "NVIDIA L40S", "memory_mb": 46068.0}]})
    monkeypatch.setattr(runtime.v3_runtime, "_cuda_preflight", lambda control, binding, installed: {})
    monkeypatch.setattr(runtime, "legacy_environment", lambda binding, project: nullcontext())
    monkeypatch.setattr(runtime.frozen, "managed_services", lambda project, control, models: fake.launcher(control, models[0]))
    monkeypatch.setattr(runtime, "gpu_processes", fake.lister)
    monkeypatch.setattr(runtime, "token_check", lambda output, examples: {r: {"status": "ok", "summary": {}} for r in identity.ROLES})
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    out = tmp_path / "study"
    inputs = ["--v1-dir", str(V1_DIR), "--v3-dir", str(V3_DIR), "--size-control-dir", str(SIZE_DIR)]

    def command(*args):
        code = cli.main([*args, "--output-dir", str(out)])
        if code != 0:
            error = out / "diagnostics/last_error.json"
            pytest.fail(f"{args[0]} exited {code}: " + (json.loads(error.read_text())["traceback"] if error.exists() else ""))

    try:
        command("prepare", *inputs)
        command("init-runtime", "--session-hours-left", "1")
        assert json.loads((out / "runtime/session.json").read_text())["allocation"]["slurm_job_id"] == "rehearsal-1"
        command("preflight")
        for condition in identity.CONDITIONS:
            receipt = json.loads((out / f"preflight/{condition}.json").read_text())
            assert receipt["status"] == "PASS" and receipt["concurrent"]["statuses"] == {"ok": receipt["concurrent"]["report"]["offered"]}
        cascade = json.loads((out / "preflight/cascade.json").read_text())
        offered = cascade["concurrent"]["report"]["offered"]
        assert cascade["concurrent"]["engine_overlap_seconds"] > 0
        assert cascade["concurrent"]["route_reasons"] == {"preflight_forced_deferral": offered} == {"preflight_forced_deferral": cascade["concurrent"]["fallback_called"]}
        assert cascade["serial"]["failures"] == 0 and cascade["serial"]["stage_calls"] == {"startlux4b": 90, "qwen9b": 90}
        assert not fake.lister()

        command("start", *inputs, "--session-hours-left", "1")
        evaluation = json.loads((out / "evaluation.json").read_text())
        assert evaluation["status"] == "COMPLETE" and len(evaluation["cells"]) == 3
        by_condition = {c["condition"]: c for c in evaluation["cells"]}
        assert by_condition["cascade"]["details"]["cascade"]["routed"] > 0
        for cell in evaluation["cells"]:
            assert cell["primary"]["offered"] == cell["secondary"]["offered"] > 0
            assert cell["details"]["history"]["successful_responses"] == cell["primary"]["successful"]
        assert set(evaluation["historical_full_pool_accuracy"]) == set(identity.CONDITIONS)
        # One preflight deployment per condition plus one fresh deployment per cell; all stopped.
        assert len(fake.state["launched"]) == 2 * (2 + 2) and not fake.lister()
        finished = sorted(out.glob("services/*/finished.json"))
        assert len(finished) == 3
        for path in finished:
            assert json.loads(path.read_text())["stop_verification"]["gpu_compute_processes"] == []

        scored = (out / "evaluation.json").read_bytes()
        command("start", *inputs, "--session-hours-left", "1")  # a finished study is verified, never rescored
        assert (out / "evaluation.json").read_bytes() == scored
        assert (out / "exports/upgrade_study_bundle.zip").stat().st_size > 0
        assert not (out / "diagnostics/last_error.json").exists()
    finally:
        server.shutdown()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        fake.cleanup()
