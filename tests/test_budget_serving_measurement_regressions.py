"""Independent scheduler regressions: durable evidence, causal occupancy, final deadlines.

The model-call stub supplies one already parsed response. These tests exercise the
actual executor and trace boundary; they do not assert new model quality or speed.
"""
import copy
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from decision_study.adapters import decision_request
from decision_study.budget_serving import measurement, study, verify, worker
from decision_study.budget_serving.ledger import InvariantViolation, Ledger
from decision_study.common import StudyError, fingerprint


def _answer():
    return {"status": "ok", "prediction": "entailment", "final_source": "startlux4b",
            "stages": [], "route": None, "error": None}


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_worker_trace_failure_aborts_measurement(operation):
    """A complete in-memory row is not success when its evidence was not written."""
    class FailingTrace:
        failed = False

        def _fail_once(self, name):
            if name == operation and not self.failed:
                self.failed = True
                raise OSError("evidence disk unavailable")

        def write(self, data):
            self._fail_once("write")

        def flush(self):
            self._fail_once("flush")

    with pytest.raises(OSError, match="evidence disk unavailable"):
        measurement.run_open_loop(
            [{"offer_id": 0, "sample_id": "example", "arrival_seconds": 0.001}],
            lambda *args: _answer(), duration=0.005, workers=1, max_inflight=1,
            drain_seconds=1, request_timeout=1, ledger=Ledger(enforced=False), trace=FailingTrace())


def test_occupancy_transitions_remain_causal_when_a_sampled_worker_is_preempted():
    """Pause after a clock sample so another worker can reach its model call first.

    A monotonic occupancy trace must enter one worker at a time, even when a
    sampled timestamp becomes stale before the shared worker count is changed.
    The bounded wait also works if timestamp sampling is inside the counter lock:
    the first worker then proceeds first, and both offers still finish.
    """
    main_thread = threading.get_ident()
    claim_lock = threading.Lock()
    seen_threads = set()
    resume_first = threading.Event()
    first_written = threading.Event()
    claimed = [False]

    def clock():
        sampled = time.monotonic()
        ident = threading.get_ident()
        pause = False
        if ident != main_thread:
            with claim_lock:
                if ident not in seen_threads:
                    seen_threads.add(ident)
                    if not claimed[0]:
                        claimed[0] = True
                        pause = True
        if pause:
            resume_first.wait(timeout=2)
        return sampled

    class Trace(io.BytesIO):
        def write(self, data):
            count = super().write(data)
            if json.loads(data)["offer_id"] == 0:
                first_written.set()
            return count

    def call(offer, deadline, begin):
        if offer["offer_id"] == 1:
            resume_first.set()
            assert first_written.wait(timeout=3), "First offer never finished"
        return _answer()

    rows, events, facts = measurement.run_open_loop(
        [{"offer_id": i, "sample_id": f"example-{i}", "arrival_seconds": 0.001 * (i + 1)} for i in range(2)],
        call, duration=0.01, workers=2, max_inflight=2, drain_seconds=10,
        request_timeout=5, ledger=Ledger(enforced=False), trace=Trace(), clock=clock)
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert len(events) == 4
    active = 0
    for event in events:
        assert abs(event["active_workers"] - active) == 1, events
        assert 0 <= event["active_workers"] <= 2, events
        active = event["active_workers"]
    assert active == 0
    assert facts["worker_occupancy_seconds"] >= 0


def test_final_offer_deadline_keeps_raw_stage_but_refuses_late_success():
    """A timely model reply does not make a later full-offer completion timely.

    This is the 30-second request-outcome boundary (scaled to one second here).
    The separate 500/1000-ms goodput already uses the full completion timestamp.
    """
    fixture = json.loads((Path(__file__).parent / "fixtures/budget_startlux_responses.json").read_text())
    example = fixture["example"]
    old = fixture["responses"]["startlux4b"]
    clock_lock = threading.Lock()
    now = [0.0]

    def clock():
        with clock_lock:
            return now[0]

    def sleep(seconds):
        with clock_lock:
            now[0] += seconds

    def call(offer, deadline, begin):
        stage = {
            "role": "startlux4b", "endpoint": "http://127.0.0.1:18380", "interface": "startlux",
            "request_id": "captured-deadline", "request_sha256": fingerprint(decision_request(example, "startlux")),
            "started_ms": 0.0, "ended_ms": 200.0, "latency_ms": 200.0,
            "deadline_ms": 1000 * (deadline - begin), "http_started": True, "attempts": 1,
            "http_status": 200, "server_request_id": "captured-deadline-server", "raw_response": old["raw_response"],
            "raw_text": None, "response_bytes": len(json.dumps(old["raw_response"]).encode()),
            "token_counts": old["token_counts"], "prediction": old["prediction"], "probabilities": old["probabilities"],
            "late": False, "error": None, "status": "ok",
        }
        # Delay after the successful stage was recorded but before the complete
        # offer reaches its caller, such as controller preemption/bookkeeping.
        with clock_lock:
            now[0] = deadline + 0.1
        return {**_answer(), "prediction": old["prediction"], "stages": [stage]}

    schedule = [{"offer_id": 0, "sample_id": example["sample_id"], "arrival_seconds": 0.0}]
    ledger_trace = io.BytesIO()
    rows, _, _ = measurement.run_open_loop(
        schedule, call, duration=0.01, workers=1, max_inflight=1,
        drain_seconds=2, request_timeout=1, ledger=Ledger(enforced=False, journal=ledger_trace, clock=clock),
        trace=io.BytesIO(), clock=clock, sleep=sleep)
    row = rows[0]
    assert row["completion_seconds"] > row["deadline_seconds"]
    assert row["result"]["stages"][0]["status"] == "ok"
    assert row["result"]["stages"][0]["raw_response"] == old["raw_response"]
    assert row["status"] == "deadline", "The full offer completed beyond its shared deadline"
    assert row["result"]["prediction"] is None and row["result"]["final_source"] is None
    kwargs = dict(condition="direct4b", mode="direct", label=None, examples={example["sample_id"]: example},
                  policy=None, flags=None, ledger_events=[json.loads(line) for line in ledger_trace.getvalue().splitlines()],
                  journal_events=[], duration=0.01, drain_seconds=2, request_timeout=1)
    verify.verify_open_loop(rows, schedule, **kwargs)
    forged = copy.deepcopy(rows)
    forged[0]["status"] = forged[0]["result"]["status"] = "ok"
    forged[0]["result"].update(prediction=old["prediction"], final_source="startlux4b", error=None)
    with pytest.raises(InvariantViolation):
        verify.verify_open_loop(forged, schedule, **kwargs)


def test_failed_cleanup_blocks_retry_before_another_worker_starts(tmp_path, monkeypatch):
    """Retry permission does not supply missing proof that the old worker stopped."""
    output = tmp_path.resolve()
    folder = output / "stages/tokens/attempt-0001"
    folder.mkdir(parents=True)
    allocation = {"slurm_job_id": "123"}
    (folder / "supervision.json").write_text(json.dumps({"binding_hash": "binding", "allocation": allocation}))
    (folder / "supervision-failure.json").write_text(json.dumps({"cleanup_error": "unreaped descendants"}))
    ctx = SimpleNamespace(output=output, p={"cleanup_margin_seconds": 180})
    monkeypatch.setattr(study.runtime, "session", lambda *args, **kwargs: {
        "deadline_epoch": time.time() + 3600, "allocation": allocation, "binding_hash": "binding"})
    launches = []

    def supervise(*args, **kwargs):
        launches.append(args)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(study.runtime, "supervise", supervise)
    # Keep the actual recover_previous implementation: same allocation, no stop
    # receipt, so no PID/PGID may be signalled and another worker must not begin.
    with pytest.raises(StudyError, match="cleanup proof"):
        study.run_stage(ctx, "tokens", "tokens", {"watchdog_seconds": 10},
                        accepted=lambda: None, accept=lambda attempt: {"attempt": attempt.name},
                        retry_failed=True, label="tokens")
    assert launches == []
    assert not (folder.parent / "attempt-0002").exists()


def test_after_load_preserves_both_roles_when_engine_process_names_match(tmp_path, monkeypatch):
    """The shared name startlux-engine must not hide the 4B engine's failure."""
    folder = tmp_path.resolve()
    four, nine = folder / "four.log", folder / "nine.log"
    four.write_text("CUDA out of memory\n")
    nine.write_text("Engine completed normal processing\n")
    record = {"roles": [
        {"role": "startlux4b", "processes": [{"name": "startlux-engine", "log": str(four)}]},
        {"role": "startlux9b", "processes": [{"name": "startlux-engine", "log": str(nine)}]},
    ]}
    monkeypatch.setattr(worker.old, "health", lambda endpoint: {"ok": True, "status": 200})
    monkeypatch.setattr(worker, "_gpu_snapshot", lambda: {"status": "ok", "processes": []})
    worker._after_load(folder, {"startlux4b": "http://127.0.0.1:18380", "startlux9b": "http://127.0.0.1:18390"}, record)
    facts = json.loads((folder / "health.json").read_text())["engine_log"]
    assert set(facts) == {"startlux4b", "startlux9b"}, "Engine evidence must retain its role identity"
    assert facts["startlux4b"]["startlux-engine"]["failure_mentions"] == {"CUDA out of memory": 1}
    assert facts["startlux9b"]["startlux-engine"]["failure_mentions"] == {}
