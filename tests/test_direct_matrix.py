"""Software acceptance for the direct-accuracy matrix completion (protocol direct-matrix-v1).

No GPU, model or scheduler is used. The rehearsal fixture runs the real node command
sequence from an ephemeral clean git repository built from this working tree and
fakes only hardware/model boundaries (see tests/direct_matrix_rehearsal.py). Real
model behaviour, token lengths and the 9B/Kev servers are checked only by the node
preflight; fake labels here are never results.
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from decision_study.common import StudyError, file_hash, fingerprint, read_json, write_json
from decision_study.direct_matrix import audit, cli, collect, data, identity, runtime, stats, study
from decision_study.size_control import runtime as size_runtime
from decision_study.storage import load_predictions
from decision_study.upgrade_study import runtime as up_runtime

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get("DIRECT_MATRIX_EVIDENCE_ROOT", ROOT.parent / "project/solo-experiments-2026-10-30/decision-study"))
HAVE_EVIDENCE = all((EVIDENCE / "results" / name).is_dir() for name in (
    "returned-experiment-2026-11-02-101054", "returned-v2-2026-11-03-002852", "returned-v3-2026-11-03-152529",
    "returned-size-control-complete-2026-11-06"))
needs_evidence = pytest.mark.skipif(not HAVE_EVIDENCE, reason="Needs the accepted V1/V2/V3/size-control returns")


# ---------------------------------------------------------------- protocol and requests

def test_protocol_fixes_the_eight_cells_and_their_order():
    p = identity.protocol()
    assert [c["cell_id"] for c in p["cells"]] == list(identity.CELL_ORDER)
    assert sum(c["count"] for c in p["cells"]) == 20400
    assert [r for r, _ in study._groups(list(identity.CELL_ORDER))] == ["startlux9b", "qwen9b", "kev4b", "startlux9b", "qwen9b"]
    assert {r: p["roles"][r]["token_reserve"] for r in identity.ROLES} == {"startlux9b": 1, "qwen9b": 64, "kev4b": 0}
    assert p["engine"]["applies_to"] == ["startlux9b", "qwen9b"] and p["engine"]["max_num_seqs"] == 64
    assert {p["roles"][r]["gpu_memory_utilization"] for r in ("startlux9b", "qwen9b")} == {0.8}
    assert all(len(p["references"][c]["compact_predictions_hash"]) == 64 for c in identity.REFERENCE_CELLS)


class Recorder(dict):
    """An input row that records which fields a request builder reads."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


def _example(labels=("entailment", "neutral", "contradiction"), criteria_order=None):
    keys = criteria_order or sorted(labels)
    return {"sample_id": "anli:x", "task": "anli", "split": "holdout", "group_id": "anli:premise:p",
            "state": "Premise:\nA.\n\nHypothesis:\nB.", "instructions": "Classify.", "labels": list(labels),
            "criteria": {k: f"{k} means" for k in keys}}


@pytest.mark.parametrize("model", ["kev", "startlux", "qwen"])
def test_requests_use_only_task_fields_in_saved_label_order(model):
    example = Recorder(_example())
    body = collect.request_body(example, model)
    assert example.read <= {"state", "instructions", "criteria", "labels"}
    text = json.dumps(body)
    assert "gold" not in text and "answer\"" not in text
    if model == "qwen":
        assert body["structured_outputs"]["json"]["properties"]["label"]["enum"] == example["labels"]
        assert body["temperature"] == 0 and body["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        assert list(body["questions"]["decision"]["criteria"]) == example["labels"]


# ---------------------------------------------------------------- scoring

def test_direct_counts_errors_wrong_and_refuses_missing_or_extra_rows():
    examples = [{"sample_id": s} for s in "abcd"]
    gold = {"a": "x", "b": "y", "c": "x", "d": "y"}
    rows = {"a": {"status": "ok", "prediction": "x"}, "b": {"status": "ok", "prediction": "x"},
            "c": {"status": "error", "prediction": None}, "d": {"status": "ok", "prediction": "y"}}
    metric, outcome = stats.direct(examples, rows, gold, ["x", "y"])
    assert (metric["correct"], metric["n"], metric["failures"], outcome) == (2, 4, 1, [True, False, False, True])
    with pytest.raises(StudyError, match="Missing prediction"):
        stats.direct(examples, {k: v for k, v in rows.items() if k != "d"}, gold, ["x", "y"])
    with pytest.raises(StudyError, match="outside the fixed denominator"):
        stats.direct(examples, {**rows, "e": rows["a"]}, gold, ["x", "y"])


def test_bootstrap_is_the_v3_pooled_cluster_algorithm():
    from decision_study.v3 import stats as v3_stats
    rng = np.random.default_rng(1)
    examples = [{"sample_id": f"anli:{i}", "group_id": f"anli:premise:{i % 37}", "round": "R1", "task": "anli",
                 "labels": ["entailment", "neutral", "contradiction"], "state": "Premise:\nA.\n\nHypothesis:\nB."}
                for i in range(300)]
    a, b = rng.random(300) < .7, rng.random(300) < .6
    mine = stats.paired_difference(examples, a, b, cluster="group_id", repetitions=500, seed=6998)
    theirs = v3_stats.cluster_bootstrap_difference(examples, list(a), list(b), repetitions=500, seed=6998)
    assert (mine["estimate"], mine["lower"], mine["upper"]) == (theirs["estimate"], theirs["lower"], theirs["upper"])
    # Pooled question-weighted ratio, not a mean of cluster accuracies: cluster g has 3 inputs
    # (differences 1, 1, 0) and cluster h has 1 (difference -1). Draws can only be 2/3, 1/4 or -1.
    tiny = [{"sample_id": s, "group_id": g} for s, g in zip("abcd", "gggh")]
    result = stats.paired_difference(tiny, [True, True, False, False], [False, False, False, True],
                                     cluster="group_id", repetitions=2000, seed=6998)
    assert result["estimate"] == 0.25 and result["cluster_count"] == 2
    assert result["lower"] == -1.0 and result["upper"] == pytest.approx(2 / 3)
    assert stats.paired_difference(tiny, [True] * 4, [False] * 4, cluster="sample_id", repetitions=10, seed=1)["cluster_count"] == 4


# ---------------------------------------------------------------- row checks

def _row(example, model="kev", label="neutral", **extra):
    probabilities = {k: (0.9 if k == label else 0.05) for k in example["labels"]}
    row = {"sample_id": example["sample_id"], "task": example["task"], "split": example["split"], "model": model,
           "cell": "kev4b-anli", "purpose": "formal", "deployment_id": "kev4b-1", "status": "ok", "prediction": label,
           "probabilities": probabilities, "error": None, "latency_ms": 3.0,
           "raw_response": {"answers": {"decision": {"choice": label, "probabilities": probabilities}}},
           "attempts": [{"attempt": 1, "status": "ok", "request_id": "r-" + example["sample_id"], "latency_ms": 3.0}],
           "example_sha256": fingerprint(example), "request_sha256": fingerprint(collect.request_body(example, model))}
    row.update(extra)
    return row


def _check(examples, rows):
    return collect.check_rows(examples, rows, cell_id="kev4b-anli", model="kev", purpose="formal", deployments={"kev4b-1"})


def test_check_rows_refuses_missing_duplicate_unbound_and_unparsed_rows():
    examples = [dict(_example(), sample_id=f"anli:{i}") for i in range(3)]
    rows = [_row(e) for e in examples]
    assert set(_check(examples, rows)) == {e["sample_id"] for e in examples}
    cases = {
        "Missing": rows[:2],
        "Duplicate": rows + [dict(rows[0])],
        "not bound": [rows[0], {**rows[1], "example_sha256": "0" * 64}, rows[2]],
        "does not parse": [rows[0], {**rows[1], "raw_response": {"answers": {}}}, rows[2]],
        "differs from its raw response": [rows[0], {**rows[1], "prediction": "entailment"}, rows[2]],
        "carries a prediction": [rows[0], {**rows[1], "status": "error", "error": {"type": "http"},
                                           "attempts": [{"attempt": 1, "status": "error", "request_id": "x", "error": {}, "latency_ms": 1}]},
                                 rows[2]],
        "Malformed request attempt": [rows[0], {**rows[1], "attempts": [{"attempt": 2, "status": "ok", "request_id": "y",
                                                                           "latency_ms": 1}]}, rows[2]],
        "unrecorded deployment": [rows[0], {**rows[1], "deployment_id": "kev4b-2"}, rows[2]],
        "identity mismatch": [rows[0], {**rows[1], "cell": "kev4b-squad2"}, rows[2]],
    }
    for message, case in cases.items():
        with pytest.raises(StudyError, match=message):
            _check(examples, case)


def test_engine_flag_check_is_exact_and_never_crashes_on_a_trailing_flag():
    engine = lambda command: {"role": "qwen9b", "processes": [{"name": "qwen-engine", "command": command}]}
    assert runtime.engine_command_limited(engine(["vllm", "--max-num-seqs", "64"]))
    assert not runtime.engine_command_limited(engine(["vllm", "--max-num-seqs"]))
    assert not runtime.engine_command_limited(engine(["vllm", "--max-num-seqs", "256"]))
    assert not runtime.engine_command_limited(engine(["vllm"]))
    assert not runtime.engine_command_limited({"role": "kev4b", "processes": [{"name": "kev", "command": ["--max-num-seqs", "64"]}]})
    assert runtime.engine_command_limited({"role": "kev4b", "processes": [{"name": "kev", "command": ["kev.serve"]}]})


# ---------------------------------------------------------------- scheduler evidence and stop reconciliation

def _runner(stdout="", returncode=0, error=None):
    def run(command, **kwargs):
        if error:
            raise error
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return run


def test_slurm_job_state_counts_only_a_unique_terminal_accounting_record():
    assert runtime.slurm_job_state("77", runner=_runner("77|COMPLETED|2026-11-10T01:00:00\n"))["terminal"]
    assert runtime.slurm_job_state("77", runner=_runner("77|CANCELLED by 1234|2026-11-10\n"))["terminal"]
    for runner in (_runner("77|RUNNING|Unknown\n"), _runner(""), _runner("77|COMPLETED|x\n", returncode=1),
                   _runner("77|COMPLETING|Unknown\n"), _runner("770|COMPLETED|x\n"), _runner(error=OSError("no sacct"))):
        assert runtime.slurm_job_state("77", runner=runner)["terminal"] is False
    assert runtime.slurm_job_state("abc", runner=_runner("abc|COMPLETED|x\n"))["terminal"] is False


def _deployment(tmp_path, *, host, job="99999999", pid=987654321, started=None):
    folder = tmp_path / "out/services/kev4b-1"
    folder.mkdir(parents=True)
    record = {"role": "kev4b", "allocation": {"hostname": host, "slurm_job_id": job},
              "started_at": started or "2026-11-10T00:00:00+00:00",
              "processes": [{"pid": pid, "name": "kev", "command": ["fake"]}],
              "stop_verification": {"error": "residual process"}}
    write_json(folder / "finished.json", {"deployment": record})
    return tmp_path / "out", folder


@pytest.mark.parametrize("state, passed", [("RUNNING", False), (None, False), ("COMPLETED", True)])
def test_a_deployment_on_another_host_needs_terminal_scheduler_evidence(tmp_path, monkeypatch, state, passed):
    out, folder = _deployment(tmp_path, host="node-that-is-not-" + socket.gethostname())
    monkeypatch.setenv("SLURM_JOB_ID", "4243")
    calls = []

    def scheduler(job):
        calls.append(job)
        record = {"query": ["sacct"], "job_id": job, "state": state, "terminal": state == "COMPLETED"}
        return record if state else {**record, "error": "no unique accounting record for the job"}

    if passed:
        assert study.reconcile_stops(out, lister=lambda: [], scheduler=scheduler) == ["kev4b-1"]
        saved = read_json(folder / "stop-reverified.json")
        assert saved["scheduler"]["state"] == "COMPLETED" and saved["signals_sent"] is False
        assert study.deployment_stop(out, "kev4b-1")[0] == "reverified"
    else:
        with pytest.raises(StudyError, match="not proven stopped"):
            study.reconcile_stops(out, lister=lambda: [], scheduler=scheduler)
        (failed,) = folder.glob("stop-reverify-failed-*.json")
        assert read_json(failed)["passed"] is False, "a non-terminal or unknown job is never counted as ended"
        assert not (folder / "stop-reverified.json").exists()
        assert study.deployment_stop(out, "kev4b-1") == (None, None)
    assert calls == ["99999999"]


def test_a_forged_scheduler_reverification_is_not_accepted(tmp_path):
    out, folder = _deployment(tmp_path, host="node-that-is-not-" + socket.gethostname())
    old = read_json(folder / "finished.json")["deployment"]["allocation"]
    forged = {"passed": True, "signals_sent": False, "deployment_id": "kev4b-1", "gpu_compute_processes": [],
              "old_allocation": old, "current_allocation": {"hostname": socket.gethostname(), "slurm_job_id": "4243"}}
    write_json(folder / "stop-reverified.json", {**forged, "basis": "different_host_after_allocation_ended"})
    assert study.deployment_stop(out, "kev4b-1") == (None, None)
    running = {"query": ["sacct"], "job_id": "99999999", "state": "RUNNING", "terminal": False}
    write_json(folder / "stop-reverified.json", {**forged, "basis": "scheduler_accounting_of_earlier_job", "scheduler": running})
    assert study.deployment_stop(out, "kev4b-1") == (None, None), "a still-running earlier job never proves a stop"
    write_json(folder / "stop-reverified.json", {**forged, "basis": "scheduler_accounting_of_earlier_job",
                                                 "scheduler": {**running, "state": "COMPLETED", "terminal": True}})
    assert study.deployment_stop(out, "kev4b-1")[0] == "reverified"


def test_same_host_reconciliation_is_passive_and_fails_on_a_live_member(tmp_path):
    started = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        out, folder = _deployment(tmp_path, host=socket.gethostname(), pid=child.pid, started=started)
        with pytest.raises(StudyError, match="not proven stopped"):
            study.reconcile_stops(out, lister=lambda: [], scheduler=lambda job: pytest.fail("no scheduler on the same host"))
        assert child.poll() is None, "reconciliation must never signal a process"
    finally:
        child.kill()
        child.wait()
    assert study.reconcile_stops(out, lister=lambda: []) == ["kev4b-1"]
    assert read_json(folder / "stop-reverified.json")["basis"] == "recorded_host_process_table"
    other, _ = _deployment(tmp_path / "busy", host=socket.gethostname())
    with pytest.raises(StudyError, match="not proven stopped"):
        study.reconcile_stops(other, lister=lambda: [{"pid": 1, "used_memory_mib": 1.0}])


# ---------------------------------------------------------------- child watchdog and attempt acceptance

def _child(monkeypatch, script):
    monkeypatch.setattr(study, "_worker_command", lambda output, folder: [sys.executable, "-c", script, str(folder)])
    monkeypatch.setattr(runtime, "check_time", lambda output, required_seconds=0: {})


RECEIPT = ("import json, sys, time, hashlib; from pathlib import Path; f = Path(sys.argv[1]); "
           "plan = json.loads((f / 'plan.json').read_text()); "
           "h = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(); "
           "(f / 'complete.json').write_text(json.dumps({'plan_hash': h, 'complete': True})); time.sleep({delay})")


def test_on_time_exit_is_accepted_by_the_parent(tmp_path, monkeypatch):
    _child(monkeypatch, RECEIPT.replace("{delay}", "0"))
    plan = study._run_process(tmp_path, tmp_path / "attempt-a", {"kind": "test"}, 10)
    assert study.attempt_accepted(tmp_path / "attempt-a") == plan


def test_exit_after_the_watchdog_is_never_accepted_even_if_the_receipt_is_renamed_back(tmp_path, monkeypatch):
    _child(monkeypatch, RECEIPT.replace("{delay}", "3"))
    folder = tmp_path / "attempt-late"
    with pytest.raises(StudyError, match="watchdog"):
        study._run_process(tmp_path, folder, {"kind": "test"}, 1.0)
    assert (folder / "unaccepted-complete.json").exists() and not (folder / "accepted.json").exists()
    (folder / "unaccepted-complete.json").rename(folder / "complete.json")
    assert study.attempt_accepted(folder) is None


def test_hung_child_is_stopped_by_the_watchdog(tmp_path, monkeypatch):
    _child(monkeypatch, "import time; time.sleep(60)")
    began = time.monotonic()
    with pytest.raises(StudyError, match="exceeded the absolute parent watchdog"):
        study._run_process(tmp_path, tmp_path / "attempt-hung", {"kind": "test"}, 0.5)
    assert time.monotonic() - began < 15


def _verified_stop(out, deployment_id, pid=987654321):
    write_json(out / f"services/{deployment_id}/finished.json", {"deployment": {
        "processes": [{"pid": pid}], "stop_verification": {"owned_groups": [pid], "gpu_compute_processes": []}}})


def test_resume_folds_fsynced_journal_rows_into_the_recovery_record(tmp_path, monkeypatch):
    """A child killed between shard writes leaves rows only in the journal; recovery must bind them."""
    from decision_study.storage import PredictionStore
    out, cell, deployment = tmp_path / "out", "kev4b-anli", "kev4b-old"
    directory = collect.cell_dir(out, cell)
    config = {"model": "kev", "fixture": "journal ordering"}
    store = PredictionStore(directory, "kev", config, spool_dir=directory / "spool")
    for number in range(53):  # default shard size 25: 50 rows in two shards, 3 only in the fsynced journal
        store.append({"sample_id": f"anli:{number:03d}", "model": "kev", "status": "ok", "config_hash": fingerprint(config),
                      "deployment_id": deployment, "recorded_at": datetime.now(timezone.utc).isoformat()})
    with store.journal.open("ab") as stream:  # a kill in the middle of the next journal write
        stream.write(b'{"sample_id": "anli:053", "mo')
    del store
    assert len(load_predictions(directory, "kev")) == 50 and study.pending_journal_bytes(out, cell) > 0
    folder = directory / f"attempt-{deployment}"
    write_json(folder / "plan.json", {"kind": "collect", "cell_id": cell, "deployment_id": deployment,
                                      "config_hash": fingerprint(config), "watchdog_deadline_epoch": time.time() + 600})
    write_json(folder / "interrupted.json", {"type": "SIGKILL"})
    _verified_stop(out, deployment)
    monkeypatch.setattr(study, "verify_stored_config", lambda output, ctx, cell_id: config)
    monkeypatch.setattr(study, "verify_smoke", lambda *args: None)
    ctx = {"settings_hash": "fixture"}
    assert study.unreconciled_attempts(out, ctx, cell) == [folder.name]
    assert study.pending_journal_bytes(out, cell) > 0, "a plain run must not recover anything"
    assert study.reconcile_attempts(out, ctx, cell) == [folder.name]
    recovery = data.checked(folder / "recovery.json", "recovery_hash")
    assert recovery["rows"] == 53 and recovery["journal_recovery"]["sharded_rows_before"] == 50
    assert (directory / "predictions/kev/journal_recovery.json").is_file(), "frozen tail truncation is recorded"
    assert (directory / recovery["journal_recovery"]["journal_copy"]).read_bytes().endswith(b'"mo')
    PredictionStore(directory, "kev", config, spool_dir=directory / "spool")  # what the resumed child does first
    assert study.attempt_status(out, ctx, cell, folder, load_predictions(directory, "kev")) == "recovered"
    assert study.pending_journal_bytes(out, cell) == 0


def test_recovery_refuses_rows_recorded_after_the_watchdog(tmp_path, monkeypatch):
    folder = tmp_path / "cells/kev4b-anli/attempt-kev4b-1"
    folder.mkdir(parents=True)
    write_json(folder / "plan.json", {"kind": "collect", "cell_id": "kev4b-anli", "deployment_id": "kev4b-1",
                                      "watchdog_deadline_epoch": 1000.0, "config_hash": "c"})
    write_json(folder / "interrupted.json", {"error": "watchdog"})
    _verified_stop(tmp_path, "kev4b-1")  # every other recovery condition holds, so only the deadline can refuse
    monkeypatch.setattr(study, "verify_smoke", lambda *args: None)
    late = {"sample_id": "anli:1", "deployment_id": "kev4b-1", "recorded_at": "2026-11-10T00:00:00+00:00"}
    with pytest.raises(StudyError, match="after its watchdog deadline"):
        study.recovery_payload(tmp_path, {"settings_hash": "s"}, "kev4b-anli", folder, [late])
    on_time = {**late, "recorded_at": datetime.fromtimestamp(999, timezone.utc).isoformat()}
    assert study.recovery_payload(tmp_path, {"settings_hash": "s"}, "kev4b-anli", folder, [on_time])["rows"] == 1
    assert study.recovery_payload(tmp_path, {"settings_hash": "s"}, "kev4b-anli", folder, [])["state"] == "interrupted"


# ---------------------------------------------------------------- deployment lifecycle with real process groups

def _services():
    state = {"launched": []}

    @contextmanager
    def launcher(control, model):
        process = subprocess.Popen(["sleep", "60"], start_new_session=True)
        state["launched"].append(process)
        write_json(control / "runtime/last_services.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                                                            "processes": [{"name": model, "pid": process.pid, "command": ["x"]}],
                                                            "models": {model: {"health": {}, "config": {}}}})
        try:
            yield {model: "http://127.0.0.1:1"}
        finally:
            with suppress_lookup():
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)

    def lister():
        return [{"pid": p.pid, "used_memory_mib": 1.0} for p in state["launched"] if p.poll() is None]

    def cleanup():
        for p in state["launched"]:
            if p.poll() is None:
                with suppress_lookup():
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
    return SimpleNamespace(launcher=launcher, lister=lister, state=state, cleanup=cleanup)


@contextmanager
def suppress_lookup():
    try:
        yield
    except (ProcessLookupError, PermissionError):
        pass


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    fake = _services()
    controls = {}
    for role in identity.ROLES:
        (tmp_path / role / "runtime").mkdir(parents=True)
        controls[role] = tmp_path / role
    session = {"allocation": {"hostname": "test", "slurm_job_id": "1"}}
    monkeypatch.setattr(runtime, "validate", lambda output: (controls, {}, session, {}))
    monkeypatch.setattr(runtime, "legacy_environment", lambda binding, project: nullcontext())
    monkeypatch.setattr(runtime, "role_environment", lambda binding, session, role: nullcontext())
    fake.output = tmp_path / "out"
    yield fake
    fake.cleanup()


def test_deployment_stops_and_verifies_its_own_group(lifecycle):
    record = {}
    with runtime.deployment(lifecycle.output, "kev4b", launcher=lifecycle.launcher, lister=lifecycle.lister,
                            preflight=False, sink=record) as service:
        assert service["pids"] == [lifecycle.state["launched"][0].pid]
    assert record["stop_verification"]["owned_groups"] == service["pids"] and not lifecycle.lister()
    assert study._stop_verified(record)


def test_a_foreign_gpu_process_blocks_the_start_and_is_not_stopped(lifecycle):
    foreign = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        with pytest.raises(StudyError, match="outside this deployment"):
            with runtime.deployment(lifecycle.output, "qwen9b", launcher=lifecycle.launcher,
                                    lister=lambda: [{"pid": foreign.pid, "used_memory_mib": 1.0}], preflight=False):
                pass
        assert foreign.poll() is None and not lifecycle.state["launched"]
    finally:
        foreign.kill()
        foreign.wait()


def test_a_half_started_launch_is_registered_and_its_residual_worker_killed(lifecycle, monkeypatch):
    monkeypatch.setattr(up_runtime, "STOP_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(up_runtime, "STOP_TERM_SECONDS", 0.5)

    @contextmanager
    def failing(control, model):
        leader = subprocess.Popen(["sh", "-c", "(trap '' TERM; exec sleep 60) & wait"], start_new_session=True)
        lifecycle.state["launched"].append(leader)
        time.sleep(0.5)  # let the worker install its TERM trap before the launch fails
        write_json(control / "runtime/last_services.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                                                            "processes": [{"name": model, "pid": leader.pid, "command": ["x"]}]})
        try:
            raise RuntimeError("engine exited during startup")
            yield  # pragma: no cover
        finally:
            with suppress_lookup():
                os.killpg(leader.pid, signal.SIGTERM)
            leader.wait(timeout=10)

    record = {}
    with pytest.raises(RuntimeError, match="during startup"):
        with runtime.deployment(lifecycle.output, "startlux9b", launcher=failing, lister=lambda: [], preflight=False,
                                sink=record):
            pass
    assert record["startup_failed"] and record["stop_verification"]["residual_killed"]
    assert up_runtime.owned_members({lifecycle.state["launched"][0].pid}, time.time() - 60)[0] == []


def test_a_stale_service_record_is_never_signalled(lifecycle):
    bystander = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        control = lifecycle.output.parent / "kev4b"
        write_json(control / "runtime/last_services.json", {"started_at": "2026-01-01T00:00:00+00:00",
                                                            "processes": [{"name": "kev", "pid": bystander.pid}]})

        @contextmanager
        def broken(control, model):
            raise RuntimeError("failed before writing its own record")
            yield  # pragma: no cover

        record = {}
        with pytest.raises(RuntimeError):
            with runtime.deployment(lifecycle.output, "kev4b", launcher=broken, lister=lambda: [], preflight=False, sink=record):
                pass
        assert bystander.poll() is None and "stop_verification" not in record
    finally:
        bystander.kill()
        bystander.wait()


def test_a_binding_without_verified_snapshots_cannot_drive_a_node_run(tmp_path, monkeypatch):
    write_json(tmp_path / "runtime/binding.json", {"snapshots_checked": False, "source_dirs": {}})
    monkeypatch.setattr(runtime, "plans", lambda *a, **k: pytest.fail("an unverified binding must be refused first"))
    with pytest.raises(StudyError, match="verified model snapshots"):
        runtime.verify_binding(tmp_path)
    with pytest.raises(StudyError, match="verified model snapshots"):
        runtime.validate(tmp_path)


# ---------------------------------------------------------------- end-to-end rehearsal on real evidence

def build_clean_repo(destination):
    names = [n for n in subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0") if n]
    destination.mkdir(parents=True)
    for name in names:
        if (ROOT / name).is_file():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination / name)
    git = lambda *args: subprocess.run(["git", *args], cwd=destination, check=True, capture_output=True, text=True)
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=direct-matrix-rehearsal", "-c", "user.email=rehearsal@example.invalid", "-c", "commit.gpgsign=false",
        "commit", "-qm", "Ephemeral rehearsal snapshot (local only, no remote)")
    return git("rev-parse", "HEAD").stdout.strip()


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    if not HAVE_EVIDENCE:
        pytest.skip("Needs the accepted V1/V2/V3/size-control returns")
    # Always outside this checkout: production verification refuses an output inside the source root.
    base = tmp_path_factory.mktemp("dm")
    repo, work = base / "repo", base / "work"
    commit = build_clean_repo(repo)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-I", str(repo / "tests/direct_matrix_rehearsal.py"), "--repo", str(repo),
                             "--evidence", str(EVIDENCE), "--work", str(work)], env=env, capture_output=True, text=True, timeout=1500)
    (base / "rehearsal.log").write_text(result.stdout + result.stderr)
    keep = os.environ.get("DIRECT_MATRIX_REHEARSAL_EVIDENCE")
    if keep:  # copies only; the run itself stays in the temporary directory
        target = Path(keep)
        target.mkdir(parents=True, exist_ok=False)
        for source in (base / "rehearsal.log", work / "summary.json",
                       work / "results/aux-direct-matrix-v1/exports/direct_matrix_bundle.zip"):
            if source.is_file():
                shutil.copy2(source, target / source.name)
        (target / "ephemeral-commit.txt").write_text(commit + "\n")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    summary = json.loads((work / "summary.json").read_text())
    return SimpleNamespace(repo=repo, work=work, commit=commit, summary=summary,
                           output=Path(summary["output"]), extracted=work / "returned-bundle")


@needs_evidence
def test_rehearsal_runs_the_real_node_sequence_and_every_guard_fires(rehearsal):
    steps = {s["name"]: s for s in rehearsal.summary["steps"]}
    assert [s["code"] for s in rehearsal.summary["steps"]] == [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0]
    assert "Child process exited -9" in steps["start-mid-cell-kill"]["error"]
    assert "Child process exited 2" in steps["resume-fault-a"]["error"]
    for name in ("plain-rerun", "new-allocation-without-resume", "plain-rerun-a", "plain-rerun-b"):
        assert "--resume" in steps[name]["error"] and steps[name]["launched"] == 0
    assert "--new-allocation" in steps["other-job-without-ack"]["error"] and steps["other-job-without-ack"]["launched"] == 0
    assert "Child process exited -9" in steps["resume-fault-b"]["error"]
    assert steps["start-again"]["launched"] == 0 and rehearsal.summary["evaluation_unchanged"]
    assert rehearsal.summary["faults_fired"] == ["startlux9b-squad2", "kev4b-squad2", "kev4b-anli"]
    assert rehearsal.summary["scheduler_calls"] == [] and rehearsal.summary["alive_after"] == []


@needs_evidence
def test_interrupted_attempts_are_recovered_without_recollection(rehearsal):
    s = rehearsal.summary
    assert s["after_fault_a"] == s["after_complete"], "accepted rows must never be recollected"
    states = {}
    for cell in ("kev4b-squad2", "kev4b-anli"):
        paths = s["recoveries"][cell]
        assert len(paths) == 1
        saved = data.checked(rehearsal.output / paths[0], "recovery_hash")
        states[cell] = saved["state"]
        assert saved["acknowledged_by"] == "--resume" and saved["rows"] == identity.COUNTS[cell.split("-")[1]]
        completion = read_json(rehearsal.output / f"cells/{cell}/completion.json")
        assert [d["attempt"] for d in completion["deployments"].values()] == ["recovered"]
    assert states == {"kev4b-squad2": "interrupted", "kev4b-anli": "late_or_failed_exit_after_completion_receipt"}


@needs_evidence
def test_a_mid_cell_kill_keeps_journal_rows_and_collects_only_the_remainder(rehearsal):
    m = rehearsal.summary["mid_cell"]
    assert (m["sharded"], m["journal"]) == (50, 3), "SIGKILL after 53 appends: 2 shards plus 3 fsynced journal rows"
    assert m["journal_after_plain_rerun"] == 3 and m["journal_after_refusals"] == 3, "no recovery without --resume"
    assert m["final_rows"] == 4000 and m["final_journal"] == 0
    assert m["old_rows_unchanged"] and m["old_shards_unchanged"], "the 53 earlier rows are kept byte for byte"
    assert sorted(m["deployments"].values()) == [53, 3947] and len(m["attempts_final"]) == 2
    assert m["formal_requests_per_input"] == [1], "every input was requested exactly once in formal collection"
    folder = rehearsal.output / "cells/startlux9b-squad2" / m["attempts"][0]
    recovery = data.checked(folder / "recovery.json", "recovery_hash")
    assert recovery["rows"] == 53 and recovery["state"] == "interrupted"
    journal = recovery["journal_recovery"]
    assert (journal["sharded_rows_before"], journal["sharded_rows_after"]) == (50, 53) and journal["journal_bytes"] > 0
    assert (rehearsal.output / "cells/startlux9b-squad2" / journal["journal_copy"]).read_bytes().count(b"\n") == 3
    completion = read_json(rehearsal.output / "cells/startlux9b-squad2/completion.json")
    assert [d["attempt"] for _, d in sorted(completion["deployments"].items())] == ["recovered", "accepted"]


@needs_evidence
def test_offline_audit_passes_from_the_bundle_alone(rehearsal):
    report = rehearsal.summary["audit"]
    assert report["status"] == "PASS" and report["references_match_accepted"] and len(report["cells"]) == 20
    assert rehearsal.summary["audit_opened_outside_bundle_and_checkout"] == []
    assert rehearsal.summary["audit_opened_evidence_or_node_paths"] == []
    assert report["cells"]["qwen9b-squad2"]["failures"] == 1, "the injected terminal error stays in the denominator"
    accepted = identity.protocol()["references"]
    assert all(report["cells"][c]["correct"] == accepted[c]["accepted_correct"] for c in identity.REFERENCE_CELLS)
    assert audit.audit(rehearsal.extracted)["evaluation_hash"] == report["evaluation_hash"]


@needs_evidence
def test_prepared_inputs_keep_original_values_and_line_hashes(rehearsal):
    out = rehearsal.output
    clinc = data.formal_examples(out, "clinc")
    assert len(clinc) == 600 and all("input_hash" not in r for r in clinc)
    anli = data.formal_examples(out, "anli")
    assert list(anli[0]["criteria"]) != anli[0]["labels"], "regression: canonical criteria order differs from label order"
    assert list(clinc[0]["criteria"]) != clinc[0]["labels"]
    lines = read_json(out / "data/source_lines.json")
    source = (EVIDENCE / "results/returned-experiment-2026-11-02-101054/data/examples.jsonl").read_bytes().split(b"\n")
    import hashlib
    first = lines["clinc"][0]
    assert hashlib.sha256(source[first["line"] - 1]).hexdigest() == first["line_sha256"]
    assert json.loads(source[first["line"] - 1]) == clinc[0]


# ---------------------------------------------------------------- tampering after a valid run

@pytest.fixture
def node_assets(monkeypatch):
    """Fake only the node-only snapshot metadata reads so production verification can run locally."""
    import hashlib
    monkeypatch.setattr(runtime, "_metadata", lambda path: {"config.json": hashlib.sha256(str(path).encode()).hexdigest()})
    monkeypatch.setattr(size_runtime, "_verify_model_metadata", lambda path: None)


@contextmanager
def tampered(*paths):
    saved = {p: (p.read_bytes() if p.is_file() else None) for p in paths}
    trees = {p: p.with_name(p.name + ".saved") for p in paths if p.is_dir()}
    for tree, copy in trees.items():
        shutil.copytree(tree, copy)
    try:
        yield
    finally:
        for tree, copy in trees.items():
            if tree.exists():
                shutil.rmtree(tree)
            copy.rename(tree)
        for path, payload in saved.items():
            if path in trees:
                continue
            if payload is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_bytes(payload)


def _rejected_everywhere(out, cell, message=None):
    ctx = study.context(out)
    for call in (lambda: study.completed(out, ctx, cell), lambda: study.make_freeze(out, require_existing=True),
                 lambda: study.evaluate(out), lambda: audit.audit(out)):
        with pytest.raises((StudyError, OSError, KeyError), match=message):
            call()


@needs_evidence
def test_the_untampered_run_verifies_in_place(rehearsal, node_assets):
    out = rehearsal.output
    assert study.make_freeze(out, require_existing=True)["freeze_hash"] == read_json(out / "freeze.json")["freeze_hash"]
    assert study.evaluate(out) == read_json(out / "evaluation.json")
    assert audit.audit(out)["status"] == "PASS"


@needs_evidence
@pytest.mark.parametrize("role, cell", [("kev4b", "kev4b-anli"), ("qwen9b", "qwen9b-clinc"), ("startlux9b", "startlux9b-squad1")])
def test_a_removed_or_changed_raw_token_report_is_refused(rehearsal, node_assets, role, cell):
    out = rehearsal.output
    report = out / read_json(out / "preflight/tokens.json")["reports"][role]["file"]
    with tampered(report):
        report.unlink()
        _rejected_everywhere(out, cell, "Raw token report")
    with tampered(report):
        value = read_json(report)
        value["rows"][0]["input_tokens"] += 1
        write_json(report, value)
        _rejected_everywhere(out, cell, "Raw token report")


@needs_evidence
def test_a_wrong_reserve_or_overlong_input_is_refused_even_with_a_repaired_binding(rehearsal, node_assets):
    out = rehearsal.output
    tokens = out / "preflight/tokens.json"
    report = out / read_json(tokens)["reports"]["qwen9b"]["file"]
    inputs = study.role_inputs(out)["qwen9b"]
    for change in ({"reserved_output_tokens": 1}, {"input_tokens": 8192 - 63}):
        with tampered(report, tokens):
            value = read_json(report)
            value["rows"][5].update(change)
            write_json(report, value)
            receipt = read_json(tokens)
            receipt["reports"]["qwen9b"]["sha256"] = file_hash(report)
            write_json(tokens, receipt)
            with pytest.raises(StudyError, match="within the frozen limits"):
                runtime.verify_token_report(out, "qwen9b", receipt["reports"]["qwen9b"], inputs)


@needs_evidence
def test_removed_preflight_rows_are_refused(rehearsal, node_assets):
    out = rehearsal.output
    folder = out / read_json(out / "preflight/kev4b.json")["evidence_dir"]
    with tampered(folder / "development-anli/rows.json"):
        (folder / "development-anli/rows.json").unlink()
        _rejected_everywhere(out, "kev4b-anli")


def _preflight_alone_refuses(out, role, message):
    """The preflight check by itself, so no later receipt binding can mask a missing guard."""
    ctx = study.context(out)
    token_sha, longest, _ = study.verify_token_receipt(out, ctx, roles=(role,))
    with pytest.raises(StudyError, match=message):
        study.verify_preflight(out, ctx, role, token_sha=token_sha, longest=longest)


def _repair_preflight(out, role, folder):
    receipt = read_json(out / f"preflight/{role}.json")
    receipt["evidence"] = study._evidence(folder)
    receipt["stages"] = {k: v for k, v in receipt["stages"].items() if (folder / k).is_dir()}
    write_json(folder / "receipt.json", receipt)
    write_json(out / f"preflight/{role}.json", receipt)
    tokens = read_json(out / "preflight/tokens.json")
    return receipt, tokens


@needs_evidence
def test_a_deleted_required_stage_is_refused_even_with_repaired_receipts(rehearsal, node_assets):
    out = rehearsal.output
    folder = out / read_json(out / "preflight/kev4b.json")["evidence_dir"]
    with tampered(folder, out / "preflight/kev4b.json"):
        shutil.rmtree(folder / "development-anli")
        _repair_preflight(out, "kev4b", folder)
        _preflight_alone_refuses(out, "kev4b", "exactly the required stages")
        _rejected_everywhere(out, "kev4b-anli", "exactly the required stages")


@needs_evidence
def test_an_extra_unrequired_stage_is_refused_even_with_repaired_receipts(rehearsal, node_assets):
    """Only the exact inventory check can see a stage nobody asked for; every listed stage still verifies."""
    out = rehearsal.output
    folder = out / read_json(out / "preflight/kev4b.json")["evidence_dir"]
    with tampered(folder, out / "preflight/kev4b.json"):
        shutil.copytree(folder / "development-anli", folder / "development-extra")
        receipt = read_json(out / "preflight/kev4b.json")
        receipt["evidence"] = study._evidence(folder)
        receipt["stages"]["development-extra"] = receipt["stages"]["development-anli"]
        write_json(folder / "receipt.json", receipt)
        write_json(out / "preflight/kev4b.json", receipt)
        _preflight_alone_refuses(out, "kev4b", "exactly the required stages")


@needs_evidence
def test_a_shrunk_development_stage_is_refused_even_with_repaired_hashes(rehearsal, node_assets):
    out = rehearsal.output
    folder = out / read_json(out / "preflight/kev4b.json")["evidence_dir"]
    stage = folder / "development-squad"
    with tampered(folder, out / "preflight/kev4b.json"):
        plan = read_json(stage / "plan.json")
        plan["sample_ids"] = plan["sample_ids"][1:]
        write_json(stage / "plan.json", plan)
        write_json(stage / "rows.json", read_json(stage / "rows.json")[1:])
        complete = {**read_json(stage / "complete.json"), "plan_hash": fingerprint(plan), "count": len(plan["sample_ids"])}
        write_json(stage / "complete.json", complete)
        write_json(stage / "accepted.json", {**read_json(stage / "accepted.json"), "plan_hash": fingerprint(plan),
                                             "complete_sha256": file_hash(stage / "complete.json")})
        _repair_preflight(out, "kev4b", folder)
        _preflight_alone_refuses(out, "kev4b", "differs from the required inputs")
        _rejected_everywhere(out, "kev4b-squad2", "differs from the required inputs")


@needs_evidence
def test_stop_smoke_and_raw_formal_output_closures(rehearsal, node_assets):
    out = rehearsal.output
    cell = "qwen9b-clinc"
    (deployment,) = read_json(out / f"cells/{cell}/completion.json")["deployments"]
    finished = out / f"services/{deployment}/finished.json"
    with tampered(finished):
        value = read_json(finished)
        value["deployment"]["stop_verification"] = {"error": "residual process"}
        write_json(finished, value)
        _rejected_everywhere(out, cell)
    smoke = out / f"cells/{cell}/attempt-{deployment}/smoke.json"
    with tampered(smoke):
        value = read_json(smoke)
        value["current"][0]["latency_ms"] += 1
        write_json(smoke, value)
        _rejected_everywhere(out, cell)
    shard = sorted((out / f"cells/{cell}/predictions/qwen").glob("shard-*.json"))[0]
    with tampered(shard):
        value = read_json(shard)
        raw = value["rows"][0]["raw_response"]
        labels = data.formal_examples(out, "clinc")[0]["labels"]
        other = next(l for l in labels if l != value["rows"][0]["prediction"])
        raw["choices"][0]["message"]["content"] = json.dumps({"label": other})
        value["sha256"] = fingerprint(value["rows"])
        write_json(shard, value)
        _rejected_everywhere(out, cell, "differs from its raw response")


@needs_evidence
def test_an_unaccepted_attempt_cannot_complete_a_cell(rehearsal, node_assets):
    out = rehearsal.output
    cell = "qwen9b-squad1"
    (deployment,) = read_json(out / f"cells/{cell}/completion.json")["deployments"]
    folder = out / f"cells/{cell}/attempt-{deployment}"
    with tampered(folder, out / f"cells/{cell}/completion.json"):
        (out / f"cells/{cell}/completion.json").unlink()
        (folder / "accepted.json").unlink()
        ctx = study.context(out)
        assert study.unreconciled_attempts(out, ctx, cell) == [folder.name]
        assert study.finalize(out, ctx, cell) is None
    with tampered(folder):
        (folder / "complete.json").rename(folder / "unaccepted-complete.json")
        _rejected_everywhere(out, cell)


@needs_evidence
def test_changed_historical_labels_with_unchanged_totals_are_refused(rehearsal):
    out = rehearsal.output
    cell = "startlux4b-squad2"
    path = out / f"references/{cell}.jsonl"
    gold = read_json(out / "sealed/squad2-gold.json")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    right = next(i for i, r in enumerate(rows) if r["prediction"] == gold[r["sample_id"]])
    wrong = next(i for i, r in enumerate(rows) if r["status"] == "ok" and r["prediction"] != gold[r["sample_id"]])
    flip = lambda label: "unanswerable" if label == "answerable" else "answerable"
    rows[right]["prediction"], rows[wrong]["prediction"] = flip(rows[right]["prediction"]), flip(rows[wrong]["prediction"])
    assert sum(r["prediction"] == gold[r["sample_id"]] for r in rows) == 3671
    provenance, manifest = out / "references/provenance.json", out / "data/manifest.json"
    with tampered(path, provenance, manifest):
        path.write_bytes(b"".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in rows))
        value = read_json(provenance)
        value[cell]["compact_hash"] = fingerprint(rows)
        provenance.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        m = read_json(manifest)
        m["files"][f"references/{cell}.jsonl"] = file_hash(path)
        m["files"]["references/provenance.json"] = file_hash(provenance)
        m = data.seal({k: v for k, v in m.items() if k != "manifest_hash"}, "manifest_hash")
        write_json(manifest, m)
        with pytest.raises(StudyError, match="Prepared reference cell changed"):
            data.verify(out)
        settings = read_json(out / "settings.json")
        with pytest.raises(StudyError, match="Prepared reference cell changed"):
            study.prepare(settings["source_dirs"], out, source=settings["source"])


@needs_evidence
def test_exported_gold_subsets_are_bound_to_their_pins(rehearsal):
    gold = rehearsal.extracted / "sealed/anli-gold.json"
    with tampered(gold):
        value = read_json(gold)
        sid = next(iter(value))
        value[sid] = "neutral" if value[sid] != "neutral" else "entailment"
        write_json(gold, value)
        with pytest.raises(StudyError, match="pinned fingerprint"):
            data.load_sealed_gold(rehearsal.extracted)  # the pin alone, before any re-score comparison
        with pytest.raises(StudyError, match="pinned fingerprint"):
            audit.audit(rehearsal.extracted)


@needs_evidence
def test_status_validates_receipts_and_reports_tampering(rehearsal, capsys):
    out = rehearsal.output
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        _status_checks(out)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _status_checks(out):
    assert cli.main(["status", "--output-dir", str(out)]) == 0
    evaluation = out / "evaluation.json"
    with tampered(evaluation):
        value = read_json(evaluation)
        value["new_predictions"] = 0
        write_json(evaluation, value)
        result = study.status(out)
        assert result["evaluation"].startswith("invalid") and result["problems"]
        assert cli.main(["status", "--output-dir", str(out)]) == 1


@needs_evidence
def test_session_initialization_never_extends_or_silently_switches_allocations(rehearsal, node_assets, monkeypatch, tmp_path):
    settings = read_json(rehearsal.output / "settings.json")
    out = tmp_path / "init"
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(StudyError, match="SLURM_JOB_ID"):
        runtime.initialize(settings["source_dirs"], out, 1)
    monkeypatch.setenv("SLURM_JOB_ID", "5001")
    first = runtime.initialize(settings["source_dirs"], out, 1)["session"]
    later = runtime.initialize(settings["source_dirs"], out, 5)["session"]
    assert later["deadline_epoch"] <= first["deadline_epoch"]
    with pytest.raises(StudyError, match="same Slurm job"):
        runtime.initialize(settings["source_dirs"], out, 1, new_allocation=True)
    monkeypatch.setenv("SLURM_JOB_ID", "5002")
    with pytest.raises(StudyError, match="--new-allocation"):
        runtime.initialize(settings["source_dirs"], out, 1)
    runtime.initialize(settings["source_dirs"], out, 1, new_allocation=True)
    assert list((out / "runtime/session-history").glob("*.json"))
    assert read_json(out / "runtime/binding.json")["snapshots_checked"] is True
