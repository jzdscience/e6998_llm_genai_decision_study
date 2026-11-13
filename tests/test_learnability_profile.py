"""Development profile boundaries, evidence binding and owned process cleanup.

Artificial fixtures only. The Linux subreaper test exercises real subprocesses,
not model services; it is explicitly skipped on macOS.
"""
from contextlib import contextmanager, nullcontext
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import pytest

from decision_study.common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from decision_study.learnability_profile import study as s


def row(number):
    return {"sample_id": f"anli:artificial-{number:03d}", "task": "anli", "split": "development",
            "group_id": f"g{number}", "round": f"R{number//30+1}", "source_split": "train",
            "state": f"Artificial fixture {number}", "instructions": "Classify",
            "labels": ["entailment", "neutral", "contradiction"],
            "criteria": {"entailment": "E", "neutral": "N", "contradiction": "C"},
            "premise_hash": f"p{number}", "input_hash": f"i{number}"}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    runtime = tmp_path/"runtime"
    dirs = {"v1": runtime/"results/solo-v1", "v3": runtime/"results/solo-v3",
            "size_control": runtime/"results/aux-size-control-v1-r2"}
    for d in dirs.values():
        d.mkdir(parents=True)
    rows = [row(n) for n in range(90)]
    source = dirs["v3"]/"data/examples.jsonl"
    source.parent.mkdir()
    source.write_text("".join(json.dumps(r)+"\n" for r in rows))
    # Deliberately invalid files: any accidental gold read should fail the test.
    for relative in ("data/fit_gold.json", "sealed/holdout_gold.json"):
        p = dirs["v3"]/relative
        p.parent.mkdir(exist_ok=True)
        p.write_text("DO NOT READ GOLD")
    monkeypatch.setattr(s, "V3_EXAMPLES_SHA256", file_hash(source))
    monkeypatch.setattr(s, "DEVELOPMENT_SHA256", fingerprint(rows))
    monkeypatch.setattr(s, "source_identity", lambda: {"files": {"fake.py": "a"*64}, "git_commit": "fake",
                                                     "git_dirty": True, "note": "artificial fixture"})
    controls = {}
    for role in s.ROLES:
        model = s.INTERFACE[role]
        snapshot = tmp_path/"snapshots"/s.REVISIONS[role]
        snapshot.mkdir(parents=True)
        controls[role] = {"lock.json": {"max_model_len": 8192, "models": {model: {"revision": s.REVISIONS[role]}}},
                          "installations.json": {"models": {model: {"revision": s.REVISIONS[role], "model_path": str(snapshot)}}},
                          "selected_profiles.json": {model: "primary"}, "selected_memory.json": {model: .5}}
    binding = {"v1_dir": str(dirs["v1"]), "v3_dir": str(dirs["v3"]), "size_control_dir": str(dirs["size_control"]),
               "source_work_dir": str(dirs["v1"]/"runtime/work"), "work_dir": str(runtime/"results/aux-anli-learnability-profile-v1/runtime-work")}
    monkeypatch.setattr(s.old, "plans", lambda *a, **kw: (copy.deepcopy(binding), copy.deepcopy(controls)))
    output = runtime/"results/aux-anli-learnability-profile-v1"
    s.prepare(dirs, output)
    allocation = {"hostname": "test-node", "slurm_job_id": "123"}
    monkeypatch.setattr(s.v3, "_allocation", lambda: allocation)
    monkeypatch.setattr(s, "slurm_hours_left", lambda: 2)
    s.initialize(output)
    selftest = output/"ownership-selftest/artificial"
    selftest.mkdir(parents=True)
    for name in ("actor-ready.json", "decoy.json", "actor-cleanup.jsonl", "outer-cleanup.jsonl", "outer-stop.json"):
        write_json(selftest/name, {"artificial_test_fixture": True})
    write_json(selftest/"actor-result.json", {"status": "PASS", "decoy_survived": True})
    receipt = {"status": "PASS", "settings_hash": fingerprint(read_json(output/"settings.json")),
               "allocation": allocation.copy(), "folder": str(selftest.relative_to(output)),
               "files": {p.name: file_hash(p) for p in selftest.iterdir()}}
    write_json(selftest/"receipt.json", receipt)
    write_json(output/"runtime/ownership-selftest.json", {"file": str((selftest/"receipt.json").relative_to(output)),
                                                         "sha256": file_hash(selftest/"receipt.json")})
    return output, dirs, rows, allocation


def fake_runtime(role):
    return {"model_revision": s.REVISIONS[role], "max_model_len": 8192,
            "gpu_memory_utilization": .8, "truncate": False}


def predictions(rows, role):
    result = []
    for number, r in enumerate(rows):
        probs = {"entailment": .9, "neutral": .05, "contradiction": .05}
        raw = ({"choices": [{"finish_reason": "stop", "message": {"content": '{"label":"entailment"}'}}]}
               if role == "qwen9b" else {"answers": {"decision": {"choice": "entailment", "probabilities": probs}}})
        result.append({"sample_id": r["sample_id"], "task": "anli", "split": "development", "model": s.INTERFACE[role],
                       "status": "ok", "prediction": "entailment", "probabilities": None if role == "qwen9b" else probs,
                       "raw_response": raw, "attempts": [{"attempt": 1, "status": "ok"}], "latency_ms": 2.0,
                       "request_started_monotonic": number*.002, "request_finished_monotonic": (number+1)*.002,
                       "observed_request_seconds": .002, "profile_acceptance": "on_time",
                       "input_sha256": fingerprint(r), "request_sha256": fingerprint(s.request_body(r, s.INTERFACE[role]))})
    return result


def completed_attempt(output, rows, role):
    folder = output/"attempts"/role/"0001"
    folder.mkdir(parents=True)
    settings = read_json(output/"settings.json")
    sess = read_json(output/"runtime/session.json")
    plan = {"role": role, "settings_hash": fingerprint(settings), "allocation": sess["allocation"],
            "started_epoch": time.time()-1, "deadline_epoch": time.time()+120, "last_services_before": None}
    plan["ownership_selftest"] = read_json(output/"runtime/ownership-selftest.json")
    write_json(folder/"plan.json", plan)
    runtime = fake_runtime(role)
    token_path = output/f"runtime-controls/{role}/runtime/tokens-interactive-{s.INTERFACE[role]}-{s.frozen._hash(rows)[:16]}.json"
    tokens = {"status": "ok", "model": s.INTERFACE[role], "model_revision": s.REVISIONS[role], "max_model_len": 8192,
              "n_examples": len(rows), "truncated": False, "overlong": [], "effective_config_hash": fingerprint(runtime),
              "rows": [{"sample_id": r["sample_id"], "input_tokens": 50, "reserved_output_tokens": 64 if role == "qwen9b" else 1}
                       for r in rows]}
    write_json(token_path, tokens)
    write_json(folder/"tokens.json", {"file": str(token_path.relative_to(output)), "sha256": file_hash(token_path),
                                     "runtime_hash": fingerprint(runtime)})
    write_json(folder/"hardware.json", {"artificial": True})
    (folder/"resources.jsonl").write_text('{"artificial":true}\n')
    (folder/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in predictions(rows, role)))
    processes = [{"name": s.INTERFACE[role]+"-engine", "pid": 123456,
                  "command": ["python", "-m", "vllm.entrypoints.openai.api_server", "--gpu-memory-utilization", "0.8", "--max-num-seqs", "64"]}]
    stop = {"gpu_compute_processes": [], "owned_groups": [123456]}
    write_json(folder/"deployment.json", {"processes": processes, "runtime": runtime, "stop": stop,
                                           "startup_seconds": 5.0, "collection_seconds": .18,
                                           "startup_started_monotonic": 0.0, "ready_monotonic": 5.0,
                                           "collection_started_monotonic": 0.0, "collection_finished_monotonic": .18,
                                           "latency_ms": {"median": 2.0, "max": 2.0}})
    files = {n: file_hash(folder/n) for n in ("plan.json", "tokens.json", "hardware.json", "resources.jsonl", "rows.jsonl", "deployment.json")}
    write_json(folder/"complete.json", {"role": role, "count": len(rows), "files": files})
    write_json(folder/"parent-stop.json", stop)
    write_json(folder/"accepted.json", {"role": role, "returncode": 0, "plan_hash": fingerprint(plan),
                                        "complete_sha256": file_hash(folder/"complete.json"),
                                        "parent_stop_sha256": file_hash(folder/"parent-stop.json"),
                                        "exited_epoch": time.time(), "accepted_epoch": time.time(),
                                        "deadline_epoch": plan["deadline_epoch"]})
    return folder


def test_prepare_only_uses_exact_exposed_train_rows_and_new_controls(prepared):
    output, dirs, rows, _ = prepared
    settings, got = s.context(output)
    assert got == rows and settings["source"]["git_dirty"] is True
    for role in s.ROLES:
        assert read_json(s.control(output, role)/"runtime/selected_memory.json") == {s.INTERFACE[role]: .8}
    assert (dirs["v3"]/"sealed/holdout_gold.json").read_text() == "DO NOT READ GOLD"


def test_new_eval_row_cannot_enter_profile_even_if_source_hash_is_rebound(prepared, monkeypatch):
    _, dirs, _, _ = prepared
    p = dirs["v3"]/"data/examples.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    rows[0]["source_split"] = "dev"
    p.write_text("".join(json.dumps(r)+"\n" for r in rows))
    monkeypatch.setattr(s, "V3_EXAMPLES_SHA256", file_hash(p))
    monkeypatch.setattr(s, "DEVELOPMENT_SHA256", fingerprint(rows))
    with pytest.raises(StudyError, match="train-only"):
        s.inputs(dirs["v3"])


@pytest.mark.parametrize("target", ["solo-v1", "solo-v3", "aux-size-control-v1-r2"])
def test_cli_rejects_old_roots_before_any_lock_or_diagnostic_write(prepared, target):
    output, _, _, _ = prepared
    runtime = output.parent.parent
    old = runtime/"results"/target
    before = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert s.main(["start", "--runtime-root", str(runtime), "--output-dir", str(old)]) == 2
    assert s.main(["worker", "--output-dir", str(old), "--attempt-dir", str(old/"attempts/invalid")]) == 2
    after = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert before == after
    assert not (old/".active.lock").exists()


def test_session_does_not_extend_and_handover_requires_verified_ended_job(prepared, monkeypatch):
    output, _, _, allocation = prepared
    before = read_json(output/"runtime/session.json")["deadline_epoch"]
    monkeypatch.setattr(s, "slurm_hours_left", lambda: 20)
    assert s.initialize(output)["deadline_epoch"] == before
    with pytest.raises(StudyError, match="same allocation"):
        s.initialize(output, new_allocation=True)
    allocation["slurm_job_id"] = "124"
    with pytest.raises(StudyError, match="genuinely new"):
        s.initialize(output)
    monkeypatch.setattr(s, "slurm_job_state", lambda job: {"terminal": False, "job_id": job})
    with pytest.raises(StudyError, match="earlier Slurm"):
        s.initialize(output, new_allocation=True)
    monkeypatch.setattr(s, "slurm_job_state", lambda job: {"terminal": True, "job_id": job})
    assert s.initialize(output, new_allocation=True)["allocation"]["slurm_job_id"] == "124"


@pytest.mark.parametrize("what", ["source", "input", "control"])
def test_bound_sources_inputs_and_controls_cannot_change(prepared, monkeypatch, what):
    output, _, _, _ = prepared
    if what == "source":
        monkeypatch.setattr(s, "source_identity", lambda: {"files": {"fake.py": "b"*64}})
    elif what == "input":
        path = output/"data/development.json"
        value = read_json(path); value[0]["state"] += " changed"; write_json(path, value)
    else:
        write_json(s.control(output, "qwen9b")/"runtime/selected_memory.json", {"qwen": .4})
    with pytest.raises(StudyError):
        s.context(output)


@pytest.mark.parametrize("role", s.ROLES)
def test_completed_role_reparses_responses_and_requires_tokens_launch_and_stop(prepared, role):
    output, _, rows, _ = prepared
    folder = completed_attempt(output, rows, role)
    assert s.verify_attempt(output, folder, role)["count"] == 90
    assert s.verify_attempt(output, folder, role, offline=True)["count"] == 90
    receipt = read_json(folder/"tokens.json")
    raw_path = output/receipt["file"]
    raw = read_json(raw_path); raw["rows"][0]["input_tokens"] = 9000; write_json(raw_path, raw)
    with pytest.raises(StudyError, match="token report"):
        s.verify_attempt(output, folder, role)


@pytest.mark.parametrize("mutation", ["cap", "stop", "raw", "accepted", "negative_time", "aggregate"])
def test_completed_attempt_rejects_semantic_tampering_after_rehash(prepared, mutation):
    output, _, rows, _ = prepared
    role = "qwen9b"; folder = completed_attempt(output, rows, role)
    if mutation in {"cap", "stop", "negative_time", "aggregate"}:
        p = folder/"deployment.json"; value = read_json(p)
        if mutation == "cap": value["processes"][0]["command"][-1] = "256"
        elif mutation == "stop": value["stop"]["gpu_compute_processes"] = [{"pid": 999}]
        elif mutation == "negative_time": value["collection_seconds"] = -1
        else: value["latency_ms"]["median"] = 999
        write_json(p, value)
    elif mutation == "raw":
        p = folder/"rows.jsonl"; values = s.read_rows(p)
        values[0]["raw_response"]["choices"][0]["message"]["content"] = '{"label":"neutral"}'
        p.write_text("".join(json.dumps(r)+"\n" for r in values))
    else:
        p = folder/"accepted.json"; value = read_json(p); value["exited_epoch"] = time.time()+10000; write_json(p, value)
    if mutation != "accepted":
        complete = read_json(folder/"complete.json"); complete["files"][p.name] = file_hash(p); write_json(folder/"complete.json", complete)
        accept = read_json(folder/"accepted.json"); accept["complete_sha256"] = file_hash(folder/"complete.json"); write_json(folder/"accepted.json", accept)
    with pytest.raises(StudyError):
        s.verify_attempt(output, folder, role)


def test_startup_failure_keeps_owned_engine_for_stop_verification(prepared, monkeypatch):
    output, _, _, _ = prepared
    role = "startlux4b"; calls = []; record = {}
    monkeypatch.setattr(s.old, "assert_gpu_owned_by", lambda groups, **kw: [])
    monkeypatch.setattr(s, "observe_stopped", lambda groups, **kw: calls.append(set(groups)) or {"gpu_compute_processes": [], "owned_groups": sorted(groups)})
    @contextmanager
    def launch(target, model, **kwargs):
        assert kwargs["max_num_seqs"] == 64
        write_json(target/"runtime/last_services.json", {"started_at": utc_now(), "processes": [{"pid": 444}]})
        raise RuntimeError("frontend failed after engine start")
        yield  # pragma: no cover
    with pytest.raises(RuntimeError, match="frontend failed"):
        with s.deployment(output, role, sink=record, launch=launch):
            pass
    assert calls == [{444}]
    assert record["stop"]["owned_groups"] == [444]


def test_engine_launcher_path_is_used_on_normal_entry(prepared, monkeypatch):
    output, _, _, _ = prepared
    role = "qwen9b"; called = []; record = {}
    monkeypatch.setattr(s.old, "assert_gpu_owned_by", lambda groups, **kw: [])
    monkeypatch.setattr(s, "observe_stopped", lambda groups, **kw: {"gpu_compute_processes": [], "owned_groups": sorted(groups)})
    monkeypatch.setattr(s.old, "engine_log_facts", lambda p: {})
    @contextmanager
    def launch(target, model, *, max_num_seqs):
        called.append((model, max_num_seqs))
        processes = [{"pid": 444, "name": "qwen-engine", "command": ["--gpu-memory-utilization", "0.8", "--max-num-seqs", "64"]}]
        write_json(target/"runtime/last_services.json", {"started_at": utc_now(), "processes": processes,
                                                        "models": {model: {"config": fake_runtime(role), "health": {"ok": True}}}})
        yield {model: "http://127.0.0.1:18000"}
    monkeypatch.setattr(s, "managed_service", launch)
    with s.deployment(output, role, sink=record) as endpoint:
        assert endpoint == "http://127.0.0.1:18000"
    assert called == [("qwen", 64)] and record["stop"]["owned_groups"] == [444]


def test_stop_child_never_signals_a_reused_exited_leader_pid(monkeypatch):
    class Exited:
        pid = 900
        def poll(self): return 0
        def terminate(self): pytest.fail("Exited PID might have been recycled")
        def kill(self): pytest.fail("Exited PID might have been recycled")
    s.stop_child(Exited(), time.time()-1)


def test_worker_stop_observation_never_signals_residual_members(monkeypatch):
    monkeypatch.setattr(s.old, "process_table", lambda: [{"pgid": 900, "pid": 901}])
    monkeypatch.setattr(s.old, "_signal", lambda *a: pytest.fail("unverified process signalled"))
    with pytest.raises(StudyError, match="parent ancestry"):
        s.observe_stopped({900}, lister=lambda: [], grace=0)


def test_worker_failure_never_accepts_and_cleanup_receipt_is_preserved(prepared, monkeypatch):
    output, _, _, _ = prepared
    monkeypatch.setattr(s, "subreaper", nullcontext)
    monkeypatch.setattr(s, "worker_command", lambda *a: [sys.executable, "-c", "raise SystemExit(3)"])
    monkeypatch.setattr(s, "stop_child", lambda *a: None)
    monkeypatch.setattr(s, "recover_stop", lambda *a, **kw: {"gpu_compute_processes": [], "owned_groups": []})
    with pytest.raises(StudyError, match="exit 3"):
        s.run_attempt(output, "qwen9b")
    folder = s.attempts(output, "qwen9b")[0]
    assert not (folder/"accepted.json").exists()
    assert read_json(folder/"failure.json")["parent_stop"]["gpu_compute_processes"] == []
    with pytest.raises(StudyError, match="retry-failed"):
        # Prior roles can be accepted fixtures so the retry check reaches Qwen.
        completed_attempt(output, s.context(output)[1], "startlux4b")
        s.run(output)


def test_complete_three_roles_resume_without_launch_and_offline_audit(prepared, monkeypatch):
    output, _, rows, _ = prepared
    for role in s.ROLES:
        completed_attempt(output, rows, role)
    monkeypatch.setattr(s, "run_attempt", lambda *a: pytest.fail("accepted role relaunched"))
    assert len(s.run(output)["roles"]) == 3
    assert s.audit(output)["status"] == "PASS"


@pytest.mark.parametrize("late", [False, True])
def test_whole_worker_fake_service_records_rows_then_stops_before_complete(prepared, monkeypatch, late):
    output, _, rows, _ = prepared
    role = "qwen9b"; folder = output/"attempts"/role/"worker-check"; folder.mkdir(parents=True)
    settings = read_json(output/"settings.json"); sess = read_json(output/"runtime/session.json")
    plan = {"role": role, "settings_hash": fingerprint(settings), "allocation": sess["allocation"],
            "started_epoch": time.time(), "deadline_epoch": time.time()+60, "last_services_before": None}
    plan["ownership_selftest"] = read_json(output/"runtime/ownership-selftest.json")
    write_json(folder/"plan.json", plan)
    monkeypatch.setattr(s.frozen, "doctor", lambda *a, **kw: {"gpu": [{"name": "NVIDIA L40S", "memory_mb": 46068}]})
    monkeypatch.setattr(s.v3, "_cuda_preflight", lambda *a: {})
    monkeypatch.setattr(s.frozen, "runtime_config", lambda *a: fake_runtime(role))
    def token_fake(project, target, model, examples):
        value = {"status": "ok", "model": model, "model_revision": s.REVISIONS[role], "max_model_len": 8192,
                 "n_examples": len(examples), "truncated": False, "overlong": [], "effective_config_hash": fingerprint(fake_runtime(role)),
                 "rows": [{"sample_id": r["sample_id"], "input_tokens": 50, "reserved_output_tokens": 64} for r in examples]}
        write_json(target/f"runtime/tokens-interactive-{model}-{s.frozen._hash(examples)[:16]}.json", value)
        return value
    monkeypatch.setattr(s.frozen, "token_preflight_examples", token_fake)
    monkeypatch.setattr(s.old, "assert_gpu_owned_by", lambda *a, **kw: [])
    events = []
    def stopped(groups, **kw):
        events.append("stopped")
        assert not (folder/"complete.json").exists()
        return {"gpu_compute_processes": [], "owned_groups": sorted(groups)}
    monkeypatch.setattr(s, "observe_stopped", stopped)
    monkeypatch.setattr(s.old, "engine_log_facts", lambda *a: {})
    @contextmanager
    def launch(target, model, *, max_num_seqs):
        assert max_num_seqs == 64
        write_json(target/"runtime/last_services.json", {"started_at": utc_now(),
            "processes": [{"pid": 444, "name": "qwen-engine", "command": ["--gpu-memory-utilization", "0.8", "--max-num-seqs", "64"]}],
            "models": {model: {"config": fake_runtime(role), "health": {"ok": True}}}})
        yield {model: "http://127.0.0.1:18000"}
    monkeypatch.setattr(s, "managed_service", launch)
    def monitor(pids, path):
        path.write_text('{"artificial":true}\n')
        return nullcontext()
    monkeypatch.setattr(s, "ResourceMonitor", monitor)
    by_id = {r["sample_id"]: r for r in predictions(rows, role)}
    offset = [0]; original_monotonic = time.monotonic
    class Clock:
        def monotonic(self): return original_monotonic()+offset[0]
        def __getattr__(self, name): return getattr(time, name)
    monkeypatch.setattr(s, "time", Clock())
    class Client:
        def __init__(self, model, endpoint, **kw):
            assert model == "qwen" and kw == {"timeout": 120, "max_attempts": 1}
        def predict(self, example):
            events.append(example["sample_id"])
            value = copy.deepcopy(by_id[example["sample_id"]]); value["latency_ms"] = 0.0
            if late: offset[0] += 121
            return value
    monkeypatch.setattr(s, "ModelClient", Client)
    if late:
        with pytest.raises(StudyError):
            s.worker(output, folder)
        assert not (folder/"complete.json").exists()
        preserved = s.read_rows(folder/"rows.jsonl")
        assert len(preserved) == 1  # Reject immediately, before the next request.
        assert preserved[0]["profile_acceptance"] == "late"
    else:
        s.worker(output, folder)
        assert len(events) == 91
        assert s.verify_attempt(output, folder, role, accepted=False)["count"] == 90
    assert events[-1] == "stopped"
    assert not (folder/"accepted.json").exists()  # Only the parent can accept an on-time exit.


def test_absolute_watchdog_stops_real_worker_and_keeps_failed_attempt(prepared, monkeypatch):
    output, _, _, _ = prepared
    real_time = time.time; real_popen = subprocess.Popen; offset = [0]; launched = []
    class Clock:
        def time(self): return real_time()+offset[0]
        def __getattr__(self, name): return getattr(time, name)
    def popen(args, **kw):
        proc = real_popen(args, **kw)
        if args[:2] == [sys.executable, "-c"]:
            launched.append(proc)
            offset[0] = 3601  # The monotonic-sized stage's absolute deadline has now expired.
        return proc
    monkeypatch.setattr(s, "time", Clock())
    monkeypatch.setattr(s, "subreaper", nullcontext)
    monkeypatch.setattr(s, "worker_command", lambda *a: [sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(s.subprocess, "Popen", popen)
    monkeypatch.setattr(s, "recover_stop", lambda *a, **kw: {"gpu_compute_processes": [], "owned_groups": []})
    with pytest.raises(StudyError, match="absolute watchdog"):
        s.run_attempt(output, "qwen9b")
    assert launched and launched[0].poll() is not None
    folder = s.attempts(output, "qwen9b")[0]
    assert (folder/"failure.json").exists() and not (folder/"accepted.json").exists()


def test_subreaper_failure_prevents_worker_launch(prepared, monkeypatch):
    output, _, _, _ = prepared
    @contextmanager
    def blocked():
        raise StudyError("cannot establish subreaper")
        yield  # pragma: no cover
    monkeypatch.setattr(s, "subreaper", blocked)
    monkeypatch.setattr(s, "worker_command", lambda *a: pytest.fail("worker launched without ownership"))
    with pytest.raises(StudyError, match="subreaper"):
        s.run_attempt(output, "qwen9b")


def test_export_prunes_cache_and_rejects_symlink_directories(prepared):
    output, _, _, _ = prepared
    cache = output/"runtime-work"/"model"; cache.mkdir(parents=True); (cache/"weight").write_text("not exported")
    bundle = s.export(output)
    with zipfile.ZipFile(bundle) as z:
        assert "BUNDLE_MANIFEST.json" in z.namelist()
        assert not any(n.startswith("runtime-work/") for n in z.namelist())
    (output/"linked-evidence").symlink_to(output/"data", target_is_directory=True)
    with pytest.raises(StudyError, match="Symlinked directory"):
        s.export(output)


def test_current_ancestry_excludes_recycled_recorded_pids_and_handles_deeper_adoptions(tmp_path, monkeypatch):
    began = time.time()-1
    root = os.getpid()
    active = {111: {"pid": 111, "ppid": root, "pgid": 111, "started": began+.1, "zombie": False}}
    decoy = {"pid": 900, "ppid": 1, "pgid": 900, "started": began+.2, "zombie": False}
    monkeypatch.setattr(s, "process_rows", lambda: [*active.values(), decoy])
    monkeypatch.setattr(s.old, "_launch_record", lambda *a: {"processes": [{"pid": 900}]})
    monkeypatch.setattr(s.old, "assert_gpu_owned_by", lambda groups: [])
    signalled = []
    monkeypatch.setattr(s.os, "pidfd_open", lambda pid: pid+1000000, raising=False)
    original_close = os.close
    monkeypatch.setattr(s.os, "close", lambda fd: None if fd >= 1000000 else original_close(fd))
    def send(fd, sig):
        pid = fd-1000000
        signalled.append(pid)
        assert pid != 900, "Reused historical PID is not owned by ancestry"
        active.pop(pid)
        # More than two adoption generations; a fixed two-pass cleanup misses these.
        if pid < 115:
            active[pid+1] = {"pid": pid+1, "ppid": root, "pgid": pid+1, "started": began+.3, "zombie": False}
    monkeypatch.setattr(s.signal, "pidfd_send_signal", send, raising=False)
    result = s.recover_stop(tmp_path, "qwen9b", {"started_epoch": began, "last_services_before": None},
                            worker_pid=900, max_seconds=3, term_seconds=0)
    assert signalled == [111, 112, 113, 114, 115]
    assert result["recorded_engine_groups"] == [900] and not active


def test_pidfd_signal_rechecks_ancestry_after_open(monkeypatch):
    calls = []
    selections = iter([{111: {"pid": 111}}, {}])
    monkeypatch.setattr(s, "owned_descendants", lambda since: next(selections))
    monkeypatch.setattr(s.os, "pidfd_open", lambda pid: 1000001, raising=False)
    original_close = os.close
    monkeypatch.setattr(s.os, "close", lambda fd: calls.append("closed") if fd == 1000001 else original_close(fd))
    monkeypatch.setattr(s.signal, "pidfd_send_signal", lambda *a: pytest.fail("PID no longer belongs to our ancestry"), raising=False)
    assert s.signal_descendant(111, time.time()-1, signal.SIGTERM) is False
    assert calls == ["closed"]


def test_missing_or_changed_ownership_gate_prevents_model_work(prepared, monkeypatch):
    output, _, _, _ = prepared
    (output/"runtime/ownership-selftest.json").unlink()
    monkeypatch.setattr(s, "run_attempt", lambda *a: pytest.fail("model work before Linux ownership gate"))
    with pytest.raises((StudyError, FileNotFoundError)):
        s.run(output)


@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux child-subreaper ownership; node acceptance still required")
def test_linux_subeaper_adopts_separate_session_orphan_and_stops_it(tmp_path, monkeypatch):
    # Leader exits; its separate-session child ignores TERM. Kernel subreaper
    # adoption supplies ownership even after PPID changes and PGID differs.
    pidfile = tmp_path/"pid"
    child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    code = "import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c',sys.argv[2]],start_new_session=True); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(.3)"
    began = time.time()
    monkeypatch.setattr(s.old, "assert_gpu_owned_by", lambda groups: [])
    with s.subreaper():
        parent = subprocess.Popen([sys.executable, "-c", code, str(pidfile), child_code], start_new_session=True)
        parent.wait(timeout=5)
        child = int(pidfile.read_text())
        try:
            assert child in s.adopted_groups(began)
            stop = s.recover_stop(tmp_path, "qwen9b", {"started_epoch": began, "last_services_before": None},
                                  worker_pid=parent.pid, max_seconds=5, term_seconds=.1)
            assert {"pid": child, "signal": "SIGKILL"} in stop["signals"]
            assert s.adopted_groups(began) == set()
        finally:
            try: os.kill(child, signal.SIGKILL)
            except ProcessLookupError: pass
            try: os.waitpid(child, os.WNOHANG)
            except ChildProcessError: pass


@pytest.mark.skipif(sys.platform != "linux", reason="Native Linux node gate; no GPU/model calls")
def test_native_ownership_gate_double_fork_and_fresh_unrelated_decoy(prepared):
    from decision_study.learnability_profile import ownership
    output, _, _, _ = prepared
    receipt = ownership.run_selftest(output)
    assert receipt["status"] == "PASS"
    pointer = ownership.require_receipt(output)
    assert file_hash(output/pointer["file"]) == pointer["sha256"]
    actor = read_json(output/receipt["folder"]/"actor-result.json")
    assert actor["decoy_survived"] is True
    assert not any(r["pid"] == actor["decoy_pid"] for r in actor["stop"]["signals"])
    assert actor["stop"]["gpu_checked"] is False
