"""Artificial gold-free cases for resumable collection and complete token pooling."""
from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from decision_study.common import StudyError, file_hash, fingerprint, read_json, write_json
from decision_study.learnability import collection as c


def row(i):
    return {"sample_id": f"anli:artificial-{i}", "task": "anli", "split": "fitting", "group_id": f"g{i}",
            "round": "R1", "source_split": "train", "state": f"Premise:\nP{i}\n\nHypothesis:\nH{i}",
            "instructions": "Classify", "labels": ["entailment", "neutral", "contradiction"],
            "criteria": {"entailment": "E", "neutral": "N", "contradiction": "C"}, "premise_hash": f"p{i}", "input_hash": f"i{i}"}


def pred(r, role="qwen9b"):
    probs = {"entailment": .9, "neutral": .05, "contradiction": .05}
    raw = ({"choices": [{"finish_reason": "stop", "message": {"content": '{"label":"entailment"}'}}]}
           if role == "qwen9b" else {"answers": {"decision": {"choice": "entailment", "probabilities": probs}}})
    return {"sample_id": r["sample_id"], "task": "anli", "split": r["split"], "model": c.p.INTERFACE[role],
            "status": "ok", "prediction": "entailment", "probabilities": None if role == "qwen9b" else probs,
            "raw_response": raw, "attempts": [{"attempt": 1, "status": "ok"}], "latency_ms": 1.,
            "request_started_monotonic": 10., "request_finished_monotonic": 10.001,
            "observed_request_seconds": .001, "profile_acceptance": "on_time",
            "input_sha256": fingerprint(r), "request_sha256": fingerprint(c.p.request_body(r, c.p.INTERFACE[role]))}


def stopped(plan=None):
    return {"plan_hash": fingerprint(plan) if plan is not None else "artificial-unbound",
            "gpu_compute_processes": [], "gpu_checked": True,
            "ownership": "current descendants of sole-worker Linux subreaper; kernel pidfds; historical PIDs never authorize signals"}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    output, profile = tmp_path/"collection", tmp_path/"profile"
    rows = [row(i) for i in range(3)]
    profile_settings = {"source_dirs": {"v1": str(tmp_path/"old1"), "v3": str(tmp_path/"old3"), "size_control": str(tmp_path/"old9")}}
    controls = {role: {n: {"artificial": n, "role": role} for n in c.p.CONTROL_NAMES} for role in c.ROLES}
    binding = {"control_identity": {role: {n: fingerprint(v) for n, v in d.items()} for role, d in controls.items()}}
    monkeypatch.setattr(c.p, "plans", lambda *args: (copy.deepcopy(binding), copy.deepcopy(controls)))
    monkeypatch.setattr(c.p, "context", lambda *args, **kw: (copy.deepcopy(profile_settings), []))
    monkeypatch.setattr(c.p, "source_identity", lambda: {"files": {"fake.py": "f"*64}})
    allocation = {"hostname": "test", "slurm_job_id": "123"}
    sess = {"allocation": allocation, "deadline_epoch": time.time()+3600}
    monkeypatch.setattr(c.p, "session", lambda *args: copy.deepcopy(sess))
    monkeypatch.setattr(c.ownership, "require_receipt", lambda *args: {"file": "artificial", "sha256": "a"*64})
    monkeypatch.setattr(c.ownership, "verify_receipt", lambda *args: None)
    settings = c.prepare(output, profile, rows, {"artificial_study": "test"})
    return output, profile, rows, settings, allocation


def fake_attempt(prepared, role="qwen9b", start=0, count=3, name="0001"):
    output, profile, rows, settings, allocation = prepared
    folder = output/f"collection/attempts/{role}/{name}";folder.mkdir(parents=True)
    plan = {"role": role, "start": start, "collection_hash": fingerprint(settings), "allocation": allocation,
            "ownership_selftest": {"file": "artificial"}, "started_epoch": time.time()-1,
            "deadline_epoch": time.time()+300, "last_services_before": None}
    write_json(folder/"plan.json", plan)
    runtime = {"model_revision": c.p.REVISIONS[role], "max_model_len": 8192, "gpu_memory_utilization": .8, "truncate": False}
    launch = {"runtime": runtime, "processes": [{"pid": 888, "name": c.p.INTERFACE[role]+"-engine",
              "command": ["python", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.8"]}]}
    write_json(folder/"deployment-open.json", launch)
    report_path = output/f"runtime-controls/{role}/runtime/tokens-interactive-{c.p.INTERFACE[role]}-{c.p.frozen._hash(rows)[:16]}.json"
    report = {"status": "ok", "model": c.p.INTERFACE[role], "model_revision": c.p.REVISIONS[role],
              "max_model_len": 8192, "n_examples": len(rows), "truncated": False, "overlong": [],
              "effective_config_hash": fingerprint(runtime), "rows": [{"sample_id": r["sample_id"], "input_tokens": 50,
                    "reserved_output_tokens": 64 if role == "qwen9b" else 1} for r in rows]}
    write_json(report_path, report)
    write_json(folder/"tokens.json", {"file": str(report_path.relative_to(output)), "sha256": file_hash(report_path),
                                     "runtime_hash": fingerprint(runtime)})
    c._write_chunk(output, folder, role, start, rows, [pred(r, role) for r in rows[start:start+count]])
    return folder


def test_prepare_gold_free_and_exact_source_control_binding(prepared):
    output, profile, rows, settings, _ = prepared
    assert c.context(output)[1] == rows
    changed = copy.deepcopy(rows);changed[0]["gold"] = "entailment"
    with pytest.raises(StudyError, match="gold-free"):
        c.prepare(output, profile, changed, {})
    write_json(output/"runtime-controls/qwen9b/runtime/lock.json", {"substituted": True})
    with pytest.raises(StudyError, match="control changed"):
        c.context(output)


@pytest.mark.parametrize("role", c.ROLES)
def test_chunks_reparse_raw_and_require_stop_before_acceptance(prepared, role):
    output = prepared[0];folder = fake_attempt(prepared, role)
    rows, used = c.chunks(output, role)
    assert len(rows) == 3 and used == [folder]
    with pytest.raises(StudyError, match="verified stop"):
        c.predictions(output, role)
    write_json(folder/"parent-stop.json", stopped(read_json(folder/"plan.json")))
    assert c.predictions(output, role, offline=True) == rows
    # Valid artifact hashes alone do not authorize a changed classification.
    target = folder/"chunks/00000000/rows.json";changed = read_json(target);changed[0]["prediction"] = "neutral";write_json(target, changed)
    receipt = folder/"chunks/00000000/receipt.json";value = read_json(receipt);value["rows_sha256"] = file_hash(target);write_json(receipt, value)
    with pytest.raises(StudyError, match="raw response"):
        c.chunks(output, role)


def test_resume_chunks_are_contiguous_and_verified_tokens_once_per_attempt(prepared, monkeypatch):
    output = prepared[0];folder = fake_attempt(prepared, count=1)
    c._write_chunk(output, folder, "qwen9b", 1, prepared[2], [pred(r) for r in prepared[2][1:]])
    calls = [];original = c.p.verify_tokens
    monkeypatch.setattr(c.p, "verify_tokens", lambda *args: (calls.append(1), original(*args))[1])
    assert len(c.chunks(output, "qwen9b")[0]) == 3
    assert calls == [1]
    fake_attempt(prepared, start=0, count=1, name="0002")
    with pytest.raises(StudyError, match="overlap"):
        c.chunks(output, "qwen9b")


def test_late_or_retried_response_not_committed(prepared):
    rows = prepared[2];value = pred(rows[0]);value.update(request_finished_monotonic=131., observed_request_seconds=121., profile_acceptance="late")
    with pytest.raises(StudyError, match="late"):
        c.validate_predictions(rows[:1], [value], "qwen9b")
    value = pred(rows[0]);value["attempts"].append({"attempt": 2, "status": "ok"})
    with pytest.raises(StudyError, match="retry"):
        c.validate_predictions(rows[:1], [value], "qwen9b")


def test_failure_requires_explicit_retry_and_new_allocation_requires_terminal_proof(prepared, monkeypatch):
    output, _, _, _, allocation = prepared;folder = fake_attempt(prepared)
    write_json(folder/"failure.json", {"error": "artificial"})
    write_json(folder/"parent-stop.json", stopped(read_json(folder/"plan.json")))
    with pytest.raises(StudyError, match="explicitly"):
        c._clear_previous(output, "qwen9b", {"allocation": allocation}, False)
    c._clear_previous(output, "qwen9b", {"allocation": allocation}, True)
    (folder/"parent-stop.json").unlink()
    with pytest.raises(StudyError, match="no confirmed cleanup"):
        c._clear_previous(output, "qwen9b", {"allocation": allocation}, True)
    new = {"hostname": "other", "slurm_job_id": "456"}
    monkeypatch.setattr(c.p, "slurm_job_state", lambda *args: {"terminal": False})
    with pytest.raises(StudyError, match="earlier allocation"):
        c._clear_previous(output, "qwen9b", {"allocation": new}, True)
    monkeypatch.setattr(c.p, "slurm_job_state", lambda *args: {"terminal": True, "job_id": "123"})
    monkeypatch.setattr(c.p.old, "assert_gpu_owned_by", lambda groups: [])
    c._clear_previous(output, "qwen9b", {"allocation": new}, True)
    assert c.predictions(output, "qwen9b")
    assert read_json(folder/"recovered-stop.json")["mode"].startswith("read-only")


def test_completed_role_resumes_without_launch_and_low_allowance_checkpoints(prepared, monkeypatch):
    output, profile, rows, _, allocation = prepared;folder = fake_attempt(prepared)
    write_json(folder/"parent-stop.json", stopped(read_json(folder/"plan.json")))
    monkeypatch.setattr(c.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("completed role relaunched"))
    assert c.collect_role(output, profile, "qwen9b")["status"] == "COMPLETE"
    monkeypatch.setattr(c.p, "session", lambda *args: {"allocation": allocation, "deadline_epoch": time.time()+100})
    assert c.collect_role(output, profile, "startlux4b")["status"] == "CHECKPOINT"


class FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens):
        result = []
        for text in texts:
            ids = list(range(1, int(text)+1))
            if add_special_tokens: ids = [101, *ids, 102]
            result.append(SimpleNamespace(ids=ids, type_ids=[0]*len(ids), attention_mask=[1]*len(ids), overflowing=[]))
        return result


class FakeSession:
    def __init__(self): self.calls = []
    def run(self, outputs, feed):
        self.calls.append(feed["input_ids"].copy())
        ids = feed["input_ids"].astype(np.float32)
        result = np.ones((*ids.shape, 384), dtype=np.float32)
        result[:, :, 0] = ids
        result[:, :, 1] = ids**2
        return [result]


def test_embedding_full_chunk_coverage_and_weighted_raw_means():
    tokenizer, session = FakeTokenizer(), FakeSession()
    matrix, lengths, count = c.field_embeddings(["255", "3"], tokenizer, session)
    assert lengths == [255, 3] and count == 3
    assert max(a.shape[1] for a in session.calls) == 256
    # Independent expected weighted means, including CLS/SEP per chunk, then one normalization.
    means = []
    for block in (list(range(1, 255)), [255]):
        tokens = np.asarray([101, *block, 102], dtype=np.float32)
        m = np.ones(384);m[0] = tokens.mean();m[1] = (tokens**2).mean();means.append(m)
    expected = (254*means[0]+means[1])/255;expected /= np.linalg.norm(expected)
    assert np.allclose(matrix[0], expected, atol=1e-7)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)


def test_embedding_rejects_truncation_or_bad_template():
    tokenizer = FakeTokenizer();original = tokenizer.encode_batch
    def broken(texts, add_special_tokens):
        values = original(texts, add_special_tokens)
        values[0].overflowing = ["lost tokens"]
        return values
    tokenizer.encode_batch = broken
    with pytest.raises(StudyError, match="truncation"):
        c.field_embeddings(["300"], tokenizer, FakeSession())


def test_offline_embedding_receipts_preserve_order_and_unit_shape(prepared):
    output, _, rows, settings, _ = prepared
    identity = {"encoding": c.ENCODING, "inputs_hash": fingerprint(rows), "collection_hash": fingerprint(settings), "model_identity": {"fake": True}}
    write_json(output/"embeddings/identity.json", identity)
    folder = output/"embeddings/chunks/00000000";folder.mkdir(parents=True)
    matrix = np.zeros((3, 1536), dtype=np.float32);matrix[:, 0] = 1
    path = folder/"vectors.npy"
    with path.open("xb") as handle: np.save(handle, matrix, allow_pickle=False)
    receipt = {"identity_hash": fingerprint(identity), "start": 0, "count": 3, "ids": [r["sample_id"] for r in rows],
               "vectors": "vectors.npy", "sha256": file_hash(path), "coverage": {"identity": {"fake": True}, "field_tokens": [3]*6, "chunks": 6}}
    write_json(folder/"receipt.json", receipt)
    ids, got = c.embeddings(output, offline=True)
    assert ids == receipt["ids"] and np.array_equal(got, matrix)
    receipt["ids"].reverse();write_json(folder/"receipt.json", receipt)
    with pytest.raises(StudyError, match="identity changed"):
        c.embeddings(output)


@pytest.mark.parametrize("behavior", ["normal", "late", "checkpoint"])
def test_worker_preserves_requests_and_commits_only_successful_complete_chunks(prepared, monkeypatch, behavior):
    import shutil
    output, _, rows, settings, _ = prepared
    folder = fake_attempt(prepared)
    shutil.rmtree(folder/"chunks")  # This test's artificial attempt has not collected anything yet.
    launch = read_json(folder/"deployment-open.json")
    (folder/"deployment-open.json").unlink()
    plan = read_json(folder/"plan.json")
    plan["deadline_epoch"] = time.time()+(80 if behavior == "checkpoint" else 600)
    write_json(folder/"plan.json", plan)
    monkeypatch.setattr(c, "legacy_environment", lambda *args: nullcontext())
    monkeypatch.setattr(c.p.v3, "_environment", lambda *args: nullcontext())
    monkeypatch.setattr(c.p.frozen, "doctor", lambda *args, **kw: {"gpu": [{"name": "NVIDIA L40S", "memory_mb": 46068}]})
    monkeypatch.setattr(c.p.v3, "_cuda_preflight", lambda *args: {})
    monkeypatch.setattr(c.p, "token_check", lambda *args: read_json(folder/"tokens.json"))
    monkeypatch.setattr(c, "hardware", lambda: {"artificial": True})
    monkeypatch.setattr(c, "ResourceMonitor", lambda *args: nullcontext())
    stops = []
    @contextmanager
    def deployment(output, role, sink):
        sink.update(launch)
        try: yield "http://artificial"
        finally: sink["stop"] = stopped();stops.append(True)
    monkeypatch.setattr(c.p, "deployment", deployment)
    class FakeClient:
        def __init__(self, *args, **kwargs):
            assert kwargs["max_attempts"] == 1
        def predict(self, value):
            result = pred(value);result["latency_ms"] = 0.
            return result
    monkeypatch.setattr(c, "ModelClient", FakeClient)
    if behavior == "late":
        clock = iter([0., 121.]);monkeypatch.setattr(c.time, "monotonic", lambda: next(clock))
        with pytest.raises(StudyError, match="late"):
            c.worker(output, folder)
        assert len(c.read_rows(folder/"requests.jsonl")) == 1
        assert not (folder/"worker-finished.json").exists()
        assert not list((folder/"chunks").glob("*/receipt.json"))
    else:
        c.worker(output, folder)
        value = read_json(folder/"worker-finished.json")
        assert value["status"] == ("COMPLETE" if behavior == "normal" else "CHECKPOINT")
        assert value["end"] == (3 if behavior == "normal" else 0)
        assert len(c.chunks(output, "qwen9b")[0]) == value["end"]
    assert stops == [True]
    assert read_json(folder/"deployment.json")["stop"]["gpu_compute_processes"] == []


def test_parent_watchdog_records_failure_and_owned_cleanup(prepared, monkeypatch):
    output, profile, _, _, _ = prepared
    now = [time.time()]
    monkeypatch.setattr(c.time, "time", lambda: now[0])
    monkeypatch.setattr(c.p, "subreaper", lambda: nullcontext())
    class Process:
        pid = 444
        returncode = None
        def poll(self): return None
        def wait(self, timeout):
            now[0] += 4000
            raise c.subprocess.TimeoutExpired("artificial", timeout)
    proc = Process();calls = []
    monkeypatch.setattr(c.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(c.p, "stop_child", lambda child, since: calls.append(child.pid))
    monkeypatch.setattr(c.p, "recover_stop", lambda *args, **kwargs: stopped())
    with pytest.raises(StudyError, match="absolute allocation watchdog"):
        c.collect_role(output, profile, "qwen9b")
    folder = c.attempts(output, "qwen9b")[0]
    assert calls == [444] and (folder/"failure.json").exists() and (folder/"parent-stop.json").exists()
    assert not (folder/"accepted.json").exists()


@pytest.mark.parametrize("mutation", ["wrong_old_job", "same_new_job", "missing_old_job"])
def test_recovered_stop_rejects_rebound_other_job_proof(prepared, mutation):
    output = prepared[0];folder = fake_attempt(prepared)
    path = folder/"recovered-stop.json"
    value = {"plan_hash": fingerprint(read_json(folder/"plan.json")), "gpu_compute_processes": [], "old_allocation_proof": {"job_id": "123", "terminal": True},
             "new_allocation": {"hostname": "new-node", "slurm_job_id": "456"}}
    write_json(path, value)
    assert c.verify_stop(output, folder) == value
    if mutation == "wrong_old_job": value["old_allocation_proof"]["job_id"] = "999"
    elif mutation == "same_new_job": value["new_allocation"]["slurm_job_id"] = "123"
    else: value["old_allocation_proof"].pop("job_id")
    write_json(path, value)
    # Even a newly recomputed container/file hash cannot repair the semantic
    # binding to this attempt's recorded Slurm job.
    write_json(folder/"artificial-wrapper.json", {"recovered_stop_sha256": file_hash(path)})
    with pytest.raises(StudyError, match="this attempt's old job"):
        c.verify_stop(output, folder)
    with pytest.raises(StudyError, match="this attempt's old job"):
        c.predictions(output, "qwen9b", offline=True)


@pytest.mark.parametrize("kind", ["parent-stop.json", "recovered-stop.json"])
def test_stop_receipt_requires_exact_attempt_plan_hash(prepared, kind):
    output = prepared[0];folder = fake_attempt(prepared)
    plan = read_json(folder/"plan.json")
    if kind == "parent-stop.json": value = stopped(plan)
    else:
        value = {"plan_hash": fingerprint(plan), "gpu_compute_processes": [],
                 "old_allocation_proof": {"job_id": "123", "terminal": True},
                 "new_allocation": {"hostname": "new-node", "slurm_job_id": "456"}}
    write_json(folder/kind, value)
    c.verify_stop(output, folder)
    value["plan_hash"] = fingerprint({**plan, "start": 99})
    write_json(folder/kind, value)
    with pytest.raises(StudyError, match="attempt plan"):
        c.verify_stop(output, folder)


def test_wrong_job_accounting_cannot_create_recovery_receipt(prepared, monkeypatch):
    output = prepared[0];folder = fake_attempt(prepared)
    monkeypatch.setattr(c.p, "slurm_job_state", lambda *args: {"terminal": True, "job_id": "999"})
    monkeypatch.setattr(c.p.old, "assert_gpu_owned_by", lambda *args: pytest.fail("wrong job admitted before GPU check"))
    with pytest.raises(StudyError, match="this interrupted earlier allocation"):
        c._clear_previous(output, "qwen9b", {"allocation": {"hostname": "new", "slurm_job_id": "456"}}, True)
    assert not (folder/"recovered-stop.json").exists()
