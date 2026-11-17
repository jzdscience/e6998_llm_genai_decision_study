"""Whole-CLI simulation of anli-budget-serving-v1 with only provision boundaries faked.

The one-command controller, the real ``runtime.supervise`` and the real worker command
run unchanged. Faked: Slurm queue/accounting, the Linux subreaper/descendant recovery
and native ownership actor (unavailable on macOS), GPU listing, and the StartLux
services, which a ``sitecustomize`` harness replaces inside the worker by loopback HTTP
frontends returning the captured StartLux response structure. Study 4 artifacts are
synthetic and smaller; no real ANLI label is read. Durations are shortened; the
production protocol values are validated separately in test_budget_serving.py.
"""
import contextlib
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest

from decision_study.budget_serving import cli, evaluation, export, inputs, protocol, runtime, study
from decision_study.common import StudyError, file_hash, fingerprint, read_json, read_jsonl, write_json
from decision_study.learnability import data as learn_data
from decision_study.learnability import modeling
from decision_study.service_study.identity import fixed_write

LABELS = list(protocol.LABELS)
CRITERIA = {"entailment": "The premise supports the hypothesis as necessarily true.",
            "neutral": "The premise neither entails nor contradicts the hypothesis.",
            "contradiction": "The premise supports the hypothesis as necessarily false."}
WORKLOAD = 40

HARNESS = r'''
"""Provision-boundary fakes shared by the test controller and the supervised worker."""
import contextlib
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

CONFIG = json.loads(Path(os.environ["BUDGET_FAKE_CONFIG"]).read_text())
LABELS = ("entailment", "neutral", "contradiction")
DOCTOR = {"gpu": [{"index": 0, "name": "NVIDIA L40S", "memory_mb": 46068.0, "driver": "570.86.15", "compute_capability": "8.9"}],
          "gpu_ready": True, "status": "ok", "warnings": [], "selected_gpu_index": 0, "reserved_gpu_count": 1,
          "timestamp": "simulated"}


def install_common(setattr):
    from decision_study.budget_serving import inputs, protocol, runtime
    from decision_study.learnability_profile import study as profile
    setattr(protocol, "CONFIG", Path(CONFIG["protocol_path"]))
    setattr(protocol, "validate", lambda value: value)
    setattr(protocol, "TOKEN_REPORT_SECONDS", CONFIG["token_report_seconds"])
    setattr(inputs.study_data, "DEV_MEMBERSHIP_SHA256", CONFIG["membership_sha256"])
    setattr(profile, "V3_EXAMPLES_SHA256", CONFIG["v3_examples_sha256"])
    setattr(profile, "DEVELOPMENT_SHA256", CONFIG["development_sha256"])
    setattr(runtime.size_runtime, "_verify_model_metadata", lambda path: None)
    for name in ("PINNED_WORKLOAD_FINGERPRINT", "PINNED_CELL_FINGERPRINT", "PINNED_POLICY_FINGERPRINT"):
        setattr(inputs, name, CONFIG["pinned"][name])


def probabilities(role, state):
    digest = hashlib.sha256(f"{role}|{state}".encode()).digest()
    raw = [digest[0] + 1.0, digest[1] + 1.0, digest[2] + 1.0]
    return {label: value / sum(raw) for label, value in zip(LABELS, raw)}


class Frontend(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("x-request-id", self.headers.get("X-Request-ID", ""))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state, role = body["state"], self.server.role
        time.sleep(CONFIG["latency"][role])
        # Some formal 4B calls fail like an overloaded frontend or answer HTTP 200 with a JSON null
        # body; development inputs never fail.
        marker = hashlib.sha256(state.encode()).digest()[3] % 9
        if role == "startlux4b" and "Formal premise" in state and marker == 0:
            return self._send(503, {"detail": "simulated overload"})
        if role == "startlux4b" and "Formal premise" in state and marker == 1:
            return self._send(200, None)
        p = probabilities(role, state)
        choice = max(p, key=p.get)
        self._send(200, {"answers": {"decision": {"choice": choice, "confidence": p[choice], "probabilities": p, "type": "choice"}},
                         "model": "startlux", "usage": {"input_tokens": len(state) // 4, "output_tokens": 0}})


def write_doctor(target):
    from decision_study.common import write_json
    write_json(target / "runtime/doctor.json", DOCTOR)


@contextlib.contextmanager
def deployment(output, condition, *, sink):
    from decision_study import runtime as frozen
    from decision_study.budget_serving import runtime
    from decision_study.common import read_json, utc_now
    plan = read_json(Path(os.environ["DECISION_BUDGET_WORKER_ATTEMPT"]) / "plan.json")
    label = (plan.get("cell") or {}).get("cell_id") or plan.get("deployment")
    sink.update(condition=condition, started_at=utc_now(), roles=[], processes=[], allocation=runtime.session(output)["allocation"])
    servers = []
    try:
        marker = Path(CONFIG["fail_marker"])
        if label == CONFIG.get("fail_once") and not marker.exists():
            marker.write_text(label)
            raise RuntimeError("simulated frontend startup failure on the first attempt")
        for role in runtime.condition_roles(condition):
            target = runtime.control(output, condition, role)
            write_doctor(target)
            lock, installed = read_json(target / "runtime/lock.json"), read_json(target / "runtime/installations.json")
            processes = []
            for spec in runtime.engine_specs(runtime.PROJECT, target, "startlux", "primary", lock, installed, runtime.MAX_NUM_SEQS):
                log = target / "runtime/logs" / f"{spec['name']}-{time.time_ns()}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("simulated service log\nprefix-bidirectional prompt attention ON\n")
                processes.append({"name": spec["name"], "pid": os.getpid(), "command": spec["command"], "log": str(log)})
            port = lock["models"]["startlux"]["port"]
            server = ThreadingHTTPServer(("127.0.0.1", port), Frontend)
            server.daemon_threads, server.role = True, role
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
            sink["processes"].extend(processes)
            sink["roles"].append({"role": role, "gpu_before_start": [], "processes": processes, "startup_failed": False,
                                  "endpoint": f"http://127.0.0.1:{port}", "startup_seconds": 0.01,
                                  "health": {"status": "ok", "prompt_attention": "bidirectional", "defaults": {"think_mode": "never"}},
                                  "runtime": frozen.runtime_config(target, "startlux"), "engine_log": {}})
        sink["gpu_after_start"] = []
        sink["ready_at"] = utc_now()
        yield {item["role"]: item["endpoint"] for item in sink["roles"]}
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        sink["stopped_at"] = utc_now()


def token_check(output, condition, role, examples):
    from decision_study import runtime as frozen
    from decision_study.budget_serving import runtime
    from decision_study.common import file_hash, fingerprint, write_json
    target = runtime.control(output, condition, role)
    write_doctor(target)
    config = frozen.runtime_config(target, "startlux")
    path = target / "runtime" / f"tokens-interactive-startlux-{frozen._hash(examples)[:16]}.json"
    write_json(path, {"status": "ok", "model": "startlux", "model_revision": runtime.REVISIONS[role], "max_model_len": 8192,
                      "n_examples": len(examples), "truncated": False, "overlong": [], "effective_config_hash": fingerprint(config),
                      "rows": [{"sample_id": e["sample_id"], "input_tokens": 40 + len(e["state"]) // 8, "reserved_output_tokens": 1}
                               for e in examples]})
    receipt = {"file": str(path.relative_to(runtime.normal(output))), "sha256": file_hash(path), "runtime_hash": fingerprint(config)}
    runtime.verify_tokens(output, condition, role, examples, receipt)
    return receipt


def install_worker():
    from decision_study.budget_serving import runtime
    from decision_study.upgrade_study import runtime as old
    install_common(setattr)
    runtime.deployment = deployment
    runtime.token_check = token_check
    old.gpu_processes = lambda: []
'''

SITECUSTOMIZE = '''import os
if os.environ.get("BUDGET_FAKE_CONFIG") and os.environ.get("DECISION_BUDGET_WORKER_ATTEMPT"):
    import budget_fakes
    budget_fakes.install_worker()
'''


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def row(i, split, source_split, prefix):
    premise = f"{prefix} premise {i // 2} about an authored scene."
    hypothesis = f"{prefix} hypothesis {i} for that scene."
    return {"sample_id": f"artificial:{split}:{i:04d}", "task": "anli", "split": split, "group_id": f"g:{prefix}:{i // 2}",
            "round": f"R{i % 3 + 1}", "source_split": source_split, "state": f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}",
            "instructions": "Classify the relationship between the premise and hypothesis.", "labels": LABELS, "criteria": CRITERIA,
            "premise_hash": hashlib.sha256(premise.encode()).hexdigest(),
            "input_hash": hashlib.sha256(f"{premise}|{hypothesis}".encode()).hexdigest()}


def head(intercept, slopes, l2):
    return {"kind": "logistic", "mean": [0.0] * 7, "scale": [1.0] * 7, "coefficients": [intercept, *slopes], "l2": l2,
            "n": 18000, "positives": 15000, "iterations": 9, "objective": 0.3, "gradient_inf_norm": 1e-7}


def historical_runtime(root):
    """Accepted V1/size-control controls and pinned snapshot directories (no weights)."""
    v1, size = root / "results/solo-v1", root / "results/aux-size-control-v1-r2"
    defaults = read_json(runtime.PROJECT / "configs/runtime_defaults.json")
    base = {**copy.deepcopy(defaults), "lock_hash": fingerprint(defaults), "defaults_hash": fingerprint(defaults),
            "created_at": "historical", "verified_sources": []}
    name = "vllm-" + defaults["packages"]["vllm_primary"]
    python = v1 / "runtime/work/envs" / name / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("stub interpreter")
    env = {"python": str(python), "freeze_sha256": "a" * 64}
    models = {}
    for role, work in [("startlux4b", v1 / "runtime/work"), ("startlux9b", size / "runtime-work")]:
        snapshot = work / "cache/huggingface/hub/models--pinned/snapshots" / runtime.REVISIONS[role]
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"fixture")
        models[role] = {"model_path": str(snapshot), "revision": runtime.REVISIONS[role]}
    write_json(v1 / "runtime/lock.json", base)
    write_json(v1 / "runtime/installations.json", {"lock_hash": base["lock_hash"], "environments": {name: env},
                                                   "models": {"startlux": models["startlux4b"]}})
    nine = copy.deepcopy(base)
    nine["models"]["startlux"].update(repo=runtime.REPOS["startlux9b"], revision=runtime.REVISIONS["startlux9b"], native_temperature=1.1014)
    nine_hash = fingerprint({k: nine[k] for k in defaults})
    nine.update(lock_hash=nine_hash, defaults_hash=nine_hash, **{"size-control_source": {"parent_lock_hash": base["lock_hash"]}})
    write_json(size / "runtime-controls/startlux9b/runtime/lock.json", nine)
    write_json(size / "runtime-controls/startlux9b/runtime/installations.json",
               {"lock_hash": nine_hash, "environments": {name: env}, "models": {"startlux": models["startlux9b"]}})
    write_json(size / "runtime/binding.json", {"work_dir": str(size / "runtime-work")})


def synthetic_study4(source):
    workload = [row(i, "eligible_dev", "dev", "Formal") for i in range(WORKLOAD)]
    other = [row(i, "fitting", "train", "Fitting") for i in range(6)]
    jsonl(source / "data/examples.jsonl", other + workload)
    membership = learn_data.sha(learn_data.jsonl(learn_data.membership(workload)))
    fixed_write(source / "data/manifest.json", {"eligible_dev_membership_sha256": membership, "dev_labels_used": False,
                "files": {"data/examples.jsonl": {"sha256": file_hash(source / "data/examples.jsonl"), "bytes": 1}}})
    candidates = [{"l2": l2, "local_head": head(-1.0, [4.0, 0, 0, 0, 0, 0, 0], l2), "fallback_head": head(0.6, [0.0] * 7, l2),
                   "selection": {"correct": correct}} for l2, correct in ((1.0, 700), (0.1, 705), (0.01, 710))]
    cell = {"fallback": "startlux9b", "fit_size": 18000, "arm": "front", "usable_fit_count": 18000, "candidates": candidates,
            "selected_index": 2, "deployment_threshold": {"threshold": 0.05, "deferred": 87, "net_correct": 5}}
    models = {"recipe": modeling.RECIPE, "evaluation_gold_used": False, "fit_sizes": list(modeling.FIT_SIZES),
              "cells": {"startlux9b/n18000/front": cell}}
    fixed_write(source / "fitting/models.json", models)
    scores = [0.1 if i % 4 == 0 else -0.1 for i in range(WORKLOAD)]
    fixed_write(source / "fitting/scores-eligible_dev.json", {
        "sample_ids": [r["sample_id"] for r in workload], "evaluation_gold_used": False, "models_sha256": fingerprint(models),
        "front_failed": [False] * WORKLOAD, "cells": {"startlux9b/n18000/front": {"scores": scores}}})
    fixed_write(source / "fitting/complete.json", {"models_sha256": file_hash(source / "fitting/models.json"),
                "scores": {"eligible_dev": file_hash(source / "fitting/scores-eligible_dev.json")},
                "evaluation_gold_used": False, "recipe": modeling.RECIPE})
    fixed_write(source / "freeze.json", {"models_and_scores_frozen_before_evaluation_gold": True, "files": {
        name: file_hash(source / name) for name in ("data/examples.jsonl", "data/manifest.json", "fitting/models.json",
                                                    "fitting/complete.json", "fitting/scores-eligible_dev.json")}})
    gold = {r["sample_id"]: LABELS[hashlib.sha256(r["sample_id"].encode()).digest()[0] % 3] for r in workload}
    gold.update({f"artificial:train_holdout:{i}": "neutral" for i in range(3)})
    fixed_write(source / "sealed/evaluation_gold.json", gold)
    fixed_write(source / "evaluation/results.json", {"simulated": True})
    files = {name: file_hash(source / name) for name in protocol.SOURCE_KEYS}
    picked = candidates[2]
    policy = {"cell": "startlux9b/n18000/front", "selected_index": 2, "l2": 0.01, "threshold": 0.05,
              "local_head": picked["local_head"], "fallback_head": picked["fallback_head"]}
    pinned = {"PINNED_WORKLOAD_FINGERPRINT": fingerprint(sorted(workload, key=lambda r: r["sample_id"])),
              "PINNED_CELL_FINGERPRINT": fingerprint(read_json(source / "fitting/models.json")["cells"]["startlux9b/n18000/front"]),
              "PINNED_POLICY_FINGERPRINT": fingerprint(policy)}
    return {"files": files, "membership": membership, "deferred": sum(s >= 0.05 for s in scores), "gold_count": len(gold),
            "premise_groups": len({r["premise_hash"] for r in workload}), "pinned": pinned}


def synthetic_v3(v3):
    development = [row(i, "development", "train", "Development") for i in range(90)]
    jsonl(v3 / "data/examples.jsonl", development + [row(i, "holdout", "test", "Holdout") for i in range(4)])
    return {"sha256": file_hash(v3 / "data/examples.jsonl"), "fingerprint": fingerprint(sorted(development, key=lambda r: r["sample_id"]))}


def shortened_protocol(source, development):
    p = json.loads(protocol.CONFIG.read_text())
    p.update(rates_per_second=[10, 40], repeats=1, cell_count=10, duration_seconds=1.0, drain_seconds=2,
             request_timeout_seconds=2, startup_timeout_seconds=20, watchdog_grace_seconds=1, source_files=source["files"])
    p["warmup"]["count"] = 3
    p["preflight"]["concurrent"].update(duration_seconds=1.0, rate_per_second=40)  # dense enough for role overlap
    p["preflight"]["policy_smoke"]["duration_seconds"] = 1.0
    p["workload"].update(count=WORKLOAD, premise_groups=source["premise_groups"])
    p["policy"].update(threshold=0.05, offline_regression={"selection_deferred": 87, "selection_count": 900,
                       "eligible_dev_threshold_deferred": source["deferred"], "eligible_dev_count": WORKLOAD})
    p["gold"].update(sha256=source["files"]["sealed/evaluation_gold.json"], source_count=source["gold_count"])
    p["development"].update(source_sha256=development["sha256"], rows_fingerprint=development["fingerprint"])
    return p


def fake_recover(since, *, check_gpu=True, max_seconds=60, term_seconds=10):
    return {"observed_owned_processes": [], "signals": [], "gpu_checked": check_gpu, "gpu_compute_processes": [] if check_gpu else None,
            "ownership": "current descendants of sole-worker Linux subreaper; stable kernel descriptors (simulated on this host)",
            "seconds": 0.0}


def fake_selftest(output):
    """Writes the runtime's receipt layout; the real verify_native_selftest checks it."""
    binding, session = runtime.validate_controls(output), runtime.session(output)
    folder = runtime.safe(output, f"ownership-selftest/{time.time_ns()}")
    folder.mkdir(parents=True)
    values = {"actor-ready.json": {"ready": True}, "decoy.json": {"pid": 1, "created_epoch": time.time()},
              "actor-result.json": {"status": "PASS", "decoy_pid": 1, "decoy_survived": True, "simulated": True},
              "outer-stop.json": {"signals": [], "simulated": True}}
    for name, value in values.items():
        fixed_write(folder / name, value)
    (folder / "actor-cleanup.jsonl").write_text('{"simulated":true}\n')
    names = ("actor-ready.json", "decoy.json", "actor-result.json", "actor-cleanup.jsonl", "outer-stop.json")
    receipt = {"status": "PASS", "binding_hash": fingerprint(binding), "allocation": session["allocation"],
               "files": {n: file_hash(folder / n) for n in names}, "folder": str(folder.relative_to(runtime.normal(output)))}
    fixed_write(folder / "receipt.json", receipt)
    pointer = {"file": str((folder / "receipt.json").relative_to(runtime.normal(output))), "sha256": file_hash(folder / "receipt.json")}
    write_json(runtime.safe(output, "runtime/ownership-selftest.json"), pointer)
    return pointer


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    historical_runtime(root)
    source = synthetic_study4(root / "results/aux-anli-learnability-v1-r2")
    development = synthetic_v3(root / "results/solo-v3")
    p = shortened_protocol(source, development)
    write_json(tmp_path / "protocol.json", p)
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "budget_fakes.py").write_text(HARNESS)
    (harness / "sitecustomize.py").write_text(SITECUSTOMIZE)
    order = study.planned_order(p)
    config = {"protocol_path": str(tmp_path / "protocol.json"), "token_report_seconds": 5, "membership_sha256": source["membership"],
              "v3_examples_sha256": development["sha256"], "development_sha256": development["fingerprint"],
              "latency": {"startlux4b": 0.01, "startlux9b": 0.02}, "fail_once": order[1]["cell_id"], "pinned": source["pinned"],
              "fail_marker": str(tmp_path / "failed-once.marker")}
    write_json(tmp_path / "fake-config.json", config)
    monkeypatch.setenv("BUDGET_FAKE_CONFIG", str(tmp_path / "fake-config.json"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(harness), str(protocol.PROJECT)]))
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    for name in ("DECISION_STUDY_DEADLINE_EPOCH", "DECISION_BUDGET_WORKER_OUTPUT", "DECISION_BUDGET_WORKER_ATTEMPT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.syspath_prepend(str(harness))
    sys.modules.pop("budget_fakes", None)
    importlib.import_module("budget_fakes").install_common(monkeypatch.setattr)
    control = {"hours": 2.0, "ended": set()}
    monkeypatch.setattr(runtime, "slurm_hours_left", lambda: control["hours"])
    monkeypatch.setattr(runtime, "slurm_job_state", lambda job: {"job_id": str(job), "terminal": str(job) in control["ended"],
                                                                "state": "COMPLETED" if str(job) in control["ended"] else "RUNNING"})
    monkeypatch.setattr(runtime.life, "subreaper", lambda: contextlib.nullcontext())
    monkeypatch.setattr(runtime, "_recover_descendants", fake_recover)
    monkeypatch.setattr(runtime, "native_selftest", fake_selftest)
    monkeypatch.setattr(runtime.old, "assert_gpu_owned_by", lambda *a, **k: [])
    output = root / "results/aux-anli-budget-serving-v1"
    gold_reads, real_gold = [], inputs.load_gold

    def load_gold(*args):
        gold_reads.append(study.path(output, "freeze.json").exists())
        return real_gold(*args)
    monkeypatch.setattr(inputs, "load_gold", load_gold)
    return SimpleNamespace(root=root, output=output, p=p, order=order, control=control, gold_reads=gold_reads, tmp=tmp_path,
                           monkeypatch=monkeypatch)


def start(world, *extra):
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        return cli.main(["start", "--runtime-root", str(world.root), "--output-dir", str(world.output), *extra])
    finally:
        for signum, handler in saved.items():
            signal.signal(signum, handler)


def attempts(world, cell_id):
    return sorted((world.output / "cells" / cell_id).glob("attempt-*"))


def extract(bundle_line, target):
    path = Path(bundle_line.split(": ", 1)[1].strip())
    if path.name.endswith(".parts.json"):
        path = export.join(path, target.with_suffix(".zip"))
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    return target


def test_one_command_fails_retries_checkpoints_and_resumes_on_a_genuinely_new_job(world, capsys):
    order, output = world.order, world.output
    failing, checkpoint_at = order[1]["cell_id"], order[5]["cell_id"]

    # 1. Technical failure: preserved evidence, diagnostic bundle, stop.
    assert start(world) == 2
    out = capsys.readouterr()
    assert "DIAGNOSTIC BUNDLE:" in out.out and "BLOCKED" in out.err
    assert read_json(output / "preflight/tokens.json")["status"] == "PASS"
    assert all(read_json(output / f"preflight/{d}.json")["status"] == "PASS" for d in protocol.DEPLOYMENTS)
    assert (output / f"cells/{order[0]['cell_id']}/accepted.json").exists()
    [failed] = attempts(world, failing)
    assert (failed / "supervision-failure.json").exists() and (failed / "parent-stop.json").exists()
    assert list((failed).glob("worker-error-*.json")) and not (failed / "supervised.json").exists()

    # 2. The same command never silently retries a failed attempt.
    assert start(world) == 2
    assert "--retry-failed" in capsys.readouterr().err
    assert len(attempts(world, failing)) == 1

    # 3. Explicit retry; the allocation then runs short before a later cell: checkpoint, no attempt folder.
    real_stage = study.cell_stage

    def shortened(ctx, cell, **kwargs):
        if cell["cell_id"] == checkpoint_at:
            session = read_json(output / "runtime/session.json")
            write_json(output / "runtime/session.json", {**session, "deadline_epoch": time.time() + 300})
        return real_stage(ctx, cell, **kwargs)
    world.monkeypatch.setattr(study, "cell_stage", shortened)
    assert start(world, "--retry-failed") == 0
    out = capsys.readouterr().out
    assert "STUDY_CHECKPOINT" in out and "CHECKPOINT BUNDLE:" in out
    assert [a.name for a in attempts(world, failing)] == ["attempt-0001", "attempt-0002"]
    accepted = read_json(output / f"cells/{failing}/accepted.json")
    assert accepted["attempt"].endswith("attempt-0002") and accepted["prior_attempts"] == [{"attempt": "attempt-0001", "state": "failed"}]
    assert attempts(world, checkpoint_at) == []
    world.monkeypatch.setattr(study, "cell_stage", real_stage)

    # 4. An interrupted attempt in the same allocation is never signalled or replaced.
    session = read_json(output / "runtime/session.json")
    planted = output / f"cells/{checkpoint_at}/attempt-0001"
    planted.mkdir(parents=True)
    write_json(planted / "supervision.json", {"command": ["python"], "allocation": session["allocation"],
                                              "binding_hash": session["binding_hash"], "started_epoch": time.time(),
                                              "deadline_epoch": time.time() + 60})
    assert start(world) == 2
    assert "this allocation" in capsys.readouterr().err
    assert not (planted / "recovered-stop.json").exists()

    # 5. A genuinely different Slurm job with terminal proof for the old one resumes automatically.
    world.control["ended"].add("111")
    world.monkeypatch.setenv("SLURM_JOB_ID", "222")
    assert start(world) == 0
    out = capsys.readouterr().out
    assert "STUDY_COMPLETE" in out and "RESULT BUNDLE:" in out
    bundle_line = next(line for line in out.splitlines() if line.startswith("RESULT BUNDLE:"))
    transfer = read_json(Path(next(line for line in out.splitlines() if line.startswith("TRANSFER MANIFEST:")).split(": ", 1)[1]))
    bundle_path = Path(bundle_line.split(": ", 1)[1])
    assert transfer["main_result"]["files"][0] == {"name": bundle_path.name, "bytes": bundle_path.stat().st_size,
                                                   "sha256": file_hash(bundle_path)}
    assert transfer["optional_supplement"]["directory"] == "exports/learnability-embeddings"
    # Genuine HTTP 200 JSON-null answers were accepted as counted failed calls in the formal cells.
    nulls = 0
    for cell in order:
        attempt = output / read_json(output / f"cells/{cell['cell_id']}/accepted.json")["attempt"]
        for row in read_jsonl(attempt / "traces.jsonl"):
            for stage in (row.get("result") or {}).get("stages", []):
                nulls += (stage["error"] or {}).get("phase") == "answer" and stage["raw_response"] is None
    assert nulls > 0
    assert "INCOMPLETE" in out  # the synthetic Study 4 root has no accepted embedding arrays; the study still completes
    recovered = read_json(planted / "recovered-stop.json")
    assert recovered["old_allocation_proof"]["terminal"] and recovered["new_allocation"]["slurm_job_id"] == "222"
    resumed = read_json(output / f"cells/{checkpoint_at}/accepted.json")
    assert resumed["prior_attempts"] == [{"attempt": "attempt-0001", "state": "interrupted"}]
    assert list((output / "runtime/allocation-checks").glob("*.json")) and len(list((output / "ownership-selftest").iterdir())) == 2
    assert {read_json(output / f"cells/{c['cell_id']}/accepted.json")["allocation"]["slurm_job_id"] for c in order} == {"111", "222"}

    # Gold was parsed only after the freeze of every trace, ledger and verified stop.
    freeze = read_json(output / "freeze.json")
    assert world.gold_reads and all(world.gold_reads)
    assert read_json(output / "evaluation/access.json")["freeze_sha256"] == file_hash(output / "freeze.json")
    assert freeze["gold_labels_parsed"] is False and len(freeze["cells"]) == 10
    results = read_json(output / "evaluation/results.json")
    assert len(results["cells"]) == 10 and len(results["summaries"]) == 10 and len(results["paired_differences"]) == 20
    for cell in results["cells"]:
        metrics = cell["metrics"]
        assert metrics["offered"] == metrics["admitted"] + metrics["client_rejected"]
        assert metrics["successful"] + metrics["errors"] + metrics["deadlines"] == metrics["admitted"]
        if cell["condition"] in protocol.CAPPED:
            assert cell["ledger"]["reserved"] <= cell["ledger"]["offered_received"] // 10
            assert metrics["routing"]["budget_granted"] == cell["ledger"]["reserved"]
            assert metrics["model_attempts"]["startlux9b"] <= cell["ledger"]["reserved"]
        if cell["condition"] in protocol.PAIR_CONDITIONS:
            assert metrics["front_fallback_pairs"]["applicable"] <= metrics["model_attempts"]["startlux9b"]
        if cell["condition"] == "threshold":
            shadow = cell["counterfactual_cap"]
            assert shadow["would_grant"] <= shadow["offers"] // 10
            assert shadow["candidates_reaching_fallback"] == metrics["routing"]["candidates"]  # uncapped: every candidate
    assert read_json(output / "audit.json")["status"] == "PASS"
    assert (output / "REPORT.md").read_text().startswith("# StartLux-4B")
    assert list((output / "supplements").glob("embedding-capture-*.json"))

    # Gold affects only evaluation: different labels change metrics, never the frozen routing evidence.
    ctx = study.context(output)
    flipped = {sid: LABELS[(LABELS.index(label) + 1) % 3] for sid, label in read_json(output / "evaluation/gold.json").items()}
    other = evaluation.compute(ctx, flipped)
    assert other["cells"] != results["cells"]
    study.verify_freeze(ctx, deep=False)

    # 6. Rerunning a complete study re-audits without new GPU attempts.
    before = sorted(str(p) for p in output.glob("cells/*/attempt-*"))
    assert start(world) == 0
    assert "STUDY_COMPLETE" in capsys.readouterr().out
    assert sorted(str(p) for p in output.glob("cells/*/attempt-*")) == before

    # 7. The returned compact bundle audits offline from raw evidence; forged evidence fails even with rehashed receipts.
    extracted = extract(bundle_line, world.tmp / "returned")
    assert not (extracted / "runtime-work").exists()
    assert cli.main(["audit", "--output-dir", str(extracted)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert cli.main(["audit", "--output-dir", str(extracted),
                     "--gold-source", str(world.root / "results/aux-anli-learnability-v1-r2/sealed/evaluation_gold.json")]) == 0
    capsys.readouterr()
    real_identity = protocol.source_identity

    def other_checkout():
        value = real_identity()
        value["files"]["decision_study/budget_serving/router.py"] = "0" * 64
        return value
    world.monkeypatch.setattr(protocol, "source_identity", other_checkout)
    with pytest.raises(StudyError, match="released checkout"):
        study.context(extracted, offline=True)
    world.monkeypatch.setattr(protocol, "source_identity", real_identity)
    forged_gold(extracted, capsys)
    forged_cell(world, extract(bundle_line, world.tmp / "returned-1"), capsys)
    forged_preflight(world, world.tmp / "returned-2", bundle_line, capsys)
    forged_freeze(world, bundle_line, capsys)
    forged_history(world, bundle_line, failing, capsys)


def forged_history(world, bundle_line, failing, capsys):
    """Deleting the failed attempt and dropping it from the freeze hides no accounting."""
    import shutil
    output = extract(bundle_line, world.tmp / "returned-history")
    failed = output / f"cells/{failing}/attempt-0001"
    assert (failed / "supervision-failure.json").exists()
    shutil.rmtree(failed)
    prefix = str(failed.relative_to(output)) + "/"
    _rewrite_freeze(output, lambda frozen: frozen.update(files={k: v for k, v in frozen["files"].items() if not k.startswith(prefix)}))
    assert cli.main(["audit", "--output-dir", str(output)]) == 2
    assert "Stage attempt history differs" in capsys.readouterr().err


def _rewrite_freeze(output, change):
    """Change the freeze record, then rewrite its self hash and every evaluation link to it."""
    frozen = read_json(output / "freeze.json")
    change(frozen)
    frozen["self_hash"] = fingerprint({k: v for k, v in frozen.items() if k != "self_hash"})
    write_json(output / "freeze.json", frozen)
    access = read_json(output / "evaluation/access.json")
    access["freeze_sha256"] = file_hash(output / "freeze.json")
    write_json(output / "evaluation/access.json", access)
    complete = read_json(output / "evaluation/complete.json")
    complete.update(freeze_sha256=file_hash(output / "freeze.json"), access_sha256=file_hash(output / "evaluation/access.json"))
    write_json(output / "evaluation/complete.json", complete)


def forged_freeze(world, bundle_line, capsys):
    """Full offline audits of rewritten freezes; every underlying receipt and trace is left untouched."""
    finder = extract(bundle_line, world.tmp / "returned-finder")
    for folder in (finder, finder / "cells", next((finder / "cells").iterdir())):
        (folder / ".DS_Store").write_bytes(b"Finder metadata")  # never evidence; must not break an audit
    assert cli.main(["audit", "--output-dir", str(finder)]) == 0
    assert json.loads(capsys.readouterr().out)["frozen_files"] == len(read_json(finder / "freeze.json")["files"]) > 300

    def joint(frozen):  # the independent protocol reviewer's reproduction
        frozen["files"] = {}
        frozen["tokens_sha256"] = frozen["source_files_fingerprint"] = "0" * 64
        frozen["preflight_sha256"] = {k: "0" * 64 for k in frozen["preflight_sha256"]}
        frozen["cells"] = {k: "0" * 64 for k in frozen["cells"]}

    def bindings_only(frozen):
        frozen["cells"][sorted(frozen["cells"])[0]] = "0" * 64

    def source_only(frozen):
        frozen["source_files_fingerprint"] = "0" * 64

    def inventory_only(frozen):
        del frozen["files"][next(name for name in sorted(frozen["files"]) if name.endswith("/traces.jsonl"))]
    cases = [("joint", joint, "Freeze bindings differ"), ("bindings", bindings_only, "Freeze bindings differ"),
             ("source", source_only, "Freeze bindings differ"), ("inventory", inventory_only, "Frozen evidence inventory differs")]
    for name, change, reason in cases:
        output = extract(bundle_line, world.tmp / f"returned-freeze-{name}")
        _rewrite_freeze(output, change)
        assert cli.main(["audit", "--output-dir", str(output)]) == 2, name
        assert reason in capsys.readouterr().err, name


def forged_gold(extracted, capsys):
    """A relabelled evaluated subset with every evaluation hash rewritten still fails the pinned gold."""
    gold = read_json(extracted / "evaluation/gold.json")
    first = sorted(gold)[0]
    gold[first] = LABELS[(LABELS.index(gold[first]) + 1) % 3]
    write_json(extracted / "evaluation/gold.json", gold)
    complete = read_json(extracted / "evaluation/complete.json")
    complete["gold_sha256"] = file_hash(extracted / "evaluation/gold.json")
    write_json(extracted / "evaluation/complete.json", complete)
    assert cli.main(["audit", "--output-dir", str(extracted)]) == 2
    assert "exact eligible-dev subset" in capsys.readouterr().err


def _rehash_cell(output, cell_id, changed):
    """An adversary rewrites every hash and receipt that covers the changed files."""
    receipt_path = output / f"cells/{cell_id}/accepted.json"
    receipt = read_json(receipt_path)
    folder = output / receipt["attempt"]
    complete = read_json(folder / "complete.json")
    for name in changed:
        complete["files"][name] = file_hash(folder / name)
    write_json(folder / "complete.json", complete)
    receipt.update(evidence=complete["files"], complete_sha256=file_hash(folder / "complete.json"))
    write_json(receipt_path, receipt)
    _rehash_freeze(output, [str((folder / name).relative_to(output)) for name in changed] +
                   [str((folder / "complete.json").relative_to(output)), str(receipt_path.relative_to(output))])


def _rehash_freeze(output, names):
    freeze = read_json(output / "freeze.json")
    for name in names:
        freeze["files"][name] = file_hash(output / name)
    for cell_id in freeze["cells"]:
        freeze["cells"][cell_id] = file_hash(output / f"cells/{cell_id}/accepted.json")
    freeze["preflight_sha256"] = {d: file_hash(output / f"preflight/{d}.json") for d in freeze["preflight_sha256"]}
    freeze["self_hash"] = fingerprint({k: v for k, v in freeze.items() if k != "self_hash"})
    write_json(output / "freeze.json", freeze)
    for name in ("evaluation/access.json", "evaluation/complete.json"):
        value = read_json(output / name)
        value["freeze_sha256"] = file_hash(output / "freeze.json")
        if name.endswith("complete.json"):
            value["access_sha256"] = file_hash(output / "evaluation/access.json")
        write_json(output / name, value)


def forged_cell(world, extracted, capsys):
    cell = next(c for c in world.order if c["condition"] == "budget" and c["rate"] == 40)
    folder = extracted / read_json(extracted / f"cells/{cell['cell_id']}/accepted.json")["attempt"]
    rows = read_jsonl(folder / "traces.jsonl")
    target = next((r for r in rows if r["result"] and (r["result"]["route"] or {}).get("budget_denied")), None)
    assert target is not None, "the simulated budget cell must deny at least one candidate"
    target["result"]["route"].update(budget_granted=True, budget_denied=False)  # hide a denial
    (folder / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    _rehash_cell(extracted, cell["cell_id"], ["traces.jsonl"])
    assert cli.main(["audit", "--output-dir", str(extracted)]) == 2
    assert "Route permit differs from the ledger replay" in capsys.readouterr().err  # semantic, not a hash mismatch
    ctx = study.context(extracted, offline=True)
    with pytest.raises(StudyError, match="Route permit differs from the ledger replay"):
        study.accepted_cell(ctx, cell, offline=True)


def forged_preflight(world, target, bundle_line, capsys):
    extracted = extract(bundle_line, target)
    receipt = read_json(extracted / "preflight/pair.json")
    folder = extracted / receipt["attempt"]
    events = read_jsonl(folder / "smoke-budget/ledger.jsonl")
    index = next(i for i, e in enumerate(events) if e["type"] in {"reserve", "deny"})
    del events[index]  # drop one permit decision and renumber the journal
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
    (folder / "smoke-budget/ledger.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    complete = read_json(folder / "complete.json")
    complete["files"]["smoke-budget/ledger.jsonl"] = file_hash(folder / "smoke-budget/ledger.jsonl")
    write_json(folder / "complete.json", complete)
    receipt.update(evidence=complete["files"], complete_sha256=file_hash(folder / "complete.json"))
    write_json(extracted / "preflight/pair.json", receipt)
    _rehash_freeze(extracted, [str((folder / "smoke-budget/ledger.jsonl").relative_to(extracted)),
                               str((folder / "complete.json").relative_to(extracted)), "preflight/pair.json"])
    ctx = study.context(extracted, offline=True)
    with pytest.raises(StudyError, match="Ledger"):
        study.accepted_preflight(ctx, "pair", offline=True)
    assert cli.main(["audit", "--output-dir", str(extracted)]) == 2
    assert "Ledger" in capsys.readouterr().err


def test_output_identity_and_source_changes_block_before_any_write(world, capsys, monkeypatch):
    unrelated = world.root / "results/aux-anli-budget-serving-v1-r2"
    unrelated.mkdir(parents=True)
    (unrelated / "notes.txt").write_text("someone else's file")
    assert cli.main(["start", "--runtime-root", str(world.root), "--output-dir", str(unrelated)]) == 2
    assert "unrelated files" in capsys.readouterr().err and sorted(p.name for p in unrelated.iterdir()) == ["notes.txt"]
    elsewhere = world.root / "results/other-study"
    assert cli.main(["start", "--runtime-root", str(world.root), "--output-dir", str(elsewhere)]) == 2
    assert not elsewhere.exists()
    ctx = study.prepare(world.output, world.root)
    assert study.path(world.output, "study.json").exists() and ctx.settings_sha256
    real = protocol.source_identity

    def changed():
        value = real()
        value["files"]["decision_study/budget_serving/serve.py"] = "0" * 64
        return value
    monkeypatch.setattr(protocol, "source_identity", changed)
    with pytest.raises(StudyError, match="fresh output"):
        study.prepare(world.output, world.root)
    monkeypatch.setattr(protocol, "source_identity", real)
    monkeypatch.setenv("DECISION_STUDY_DEADLINE_EPOCH", str(time.time() + 99999))
    assert start(world) == 2
    assert "ambient" in capsys.readouterr().err
