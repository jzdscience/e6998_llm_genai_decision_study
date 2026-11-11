"""End-to-end CLI rehearsal of the direct-matrix node sequence on the real accepted evidence.

Run from a clean committed checkout (an ephemeral test repository; no remote):

    python -I tests/direct_matrix_rehearsal.py --repo REPO --evidence EVIDENCE_ROOT --work WORK_DIR

Production code runs unmodified for preparation, source and protocol identity,
runtime initialization and binding, session and allocation checks, preflight,
collection children, PredictionStore, completion, freeze, evaluation, export and
the offline audit. Faked, and only these: starting the service commands (V1's own
command builders run; each service is a dummy process group), nvidia-smi, the
hardware doctor, the CUDA toolkit check, the tokenizer run (its V1-format report is
written in place), reads of node-only snapshot metadata, the legacy cwd scope for
node environments, and the model servers (a local HTTP server). Three faults are
injected through the child-process command: a SIGKILL after the 53rd formal row of
startlux9b-squad2 (50 rows in shards, 3 only in the fsynced journal), a crash after
the last row of kev4b-squad2, and a SIGKILL after kev4b-anli's completion receipt.
Fake labels are arbitrary; nothing here is an experimental result.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def _probabilities(labels, index):
    other = round(0.1 / (len(labels) - 1), 4)
    values = [other] * len(labels)
    values[index] = round(1 - other * (len(labels) - 1), 4)
    return dict(zip(labels, values))


def _pick(text, labels):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % len(labels)


class Models(BaseHTTPRequestHandler):
    failing_states = set()
    requests = 0
    decision_counts = {}  # (startlux|kev, sha256(state)) -> number of SystemOne requests
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests += 1
        if self.path == "/v1/systemone":
            key = ("startlux" if "think_mode" in body else "kev", hashlib.sha256(body["state"].encode()).hexdigest())
            with self.lock:
                self.decision_counts[key] = self.decision_counts.get(key, 0) + 1
            question = body["questions"]["decision"]
            labels = list(question["criteria"])
            index = _pick(body["state"] + "|" + "|".join(labels), labels)
            probabilities = _probabilities(labels, index)
            self._send(200, {"answers": {"decision": {"type": "choice", "choice": labels[index],
                                                      "confidence": probabilities[labels[index]],
                                                      "probabilities": probabilities}},
                             "model": "rehearsal-fake", "usage": {"input_tokens": 10, "output_tokens": 1}})
        elif self.path == "/v1/chat/completions":
            content = body["messages"][-1]["content"]
            if any(state in content for state in self.failing_states):
                self._send(500, {"error": "injected terminal failure"})
                return
            labels = body["structured_outputs"]["json"]["properties"]["label"]["enum"]
            label = labels[_pick(content, labels)]
            self._send(200, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"label": label})}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        else:
            self._send(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    # Resolve every path before changing directory: outputs must stay outside the clean checkout.
    repo, work, evidence = args.repo.resolve(), args.work.resolve(), args.evidence.resolve()
    if work == repo or repo in work.parents:
        raise SystemExit("--work must be outside the checkout")
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True

    from decision_study import runtime as frozen
    from decision_study.common import read_json
    from decision_study.direct_matrix import audit, cli, runtime, study
    from decision_study.size_control import runtime as size_runtime
    from decision_study.upgrade_study import launcher as up_launcher, runtime as up_runtime
    from decision_study.v3 import runtime as v3_runtime

    results = work / "results"
    output = results / "aux-direct-matrix-v1"
    results.mkdir(parents=True, exist_ok=False)
    ev = evidence / "results"
    inputs = ["--v1-dir", str(ev / "returned-experiment-2026-11-02-101054"), "--v2-dir", str(ev / "returned-v2-2026-11-03-002852"),
              "--v3-dir", str(ev / "returned-v3-2026-11-03-152529"),
              "--size-control-dir", str(ev / "returned-size-control-complete-2026-11-06")]

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    launched = []
    faults = {"startlux9b-squad2": "kill-mid-cell", "kev4b-squad2": "crash-before-receipt", "kev4b-anli": "kill-after-receipt"}
    fired, scheduler_calls = [], []

    def lister():
        return [{"pid": p.pid, "used_memory_mib": 1.0} for p in launched if p.poll() is None]

    def start(specs, control, model, extra):
        meta, processes = {"started_at": datetime.now(timezone.utc).isoformat(), "models": {}, "processes": [], **extra}, []
        for spec in specs:
            process = subprocess.Popen(["sleep", "600"], start_new_session=True)
            launched.append(process)
            processes.append(process)
            meta["processes"].append({"name": spec["name"], "pid": process.pid, "command": spec["command"], "log": ""})
            (control / "runtime/last_services.json").write_text(json.dumps(meta))
        meta["models"][model] = {"health": {"status": "ok"}, "config": frozen.runtime_config(control, model)}
        (control / "runtime/last_services.json").write_text(json.dumps(meta))
        return processes

    def stop(processes):
        for process in processes:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)

    @contextlib.contextmanager
    def fake_vllm(control, model, *, max_num_seqs, project_dir=repo):
        control = Path(control).resolve()
        lock, installed = (read_json(control / "runtime" / n) for n in ("lock.json", "installations.json"))
        specs = up_launcher.engine_specs(project_dir, control, model, frozen._resolve_profile(control, model, None),
                                         lock, installed, max_num_seqs)
        processes = start(specs, control, model, {"max_num_seqs": max_num_seqs})
        try:
            yield {model: endpoint}
        finally:
            stop(processes)

    @contextlib.contextmanager
    def fake_v1(project_dir, control, models, profile=None):
        (model,) = models
        control = Path(control).resolve()
        lock, installed = (read_json(control / "runtime" / n) for n in ("lock.json", "installations.json"))
        specs = frozen._service_specs(project_dir, control, model, frozen._resolve_profile(control, model, None), lock, installed)
        processes = start(specs, control, model, {})
        try:
            yield {model: endpoint}
        finally:
            stop(processes)

    def fake_doctor(control, require_gpu=False, check_network=False):
        path = Path(control) / "runtime/doctor.json"
        report = {**read_json(path), "timestamp": datetime.now(timezone.utc).isoformat(), "rehearsal": True}
        path.write_text(json.dumps(report))
        return report

    def fake_tokens(project, control, model, examples, profile=None):
        control = Path(control)
        reserve = {"kev": 0, "startlux": 1, "qwen": 64}[model]
        lock = read_json(control / "runtime/lock.json")
        report = {"status": "ok", "model": model, "model_revision": lock["models"][model]["revision"],
                  "max_model_len": lock["max_model_len"], "n_examples": len(examples), "overlong": [],
                  "renderer": "rehearsal-fake", "truncated": False,
                  "rows": [{"sample_id": e["sample_id"], "input_tokens": 40 + len(e["state"]) // 4,
                            "reserved_output_tokens": reserve,
                            "token_ids_sha256": hashlib.sha256(e["state"].encode()).hexdigest()} for e in examples],
                  "effective_config_hash": frozen._hash(frozen.runtime_config(control, model))}
        (control / "runtime" / f"tokens-interactive-{model}-{frozen._hash(examples)[:16]}.json").write_text(json.dumps(report))
        return report

    def fake_metadata(path):
        return {"config.json": hashlib.sha256(str(path).encode()).hexdigest()}

    def no_scheduler(job_id, runner=None):
        scheduler_calls.append(job_id)
        raise AssertionError("the rehearsal never needs a scheduler query")

    real_command = study._worker_command

    def worker_command(out, folder):
        cell = folder.parent.name
        if folder.name.startswith("attempt-") and cell in faults and cell not in fired:
            fired.append(cell)
            return [sys.executable, str(repo / "tests/direct_matrix_fault_worker.py"), faults[cell],
                    "--root", str(out), "--attempt", str(folder)]
        return real_command(out, folder)

    up_runtime.gpu_processes = lister
    frozen.doctor = fake_doctor
    frozen.managed_services = fake_v1
    frozen.token_preflight_examples = fake_tokens
    v3_runtime._cuda_preflight = lambda control, binding, installed: {}
    runtime.legacy_environment = lambda binding, project: contextlib.nullcontext()
    runtime.managed_service = fake_vllm
    runtime._metadata = fake_metadata
    runtime.slurm_job_state = no_scheduler
    size_runtime._verify_model_metadata = lambda path: None
    study._worker_command = worker_command

    # One squad2 input (the shortest, so never a preflight longest-input check) always fails for Qwen.
    squad2 = [json.loads(line) for line in (ev / "returned-v2-2026-11-03-002852/data/examples.jsonl").read_text().splitlines()
              if line.strip() and '"holdout"' in line]
    failing = min(squad2, key=lambda r: (len(r["state"]), r["sample_id"]))
    Models.failing_states = {failing["state"]}

    steps = []

    def command(name, argv, job="4242"):
        os.environ["SLURM_JOB_ID"] = job
        before = len(launched)
        code = cli.main([*argv, "--output-dir", str(output)])
        error = output / "diagnostics/last_error.json"
        steps.append({"name": name, "argv": argv, "job": job, "code": code, "launched": len(launched) - before,
                      "error": json.loads(error.read_text())["error"] if code and error.exists() else None})
        print(f"STEP {name}: exit {code}", flush=True)
        return code

    def shard_hashes(cell):
        folder = output / "cells" / cell / "predictions"
        return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("shard-*.json"))}

    def durable(cell, model):
        """Rows in shards and rows only in the fsynced journal, read without opening a PredictionStore."""
        from decision_study.storage import load_predictions
        directory = output / "cells" / cell
        sharded = load_predictions(directory, model)
        journal = study.journal_path(directory, model, read_json(directory / f"predictions/{model}/config.json"))
        lines = [json.loads(line) for line in journal.read_bytes().splitlines() if line.strip()] if journal.is_file() else []
        return sharded, lines

    summary = {"output": str(output), "failing_sample": failing["sample_id"]}
    try:
        command("prepare", ["prepare", *inputs])
        command("init", ["init-runtime", "--session-hours-left", "2"])
        command("preflight", ["preflight"])
        command("start-mid-cell-kill", ["start", *inputs, "--session-hours-left", "2"])
        sharded, journal = durable("startlux9b-squad2", "startlux")
        summary["mid_cell"] = {"sharded": len(sharded), "journal": len(journal),
                               "rows": {r["sample_id"]: r for r in sharded + journal},
                               "shards": shard_hashes("startlux9b-squad2"),
                               "attempts": sorted(p.name for p in (output / "cells/startlux9b-squad2").glob("attempt-*"))}
        command("plain-rerun", ["start", *inputs, "--session-hours-left", "2"])
        summary["mid_cell"]["journal_after_plain_rerun"] = len(durable("startlux9b-squad2", "startlux")[1])
        command("other-job-without-ack", ["start", *inputs, "--session-hours-left", "2"], job="4243")
        command("new-allocation-without-resume", ["start", *inputs, "--session-hours-left", "2", "--new-allocation"], job="4243")
        summary["mid_cell"]["journal_after_refusals"] = len(durable("startlux9b-squad2", "startlux")[1])
        command("resume-fault-a", ["start", *inputs, "--session-hours-left", "2", "--resume"], job="4243")
        summary["after_fault_a"] = {"kev4b-squad2": shard_hashes("kev4b-squad2"),
                                    "attempts": sorted(p.name for p in (output / "cells/kev4b-squad2").glob("attempt-*"))}
        command("plain-rerun-a", ["start", *inputs, "--session-hours-left", "2"], job="4243")
        command("resume-fault-b", ["start", *inputs, "--session-hours-left", "2", "--resume"], job="4243")
        command("plain-rerun-b", ["start", *inputs, "--session-hours-left", "2"], job="4243")
        command("resume-complete", ["start", *inputs, "--session-hours-left", "2", "--resume"], job="4243")
        summary["after_complete"] = {"kev4b-squad2": shard_hashes("kev4b-squad2"),
                                     "attempts": sorted(p.name for p in (output / "cells/kev4b-squad2").glob("attempt-*"))}
        final, leftover = durable("startlux9b-squad2", "startlux")
        by_id = {r["sample_id"]: r for r in final}
        old = summary["mid_cell"]["rows"]
        summary["mid_cell"].update(
            final_rows=len(final), final_journal=len(leftover), old_rows_unchanged=all(by_id[s] == r for s, r in old.items()),
            old_shards_unchanged=all(shard_hashes("startlux9b-squad2").get(k) == v for k, v in summary["mid_cell"]["shards"].items()),
            deployments={d: sum(r["deployment_id"] == d for r in final) for d in sorted({r["deployment_id"] for r in final})},
            attempts_final=sorted(p.name for p in (output / "cells/startlux9b-squad2").glob("attempt-*")))
        longest = read_json(output / "preflight/startlux9b.json")
        stage = output / longest["evidence_dir"] / "longest-startlux9b-squad2/plan.json"
        preflight_ids = set(read_json(stage)["sample_ids"])
        formal = [json.loads(line) for line in (output / "data/formal/squad2.jsonl").read_text().splitlines()]
        counts = {e["sample_id"]: Models.decision_counts.get(("startlux", hashlib.sha256(e["state"].encode()).hexdigest()), 0)
                  - (e["sample_id"] in preflight_ids) for e in formal}
        summary["mid_cell"]["formal_requests_per_input"] = sorted(set(counts.values()))
        del summary["mid_cell"]["rows"]
        evaluation_bytes = (output / "evaluation.json").read_bytes()
        command("start-again", ["start", *inputs, "--session-hours-left", "2"], job="4243")
        summary["evaluation_unchanged"] = (output / "evaluation.json").read_bytes() == evaluation_bytes
        command("status", ["status"], job="4243")
        summary["status"] = study.status(output)
        summary["recoveries"] = {cell: sorted(str(p.relative_to(output)) for p in (output / "cells" / cell).glob("attempt-*/recovery.json"))
                                 for cell in faults}
        # Offline audit of the returned bundle: extracted elsewhere, opening nothing outside it and this checkout.
        bundle = output / "exports/direct_matrix_bundle.zip"
        extracted = work / "returned-bundle"
        opened = []

        def hook(event, payload):
            if event == "open" and isinstance(payload[0], (str, bytes, os.PathLike)):
                opened.append(os.fsdecode(payload[0]))

        audit.extract_bundle(bundle, extracted)
        sys.addaudithook(hook)
        report = audit.audit(extracted)
        summary["audit"] = report
        outside = sorted({path for path in opened if not (Path(path).resolve().is_relative_to(extracted)
                                                          or Path(path).resolve().is_relative_to(repo)
                                                          or Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())
                                                          or Path(path).resolve().is_relative_to(Path(sys.base_prefix).resolve()))})
        summary["audit_opened_outside_bundle_and_checkout"] = outside
        summary["audit_opened_evidence_or_node_paths"] = sorted(
            path for path in opened if path.startswith(str(evidence)) or path.startswith("/home/"))
    finally:
        summary.update(steps=steps, faults_fired=fired, scheduler_calls=scheduler_calls, requests=Models.requests,
                       alive_after=lister())
        server.shutdown()
        for process in launched:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
        (work / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
