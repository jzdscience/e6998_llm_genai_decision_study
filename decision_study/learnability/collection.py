"""Resumable, gold-free collection using the accepted development-profile runtime.

Successful immutable request chunks survive a later interruption. They are not a
finished role until every contributing attempt has a verified stop. No historic
PID authorizes a signal. The sole-worker parent uses the profile's pidfd/subreaper
ownership implementation without modifying it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from ..common import StudyError, canonical, file_hash, fingerprint, read_json, utc_now, write_json
from ..adapters import ModelClient, parse_response
from ..service_study.identity import fixed_write
from ..service_study.resources import ResourceMonitor, hardware
from ..runtime_compat import legacy_environment
from ..learnability_profile import study as p
from ..learnability_profile import ownership
from ..cheap_baselines import encoder

CHUNK_SIZE = 128
REQUEST_SECONDS = 120
CLEANUP_SECONDS = 120
ROLES = p.ROLES
SPLITS = {"fitting", "selection", "eligible_dev", "train_holdout"}
COLLECTION_PROTOCOL = {"id": "anli-learnability-collection-v1", "chunk_size": CHUNK_SIZE,
                       "request_acceptance_seconds": REQUEST_SECONDS, "max_attempts": 1,
                       "cleanup_reserve_seconds": CLEANUP_SECONDS, "roles": list(ROLES)}
ENCODING = {"encoder": encoder.ENCODER_SPEC, "content_tokens_per_chunk": 254,
            "pair": "l2(concat(u,v,abs(u-v),u*v))", "dimension": 1536,
            "chunk_pooling": "content-count-weighted raw attention-mask means, then one field L2",
            "threads": 1, "truncation": False}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate_inputs(rows):
    if not rows or len({r.get("sample_id") for r in rows}) != len(rows):
        raise StudyError("Collection requires unique nonempty model inputs")
    for row in rows:
        if (set(row) != p.INPUT_FIELDS or row["task"] != "anli" or row["split"] not in SPLITS
                or row["labels"] != ["entailment", "neutral", "contradiction"]):
            raise StudyError("Collection inputs differ from the gold-free canonical schema")


def runtime_plan(source_dirs, output):
    binding, controls = p.plans(source_dirs, output)
    binding = copy.deepcopy(binding)
    binding.update(protocol=COLLECTION_PROTOCOL["id"], collection_revision=1)
    binding.pop("profile_revision", None)
    return binding, controls


def prepare(output, profile_output, rows, study_identity):
    """Copy runtime controls into the new study root; no gold is accepted by this API."""
    output, profile_output = p.normal(output), p.normal(profile_output)
    if output == profile_output or output in profile_output.parents or profile_output in output.parents:
        raise StudyError("Formal collection and development profile need separate roots")
    validate_inputs(rows)
    profile_settings, _ = p.context(profile_output)
    binding, controls = runtime_plan(profile_settings["source_dirs"], output)
    settings = {"protocol": COLLECTION_PROTOCOL, "profile_output": str(profile_output),
                "profile_settings_hash": fingerprint(profile_settings), "study_identity": study_identity,
                "inputs_hash": fingerprint(rows), "binding": binding, "source": p.source_identity()["files"]}
    fixed_write(p.safe(output, "collection/settings.json"), settings)
    fixed_write(p.safe(output, "collection/inputs.json"), rows)
    for role in ROLES:
        for name in p.CONTROL_NAMES:
            fixed_write(p.safe(p.control(output, role), "runtime/"+name), controls[role][name])
    return settings


def context(output, *, offline=False):
    settings = read_json(p.safe(output, "collection/settings.json"))
    rows = read_json(p.safe(output, "collection/inputs.json"))
    validate_inputs(rows)
    if (settings["protocol"] != COLLECTION_PROTOCOL or settings["source"] != p.source_identity()["files"]
            or settings["inputs_hash"] != fingerprint(rows)):
        raise StudyError("Collection source, protocol or exact inputs changed")
    for role in ROLES:
        for name, digest in settings["binding"]["control_identity"][role].items():
            if fingerprint(read_json(p.safe(p.control(output, role), "runtime/"+name))) != digest:
                raise StudyError("Formal runtime control changed")
    if not offline:
        profile_settings, _ = p.context(settings["profile_output"])
        if fingerprint(profile_settings) != settings["profile_settings_hash"]:
            raise StudyError("Accepted development profile settings changed")
        binding, _ = runtime_plan(profile_settings["source_dirs"], output)
        if binding != settings["binding"]:
            raise StudyError("Bound runtime environment changed")
    return settings, rows


def validate_predictions(rows, predictions, role):
    if len(rows) != len(predictions):
        raise StudyError("Incomplete prediction chunk")
    for row, pred in zip(rows, predictions):
        if (pred.get("sample_id") != row["sample_id"] or pred.get("task") != "anli"
                or pred.get("split") != row["split"] or pred.get("model") != p.INTERFACE[role]
                or pred.get("status") != "ok" or pred.get("input_sha256") != fingerprint(row)
                or pred.get("request_sha256") != fingerprint(p.request_body(row, p.INTERFACE[role]))):
            raise StudyError("Prediction does not bind exact request and input")
        if parse_response(pred.get("raw_response"), row, p.INTERFACE[role]) != (pred.get("prediction"), pred.get("probabilities")):
            raise StudyError("Prediction differs from raw response")
        attempts = pred.get("attempts", [])
        if len(attempts) != 1 or attempts[0].get("attempt") != 1 or attempts[0].get("status") != "ok":
            raise StudyError("Formal prediction includes an unapproved request retry")
        a, b, dt, latency = (pred.get(key) for key in
            ("request_started_monotonic", "request_finished_monotonic", "observed_request_seconds", "latency_ms"))
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in (a, b, dt, latency)) or not math.isclose(b-a, dt, abs_tol=1e-8)
                or dt > REQUEST_SECONDS or latency/1000 > dt+.001 or pred.get("profile_acceptance") != "on_time"):
            raise StudyError("Request was late or its timing is inconsistent")


def attempts(output, role):
    folder = p.safe(output, "collection/attempts/"+role)
    return sorted((f for f in folder.iterdir() if f.is_dir()), key=lambda f: f.name) if folder.exists() else []


def chunks(output, role, *, offline=False):
    """Validate and return every committed successful chunk, including before a later failure."""
    settings, rows = context(output, offline=offline)
    pieces = []
    profile_root = p.normal(output).parent/"profile" if offline else Path(settings["profile_output"])
    profile_settings, _ = p.context(profile_root, offline=offline)
    if fingerprint(profile_settings) != settings["profile_settings_hash"]:
        raise StudyError("Bundled development profile settings changed")
    for folder in attempts(output, role):
        receipts = sorted((folder/"chunks").glob("*/receipt.json"))
        if not receipts:
            continue
        plan = read_json(p.safe(folder, "plan.json"))
        launch = read_json(p.safe(folder, "deployment-open.json"))
        token_receipt = read_json(p.safe(folder, "tokens.json"))
        ownership.verify_receipt(profile_root, plan["ownership_selftest"], profile_settings, plan["allocation"])
        p.verify_tokens(output, role, rows, token_receipt)
        p.verify_engine(launch, role)
        if (plan["collection_hash"] != fingerprint(settings) or plan["role"] != role
                or token_receipt["runtime_hash"] != fingerprint(launch["runtime"])):
            raise StudyError("Chunk attempt does not bind its collection and runtime")
        for receipt_path in receipts:
            receipt = read_json(p.safe(receipt_path.parent, "receipt.json"))
            if (receipt.get("plan_hash") != fingerprint(plan)
                    or receipt.get("deployment_sha256") != file_hash(p.safe(folder, "deployment-open.json"))
                    or receipt.get("tokens_sha256") != file_hash(p.safe(folder, "tokens.json"))):
                raise StudyError("Chunk does not bind its plan, token report and actual engine")
            predictions_path = p.safe(receipt_path.parent, "rows.json")
            if receipt["rows_sha256"] != file_hash(predictions_path):
                raise StudyError("Committed chunk bytes changed")
            predictions = read_json(predictions_path)
            start, count = receipt["start"], receipt["count"]
            if (not isinstance(start, int) or isinstance(start, bool) or start < 0
                    or not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= CHUNK_SIZE
                    or start+count > len(rows)):
                raise StudyError("Invalid chunk range")
            validate_predictions(rows[start:start+count], predictions, role)
            pieces.append((start, predictions, folder))
    pieces.sort(key=lambda piece: piece[0])
    result, used = [], []
    for start, predictions, folder in pieces:
        if start != len(result):
            raise StudyError("Prediction chunks overlap or contain a gap")
        result.extend(predictions)
        if folder not in used:
            used.append(folder)
    return result, used


def _write_chunk(output, folder, role, start, rows, predictions):
    validate_predictions(rows[start:start+len(predictions)], predictions, role)
    target = p.safe(folder, f"chunks/{start:08d}")
    fixed_write(p.safe(target, "rows.json"), predictions)
    fixed_write(p.safe(target, "receipt.json"), {
        "role": role, "start": start, "count": len(predictions),
        "rows_sha256": file_hash(p.safe(target, "rows.json")),
        "plan_hash": fingerprint(read_json(p.safe(folder, "plan.json"))),
        "tokens_sha256": file_hash(p.safe(folder, "tokens.json")),
        "deployment_sha256": file_hash(p.safe(folder, "deployment-open.json")), "at": utc_now()})


def worker(output, folder):
    settings, rows = context(output)
    plan = read_json(p.safe(folder, "plan.json"))
    role = plan["role"]
    if plan["collection_hash"] != fingerprint(settings):
        raise StudyError("Worker collection identity changed")
    profile_settings, _ = p.context(settings["profile_output"])
    ownership.verify_receipt(settings["profile_output"], plan["ownership_selftest"], profile_settings, plan["allocation"])
    session = p.session(settings["profile_output"])
    if session["allocation"] != plan["allocation"]:
        raise StudyError("Worker allocation changed")
    session = {**session, "deadline_epoch": min(session["deadline_epoch"], plan["deadline_epoch"])}
    done, _ = chunks(output, role)
    if len(done) != plan["start"]:
        raise StudyError("Worker resume prefix changed")
    target, binding = p.control(output, role), settings["binding"]
    record = {"role": role}
    with legacy_environment(binding, p.PROJECT), p.v3._environment(binding, session):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        write_json(p.safe(target, "runtime/session.json"), session)
        doctor = p.frozen.doctor(target, require_gpu=True)
        gpus = doctor.get("gpu", [])
        if len(gpus) != 1 or gpus[0].get("name") != "NVIDIA L40S" or gpus[0].get("memory_mb", 0) < 44000:
            raise StudyError("Formal collection requires one L40S with at least 44,000 MiB")
        installed = read_json(p.safe(target, "runtime/installations.json"))
        os.environ.update(p.v3._cuda_preflight(target, binding, installed))
        token_path = p.safe(output, f"collection/tokens/{role}.json")
        if token_path.exists():
            token_receipt = read_json(token_path)
            p.verify_tokens(output, role, rows, token_receipt)
            if token_receipt["runtime_hash"] != fingerprint(p.frozen.runtime_config(target, p.INTERFACE[role])):
                raise StudyError("Resumed tokens belong to a different runtime")
        else:
            token_receipt = p.token_check(output, role, rows)
            fixed_write(token_path, token_receipt)
        fixed_write(p.safe(folder, "tokens.json"), token_receipt)
        fixed_write(p.safe(folder, "hardware.json"), hardware())
        try:
            with p.deployment(output, role, sink=record) as endpoint:
                fixed_write(p.safe(folder, "deployment-open.json"), record)
                client = ModelClient(p.INTERFACE[role], endpoint, timeout=REQUEST_SECONDS, max_attempts=1)
                with ResourceMonitor([os.getpid(), *[v["pid"] for v in record["processes"]]], p.safe(folder, "resources.jsonl")):
                    start, pending = len(done), []
                    with p.safe(folder, "requests.jsonl").open("x") as raw:
                        for index in range(start, len(rows)):
                            # A planned allocation checkpoint is success, not a technical failure.
                            if time.time()+REQUEST_SECONDS+10 >= plan["deadline_epoch"]:
                                break
                            row = rows[index]
                            a = time.monotonic(); pred = client.predict(row); b = time.monotonic()
                            pred.update(input_sha256=fingerprint(row), request_sha256=fingerprint(p.request_body(row, p.INTERFACE[role])),
                                        request_started_monotonic=a, request_finished_monotonic=b,
                                        observed_request_seconds=b-a, profile_acceptance="on_time" if b-a <= REQUEST_SECONDS else "late")
                            raw.write(json.dumps(pred, ensure_ascii=False, allow_nan=False)+"\n"); raw.flush(); os.fsync(raw.fileno())
                            validate_predictions([row], [pred], role)
                            pending.append(pred)
                            if len(pending) == CHUNK_SIZE:
                                _write_chunk(output, folder, role, start, rows, pending)
                                start += len(pending); pending = []
                        if pending:
                            _write_chunk(output, folder, role, start, rows, pending)
                            start += len(pending)
            fixed_write(p.safe(folder, "deployment.json"), record)
            fixed_write(p.safe(folder, "worker-finished.json"), {"start": plan["start"], "end": start,
                        "status": "COMPLETE" if start == len(rows) else "CHECKPOINT", "at": utc_now(),
                        "deployment_sha256": file_hash(p.safe(folder, "deployment.json"))})
        finally:
            if not p.safe(folder, "deployment.json").exists():
                fixed_write(p.safe(folder, "deployment.json"), record)


def verify_stop(output, folder):
    path = p.safe(folder, "parent-stop.json")
    if not path.exists():
        path = p.safe(folder, "recovered-stop.json")
    if not path.exists():
        raise StudyError("Contributing attempt has no verified stop")
    value = read_json(path)
    plan = read_json(p.safe(folder, "plan.json"))
    if value.get("plan_hash") != fingerprint(plan):
        raise StudyError("Stop receipt does not bind this attempt plan")
    if value.get("gpu_compute_processes") != []:
        raise StudyError("Contributing attempt has no verified stop")
    if path.name == "parent-stop.json":
        if value.get("gpu_checked") is not True or not value.get("ownership", "").startswith("current descendants of sole-worker Linux subreaper"):
            raise StudyError("Parent stop does not establish owned-process cleanup")
    else:
        old_job = plan.get("allocation", {}).get("slurm_job_id")
        proof = value.get("old_allocation_proof", {})
        new_job = value.get("new_allocation", {}).get("slurm_job_id")
        if (not isinstance(old_job, str) or not old_job.isdigit()
                or proof.get("terminal") is not True or proof.get("job_id") != old_job
                or not isinstance(new_job, str) or not new_job.isdigit() or new_job == old_job):
            raise StudyError("Recovered stop lacks a terminal proof for this attempt's old job and a distinct new job")
    return value


def _clear_previous(output, role, session, retry_failed):
    for folder in attempts(output, role):
        plan = read_json(p.safe(folder, "plan.json"))
        if p.safe(folder, "failure.json").exists() and not retry_failed:
            raise StudyError(f"Previous {role} attempt failed; inspect diagnostics and explicitly use --retry-failed")
        try:
            verify_stop(output, folder)
            continue
        except StudyError:
            pass
        # Metadata from an earlier process is never authority to send signals.
        if plan["allocation"]["slurm_job_id"] == session["allocation"]["slurm_job_id"]:
            raise StudyError("Earlier controller has no confirmed cleanup; preserve it and resume after its allocation ends")
        proof = p.slurm_job_state(plan["allocation"]["slurm_job_id"])
        if proof.get("terminal") is not True or proof.get("job_id") != plan["allocation"]["slurm_job_id"]:
            raise StudyError("Cannot prove this interrupted earlier allocation ended")
        gpu = p.old.assert_gpu_owned_by(set())
        fixed_write(p.safe(folder, "recovered-stop.json"), {"plan_hash": fingerprint(plan), "old_allocation_proof": proof,
                    "new_allocation": session["allocation"], "gpu_compute_processes": gpu,
                    "mode": "read-only terminal-allocation recovery; no historical PID signalled", "at": utc_now()})


def collect_role(output, profile_output, role, *, retry_failed=False):
    if role not in ROLES:
        raise StudyError("Unknown collection role")
    settings, rows = context(output)
    if str(p.normal(profile_output)) != settings["profile_output"]:
        raise StudyError("Collection profile root changed")
    session = p.session(profile_output)
    ownership_proof = ownership.require_receipt(profile_output)
    _clear_previous(output, role, session, retry_failed)
    done, used = chunks(output, role)
    if len(done) == len(rows):
        for folder in used: verify_stop(output, folder)
        return {"status": "COMPLETE", "role": role, "count": len(done), "total": len(rows)}
    until = session["deadline_epoch"]-CLEANUP_SECONDS
    if until-time.time() < REQUEST_SECONDS+300:
        return {"status": "CHECKPOINT", "role": role, "count": len(done), "total": len(rows), "reason": "allocation reserve"}
    folder = p.safe(output, f"collection/attempts/{role}/{time.time_ns()}")
    folder.mkdir(parents=True)
    meta = p.safe(p.control(output, role), "runtime/last_services.json")
    plan = {"role": role, "start": len(done), "collection_hash": fingerprint(settings),
            "allocation": session["allocation"], "started_epoch": time.time(), "deadline_epoch": until,
            "ownership_selftest": ownership_proof, "last_services_before": file_hash(meta) if meta.exists() else None}
    fixed_write(p.safe(folder, "plan.json"), plan)
    proc = None
    with p.subreaper():
        try:
            command = [sys.executable, "-B", "-m", "decision_study.learnability.collection", "worker",
                       "--output-dir", str(p.normal(output)), "--attempt-dir", str(folder)]
            with p.safe(folder, "worker.log").open("x") as log:
                proc = subprocess.Popen(command, cwd=p.PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            fixed_write(p.safe(folder, "worker-process.json"), {"pid": proc.pid, "pgid": proc.pid})
            while proc.poll() is None:
                if time.time() >= until:
                    raise StudyError("Formal collection worker exceeded its absolute allocation watchdog")
                p.session(profile_output)
                try: proc.wait(timeout=min(.5, max(.001, until-time.time())))
                except subprocess.TimeoutExpired: pass
            if proc.returncode != 0 or time.time() > until:
                raise StudyError(f"Formal collection worker did not finish successfully on time: {role} exit={proc.returncode}")
            result = read_json(p.safe(folder, "worker-finished.json"))
            if result.get("status") not in {"COMPLETE", "CHECKPOINT"}:
                raise StudyError("Missing complete/checkpoint worker receipt")
            stopped = p.recover_stop(output, role, plan, worker_pid=proc.pid, journal=p.safe(folder, "parent-cleanup.jsonl"))
            fixed_write(p.safe(folder, "parent-stop.json"), {**stopped, "plan_hash": fingerprint(plan)})
            done, used = chunks(output, role)
            if (result["end"] != len(done) or result["start"] != plan["start"]
                    or result["deployment_sha256"] != file_hash(p.safe(folder, "deployment.json"))
                    or (result["status"] == "COMPLETE") != (len(done) == len(rows))):
                raise StudyError("Completed prefix differs from worker receipt")
            fixed_write(p.safe(folder, "accepted.json"), {"status": result["status"], "plan_hash": fingerprint(plan),
                        "worker_finished_sha256": file_hash(p.safe(folder, "worker-finished.json")),
                        "parent_stop_sha256": file_hash(p.safe(folder, "parent-stop.json")), "accepted_at": utc_now()})
            return {"status": result["status"], "role": role, "count": len(done), "total": len(rows)}
        except BaseException as exc:
            failure = {"error": str(exc), "type": type(exc).__name__, "at": utc_now()}
            if proc is not None:
                try: p.stop_child(proc, plan["started_epoch"])
                except BaseException as stop_error: failure["worker_stop_error"] = str(stop_error)
            try:
                stopped = p.recover_stop(output, role, plan, worker_pid=proc.pid if proc else None,
                                         journal=p.safe(folder, "parent-cleanup.jsonl"))
                fixed_write(p.safe(folder, "parent-stop.json"), {**stopped, "plan_hash": fingerprint(plan)})
            except BaseException as stop_error: failure["cleanup_error"] = str(stop_error)
            fixed_write(p.safe(folder, "failure.json"), failure)
            raise


def predictions(output, role, *, require_complete=True, offline=False):
    _, inputs = context(output, offline=offline)
    rows, used = chunks(output, role, offline=offline)
    if require_complete and len(rows) != len(inputs):
        raise StudyError("Formal role predictions are incomplete")
    for folder in used: verify_stop(output, folder)
    return rows


# Frozen full-field MiniLM descriptor. These functions do not depend on the
# uncommitted CPU-matrix implementation and never receive task gold.
def _pair_text(row):
    prefix, delimiter = "Premise:\n", "\n\nHypothesis:\n"
    text = row["state"]
    if not text.startswith(prefix) or delimiter not in text:
        raise StudyError("ANLI state does not use the frozen premise/hypothesis template")
    a, b = text[len(prefix):].split(delimiter, 1)
    if not a.strip() or not b.strip():
        raise StudyError("Empty ANLI field")
    return a, b


def field_embeddings(texts, tokenizer, session):
    """254 content tokens per block; pool every token without truncation."""
    ids = tokenizer.encode_batch(texts, add_special_tokens=False)
    wrapped = tokenizer.encode_batch(texts, add_special_tokens=True)
    if len(ids) != len(texts) or len(wrapped) != len(texts):
        raise StudyError("Encoder tokenizer dropped a field")
    sums = np.zeros((len(texts), 384), dtype=np.float64)
    totals = np.zeros(len(texts), dtype=np.int64)
    blocks, lengths = [], []
    for i, (plain, full) in enumerate(zip(ids, wrapped)):
        content = list(plain.ids)
        if (not content or plain.overflowing or full.overflowing or full.ids != [101, *content, 102]
                or full.type_ids != [0]*len(full.ids) or full.attention_mask != [1]*len(full.ids)):
            raise StudyError("Encoder field tokenization/truncation/template differs")
        lengths.append(len(content))
        blocks.extend((i, content[j:j+254]) for j in range(0, len(content), 254))
    for start in range(0, len(blocks), 32):
        batch = blocks[start:start+32]; width = max(len(b)+2 for _, b in batch)
        input_ids = np.zeros((len(batch), width), dtype=np.int64); mask = np.zeros_like(input_ids)
        for j, (_, block) in enumerate(batch):
            input_ids[j, :len(block)+2] = [101, *block, 102]; mask[j, :len(block)+2] = 1
        hidden = np.asarray(session.run(["last_hidden_state"], {"input_ids": input_ids, "attention_mask": mask,
                                  "token_type_ids": np.zeros_like(input_ids)})[0], dtype=np.float32)
        if hidden.shape != (len(batch), width, 384) or not np.isfinite(hidden).all():
            raise StudyError("Encoder hidden-state dimensions/values differ")
        weights = mask.astype(np.float32)
        means = (hidden*weights[..., None]).sum(axis=1)/weights.sum(axis=1)[:, None]
        for j, (i, block) in enumerate(batch): sums[i] += len(block)*means[j]; totals[i] += len(block)
    if totals.tolist() != lengths:
        raise StudyError("Encoder chunks did not cover every content token once")
    vectors = sums/totals[:, None]; norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms <= 0).any():
        raise StudyError("Invalid encoder field vectors")
    return np.ascontiguousarray(vectors/norms, dtype=np.float32), lengths, len(blocks)


def encode_rows(rows, model_dir, *, batch_size=128, skip_starts=()):
    """Yield deterministic descriptor batches, using only pinned ONNX CPU bytes."""
    identity = encoder.verify(model_dir, {"encoder": encoder.ENCODER_SPEC})
    root = encoder._root(model_dir)
    tokenizer = encoder._load_tokenizer(encoder._path(root, "tokenizer.json"))
    if {t: tokenizer.token_to_id(t) for t in ("[CLS]", "[SEP]", "[PAD]")} != {"[CLS]": 101, "[SEP]": 102, "[PAD]": 0}:
        raise StudyError("Encoder special IDs changed")
    session = encoder._load_session(encoder._path(root, "onnx/model.onnx"), 1)
    if ({n.name: n.type for n in session.get_inputs()} != {"input_ids": "tensor(int64)", "attention_mask": "tensor(int64)", "token_type_ids": "tensor(int64)"}
            or [n.name for n in session.get_outputs()] != ["last_hidden_state"]):
        raise StudyError("Encoder ONNX signature changed")
    for start in range(0, len(rows), batch_size):
        if start in skip_starts:
            continue
        part = rows[start:start+batch_size]; pairs = [_pair_text(r) for r in part]
        fields, lengths, count = field_embeddings([t for pair in pairs for t in pair], tokenizer, session)
        u, v = fields[0::2].astype(np.float64), fields[1::2].astype(np.float64)
        pair = np.concatenate([u, v, abs(u-v), u*v], axis=1)
        pair /= np.linalg.norm(pair, axis=1, keepdims=True)
        yield start, np.ascontiguousarray(pair, dtype=np.float32), {"identity": identity, "field_tokens": lengths, "chunks": count}


def embedding_worker(output, model_dir):
    from importlib import metadata
    settings, rows = context(output)
    expected = read_json(p.PROJECT/"configs/clinc_cheap_protocol.json")["environment"]
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected:
        raise StudyError("Use the existing pinned cheap-baseline CPU environment")
    root = p.safe(output, "embeddings")
    identity = {"encoding": ENCODING, "inputs_hash": fingerprint(rows), "collection_hash": fingerprint(settings),
                "environment": actual, "model_identity": encoder.verify(model_dir, {"encoder": encoder.ENCODER_SPEC})}
    fixed_write(p.safe(root, "identity.json"), identity)
    completed = []
    for start in range(0, len(rows), 128):
        folder = p.safe(root, f"chunks/{start:08d}")
        if p.safe(folder, "receipt.json").exists():
            verify_embedding_chunk(folder, identity, rows, start)
            completed.append(start)
    for start, matrix, coverage in encode_rows(rows, model_dir, skip_starts=set(completed)):
        folder = p.safe(root, f"chunks/{start:08d}")
        receipt_path = p.safe(folder, "receipt.json")
        if receipt_path.exists():
            verify_embedding_chunk(folder, identity, rows, start)
            continue
        # Each attempt gets its own immutable file; an interruption leaves an
        # unreferenced .npy which is retained and never accepted as complete.
        folder.mkdir(parents=True, exist_ok=True)
        path = p.safe(folder, f"vectors-{time.time_ns()}.npy")
        with path.open("xb") as stream: np.save(stream, matrix, allow_pickle=False)
        fixed_write(receipt_path, {"start": start, "count": len(matrix), "identity_hash": fingerprint(identity),
                    "ids": [r["sample_id"] for r in rows[start:start+len(matrix)]],
                    "vectors": path.name, "sha256": file_hash(path), "coverage": coverage})
    embeddings(output)
    fixed_write(p.safe(root, "complete.json"), {"identity_hash": fingerprint(identity), "count": len(rows),
                "dimension": 1536, "scope": "Frozen CPU field-pair features; no task gold"})


def verify_embedding_chunk(folder, identity, rows, start):
    receipt = read_json(p.safe(folder, "receipt.json"))
    matrix_path = p.safe(folder, receipt["vectors"])
    n = min(128, len(rows)-start)
    if (receipt["identity_hash"] != fingerprint(identity) or receipt["start"] != start or receipt["count"] != n
            or receipt["ids"] != [r["sample_id"] for r in rows[start:start+n]] or file_hash(matrix_path) != receipt["sha256"]):
        raise StudyError("Embedding chunk identity changed")
    with matrix_path.open("rb") as stream: matrix = np.load(stream, allow_pickle=False)
    if (matrix.dtype != np.float32 or matrix.shape != (n, 1536) or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5, rtol=0)):
        raise StudyError("Embedding chunk has invalid shape/values")
    coverage = receipt["coverage"]
    if (coverage["identity"] != identity["model_identity"] or len(coverage["field_tokens"]) != 2*n
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in coverage["field_tokens"])
            or coverage["chunks"] != sum((v+253)//254 for v in coverage["field_tokens"])):
        raise StudyError("Embedding chunk token coverage differs")
    return matrix


def embeddings(output, *, offline=False):
    settings, rows = context(output, offline=offline)
    identity = read_json(p.safe(output, "embeddings/identity.json"))
    if (identity["encoding"] != ENCODING or identity["inputs_hash"] != fingerprint(rows)
            or identity["collection_hash"] != fingerprint(settings)):
        raise StudyError("Embedding settings changed")
    arrays = [verify_embedding_chunk(p.safe(output, f"embeddings/chunks/{i:08d}"), identity, rows, i)
              for i in range(0, len(rows), 128)]
    return [r["sample_id"] for r in rows], np.concatenate(arrays, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "embed"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args(argv)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda sig, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {sig}")))
    if args.command == "worker":
        if args.attempt_dir is None or p.normal(args.output_dir) not in p.normal(args.attempt_dir).parents:
            parser.error("Worker attempt must be inside its output")
        worker(args.output_dir, args.attempt_dir)
    else:
        if args.model_dir is None: parser.error("embed requires --model-dir")
        embedding_worker(args.output_dir, args.model_dir)
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
