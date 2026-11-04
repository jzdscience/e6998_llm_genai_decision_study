"""Resumable v3 collection and a freeze-before-holdout evaluation boundary."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import zipfile

from ..adapters import ModelClient, decision_request, qwen_request, parse_response
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..runner import predict_examples, _probability_comparison
from ..storage import PredictionStore, load_predictions
from . import data, policies, runtime, stats
from .identity import ROLES, interface, job_root, protocol, source_identity, freeze_hash


class BoundClient(ModelClient):
    def predict(self, example):
        return {**super().predict(example), "input_hash": example["input_hash"]}


def _policy_rows(role, rows):
    """Role-tagged view only, after verified job/config/model-revision binding."""
    return {row["sample_id"]: {**row, "model": role} for row in rows}


def config_for(output, role, phase):
    manifest = data.verify_fit(output)
    if manifest["protocol_identity"]["canonical_sha256"] != fingerprint(protocol()):
        raise StudyError("Prepared data belongs to a different v3 protocol")
    examples = data.load_examples(output, role=phase)
    model = interface(role)
    renderer = qwen_request if model == "qwen" else lambda e: decision_request(e, model)
    effective = runtime.runtime_identity(output, role)["runtime"]
    expected = protocol()["models"][role]
    if effective["model_repo"] != expected["repo"] or effective["model_revision"] != expected["revision"]:
        raise StudyError(f"Model identity differs from v3 protocol: {role}")
    required = {"precision": "bfloat16", "max_model_len": 8192, "prefix_cache": False,
                "truncate": False, "backend": "vllm", "vllm_version": "0.31.0",
                "gpu_memory_utilization": expected["gpu_memory_utilization"]}
    if model == "qwen":
        required.update(thinking=False, temperature=0, max_output_tokens=64, constrained_json=True)
    else:
        required.update(think_mode="never", none_check=False, allow_none=False)
    if any(effective.get(k) != v for k, v in required.items()):
        raise StudyError(f"Effective inference settings differ from v3 protocol: {role}")
    return {"protocol": "solo-v3-anli", "role": role, "phase": phase, "model": model,
            "runtime": effective, "sources": source_identity(), "protocol_hash": fingerprint(protocol()),
            "data_manifest_hash": fingerprint(manifest), "expected_predictions": len(examples),
            "request_hash": fingerprint([{"id": e["sample_id"], "request": renderer(e)} for e in examples]),
            "max_attempts": 2, "timeout_seconds": 120, "prediction_concurrency": 1}


def _check_rows(examples, rows, model, config_hash=None, model_revision=None):
    expected = {e["sample_id"]: e for e in examples}
    by_id = {r["sample_id"]: r for r in rows}
    if len(rows) != len(by_id) or set(by_id) != set(expected):
        raise StudyError("Missing, duplicate or unexpected prediction IDs")
    request_ids = set()
    for sid, row in by_id.items():
        e = expected[sid]
        if row.get("model") != model or row.get("input_hash") != e["input_hash"]:
            raise StudyError("Prediction input/model mismatch")
        if row.get("task") != e["task"] or row.get("split") != e["split"]:
            raise StudyError("Prediction task/split mismatch")
        if config_hash is not None and row.get("config_hash") != config_hash:
            raise StudyError("Prediction configuration mismatch")
        if model_revision is not None and row.get("model_revision") != model_revision:
            raise StudyError("Prediction checkpoint revision mismatch")
        attempts = row.get("attempts", [])
        if len(attempts) not in (1, 2) or attempts[-1].get("status") != row.get("status"):
            raise StudyError("Missing or inconsistent request attempts")
        for attempt in attempts:
            request_id = attempt.get("request_id")
            if not isinstance(request_id, str) or not request_id or request_id in request_ids:
                raise StudyError("Missing or duplicate request ID")
            request_ids.add(request_id)
        if not isinstance(row.get("latency_ms"), (float, int)) or not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
            raise StudyError("Invalid request latency")
        if row.get("status") == "ok":
            pred, probs = parse_response(row["raw_response"], e, model)
            if pred != row.get("prediction") or probs != row.get("probabilities"):
                raise StudyError("Stored prediction differs from original response")
        elif row.get("status") != "error":
            raise StudyError("Unknown terminal prediction status")
    return by_id


def _verify_smoke(output, jobdir, config):
    path = Path(jobdir) / "smoke.json"
    saved = read_json(path)
    dev = data.load_examples(output, role="development")
    if saved["config_hash"] != fingerprint(config):
        raise StudyError("Smoke configuration changed")
    for key in ("reference", "current"):
        _check_rows(dev, saved[key], config["model"])
    checks = _probability_comparison(dev, saved["reference"], saved["current"])
    if checks != saved["checks"] or not all(x["passed"] for x in checks.values()):
        raise StudyError("Serial reproducibility smoke failed")
    return file_hash(path)


def _verify_tokens(output, jobdir, config):
    path = Path(jobdir) / "token_preflight.json"
    value = read_json(path)
    examples = data.load_examples(output, role=config["phase"]) + data.load_examples(output, role="development")
    expected = {e["sample_id"] for e in examples}
    rows = value.get("rows", [])
    fixed = {"status": "ok", "model": config["model"],
             "model_revision": config["runtime"]["model_revision"], "max_model_len": 8192,
             "n_examples": len(examples), "truncated": False, "overlong": [],
             "effective_config_hash": fingerprint(config["runtime"])}
    if any(value.get(k) != v for k, v in fixed.items()) or len(rows) != len(expected) or {r["sample_id"] for r in rows} != expected:
        raise StudyError("Exact-token preflight identity or sample coverage differs")
    reserve = 64 if config["model"] == "qwen" else 1
    for row in rows:
        count = row.get("input_tokens")
        digest = row.get("token_ids_sha256", "")
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1
                or row.get("reserved_output_tokens") != reserve or count + reserve > 8192
                or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise StudyError("Malformed or overlong exact-token record")
    return file_hash(path)


def verified_job(output, role, phase):
    directory = job_root(output, role, phase)
    config = config_for(output, role, phase)
    model = interface(role)
    if read_json(directory / "predictions" / model / "config.json") != config:
        raise StudyError("Stored job configuration differs from the current frozen inputs/runtime/source")
    rows = load_predictions(directory, model)
    examples = data.load_examples(output, role=phase)
    _check_rows(examples, rows, model, fingerprint(config), config["runtime"]["model_revision"])
    done = read_json(directory / "completion.json")
    checks = {"config_hash": fingerprint(config), "count": len(examples),
              "predictions_hash": fingerprint(sorted(rows, key=lambda r: r["sample_id"])),
              "smoke_sha256": _verify_smoke(output, directory, config),
              "token_preflight_sha256": _verify_tokens(output, directory, config),
              "failures": sum(r["status"] != "ok" for r in rows)}
    if any(done.get(k) != value for k, value in checks.items()):
        raise StudyError(f"Completion receipt mismatch: {role}/{phase}")
    if phase == "holdout":
        frozen = read_json(Path(output) / "policy_freeze.json")
        freeze_time = datetime.fromisoformat(frozen["created_at"])
        if done.get("freeze_hash") != frozen["freeze_hash"] or any(
                datetime.fromisoformat(r["recorded_at"]) < freeze_time for r in rows):
            raise StudyError("Holdout prediction precedes or differs from policy freeze")
    return rows, done


def run_job(v1_dir, output, role, phase):
    output = Path(output)
    if phase == "holdout":
        freeze = make_freeze(output, require_existing=True)
    else:
        freeze = None
    directory = job_root(output, role, phase)
    if (directory / "completion.json").exists():
        return verified_job(output, role, phase)[1]
    if (output / "holdout_access.json").exists():
        raise StudyError("Cannot collect new predictions after first holdout evaluation")
    examples = data.load_examples(output, role=phase)
    dev = data.load_examples(output, role="development")
    with runtime.runtime_context(v1_dir, output, role) as (control, budget):
        runtime.ensure_download(output, role, control=control)
        tokens = runtime.token_preflight(output, role, examples + dev, control=control)
        if tokens.get("status") != "ok":
            raise StudyError("Exact-token preflight failed; no truncation or silent replacement")
        config = config_for(output, role, phase)
        write_json(directory / "token_preflight.json", tokens)
        _verify_tokens(output, directory, config)
        store = PredictionStore(directory, interface(role), config)
        if not set(store.rows) <= {e["sample_id"] for e in examples}:
            raise StudyError("Unexpected persisted prediction IDs")
        with runtime.managed_services(output, role, control=control) as endpoint:
            client = BoundClient(interface(role), endpoint)
            # Development-only, same-model reproducibility; not independent correctness.
            reference, current = [], []
            for target in (reference, current):
                for example in dev:
                    budget.check()
                    target.append(client.predict(example))
            checks = _probability_comparison(dev, reference, current)
            smoke = {"config_hash": fingerprint(config), "created_at": utc_now(),
                     "reference": reference, "current": current, "checks": checks,
                     "scope": "Repeated serial development responses; no accuracy gate or independent reference implementation"}
            # Keep each startup, including failed runs, separate; latest smoke is bound by completion.
            attempt = directory / "smoke-attempts" / (utc_now().replace(":", "-") + ".json")
            write_json(attempt, smoke)
            write_json(directory / "smoke.json", smoke)
            _verify_smoke(output, directory, config)
            result = predict_examples(examples, client, store, budget=budget)
        rows = list(store.rows.values())
        _check_rows(examples, rows, interface(role), fingerprint(config), config["runtime"]["model_revision"])
        receipt = {"role": role, "phase": phase, "config_hash": fingerprint(config),
                   "count": len(rows), "predictions_hash": fingerprint(sorted(rows, key=lambda r: r["sample_id"])),
                   "failures": sum(r["status"] != "ok" for r in rows),
                   "smoke_sha256": _verify_smoke(output, directory, config),
                   "token_preflight_sha256": _verify_tokens(output, directory, config),
                   "created_at": utc_now(), "sampling": result, "scored_holdout": False,
                   "freeze_hash": freeze["freeze_hash"] if freeze else None}
        write_json(directory / "completion.json", receipt)
    return verified_job(output, role, phase)[1]


def _fit_payload(output):
    manifest = data.verify_fit(output)
    examples = data.load_examples(output, role="fit")
    calibration = [e for e in examples if e["split"] == "calibration"]
    validation = [e for e in examples if e["split"] == "validation"]
    gold = data.load_fit_gold(output)
    jobs = {role: verified_job(output, role, "fit") for role in ROLES}
    policy_rows = {role: _policy_rows(role, jobs[role][0]) for role in ROLES}
    frozen, validation_metrics = {}, {}
    for fallback in ("qwen4b", "qwen9b"):
        fitted = policies.fit_policies(calibration, policy_rows["startlux4b"], policy_rows[fallback],
                                       gold, fallback_model=fallback, l2=protocol()["policies"]["l2"])
        frozen[fallback], validation_metrics[fallback] = {}, {}
        for name, policy in fitted.items():
            locked = policies.freeze_budget(policy, validation, policy_rows["startlux4b"],
                                            budget=protocol()["policies"]["validation_budget"])
            frozen[fallback][name] = locked
            validation_metrics[fallback][name] = stats.evaluate_policy(
                locked, validation, policy_rows["startlux4b"], policy_rows[fallback], gold)[0]
    return {"schema_version": 1, "protocol": "solo-v3-anli",
            "data_manifest_hash": fingerprint(manifest), "protocol_hash": fingerprint(protocol()),
            "sources": source_identity(), "fit_prediction_hashes": {r: jobs[r][1]["predictions_hash"] for r in ROLES},
            "policies": frozen, "validation_metrics": validation_metrics,
            "primary": {"fallback": "qwen9b", "candidate": "positive_gain", "reference": "confidence"}}


def make_freeze(output, *, require_existing=False):
    path = Path(output) / "policy_freeze.json"
    if require_existing and not path.exists():
        raise StudyError("Fit and freeze all policies before collecting/scoring holdout")
    payload = _fit_payload(output)
    if path.exists():
        saved = read_json(path)
        payload["created_at"] = saved["created_at"]
        payload["freeze_hash"] = freeze_hash(payload)
        if payload != saved:
            raise StudyError("Policy freeze differs from fitting evidence, data, protocol or source")
        return saved
    if (Path(output) / "holdout_access.json").exists():
        raise StudyError("Cannot create a new policy freeze after evaluation")
    payload["created_at"] = utc_now()
    payload["freeze_hash"] = freeze_hash(payload)
    write_json(path, payload)
    return payload


def evaluate(output):
    output = Path(output)
    frozen = make_freeze(output, require_existing=True)
    examples = data.load_examples(output, role="holdout")
    jobs = {role: verified_job(output, role, "holdout") for role in ROLES}
    policy_rows = {role: _policy_rows(role, jobs[role][0]) for role in ROLES}
    access_path = output / "holdout_access.json"
    access = {"freeze_hash": frozen["freeze_hash"], "data_manifest_hash": frozen["data_manifest_hash"],
              "protocol_hash": frozen["protocol_hash"], "holdout_predictions": {r: jobs[r][1]["predictions_hash"] for r in ROLES}}
    if access_path.exists():
        old = read_json(access_path)
        if any(old.get(k) != v for k, v in access.items()):
            raise StudyError("First holdout access is bound to different evidence")
    else:
        write_json(access_path, {**access, "created_at": utc_now()})
    # This is the first call allowed to parse holdout labels.
    gold = data.load_holdout_gold_for_evaluation(output, expected_freeze_hash=frozen["freeze_hash"])
    direct_results = {role: stats.evaluate_direct(examples, policy_rows[role], gold, model=role) for role in ROLES}
    direct = {role: pair[0] for role, pair in direct_results.items()}
    direct_rounds = {role: {round_name: stats.evaluate_direct(
        [e for e in examples if e["round"] == round_name], policy_rows[role], gold, model=role)[0]
        for round_name in ("R1", "R2", "R3")} for role in ROLES}
    complementarity = {}
    for fallback in ("qwen4b", "qwen9b"):
        pairs = list(zip(direct_results["startlux4b"][1], direct_results[fallback][1]))
        complementarity[fallback] = {
            "both_correct": sum(a and b for a, b in pairs),
            "both_wrong": sum(not a and not b for a, b in pairs),
            "local_correct_fallback_wrong": sum(a and not b for a, b in pairs),
            "fallback_correct_local_wrong": sum(not a and b for a, b in pairs),
            "oracle_either_correct_accuracy": sum(a or b for a, b in pairs) / len(pairs),
            "notice": "Label-informed descriptive upper bound, not a deployable policy"}
    measured, correct, rounds = {}, {}, {}
    for fallback, by_name in frozen["policies"].items():
        measured[fallback], correct[fallback], rounds[fallback] = {}, {}, {}
        for name, policy in by_name.items():
            metric, outcomes = stats.evaluate_policy(policy, examples, policy_rows["startlux4b"], policy_rows[fallback], gold)
            measured[fallback][name], correct[fallback][name] = metric, outcomes
            rounds[fallback][name] = {round_name: stats.evaluate_policy(
                policy, [e for e in examples if e["round"] == round_name], policy_rows["startlux4b"], policy_rows[fallback], gold)[0]
                for round_name in ("R1", "R2", "R3")}
    difference = stats.cluster_bootstrap_difference(examples, correct["qwen9b"]["positive_gain"],
                correct["qwen9b"]["confidence"], repetitions=2000, seed=6998)
    result = {"status": "complete", "protocol": "solo-v3-anli", "freeze_hash": frozen["freeze_hash"],
              "data_manifest_hash": frozen["data_manifest_hash"], "primary_difference": difference,
              "n": len(examples), "group_count": len({e["group_id"] for e in examples}),
              "direct": direct, "policies": measured, "rounds_descriptive": rounds,
              "direct_rounds_descriptive": direct_rounds, "complementarity_descriptive": complementarity,
              "inputs_hash": fingerprint({**access, "gold": gold}),
              "scope": "Prospective quality only. Serial collection times do not measure co-resident cascade goodput."}
    target = output / "evaluation.json"
    if target.exists():
        old = read_json(target)
        result["created_at"] = old["created_at"]
        result["evaluation_hash"] = fingerprint(result)
        if result != old:
            raise StudyError("Evaluation differs from previously recorded result; refusing overwrite")
    else:
        result["created_at"] = utc_now()
        result["evaluation_hash"] = fingerprint(result)
        write_json(target, result)
    receipt_path = output / "evaluation_receipt.json"
    receipt = {"event": "holdout_evaluated", "evaluation_hash": result["evaluation_hash"],
               "freeze_hash": frozen["freeze_hash"], "inputs_hash": result["inputs_hash"],
               "created_at": result["created_at"]}
    if receipt_path.exists() and read_json(receipt_path) != receipt:
        raise StudyError("Existing evaluation receipt conflicts")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    return result


def status(output):
    output = Path(output)
    manifest = data.verify_fit(output)
    jobs = {}
    for phase in ("fit", "holdout"):
        for role in ROLES:
            p = job_root(output, role, phase) / "completion.json"
            key = f"{role}-{phase}"
            jobs[key] = {"complete": False}
            if p.exists():
                rows, receipt = verified_job(output, role, phase)
                jobs[key] = {"complete": True, "count": len(rows), "failures": receipt["failures"]}
    evaluation_present = (output / "evaluation.json").exists()
    return {"protocol": "solo-v3-anli", "counts": manifest["counts"], "jobs": jobs,
            "freeze_present": (output / "policy_freeze.json").exists(),
            "evaluation_present": evaluation_present,
            "notice": "Status does not parse sealed gold or execute GPU inference; presence alone is not independent result acceptance."}


def export(output):
    output = Path(output).resolve()
    destination = output / "exports/v3_bundle.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for parent, dirs, names in os.walk(output, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {"work", "runtime-work", "exports", "__pycache__", ".git", "envs", "cache", "models", "weights"}
                   and not d.startswith((".venv", "venv")) and not (Path(parent) / d).is_symlink()]
        for name in names:
            p = Path(parent) / name
            if p.is_file() and not p.is_symlink() and name not in {".active.lock", "bundle_manifest.json"} and not name.endswith(".tmp"):
                files.append(p)
    manifest = {"protocol": "solo-v3-anli", "created_at": utc_now(),
                "files": {str(p.relative_to(output)): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(files)}}
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.relative_to(output))
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    temporary.replace(destination)
    return destination
