"""One fixed StartLux9B job, resumable collection and paired retrospective scoring."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import zipfile
from ..adapters import decision_request
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..runner import predict_examples, _probability_comparison
from ..storage import PredictionStore, load_predictions
from ..v3.study import BoundClient, _check_rows
from ..v3 import stats
from . import data, runtime
from .identity import ROLE, protocol, source_identity


def job_root(output):
    return runtime._safe(output, "predictions-job")


def config_for(output):
    manifest = data.verify(output)
    examples = data.examples(output, "holdout")
    effective = runtime.runtime_identity(output, ROLE)["runtime"]
    model = protocol()["model"]
    required = {k: model[k] for k in ("native_temperature", "gpu_memory_utilization", "precision", "max_model_len",
                                    "think_mode", "bidirectional", "none_check", "allow_none", "prefix_cache", "truncate")}
    required.update(model_repo=model["repo"], model_revision=model["revision"], backend="vllm", vllm_version="0.31.0")
    if any(effective.get(k) != v for k, v in required.items()):
        raise StudyError("Effective StartLux9B inference settings differ from the fixed protocol")
    return {"protocol": protocol()["protocol"], "role": ROLE, "model": "startlux", "runtime": effective,
            "sources": source_identity(), "protocol_hash": fingerprint(protocol()),
            "manifest_hash": manifest["manifest_hash"], "expected_predictions": len(examples),
            "request_hash": fingerprint([{"id": e["sample_id"], "request": decision_request(e, "startlux")} for e in examples]),
            "max_attempts": 2, "timeout_seconds": 120, "prediction_concurrency": 1}


def make_freeze(output, *, require_existing=False):
    path = runtime._safe(output, "freeze.json")
    config = config_for(output)
    payload = {"schema_version": 1, "protocol": protocol()["protocol"], "config": config,
               "manifest_hash": config["manifest_hash"], "scope": protocol()["scope"],
               "evaluation": protocol()["evaluation"]}
    if path.exists():
        saved = data.checked(path, "freeze_hash")
        if {k: v for k, v in saved.items() if k not in {"created_at", "freeze_hash"}} != payload:
            raise StudyError("Frozen size control differs from source/data/runtime; preserve this run")
        return saved
    if require_existing or runtime._safe(output, "evaluation-access.json").exists():
        raise StudyError("An existing size-control freeze is required")
    if list((job_root(output) / "predictions/startlux").glob("shard-*.json")):
        raise StudyError("Formal predictions exist without their original freeze")
    result = data.seal({**payload, "created_at": utc_now()}, "freeze_hash")
    write_json(path, result)
    return result


def verify_tokens(output, config):
    path = job_root(output) / "token_preflight.json"
    value = read_json(path)
    examples = data.examples(output, "holdout") + data.examples(output, "development")
    expected = {e["sample_id"] for e in examples}
    rows = value.get("rows", [])
    fixed = {"status": "ok", "model": "startlux", "model_revision": config["runtime"]["model_revision"],
             "max_model_len": 8192, "n_examples": len(examples), "truncated": False, "overlong": [],
             "effective_config_hash": fingerprint(config["runtime"])}
    if any(value.get(k) != v for k, v in fixed.items()) or len(rows) != len(expected) or {r["sample_id"] for r in rows} != expected:
        raise StudyError("Exact-token preflight identity or coverage differs")
    for row in rows:
        n, sha = row.get("input_tokens"), row.get("token_ids_sha256", "")
        if (isinstance(n, bool) or not isinstance(n, int) or n < 1 or n + 1 > 8192
                or row.get("reserved_output_tokens") != 1 or not isinstance(sha, str)
                or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
            raise StudyError("Malformed or overlong exact-token record")
    return file_hash(path)


def verify_smoke(output, config):
    path = job_root(output) / "smoke.json"
    saved = read_json(path)
    dev = data.examples(output, "development")
    if saved.get("config_hash") != fingerprint(config):
        raise StudyError("Smoke configuration changed")
    for key in ("reference", "current"):
        _check_rows(dev, saved[key], "startlux")
    checks = _probability_comparison(dev, saved["reference"], saved["current"])
    if checks != saved["checks"] or not all(x["passed"] for x in checks.values()):
        raise StudyError("Serial repeatability smoke failed")
    return file_hash(path)


def verified_job(output):
    freeze = make_freeze(output, require_existing=True)
    config = config_for(output)
    directory = job_root(output)
    if read_json(directory / "predictions/startlux/config.json") != config:
        raise StudyError("Stored job configuration changed")
    examples = data.examples(output, "holdout")
    rows = load_predictions(directory, "startlux")
    _check_rows(examples, rows, "startlux", fingerprint(config), config["runtime"]["model_revision"])
    done = data.checked(directory / "completion.json", "completion_hash")
    expected = {"config_hash": fingerprint(config), "freeze_hash": freeze["freeze_hash"], "count": len(examples),
                "predictions_hash": fingerprint(sorted(rows, key=lambda r: r["sample_id"])),
                "failures": sum(r["status"] != "ok" for r in rows),
                "smoke_sha256": verify_smoke(output, config), "token_preflight_sha256": verify_tokens(output, config)}
    if any(done.get(k) != v for k, v in expected.items()):
        raise StudyError("Completion receipt differs from actual evidence")
    frozen_at = datetime.fromisoformat(freeze["created_at"])
    if frozen_at.tzinfo is None or any(datetime.fromisoformat(r["recorded_at"]) < frozen_at for r in rows):
        raise StudyError("Formal prediction predates the size-control freeze")
    return rows, done


def run_job(v1_dir, output, *, smoke_only=False):
    output = Path(output)
    directory = job_root(output)
    if (directory / "completion.json").exists():
        return verified_job(output)[1]
    if runtime._safe(output, "evaluation-access.json").exists():
        raise StudyError("Cannot collect after first evaluation access")
    examples = data.examples(output, "holdout")
    dev = data.examples(output, "development")
    with runtime.runtime_context(v1_dir, output, ROLE) as (control, budget):
        runtime.ensure_download(output, ROLE, control=control)
        tokens = runtime.token_preflight(output, ROLE, examples + dev, control=control)
        if tokens.get("status") != "ok":
            raise StudyError("Exact-token preflight failed; no truncation or alternate checkpoint")
        config = config_for(output)
        write_json(directory / "token_preflight.json", tokens)
        verify_tokens(output, config)
        freeze = make_freeze(output)
        with runtime.managed_services(output, ROLE, control=control) as endpoint:
            client = BoundClient("startlux", endpoint)
            reference, current = [], []
            for target in (reference, current):
                for example in dev:
                    budget.check()
                    target.append(client.predict(example))
            smoke = {"config_hash": fingerprint(config), "created_at": utc_now(),
                     "reference": reference, "current": current,
                     "checks": _probability_comparison(dev, reference, current),
                     "scope": "Two serial development passes; repeatability, not independent backend correctness"}
            attempt = directory / "smoke-attempts" / (utc_now().replace(":", "-") + ".json")
            write_json(attempt, smoke)
            write_json(directory / "smoke.json", smoke)
            verify_smoke(output, config)
            if smoke_only:
                return {"status": "smoke_passed", "formal_predictions_collected": False, "freeze_hash": freeze["freeze_hash"]}
            store = PredictionStore(directory, "startlux", config, spool_dir=output / "runtime-work/spool")
            if not set(store.rows) <= {e["sample_id"] for e in examples}:
                raise StudyError("Unexpected persisted prediction IDs")
            sampling = predict_examples(examples, client, store, budget=budget)
        rows = list(store.rows.values())
        _check_rows(examples, rows, "startlux", fingerprint(config), config["runtime"]["model_revision"])
        done = data.seal({"config_hash": fingerprint(config), "freeze_hash": freeze["freeze_hash"], "count": len(rows),
                         "predictions_hash": fingerprint(sorted(rows, key=lambda r: r["sample_id"])),
                         "failures": sum(r["status"] != "ok" for r in rows), "sampling": sampling,
                         "smoke_sha256": verify_smoke(output, config), "token_preflight_sha256": verify_tokens(output, config),
                         "created_at": utc_now(), "scored": False}, "completion_hash")
        write_json(directory / "completion.json", done)
    return verified_job(output)[1]


def evaluate(output):
    output = Path(output)
    freeze = make_freeze(output, require_existing=True)
    rows, done = verified_job(output)
    examples = data.examples(output, "holdout")
    refs = data.references(output)
    access_path, result_path = runtime._safe(output, "evaluation-access.json"), runtime._safe(output, "evaluation.json")
    payload = {"freeze_hash": freeze["freeze_hash"], "completion_hash": done["completion_hash"],
               "manifest_hash": freeze["manifest_hash"],
               "reference_hashes": {r: fingerprint(p) for r, p in refs.items()}}
    if access_path.exists():
        access = data.checked(access_path, "access_hash")
        if any(access.get(k) != v for k, v in payload.items()):
            raise StudyError("Evaluation access is bound to different predictions")
        if not result_path.exists():
            raise StudyError("Evaluation access exists without a completed result; preserve partial evidence for review")
    else:
        if result_path.exists():
            raise StudyError("Evaluation result exists without its original access receipt")
        access = data.seal({**payload, "created_at": utc_now()}, "access_hash")
        write_json(access_path, access)
    gold = data.gold_for_evaluation(output, freeze, access)
    aligned = {role: {r["sample_id"]: {**r, "model": role} for r in predictions}
               for role, predictions in {**refs, ROLE: rows}.items()}
    direct, outcomes, rounds = {}, {}, {}
    for role, predictions in aligned.items():
        direct[role], outcomes[role] = stats.evaluate_direct(examples, predictions, gold, model=role)
        rounds[role] = {r: stats.evaluate_direct([e for e in examples if e["round"] == r], predictions, gold, model=role)[0]
                        for r in ("R1", "R2", "R3")}
    comparisons = {}
    p = protocol()["evaluation"]
    for reference in ("qwen9b", "startlux4b"):
        pairs = list(zip(outcomes[ROLE], outcomes[reference]))
        comparisons[f"{ROLE}_minus_{reference}"] = {
            "difference": stats.cluster_bootstrap_difference(examples, outcomes[ROLE], outcomes[reference],
                repetitions=p["bootstrap_repetitions"], seed=p["bootstrap_seed"]),
            "both_correct": sum(a and b for a, b in pairs), "both_wrong": sum(not a and not b for a, b in pairs),
            "candidate_only_correct": sum(a and not b for a, b in pairs),
            "reference_only_correct": sum(not a and b for a, b in pairs)}
    original = read_json(output / "references/provenance.json")["original_evaluation"]["direct"]
    if any(direct[role] != original[role] for role in refs):
        raise StudyError("Historical direct metrics differ from the accepted V3 evaluation")
    result = {"status": "complete", "protocol": protocol()["protocol"], "freeze_hash": freeze["freeze_hash"],
              "access_hash": access["access_hash"], "completion_hash": done["completion_hash"],
              "direct": direct, "rounds_descriptive": rounds, "comparisons_descriptive": comparisons,
              "gold_sha256": file_hash(output / "sealed/holdout_gold.json"), "n": len(examples),
              "scope": protocol()["scope"], "limits": p["claim_limit"]}
    if result_path.exists():
        old = data.checked(result_path, "evaluation_hash")
        result["created_at"] = old["created_at"]
        result = data.seal(result, "evaluation_hash")
        if result != old:
            raise StudyError("Recomputed evaluation differs; refusing overwrite")
    else:
        result = data.seal({**result, "created_at": utc_now()}, "evaluation_hash")
        write_json(result_path, result)
    return result


def status(output):
    output = Path(output)
    manifest = data.verify(output)
    complete = (job_root(output) / "completion.json").exists()
    done = verified_job(output)[1] if complete else None
    frozen = (output / "freeze.json").exists()
    if frozen:
        make_freeze(output, require_existing=True)
    evaluation = (output / "evaluation.json").exists()
    if evaluation:
        value = data.checked(output / "evaluation.json", "evaluation_hash")
        access = data.checked(output / "evaluation-access.json", "access_hash")
        freeze = make_freeze(output, require_existing=True)
        if (value.get("status") != "complete" or value.get("protocol") != protocol()["protocol"]
                or not done or value.get("n") != done["count"]
                or value.get("completion_hash") != done["completion_hash"]
                or access.get("completion_hash") != done["completion_hash"]
                or value.get("access_hash") != access["access_hash"]
                or value.get("freeze_hash") != freeze["freeze_hash"]
                or access.get("freeze_hash") != freeze["freeze_hash"]
                or access.get("manifest_hash") != manifest["manifest_hash"]):
            raise StudyError("Evaluation identity differs from the completed job")
    return {"protocol": protocol()["protocol"], "prepared_count": manifest["count"], "frozen": frozen,
            "collection_complete": complete, "completion": done, "evaluation_present": evaluation,
            "notice": "Read-only artifact validation; no GPU or test gold read"}


def export(output):
    output = Path(output).absolute()
    destination = runtime._safe(output, "exports/size_control_bundle.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for parent, dirs, names in os.walk(output, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {"runtime-work", "exports", "__pycache__", ".git", "models", "weights", "envs", "cache"}
                   and not d.startswith((".venv", "venv")) and not (Path(parent) / d).is_symlink()]
        for name in names:
            path = Path(parent) / name
            if path.is_file() and not path.is_symlink() and name not in {".active.lock", "bundle_manifest.json"} and not name.endswith(".tmp"):
                files.append(path)
    manifest = {"protocol": protocol()["protocol"], "created_at": utc_now(),
                "files": {str(p.relative_to(output)): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(files)}}
    temp = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(output))
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    temp.replace(destination)
    return destination
