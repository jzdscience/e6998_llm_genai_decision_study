"""Protocol gates, live HTTP prediction, and correctness acceptance checks."""
from __future__ import annotations

import concurrent.futures
import time
import uuid
from pathlib import Path

import numpy as np

from .adapters import ModelClient, decision_request, qwen_request
from .common import StudyError, file_hash, fingerprint, read_json, set_status, utc_now, write_json
from .storage import PredictionStore, load_predictions


def protocol_config(output_dir: Path, model: str) -> dict:
    from . import runtime
    from .data import load_examples, verify_data
    verify_data(output_dir)
    manifest = read_json(Path(output_dir) / "data" / "manifest.json")
    examples = load_examples(output_dir)
    renderer = qwen_request if model == "qwen" else lambda ex: decision_request(ex, model)
    # Hash actual serialized requests, including their explicit option order.
    prompt_hash = fingerprint([{"id": e["sample_id"], "labels": e["labels"], "request": renderer(e)} for e in examples])
    return {
        "schema_version": 1, "protocol": "solo-v1", "model": model,
        "data_fingerprint": manifest["fingerprint"], "prompt_hash": prompt_hash,
        "controller_source_hash": fingerprint({
            name: file_hash(Path(__file__).parent / name)
            for name in ("adapters.py", "runner.py", "runtime.py", "storage.py", "common.py")
        }),
        "runtime": runtime.runtime_config(output_dir, model),
        "timeout_seconds": 120, "max_attempts": 2, "retry_delay_seconds": 1,
        "prediction_concurrency": 1, "max_model_len": 8192,
    }


def _decorate(row: dict, config: dict, session_id: str) -> dict:
    return {**row, "config_hash": fingerprint(config), "session_id": session_id,
            "model_revision": config["runtime"].get("model_revision"), "recorded_at": utc_now()}


def predict_examples(examples: list[dict], client: ModelClient, store: PredictionStore,
                     *, budget=None, progress=print) -> dict:
    session_id = str(uuid.uuid4())
    remaining = [e for e in examples if e["sample_id"] not in store.rows]
    started, errors = time.perf_counter(), 0
    try:
        for i, example in enumerate(remaining, 1):
            if budget:
                budget.check()
            row = _decorate(client.predict(example), store.config, session_id)
            store.append(row)
            errors += row["status"] != "ok"
            if i % 25 == 0 or i == len(remaining):
                progress(f"{client.model}: {i}/{len(remaining)} new, {len(store.rows)} persisted/queued; {errors} failures")
            if i == 50 and budget:
                budget.estimate_remaining((time.perf_counter()-started)/i, len(remaining)-i)
            if i >= 50 and errors / i > .01:
                raise StudyError("More than 1% terminal runtime errors after 50 inputs. Stop and inspect errors; labels/accuracy do not trigger this check.")
    finally:
        store.sync()
    return {"requested": len(examples), "new": len(remaining), "terminal_errors": errors,
            "elapsed_seconds": time.perf_counter()-started, "session_id": session_id}


def _probability_comparison(examples: list[dict], reference: list[dict], candidate: list[dict]) -> dict:
    left = {r["sample_id"]: r for r in reference}
    right = {r["sample_id"]: r for r in candidate}
    expected = {e["sample_id"] for e in examples}
    if len(left) != len(reference) or len(right) != len(candidate) or set(left) != expected or set(right) != expected:
        raise StudyError("Smoke missing, duplicated, or mismatched sample IDs")
    reports = {}
    for task in sorted({e["task"] for e in examples}):
        subset = [e for e in examples if e["task"] == task]
        deltas, same = [], []
        for e in subset:
            a, b = left[e["sample_id"]], right[e["sample_id"]]
            if a["status"] != "ok" or b["status"] != "ok":
                raise StudyError(f"Smoke runtime failure on {e['sample_id']}")
            same.append(a["prediction"] == b["prediction"])
            if a.get("probabilities") is not None:
                if b.get("probabilities") is None:
                    raise StudyError("Candidate omitted full probabilities")
                deltas.append(max(abs(a["probabilities"][l]-b["probabilities"][l]) for l in e["labels"]))
        reports[task] = {
            "count": len(subset), "argmax_agreement": float(np.mean(same)),
            "mean_max_probability_delta": float(np.mean(deltas)) if deltas else None,
            "p95_max_probability_delta": float(np.quantile(deltas, .95)) if deltas else None,
            "passed": bool(np.mean(same) >= .98 and (not deltas or (np.mean(deltas) <= .01 and np.quantile(deltas, .95) <= .05))),
        }
    return reports


def smoke_model(project_dir: Path, output_dir: Path, model: str, *, budget=None) -> dict:
    from . import runtime
    from .data import load_examples
    config = protocol_config(output_dir, model)
    dev = load_examples(output_dir, split="development")
    path = Path(output_dir) / "smoke" / f"{model}.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("config_hash") == fingerprint(config) and cached.get("passed"):
            return cached
    # Reference model exits before the production server starts, so this also fits one GPU.
    reference = runtime.reference_predictions(project_dir, output_dir, model, dev) if model != "qwen" else None
    if budget:
        budget.check()
    with runtime.managed_services(project_dir, output_dir, [model]) as endpoints:
        client = ModelClient(model, endpoints[model])
        serial = []
        for e in dev:
            if budget:
                budget.check()
            serial.append(client.predict(e))
    reference_check = _probability_comparison(dev, reference or serial, serial)
    concurrency_trials, selected_concurrency = [], 1
    parallel, concurrency_check = [], None
    reference_passed = all(v["passed"] for v in reference_check.values())
    if reference_passed:
        for concurrency in (8, 4):
            if budget:
                budget.check()
            trial_rows = []
            try:
                with runtime.managed_services(project_dir, output_dir, [model]) as endpoints:
                    client = ModelClient(model, endpoints[model])
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        trial_rows = list(pool.map(client.predict, dev))
                check = _probability_comparison(dev, serial, trial_rows)
                concurrency_trials.append({"concurrency": concurrency, "checks": check, "rows": trial_rows})
                if all(v["passed"] for v in check.values()):
                    parallel, concurrency_check, selected_concurrency = trial_rows, check, concurrency
                    break
            except (StudyError, RuntimeError) as exc:
                concurrency_trials.append({"concurrency": concurrency, "error": str(exc), "rows": trial_rows})
    client_concurrency_verified = selected_concurrency > 1
    report = {
        "model": model, "config_hash": fingerprint(config), "config": config,
        "reference_available": reference is not None,
        "reference_check": reference_check, "concurrency_check": concurrency_check,
        "selected_concurrency": selected_concurrency, "concurrency_trials": concurrency_trials,
        "concurrency_status": "verified" if client_concurrency_verified else "serial_only",
        "client_concurrency_verified": client_concurrency_verified,
        # Serial fallback is valid for quality inference. It is not evidence of
        # concurrent request correctness or server-side batching support.
        "passed": reference_passed and (not client_concurrency_verified or
                                        all(v["passed"] for v in concurrency_check.values())),
        "created_at": utc_now(), "reference": reference, "serial": serial, "concurrent": parallel,
    }
    write_json(path, report)
    set_status(output_dir, f"smoke/{model}", "complete" if report["passed"] else "blocked",
               config_hash=report["config_hash"])
    if not report["passed"]:
        raise StudyError(f"{model} smoke differences exceeded diagnostic thresholds. See {path}; no formal predictions started.")
    return report


def require_smoke(output_dir: Path, model: str, config: dict) -> None:
    token_path = Path(output_dir) / "runtime" / f"tokens-{model}.json"
    if not token_path.exists():
        raise StudyError("Exact tokenizer preflight is missing")
    tokens = read_json(token_path)
    if (tokens.get("status") != "ok" or tokens.get("effective_config_hash") != fingerprint(config["runtime"])
            or tokens.get("examples_sha256") != file_hash(Path(output_dir) / "data" / "examples.jsonl")):
        raise StudyError("Tokenizer preflight does not match current inputs and runtime")
    path = Path(output_dir) / "smoke" / f"{model}.json"
    if not path.exists():
        raise StudyError(f"Run smoke for {model} before formal inference")
    smoke = read_json(path)
    if not smoke.get("passed") or smoke.get("config_hash") != fingerprint(config):
        raise StudyError("No passed smoke for this exact data, prompt, runtime, and GPU configuration")


def run_model(project_dir: Path, output_dir: Path, model: str, *, budget=None) -> dict:
    from . import runtime
    from .analysis import fit_model
    from .data import load_examples, load_gold
    examples, gold = load_examples(output_dir), load_gold(output_dir)
    config = protocol_config(output_dir, model)
    require_smoke(output_dir, model, config)
    store = PredictionStore(output_dir, model, config)
    phase = f"predict/{model}"
    set_status(output_dir, phase, "running", config_hash=store.config_hash)
    try:
        with runtime.managed_services(project_dir, output_dir, [model]) as endpoints:
            client = ModelClient(model, endpoints[model])
            if model != "qwen":
                fit_examples = [e for e in examples if e["split"] in {"calibration", "validation"}]
                predict_examples(fit_examples, client, store, budget=budget)
                fit_model(examples, gold, list(store.rows.values()), model, output_dir)
            test_examples = [e for e in examples if e["split"] in {"test", "stress"}]
            result = predict_examples(test_examples, client, store, budget=budget)
        expected = 3450 if model == "qwen" else 6250
        if len(store.rows) != expected:
            raise StudyError(f"Expected {expected} terminal predictions for {model}, got {len(store.rows)}")
        set_status(output_dir, phase, "complete", terminal=len(store.rows), config_hash=store.config_hash,
                   failures=sum(r["status"] != "ok" for r in store.rows.values()))
        return result
    except BaseException as exc:
        store.sync()
        set_status(output_dir, phase, "blocked", error=str(exc), terminal=len(store.rows))
        raise


def analyze(output_dir: Path, *, bootstrap_reps=2000, random_reps=100) -> dict:
    from .analysis import evaluate_experiment
    from .data import load_examples, load_gold, verify_data
    verify_data(output_dir)
    predictions = {m: load_predictions(output_dir, m) for m in ("kev", "startlux", "qwen")}
    calibrations = {m: read_json(Path(output_dir) / "calibration" / f"{m}.json") for m in ("kev", "startlux")}
    result = evaluate_experiment(load_examples(output_dir), load_gold(output_dir), predictions,
                                 calibrations, output_dir, bootstrap_reps=bootstrap_reps, random_reps=random_reps)
    from .plots import plot_results
    plot_results(Path(output_dir) / "analysis")
    set_status(output_dir, "analysis", "complete", bootstrap_reps=bootstrap_reps, random_reps=random_reps)
    return result
