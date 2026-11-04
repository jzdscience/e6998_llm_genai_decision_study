"""Portable, checksummed v1 evidence and pure CPU replay/threshold handlers.

The exporter reads frozen evidence only. It neither instantiates PredictionStore
nor invokes the v1 demo wrapper, which would write an operational lock.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from ...analysis import apply_gate, rescale_probabilities, routing_metrics, validate_calibration
from ...common import fingerprint
from ...storage import load_predictions

SCHEMA_VERSION = 1
MODELS = ("kev", "startlux", "qwen")
DECISION_MODELS = ("kev", "startlux")
CONDITIONS = ("released", "refitted")
CURATED_CASES = (
    ("Local decision helps", "clinc:test:2921:2740289483263792", "local_helps"),
    ("Fallback rescues", "squad:5a836496e60761001a2eb65a", "qwen_rescue"),
    ("Fallback harms", "squad:5acff4b177cf76001a686616", "fallback_harms"),
    ("Confident error", "squad:56de11154396321400ee25aa", "confident_miss"),
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_source(directory: Path, kind: str) -> dict:
    manifest = _read(directory / "bundle_manifest.json")
    if manifest.get("kind") != kind or manifest.get("complete") is not True:
        raise ValueError(f"Expected a completed {kind} evidence bundle")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Source bundle has no file checksums")
    for name, expected in files.items():
        relative = Path(name)
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            raise ValueError("Unsafe source manifest path")
        if not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing source evidence: {name}")
        if _digest(path) != expected:
            raise ValueError(f"Source evidence checksum mismatch: {name}")
    return manifest


def _public_prediction(row: dict) -> dict:
    # Explicit allowlist: raw runtime configs, host paths and service URLs never
    # enter the portable teaching artifact. These are normalized saved records.
    return {key: row.get(key) for key in (
        "prediction", "probabilities", "status", "config_hash", "model_revision",
    )}


def build(v1_dir: Path | str, benchmark_dir: Path | str, target_dir: Path | str) -> Path:
    """Validate source manifests and write a new portable bundle; return lab.json.

    target_dir must not exist and must be outside both evidence trees. The complete
    v1 prediction count is provenance; this UI includes only the 3,450 unique
    test/stress inputs with actual Qwen predictions, never missing fitting outputs.
    """
    source = Path(v1_dir).resolve()
    bench = Path(benchmark_dir).resolve()
    target = Path(target_dir).resolve()
    if target.exists():
        raise ValueError("Lab target already exists; choose a new directory")
    if target.is_relative_to(source) or target.is_relative_to(bench):
        raise ValueError("Lab output must be outside the frozen evidence directories")
    formal_manifest = _verify_source(source, "experiment")
    benchmark_manifest = _verify_source(bench, "benchmark")
    examples = [json.loads(line) for line in (source / "data/examples.jsonl").read_text().splitlines() if line]
    gold = _read(source / "data/gold.json")
    metrics = _read(source / "analysis/metrics.json")
    if metrics.get("evidence_type") != "model_predictions" or metrics.get("status") != "complete":
        raise ValueError("Lab requires completed real model evidence, not software fixtures")
    predictions = {model: load_predictions(source, model) for model in MODELS}
    for model, rows in predictions.items():
        if len(rows) != metrics["prediction_counts"][model] or any(r.get("is_test_fixture") for r in rows):
            raise ValueError("Prediction counts or evidence type differ from the formal report")
    indices = {model: {row["sample_id"]: row for row in rows} for model, rows in predictions.items()}
    calibrations = {model: _read(source / "calibration" / f"{model}.json") for model in DECISION_MODELS}
    for model in DECISION_MODELS:
        validate_calibration(examples, gold, predictions[model], calibrations[model], model)
    tasks, samples = {}, {}
    for ex in examples:
        if ex["split"] not in ("test", "stress"):
            continue
        sid, task = ex["sample_id"], ex["task"]
        if sid in samples or any(sid not in index for index in indices.values()):
            raise ValueError("Test/stress inputs must have unique, complete three-model predictions")
        definition = {key: ex[key] for key in ("labels", "criteria", "instructions")}
        if task in tasks and tasks[task] != definition:
            raise ValueError("Task definition changes within the saved workload")
        tasks[task] = definition
        samples[sid] = {
            "sample_id": sid, "task": task, "split": ex["split"], "group_id": ex["group_id"],
            "state": ex["state"], "gold": gold[sid],
            "predictions": {model: _public_prediction(index[sid]) for model, index in indices.items()},
        }
    settings = _read(bench / "benchmark/settings.json")
    if settings["config_hashes"] != metrics["config_hashes"]:
        raise ValueError("Benchmark and quality results use different model configurations")
    if any(settings["calibration_hashes"][m] != fingerprint(calibrations[m]) for m in DECISION_MODELS):
        raise ValueError("Benchmark and quality results use different frozen gates")
    cells = _read(bench / "benchmark/summary.json")
    public_cells = []
    allowed = ("task", "deployment", "concurrency", "repeat", "inputs", "unique_inputs", "cycles",
               "elapsed_seconds", "p50_ms", "p95_ms", "throughput_inputs_per_second",
               "successful_inputs_per_second", "failure_rate", "accuracy", "fallback_fraction")
    for cell in cells:
        if cell["settings_hash"] != fingerprint(settings):
            raise ValueError("Benchmark cell settings mismatch")
        public_cells.append({**{key: cell[key] for key in allowed}, "workload_hash": fingerprint(cell["workload_ids"])})
    payload = {
        "schema_version": SCHEMA_VERSION, "kind": "decision-lab", "mode": "recorded",
        "evidence_type": "model_predictions", "tasks": tasks, "samples": samples,
        "calibrations": calibrations, "metrics": metrics, "benchmark_cells": public_cells,
        "cases": [{"label": label, "sample_id": sid, "category": category, "model": "startlux"}
                  for label, sid, category in CURATED_CASES if sid in samples],
        "provenance": {
            "formal_manifest_sha256": _digest(source / "bundle_manifest.json"),
            "benchmark_manifest_sha256": _digest(bench / "bundle_manifest.json"),
            "formal_created_at": formal_manifest["created_at"],
            "benchmark_created_at": benchmark_manifest["created_at"],
            "config_hashes": metrics["config_hashes"],
            "benchmark_settings_hash": fingerprint(settings),
            "selection": "Four outcome-selected illustrative cases; not an unbiased sample.",
        },
    }
    _validate_payload(payload)
    # Cross-check the exact workload semantics and frozen gates before exporting.
    for task, workloads in metrics["datasets"].items():
        for workload, original in workloads.items():
            for model in DECISION_MODELS:
                for condition in CONDITIONS:
                    observed = sandbox(payload, model, task, workload, condition=condition)
                    expected = original["routes"][f"{model}/{condition}"]
                    for field in ("n", "accepted", "accepted_errors", "coverage", "selective_risk", "deferral_count", "cascade_accuracy"):
                        a, b = observed[field], expected[field]
                        if a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, abs_tol=1e-12)):
                            raise ValueError(f"Frozen metric mismatch: {task}/{workload}/{model}/{condition}/{field}")
    target.mkdir(parents=True)
    path = target / "lab.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "decision-lab", "files": {"lab.json": _digest(path)}}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _validate_payload(data: dict) -> None:
    if data.get("schema_version") != SCHEMA_VERSION or data.get("kind") != "decision-lab" or data.get("mode") != "recorded":
        raise ValueError("Unsupported Decision Lab schema or mode")
    if data.get("evidence_type") != "model_predictions":
        raise ValueError("Only archived real-model evidence can be displayed")
    if set(data["tasks"]) != {"clinc", "squad"} or set(data["calibrations"]) != set(DECISION_MODELS):
        raise ValueError("Lab requires both v1 tasks and frozen decision-model calibrations")
    if (not isinstance(data["provenance"], dict) or not isinstance(data["metrics"], dict)
            or not isinstance(data["benchmark_cells"], list) or not data["benchmark_cells"]):
        raise ValueError("Missing source identity, formal metrics or measured performance")
    samples = data.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError("Lab bundle has no saved examples")
    for sid, sample in samples.items():
        if sample["sample_id"] != sid or sample["split"] not in ("test", "stress") or sample["task"] not in data["tasks"]:
            raise ValueError("Invalid saved example identity")
        labels = data["tasks"][sample["task"]]["labels"]
        if sample["gold"] not in labels or not isinstance(sample["state"], str) or set(sample["predictions"]) != set(MODELS):
            raise ValueError("Incomplete saved example or predictions")
        for model, prediction in sample["predictions"].items():
            if prediction["status"] not in ("ok", "error"):
                raise ValueError("Unknown saved prediction status")
            if prediction["status"] == "ok" and prediction["prediction"] not in labels:
                raise ValueError("Saved prediction has unknown label")
            probabilities = prediction.get("probabilities")
            if model == "qwen" and probabilities is not None:
                raise ValueError("This Qwen evidence contract does not provide probabilities")
            if model in DECISION_MODELS and prediction["status"] == "ok":
                if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
                    raise ValueError("Missing or inconsistent decision probabilities")
                if any(isinstance(p, bool) for p in probabilities.values()):
                    raise ValueError("Boolean probability")
                rescale_probabilities([probabilities[label] for label in labels], 1.0)
    for model in DECISION_MODELS:
        for task in data["tasks"]:
            cal = data["calibrations"][model]["tasks"][task]
            temperature = cal["temperature"]["temperature"]
            if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Invalid archived temperature")
            for condition in CONDITIONS:
                apply_gate(np.array([0.5]), np.array([True]), cal["gates"][condition])
    for case in data.get("cases", []):
        if case["sample_id"] not in samples or case["model"] not in DECISION_MODELS:
            raise ValueError("Curated case is not in the saved evidence")
    deployments = {"kev_direct", "startlux_direct", "qwen_direct", "kev_to_qwen", "startlux_to_qwen"}
    seen = set()
    for cell in data["benchmark_cells"]:
        key = (cell["task"], cell["deployment"], cell["concurrency"], cell["repeat"])
        if (key in seen or cell["task"] not in data["tasks"] or cell["deployment"] not in deployments
                or cell["concurrency"] not in (1, 8) or cell["repeat"] not in (0, 1, 2)):
            raise ValueError("Invalid or repeated measured performance cell")
        seen.add(key)
        for field in ("p50_ms", "p95_ms", "throughput_inputs_per_second", "successful_inputs_per_second", "elapsed_seconds"):
            value = cell[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid measured performance value")
        for field in ("failure_rate", "accuracy", "fallback_fraction"):
            value = cell[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid measured performance fraction")


def load_bundle(bundle_dir: Path | str) -> dict:
    """Verify the portable file's checksum and schema before serving it."""
    directory = Path(bundle_dir).resolve()
    if directory.name == "lab.json" and directory.is_file():
        directory = directory.parent
    manifest = _read(directory / "manifest.json")
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != "decision-lab"
            or set(manifest.get("files", {})) != {"lab.json"}):
        raise ValueError("Invalid Decision Lab checksum manifest")
    path = directory / "lab.json"
    if path.is_symlink() or _digest(path) != manifest["files"]["lab.json"]:
        raise ValueError("Decision Lab checksum mismatch")
    data = _read(path)
    try:
        _validate_payload(data)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Incomplete Decision Lab payload") from exc
    return data


def _policy(data: dict, model: str, task: str, condition: str) -> tuple[float, dict]:
    if model not in DECISION_MODELS or task not in data["tasks"] or condition not in CONDITIONS:
        raise ValueError("Choose a saved task, Kev/StartLux, and released/refitted probabilities")
    cal = data["calibrations"][model]["tasks"][task]
    return (1.0 if condition == "released" else cal["temperature"]["temperature"], dict(cal["gates"][condition]))


def _probabilities(data: dict, sample: dict, model: str, temperature: float) -> dict | None:
    row = sample["predictions"][model]
    if row["status"] != "ok":
        return None
    labels = data["tasks"][sample["task"]]["labels"]
    values = np.array([row["probabilities"][label] for label in labels], dtype=float)
    values /= values.sum()
    transformed = rescale_probabilities(values, temperature)
    return dict(zip(labels, map(float, transformed)))


def sample_replay(data: dict, model: str, sample_id: str, condition: str = "refitted") -> dict:
    """Reconstruct an archived frozen decision; this is not a model invocation."""
    if sample_id not in data["samples"]:
        raise ValueError("Unknown sample ID; choose an example from the saved evidence")
    sample = data["samples"][sample_id]
    temperature, gate = _policy(data, model, sample["task"], condition)
    probabilities = _probabilities(data, sample, model, temperature)
    confidence = max(probabilities.values()) if probabilities else None
    accepted = bool(apply_gate([confidence if confidence is not None else np.nan], [probabilities is not None], gate)[0])
    local, fallback = sample["predictions"][model], sample["predictions"]["qwen"]
    final = local if accepted else fallback
    return {
        "mode": "recorded_replay", "sample_id": sample_id, "task": sample["task"],
        "state": sample["state"], "gold": sample["gold"], "model": model,
        "condition": condition, "temperature": temperature, "gate": gate,
        "probabilities": probabilities, "confidence": confidence,
        "accepted": accepted, "deferred": not accepted,
        "local_prediction": local["prediction"], "qwen_prediction": fallback["prediction"],
        "qwen_probabilities": None, "qwen_used_by_policy": not accepted,
        "final_prediction": final["prediction"], "status": final["status"],
        "correct": final["status"] == "ok" and final["prediction"] == sample["gold"],
        "source_predictions": sample["predictions"],
        "notice": "Reconstructed from saved predictions. Qwen shown on accepted cases is a separately collected baseline; no model runs or live timing.",
    }


def workload_samples(data: dict, task: str, workload: str = "main") -> list[dict]:
    if task not in data["tasks"] or workload not in ("main", "stress") or (task != "clinc" and workload != "main"):
        raise ValueError("Choose CLINC main/stress or SQuAD main")
    rows = [sample for sample in data["samples"].values() if sample["task"] == task]
    if workload == "main":
        return [sample for sample in rows if sample["split"] == "test"]
    return [sample for sample in rows if sample["split"] == "stress" or (sample["split"] == "test" and sample["gold"] != "oos")]


def sandbox(data: dict, model: str, task: str, workload: str = "main", threshold: float | None = None,
            condition: str = "refitted", accept_none: bool = False) -> dict:
    """Calculate quality only. A changed threshold never receives invented speed."""
    temperature, frozen_gate = _policy(data, model, task, condition)
    gate = dict(frozen_gate)
    if threshold is not None:
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be a finite number between 0 and 1")
        gate = {"threshold": float(threshold), "accept_none": bool(accept_none)}
    elif accept_none:
        gate = {"threshold": None, "accept_none": True}
    rows = workload_samples(data, task, workload)
    if not rows:
        raise ValueError("No saved examples in this workload")
    labels = data["tasks"][task]["labels"]
    valid = np.array([sample["predictions"][model]["status"] == "ok" for sample in rows])
    confidence = np.full(len(rows), np.nan)
    if valid.any():
        p = np.array([[sample["predictions"][model]["probabilities"][label] for label in labels]
                      for sample, ok in zip(rows, valid) if ok], dtype=float)
        p /= p.sum(axis=1, keepdims=True)
        confidence[valid] = rescale_probabilities(p, temperature).max(axis=1)
    accepted = apply_gate(confidence, valid, gate)
    local_correct = np.array([s["predictions"][model]["status"] == "ok" and s["predictions"][model]["prediction"] == s["gold"] for s in rows])
    qwen_correct = np.array([s["predictions"]["qwen"]["status"] == "ok" and s["predictions"]["qwen"]["prediction"] == s["gold"] for s in rows])
    metrics, _ = routing_metrics(local_correct, qwen_correct, accepted)
    deferred = ~accepted
    metrics.update({
        "model": model, "task": task, "workload": workload, "condition": condition,
        "temperature": temperature, "gate": gate, "frozen_gate": frozen_gate,
        "matches_frozen_gate": gate.get("threshold") == frozen_gate.get("threshold") and gate.get("accept_none", False) == frozen_gate.get("accept_none", False),
        "local_accuracy": float(local_correct.mean()), "qwen_accuracy": float(qwen_correct.mean()),
        "rescued": int((deferred & ~local_correct & qwen_correct).sum()),
        "harmed": int((deferred & local_correct & ~qwen_correct).sum()),
        "mode": "exploratory_saved_predictions", "performance_estimate": None,
        "notice": "Previously viewed data; threshold exploration does not refit or change the frozen study. No performance interpolation is available.",
    })
    metrics["net_correct"] = metrics["rescued"] - metrics["harmed"]
    return metrics


def measured_performance(data: dict, task: str, concurrency: int) -> list[dict]:
    if task not in data["tasks"] or concurrency not in (1, 8):
        raise ValueError("Measured settings are CLINC/SQuAD and concurrency 1/8")
    cells = [row for row in data["benchmark_cells"] if row["task"] == task and row["concurrency"] == concurrency]
    result = []
    for deployment in sorted({row["deployment"] for row in cells}):
        selected = [row for row in cells if row["deployment"] == deployment]
        result.append({
            "deployment": deployment, "rounds": len(selected), "unique_inputs": selected[0]["unique_inputs"],
            "throughput": mean(row["successful_inputs_per_second"] for row in selected),
            "p50_ms": mean(row["p50_ms"] for row in selected), "p95_ms": mean(row["p95_ms"] for row in selected),
            "fallback_fraction": mean(row["fallback_fraction"] for row in selected),
            "failure_rate": mean(row["failure_rate"] for row in selected),
        })
    return result
