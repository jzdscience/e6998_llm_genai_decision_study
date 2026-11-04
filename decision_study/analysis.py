"""Frozen-prediction analysis. No model calls, test-set fitting, or implicit imputation.

Every statistical comparison uses the same sampled observations for all systems.
The headline intervals condition on the frozen models, prompts, temperature and gate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
ANALYSIS_SOURCE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEED = 6998
EPSILON = 1e-12
DECISION_MODELS = ("kev", "startlux")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prediction_index(predictions, model=None):
    index = {}
    hashes = set()
    for row in predictions:
        sid = row["sample_id"]
        if sid in index:
            raise ValueError(f"Duplicate prediction: {sid}")
        if model is not None and row.get("model") != model:
            raise ValueError(f"Expected model {model}, found {row.get('model')}")
        if not row.get("config_hash"):
            raise ValueError(f"Missing config_hash: {sid}")
        hashes.add(row["config_hash"])
        index[sid] = row
    if len(hashes) != 1:
        raise ValueError(f"Expected one frozen config_hash per model, got {len(hashes)}")
    return index, next(iter(hashes))


def _require_rows(examples, index):
    missing = [x["sample_id"] for x in examples if x["sample_id"] not in index]
    if missing:
        raise ValueError(f"Incomplete predictions: {len(missing)} missing; first {missing[:5]}")


def _semantic_prediction_hash(examples, index):
    fields = ("sample_id", "model", "config_hash", "status", "prediction", "probabilities", "is_test_fixture")
    return _hash([{k: index[x["sample_id"]].get(k) for k in fields}
                  for x in sorted(examples, key=lambda x: x["sample_id"])])


def _input_hash(examples, gold):
    return _hash([{"example": x, "gold": gold[x["sample_id"]]}
                  for x in sorted(examples, key=lambda x: x["sample_id"])])


def _arrays(examples, gold, index, require_probabilities=True):
    if not examples:
        raise ValueError("No examples")
    labels = list(examples[0]["labels"])
    n = len(examples)
    p = np.full((n, len(labels)), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    predictions = np.full(n, -1, dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    failures = {}
    for i, ex in enumerate(examples):
        if list(ex["labels"]) != labels:
            raise ValueError("Inconsistent label order within task")
        sid = ex["sample_id"]
        if gold.get(sid) not in labels:
            raise ValueError(f"Missing or unknown gold label: {sid}")
        y[i] = labels.index(gold[sid])
        row = index[sid]
        reason = None
        if row.get("status") != "ok":
            error = row.get("error")
            reason = str(row.get("error_type") or (error.get("type") if isinstance(error, dict) else None) or "runtime_error")
        elif row.get("prediction") not in labels:
            reason = "invalid_label"
        else:
            predictions[i] = labels.index(row["prediction"])
            if require_probabilities:
                probs = row.get("probabilities")
                try:
                    if not isinstance(probs, dict) or set(probs) != set(labels):
                        raise ValueError("label set")
                    if any(isinstance(probs[label], bool) for label in labels):
                        raise ValueError("boolean probability")
                    values = np.asarray([probs[label] for label in labels], dtype=np.float64)
                    if (not np.all(np.isfinite(values)) or np.any(values < 0)
                            or np.any(values > 1) or abs(values.sum() - 1) > 1e-5):
                        raise ValueError("probability domain")
                    values = values / values.sum()
                    if values[predictions[i]] < values.max() - 1e-8:
                        raise ValueError("prediction/probability disagreement")
                    p[i] = values
                except (ValueError, TypeError, OverflowError):
                    reason = "invalid_probabilities"
        if reason is None:
            valid[i] = True
        else:
            predictions[i] = -1
            failures[reason] = failures.get(reason, 0) + 1
    return {"p": p, "valid": valid, "pred": predictions, "y": y,
            "correct": valid & (predictions == y), "labels": labels,
            "failures": failures}


def rescale_probabilities(probability_matrix, temperature):
    """Apply an *additional* temperature to released probabilities, in float64."""
    p = np.asarray(probability_matrix, dtype=np.float64)
    if p.ndim not in (1, 2) or p.shape[-1] < 2:
        raise ValueError("Expected a class distribution or matrix with at least two classes")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=-1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probabilities must sum to one")
    z = np.log(np.clip(p, EPSILON, 1.0)) / float(temperature)
    z -= z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=-1, keepdims=True)


def fit_temperature(probability_matrix, gold_indices):
    """Bounded golden-section minimization of calibration NLL; T=1 is a candidate."""
    p = np.asarray(probability_matrix, dtype=np.float64)
    y = np.asarray(gold_indices, dtype=np.int64)
    if p.ndim != 2 or len(p) == 0 or len(y) != len(p):
        raise ValueError("Temperature fitting needs a non-empty aligned calibration matrix")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("Gold index outside class range")
    rescale_probabilities(p, 1.0)  # validate before optimization
    logp = np.log(np.clip(p, EPSILON, 1.0))

    def objective(log_temperature):
        z = logp / math.exp(log_temperature)
        maximum = z.max(axis=1)
        return float(np.mean(maximum + np.log(np.exp(z - maximum[:, None]).sum(axis=1))
                             - z[np.arange(len(y)), y]))

    lower, upper = math.log(0.05), math.log(20.0)
    lo, hi = lower, upper
    ratio = (math.sqrt(5) - 1) / 2
    left, right = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fl, fr = objective(left), objective(right)
    for _ in range(160):
        if hi - lo < 1e-8:
            break
        if fl < fr:
            hi, right, fr = right, left, fl
            left = hi - ratio * (hi - lo)
            fl = objective(left)
        else:
            lo, left, fl = left, right, fr
            right = lo + ratio * (hi - lo)
            fr = objective(right)
    baseline = objective(0.0)
    candidate = min([lower, upper, (lo + hi) / 2], key=objective)
    best = objective(candidate)
    if baseline - best <= 1e-10:
        candidate, best = 0.0, baseline
    temperature = math.exp(candidate)
    return {"temperature": temperature, "nll_before": baseline, "nll_after": best,
            "n_valid": len(y), "bounds": [0.05, 20.0], "epsilon": EPSILON,
            "method": "bounded_log_temperature_golden_section_with_T1_candidate",
            "boundary_solution": abs(candidate - lower) < 1e-6 or abs(candidate - upper) < 1e-6}


def select_gate(confidences, correct, valid=None, max_risk=0.05, min_accepted=50):
    confidence = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if confidence.ndim != 1 or correct.shape != confidence.shape:
        raise ValueError("Gate arrays must be aligned vectors")
    if not 0 <= max_risk <= 1 or min_accepted < 1:
        raise ValueError("Invalid gate policy")
    valid = (np.ones(len(confidence), dtype=bool) if valid is None
             else np.asarray(valid, dtype=bool))
    if valid.shape != confidence.shape:
        raise ValueError("Invalid validity mask")
    if np.any(valid & (~np.isfinite(confidence) | (confidence < 0) | (confidence > 1))):
        raise ValueError("Valid rows require confidence in [0, 1]")
    chosen = {"threshold": None, "accept_none": True, "validation_accepted": 0,
              "validation_errors": 0, "validation_risk": None,
              "validation_coverage": 0.0, "validation_n": len(confidence),
              "max_empirical_risk": max_risk, "min_accepted": min_accepted,
              "risk_guarantee": False, "ties": "accept_entire_equal_score_group"}
    order = np.flatnonzero(valid)
    order = order[np.argsort(-confidence[order], kind="stable")]
    errors = 0
    for j, idx in enumerate(order):
        errors += int(not correct[idx])
        count = j + 1
        if count < len(order) and confidence[order[j + 1]] == confidence[idx]:
            continue
        if count >= min_accepted and errors / count <= max_risk + 1e-15:
            chosen.update(threshold=float(confidence[idx]), accept_none=False,
                          validation_accepted=count, validation_errors=errors,
                          validation_risk=errors / count,
                          validation_coverage=count / len(confidence))
    return chosen


def apply_gate(confidences, valid, gate):
    confidence = np.asarray(confidences, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if confidence.shape != valid.shape:
        raise ValueError("Confidence/valid mask mismatch")
    if gate.get("accept_none"):
        return np.zeros(confidence.shape, dtype=bool)
    threshold = gate.get("threshold")
    if threshold is None or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen gate")
    return valid & np.isfinite(confidence) & (confidence >= threshold)


def fit_model(examples, gold, predictions, model, output_dir):
    if model not in DECISION_MODELS:
        raise ValueError("Only decision models receive a fitted temperature/gate")
    index, config_hash = _prediction_index(predictions, model)
    fitting = [x for x in examples if x["split"] in ("calibration", "validation")]
    unexpected = set(index) - {x["sample_id"] for x in examples}
    if unexpected:
        raise ValueError(f"Unexpected prediction IDs for {model}: {sorted(unexpected)[:5]}")
    _require_rows(fitting, index)
    result = {"schema_version": SCHEMA_VERSION, "analysis_source_hash": ANALYSIS_SOURCE_HASH, "model": model, "config_hash": config_hash,
              "prediction_hash": _semantic_prediction_hash(fitting, index),
              "input_hash": _input_hash(fitting, gold), "tasks": {},
              "note": "Additional temperature on released probabilities. No weight training. No risk guarantee."}
    for task in sorted({x["task"] for x in fitting}):
        cal = [x for x in fitting if x["task"] == task and x["split"] == "calibration"]
        val = [x for x in fitting if x["task"] == task and x["split"] == "validation"]
        if not cal or not val:
            raise ValueError(f"Missing calibration or validation split for {task}")
        ca, va = _arrays(cal, gold, index), _arrays(val, gold, index)
        fitted = fit_temperature(ca["p"][ca["valid"]], ca["y"][ca["valid"]])
        gates = {}
        for condition, temperature in (("released", 1.0), ("refitted", fitted["temperature"])):
            confidence = np.full(len(val), np.nan)
            confidence[va["valid"]] = rescale_probabilities(va["p"][va["valid"]], temperature).max(axis=1)
            gates[condition] = select_gate(confidence, va["correct"], va["valid"])
        result["tasks"][task] = {"temperature": fitted, "gates": gates,
                                   "calibration_count": len(cal), "validation_count": len(val),
                                   "calibration_valid": int(ca["valid"].sum()),
                                   "validation_valid": int(va["valid"].sum()),
                                   "calibration_failures": ca["failures"],
                                   "validation_failures": va["failures"]}
    result["artifact_hash"] = _hash(result)
    _write_json(Path(output_dir) / "calibration" / f"{model}.json", result)
    return result



def validate_calibration(examples, gold, predictions, artifact, model):
    """Validate a frozen gate before offline, replay, live, or benchmark use.

    This is cheap provenance validation only: it never refits or inspects test
    correctness. Artifact self-hashing detects accidental edits, not adversaries
    capable of replacing both artifact and checksum.
    """
    if model not in DECISION_MODELS or artifact.get("model") != model:
        raise ValueError(f"Calibration/model mismatch for {model}")
    if artifact.get("analysis_source_hash") != ANALYSIS_SOURCE_HASH:
        raise ValueError(f"Calibration/analysis source mismatch for {model}; refit from saved calibration/validation predictions")
    content = {k: v for k, v in artifact.items() if k != "artifact_hash"}
    if not artifact.get("artifact_hash") or artifact["artifact_hash"] != _hash(content):
        raise ValueError(f"Calibration artifact checksum mismatch for {model}")
    if len({x["sample_id"] for x in examples}) != len(examples):
        raise ValueError("Duplicate sample_id in inputs")
    index, config_hash = _prediction_index(predictions, model)
    unexpected = set(index) - {x["sample_id"] for x in examples}
    if unexpected:
        raise ValueError(f"Unexpected prediction IDs for {model}: {sorted(unexpected)[:5]}")
    fitting = [x for x in examples if x["split"] in ("calibration", "validation")]
    _require_rows(fitting, index)
    if artifact["config_hash"] != config_hash:
        raise ValueError(f"Calibration/config mismatch for {model}")
    if artifact["input_hash"] != _input_hash(fitting, gold):
        raise ValueError(f"Calibration/input mismatch for {model}")
    if artifact["prediction_hash"] != _semantic_prediction_hash(fitting, index):
        raise ValueError(f"Calibration/prediction mismatch for {model}")
    if set(artifact.get("tasks", {})) != {x["task"] for x in fitting}:
        raise ValueError(f"Calibration tasks mismatch for {model}")


def probability_metrics(p, y, bins=15, prediction_indices=None):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.int64)
    if len(p) == 0:
        return {"n_valid": 0, "nll": None, "brier": None, "ece": None, "bins": []}
    p = rescale_probabilities(p, 1.0)
    predicted = p.argmax(axis=1) if prediction_indices is None else np.asarray(prediction_indices, dtype=np.int64)
    if predicted.shape != y.shape or np.any(predicted < 0) or np.any(predicted >= p.shape[1]):
        raise ValueError("Invalid prediction indices for probability metrics")
    confidence = p.max(axis=1)
    correct = predicted == y
    targets = np.eye(p.shape[1], dtype=np.float64)[y]
    ece = 0.0
    reliability = []
    bin_id = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    for b in range(bins):
        mask = bin_id == b
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(accuracy - mean_confidence)
        reliability.append({"lower": b / bins, "upper": (b + 1) / bins, "n": count,
                            "accuracy": accuracy, "confidence": mean_confidence})
    return {"n_valid": len(y), "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], EPSILON, 1)).mean()),
            "brier": float(np.square(p - targets).sum(axis=1).mean()),
            "ece": ece, "ece_definition": "15 equal-width top-label bins; final bin includes 1",
            "brier_definition": "mean over cases of sum over classes of squared probability error",
            "bins": reliability}


def classification_metrics(y, pred, labels):
    y, pred = np.asarray(y), np.asarray(pred)
    matrix = np.zeros((len(labels), len(labels) + 1), dtype=np.int64)
    for actual, guess in zip(y, pred):
        matrix[actual, guess if guess >= 0 else len(labels)] += 1
    by_class = {}
    f1s = []
    for j, label in enumerate(labels):
        tp, support, predicted = int(matrix[j, j]), int(matrix[j].sum()), int(matrix[:, j].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": support, "predicted": predicted,
                           "precision_defined": predicted > 0, "recall_defined": support > 0}
        f1s.append(f1)
    return {"n": len(y), "accuracy": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1s)),
            "failure_rate": float(np.mean(pred < 0)), "per_class": by_class,
            "confusion": {"true_labels": labels, "prediction_labels": labels + ["__failure__"],
                          "counts": matrix.tolist()}}


def routing_metrics(local_correct, qwen_correct, accepted):
    local_correct = np.asarray(local_correct, dtype=bool)
    qwen_correct = np.asarray(qwen_correct, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool)
    final_correct = np.where(accepted, local_correct, qwen_correct)
    n_accepted = int(accepted.sum())
    errors = int((accepted & ~local_correct).sum())
    risk = errors / n_accepted if n_accepted else None
    return {"n": len(accepted), "accepted": n_accepted, "accepted_errors": errors,
            "coverage": float(accepted.mean()), "selective_risk": risk,
            "deferral_count": int((~accepted).sum()), "fallback_fraction": float((~accepted).mean()),
            "cascade_accuracy": float(final_correct.mean()),
            "target_miss": risk > 0.05 if risk is not None else None,
            "risk_guarantee": False}, final_correct


def routing_controls(local_correct, qwen_correct, mandatory, deferral_count, random_reps=100, seed=SEED):
    """Matched exact quota, including fixed mandatory deferrals; oracle uses actual gain."""
    local = np.asarray(local_correct, dtype=bool)
    qwen = np.asarray(qwen_correct, dtype=bool)
    mandatory = np.asarray(mandatory, dtype=bool)
    n = len(local)
    if qwen.shape != local.shape or mandatory.shape != local.shape:
        raise ValueError("Unaligned routing arrays")
    remaining = np.flatnonzero(~mandatory)
    extra = int(deferral_count) - int(mandatory.sum())
    if extra < 0 or extra > len(remaining) or random_reps < 1:
        raise ValueError("Invalid matched deferral quota/repetitions")
    rng = np.random.default_rng(seed)
    random_accuracies = []
    for _ in range(random_reps):
        deferred = mandatory.copy()
        if extra:
            deferred[rng.choice(remaining, size=extra, replace=False)] = True
        random_accuracies.append(float(np.where(deferred, qwen, local).mean()))
    gain = qwen.astype(np.int64) - local.astype(np.int64)
    order = remaining[np.argsort(-gain[remaining], kind="stable")]
    oracle_deferred = mandatory.copy()
    oracle_deferred[order[:extra]] = True
    oracle_correct = np.where(oracle_deferred, qwen, local)
    expected_correct = int(qwen[mandatory].sum()) + int(local[~mandatory].sum())
    if len(remaining):
        expected_correct += extra / len(remaining) * gain[remaining].sum()
    return {"deferral_count": int(deferral_count), "mandatory_deferrals": int(mandatory.sum()),
            "random": {"mean_accuracy": float(np.mean(random_accuracies)),
                       "expected_accuracy": float(expected_correct / n),
                       "sd_accuracy": float(np.std(random_accuracies, ddof=1)) if random_reps > 1 else 0.0,
                       "randomization_range_95": np.quantile(random_accuracies, [.025, .975]).tolist(),
                       "repetitions": random_reps, "seed": seed,
                       "note": "Randomization variability, not a dataset confidence interval or a lower bound"},
            "oracle": {"accuracy": float(oracle_correct.mean()), "quota": "exactly_k",
                       "note": "Hindsight gold-dependent upper bound for these predictions and mandatory deferrals",
                       "negative_gain_deferrals": int((oracle_deferred & ~mandatory & (gain < 0)).sum())}}


def risk_coverage_curve(confidence, correct, valid):
    confidence, correct, valid = np.asarray(confidence), np.asarray(correct, dtype=bool), np.asarray(valid, dtype=bool)
    order = np.flatnonzero(valid)
    order = order[np.argsort(-confidence[order], kind="stable")]
    curve = [{"threshold": None, "coverage": 0.0, "risk": None, "accepted": 0, "errors": 0}]
    errors = 0
    for j, idx in enumerate(order):
        errors += int(not correct[idx])
        count = j + 1
        if count < len(order) and confidence[order[count]] == confidence[idx]:
            continue
        curve.append({"threshold": float(confidence[idx]), "coverage": count / len(correct),
                      "risk": errors / count, "accepted": count, "errors": errors})
    return curve


def accuracy_deferral_curve(confidence, local, qwen, valid):
    local, qwen, valid = np.asarray(local, bool), np.asarray(qwen, bool), np.asarray(valid, bool)
    confidence = np.asarray(confidence)
    n = len(local)
    order = np.flatnonzero(valid)
    order = order[np.argsort(-confidence[order], kind="stable")]
    final_correct = int(qwen.sum())
    mandatory = ~valid
    gain = qwen.astype(np.int64) - local.astype(np.int64)
    optional_gains = np.sort(gain[valid])[::-1]
    prefix = np.concatenate(([0], optional_gains.cumsum()))
    base = int(qwen[mandatory].sum()) + int(local[valid].sum())

    def point(count):
        extra = int(valid.sum()) - count
        random = base + (extra / len(optional_gains) * optional_gains.sum() if len(optional_gains) else 0)
        return {"deferral_fraction": (n - count) / n, "deferral_count": n - count,
                "confidence_accuracy": final_correct / n,
                "random_expected_accuracy": float(random / n),
                "oracle_accuracy": float((base + prefix[extra]) / n)}

    curve = [point(0)]
    for j, idx in enumerate(order):
        final_correct += int(local[idx]) - int(qwen[idx])
        count = j + 1
        if count == len(order) or confidence[order[count]] != confidence[idx]:
            curve.append(point(count))
    return curve



def routing_reference_curve(local, qwen, valid):
    """Random/oracle references at every legal exact quota, independent of score ties."""
    local, qwen, valid = np.asarray(local, bool), np.asarray(qwen, bool), np.asarray(valid, bool)
    mandatory = ~valid
    gain = qwen.astype(np.int64) - local.astype(np.int64)
    optional = np.sort(gain[valid])[::-1]
    prefix = np.concatenate(([0], optional.cumsum()))
    base = int(qwen[mandatory].sum()) + int(local[valid].sum())
    n = len(local)
    return [{"deferral_count": int(mandatory.sum()) + k,
             "deferral_fraction": (int(mandatory.sum()) + k) / n,
             "random_expected_accuracy": float((base + (k / len(optional) * optional.sum() if len(optional) else 0)) / n),
             "oracle_accuracy": float((base + prefix[k]) / n)} for k in range(len(optional) + 1)]


def _resample_indices(examples, gold, rng):
    groups = {}
    if examples[0]["task"] == "squad":
        for i, ex in enumerate(examples):
            groups.setdefault(ex["group_id"], []).append(i)
        group_values = list(groups.values())
        chosen = rng.integers(0, len(group_values), size=len(group_values))
        return np.concatenate([group_values[j] for j in chosen])
    for i, ex in enumerate(examples):
        groups.setdefault(gold[ex["sample_id"]], []).append(i)
    return np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()])


def _interval(values, *, degenerate_reason=None):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    answer = {"lower": None, "upper": None, "level": .95, "finite_replicates": len(finite),
              "replicates": len(values), "method": "paired_percentile_bootstrap"}
    if degenerate_reason:
        answer["warning"] = degenerate_reason
    elif len(finite):
        answer.update(lower=float(np.quantile(finite, .025)), upper=float(np.quantile(finite, .975)))
        if np.ptp(finite) == 0:
            answer["warning"] = "degenerate_empirical_distribution_not_a_risk_guarantee"
        elif len(finite) < .95 * len(values):
            answer["warning"] = "many_bootstrap_samples_have_no_accepted_cases"
    else:
        answer["warning"] = "undefined_no_finite_bootstrap_replicates"
    return answer


def paired_bootstrap(examples, gold, conditions, repetitions=2000, seed=SEED):
    """Conditions have correct, optionally accepted and local_correct arrays."""
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    draws = {name: {"accuracy": np.empty(repetitions)} for name in conditions}
    for name, condition in conditions.items():
        if "accepted" in condition:
            draws[name].update(coverage=np.empty(repetitions), selective_risk=np.full(repetitions, np.nan))
    for r in range(repetitions):
        ids = _resample_indices(examples, gold, rng)
        for name, condition in conditions.items():
            draws[name]["accuracy"][r] = np.asarray(condition["correct"])[ids].mean()
            if "accepted" in condition:
                acc = np.asarray(condition["accepted"])[ids]
                draws[name]["coverage"][r] = acc.mean()
                if acc.any():
                    draws[name]["selective_risk"][r] = (~np.asarray(condition["local_correct"])[ids][acc]).mean()
    intervals = {}
    for name, metrics in draws.items():
        intervals[name] = {}
        for metric, samples in metrics.items():
            reason = None
            if metric == "selective_risk":
                acc = conditions[name]["accepted"]
                errors = int((acc & ~conditions[name]["local_correct"]).sum())
                if not acc.any():
                    reason = "no_accepted_cases"
                elif errors in (0, int(acc.sum())):
                    reason = "boundary_error_count_bootstrap_cannot_estimate_unseen_tail"
            interval = _interval(samples, degenerate_reason=reason)
            if metric == "selective_risk" and 0 < int(conditions[name]["accepted"].sum()) < 50:
                interval["small_accepted_sample"] = True
            intervals[name][metric] = interval
    differences = {}
    for name in conditions:
        if name == "direct/qwen":
            continue
        if "direct/qwen" in draws:
            differences[f"{name} - direct/qwen"] = _interval(draws[name]["accuracy"] - draws["direct/qwen"]["accuracy"])
        if name.startswith("cascade/"):
            local_name = "direct/" + name.split("/")[1]
            differences[f"{name} - {local_name}"] = _interval(draws[name]["accuracy"] - draws[local_name]["accuracy"])
    return {"unit": "article" if examples[0]["task"] == "squad" else "example_stratified_by_gold_class",
            "cluster_count": len({x["group_id"] for x in examples}) if examples[0]["task"] == "squad" else None,
            "seed": seed, "repetitions": repetitions, "intervals": intervals,
            "paired_accuracy_differences": differences,
            "note": "Conditional on frozen models/configuration/calibration/gates; no future selective-risk guarantee."}


def _dataset_analysis(examples, gold, indices, calibrations, repetitions, random_reps):
    task = examples[0]["task"]
    arrays = {model: _arrays(examples, gold, index, model != "qwen") for model, index in indices.items()}
    result = {"n": len(examples), "direct": {}, "probability": {}, "routes": {}, "complementarity": {}}
    curves = {"risk_coverage": {}, "accuracy_deferral": {}, "reliability": {}, "routing_reference": {}}
    conditions = {}
    for model, a in arrays.items():
        result["direct"][model] = classification_metrics(a["y"], a["pred"], a["labels"])
        result["direct"][model]["failure_types"] = a["failures"]
        rows = [indices[model][x["sample_id"]] for x in examples]
        result["direct"][model]["attempts"] = sum(max(1, len(r.get("attempts", []))) for r in rows)
        result["direct"][model]["retried_cases"] = sum(len(r.get("attempts", [])) > 1 for r in rows)
        conditions[f"direct/{model}"] = {"correct": a["correct"]}
    qwen = arrays["qwen"]["correct"]
    for model in DECISION_MODELS:
        a = arrays[model]
        result["complementarity"][model] = {
            "both_correct": int((a["correct"] & qwen).sum()),
            "local_only_correct": int((a["correct"] & ~qwen).sum()),
            "qwen_only_correct": int((~a["correct"] & qwen).sum()),
            "both_wrong": int((~a["correct"] & ~qwen).sum()), "n": len(examples)}
        curves["routing_reference"][model] = routing_reference_curve(a["correct"], qwen, a["valid"])
        fitted = calibrations[model]["tasks"][task]
        for condition, temperature in (("released", 1.0), ("refitted", fitted["temperature"]["temperature"])):
            name = f"{model}/{condition}"
            p = rescale_probabilities(a["p"][a["valid"]], temperature)
            probs = probability_metrics(p, a["y"][a["valid"]], prediction_indices=a["pred"][a["valid"]])
            probs.update(n_total=len(examples), invalid_count=int((~a["valid"]).sum()), temperature=temperature)
            result["probability"][name] = probs
            confidence = np.full(len(examples), np.nan)
            confidence[a["valid"]] = p.max(axis=1)
            accepted = apply_gate(confidence, a["valid"], fitted["gates"][condition])
            route, correct = routing_metrics(a["correct"], qwen, accepted)
            route["gate"] = fitted["gates"][condition]
            route["controls"] = routing_controls(a["correct"], qwen, ~a["valid"], route["deferral_count"], random_reps)
            result["routes"][name] = route
            conditions[f"cascade/{name}"] = {"correct": correct, "accepted": accepted,
                                              "local_correct": a["correct"]}
            curves["risk_coverage"][name] = risk_coverage_curve(confidence, a["correct"], a["valid"])
            curves["accuracy_deferral"][name] = accuracy_deferral_curve(confidence, a["correct"], qwen, a["valid"])
            curves["reliability"][name] = probs["bins"]
    result["uncertainty"] = paired_bootstrap(examples, gold, conditions, repetitions)
    return result, curves, conditions, arrays


def _stress_changes(main, stress, gold, main_conditions, stress_conditions, repetitions):
    """Same resampled banking observations in both workloads, separate OOS pools."""
    main_lookup = {x["sample_id"]: i for i, x in enumerate(main)}
    stress_lookup = {x["sample_id"]: i for i, x in enumerate(stress)}
    banking = [x for x in main if gold[x["sample_id"]] != "oos"]
    if {x["sample_id"] for x in banking} != {x["sample_id"] for x in stress if gold[x["sample_id"]] != "oos"}:
        raise ValueError("CLINC stress comparison must share exactly the main banking pool")
    main_oos = np.array([i for i, x in enumerate(main) if gold[x["sample_id"]] == "oos"])
    stress_oos = np.array([i for i, x in enumerate(stress) if gold[x["sample_id"]] == "oos"])
    if not len(main_oos) or not len(stress_oos):
        raise ValueError("Both stress comparison OOS pools must be non-empty")
    rng = np.random.default_rng(SEED + 1)
    metrics = {name: {"accuracy": [], **({"coverage": [], "selective_risk": []} if "accepted" in c else {})}
               for name, c in main_conditions.items()}
    for _ in range(repetitions):
        chosen = [banking[j]["sample_id"] for j in _resample_indices(banking, gold, rng)]
        m = np.concatenate(([main_lookup[s] for s in chosen], rng.choice(main_oos, len(main_oos), replace=True))).astype(int)
        s = np.concatenate(([stress_lookup[s] for s in chosen], rng.choice(stress_oos, len(stress_oos), replace=True))).astype(int)
        for name, c in main_conditions.items():
            d = stress_conditions[name]
            metrics[name]["accuracy"].append(float(d["correct"][s].mean() - c["correct"][m].mean()))
            if "accepted" in c:
                a, b = c["accepted"][m], d["accepted"][s]
                metrics[name]["coverage"].append(float(b.mean() - a.mean()))
                risk_m = (~c["local_correct"][m][a]).mean() if a.any() else np.nan
                risk_s = (~d["local_correct"][s][b]).mean() if b.any() else np.nan
                metrics[name]["selective_risk"].append(float(risk_s - risk_m))
    return {"direction": "stress_minus_main", "method": "paired bootstrap sharing banking draws",
            "note": "Workload shift includes OOS pool composition and prevalence; not a causal pure-prevalence effect.",
            "intervals": {name: {k: _interval(v) for k, v in values.items()} for name, values in metrics.items()}}


def _prevalence_curve(pool, gold, conditions):
    oos = np.array([gold[x["sample_id"]] == "oos" for x in pool])
    if not oos.any() or oos.all():
        raise ValueError("Prevalence curve needs both banking and OOS")
    result = {}
    for name, c in conditions.items():
        points = []
        for fraction in sorted(set(np.linspace(0, 1, 11).tolist() + [.25, .75])):
            weights = np.where(oos, fraction / oos.sum(), (1 - fraction) / (~oos).sum())
            point = {"oos_fraction": fraction, "accuracy": float(weights @ c["correct"])}
            if "accepted" in c:
                coverage = float(weights @ c["accepted"])
                point.update(coverage=coverage, selective_risk=float(weights @ (c["accepted"] & ~c["local_correct"])) / coverage if coverage else None)
            points.append(point)
        result[name] = points
    return {"banking_n": int((~oos).sum()), "oos_n": int(oos.sum()), "series": result,
            "note": "Reweighting fixed conditional pools with frozen gates; points are dependent, not fresh test sets."}


def _error_cases(examples, gold, indices, calibrations, limit=30):
    cases = []
    for task in sorted({x["task"] for x in examples}):
        test = [x for x in examples if x["task"] == task and x["split"] == "test"]
        qa = _arrays(test, gold, indices["qwen"], False)
        buckets = {}
        for model in DECISION_MODELS:
            a = _arrays(test, gold, indices[model])
            f = calibrations[model]["tasks"][task]
            confidence = np.full(len(test), np.nan)
            confidence[a["valid"]] = rescale_probabilities(a["p"][a["valid"]], f["temperature"]["temperature"]).max(axis=1)
            accepted = apply_gate(confidence, a["valid"], f["gates"]["refitted"])
            for i, ex in enumerate(test):
                lc, qc = bool(a["correct"][i]), bool(qa["correct"][i])
                if lc and qc:
                    continue
                category = "local_wrong_qwen_right" if not lc and qc else "local_right_qwen_wrong" if lc else "both_wrong"
                item = {"task": task, "sample_id": ex["sample_id"], "group_id": ex["group_id"],
                        "model": model, "category": category, "state": ex["state"],
                        "instructions": ex["instructions"], "gold": gold[ex["sample_id"]],
                        "local_prediction": indices[model][ex["sample_id"]].get("prediction"),
                        "qwen_prediction": indices["qwen"][ex["sample_id"]].get("prediction"),
                        "accepted": bool(accepted[i]), "refitted_confidence": confidence[i],
                        "official_oos": gold[ex["sample_id"]] == "oos",
                        "probabilities_released": indices[model][ex["sample_id"]].get("probabilities")}
                bucket = (model, category, "oos" if item["official_oos"] else "other")
                buckets.setdefault(bucket, []).append(item)
        for items in buckets.values():
            items.sort(key=lambda x: _hash(["6998-solo-v1", "error", x["model"], x["sample_id"]]))
        task_cases = []
        while len(task_cases) < limit and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(task_cases) < limit:
                    task_cases.append(buckets[key].pop(0))
        cases.extend(task_cases)
    return cases


def _write_summary_csv(directory, metrics):
    rows = []
    for task, workloads in metrics["datasets"].items():
        for workload, result in workloads.items():
            for model, values in result["direct"].items():
                rows.append({"task": task, "workload": workload, "system": f"direct/{model}",
                             "n": values["n"], "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                             "failure_rate": values["failure_rate"]})
            for name, values in result["routes"].items():
                rows.append({"task": task, "workload": workload, "system": f"cascade/{name}",
                             "n": values["n"], "accuracy": values["cascade_accuracy"],
                             "coverage": values["coverage"], "selective_risk": values["selective_risk"],
                             "accepted": values["accepted"], "accepted_errors": values["accepted_errors"],
                             "target_miss": values["target_miss"],
                             "random_accuracy": values["controls"]["random"]["mean_accuracy"],
                             "oracle_accuracy": values["controls"]["oracle"]["accuracy"]})
    for row in rows:
        row["evidence_type"] = metrics["evidence_type"]
    fields = sorted({k for row in rows for k in row})
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    probability_rows = []
    for task, workloads in metrics["datasets"].items():
        for workload, result in workloads.items():
            for name, values in result["probability"].items():
                probability_rows.append({"task": task, "workload": workload, "condition": name,
                                         **{k: values[k] for k in ("n_total", "n_valid", "invalid_count", "temperature", "nll", "brier", "ece")}})
    with (directory / "probability_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(probability_rows[0]))
        writer.writeheader()
        writer.writerows(probability_rows)


def evaluate_experiment(examples, gold, predictions_by_model, calibrations, output_dir,
                        bootstrap_reps=2000, random_reps=100):
    """Strict complete evaluation; absent cases are an error, not omitted denominators."""
    if set(predictions_by_model) != {"kev", "startlux", "qwen"}:
        raise ValueError("Full evaluation requires kev, startlux and qwen predictions")
    if set(calibrations) != set(DECISION_MODELS):
        raise ValueError("Both decision-model calibration artifacts are required")
    if len({x["sample_id"] for x in examples}) != len(examples):
        raise ValueError("Duplicate sample_id in inputs")
    indices, hashes = {}, {}
    fitting = [x for x in examples if x["split"] in ("calibration", "validation")]
    test = [x for x in examples if x["split"] in ("test", "stress")]
    for model, rows in predictions_by_model.items():
        indices[model], hashes[model] = _prediction_index(rows, model)
        _require_rows(test, indices[model])
        unexpected = set(indices[model]) - {x["sample_id"] for x in examples}
        if unexpected:
            raise ValueError(f"Unexpected prediction IDs for {model}: {sorted(unexpected)[:5]}")
        if model != "qwen":
            validate_calibration(examples, gold, rows, calibrations[model], model)
    fixture_flags = {bool(row.get("is_test_fixture", False)) for rows in predictions_by_model.values() for row in rows}
    if len(fixture_flags) > 1:
        raise ValueError("Cannot mix software test fixtures and model results")
    evidence_type = "software_test_fixture" if fixture_flags == {True} else "model_predictions"
    metrics = {"schema_version": SCHEMA_VERSION, "analysis_source_hash": ANALYSIS_SOURCE_HASH, "status": "complete", "evidence_type": evidence_type, "config_hashes": hashes,
               "input_hash": _input_hash(examples, gold), "prediction_counts": {k: len(v) for k, v in indices.items()},
               "datasets": {}, "protocol": {"bootstrap_reps": bootstrap_reps, "random_reps": random_reps,
               "probability_epsilon": EPSILON, "seed": SEED, "invalid_predictions": "count_as_incorrect",
               "probability_metrics": "valid_full_distributions_only; denominators_reported",
               "risk_target": "empirical_validation_5_percent_not_a_guarantee"}}
    curves, condition_cache, example_cache = {}, {}, {}
    for task in sorted({x["task"] for x in test}):
        workloads = {"main": [x for x in test if x["task"] == task and x["split"] == "test"]}
        if task == "clinc":
            workloads["stress"] = [x for x in workloads["main"] if gold[x["sample_id"]] != "oos"] + [x for x in test if x["task"] == task and x["split"] == "stress"]
        metrics["datasets"][task], curves[task] = {}, {}
        for workload, rows in workloads.items():
            result, chart, conditions, _ = _dataset_analysis(rows, gold, indices, calibrations, bootstrap_reps, random_reps)
            metrics["datasets"][task][workload] = result
            curves[task][workload] = chart
            condition_cache[(task, workload)] = conditions
            example_cache[(task, workload)] = rows
    if "clinc" in metrics["datasets"]:
        metrics["clinc_paired_stress_change"] = _stress_changes(example_cache[("clinc", "main")], example_cache[("clinc", "stress")], gold,
            condition_cache[("clinc", "main")], condition_cache[("clinc", "stress")], bootstrap_reps)
        pool = [x for x in test if x["task"] == "clinc"]
        # Build the full conditional pool once; no bootstrap/model execution is needed for its reweighting.
        arrays = {m: _arrays(pool, gold, indices[m], m != "qwen") for m in indices}
        conditions = {f"direct/{m}": {"correct": a["correct"]} for m, a in arrays.items()}
        for model in DECISION_MODELS:
            a = arrays[model]
            fitted = calibrations[model]["tasks"]["clinc"]
            for name, temperature in (("released", 1.0), ("refitted", fitted["temperature"]["temperature"])):
                confidence = np.full(len(pool), np.nan)
                confidence[a["valid"]] = rescale_probabilities(a["p"][a["valid"]], temperature).max(axis=1)
                accepted = apply_gate(confidence, a["valid"], fitted["gates"][name])
                conditions[f"cascade/{model}/{name}"] = {"correct": np.where(accepted, a["correct"], arrays["qwen"]["correct"]),
                    "accepted": accepted, "local_correct": a["correct"]}
        curves["clinc"]["prevalence"] = _prevalence_curve(pool, gold, conditions)
    output = Path(output_dir) / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "curves.json", curves)
    _write_summary_csv(output, metrics)
    cases = _error_cases(examples, gold, indices, calibrations)
    with (output / "error_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(_jsonable(case), ensure_ascii=False, allow_nan=False) + "\n")
    return _jsonable(metrics)
