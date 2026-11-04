"""Small, serializable policies using only information available before fallback.

All logistic heads are fitted on calibration labels. Validation is used only to
freeze online thresholds and select a candidate; no test-time top-k operation is
hidden in the deployment API.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..analysis import rescale_probabilities
from ..common import StudyError

L2 = 0.01
FEATURE_VERSION = 1


def local_information(example: dict, row: dict) -> tuple[float, int] | None:
    """Return released confidence/label index, or None for a terminal failure."""
    if row.get("sample_id") != example.get("sample_id"):
        raise StudyError("Policy input and local prediction IDs differ")
    if row.get("status") == "error":
        return None
    if row.get("status") != "ok":
        raise StudyError("Policy requires a completed local prediction")
    labels = example["labels"]
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise StudyError("Local probability labels do not match the input")
    p = np.asarray([probabilities[label] for label in labels], dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)):
        raise StudyError("Invalid local probability vector")
    if row.get("prediction") not in labels:
        raise StudyError("Local label is outside the task schema")
    index = labels.index(row["prediction"])
    if p[index] < p.max() - 1e-8:
        raise StudyError("Local label disagrees with its probability maximum")
    return float(p.max()), index


def _features(example: dict, row: dict, *, include_length: bool) -> list[float]:
    info = local_information(example, row)
    if info is None:
        raise StudyError("A failed local prediction has no router features")
    confidence, index = info
    c = min(max(confidence, 1e-6), 1 - 1e-6)
    values = [math.log(c / (1 - c))]
    values += [float(index == j) for j in range(len(example["labels"]) - 1)]
    if include_length:
        values.append(math.log1p(len(example["state"])))
    return values


def _sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def fit_logistic(features, target, *, l2: float = L2) -> dict:
    """Deterministic ridge logistic regression with an unpenalized intercept."""
    x, y = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if (x.ndim != 2 or len(x) == 0 or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()
            or not math.isfinite(l2) or l2 <= 0):
        raise StudyError("Invalid logistic fitting data")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    result = {"feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
              "l2": l2, "n": len(y), "positives": int(y.sum())}
    if y.min() == y.max():
        return {**result, "kind": "constant", "probability": float((y.sum() + 1) / (len(y) + 2)),
                "note": "Beta(1,1) smoothing for a constant training target"}
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(y.mean() / (1 - y.mean()))
    penalty = np.full(len(beta), l2)
    penalty[0] = 0

    def objective(b):
        logits = design @ b
        return float(np.mean(np.logaddexp(0, logits) - y * logits) + .5 * np.sum(penalty * b * b))

    converged = False
    for iteration in range(100):
        p = _sigmoid(design @ beta)
        gradient = design.T @ (p - y) / len(y) + penalty * beta
        if np.max(np.abs(gradient)) < 1e-9:
            converged = True
            break
        hessian = design.T @ ((p * (1 - p))[:, None] * design) / len(y) + np.diag(penalty)
        try:
            direction = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-12, gradient)
        except np.linalg.LinAlgError as exc:
            raise StudyError("Logistic solver failed") from exc
        before, step = objective(beta), 1.0
        while step >= 2 ** -20 and objective(beta - step * direction) > before:
            step *= .5
        if step < 2 ** -20:
            raise StudyError("Logistic fitting line search did not converge")
        beta -= step * direction
    if not converged:
        raise StudyError("Logistic fitting exceeded its fixed iteration limit")
    return {**result, "kind": "logistic", "coefficients": beta.tolist(),
            "iterations": iteration, "objective": objective(beta)}


def _head_score(head: dict, values: list[float]) -> float:
    if head["kind"] == "constant":
        return float(head["probability"])
    x = (np.asarray(values) - np.asarray(head["feature_mean"])) / np.asarray(head["feature_scale"])
    beta = np.asarray(head["coefficients"])
    return float(_sigmoid(beta[0] + x @ beta[1:]))


def fit_candidates(examples: list[dict], gold: dict, local: dict, qwen: dict,
                   *, l2: float = L2) -> dict[str, dict]:
    if not examples or any(x.get("split") != "calibration" for x in examples):
        raise StudyError("Policy fitting requires calibration examples only")
    task, labels = examples[0]["task"], examples[0]["labels"]
    if any(x["task"] != task or x["labels"] != labels for x in examples):
        raise StudyError("Cannot fit one policy across task schemas")
    usable = [x for x in examples if local_information(x, local[x["sample_id"]]) is not None]
    if not usable:
        raise StudyError("No usable calibration predictions")
    targets = {
        name: [int(rows[x["sample_id"]]["status"] == "ok"
                   and rows[x["sample_id"]].get("prediction") == gold[x["sample_id"]]) for x in usable]
        for name, rows in (("local", local), ("qwen", qwen))
    }
    shared = {"task": task, "labels": list(labels), "feature_version": FEATURE_VERSION,
              "fit_split": "calibration", "fit_count": len(examples),
              "usable_fit_count": len(usable), "local_failures_excluded_from_features": len(examples) - len(usable)}
    class_features = [_features(x, local[x["sample_id"]], include_length=False) for x in usable]
    gain_features = [_features(x, local[x["sample_id"]], include_length=True) for x in usable]
    return {
        "confidence": {**shared, "family": "confidence"},
        "class_conditional": {**shared, "family": "class_conditional", "include_length": False,
                              "local_head": fit_logistic(class_features, targets["local"], l2=l2)},
        "gain": {**shared, "family": "gain", "include_length": True,
                 "local_head": fit_logistic(gain_features, targets["local"], l2=l2),
                 "qwen_head": fit_logistic(gain_features, targets["qwen"], l2=l2),
                 "target": "P(Qwen correct | local features) - P(local correct | local features)"},
    }


def score_policy(policy: dict, example: dict, local_row: dict) -> float:
    """Larger values mean greater preference for fallback; never reads gold/Qwen."""
    if example.get("task") != policy["task"] or example["labels"] != policy["labels"]:
        raise StudyError("Policy and request task schemas differ")
    info = local_information(example, local_row)
    if info is None:
        return math.inf
    confidence, _ = info
    family = policy["family"]
    if family == "confidence":
        return 1 - confidence
    if family == "original":
        vector = [[local_row["probabilities"][label] for label in policy["labels"]]]
        return 1 - float(rescale_probabilities(vector, policy["temperature"]).max())
    if family not in {"class_conditional", "gain"}:
        raise StudyError(f"Unknown policy family: {family}")
    features = _features(example, local_row, include_length=policy["include_length"])
    local_p = _head_score(policy["local_head"], features)
    return 1 - local_p if family == "class_conditional" else _head_score(policy["qwen_head"], features) - local_p


def should_defer(policy: dict, example: dict, local_row: dict) -> bool:
    score = score_policy(policy, example, local_row)
    if math.isinf(score):
        return True
    if policy["family"] == "original":
        gate = policy["gate"]
        return bool(gate["accept_none"] or 1 - score < gate["threshold"])
    decision = policy.get("decision")
    if not decision:
        raise StudyError("Policy has no frozen online threshold")
    if decision["mode"] == "none":
        return False
    if decision["mode"] == "all":
        return True
    if decision["mode"] != "threshold":
        raise StudyError("Unknown online decision mode")
    return score >= decision["threshold"]


def freeze_budget(policy: dict, examples: list[dict], local: dict, budget: float) -> dict:
    if not examples or any(x.get("split") != "validation" for x in examples):
        raise StudyError("Online threshold selection requires validation only")
    if not 0 <= budget <= 1 or not math.isfinite(budget):
        raise StudyError("Invalid fallback budget")
    scores = [score_policy(policy, x, local[x["sample_id"]]) for x in examples]
    mandatory = sum(math.isinf(s) for s in scores)
    target = math.floor(budget * len(examples) + 1e-12)
    allowed = max(0, target - mandatory)
    valid_scores = np.array([s for s in scores if math.isfinite(s)])
    used, threshold = 0, None
    for score in sorted(set(valid_scores.tolist()), reverse=True):
        count = int((valid_scores == score).sum())
        if used + count > allowed:
            break
        used += count
        threshold = score
    mode = "none" if used == 0 else "all" if used == len(valid_scores) else "threshold"
    frozen = copy.deepcopy(policy)
    frozen["decision"] = {"mode": mode, "threshold": threshold if mode == "threshold" else None,
                          "comparison": ">=", "target_budget": budget, "selection_split": "validation",
                          "selection_n": len(examples), "selection_deferrals": mandatory + used,
                          "mandatory_failures": mandatory, "budget_exceeded_by_failures": mandatory > target,
                          "ties": "defer_entire_equal_score_group_or_none"}
    return frozen
