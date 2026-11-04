"""V3 ANLI routers fitted on calibration and thresholded on validation only.

This module performs no I/O. Serving reads only the local prediction and public
input; fallback output and gold are used exclusively for calibration targets.
The primary family/back-model choice is specified by the study, never selected
here from validation accuracy. All serialized parameters are plain JSON values.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from ..common import StudyError

LABELS = ("entailment", "neutral", "contradiction")
L2 = 0.01
FEATURE_VERSION = 1
FAMILIES = ("confidence", "class_conditional", "fixed_gain", "positive_gain")
FEATURE_NAMES = ["confidence_logit", *[f"predicted_{label}" for label in LABELS], "log1p_state_chars"]


def _examples(examples: Sequence[dict], *, split: str | None = None) -> None:
    if not examples:
        raise StudyError("Examples must be nonempty")
    ids = [x.get("sample_id") for x in examples]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise StudyError("Example IDs must be nonempty and unique")
    for example in examples:
        if example.get("task") != "anli" or tuple(example.get("labels", ())) != LABELS:
            raise StudyError("Expected the frozen ANLI three-class schema")
        if not isinstance(example.get("state"), str):
            raise StudyError("Expected public text state")
        if split is not None and example.get("split") != split:
            raise StudyError(f"Expected {split} examples only")


def _row(example: dict, rows: Mapping[str, dict], *, model: str | None = None) -> dict:
    try:
        row = rows[example["sample_id"]]
    except KeyError as exc:
        raise StudyError("Missing aligned prediction") from exc
    if row.get("sample_id") != example["sample_id"]:
        raise StudyError("Prediction ID mismatch")
    if row.get("status") not in {"ok", "error"}:
        raise StudyError("Expected a terminal prediction status")
    if model is not None and row.get("model") != model:
        raise StudyError("Prediction model identity mismatch")
    for key in ("task", "split"):
        if key in row and row[key] != example.get(key):
            raise StudyError(f"Prediction {key} mismatch")
    if row["status"] == "ok" and row.get("prediction") not in LABELS:
        raise StudyError("Prediction outside the ANLI label schema")
    return row


def local_information(example: dict, row: dict) -> tuple[float, int] | None:
    """Validate released probabilities; errors have no router feature vector."""
    _examples([example])
    _row(example, {row.get("sample_id"): row}, model="startlux4b")
    if row["status"] == "error":
        return None
    probabilities = row.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
        raise StudyError("Local probability labels do not match ANLI")
    try:
        p = np.asarray([probabilities[label] for label in LABELS], dtype=float)
    except (ValueError, TypeError) as exc:
        raise StudyError("Invalid local probabilities") from exc
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)):
        raise StudyError("Invalid local probability vector")
    index = LABELS.index(row["prediction"])
    if p[index] < p.max() - 1e-8:
        raise StudyError("Local prediction disagrees with probability maximum")
    return float(p.max()), index


def feature_values(example: dict, row: dict, *, include_length: bool = True) -> list[float]:
    info = local_information(example, row)
    if info is None:
        raise StudyError("A failed prediction has no router features")
    confidence, index = info
    c = min(max(confidence, 1e-6), 1 - 1e-6)
    values = [math.log(c / (1 - c))]
    values.extend(float(index == j) for j in range(3))
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


def fit_policies(examples: Sequence[dict], localrows: Mapping[str, dict],
                 fallbackrows: Mapping[str, dict], gold: Mapping[str, str], *,
                 fallback_model: str, l2: float = L2) -> dict[str, dict]:
    """Fit four predeclared families on calibration correctness, without selection.

    Gold/row mappings may additionally contain validation IDs, but only the
    supplied calibration examples enter targets or feature standardization.
    Terminal local failures are excluded from head fitting and routed forcibly
    at deployment. A failed fallback counts as incorrect in its head target.
    """
    _examples(examples, split="calibration")
    if not isinstance(fallback_model, str) or not fallback_model or fallback_model == "startlux4b":
        raise StudyError("A distinct fallback model identity is required")
    usable, local_target, fallback_target = [], [], []
    for example in examples:
        sid = example["sample_id"]
        if gold.get(sid) not in LABELS:
            raise StudyError("Missing or invalid calibration gold")
        local = _row(example, localrows, model="startlux4b")
        fallback = _row(example, fallbackrows, model=fallback_model)
        if local_information(example, local) is None:
            continue
        usable.append(example)
        local_target.append(int(local["prediction"] == gold[sid]))
        fallback_target.append(int(fallback["status"] == "ok" and fallback["prediction"] == gold[sid]))
    if not usable:
        raise StudyError("No usable calibration local predictions")
    features = [feature_values(x, localrows[x["sample_id"]]) for x in usable]
    gain_local_head = fit_logistic(features, local_target, l2=l2)
    fallback_head = fit_logistic(features, fallback_target, l2=l2)
    shared = {"task": "anli", "labels": list(LABELS), "feature_version": FEATURE_VERSION,
              "fit_split": "calibration", "fit_count": len(examples), "usable_fit_count": len(usable),
              "local_failures_excluded_from_features": len(examples) - len(usable),
              "local_model": "startlux4b", "fallback_model": fallback_model}
    result = {"confidence": {**shared, "family": "confidence", "feature_names": ["confidence"]},
              "class_conditional": {**shared, "family": "class_conditional", "include_length": False,
                  "feature_names": FEATURE_NAMES[:-1],
                  "local_head": fit_logistic([x[:-1] for x in features], local_target, l2=l2)}}
    for family in ("fixed_gain", "positive_gain"):
        result[family] = {**shared, "family": family, "include_length": True,
                          "feature_names": list(FEATURE_NAMES),
                          "local_head": copy.deepcopy(gain_local_head),
                          "fallback_head": copy.deepcopy(fallback_head),
                          "target": "P(fallback correct | local features) - P(local correct | local features)",
                          "positive_only": family == "positive_gain"}
    return result


def score_policy(policy: dict, example: dict, localrow: dict) -> float:
    """Higher means more preference for fallback; no gold/fallback argument."""
    if policy.get("task") != "anli" or tuple(policy.get("labels", ())) != LABELS:
        raise StudyError("Policy schema differs from frozen ANLI schema")
    family = policy.get("family")
    if family not in FAMILIES:
        raise StudyError("Unknown V3 policy family")
    info = local_information(example, localrow)
    if info is None:
        return math.inf
    if family == "confidence":
        return 1 - info[0]
    include_length = family != "class_conditional"
    if policy.get("include_length") is not include_length:
        raise StudyError("Policy feature layout differs from its family")
    values = feature_values(example, localrow, include_length=include_length)
    local_p = _head_score(policy["local_head"], values)
    score = 1 - local_p if family == "class_conditional" else _head_score(policy["fallback_head"], values) - local_p
    if not math.isfinite(score):
        raise StudyError("Nonfinite policy score")
    return score


def route(policy: dict, example: dict, localrow: dict) -> bool:
    """Apply one frozen online decision; local failures override every budget.

    Holdout inputs are never ranked or batched to reproduce a nominal quota.
    Positive-gain routers keep BOTH strict positive utility and the frozen
    validation threshold, including if all validation candidates were selected.
    """
    decision = policy.get("decision")
    if not isinstance(decision, dict) or decision.get("mode") not in {"none", "threshold"}:
        raise StudyError("Policy needs a frozen online threshold")
    threshold = decision.get("threshold")
    if decision["mode"] == "threshold" and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise StudyError("Invalid frozen threshold")
    score = score_policy(policy, example, localrow)
    if math.isinf(score):
        return True
    if decision["mode"] == "none":
        return False
    if policy["family"] == "positive_gain" and score <= 0:
        return False
    return score >= threshold


def freeze_budget(policy: dict, examples: Sequence[dict], localrows: Mapping[str, dict],
                  budget: float = .1) -> dict:
    """Freeze a largest-first whole-tie threshold, counting forced failures.

    If the next highest tie group cannot fit, stop: skipping it to admit a lower
    group cannot be represented by a monotone online threshold. Positive gain
    excludes every score <=0. No label is read for threshold selection.
    """
    _examples(examples, split="validation")
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise StudyError("Invalid validation fallback budget")
    scores = [score_policy(policy, x, _row(x, localrows, model="startlux4b")) for x in examples]
    mandatory = sum(math.isinf(score) for score in scores)
    target = math.floor(budget * len(examples) + 1e-12)
    allowed = max(0, target - mandatory)
    eligible = [score for score in scores if math.isfinite(score)
                and (policy["family"] != "positive_gain" or score > 0)]
    used, threshold = 0, None
    for score, count in sorted(Counter(eligible).items(), reverse=True):
        if used + count > allowed:
            break
        used += count
        threshold = score
    frozen = copy.deepcopy(policy)
    frozen["decision"] = {"mode": "none" if used == 0 else "threshold", "threshold": threshold,
                          "comparison": ">=", "positive_score_required": policy["family"] == "positive_gain",
                          "target_budget": budget, "target_count": target, "selection_split": "validation",
                          "selection_n": len(examples), "selection_deferrals": mandatory + used,
                          "selection_optional_deferrals": used, "eligible_positive_or_unrestricted": len(eligible),
                          "mandatory_failures": mandatory, "budget_exceeded_by_failures": mandatory > target,
                          "ties": "defer_entire_equal_score_group_or_none",
                          "failure_rule": "force_fallback_even_if_budget_exceeded",
                          "none_reason": "no_eligible_group_fits" if used == 0 else None}
    return frozen
