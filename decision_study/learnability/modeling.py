"""Frozen ANLI routing recipe: deterministic NumPy fitting and post-freeze evaluation.

This module performs no file I/O. Fitting accepts only fitting/selection rows and
matching training gold. Scoring accepts public inputs/front predictions only;
evaluation gold enters only evaluate_suite(), which never fits or selects a model.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np

from ..common import StudyError, fingerprint

LABELS = ("entailment", "neutral", "contradiction")
FALLBACKS = ("qwen9b", "startlux9b")
ARMS = ("front", "text")
FIT_SIZES = (1800, 6000, 18000)
EMBEDDING_DIM = 1536
L2_GRID = (1.0, 0.1, 0.01)
QUOTA = 0.1
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261113
RANDOM_SEED = 20261112
RECIPE = {"id": "anli-learnability-numpy-gain-v1", "feature_names":
          [*[f"probability_{x}" for x in LABELS], *[f"predicted_{x}" for x in LABELS], "log1p_state_chars"],
          "text_dimension": EMBEDDING_DIM, "text": "normalized concat[u,v,abs(u-v),u*v]; frozen MiniLM field embeddings",
          "l2_grid": list(L2_GRID), "objective": "mean binary cross entropy + L2/2 times squared slopes; unpenalized intercept",
          "solver": "deterministic L-BFGS", "history": 10, "max_iterations": 1000,
          "gradient_tolerance": 1e-6, "armijo": 1e-4,
          "constant_target": "Jeffreys smoothing: (sum(y)+0.5)/(n+1)",
          "selection": "largest selection correct count at exact floor(0.1*N) quota; tie favors larger L2",
          "selection_threshold": "maximize selection net correct with positive predicted gain and <= floor(0.1*N) voluntary deferrals; tie favors fewer deferrals",
          "failure_rule": "front failures force deferral within quota; if failures exceed quota cell fails",
          "inference": "paired normalized-premise cluster bootstrap, quota recomputed in every draw; conditional on fitted models",
          "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
          "minimum_useful_difference_pp": 1.0, "minimum_effect_interpretation": "descriptive threshold only; no formal equivalence claim"}


def require(condition, message):
    if not condition:
        raise StudyError(message)


def _examples(examples, splits):
    examples = list(examples)
    ids = [r.get("sample_id") for r in examples]
    require(bool(examples) and all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids),
            "Examples need unique nonempty IDs")
    for row in examples:
        require(not {"gold", "answer", "correct", "target", "gold_label"}.intersection(row), "Example inputs must not contain correctness/gold fields")
        require(row.get("task") == "anli" and row.get("split") in splits and tuple(row.get("labels", ())) == LABELS,
                "ANLI example schema or permitted split differs")
        require(isinstance(row.get("state"), str) and isinstance(row.get("premise_hash"), str) and row["premise_hash"],
                "Examples need public state and normalized premise hash")
        require(row.get("round") in {"R1", "R2", "R3"}, "ANLI round differs")
    return examples, ids


def _gold(gold, ids):
    require(isinstance(gold, Mapping) and set(gold) == set(ids), "Gold IDs must match exactly the permitted role(s)")
    require(all(gold[sid] in LABELS for sid in ids), "Gold label differs from the fixed ANLI schema")


def _predictions(examples, predictions):
    require(isinstance(predictions, Mapping), "Predictions must be keyed by sample ID")
    selected = []
    for example in examples:
        row = predictions.get(example["sample_id"])
        require(isinstance(row, Mapping) and row.get("sample_id") == example["sample_id"]
                and row.get("status") in {"ok", "error"}, "Missing or invalid terminal prediction")
        if row["status"] == "ok":
            require(row.get("prediction") in LABELS, "Prediction has an invalid label")
        selected.append(row)
    return selected


def front_features(examples, front_rows):
    """Return the seven fixed features, failed-front mask and confidence."""
    predictions = _predictions(examples, front_rows)
    matrix = np.zeros((len(examples), 7), dtype=np.float64)
    failed = np.zeros(len(examples), dtype=bool)
    confidence = np.zeros(len(examples), dtype=np.float64)
    for i, (example, row) in enumerate(zip(examples, predictions)):
        if row["status"] == "error":
            failed[i] = True
            continue
        values = row.get("probabilities")
        require(isinstance(values, Mapping) and set(values) == set(LABELS), "Front probability labels differ")
        try:
            p = np.asarray([values[label] for label in LABELS], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise StudyError("Invalid front probabilities") from exc
        require(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                and abs(float(p.sum())-1) <= 1e-6, "Invalid front probabilities")
        index = LABELS.index(row["prediction"])
        require(p[index] >= p.max()-1e-8, "Front label disagrees with probability maximum")
        matrix[i, :3] = p
        matrix[i, 3+index] = 1
        matrix[i, 6] = math.log1p(len(example["state"]))
        confidence[i] = p.max()
    return matrix, failed, confidence


def feature_matrices(examples, front_rows, embeddings_by_id):
    front, failed, confidence = front_features(examples, front_rows)
    require(isinstance(embeddings_by_id, Mapping), "Embeddings must be keyed by ID")
    try:
        text = np.asarray([embeddings_by_id[r["sample_id"]] for r in examples], dtype=np.float64)
    except (TypeError, ValueError, KeyError) as exc:
        raise StudyError("Missing or invalid frozen text embeddings") from exc
    require(text.shape == (len(examples), EMBEDDING_DIM) and np.isfinite(text).all(), "Frozen embedding dimension/values differ")
    require(np.allclose(np.linalg.norm(text, axis=1), 1, atol=2e-5, rtol=0), "Text pair descriptors must be L2-normalized")
    return {"front": front, "text": np.column_stack([front, text])}, failed, confidence


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1/(1+np.exp(-z[positive]))
    e = np.exp(z[~positive])
    out[~positive] = e/(1+e)
    return out


def logistic_objective_gradient(beta, x, y, l2):
    """Average objective; x is already standardized. Intercept is beta[0]."""
    z = beta[0]+x @ beta[1:]
    residual = _sigmoid(z)-y
    value = float(np.mean(np.logaddexp(0, z)-y*z)+.5*l2*np.dot(beta[1:], beta[1:]))
    gradient = np.concatenate(([residual.mean()], x.T @ residual/len(y)+l2*beta[1:]))
    return value, gradient


def fit_head(features, target, l2):
    x, y = np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)
    require(x.ndim == 2 and len(x) and y.shape == (len(x),) and np.isfinite(x).all()
            and np.isin(y, [0, 1]).all() and l2 in L2_GRID, "Invalid logistic fitting data or regularization")
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    result = {"n": len(y), "positives": int(y.sum()), "mean": mean.tolist(), "scale": scale.tolist(), "l2": l2}
    if y.min() == y.max():
        return {**result, "kind": "constant", "probability": float((y.sum()+.5)/(len(y)+1)), "iterations": 0}
    x = np.ascontiguousarray((x-mean)/scale)
    beta = np.zeros(x.shape[1]+1)
    beta[0] = math.log(y.mean()/(1-y.mean()))
    value, gradient = logistic_objective_gradient(beta, x, y, l2)
    history = []
    for iteration in range(RECIPE["max_iterations"]+1):
        if np.max(np.abs(gradient)) <= RECIPE["gradient_tolerance"]:
            return {**result, "kind": "logistic", "coefficients": beta.tolist(), "iterations": iteration,
                    "objective": value, "gradient_inf_norm": float(np.max(np.abs(gradient)))}
        if iteration == RECIPE["max_iterations"]:
            break
        q = gradient.copy()
        alphas = []
        for delta, change, rho in reversed(history):
            alpha = rho*np.dot(delta, q)
            alphas.append(alpha)
            q -= alpha*change
        if history:
            delta, change, _ = history[-1]
            q *= np.dot(delta, change)/np.dot(change, change)
        for (delta, change, rho), alpha in zip(history, reversed(alphas)):
            q += delta*(alpha-rho*np.dot(change, q))
        direction = -q
        slope = float(np.dot(gradient, direction))
        if slope >= 0 or not math.isfinite(slope):
            direction = -gradient
            slope = -float(np.dot(gradient, gradient))
            history = []
        step, accepted = 1.0, False
        for _ in range(40):
            trial = beta+step*direction
            next_value, next_gradient = logistic_objective_gradient(trial, x, y, l2)
            if math.isfinite(next_value) and np.isfinite(next_gradient).all() and next_value <= value+RECIPE["armijo"]*step*slope:
                accepted = True
                break
            step *= .5
        require(accepted, "Fixed logistic line search failed; no unconverged model accepted")
        delta, change = trial-beta, next_gradient-gradient
        curvature = float(np.dot(delta, change))
        if curvature > 1e-12*np.linalg.norm(delta)*np.linalg.norm(change):
            history.append((delta, change, 1/curvature))
            history = history[-RECIPE["history"]:]
        beta, value, gradient = trial, next_value, next_gradient
    raise StudyError("Fixed logistic solver exceeded its iteration budget")


def score_head(head, features):
    x = np.asarray(features, dtype=np.float64)
    mean, scale = np.asarray(head["mean"]), np.asarray(head["scale"])
    require(x.ndim == 2 and mean.shape == scale.shape == (x.shape[1],) and np.isfinite(x).all()
            and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), "Saved head feature identity differs")
    if head["kind"] == "constant":
        p = head["probability"]
        require(isinstance(p, (float, int)) and math.isfinite(p) and 0 < p < 1, "Invalid constant head")
        return np.full(len(x), p, dtype=np.float64)
    require(head["kind"] == "logistic", "Unknown head kind")
    beta = np.asarray(head["coefficients"], dtype=np.float64)
    require(beta.shape == (x.shape[1]+1,) and np.isfinite(beta).all(), "Saved logistic coefficient identity differs")
    return _sigmoid(beta[0]+((x-mean)/scale) @ beta[1:])


def exact_quota(scores, sample_ids, forced=None):
    """Top-k with stable sample-ID ties; duplicate IDs are allowed in bootstrap draws."""
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(sample_ids, dtype=str)
    forced = np.zeros(len(ids), dtype=bool) if forced is None else np.asarray(forced, dtype=bool)
    require(scores.shape == ids.shape == forced.shape and len(ids) and np.isfinite(scores).all(), "Invalid quota scores")
    k = int(math.floor(QUOTA*len(ids)))
    require(int(forced.sum()) <= k, "Forced front failures exceed the exact routing quota")
    chosen = forced.copy()
    remaining = np.flatnonzero(~forced)
    order = np.lexsort((remaining, ids[remaining], -scores[remaining]))
    chosen[remaining[order[:k-int(forced.sum())]]] = True
    return chosen


def _correct(examples, rows, gold):
    return np.asarray([r["status"] == "ok" and r["prediction"] == gold[e["sample_id"]]
                       for e, r in zip(examples, _predictions(examples, rows))], dtype=bool)


def _counts(front_correct, back_correct, deferred):
    final = np.where(deferred, back_correct, front_correct)
    rescue = int(np.sum(deferred & ~front_correct & back_correct))
    harm = int(np.sum(deferred & front_correct & ~back_correct))
    return {"n": len(final), "correct": int(final.sum()), "accuracy": float(final.mean()),
            "deferred": int(deferred.sum()), "defer_fraction": float(deferred.mean()),
            "rescues": rescue, "harms": harm, "net_correct_vs_front": rescue-harm}


def _threshold(scores, ids, front_correct, back_correct, forced):
    # Fixed selection recipe: nonnegative gain, <=10% voluntary budget, allow none.
    k = int(math.floor(QUOTA*len(ids)))
    require(int(forced.sum()) <= k, "Selection forced failures exceed quota")
    best = {"threshold": None, "net_correct": int(np.sum(back_correct[forced])-np.sum(front_correct[forced])),
            "deferred": int(forced.sum())}
    for value in sorted(set(float(x) for x in scores if x > 0), reverse=True):
        mask = forced | (scores >= value)
        if int(mask.sum()) > k:
            continue
        gain = int(np.sum(back_correct[mask])-np.sum(front_correct[mask]))
        if gain > best["net_correct"]:
            best = {"threshold": value, "net_correct": gain, "deferred": int(mask.sum())}
    return best


def fit_suite(fit_examples, selection_examples, front_rows, fallback_rows_by_role, train_gold,
              embeddings_by_id, fit_ids_by_size):
    fitting, fit_ids = _examples(fit_examples, {"fitting"})
    selection, selection_ids = _examples(selection_examples, {"selection"})
    require(not set(fit_ids) & set(selection_ids), "Fitting/selection IDs overlap")
    require(not {r["premise_hash"] for r in fitting} & {r["premise_hash"] for r in selection}, "Fitting/selection premises overlap")
    _gold(train_gold, fit_ids+selection_ids)
    require(set(fit_ids_by_size) == {str(n) for n in FIT_SIZES}, "Nested fitting budgets differ")
    previous = set()
    for n in FIT_SIZES:
        ids = fit_ids_by_size[str(n)]
        require(len(ids) == n and len(set(ids)) == n and previous <= set(ids) <= set(fit_ids), "Fitting IDs are not valid nested budgets")
        previous = set(ids)
    require(previous == set(fit_ids), "Maximum fitting budget must match the fitting pool exactly")
    require(set(fallback_rows_by_role) == set(FALLBACKS), "Fallback roles differ")
    fit_x, failed, _ = feature_matrices(fitting, front_rows, embeddings_by_id)
    sel_x, sel_failed, _ = feature_matrices(selection, front_rows, embeddings_by_id)
    front_y = _correct(fitting, front_rows, train_gold)
    sel_front = _correct(selection, front_rows, train_gold)
    back_y = {role: _correct(fitting, fallback_rows_by_role[role], train_gold) for role in FALLBACKS}
    sel_back = {role: _correct(selection, fallback_rows_by_role[role], train_gold) for role in FALLBACKS}
    by_id = {sid: i for i, sid in enumerate(fit_ids)}
    cells = {}
    for n in FIT_SIZES:
        index = np.asarray([by_id[sid] for sid in fit_ids_by_size[str(n)]], dtype=int)
        index = index[~failed[index]]
        require(len(index) > 0, "No usable fitting front predictions")
        for arm in ARMS:
            candidates = {role: [] for role in FALLBACKS}
            for l2 in L2_GRID:
                local_head = fit_head(fit_x[arm][index], front_y[index], l2)
                local_scores = score_head(local_head, sel_x[arm])
                for role in FALLBACKS:
                    fallback_head = fit_head(fit_x[arm][index], back_y[role][index], l2)
                    gain = score_head(fallback_head, sel_x[arm])-local_scores
                    gain[sel_failed] = 0
                    mask = exact_quota(gain, selection_ids, sel_failed)
                    counts = _counts(sel_front, sel_back[role], mask)
                    candidates[role].append({"l2": l2, "local_head": local_head, "fallback_head": fallback_head,
                                             "selection": counts})
            for role in FALLBACKS:
                choices = candidates[role]
                winner = max(range(len(choices)), key=lambda j: (choices[j]["selection"]["correct"], choices[j]["l2"]))
                picked = choices[winner]
                gain = score_head(picked["fallback_head"], sel_x[arm])-score_head(picked["local_head"], sel_x[arm])
                gain[sel_failed] = 0
                key = f"{role}/n{n}/{arm}"
                cells[key] = {"fallback": role, "fit_size": n, "arm": arm, "usable_fit_count": len(index),
                              "candidates": choices, "selected_index": winner,
                              "deployment_threshold": _threshold(gain, selection_ids, sel_front, sel_back[role], sel_failed)}
    return {"recipe": RECIPE, "fit_ids_sha256": fingerprint(fit_ids_by_size), "selection_ids_sha256": fingerprint(selection_ids),
            "fitting_premises_sha256": fingerprint(sorted({r["premise_hash"] for r in fitting})),
            "selection_premises_sha256": fingerprint(sorted({r["premise_hash"] for r in selection})),
            "fit_sizes": list(FIT_SIZES), "cells": cells, "evaluation_gold_used": False}


def score_suite(models, evaluation_examples, front_rows, embeddings_by_id):
    examples, ids = _examples(evaluation_examples, {"eligible_dev", "train_holdout"})
    require(len({r["split"] for r in examples}) == 1, "Score each evaluation population separately")
    require(models.get("recipe") == RECIPE and models.get("fit_sizes") == list(FIT_SIZES), "Fitted recipe differs")
    require(models.get("evaluation_gold_used") is False, "Fitting record claims evaluation gold use")
    features, failed, confidence = feature_matrices(examples, front_rows, embeddings_by_id)
    cells = {}
    expected = {f"{role}/n{n}/{arm}" for role in FALLBACKS for n in FIT_SIZES for arm in ARMS}
    require(set(models["cells"]) == expected, "Fitted cell inventory differs")
    for key, model in models["cells"].items():
        require(key == f"{model['fallback']}/n{model['fit_size']}/{model['arm']}", "Fitted cell identity differs")
        choices = model["candidates"]
        require([x["l2"] for x in choices] == list(L2_GRID), "Search budget differs")
        winner = max(range(len(choices)), key=lambda j: (choices[j]["selection"]["correct"], choices[j]["l2"]))
        require(model["selected_index"] == winner, "Selected model violates the frozen selection/tie rule")
        picked = choices[winner]
        gain = score_head(picked["fallback_head"], features[model["arm"]])-score_head(picked["local_head"], features[model["arm"]])
        gain[failed] = 0
        cells[key] = {"fallback": model["fallback"], "fit_size": model["fit_size"], "arm": model["arm"],
                      "scores": gain.tolist(), "deployment_threshold": model["deployment_threshold"]}
    return {"recipe": RECIPE, "models_sha256": fingerprint(models), "examples_sha256": fingerprint(examples),
            "split": examples[0]["split"], "sample_ids": ids, "front_failed": failed.tolist(),
            "front_confidence": confidence.tolist(), "cells": cells, "evaluation_gold_used": False}


def _paired_bootstrap(examples, text, front, failed, front_correct, back_correct):
    ids = np.asarray([r["sample_id"] for r in examples], dtype=str)
    groups = {}
    for i, row in enumerate(examples):
        groups.setdefault(row["premise_hash"], []).append(i)
    members = [np.asarray(groups[g], dtype=int) for g in sorted(groups)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    failed_draws = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        picked = rng.integers(len(members), size=len(members))
        index = np.concatenate([members[g] for g in picked])
        try:
            a = exact_quota(text[index], ids[index], failed[index])
            b = exact_quota(front[index], ids[index], failed[index])
        except StudyError:
            failed_draws += 1
            continue
        ca = np.where(a, back_correct[index], front_correct[index])
        cb = np.where(b, back_correct[index], front_correct[index])
        deltas.append(float((ca.astype(float)-cb).mean()))
    # Dropping infeasible bootstrap draws would condition the requested interval.
    # Keep the point result but explicitly withhold that interval instead.
    if failed_draws:
        return {"status": "UNAVAILABLE_FORCED_FAILURE_QUOTA", "replicates": BOOTSTRAP_REPLICATES,
                "infeasible_draws": failed_draws, "ci95": None, "premise_groups": len(members)}
    lo, hi = np.quantile(deltas, [.025, .975])
    return {"status": "PASS", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "ci95": [float(lo), float(hi)], "premise_groups": len(members),
            "scope": "conditional on fitted models; paired normalized-premise percentile bootstrap; rerank exact quota per draw"}


def evaluate_suite(frozen_scores, evaluation_examples, front_rows, fallback_rows_by_role, eval_gold):
    examples, ids = _examples(evaluation_examples, {"eligible_dev", "train_holdout"})
    require(len({r["split"] for r in examples}) == 1, "Evaluate populations separately")
    _gold(eval_gold, ids)
    require(set(fallback_rows_by_role) == set(FALLBACKS), "Fallback roles differ")
    require(frozen_scores.get("recipe") == RECIPE and frozen_scores.get("sample_ids") == ids
            and frozen_scores.get("examples_sha256") == fingerprint(examples)
            and frozen_scores.get("split") == examples[0]["split"] and frozen_scores.get("evaluation_gold_used") is False,
            "Frozen score input/recipe identity differs")
    _, failed, confidence = front_features(examples, front_rows)
    require(frozen_scores["front_failed"] == failed.tolist() and frozen_scores["front_confidence"] == confidence.tolist(),
            "Frozen front observation differs")
    front_correct = _correct(examples, front_rows, eval_gold)
    backs = {role: _correct(examples, fallback_rows_by_role[role], eval_gold) for role in FALLBACKS}
    expected = {f"{role}/n{n}/{arm}" for role in FALLBACKS for n in FIT_SIZES for arm in ARMS}
    require(set(frozen_scores["cells"]) == expected, "Frozen score cell inventory differs")
    cells = {}
    for key, cell in frozen_scores["cells"].items():
        require(key == f"{cell['fallback']}/n{cell['fit_size']}/{cell['arm']}", "Frozen cell identity differs")
        gain = np.asarray(cell["scores"], dtype=np.float64)
        mask = exact_quota(gain, ids, failed)
        threshold = cell["deployment_threshold"]["threshold"]
        require(threshold is None or (isinstance(threshold, (float, int)) and math.isfinite(threshold) and threshold > 0),
                "Invalid frozen deployment threshold")
        deployment = failed.copy() if threshold is None else failed | (gain >= threshold)
        cells[key] = {**_counts(front_correct, backs[cell["fallback"]], mask),
                      "deployment_threshold_result": _counts(front_correct, backs[cell["fallback"]], deployment)}
    baselines = {"direct_startlux4b": {"n": len(ids), "correct": int(front_correct.sum()), "accuracy": float(front_correct.mean())}}
    random_priority = np.asarray([int(hashlib.sha256(f"{RANDOM_SEED}:{sid}".encode()).hexdigest()[:13], 16) for sid in ids], dtype=float)
    for role in FALLBACKS:
        back = backs[role]
        baselines[f"direct_{role}"] = {"n": len(ids), "correct": int(back.sum()), "accuracy": float(back.mean())}
        for name, score in (("confidence10", -confidence), ("random10", random_priority),
                            ("oracle10_hindsight", back.astype(float)-front_correct.astype(float))):
            baselines[f"{role}/{name}"] = _counts(front_correct, back, exact_quota(score, ids, failed))
    comparisons = {}
    for role in FALLBACKS:
        for n in FIT_SIZES:
            text_key, front_key = f"{role}/n{n}/text", f"{role}/n{n}/front"
            value = {"accuracy_difference": cells[text_key]["accuracy"]-cells[front_key]["accuracy"],
                     "text_cell": text_key, "front_cell": front_key,
                     "primary": examples[0]["split"] == "eligible_dev" and role == "qwen9b" and n == max(FIT_SIZES)}
            if role == "qwen9b" and n == max(FIT_SIZES):
                value["interval"] = _paired_bootstrap(examples, np.asarray(frozen_scores["cells"][text_key]["scores"]),
                    np.asarray(frozen_scores["cells"][front_key]["scores"]), failed, front_correct, backs[role])
                text_mask = exact_quota(frozen_scores["cells"][text_key]["scores"], ids, failed)
                front_mask = exact_quota(frozen_scores["cells"][front_key]["scores"], ids, failed)
                delta = np.where(text_mask, backs[role], front_correct).astype(float)-np.where(front_mask, backs[role], front_correct)
                weights = {"R1": .3125, "R2": .3125, "R3": .375}
                require(set(r["round"] for r in examples) == set(weights), "Round sensitivity requires all three rounds")
                value["official_round_weight_sensitivity"] = sum(w*float(delta[[i for i, e in enumerate(examples) if e["round"] == rnd]].mean()) for rnd, w in weights.items())
                value["weighting_limit"] = "Cannot remove within-round historical exclusion bias or recover excluded examples"
            comparisons[f"{role}/n{n}"] = value
    return {"split": examples[0]["split"], "n": len(ids), "frozen_scores_sha256": fingerprint(frozen_scores),
            "cells": cells, "baselines": baselines, "comparisons": comparisons,
            "interpretation": "Question-weighted eligible subset; one primary comparison. Other cells/holdout/round sensitivity are descriptive; no multiplicity correction or equivalence claim."}
