"""Pure V3 metrics and paired premise-cluster bootstrap; no files or models.

The study owns access/identity gates. These functions consume already-authorized
aligned inputs and never select a policy or tune a threshold on evaluation data.
Prepared group_id is the normalized-premise identity, without a round prefix, so
repeated premises remain one cluster even when they occur in different rounds.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ..common import StudyError
from .policies import LABELS, _examples, _row, route


def _gold(examples: Sequence[dict], gold: Mapping[str, str]) -> None:
    _examples(examples)
    if any(gold.get(x["sample_id"]) not in LABELS for x in examples):
        raise StudyError("Missing or invalid evaluation gold")


def _correct(row: dict, label: str) -> bool:
    return row["status"] == "ok" and row["prediction"] == label


def evaluate_policy(policy: dict, examples: Sequence[dict], localrows: Mapping[str, dict],
                    fallbackrows: Mapping[str, dict], gold: Mapping[str, str]) -> tuple[dict, list[bool]]:
    """Return metrics and aligned per-question correctness, counting failures wrong."""
    _gold(examples, gold)
    if not policy.get("fallback_model"):
        raise StudyError("Policy lacks fallback model identity")
    local, fallback, deferred, failures, forced = [], [], [], [], []
    for example in examples:
        sid = example["sample_id"]
        first = _row(example, localrows, model="startlux4b")
        second = _row(example, fallbackrows, model=policy["fallback_model"])
        send = route(policy, example, first)
        local.append(_correct(first, gold[sid]))
        fallback.append(_correct(second, gold[sid]))
        deferred.append(send)
        forced.append(first["status"] == "error")
        failures.append((second if send else first)["status"] == "error")
    lc, fc, sent = np.array(local), np.array(fallback), np.array(deferred)
    final = np.where(sent, fc, lc)
    rescue = int((sent & ~lc & fc).sum())
    harm = int((sent & lc & ~fc).sum())
    metric = {"n": len(examples), "correct": int(final.sum()), "accuracy": float(final.mean()),
              "deferrals": int(sent.sum()), "actual_deferral_rate": float(sent.mean()),
              "forced_deferrals": sum(forced), "optional_deferrals": int(sent.sum()) - sum(forced),
              "rescue": rescue, "harm": harm, "net_correct_gain": rescue - harm,
              "local_correct": int(lc.sum()), "fallback_correct": int(fc.sum()),
              "final_failures": sum(failures), "target_budget": policy["decision"]["target_budget"],
              "fallback_model": policy["fallback_model"], "family": policy["family"]}
    assert metric["net_correct_gain"] == metric["correct"] - metric["local_correct"]
    return metric, final.tolist()


def evaluate_direct(examples: Sequence[dict], rows: Mapping[str, dict], gold: Mapping[str, str],
                    *, model: str) -> tuple[dict, list[bool]]:
    """Direct-model baseline; no artificial deferral fraction is assigned."""
    _gold(examples, gold)
    aligned = [_row(x, rows, model=model) for x in examples]
    correct = [_correct(row, gold[x["sample_id"]]) for x, row in zip(examples, aligned)]
    metric = {"model": model, "n": len(examples), "correct": sum(correct),
              "accuracy": sum(correct) / len(examples),
              "failures": sum(row["status"] == "error" for row in aligned)}
    return metric, correct


def cluster_bootstrap_difference(examples: Sequence[dict], candidate_correct: Sequence[bool],
                                 reference_correct: Sequence[bool], *,
                                 repetitions: int = 2000, seed: int = 6998) -> dict:
    """Paired percentile interval for pooled question accuracy difference.

    Sample G premise clusters with replacement and divide the summed question
    differences by the summed question counts in each draw. This is not a mean
    of cluster accuracies or a round-stratified bootstrap. Cross-round copies of
    a premise share their prepared group_id and therefore always move together.
    """
    _examples(examples)
    candidate = np.asarray(candidate_correct)
    reference = np.asarray(reference_correct)
    for array in (candidate, reference):
        if array.shape != (len(examples),) or not np.isin(array, [False, True]).all():
            raise StudyError("Bootstrap needs aligned binary correctness arrays")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise StudyError("Bootstrap repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StudyError("Bootstrap seed must be a nonnegative integer")
    groups = defaultdict(list)
    for index, example in enumerate(examples):
        group = example.get("group_id")
        if not isinstance(group, str) or not group:
            raise StudyError("A prepared normalized-premise group_id is required")
        groups[group].append(index)
    if len(groups) < 2:
        raise StudyError("At least two premise clusters are required")
    difference = candidate.astype(int) - reference.astype(int)
    keys = sorted(groups)
    sizes = np.asarray([len(groups[key]) for key in keys], dtype=np.int64)
    totals = np.asarray([difference[groups[key]].sum() for key in keys], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, len(keys), size=len(keys))
        draws[index] = totals[selected].sum() / sizes[selected].sum()
    return {"estimate": float(difference.mean()), "lower": float(np.quantile(draws, .025)),
            "upper": float(np.quantile(draws, .975)), "confidence_level": .95,
            "method": "paired_normalized_premise_percentile_bootstrap", "cluster_count": len(groups),
            "question_count": len(examples), "minimum_cluster_size": int(sizes.min()),
            "maximum_cluster_size": int(sizes.max()), "replicates": repetitions, "seed": seed,
            "cross_round_clusters": sum(len({examples[i].get("round") for i in members}) > 1
                                        for members in groups.values()),
            "warning": "degenerate_empirical_distribution_not_a_guarantee" if np.ptp(draws) == 0 else None,
            "estimand": "pooled question-weighted accuracy difference across the prespecified ANLI rounds",
            "note": "Conditional on frozen policies; range excludes neither training exposure nor other dataset limitations. No equivalence or future-risk guarantee."}
