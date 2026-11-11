"""Direct accuracy and paired cluster-bootstrap differences for any of the four tasks.

Pure functions over already-authorized, aligned inputs. The bootstrap is the
V2/V3 algorithm (sorted cluster keys, ``numpy.random.default_rng(seed)``, pooled
question-weighted ratio per draw); only the cluster key is task-specific.
"""
from collections import defaultdict

import numpy as np

from ..common import StudyError


def correct(row, label):
    return bool(row["status"] == "ok" and row.get("prediction") == label)


def direct(examples, rows, gold, labels):
    """Accuracy over the full fixed denominator; terminal errors count wrong; nothing missing allowed."""
    outcomes, failures = [], 0
    for example in examples:
        sid = example["sample_id"]
        row = rows.get(sid)
        if row is None:
            raise StudyError(f"Missing prediction for {sid}; a missing cell is never scored as zero")
        if row["status"] not in {"ok", "error"} or (row["status"] == "ok" and row.get("prediction") not in labels):
            raise StudyError(f"Invalid terminal prediction for {sid}")
        if gold.get(sid) not in labels:
            raise StudyError(f"Missing or invalid gold for {sid}")
        failures += row["status"] == "error"
        outcomes.append(correct(row, gold[sid]))
    if len(rows) != len(examples):
        raise StudyError("Prediction set has rows outside the fixed denominator")
    n = len(examples)
    return {"n": n, "correct": sum(outcomes), "accuracy": sum(outcomes) / n, "failures": failures}, outcomes


def cluster_keys(examples, cluster):
    if cluster == "sample_id":
        keys = [e["sample_id"] for e in examples]
    elif cluster == "group_id":
        keys = [e.get("group_id") for e in examples]
    else:
        raise StudyError(f"Unknown cluster definition {cluster!r}")
    if any(not isinstance(k, str) or not k for k in keys):
        raise StudyError("Every input needs a non-empty cluster key")
    return keys


def paired_difference(examples, candidate, reference, *, cluster, repetitions, seed):
    """Paired percentile interval for candidate minus reference accuracy, resampling whole clusters."""
    a, b = np.asarray(candidate), np.asarray(reference)
    for array in (a, b):
        if array.shape != (len(examples),) or not np.isin(array, [False, True]).all():
            raise StudyError("Bootstrap needs aligned binary correctness arrays")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise StudyError("Bootstrap repetitions must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StudyError("Bootstrap seed must be a nonnegative integer")
    groups = defaultdict(list)
    for index, key in enumerate(cluster_keys(examples, cluster)):
        groups[key].append(index)
    if len(groups) < 2:
        raise StudyError("At least two clusters are required")
    difference = a.astype(int) - b.astype(int)
    keys = sorted(groups)
    sizes = np.asarray([len(groups[k]) for k in keys], dtype=np.int64)
    totals = np.asarray([difference[groups[k]].sum() for k in keys], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, len(keys), size=len(keys))
        draws[index] = totals[selected].sum() / sizes[selected].sum()
    return {"estimate": float(difference.mean()), "lower": float(np.quantile(draws, .025)),
            "upper": float(np.quantile(draws, .975)), "confidence_level": .95,
            "method": "paired_cluster_percentile_bootstrap", "cluster": cluster, "cluster_count": len(groups),
            "question_count": len(examples), "replicates": repetitions, "seed": seed,
            "candidate_only_correct": int(((a == 1) & (b == 0)).sum()),
            "reference_only_correct": int(((a == 0) & (b == 1)).sum()),
            "both_correct": int(((a == 1) & (b == 1)).sum()), "both_wrong": int(((a == 0) & (b == 0)).sum()),
            "warning": "degenerate_empirical_distribution_not_a_guarantee" if np.ptp(draws) == 0 else None}
