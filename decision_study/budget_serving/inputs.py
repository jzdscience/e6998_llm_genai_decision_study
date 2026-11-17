"""Pinned Study 4 artifacts, read directly by SHA-256 rather than through the historical context.

The historical ``learnability.study.context()`` compares the whole current source tree
with the one that produced Study 4 and would reject any new module; it is therefore not
called. Instead every byte used here is bound to the hashes fixed in the protocol and
cross-checked against the historical freeze and fitting receipts. Evaluation labels are
parsed only by ``load_gold()``, which the controller calls after the new trace freeze.
"""
from __future__ import annotations

import json
import math

from ..common import StudyError, file_hash, fingerprint, read_json
from ..learnability import data as study_data
from ..learnability import modeling
from ..learnability_profile import study as profile
from .protocol import LABELS, require

INPUT_FIELDS = profile.INPUT_FIELDS
POLICY_KEYS = ("cell", "selected_index", "l2", "threshold", "local_head", "fallback_head")
# Canonical fingerprints derived from the accepted Study 4 bytes (all 8 pinned SHA-256 checks
# passing) and reproduced independently by review. They anchor the prepared copies offline,
# so a jointly rewritten workload/cell/policy plus a regenerated hash manifest cannot pass.
PINNED_WORKLOAD_FINGERPRINT = "25e11c63687d98794c73b310d30675226710a9e32894ae386d2c96d6a6f1d0f5"
PINNED_CELL_FINGERPRINT = "ce08ef53f7b2d16034db625b5ba60c5585470a5bb7e5f75a91093032fc43af3b"
PINNED_POLICY_FINGERPRINT = "a5e63a558845ee179b1ff78e189f1b7c0eb1c59ba684412ca51a9e3eea682c0f"


def safe(root, relative):
    return profile.safe(root, relative)


def source_root(runtime_root, p):
    return safe(runtime_root, p["source_study"]["default_directory"])


def verify_source_files(source, p):
    """Byte identity of every pinned historical file. Hashing the gold file uses no labels."""
    hashes = {}
    for relative, expected in p["source_files"].items():
        path = safe(source, relative)
        require(path.is_file(), f"Pinned Study 4 file is missing: {relative}")
        actual = file_hash(path)
        require(actual == expected, f"Pinned Study 4 file differs: {relative}")
        hashes[relative] = actual
    return hashes


def _rows(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise StudyError(f"Invalid pinned example line {number}") from exc
    return rows


def validate_workload(rows, p):
    ids = [r.get("sample_id") for r in rows]
    require(len(rows) == p["workload"]["count"] and len(set(ids)) == len(ids) and ids == sorted(ids),
            "Workload must be the unique sample_id-ordered eligible-dev population")
    for row in rows:
        require(set(row) == INPUT_FIELDS and row["task"] == "anli" and row["split"] == p["workload"]["split"]
                and row["labels"] == list(LABELS) and isinstance(row["state"], str) and row["state"]
                and row["source_split"] == "dev" and row["round"] in {"R1", "R2", "R3"},
                "Workload rows differ from the gold-free canonical eligible-dev schema")
    require(study_data.sha(study_data.jsonl(study_data.membership(rows))) == study_data.DEV_MEMBERSHIP_SHA256,
            "Eligible-dev membership differs from the accepted input audit")
    require(len({r["premise_hash"] for r in rows}) == p["workload"]["premise_groups"], "Eligible-dev premise groups differ")
    require(fingerprint(rows) == PINNED_WORKLOAD_FINGERPRINT, "Workload rows differ from the accepted eligible-dev inputs")
    return rows


def load_workload(source, p):
    rows = _rows(safe(source, p["workload"]["source_file"]))
    manifest = read_json(safe(source, "data/manifest.json"))
    require(manifest.get("eligible_dev_membership_sha256") == study_data.DEV_MEMBERSHIP_SHA256
            and manifest.get("dev_labels_used") is False
            and manifest.get("files", {}).get("data/examples.jsonl", {}).get("sha256") == p["source_files"]["data/examples.jsonl"],
            "Study 4 manifest does not bind the pinned examples and eligible-dev membership")
    selected = sorted((r for r in rows if r.get("split") == p["workload"]["split"]), key=lambda r: r["sample_id"])
    return validate_workload(selected, p)


def load_development(runtime_root, p):
    spec = p["development"]
    path = safe(runtime_root, spec["source_file"])
    require(path.is_file() and file_hash(path) == spec["source_sha256"], "V3 development source differs")
    rows = profile.inputs(safe(runtime_root, "results/solo-v3"))
    require(len(rows) == spec["count"] and fingerprint(rows) == spec["rows_fingerprint"]
            and [r["sample_id"] for r in rows] == sorted(r["sample_id"] for r in rows),
            "Development inputs differ from the fixed 90 sample_id-ordered rows")
    return rows


def _winner(choices):
    # Frozen Study 4 selection/tie rule, identical to modeling.score_suite.
    return max(range(len(choices)), key=lambda j: (choices[j]["selection"]["correct"], choices[j]["l2"]))


def _finite_head(head):
    require(isinstance(head, dict) and head.get("kind") in {"logistic", "constant"}, "Saved head kind differs")
    values = [*head.get("mean", []), *head.get("scale", []), *(head.get("coefficients") or [])]
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values),
            "Saved head contains invalid numbers")
    require(len(head.get("mean", [])) == len(head.get("scale", [])) == len(modeling.RECIPE["feature_names"]),
            "Saved front head is not the seven-feature recipe")


def validate_policy_cell(cell, models_meta, p):
    """Re-derive the selected candidate from the full stored cell (offline-capable)."""
    spec = p["policy"]
    require(models_meta.get("recipe") == modeling.RECIPE and models_meta.get("evaluation_gold_used") is False,
            "Study 4 fitted recipe differs from the frozen modeling code")
    require(cell.get("fallback") == "startlux9b" and cell.get("fit_size") == 18000 and cell.get("arm") == "front",
            "Selected policy cell identity differs")
    choices = cell.get("candidates", [])
    require([c.get("l2") for c in choices] == list(modeling.L2_GRID), "Stored L2 search differs")
    winner = _winner(choices)
    require(cell.get("selected_index") == winner == spec["selected_index"] and choices[winner]["l2"] == spec["l2"],
            "Stored selection violates the frozen rule or differs from the protocol")
    threshold = cell.get("deployment_threshold", {})
    require(threshold.get("threshold") == spec["threshold"]
            and threshold.get("deferred") == spec["offline_regression"]["selection_deferred"],
            "Stored deployment threshold differs from the protocol")
    for name in ("local_head", "fallback_head"):
        _finite_head(choices[winner][name])
        require(choices[winner][name].get("l2") == spec["l2"], "Selected head regularization differs")
    picked = choices[winner]
    policy = {"cell": spec["cell"], "selected_index": winner, "l2": picked["l2"], "threshold": threshold["threshold"],
              "local_head": picked["local_head"], "fallback_head": picked["fallback_head"]}
    require(fingerprint(cell) == PINNED_CELL_FINGERPRINT and fingerprint(policy) == PINNED_POLICY_FINGERPRINT,
            "Selected policy cell differs from the accepted Study 4 models artifact")
    return policy


def load_policy(source, p, workload):
    spec = p["policy"]
    models = read_json(safe(source, spec["source_file"]))
    cell = models.get("cells", {}).get(spec["cell"])
    require(isinstance(cell, dict), "Selected Study 4 policy cell is missing")
    meta = {"recipe": models.get("recipe"), "evaluation_gold_used": models.get("evaluation_gold_used")}
    policy = validate_policy_cell(cell, meta, p)
    files = p["source_files"]
    freeze = read_json(safe(source, "freeze.json"))
    fitting = read_json(safe(source, "fitting/complete.json"))
    require(freeze.get("models_and_scores_frozen_before_evaluation_gold") is True
            and all(freeze.get("files", {}).get(k) == files[k] for k in
                    ("data/examples.jsonl", "data/manifest.json", "fitting/models.json", "fitting/complete.json",
                     "fitting/scores-eligible_dev.json")),
            "Historical freeze does not bind the pinned examples, models and scores")
    require(fitting.get("models_sha256") == files["fitting/models.json"] and fitting.get("evaluation_gold_used") is False
            and fitting.get("scores", {}).get("eligible_dev") == files["fitting/scores-eligible_dev.json"]
            and fitting.get("recipe") == modeling.RECIPE, "Historical fitting receipt does not bind the selected models")
    # Offline regression of the pinned artifacts only; saved scores never route a live request.
    scores = read_json(safe(source, "fitting/scores-eligible_dev.json"))
    stored = scores.get("cells", {}).get(spec["cell"], {})
    require(scores.get("sample_ids") == [r["sample_id"] for r in workload] and scores.get("evaluation_gold_used") is False
            and scores.get("models_sha256") == fingerprint(models), "Historical eligible-dev scores do not bind this workload/models")
    failed = scores.get("front_failed", [])
    deferred = sum(bool(f) or g >= policy["threshold"] for f, g in zip(failed, stored.get("scores", [])))
    require(len(failed) == len(workload) == len(stored.get("scores", []))
            and deferred == spec["offline_regression"]["eligible_dev_threshold_deferred"],
            "Pinned threshold regression differs from the protocol")
    return {"policy": policy, "cell": cell, "models_meta": meta,
            "binding": {"models_sha256": files["fitting/models.json"], "models_fingerprint": fingerprint(models),
                        "historical_threshold_deferred_eligible_dev": deferred,
                        "historical_selection_deferred": cell["deployment_threshold"]["deferred"],
                        "role": spec["role"], "routing_inputs": "live startlux4b response + unmodified input only"}}


def verify_policy_document(document, p):
    """Offline check of inputs/policy.json: the stored cell still yields the stored policy."""
    require(set(document) == {"policy", "cell", "models_meta", "binding"}, "Policy document inventory differs")
    expected = validate_policy_cell(document["cell"], document["models_meta"], p)
    require(document["policy"] == expected and set(expected) == set(POLICY_KEYS), "Stored policy differs from its cell")
    return document["policy"]


def load_gold(source, p, workload_ids):
    """Post-freeze only: exact eligible-dev IDs from the 3,537-ID Study 4 gold file."""
    path = safe(source, p["gold"]["source_file"])
    require(file_hash(path) == p["gold"]["sha256"], "Study 4 evaluation gold differs from its pinned SHA")
    return gold_subset(read_json(path), p, workload_ids)


def gold_subset(gold, p, workload_ids):
    require(isinstance(gold, dict) and len(gold) == p["gold"]["source_count"], "Study 4 gold count differs")
    missing = [sid for sid in workload_ids if sid not in gold]
    require(not missing, "Gold lacks eligible-dev IDs")
    subset = {sid: gold[sid] for sid in workload_ids}
    require(all(label in LABELS for label in subset.values()), "Gold label differs from the ANLI schema")
    return subset
