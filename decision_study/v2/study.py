"""Fit on original calibration, select on validation, then open a sealed holdout.

Public functions expect the caller to hold the output-directory command lock.
Fingerprints detect accidental/stale mutations; they are not digital signatures.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import platform

import numpy as np

from ..adapters import decision_request, qwen_request
from ..analysis import validate_calibration
from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl, utc_now, write_json
from ..storage import load_predictions
from ..runner import _probability_comparison
from . import data_v2
from .identity import source_identity
from .policies import fit_candidates, freeze_budget, local_information, should_defer


def _source_hashes() -> dict:
    return source_identity()


def _checked_self_hash(artifact: dict, key: str) -> None:
    expected = artifact.get(key)
    if not expected or expected != fingerprint({k: v for k, v in artifact.items() if k != key}):
        raise StudyError(f"Invalid {key}: artifact was changed")


def _protocol(manifest: dict) -> dict:
    try:
        protocol = manifest["protocol_identity"]["configuration"]
        p, e = protocol["policies"], protocol["evaluation"]
        checks = [protocol["protocol"] == "solo-v2", protocol["data"]["primary_task"] == "squad",
                  p["candidates"] == ["class_conditional", "gain"],
                  p["class_conditional_features"] == ["confidence_logit", "predicted_class"],
                  p["gain_features"] == ["confidence_logit", "predicted_class", "log1p_input_characters"],
                  p["fit_split"] == "calibration", p["selection_split"] == "validation",
                  p["baseline"] == "confidence: 1 minus released maximum probability",
                  p["selection"] == "highest validation cascade correct count; ties choose class_conditional",
                  p["threshold"] == "whole score-tie groups, largest validation deferral count not exceeding budget; runtime failures force deferral",
                  p["original_gate"] == "v1 refitted risk gate retained as secondary descriptive reference",
                  e["bootstrap_unit"] == "article", e["confidence_level"] == .95]
        if not all(checks) or not math.isfinite(p["l2"]) or p["l2"] <= 0:
            raise StudyError("Unsupported v2 fitting/evaluation protocol")
        budgets = [p["primary_budget"], *p["secondary_budgets"]]
        if len(set(budgets)) != len(budgets) or any(not 0 <= b <= 1 for b in budgets):
            raise StudyError("Invalid or duplicate protocol budgets")
        if not isinstance(e["bootstrap_reps"], int) or e["bootstrap_reps"] < 1:
            raise StudyError("Invalid bootstrap repetitions")
        return protocol
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyError("Incomplete v2 fitting/evaluation protocol") from exc


def _index(rows: list[dict], examples: list[dict], model: str, *, exact: bool = True) -> dict:
    answer = {r["sample_id"]: r for r in rows}
    ids = {x["sample_id"] for x in examples}
    if len(answer) != len(rows) or not ids <= set(answer) or (exact and ids != set(answer)):
        raise StudyError(f"Incomplete, duplicate or foreign {model} prediction IDs")
    for example in examples:
        row = answer[example["sample_id"]]
        if row.get("model") != model or row.get("status") not in {"ok", "error"}:
            raise StudyError(f"Invalid {model} prediction identity/status")
        if row["status"] == "ok" and row.get("prediction") not in example["labels"]:
            raise StudyError(f"Invalid {model} task label")
        if model == "startlux":
            local_information(example, row)
    return {sid: answer[sid] for sid in sorted(ids)}


def _job_predictions(v1_dir: Path, output_dir: Path, manifest: dict, examples: list[dict],
                     job: str, model: str) -> dict:
    """Verify saved job identity and smoke evidence without starting GPU services."""
    jobdir = output_dir / "predictions-jobs" / job
    paths = [jobdir / "completion.json", jobdir / "smoke.json", jobdir / f"predictions/{model}/config.json"]
    if not all(p.is_file() for p in paths):
        raise StudyError(f"Missing completed {job} prediction job and/or smoke receipt")
    completion, smoke, config = (read_json(p) for p in paths)
    original = read_json(v1_dir / f"predictions/{model}/config.json")
    renderer = qwen_request if model == "qwen" else lambda e: decision_request(e, model)
    expected = {"protocol": "solo-v2", "job": job, "model": model,
                "data_manifest_hash": fingerprint(manifest), "sources": _source_hashes(),
                "protocol_hash": fingerprint(manifest["protocol_identity"]["configuration"]),
                "runtime": original["runtime"], "v1_config_hash": fingerprint(original),
                "request_hash": fingerprint([{"id": e["sample_id"], "request": renderer(e)} for e in examples]),
                "expected_predictions": len(examples), "max_attempts": 2, "timeout_seconds": 120}
    if config != expected:
        raise StudyError(f"{job} prediction configuration/source/runtime does not match the frozen protocol")
    config_hash = fingerprint(config)
    if (completion.get("config_hash") != config_hash or completion.get("job") != job
            or completion.get("model") != model or completion.get("count") != len(examples)
            or smoke.get("config_hash") != config_hash):
        raise StudyError(f"{job} completion/smoke identity mismatch")
    dev = [x for x in read_jsonl(v1_dir / "data/examples.jsonl") if x["split"] == "development"]
    reference = read_json(v1_dir / f"smoke/{model}.json")["serial"]
    checks = _probability_comparison(dev, reference, smoke.get("rows", []))
    if not checks or checks != smoke.get("checks") or not all(c["passed"] for c in checks.values()):
        raise StudyError(f"{job} saved smoke evidence failed recomputation")
    rows = _index(load_predictions(jobdir, model), examples, model)
    if completion.get("failures") != sum(r["status"] != "ok" for r in rows.values()):
        raise StudyError(f"{job} completion failure count mismatch")
    return rows


def _fitting_inputs(v1_dir: Path, output_dir: Path):
    manifest = data_v2.verify_fit(output_dir)
    protocol = _protocol(manifest)
    examples = data_v2.load_examples(output_dir, role="qwen_fit")
    gold = data_v2.load_fit_gold(output_dir)
    ids = {x["sample_id"] for x in examples}
    if (len(ids) != len(examples) or set(gold) != ids
            or len(examples) != protocol["data"]["qwen_fit_count"]
            or any(x["split"] not in {"calibration", "validation"} for x in examples)):
        raise StudyError("Fitting data roles/counts do not match protocol")
    original_examples = {x["sample_id"]: x for x in read_jsonl(v1_dir / "data/examples.jsonl")}
    original_gold = read_json(v1_dir / "data/gold.json")
    for example in examples:
        sid = example["sample_id"]
        if original_examples.get(sid) != example or original_gold.get(sid) != gold[sid]:
            raise StudyError("Fitting rows/gold differ from original v1")
    local = _index(load_predictions(v1_dir, "startlux"), examples, "startlux", exact=False)
    job = output_dir / "predictions-jobs/qwen_fit"
    qwen = _job_predictions(v1_dir, output_dir, manifest, examples, "qwen_fit", "qwen")
    calibration = read_json(v1_dir / "calibration/startlux.json")
    try:
        validate_calibration(examples, gold, list(local.values()), calibration, "startlux")
    except ValueError as exc:
        raise StudyError(str(exc)) from exc
    bindings = {"manifest_hash": manifest["manifest_hash"],
                "protocol_hash": fingerprint(protocol), "fit_examples_hash": fingerprint(examples),
                "fit_gold_hash": fingerprint(gold), "local_fit_predictions_hash": fingerprint(local),
                "qwen_fit_predictions_hash": fingerprint(qwen),
                "v1_calibration_hash": file_hash(v1_dir / "calibration/startlux.json"),
                "qwen_fit_config_hash": file_hash(job / "predictions/qwen/config.json")}
    return manifest, protocol, examples, gold, local, qwen, calibration, bindings


def _correct(row: dict, gold: str) -> bool:
    return bool(row["status"] == "ok" and row.get("prediction") == gold)


def _measure(policy: dict, examples: list[dict], gold: dict, local: dict, qwen: dict):
    defer = np.array([should_defer(policy, x, local[x["sample_id"]]) for x in examples], dtype=bool)
    lc = np.array([_correct(local[x["sample_id"]], gold[x["sample_id"]]) for x in examples])
    qc = np.array([_correct(qwen[x["sample_id"]], gold[x["sample_id"]]) for x in examples])
    final = np.where(defer, qc, lc)
    errors = np.array([(qwen if d else local)[x["sample_id"]]["status"] != "ok"
                       for x, d in zip(examples, defer)])
    metric = {"n": len(examples), "correct": int(final.sum()), "accuracy": float(final.mean()),
              "deferrals": int(defer.sum()), "actual_deferral_rate": float(defer.mean()),
              "rescue": int((defer & ~lc & qc).sum()), "harm": int((defer & lc & ~qc).sum()),
              "net_correct_gain": int(final.sum() - lc.sum()), "final_failures": int(errors.sum()),
              "target_budget": policy.get("decision", {}).get("target_budget")}
    return metric, final


def fit(v1_dir: Path, output_dir: Path) -> dict:
    v1_dir, output_dir = Path(v1_dir), Path(output_dir)
    path = output_dir / "policy_freeze.json"
    if path.exists():
        return load_freeze(output_dir, v1_dir=v1_dir)
    if (output_dir / "holdout_access.json").exists() or (output_dir / "evaluation.json").exists():
        raise StudyError("Cannot fit after holdout access; use a new prospective study")
    manifest, protocol, examples, gold, local, qwen, calibration, bindings = _fitting_inputs(v1_dir, output_dir)
    cal = [x for x in examples if x["task"] == "squad" and x["split"] == "calibration"]
    val = [x for x in examples if x["task"] == "squad" and x["split"] == "validation"]
    if not cal or not val:
        raise StudyError("Missing SQuAD calibration/validation rows")
    if {x["group_id"] for x in cal} & {x["group_id"] for x in val}:
        raise StudyError("Calibration/validation article overlap")
    bases = fit_candidates(cal, gold, local, qwen, l2=protocol["policies"]["l2"])
    by_budget = {}
    primary_budget = protocol["policies"]["primary_budget"]
    for budget in [primary_budget, *protocol["policies"]["secondary_budgets"]]:
        candidates = {name: freeze_budget(p, val, local, budget) for name, p in bases.items()}
        metrics = {name: _measure(p, val, gold, local, qwen)[0] for name, p in candidates.items()}
        selected = max(("class_conditional", "gain"), key=lambda name: metrics[name]["correct"])
        by_budget[str(budget)] = {"selected_name": selected, "policies": candidates,
                                  "validation_metrics": metrics}
    primary = by_budget[str(primary_budget)]
    task = calibration["tasks"]["squad"]
    original = {"family": "original", "task": "squad", "labels": cal[0]["labels"],
                "temperature": task["temperature"]["temperature"], "gate": task["gates"]["refitted"],
                "provenance": "unchanged v1 refitted selective-risk gate; secondary reference"}
    artifact = {"schema_version": 1, "protocol": "solo-v2", "status": "frozen",
                "created_at": utc_now(), "v1_dir": str(v1_dir.resolve()),
                "evidence_type": "software_test_fixture" if any(r.get("is_test_fixture") for r in [*local.values(), *qwen.values()]) else "saved_model_predictions",
                "numerics": {"python": platform.python_version(), "numpy": np.__version__},
                "source_hashes": _source_hashes(), "bindings": bindings, "protocol_configuration": protocol,
                "fit_role": "calibration", "selection_role": "validation", "primary_task": "squad",
                "primary_budget": primary_budget, "selected_name": primary["selected_name"],
                "selected_policy": primary["policies"][primary["selected_name"]],
                "confidence_policy": primary["policies"]["confidence"], "original_policy": original,
                "candidates_by_budget": by_budget,
                "note": "Only calibration trains heads; validation freezes online thresholds and candidate choice. Holdout labels were not read."}
    artifact["freeze_hash"] = fingerprint(artifact)
    write_json(path, artifact)
    return artifact


def load_freeze(output_dir: Path, *, v1_dir: Path | None = None) -> dict:
    output_dir = Path(output_dir)
    path = output_dir / "policy_freeze.json"
    if not path.exists():
        raise StudyError("A valid policy freeze is required before holdout evaluation")
    artifact = read_json(path)
    _checked_self_hash(artifact, "freeze_hash")
    if artifact.get("source_hashes") != _source_hashes():
        raise StudyError("Policy source identity changed after freezing")
    v1_dir = Path(v1_dir or artifact["v1_dir"])
    *_, bindings = _fitting_inputs(v1_dir, output_dir)
    if artifact.get("bindings") != bindings:
        raise StudyError("Fitting/source data changed after policy freeze")
    access_path = output_dir / "holdout_access.json"
    if access_path.exists():
        access = read_json(access_path)
        _checked_self_hash(access, "access_hash")
        if access["freeze_hash"] != artifact["freeze_hash"] or access["manifest_hash"] != bindings["manifest_hash"]:
            raise StudyError("Policy/data changed after first holdout access")
    return artifact


def _paired_interval(examples: list[dict], difference: np.ndarray, repetitions: int, seed: int) -> dict:
    groups = defaultdict(list)
    for i, x in enumerate(examples):
        groups[x["group_id"]].append(i)
    if len(groups) < 2:
        raise StudyError("At least two independent articles are needed for the primary interval")
    sizes = np.array([len(groups[k]) for k in sorted(groups)])
    totals = np.array([difference[groups[k]].sum() for k in sorted(groups)])
    rng, draws = np.random.default_rng(seed), np.empty(repetitions)
    for i in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        draws[i] = totals[selected].sum() / sizes[selected].sum()
    return {"estimate": float(difference.mean()), "lower": float(np.quantile(draws, .025)),
            "upper": float(np.quantile(draws, .975)), "confidence_level": .95,
            "method": "paired_article_percentile_bootstrap", "cluster_count": len(groups),
            "replicates": repetitions, "seed": seed,
            "warning": "degenerate_empirical_distribution_not_a_guarantee" if np.ptp(draws) == 0 else None,
            "estimand": "question-weighted accuracy difference under the prespecified balanced holdout sampling",
            "note": "Conditional on frozen policies; no equivalence, power or future-risk guarantee."}


def evaluate(v1_dir: Path, output_dir: Path, *, bootstrap_reps: int | None = None) -> dict:
    output_dir = Path(output_dir)
    artifact = load_freeze(output_dir, v1_dir=Path(v1_dir))
    protocol = artifact["protocol_configuration"]
    repetitions = protocol["evaluation"]["bootstrap_reps"]
    if bootstrap_reps is not None and bootstrap_reps != repetitions:
        raise StudyError("Evaluation repetitions must equal the preregistered protocol")
    examples = data_v2.load_examples(output_dir, role="holdout")
    if not examples or any(x["task"] != "squad" or x["split"] != "holdout" for x in examples):
        raise StudyError("Primary evaluation requires the prepared SQuAD holdout")
    manifest = data_v2.verify_fit(output_dir)
    local = _job_predictions(Path(v1_dir), output_dir, manifest, examples, "holdout_startlux", "startlux")
    qwen = _job_predictions(Path(v1_dir), output_dir, manifest, examples, "holdout_qwen", "qwen")
    access_path = output_dir / "holdout_access.json"
    if not access_path.exists():
        access = {"created_at": utc_now(), "freeze_hash": artifact["freeze_hash"],
                  "manifest_hash": artifact["bindings"]["manifest_hash"],
                  "purpose": "First sealed-label access for fixed-policy evaluation"}
        access["access_hash"] = fingerprint(access)
        write_json(access_path, access)
    data_v2.verify(output_dir)
    gold = read_json(output_dir / "sealed/holdout_gold.json")
    if set(gold) != {x["sample_id"] for x in examples}:
        raise StudyError("Holdout gold does not match exact prepared IDs")
    inputs_hash = fingerprint({"examples": examples, "gold": gold, "startlux": local, "qwen": qwen})
    path = output_dir / "evaluation.json"
    if path.exists():
        previous = read_json(path)
        _checked_self_hash(previous, "evaluation_hash")
        if previous["freeze_hash"] != artifact["freeze_hash"] or previous["inputs_hash"] != inputs_hash:
            raise StudyError("Evaluation identity changed; existing holdout results are immutable")
        _record_receipt(output_dir, previous)
        return previous
    metrics, outcomes = {}, {}
    for name, p in (("selected", artifact["selected_policy"]), ("confidence", artifact["confidence_policy"]),
                    ("original_v1", artifact["original_policy"])):
        metrics[name], outcomes[name] = _measure(p, examples, gold, local, qwen)
    sensitivity = {}
    for budget, group in artifact["candidates_by_budget"].items():
        sensitivity[budget] = {name: _measure(p, examples, gold, local, qwen)[0]
                               for name, p in group["policies"].items()}
    direct = {}
    for model, index in (("startlux", local), ("qwen", qwen)):
        correct = sum(_correct(index[x["sample_id"]], gold[x["sample_id"]]) for x in examples)
        direct[model] = {"n": len(examples), "correct": correct, "accuracy": correct / len(examples),
                         "failures": sum(index[x["sample_id"]]["status"] != "ok" for x in examples)}
    result = {"schema_version": 1, "protocol": "solo-v2", "status": "complete", "created_at": utc_now(),
              "evidence_type": "software_test_fixture" if artifact["evidence_type"] == "software_test_fixture" or any(r.get("is_test_fixture") for r in [*local.values(), *qwen.values()]) else "saved_model_predictions",
              "freeze_hash": artifact["freeze_hash"], "inputs_hash": inputs_hash,
              "manifest_hash": artifact["bindings"]["manifest_hash"], "n": len(examples),
              "selected_name": artifact["selected_name"], "primary_budget": artifact["primary_budget"],
              "primary_comparison": "selected policy minus confidence baseline, both validation-frozen online thresholds",
              "primary_difference": _paired_interval(examples, outcomes["selected"].astype(int) - outcomes["confidence"].astype(int), repetitions, protocol["seed"]),
              "policy_metrics": metrics, "descriptive_direct": direct, "descriptive_budget_sensitivity": sensitivity,
              "limitations": ["Fresh project articles from original train, not proof of pretraining-unseen or same-distribution data.",
                              "Only the primary paired comparison is inferential; direct/budget comparisons are descriptive.",
                              "Actual holdout deferral may differ from validation budget; thresholds were not retuned."]}
    result["evaluation_hash"] = fingerprint(result)
    write_json(path, result)
    _record_receipt(output_dir, result)
    return result


def _record_receipt(output_dir: Path, result: dict) -> None:
    path = output_dir / "evaluation_receipts.jsonl"
    receipt = {"event": "holdout_evaluated", "freeze_hash": result["freeze_hash"],
               "evaluation_hash": result["evaluation_hash"], "inputs_hash": result["inputs_hash"],
               "created_at": result["created_at"]}
    existing = read_jsonl(path)
    if existing:
        if existing != [receipt]:
            raise StudyError("Conflicting holdout evaluation receipt")
        return
    with path.open("ab") as stream:
        stream.write(canonical(receipt) + b"\n")
