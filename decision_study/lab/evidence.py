"""Portable, read-only presentation of the two audited follow-up experiments.

The builder accepts only the recorded final V2/V3 archives. It verifies the bytes
it consumes, replays the original frozen routers, and checks their totals against
the saved evaluations. It never fits, collects predictions, or creates an access
receipt. The small output keeps illustrative cases rather than the full holdout.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from ..common import StudyError, canonical, file_hash, fingerprint
from ..v2.policies import should_defer
from ..v3.policies import route

KIND = "decision-study-multiround-lab"
SOURCES = {
    "v2": {
        "id": "v2-complete-20261103-082852",
        "archive_sha256": "f340f4b159cb21a0df35a6386db1800e263e85fc012ff85cd2449e7d21f2cdec",
        "report_manifest": "reports/v2-complete-2026-11-03/SHA256.json",
        "report_sha256": "41dded9f5943a6a542b14572d461418257d7726d8379688bc91d38c215b8342c",
    },
    "v3": {
        "id": "v3-complete-20261103-232529",
        "archive_sha256": "a8d0dc672d83c3edcb85d005e66fa4b3d000132729f98f0beaab6865dfc57a9e",
        "report_manifest": "reports/v3-complete-2026-11-03/SHA256.json",
        "report_sha256": "764fa702dde7831907c49a3dcb99b6ef988617590ee30e3eb1819564f94aad93",
    },
}
LABELS = {
    "startlux": "StartLux 4B", "startlux4b": "StartLux 4B",
    "qwen": "Qwen 4B", "qwen4b": "Qwen 4B", "qwen9b": "Qwen 9B",
}
FAMILIES = {"confidence": "Confidence", "gain": "Gain", "class_conditional": "Class-conditional",
            "fixed_gain": "Fixed gain", "positive_gain": "Positive gain", "original_v1": "Original V1 gate"}
CASE_NOTE = (
    "Twelve illustrative cases are selected deterministically by saved outcomes under the confidence "
    "reference: up to three each for local-correct retention, rescue, harm, and both models wrong. "
    "They explain routing and are not a representative sample or a new evaluation. "
    "All displayed routes use the original frozen policies; fallback predictions were separately collected."
)


def _require(condition, message):
    if not condition:
        raise StudyError(message)


def _safe(root: Path, relative: str) -> Path:
    rel = Path(relative)
    _require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe evidence path")
    path = root / rel
    _require(not any(x.is_symlink() for x in (path, *list(path.parents)[:len(rel.parts) - 1])),
             "Symlink evidence paths are not accepted")
    return path


def _self_hash(data: dict, key: str):
    _require(data.get(key) == fingerprint({k: v for k, v in data.items() if k != key}),
             f"Changed {key}; preserve the original evidence")


class _Source:
    """Verified access to one fixed, already-audited result archive and its extraction."""

    def __init__(self, root: Path, version: str):
        self.root, self.version, self.used = root, version, {}
        spec = SOURCES[version]
        index = json.loads((root / "records/BUNDLE_INDEX.json").read_text())
        entries = [x for x in index["bundles"] if x.get("id") == spec["id"]]
        _require(len(entries) == 1, "Missing unique completed result receipt")
        entry = entries[0]
        _require(entry.get("reported_state") == "complete" and entry.get("sha256") == spec["archive_sha256"],
                 "Result receipt is not the audited final archive")
        archive = _safe(root, entry["archive"])
        _require(archive.stat().st_size == entry["bytes"] and file_hash(archive) == spec["archive_sha256"],
                 "Original result archive hash/size mismatch")
        self.directory = _safe(root, entry["extracted_directory"])
        with zipfile.ZipFile(archive) as zipped:
            _require(zipped.namelist().count("bundle_manifest.json") == 1, "Archive manifest is ambiguous")
            manifest_bytes = zipped.read("bundle_manifest.json")
        _require((self.directory / "bundle_manifest.json").read_bytes() == manifest_bytes,
                 "Extracted manifest differs from the original archive")
        self.files = json.loads(manifest_bytes)["files"]
        self.used["bundle_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
        report = _safe(root, spec["report_manifest"])
        _require(file_hash(report) == spec["report_sha256"], "Audited report manifest changed")
        self.report_files = json.loads(report.read_text())["sha256"]
        self.report_used = {spec["report_manifest"]: spec["report_sha256"]}
        self.archive_sha256 = spec["archive_sha256"]

    def read(self, relative: str) -> bytes:
        _require(relative in self.files, f"Evidence absent from archive manifest: {relative}")
        data = _safe(self.directory, relative).read_bytes()
        expected = self.files[relative]
        sha = hashlib.sha256(data).hexdigest()
        _require(len(data) == expected["bytes"] and sha == expected["sha256"],
                 f"Evidence hash/size mismatch: {relative}")
        self.used[relative] = sha
        return data

    def json(self, relative: str):
        return json.loads(self.read(relative))

    def audited(self, relative: str) -> bytes:
        _require(relative in self.report_files, "Summary is not in the audited report manifest")
        data = _safe(self.root, relative).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        _require(sha == self.report_files[relative], "Audited summary hash mismatch")
        self.report_used[relative] = sha
        return data


def _rows(source: _Source, job: str, interface: str, role: str, examples: list[dict]) -> dict:
    prefix = f"predictions-jobs/{job}/predictions/{interface}/"
    config = source.json(prefix + "config.json")
    config_sha = fingerprint(config)
    collected = {}
    for relative in sorted(source.files):
        if not relative.startswith(prefix + "shard-") or not relative.endswith(".json"):
            continue
        shard = source.json(relative)
        _require(shard["config_hash"] == config_sha and shard["sha256"] == fingerprint(shard["rows"])
                 and shard["count"] == len(shard["rows"]), "Prediction shard identity mismatch")
        for row in shard["rows"]:
            sid = row["sample_id"]
            _require(sid not in collected and row["model"] == interface and row["config_hash"] == config_sha,
                     "Duplicate or mismatched prediction")
            _require(row["status"] == "ok" and not row.get("is_test_fixture"),
                     "Lab requires the completed real prediction evidence")
            collected[sid] = {**row, "model": role}
    _require(set(collected) == {x["sample_id"] for x in examples}, "Prediction/input IDs do not match")
    for example in examples:
        row = collected[example["sample_id"]]
        _require(row["prediction"] in example["labels"] and row["task"] == example["task"]
                 and row["split"] == "holdout", "Prediction task/split mismatch")
        if "input_hash" in row:
            _require(row["input_hash"] == example["input_hash"], "Prediction input identity mismatch")
    return collected


def _outcome(local_correct: bool, final_correct: bool, deferred: bool) -> str:
    if deferred and final_correct and not local_correct:
        return "rescue"
    if deferred and local_correct and not final_correct:
        return "harm"
    return "correct" if final_correct else "wrong"


def _measure(examples, gold, local, fallback, frozen, version):
    decisions = {}
    n, correct, deferred, rescue, harm = len(examples), 0, 0, 0, 0
    for ex in examples:
        sid = ex["sample_id"]
        handoff = bool(should_defer(frozen, ex, local[sid]) if version == "v2" else route(frozen, ex, local[sid]))
        final = fallback[sid]["prediction"] if handoff else local[sid]["prediction"]
        lc, fc = local[sid]["prediction"] == gold[sid], final == gold[sid]
        outcome = _outcome(lc, fc, handoff)
        decisions[sid] = {"deferred": handoff, "final_prediction": final, "correct": fc, "outcome": outcome}
        correct += fc
        deferred += handoff
        rescue += outcome == "rescue"
        harm += outcome == "harm"
    return {"n": n, "correct": correct, "accuracy": correct/n, "deferrals": deferred,
            "actual_deferral_rate": deferred/n, "rescue": rescue, "harm": harm}, decisions


def _compare_metrics(actual, saved):
    for key, value in actual.items():
        _require(saved.get(key) == value, f"Frozen replay differs from saved evaluation: {key}")


def _case_subset(examples, gold, local, fallback_rows, decisions, reference_id, local_model):
    reference_fallback = "qwen" if local_model == "startlux" else "qwen9b"
    buckets = {"Local correct and retained": [], "Confidence reference rescues": [],
               "Confidence reference harms": [], "Both models wrong": []}
    for ex in sorted(examples, key=lambda x: x["sample_id"]):
        sid = ex["sample_id"]
        ref = decisions[reference_id][sid]
        if not ref["deferred"] and ref["correct"]:
            buckets["Local correct and retained"].append(ex)
        if ref["outcome"] == "rescue":
            buckets["Confidence reference rescues"].append(ex)
        if ref["outcome"] == "harm":
            buckets["Confidence reference harms"].append(ex)
        if local[sid]["prediction"] != gold[sid] and fallback_rows[reference_fallback][sid]["prediction"] != gold[sid]:
            buckets["Both models wrong"].append(ex)
    chosen, seen = [], set()
    for reason, candidates in buckets.items():
        for ex in candidates:
            sid = ex["sample_id"]
            if sid not in seen:
                seen.add(sid)
                chosen.append((ex, reason))
                if sum(r == reason for _, r in chosen) == 3:
                    break
    _require(len(chosen) == 12, "Expected illustrative categories missing; do not silently substitute cases")
    result = []
    for ex, reason in chosen:
        sid, row = ex["sample_id"], local[ex["sample_id"]]
        result.append({"sample_id": sid, "task": ex["task"], "round": ex.get("round"),
                       "state": ex["state"], "gold": gold[sid], "labels": ex["labels"],
                       "input_hash": ex["input_hash"], "selection_reason": reason,
                       "local": {"model": local_model, "label": LABELS[local_model],
                                 "prediction": row["prediction"], "confidence": max(row["probabilities"].values()),
                                 "probabilities": row["probabilities"]},
                       "fallbacks": {key: {"model": key, "label": LABELS[key], "prediction": rows[sid]["prediction"]}
                                     for key, rows in fallback_rows.items()},
                       "routes": {key: routes[sid] for key, routes in decisions.items()}})
    return result


def _performance(source):
    relative = ".validation/v2-complete-2026-11-03-002852/condition-rate-summary.csv"
    rows = []
    for item in csv.DictReader(io.StringIO(source.audited(relative).decode())):
        rows.append({"condition": item["condition"], "offered_rate": float(item["requested_rate"]),
                     "repetitions": int(item["n_repeats"]),
                     "goodput_mean": float(item["goodput_per_second_mean"]),
                     "goodput_min": float(item["goodput_per_second_min"]),
                     "goodput_max": float(item["goodput_per_second_max"]),
                     "latency_p95_ms_mean": float(item["latency_ms_p95_mean"]),
                     "rejection_fraction_mean": float(item["rejection_fraction_all_offered_mean"]),
                     "actual_offered_rate_mean": float(item["actual_offered_rate_mean"])})
    _require(len(rows) == 12 and all(x["repetitions"] == 3 for x in rows), "Incomplete measured performance matrix")
    return {"measured": True, "duration_seconds": 30, "slo_ms": 500, "rows": rows,
            "note": "V2 validation workload, 30-second offered-arrival windows, three repetitions per condition/rate. "
                    "Goodput counts successful requests within 500 ms from scheduled arrival, including dispatch and queue delay. "
                    "Means and min/max are across repetitions, not confidence intervals. Three cascade cells came from one L40S "
                    "allocation and the other 33 cells from another; the original report retains the same-card sensitivity. "
                    "Only the selected gain cascade was measured; these speeds do not apply to other policy rows or V3."}


def _experiment(root, version):
    source = _Source(root, version)
    evaluation, frozen = source.json("evaluation.json"), source.json("policy_freeze.json")
    _require(evaluation.get("status") == "complete", "Final evaluation is required")
    _self_hash(evaluation, "evaluation_hash")
    _self_hash(frozen, "freeze_hash")
    access = source.json("holdout_access.json")
    _require(evaluation["freeze_hash"] == frozen["freeze_hash"] == access.get("freeze_hash"),
             "Evaluation/access/freeze identity mismatch")
    # All identity checks above precede reading the already-evaluated labels.
    examples = [json.loads(line) for line in source.read("data/examples.jsonl").splitlines() if line.strip()]
    examples = [ex for ex in examples if ex["split"] == "holdout"]
    gold = source.json("sealed/holdout_gold.json")
    _require(len(examples) == evaluation["n"] and set(gold) == {x["sample_id"] for x in examples},
             "Evaluated holdout IDs/count differ")
    if version == "v2":
        local_model = "startlux"
        local = _rows(source, "holdout_startlux", "startlux", local_model, examples)
        fallbacks = {"qwen": _rows(source, "holdout_qwen", "qwen", "qwen", examples)}
        policy_specs = {name: ("qwen", policy) for name, policy in frozen["candidates_by_budget"]["0.1"]["policies"].items()}
        policy_specs["original_v1"] = ("qwen", frozen["original_policy"])
        saved = {**evaluation["descriptive_budget_sensitivity"]["0.1"], "original_v1": evaluation["policy_metrics"]["original_v1"]}
        direct = evaluation["descriptive_direct"]
        candidate, reference = frozen["selected_name"], "confidence"
        title, task, formal = "V2 · Fresh SQuAD answerability", "squad", 10800
        notes = ["Policy table uses the prespecified 10% validation target; the original V1 gate is a secondary reference with no target budget.",
                 "Only the gain-minus-confidence paired comparison is primary; direct models and other policies are descriptive.",
                 "Actual holdout deferral differs from validation; no threshold was retuned on the 4,000-question holdout."]
        performance = _performance(source)
    else:
        local_model = "startlux4b"
        local = _rows(source, "startlux4b-holdout", "startlux", local_model, examples)
        fallbacks = {model: _rows(source, model + "-holdout", "qwen", model, examples) for model in ("qwen4b", "qwen9b")}
        policy_specs = {f"{model}/{family}": (model, policy) for model, group in frozen["policies"].items() for family, policy in group.items()}
        saved = {f"{model}/{family}": metrics for model, group in evaluation["policies"].items() for family, metrics in group.items()}
        direct = evaluation["direct"]
        candidate, reference = "qwen9b/positive_gain", "qwen9b/confidence"
        title, task, formal = "V3 · ANLI three-way inference", "anli", 17700
        notes = ["Both positive-gain policies froze mode=none on validation; zero holdout deferral reproduces direct StartLux, rather than identifying complementary cases.",
                 "The experiment separately collected all fallback predictions. Offline zero deferral does not mean there were no GPU fallback calls.",
                 "Confidence froze 10% deferral on validation and reached 23.21875% on holdout; no test-time top-k or retuning was used.",
                 "Only positive-gain9B minus confidence9B is primary; other models, policies and ANLI rounds are descriptive."]
        performance = {"measured": False, "rows": [], "note": "V3 was a sequential quality experiment. No co-resident 9B cascade goodput was measured; serial model times are not a cascade benchmark."}
    decisions, policies, frozen_policies = {}, [], {}
    for name, (fallback, policy) in policy_specs.items():
        actual, routes = _measure(examples, gold, local, fallbacks[fallback], policy, version)
        _compare_metrics(actual, saved[name])
        decisions[name] = routes
        frozen_policies[name] = policy
        decision = policy.get("decision")
        validation_rate = (decision["selection_deferrals"] / decision["selection_n"] if decision else
                           1 - policy["gate"]["validation_accepted"] / policy["gate"]["validation_n"])
        family = "original_v1" if name == "original_v1" else policy["family"]
        policies.append({"id": name, "label": FAMILIES[family], "family": family, "fallback": fallback,
                         **{k: actual[k] for k in ("n", "correct", "accuracy", "deferrals", "rescue", "harm")},
                         "deferral_rate": actual["actual_deferral_rate"], "target_budget": saved[name]["target_budget"],
                         "validation_deferral_rate": validation_rate,
                         "validation_scope": "V1 original gate selection pool" if name == "original_v1" else version.upper() + " validation pool",
                         "decision_mode": decision["mode"] if decision else "original_gate",
                         "is_primary_candidate": name == candidate, "is_primary_reference": name == reference})
    direct_rows = []
    for model, rows in {local_model: local, **fallbacks}.items():
        count = sum(rows[ex["sample_id"]]["prediction"] == gold[ex["sample_id"]] for ex in examples)
        _require(direct[model]["correct"] == count and direct[model]["accuracy"] == count/len(examples),
                 "Direct prediction totals differ from saved evaluation")
        direct_rows.append({"model": model, "label": LABELS[model], "n": len(examples), "correct": count, "accuracy": count/len(examples)})
    primary = evaluation["primary_difference"]
    result = {"id": version, "title": title, "task": task, "n": len(examples), "formal_predictions": formal,
              "primary": {"label": ("Gain − confidence" if version == "v2" else "Positive gain 9B − confidence 9B"),
                          "estimate_pp": 100*primary["estimate"], "lower_pp": 100*primary["lower"], "upper_pp": 100*primary["upper"],
                          **{k: primary[k] for k in ("confidence_level", "cluster_count", "method")},
                          "candidate_id": candidate, "reference_id": reference},
              "direct": direct_rows, "policies": policies, "frozen_policies": frozen_policies,
              "performance": performance, "cases": _case_subset(examples, gold, local, fallbacks, decisions, reference, local_model),
              "case_note": CASE_NOTE, "notes": notes,
              "provenance": {"bundle_sha256": source.archive_sha256, "evaluation_hash": evaluation["evaluation_hash"],
                             "freeze_hash": frozen["freeze_hash"], "source_files": source.used,
                             "audited_report_files": source.report_used}}
    if version == "v3":
        result["rounds"] = evaluation["direct_rounds_descriptive"]
    return result


def _validate_payload(data):
    _require(data.get("schema_version") == 1 and data.get("kind") == KIND
             and data.get("evidence_type") == "saved_model_predictions", "Unsupported lab evidence schema")
    _require(set(data.get("experiments", {})) == {"v2", "v3"}, "Both audited experiments are required")
    for version, experiment in data["experiments"].items():
        _require(experiment["id"] == version and experiment["n"] == {"v2": 4000, "v3": 3200}[version],
                 "Experiment identity/count mismatch")
        _require(experiment["provenance"]["bundle_sha256"] == SOURCES[version]["archive_sha256"],
                 "Unknown source result archive")
        policies = {p["id"]: p for p in experiment["policies"]}
        _require(len(policies) == len(experiment["policies"]) == {"v2": 4, "v3": 8}[version], "Missing/duplicate policy")
        for p in experiment["policies"]:
            _require(p["n"] == experiment["n"] and p["accuracy"] == p["correct"]/p["n"]
                     and p["deferral_rate"] == p["deferrals"]/p["n"], "Policy metric arithmetic mismatch")
            _require(all(isinstance(p[k], int) and 0 <= p[k] <= p["n"] for k in ("correct", "deferrals", "rescue", "harm")),
                     "Invalid policy counts")
        primary = experiment["primary"]
        _require(all(isinstance(primary[k], (int, float)) and math.isfinite(primary[k])
                     for k in ("estimate_pp", "lower_pp", "upper_pp"))
                 and primary["lower_pp"] <= primary["upper_pp"], "Invalid primary interval")
        candidate, reference = policies[primary["candidate_id"]], policies[primary["reference_id"]]
        _require(math.isclose(primary["estimate_pp"], 100*(candidate["correct"] - reference["correct"])/experiment["n"], abs_tol=1e-12),
                 "Primary difference disagrees with policy counts")
        _require(len(experiment["cases"]) == 12 and len({c["sample_id"] for c in experiment["cases"]}) == 12, "Illustrative cases missing or duplicated")
        for case in experiment["cases"]:
            ex = {"sample_id": case["sample_id"], "state": case["state"], "task": case["task"], "labels": case["labels"]}
            local = {"sample_id": case["sample_id"], "model": case["local"]["model"], "status": "ok",
                     "prediction": case["local"]["prediction"], "probabilities": case["local"]["probabilities"]}
            _require(case["gold"] in case["labels"] and set(case["routes"]) == set(policies), "Case schema differs from policies")
            for name, saved_route in case["routes"].items():
                p = experiment["frozen_policies"][name]
                handoff = bool(should_defer(p, ex, local) if version == "v2" else route(p, ex, local))
                final = case["fallbacks"][policies[name]["fallback"]]["prediction"] if handoff else local["prediction"]
                correct = final == case["gold"]
                expected = {"deferred": handoff, "final_prediction": final, "correct": correct,
                            "outcome": _outcome(local["prediction"] == case["gold"], correct, handoff)}
                _require(saved_route == expected, "Case route differs from the frozen policy replay")
        _require(experiment["performance"]["measured"] is (version == "v2"), "Unmeasured performance must remain unavailable")
        if version == "v3":
            _require(not experiment["performance"]["rows"], "V3 has no measured cascade performance")


def build(evidence_root: Path | str, target_dir: Path | str) -> Path:
    """Create a new presentation bundle; source artifacts are never modified."""
    root, target = Path(evidence_root).resolve(), Path(target_dir).resolve()
    _require(not target.exists(), "Output already exists; choose a new bundle directory")
    _require(not target.is_relative_to(root), "Presentation output must be outside the original evidence tree")
    data = {"schema_version": 1, "kind": KIND, "evidence_type": "saved_model_predictions",
            "experiments": {v: _experiment(root, v) for v in ("v2", "v3")}}
    _validate_payload(data)
    payload = canonical(data) + b"\n"
    manifest = {"schema_version": 1, "kind": KIND,
                "files": {"study.json": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}},
                "note": "Checksums detect changes; they are not digital signatures. Original result identities are retained in study.json."}
    target.mkdir(parents=True, exist_ok=False)
    (target / "study.json").write_bytes(payload)
    (target / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return target / "study.json"


def load_bundle(bundle_dir: Path | str) -> dict:
    """Validate a portable bundle without accessing the original evidence tree."""
    directory = Path(bundle_dir)
    manifest = json.loads(_safe(directory, "manifest.json").read_text())
    _require(manifest.get("schema_version") == 1 and manifest.get("kind") == KIND
             and set(manifest.get("files", {})) == {"study.json"}, "Invalid portable bundle manifest")
    content = _safe(directory, "study.json").read_bytes()
    expected = manifest["files"]["study.json"]
    _require(len(content) == expected["bytes"] and hashlib.sha256(content).hexdigest() == expected["sha256"],
             "Portable study bundle hash/size mismatch")
    data = json.loads(content)
    try:
        _validate_payload(data)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise StudyError("Malformed portable study data") from exc
    return data
