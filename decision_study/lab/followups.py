"""Portable, read-only evidence for the follow-up studies that came after V3.

The bundle covers three accepted follow-ups: the CLINC CPU baselines, the CLINC CPU/GPU service study
and the ANLI 9B serving study, plus the retrospective StartLux-9B size control. Its numbers come from
tracked summaries whose SHA-256 values are fixed by acceptance records or report manifests; a few
illustrative ANLI cases come from the original 9B serving archive. Building needs that archive and the
sealed V3 gold (both outside Git); loading the portable bundle needs nothing outside it. Nothing here
fits a model, calls a service or reruns an evaluation, and the frozen V1-V3 bundles are not touched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

from ..common import StudyError, canonical

KIND = "decision-study-followup-lab"
EVIDENCE_TYPE = "accepted_followup_summaries"
SERVING_REPORT = "reports/upgrade-9b-serving-2026-11-09"
ACCEPTANCE = ("records/upgrade-service-acceptance-2026-11-10.json",
              "3bc4c53f566ef9bf87527e613dea34dbd0f4f19f26fd63ce446aadf4e791c604")
FOLLOWUP_MANIFEST = ("reports/followups-2026-11-06/SHA256.json",
                     "da288a23de12abd5c106516753ebfa9c5f6b483bbfa300d096b4f5d446b23f1d")
CHEAP_ACCEPTANCE = ("records/cheap-baselines-acceptance-2026-11-04.json",
                    "a8669440a7e70c30031cdbb5e35b8d26f3884a54895508d1b2189b1429f9dd37")
# Tracked copies of the 9B audit receipts and the acceptance-record keys that fix their hashes.
SERVING_RECEIPTS = {
    f"{SERVING_REPORT}/evidence/metrics/recomputed.json": ".validation/upgrade-return-review-2026-11-10/metrics/recomputed.json",
    f"{SERVING_REPORT}/evidence/metrics/audit-receipt.json": ".validation/upgrade-return-review-2026-11-10/metrics/audit-receipt.json",
    f"{SERVING_REPORT}/evidence/integrity-audit.json": ".validation/upgrade-return-review-2026-11-10/integrity/integrity-audit.json",
    f"{SERVING_REPORT}/evidence/metrics/stall-analysis.json": ".validation/upgrade-return-review-2026-11-10/metrics/stall-analysis.json",
}
SERVICE_AGGREGATES = "reports/followups-2026-11-06/evidence/service/aggregates.csv"
SERVICE_RESOURCES = "reports/followups-2026-11-06/evidence/service/resource-summary.csv"
SIZE_EVALUATION = "reports/followups-2026-11-06/size-evaluation.json"
CHEAP_SUMMARY = "reports/cheap-baselines-2026-11-04/summary.csv"
# Every tracked input and the SHA-256 its acceptance record or report manifest gives it.
EXPECTED_SOURCES = {
    ACCEPTANCE[0]: ACCEPTANCE[1],
    FOLLOWUP_MANIFEST[0]: FOLLOWUP_MANIFEST[1],
    CHEAP_ACCEPTANCE[0]: CHEAP_ACCEPTANCE[1],
    f"{SERVING_REPORT}/evidence/metrics/recomputed.json": "812493682b545e6f98ef439b4ec541f8d43a3e2901db5a9baf16a7ef4eb1f8d7",
    f"{SERVING_REPORT}/evidence/metrics/audit-receipt.json": "e4d0dcc4ed75b5d485c82ee07708f64c3247a30495cd646dcd49e52698452896",
    f"{SERVING_REPORT}/evidence/integrity-audit.json": "99b1d5c0451eced1a09d97e15ecd4322a43238bf38d2b2f0efd19f56c53af4be",
    f"{SERVING_REPORT}/evidence/metrics/stall-analysis.json": "336e8533b4c3019b07c85dcc2ec63b30a8efc8ffbaa72f84501eb088a76fcaf5",
    SERVICE_AGGREGATES: "22cf52dc3bafa0ba663305de2807510c14069d8b58552a5d916ca9a020b5170e",
    SERVICE_RESOURCES: "87dd9acda01f0e9e1a0aa83dc2ca064e5d5c131156d75332916497038ede11df",
    SIZE_EVALUATION: "e0cb367314cc70b7412ea0d37d48357ffba4fac22e7b6b4d4e927c29ebd1ca20",
    CHEAP_SUMMARY: "641e60466633634cb6622ff7d6574673798fbb587dc2226c4c36579945a052f1",
}
BUNDLES = {
    "serving9b": ("upgrade-formal-20261109-180806", "9baeca3eb7f84a86fdc1f63bc16114c1a91c756bbefe323840152d7901f91be7"),
    "clinc_service": ("service-complete-20261106-170135", "d212c130c707a30fc9b99be759e0ec4386000b21fca0792dbe2867b619e0998b"),
    "cpu_baselines": ("cheap-baselines-20261105-073350", "7f00178c7bd1d86eeeb121929131447675d0053c10072f687f88d7e3ae77b2be"),
    "size_control": ("size-control-complete-20261106-171136", "f427f7762bac745d9316cb02195c09f98ab551794de1da047509be7f9fe7d1ed"),
}
GOLD = ("results/solo-v3-prepared/sealed/holdout_gold.json",
        "ed3525b721ed564a4cdf733e94ac381cfebb4ba06eb4a3a6844a1a0b37824f14")

CONDITIONS = ("startlux4b", "startlux9b", "cascade")
RATES = (10, 20, 40, 80)
SLOS = (500, 1000)
REPEATS = (0, 1, 2)
WINDOW_SECONDS = 120
TIMELY_RULE = 0.99
ACCEPTED_TOTALS = {"offered": 162801, "successful": 131935, "client_rejected": 30866}
ACCEPTED_PASSING = {"500": {"startlux4b": [10, 40], "startlux9b": [10, 20], "cascade": []},
                    "1000": {"startlux4b": [10, 40], "startlux9b": [10, 20, 40], "cascade": [10]}}
# Median correct-and-timely responses per 120-second window at 10/20/40/80 requests/s, from the
# accepted receipt (records/upgrade-service-acceptance-2026-11-10.json); goodput = count / 120.
ACCEPTED_MEDIAN_COUNTS = {"500": {"startlux4b": [871, 1782, 3595, 4591], "startlux9b": [896, 1840, 3690, 35],
                                  "cascade": [834, 1531, 44, 18]},
                          "1000": {"startlux4b": [871, 1782, 3595, 7009], "startlux9b": [896, 1840, 3723, 101],
                                   "cascade": [877, 1747, 109, 30]}}
DEPLOYMENTS = [
    {"id": "startlux4b", "label": "StartLux-4B direct", "short": "StartLux-4B", "gpu_share": "0.40",
     "services": "StartLux-Decision-v1.1-4B-Instant"},
    {"id": "startlux9b", "label": "StartLux-9B direct", "short": "StartLux-9B", "gpu_share": "0.50",
     "services": "StartLux-Decision-v1.1-9B-Instant"},
    {"id": "cascade", "label": "StartLux-4B → Qwen3.5-9B cascade", "short": "4B → Qwen-9B cascade",
     "gpu_share": "0.40 + 0.50", "services": "StartLux-4B front, Qwen3.5-9B fallback (frozen V3 class-conditional router)"},
]
CLINC_CONDITIONS = ("minilm400", "minilm1600", "startlux4b", "qwen4b")
CLINC_LABELS = {"minilm400": "MiniLM-400 (CPU)", "minilm1600": "MiniLM-1600 (CPU)",
                "startlux4b": "StartLux-4B (GPU)", "qwen4b": "Qwen3.5-4B (GPU)"}
CLINC_RATES = (5, 25, 100)
BASELINE_MODELS = ("tfidf400", "tfidf1600", "minilm400", "minilm1600", "kev", "qwen", "startlux")
BASELINE_LABELS = {"tfidf400": "TF-IDF + LR, 400 labels", "tfidf1600": "TF-IDF + LR, 1,600 labels",
                   "minilm400": "MiniLM + LR, 400 labels", "minilm1600": "MiniLM + LR, 1,600 labels",
                   "kev": "Kev-4B (decision model)", "qwen": "Qwen3.5-4B (generative)",
                   "startlux": "StartLux-4B (decision model)"}
ANLI_LABELS = ("entailment", "neutral", "contradiction")
CASE_KINDS = ("upgrade_fixes", "fallback_fixes", "fallback_harms", "late_correct")
CASE_NOTE = (
    "Four cases are chosen by their known outcome to illustrate one pattern each. The first three are the "
    "first sample ID in sorted order among the 3,200 saved test predictions where the larger front model "
    "fixes an error the router kept, where the fallback fixes an error the larger model also makes, and "
    "where the fallback breaks an answer both direct models got right; their counts are out of those 3,200 "
    "inputs. The fourth is the first correct live StartLux-9B answer that arrived after 1,000 ms in the first "
    "80 requests/s repeat; its count is out of the successful live requests in that cell, where inputs repeat. "
    "The cases explain the comparison and are not a representative sample."
)
SERVING_NOTES = [
    "Correct goodput counts the requests scheduled within the 120-second arrival window whose correct label came "
    "back within that request's own SLO, divided by 120 seconds; a timely answer may complete just after the window "
    "ends. Rejected and late requests stay in the offered base. It is measured from returned labels.",
    "The 99% rule asks every repeat to answer at least 99% of offered requests within the SLO. Passing rates are "
    "tested points; nothing between them was measured.",
    "The three deployments use different GPU memory shares (0.40, 0.50, 0.90 in total), so differences describe "
    "these fixed deployments and do not isolate model size, architecture or co-residency.",
    "Quality is retrospective on the ANLI test already scored in V3. Three repeats describe serving variability; "
    "repeated requests are not new independent quality samples.",
]


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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class _Tracked:
    """Read tracked repository files, each checked against its accepted SHA-256."""

    def __init__(self, repo: Path):
        self.repo, self.used = repo, {}
        acceptance = json.loads(self.read(ACCEPTANCE[0]))
        _require(acceptance.get("status") == "PASS", "9B serving acceptance receipt is not PASS")
        accepted = acceptance["audit_files_sha256"]
        for path, key in SERVING_RECEIPTS.items():
            _require(accepted.get(key) == EXPECTED_SOURCES[path], f"Acceptance receipt does not fix {path}")
        manifest = json.loads(self.read(FOLLOWUP_MANIFEST[0]))["sha256"]
        for path in (SERVICE_AGGREGATES, SERVICE_RESOURCES, SIZE_EVALUATION):
            _require(manifest.get(path) == EXPECTED_SOURCES[path], f"Follow-up report manifest does not fix {path}")
        cheap = json.loads(self.read(CHEAP_ACCEPTANCE[0]))["report_files_sha256"]
        _require(cheap.get(CHEAP_SUMMARY) == EXPECTED_SOURCES[CHEAP_SUMMARY], "Baseline acceptance does not fix the summary")

    def read(self, relative: str) -> bytes:
        data = _safe(self.repo, relative).read_bytes()
        _require(_sha(data) == EXPECTED_SOURCES[relative], f"Accepted summary changed: {relative}")
        self.used[relative] = EXPECTED_SOURCES[relative]
        return data

    def json(self, relative: str):
        return json.loads(self.read(relative))

    def csv(self, relative: str) -> list[dict]:
        return list(csv.DictReader(io.StringIO(self.read(relative).decode())))


def _median_range(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def _serving(tracked: _Tracked) -> dict:
    recomputed = tracked.json(f"{SERVING_REPORT}/evidence/metrics/recomputed.json")
    receipt = tracked.json(f"{SERVING_REPORT}/evidence/metrics/audit-receipt.json")
    integrity = tracked.json(f"{SERVING_REPORT}/evidence/integrity-audit.json")
    stall = tracked.json(f"{SERVING_REPORT}/evidence/metrics/stall-analysis.json")
    _require(receipt["status"] == "PASS" and not receipt["discrepancies"] and integrity["status"] == "PASS",
             "9B serving audits are not PASS")
    peaks = {x["cell_id"]: x["gpu_peak_mib"] / 1024 for x in integrity["cell_inventory"]}
    cells = []
    for c in recomputed["cells"]:
        p, s, d = c["primary"], c["secondary"], c["details"]
        _require(p["failed_admitted"] == 0 and p["offered"] == p["successful"] + p["rejected"], "Cell counts mismatch")
        cells.append({
            "cell_id": c["cell_id"], "condition": c["condition"], "rate": c["rate"], "repeat": c["repeat"],
            "offered": p["offered"], "successful": p["successful"], "rejected": p["rejected"], "correct": p["correct"],
            "slo": {str(slo): {"met": m["met_slo"], "correct_met": m["correct_met_slo"],
                               "correct_goodput": m["correct_goodput_per_second"],
                               "timely_fraction": m["slo_fraction_all_offered"]}
                    for slo, m in ((500, p), (1000, s))},
            "successful_per_second": p["successful_per_offered_window_second"],
            "latency_ms": p["latency_ms_all_admitted"],
            "live_accuracy": d["history"]["live_accuracy"],
            "matched_historical_accuracy": d["history"]["matched_subset_historical_accuracy"],
            "deferral_rate": d["cascade"]["deferral_rate_routed"] if c["condition"] == "cascade" else None,
            "sampled_gpu_peak_gib": peaks[c["cell_id"]],
        })
    cells.sort(key=lambda x: (CONDITIONS.index(x["condition"]), x["rate"], x["repeat"]))
    quality = []
    for condition in CONDITIONS:
        live = receipt["live_history_comparison"][condition]
        quality.append({
            "condition": condition,
            "historical_full_pool_accuracy": receipt["historical_full_pool_accuracy"][condition],
            "successful_responses": live["successful_response_count"],
            "pooled_live_accuracy": live["pooled_live_accuracy"],
            "pooled_matched_historical_accuracy": live["pooled_matched_historical_accuracy"],
            "live_accuracy_cell_range": live["live_accuracy_per_cell_range"],
            "inputs_with_more_than_one_live_label": receipt["label_variability"][condition]["inputs_with_more_than_one_live_label"],
        })
    resources = []
    for condition in CONDITIONS:
        for role in (["startlux4b", "qwen9b"] if condition == "cascade" else [condition]):
            launches = [x for x in integrity["launches"] if x["condition"] == condition and x["role"] == role]
            formal = [x for x in launches if not x["scope"].startswith("preflight")]
            preflight = [x for x in launches if x["scope"].startswith("preflight")]
            _require(len(formal) == 12 and len(preflight) == 1, "Unexpected engine start count")
            resources.append({
                "condition": condition, "role": role, "gpu_share": "0.40" if role == "startlux4b" else "0.50",
                "formal_starts": len(formal),
                "formal_kv_gib": sorted({x["kv_gib"][0] for x in formal}),
                "formal_kv_tokens": sorted({x["kv_tokens"][0] for x in formal}),
                "preflight_kv_gib": preflight[0]["kv_gib"][0], "preflight_kv_tokens": preflight[0]["kv_tokens"][0],
            })
    pre = integrity["preflight_summary"]
    timings = stall["late_offers_timings"]
    return {
        "title": "ANLI · a larger front model or a 9B fallback",
        "window_seconds": WINDOW_SECONDS, "rates": list(RATES), "slos_ms": list(SLOS), "repeats": len(REPEATS),
        "timely_rule": TIMELY_RULE, "deployments": DEPLOYMENTS, "cells": cells,
        "summary": _summary(cells), "passing_rates": _passing(cells),
        "totals": {"offered": receipt["offered_total"], "successful": receipt["status_counts"]["ok"],
                   "client_rejected": receipt["status_counts"]["client_rejected"],
                   "stage_predictions_checked": receipt["raw_checks"]["stage_predictions_match_raw_HTTP_response"],
                   "cascade_routes_replayed": receipt["raw_checks"]["cascade_routes_replayed_from_probabilities"]},
        "historical_deferral": {"deferred": 766, "n": 3200},
        "quality": quality, "resources": resources,
        "preflight": {
            "note": "Development preflight on 2026-11-09, about eight hours before the formal cells; forced deferral in the cascade check.",
            "concurrent_offers": pre["cascade"]["concurrent_successful"],
            "timely_500": {c: pre[c]["timely_500ms"] for c in CONDITIONS},
            "latency_p95_ms": {c: pre[c]["latency_ms"]["p95"] for c in CONDITIONS},
            "pairwise_http_stage_overlap_seconds": pre["cascade"]["pairwise_http_stage_overlap_seconds"],
            "simultaneous_http_union_seconds": pre["cascade"]["wall_clock_http_stage_overlap_union_seconds"],
        },
        "anomaly": {
            "cell_id": stall["cell_id"], "condition": "startlux4b", "rate": 20, "repeat": 0,
            "late_500": stall["late500_count"], "client_rejections": 1,
            "arrival_window_seconds": [timings["arrival_seconds"]["min"], timings["arrival_seconds"]["max"]],
            "client_queue_median_ms": timings["client_queue_ms"]["median"],
            "model_call_median_ms": timings["stage_latency_sum_ms"]["median"],
            "max_outside_model_call_ms": stall["all_offers_max_worker_outside_stage_ms"],
            "note": "Kept as measured under the protocol; the wait sat outside the recorded model-call timing and its cause was not isolated.",
        },
        "notes": SERVING_NOTES,
    }


def _summary(cells: list[dict]) -> list[dict]:
    rows = []
    for slo in SLOS:
        for condition in CONDITIONS:
            for rate in RATES:
                group = sorted((c for c in cells if c["condition"] == condition and c["rate"] == rate),
                               key=lambda c: c["repeat"])
                goodput = [c["slo"][str(slo)]["correct_goodput"] for c in group]
                timely = [c["slo"][str(slo)]["timely_fraction"] for c in group]
                rows.append({"condition": condition, "rate": rate, "slo_ms": slo,
                             "correct_goodput": {**_median_range(goodput), "repeats": goodput},
                             "timely_fraction": {"min": min(timely), "repeats": timely},
                             "passes_rule": all(t >= TIMELY_RULE for t in timely),
                             "successful_per_second_median": statistics.median(c["successful_per_second"] for c in group),
                             "latency_p95_ms_median": statistics.median(c["latency_ms"]["p95"] for c in group),
                             "offered": sum(c["offered"] for c in group),
                             "client_rejected": sum(c["rejected"] for c in group),
                             "deferral_rate": (_median_range([c["deferral_rate"] for c in group])
                                               if condition == "cascade" else None)})
    return rows


def _passing(cells: list[dict]) -> dict:
    return {str(slo): {condition: [rate for rate in RATES if all(
        c["slo"][str(slo)]["timely_fraction"] >= TIMELY_RULE
        for c in cells if c["condition"] == condition and c["rate"] == rate)] for condition in CONDITIONS}
        for slo in SLOS}


def _clinc_service(tracked: _Tracked) -> dict:
    resources = defaultdict(list)
    for r in tracked.csv(SERVICE_RESOURCES):
        resources[(r["condition"], int(float(r["rate"])))].append(r)
    rows = []
    for r in tracked.csv(SERVICE_AGGREGATES):
        key = (r["condition"], int(float(r["rate"])))
        res = resources[key]
        _require(len(res) == int(r["repeats"]) == 5, "Service resource rows missing")
        gpu = [float(x["gpu_memory_sample_peak_mib"]) / 1024 for x in res]
        rows.append({
            "condition": key[0], "rate": key[1], "repeats": int(r["repeats"]),
            "offered": int(r["offered_total"]), "successful": int(r["successful_total"]),
            "rejected": int(r["rejected_total"]), "correct_met": int(r["correct_met_slo_total"]),
            "correct_goodput": {k: float(r[f"correct_goodput_per_second_{k}"]) for k in ("median", "min", "max")},
            "successful_per_second_median": float(r["successful_per_offered_window_second_median"]),
            "latency_p95_ms_median": float(r["latency_p95_ms_median"]),
            "accuracy_successful_median": float(r["accuracy_successful_median"]),
            "cpu_core_equivalents_median": statistics.median(float(x["service_observed_cpu_core_equivalents"]) for x in res),
            "sampled_gpu_peak_gib_max": max(gpu) if max(gpu) > 0 else None,
        })
    rows.sort(key=lambda x: (CLINC_CONDITIONS.index(x["condition"]), x["rate"]))
    return {"title": "CLINC · CPU classifier versus GPU services", "rates": list(CLINC_RATES), "repeats": 5,
            "window_seconds": 180, "slo_ms": 500, "labels": CLINC_LABELS, "rows": rows,
            "note": "Four direct services on the 600 CLINC main-test inputs, five 180-second repeats per rate, 500 ms SLO "
                    "from scheduled arrival. CPU services ran one inference thread; GPU services used a 0.40 memory share. "
                    "Ranges are min-max across repeats. The test set had been scored before, so quality is retrospective."}


def _cpu_baselines(tracked: _Tracked) -> dict:
    rows = []
    for r in tracked.csv(CHEAP_SUMMARY):
        rows.append({"model": r["model"], "label": BASELINE_LABELS[r["model"]],
                     "main_correct": int(r["main_correct"]), "main_count": int(r["main_count"]),
                     "banking_correct": int(r["banking_correct"]), "banking_count": int(r["banking_count"]),
                     "oos_correct": int(r["main_oos_correct"]), "oos_count": int(r["main_oos_count"]),
                     "mixed_accuracy": float(r["mixed_accuracy"])})
    rows.sort(key=lambda x: BASELINE_MODELS.index(x["model"]))
    return {"title": "CLINC · cheap supervised classifiers", "rows": rows,
            "note": "CLINC banking intents with out-of-scope queries: 600 main-test inputs (450 banking, 150 out of scope); "
                    "mixed is the 1,300-input stress workload with 65% out of scope. CPU classifiers were trained on 400 or "
                    "1,600 labelled examples and evaluated retrospectively; LLM rows are the frozen V1 predictions."}


def _size_control(tracked: _Tracked) -> dict:
    ev = tracked.json(SIZE_EVALUATION)
    _require(ev["status"] == "complete" and ev["n"] == 3200, "Size control evaluation incomplete")
    diff = ev["comparisons_descriptive"]["startlux9b_minus_startlux4b"]
    return {"n": ev["n"], "direct": {m: {"correct": ev["direct"][m]["correct"], "accuracy": ev["direct"][m]["accuracy"]}
                                     for m in ("startlux4b", "startlux9b", "qwen9b")},
            "difference_pp": {k: 100 * diff["difference"][k] for k in ("estimate", "lower", "upper")},
            "only_9b_correct": diff["candidate_only_correct"], "only_4b_correct": diff["reference_only_correct"],
            "note": "Retrospective direct-model comparison on the 3,200 ANLI test items scored in V3; paired 95% "
                    "premise-cluster bootstrap. No routing or fitting."}


class _Archive:
    """The original 9B serving archive, opened only after its SHA-256 matches the accepted bundle."""

    def __init__(self, repo: Path, evidence_root: Path):
        index = json.loads(_safe(repo, "records/BUNDLE_INDEX.json").read_text())
        bundle_id, digest = BUNDLES["serving9b"]
        entries = [x for x in index["bundles"] if x.get("id") == bundle_id]
        _require(len(entries) == 1 and entries[0]["sha256"] == digest and entries[0]["reported_state"] == "complete",
                 "Bundle index lacks the accepted 9B serving archive")
        path = _safe(evidence_root, entries[0]["archive"])
        _require(path.is_file(), f"9B serving archive not found under the evidence root: {entries[0]['archive']}")
        _require(path.stat().st_size == entries[0]["bytes"] and _sha(path.read_bytes()) == digest,
                 "9B serving archive hash/size mismatch")
        self.zip, self.used = zipfile.ZipFile(path), {}
        self.settings = json.loads(self.member("settings.json"))
        self.freeze = json.loads(self.member("freeze.json"))

    def member(self, name: str) -> bytes:
        data = self.zip.read(name)
        self.used[name] = _sha(data)
        return data

    def bound(self, name: str) -> bytes:
        data = self.member(name)
        _require(self.settings["file_sha256"].get(name) == _sha(data), f"Archive member not bound by settings: {name}")
        return data

    def traces(self, cell_id: str) -> list[dict]:
        folder = f"cells/{cell_id}/attempt-0001"
        complete = self.member(folder + "/complete.json")
        _require(self.freeze["cells"].get(folder + "/complete.json") == _sha(complete), "Cell not bound by freeze")
        data = self.member(folder + "/traces.jsonl")
        _require(json.loads(complete)["files"]["traces.jsonl"] == _sha(data), "Trace not bound by completion receipt")
        return [json.loads(line) for line in data.splitlines() if line.strip()]


def _cases(repo: Path, evidence_root: Path, serving: dict) -> tuple[list[dict], dict]:
    archive = _Archive(repo, evidence_root)
    gold_bytes = _safe(evidence_root, GOLD[0]).read_bytes()
    _require(_sha(gold_bytes) == GOLD[1], "Sealed ANLI gold hash mismatch")
    gold = json.loads(gold_bytes)
    workload = {x["sample_id"]: x for x in json.loads(archive.bound("workload.json"))}
    history = json.loads(archive.bound("history.json"))
    _require(set(workload) == set(gold) == set(history["startlux4b"]) and len(gold) == 3200, "Case inputs incomplete")
    _require(sum(r["deferred"] for r in history["startlux4b"].values()) == serving["historical_deferral"]["deferred"],
             "Historical route count differs")

    def saved(sid):
        front, big, fallback = history["startlux4b"][sid], history["startlux9b"][sid], history["qwen9b"][sid]
        return front, big, fallback, front["deferred"]

    rules = {
        "upgrade_fixes": lambda f, b, q, d, g: f["prediction"] != g and b["prediction"] == g and not d,
        "fallback_fixes": lambda f, b, q, d, g: f["prediction"] != g and d and q["prediction"] == g and b["prediction"] != g,
        "fallback_harms": lambda f, b, q, d, g: f["prediction"] == g and d and q["prediction"] != g and b["prediction"] == g,
    }
    matches = {kind: [sid for sid in sorted(gold) if rule(*saved(sid), gold[sid])] for kind, rule in rules.items()}
    # The wider group each pattern belongs to, so a subset count is never read as the whole group.
    contexts = {
        "upgrade_fixes": ("saved test items StartLux-9B gets right and StartLux-4B gets wrong",
                          lambda f, b, q, d, g: f["prediction"] != g and b["prediction"] == g),
        "fallback_fixes": ("saved test items the frozen cascade rescues",
                           lambda f, b, q, d, g: f["prediction"] != g and d and q["prediction"] == g),
        "fallback_harms": ("saved test items the frozen cascade harms",
                           lambda f, b, q, d, g: f["prediction"] == g and d and q["prediction"] != g),
    }
    context_counts = {kind: sum(rule(*saved(sid), gold[sid]) for sid in gold) for kind, (_, rule) in contexts.items()}
    chosen = {kind: ids[0] for kind, ids in matches.items() if ids}
    _require(set(chosen) == set(rules), "An illustrative category has no example; do not substitute cases")

    live = defaultdict(lambda: defaultdict(lambda: {"successful": 0, "labels": Counter(), "timely_500": 0}))
    late_cell, late = "r0-startlux9b-rate80", None
    for cell in serving["cells"]:
        rows = archive.traces(cell["cell_id"])
        _require(len(rows) == cell["offered"], "Trace count differs from the accepted cell")
        for row in rows:
            if row["status"] != "ok":
                continue
            sid = row["sample_id"]
            if sid in chosen.values():
                entry = live[sid][cell["condition"]]
                entry["successful"] += 1
                entry["labels"][row["result"]["prediction"]] += 1
                entry["timely_500"] += row["arrival_to_completion_ms"] <= 500
        if cell["cell_id"] == late_cell:
            candidates = sorted((r for r in rows if r["status"] == "ok" and r["arrival_to_completion_ms"] > 1000
                                 and r["result"]["prediction"] == gold[r["sample_id"]]), key=lambda r: r["offer_id"])
            _require(candidates, "No late correct response in the selected cell")
            late, late_count = candidates[0], len(candidates)

    titles = {"upgrade_fixes": "The larger front model fixes an error the router kept",
              "fallback_fixes": "The fallback fixes an error the larger model also makes",
              "fallback_harms": "The fallback breaks an answer both direct models got right"}
    cases = []
    for kind in ("upgrade_fixes", "fallback_fixes", "fallback_harms"):
        sid = chosen[kind]
        front, big, fallback, deferred = saved(sid)
        item = workload[sid]
        cases.append({
            "id": kind.replace("_", "-"), "kind": kind, "title": titles[kind], "sample_id": sid, "round": item["round"],
            "state": item["state"], "labels": item["labels"], "gold": gold[sid],
            "pattern_count": len(matches[kind]), "pattern_base": len(gold), "pattern_unit": "saved test items",
            "context": {"label": contexts[kind][0], "count": context_counts[kind]},
            "saved": {"startlux4b": {"prediction": front["prediction"], "probabilities": front["probabilities"]},
                      "startlux9b": {"prediction": big["prediction"], "probabilities": big["probabilities"]},
                      "qwen9b": {"prediction": fallback["prediction"]}, "deferred": deferred,
                      "cascade_final": fallback["prediction"] if deferred else front["prediction"]},
            "live": {c: {"successful": v["successful"], "labels": dict(sorted(v["labels"].items())), "timely_500": v["timely_500"]}
                     for c, v in sorted(live[sid].items(), key=lambda kv: CONDITIONS.index(kv[0]))},
        })
    item = workload[late["sample_id"]]
    stage_ms = sum(s["latency_ms"] for s in late["result"]["stages"])
    cases.append({
        "id": "late-correct", "kind": "late_correct", "title": "A correct answer that arrived too late to count",
        "sample_id": late["sample_id"], "round": item["round"], "state": item["state"], "labels": item["labels"],
        "gold": gold[late["sample_id"]], "cell_id": late_cell, "offer_id": late["offer_id"], "condition": "startlux9b",
        "rate": 80, "repeat": 0, "pattern_count": late_count, "pattern_unit": "successful live requests in this cell",
        "pattern_base": next(c["successful"] for c in serving["cells"] if c["cell_id"] == late_cell),
        "trace": {"prediction": late["result"]["prediction"], "arrival_seconds": late["arrival_seconds"],
                  "client_queue_ms": late["client_queue_ms"], "model_call_ms": stage_ms,
                  "arrival_to_completion_ms": late["arrival_to_completion_ms"]},
    })
    provenance = {"archive_members": dict(sorted(archive.used.items())), "gold_sha256": GOLD[1]}
    return cases, provenance


def build(evidence_root: Path | str, target_dir: Path | str, repo_root: Path | str) -> Path:
    """Create a new portable follow-up bundle; no source artifact is modified."""
    repo, root, target = Path(repo_root).resolve(), Path(evidence_root).resolve(), Path(target_dir).resolve()
    _require(not target.exists(), "Output already exists; choose a new bundle directory")
    _require(not target.is_relative_to(root), "Presentation output must be outside the original evidence tree")
    tracked = _Tracked(repo)
    serving = _serving(tracked)
    cases, case_sources = _cases(repo, root, serving)
    data = {"schema_version": 1, "kind": KIND, "evidence_type": EVIDENCE_TYPE,
            "serving9b": serving, "clinc_service": _clinc_service(tracked), "cpu_baselines": _cpu_baselines(tracked),
            "size_control": _size_control(tracked), "cases": cases, "case_note": CASE_NOTE,
            "provenance": {"bundles": {k: {"id": v[0], "sha256": v[1]} for k, v in BUNDLES.items()},
                           "tracked_sources": dict(sorted(tracked.used.items())),
                           "serving_archive": case_sources,
                           "note": "Tracked sources are checked against acceptance records and report manifests; "
                                   "archive members against the accepted archive's settings, freeze and completion receipts."}}
    validate(data)
    payload = canonical(data) + b"\n"
    manifest = {"schema_version": 1, "kind": KIND,
                "files": {"followups.json": {"sha256": _sha(payload), "bytes": len(payload)}},
                "note": "Checksums detect changes; they are not digital signatures. Source identities are in followups.json."}
    target.mkdir(parents=True, exist_ok=False)
    (target / "followups.json").write_bytes(payload)
    (target / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return target / "followups.json"


def _check_cell(cell: dict):
    counts = ("offered", "successful", "rejected", "correct")
    _require(all(isinstance(cell[k], int) and not isinstance(cell[k], bool) and cell[k] >= 0 for k in counts),
             "Invalid cell counts")
    _require(cell["offered"] == cell["successful"] + cell["rejected"] and cell["correct"] <= cell["successful"],
             "Cell count arithmetic mismatch")
    _require(set(cell["slo"]) == {str(s) for s in SLOS}, "Cell SLO inventory mismatch")
    for slo, m in cell["slo"].items():
        _require(isinstance(m["met"], int) and isinstance(m["correct_met"], int)
                 and 0 <= m["correct_met"] <= min(m["met"], cell["correct"]) and m["met"] <= cell["successful"],
                 "Cell SLO count arithmetic mismatch")
        _require(_finite(m["correct_goodput"]) and _finite(m["timely_fraction"]), "Non-finite cell metric")
        _require(math.isclose(m["correct_goodput"], m["correct_met"] / WINDOW_SECONDS, rel_tol=0, abs_tol=1e-12),
                 "Correct goodput is not correct-and-timely responses per window second")
        _require(math.isclose(m["timely_fraction"], m["met"] / cell["offered"], rel_tol=0, abs_tol=1e-12),
                 "Timely fraction is not timely responses per offered request")
    for key in ("successful_per_second", "live_accuracy", "matched_historical_accuracy", "sampled_gpu_peak_gib"):
        _require(_finite(cell[key]), "Non-finite cell metric")
    _require(all(_finite(cell["latency_ms"][q]) for q in ("p50", "p95", "p99")), "Non-finite latency")
    _require(math.isclose(cell["successful_per_second"], cell["successful"] / WINDOW_SECONDS, rel_tol=0, abs_tol=1e-12),
             "Completed-request rate mismatch")
    _require(math.isclose(cell["live_accuracy"], cell["correct"] / cell["successful"], rel_tol=0, abs_tol=1e-12),
             "Live accuracy mismatch")
    _require((cell["deferral_rate"] is None) == (cell["condition"] != "cascade")
             and (cell["deferral_rate"] is None or 0 <= cell["deferral_rate"] <= 1), "Invalid deferral rate")


def _validate_serving(s: dict):
    _require(s["rates"] == list(RATES) and s["slos_ms"] == list(SLOS) and s["repeats"] == len(REPEATS)
             and s["window_seconds"] == WINDOW_SECONDS and s["timely_rule"] == TIMELY_RULE, "Serving design mismatch")
    _require([d["id"] for d in s["deployments"]] == list(CONDITIONS), "Deployment inventory mismatch")
    inventory = [(c["condition"], c["rate"], c["repeat"]) for c in s["cells"]]
    _require(len(inventory) == 36 and set(inventory) == {(c, r, k) for c in CONDITIONS for r in RATES for k in REPEATS},
             "Expected exactly the 36 measured cells")
    _require(all(c["cell_id"] == f'r{c["repeat"]}-{c["condition"]}-rate{c["rate"]}' for c in s["cells"]), "Cell IDs mismatch")
    for cell in s["cells"]:
        _check_cell(cell)
    totals = {"offered": sum(c["offered"] for c in s["cells"]), "successful": sum(c["successful"] for c in s["cells"]),
              "client_rejected": sum(c["rejected"] for c in s["cells"])}
    _require(totals == ACCEPTED_TOTALS and all(s["totals"][k] == v for k, v in totals.items()),
             "Totals differ from the accepted 9B serving study")
    _require(s["summary"] == _summary(s["cells"]), "Summary differs from the 36 cells")
    _require(s["passing_rates"] == _passing(s["cells"]) == ACCEPTED_PASSING, "99% rule pass sets differ from the accepted study")
    for row in s["summary"]:
        count = ACCEPTED_MEDIAN_COUNTS[str(row["slo_ms"])][row["condition"]][RATES.index(row["rate"])]
        _require(math.isclose(row["correct_goodput"]["median"], count / WINDOW_SECONDS, rel_tol=0, abs_tol=1e-12),
                 "Median correct goodput differs from the accepted study")
    _require([q["condition"] for q in s["quality"]] == list(CONDITIONS)
             and all(_finite(q[k]) and 0 <= q[k] <= 1 for q in s["quality"]
                     for k in ("historical_full_pool_accuracy", "pooled_live_accuracy", "pooled_matched_historical_accuracy")),
             "Invalid quality context")
    _require(len(s["resources"]) == 4 and all(_finite(x) and x > 0 for r in s["resources"]
                                               for x in (*r["formal_kv_gib"], r["preflight_kv_gib"])), "Invalid resource context")


def _validate_clinc(service: dict, baselines: dict, size: dict):
    inventory = [(r["condition"], r["rate"]) for r in service["rows"]]
    _require(service["rates"] == list(CLINC_RATES) and len(inventory) == 12
             and set(inventory) == {(c, r) for c in CLINC_CONDITIONS for r in CLINC_RATES}, "CLINC service inventory mismatch")
    for r in service["rows"]:
        g = r["correct_goodput"]
        _require(r["repeats"] == 5 and r["offered"] == r["successful"] + r["rejected"] and r["correct_met"] <= r["successful"],
                 "CLINC service count arithmetic mismatch")
        _require(all(_finite(g[k]) for k in ("median", "min", "max")) and g["min"] <= g["median"] <= g["max"],
                 "Invalid CLINC goodput range")
        _require(all(_finite(r[k]) for k in ("successful_per_second_median", "latency_p95_ms_median",
                                              "accuracy_successful_median", "cpu_core_equivalents_median")),
                 "Non-finite CLINC service metric")
    _require([r["model"] for r in baselines["rows"]] == list(BASELINE_MODELS), "Baseline inventory mismatch")
    for r in baselines["rows"]:
        _require(r["main_count"] == 600 and r["banking_count"] == 450 and r["oos_count"] == 150
                 and 0 <= r["main_correct"] <= 600 and r["banking_correct"] + r["oos_correct"] == r["main_correct"],
                 "Baseline count arithmetic mismatch")
    d = size["difference_pp"]
    _require(size["n"] == 3200 and all(_finite(d[k]) for k in d) and d["lower"] <= d["estimate"] <= d["upper"]
             and math.isclose(d["estimate"], 100 * (size["direct"]["startlux9b"]["correct"] - size["direct"]["startlux4b"]["correct"]) / 3200,
                              rel_tol=0, abs_tol=1e-9), "Invalid size-control comparison")


def _validate_cases(cases: list[dict]):
    _require(2 <= len(cases) <= 4 and len({c["id"] for c in cases}) == len(cases), "Expected two to four cases")
    for case in cases:
        _require(case["kind"] in CASE_KINDS and case["sample_id"].startswith("anli:")
                 and tuple(sorted(case["labels"])) == tuple(sorted(ANLI_LABELS)) and case["gold"] in case["labels"]
                 and isinstance(case["state"], str) and case["state"].strip(), "Invalid case identity")
        _require(isinstance(case["pattern_count"], int) and 1 <= case["pattern_count"] <= case["pattern_base"], "Invalid pattern count")
        unit = "successful live requests in this cell" if case["kind"] == "late_correct" else "saved test items"
        _require(case["pattern_unit"] == unit, "Case count unit does not match its selection")
        gold = case["gold"]
        if case["kind"] == "late_correct":
            t = case["trace"]
            _require(t["prediction"] == gold and _finite(t["arrival_to_completion_ms"]) and t["arrival_to_completion_ms"] > 1000
                     and case["rate"] in RATES and case["condition"] in CONDITIONS, "Late-correct case does not match its rule")
            continue
        _require(case["pattern_base"] == 3200 and isinstance(case["context"]["count"], int)
                 and case["pattern_count"] <= case["context"]["count"] <= case["pattern_base"]
                 and isinstance(case["context"]["label"], str) and case["context"]["label"].strip(),
                 "Case context count is inconsistent")
        s = case["saved"]
        front, big, fallback, deferred = (s["startlux4b"]["prediction"], s["startlux9b"]["prediction"],
                                          s["qwen9b"]["prediction"], s["deferred"])
        _require({front, big, fallback} <= set(case["labels"]) and isinstance(deferred, bool), "Invalid saved predictions")
        _require(s["cascade_final"] == (fallback if deferred else front), "Cascade label differs from the frozen route")
        rule = {"upgrade_fixes": front != gold and big == gold and not deferred,
                "fallback_fixes": front != gold and deferred and fallback == gold and big != gold,
                "fallback_harms": front == gold and deferred and fallback != gold and big == gold}[case["kind"]]
        _require(rule, "Case outcome does not match its selection rule")
        for condition, v in case["live"].items():
            _require(condition in CONDITIONS and sum(v["labels"].values()) == v["successful"] >= v["timely_500"] >= 0,
                     "Invalid live label counts")


def validate(data: dict):
    _require(data.get("schema_version") == 1 and data.get("kind") == KIND and data.get("evidence_type") == EVIDENCE_TYPE,
             "Unsupported follow-up evidence schema")
    provenance = data["provenance"]
    _require({k: (v["id"], v["sha256"]) for k, v in provenance["bundles"].items()} == BUNDLES,
             "Unknown source result archive")
    _require(provenance["tracked_sources"] == EXPECTED_SOURCES, "Tracked source hashes differ from the accepted ones")
    _require(provenance["serving_archive"]["gold_sha256"] == GOLD[1], "Unknown gold identity")
    _validate_serving(data["serving9b"])
    _validate_clinc(data["clinc_service"], data["cpu_baselines"], data["size_control"])
    _validate_cases(data["cases"])


def load_bundle(bundle_dir: Path | str) -> dict:
    """Validate the portable follow-up bundle without reading any external evidence."""
    directory = Path(bundle_dir)
    manifest = json.loads(_safe(directory, "manifest.json").read_text())
    _require(manifest.get("schema_version") == 1 and manifest.get("kind") == KIND
             and set(manifest.get("files", {})) == {"followups.json"}, "Invalid portable follow-up manifest")
    content = _safe(directory, "followups.json").read_bytes()
    expected = manifest["files"]["followups.json"]
    _require(len(content) == expected["bytes"] and _sha(content) == expected["sha256"],
             "Portable follow-up bundle hash/size mismatch")
    data = json.loads(content)
    try:
        validate(data)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError) as exc:
        raise StudyError("Malformed portable follow-up data") from exc
    return data
