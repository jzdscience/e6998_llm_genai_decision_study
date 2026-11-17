"""Post-freeze quality and service evaluation from actual returned labels, plus audit and report.

Gold enters only here, after ``freeze.json`` binds every raw trace, ledger, fallback
journal, service record and verified stop, and after an access receipt bound to that
freeze is written. The eligible-dev labels were already viewed in Study 4: this is a
retrospective system evaluation, not fresh confirmation. Three repeats are described
by median/min/max and paired per-cell differences only; no 95% interval is formed
from three runs, and no offline accuracy is multiplied by throughput.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

import numpy as np

from ..common import StudyError, file_hash, fingerprint, read_json, read_jsonl, utc_now
from ..service_study.identity import fixed_write
from . import inputs, protocol, study
from .protocol import BACK, LABELS, PAIR_CONDITIONS, ROLES, require

LIMITS_NOTE = [
    "Retrospective system evaluation on 1,737 already-viewed eligible-dev inputs; the cascade was chosen after a secondary result. No fresh confirmatory quality or equivalence claim.",
    "Repeated offers are workload instances, not additional independent quality observations.",
    "Three repeats are summarized by median/min/max and paired per-cell differences; no 95% interval is formed from three runs.",
    "Direct conditions use one role at 0.90 of one L40S; pair conditions co-reside 0.40 + 0.50. This is a deployment tradeoff, not a pure model-size, router or co-residency effect.",
    "The budget caps 9B calls causally at reserved <= floor(offered/10); it is not an exact 10% of calls and not a GPU-billing or dollar saving.",
    "Random uses the same hard cap and may underfill; it is not matched on realized 9B calls.",
    "Correct goodput counts actual returned labels within the SLO per 120-second offer window; offline accuracy is never multiplied by throughput.",
]


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _quantiles(values):
    return {f"p{q}": (float(np.quantile(values, q / 100)) if values else None) for q in (50, 95, 99)}


def _counts(values):
    return {str(k): v for k, v in sorted(Counter(str(v) for v in values).items())}


def cell_metrics(rows, gold, *, condition, p):
    duration, limits = p["duration_seconds"], (p["slo_ms"], p["secondary_slo_ms"])
    offered = len(rows)
    admitted = [r for r in rows if not r["rejected"]]
    success = [r for r in admitted if r["status"] == "ok"]
    correct = [r for r in success if r["result"]["prediction"] == gold[r["sample_id"]]]
    within = lambda group, limit: [r for r in group if r["arrival_to_completion_ms"] <= limit]
    stages = [s for r in admitted for s in r["result"]["stages"]]
    attempts = {role: sum(1 for s in stages if s["role"] == role and s["attempts"] == 1) for role in ROLES}
    metrics = {
        "offered": offered, "admitted": len(admitted), "client_rejected": offered - len(admitted),
        "successful": len(success), "errors": sum(r["status"] == "error" for r in admitted),
        "deadlines": sum(r["status"] == "deadline" for r in admitted), "status_counts": _counts(r["status"] for r in rows),
        "failure_types": _counts((r["result"].get("error") or {}).get("type") for r in admitted if r["status"] != "ok"),
        "correct": len(correct), "accuracy_successful": _rate(len(correct), len(success)),
        "accuracy_denominator": "successful returned labels", "correct_per_offer": _rate(len(correct), offered),
        "successful_per_offer": _rate(len(success), offered),
        "timely_successful": {str(l): len(within(success, l)) for l in limits},
        "late_successful": {str(l): len(success) - len(within(success, l)) for l in limits},
        "timely_correct": {str(l): len(within(correct, l)) for l in limits},
        "timely_correct_per_second": {str(l): len(within(correct, l)) / duration for l in limits},
        "timely_correct_per_offer": {str(l): _rate(len(within(correct, l)), offered) for l in limits},
        "latency_ms_admitted": _quantiles([r["arrival_to_completion_ms"] for r in admitted]),
        "latency_ms_successful": _quantiles([r["arrival_to_completion_ms"] for r in success]),
        "client_queue_ms": _quantiles([r["client_queue_ms"] for r in admitted]),
        "model_attempts": attempts, "startlux9b_attempts_per_offer": _rate(attempts[BACK], offered),
        "final_source": _counts(r["result"]["final_source"] for r in success),
        "correct_by_source": _counts(r["result"]["final_source"] for r in correct)}
    if condition in PAIR_CONDITIONS:
        routes = [r["result"]["route"] for r in admitted if r["result"]["route"]]
        both = [r for r in admitted if len(r["result"]["stages"]) == 2
                and all(s["status"] == "ok" for s in r["result"]["stages"])]
        front_ok = lambda r: r["result"]["stages"][0]["prediction"] == gold[r["sample_id"]]
        back_ok = lambda r: r["result"]["stages"][1]["prediction"] == gold[r["sample_id"]]
        metrics["routing"] = {
            "routed_offers": len(routes), "candidate_reasons": _counts(x["candidate_reason"] for x in routes),
            "candidates": sum(bool(x["candidate"]) for x in routes),
            "permit_requests": sum(bool(x["permit_requested"]) for x in routes),
            "budget_granted": sum(x["budget_granted"] is True for x in routes),
            "budget_denied": sum(x["budget_denied"] is True for x in routes),
            "fallback_invocations": sum(bool(x["fallback_step"]) for x in routes),
            "fallback_attempted": sum(bool(x["fallback_attempted"]) for x in routes),
            "unused_reservations": sum(x["unused_reservation_reason"] is not None for x in routes),
            "counting": "reservations = budget_granted; invocations = fallback steps; actual attempts = started 9B HTTP calls",
            "router_ms": _quantiles([x["router_ms"] for x in routes if x["router_ms"] is not None])}
        metrics["front_fallback_pairs"] = {
            "applicable": len(both), "denominator": "admitted offers whose 4B and 9B stages both returned valid labels",
            "rescue": sum(not front_ok(r) and back_ok(r) for r in both), "harm": sum(front_ok(r) and not back_ok(r) for r in both),
            "both_correct": sum(front_ok(r) and back_ok(r) for r in both),
            "both_wrong": sum(not front_ok(r) and not back_ok(r) for r in both),
            "note": "Only live pairs from this run; no historical counterfactual fills uncalled 9B answers"}
    return metrics


def shadow_cap(ledger_events, journal_events, p):
    """Gold-free, post-freeze replay of the causal cap over a threshold cell's live candidate times."""
    offers = [e["at_seconds"] for e in ledger_events if e["type"] == "offer"]
    steps = sorted(e["at_seconds"] for e in journal_events)
    numerator, denominator = p["budget"]["numerator"], p["budget"]["denominator"]
    index = reserved = denied = 0
    for at in steps:
        while index < len(offers) and offers[index] <= at:
            index += 1
        if reserved < numerator * index // denominator:
            reserved += 1
        else:
            denied += 1
    return {"candidates_reaching_fallback": len(steps), "would_grant": reserved, "would_deny": denied,
            "offers": len(offers), "scope": "Descriptive counterfactual on this threshold cell; not a measured budget condition"}


def resource_summary(path):
    used, utilization, cards = [], [], set()
    for sample in read_jsonl(path):
        for gpu in (sample.get("gpu") or {}).get("gpus") or []:
            if gpu.get("used_mib") is not None:
                used.append(gpu["used_mib"])
            if gpu.get("utilization_percent") is not None:
                utilization.append(gpu["utilization_percent"])
            cards.add(f"{gpu.get('name')}|{gpu.get('uuid')}")
    return {"gpu_memory_used_mib": {"max": max(used) if used else None, "mean": float(np.mean(used)) if used else None},
            "gpu_utilization_mean_percent": float(np.mean(utilization)) if utilization else None,
            "gpu_samples": len(used), "cards": sorted(cards),
            "scope": "nvidia-smi device samples during the open-loop window; device-wide, not per process"}


def label_variability(seen):
    return {condition: {"inputs_seen": len(by_id), "inputs_with_more_than_one_live_label": sum(len(v) > 1 for v in by_id.values())}
            for condition, by_id in sorted(seen.items())}


def summary_values(metrics, p):
    primary, secondary = str(p["slo_ms"]), str(p["secondary_slo_ms"])
    values = {f"timely_correct_per_second_{primary}ms": metrics["timely_correct_per_second"][primary],
              f"timely_correct_per_second_{secondary}ms": metrics["timely_correct_per_second"][secondary],
              f"timely_correct_{primary}ms": metrics["timely_correct"][primary],
              "accuracy_successful": metrics["accuracy_successful"], "correct_per_offer": metrics["correct_per_offer"],
              "successful_per_offer": metrics["successful_per_offer"], "client_rejected": metrics["client_rejected"],
              "errors": metrics["errors"], "deadlines": metrics["deadlines"],
              f"late_successful_{primary}ms": metrics["late_successful"][primary],
              "latency_p95_ms_admitted": metrics["latency_ms_admitted"]["p95"],
              "startlux9b_attempts_per_offer": metrics["startlux9b_attempts_per_offer"]}
    routing = metrics.get("routing")
    if routing:
        values.update(candidates=routing["candidates"], budget_granted=routing["budget_granted"],
                      budget_denied=routing["budget_denied"], unused_reservations=routing["unused_reservations"])
    return values


def _stats(values):
    present = [v for v in values if v is not None]
    if not present:
        return {"median": None, "min": None, "max": None, "values": values, "missing": len(values)}
    return {"median": float(np.median(present)), "min": min(present), "max": max(present), "values": values,
            "missing": len(values) - len(present)}


def summaries(cells, p):
    out = []
    for condition in p["conditions"]:
        for rate in p["rates_per_second"]:
            chosen = sorted((c for c in cells if c["condition"] == condition and c["rate"] == rate), key=lambda c: c["repeat"])
            values = [summary_values(c["metrics"], p) for c in chosen]
            keys = sorted(set().union(*values)) if values else []
            out.append({"condition": condition, "rate": rate, "repeats": [c["repeat"] for c in chosen],
                        "metrics": {k: _stats([v.get(k) for v in values]) for k in keys}})
    return out


def paired_differences(cells, p):
    """Later minus earlier condition (protocol order) within each repeat/rate schedule."""
    by = {(c["condition"], c["repeat"], c["rate"]): summary_values(c["metrics"], p) for c in cells}
    out = []
    order = list(p["conditions"])
    for i, earlier in enumerate(order):
        for later in order[i + 1:]:
            for rate in p["rates_per_second"]:
                diffs = defaultdict(list)
                for repeat in range(p["repeats"]):
                    a, b = by[(later, repeat, rate)], by[(earlier, repeat, rate)]
                    for key in sorted(set(a) & set(b)):
                        diffs[key].append(None if a[key] is None or b[key] is None else a[key] - b[key])
                out.append({"difference": f"{later} - {earlier}", "rate": rate,
                            "pairing": "same repeat/rate schedule and seeded inputs", "metrics": {k: _stats(v) for k, v in sorted(diffs.items())}})
    return out


def compute(ctx, gold):
    """Deterministic metrics from the accepted raw traces and the eligible-dev labels."""
    p = ctx.p
    require(set(gold) == set(ctx.examples) and all(v in LABELS for v in gold.values()), "Gold does not cover exactly the workload")
    cells, seen = [], defaultdict(lambda: defaultdict(set))
    for cell in ctx.settings["order"]:  # one cell's raw trace in memory at a time
        receipt = ctx.json(f"cells/{cell['cell_id']}/accepted.json")
        folder = study.path(ctx.output, receipt["attempt"])
        rows = sorted(read_jsonl(study.path(folder, "traces.jsonl")), key=lambda r: r["offer_id"])
        for row in rows:
            if row["status"] == "ok":
                seen[cell["condition"]][row["sample_id"]].add(row["result"]["prediction"])
        value = {**cell, "attempt": receipt["attempt"], "allocation": receipt["allocation"],
                 "prior_attempts": receipt["prior_attempts"], "metrics": cell_metrics(rows, gold, condition=cell["condition"], p=p),
                 "ledger": receipt["ledger"], "service": receipt["service"],
                 "resources": resource_summary(study.path(folder, "resources.jsonl"))}
        if cell["condition"] == "threshold":
            value["counterfactual_cap"] = shadow_cap(read_jsonl(study.path(folder, "ledger.jsonl")),
                                                     read_jsonl(study.path(folder, "fallback.jsonl")), p)
        cells.append(value)
    allocations = defaultdict(set)
    for cell in cells:
        allocations[f"r{cell['repeat']}-rate{cell['rate']}"].add(fingerprint(cell["allocation"]))
    return {"status": "COMPLETE", "protocol": p["protocol"], "revision": p["revision"],
            "primary_metric": p["primary_metric"], "cells": cells, "summaries": summaries(cells, p),
            "paired_differences": paired_differences(cells, p), "label_variability": label_variability(seen),
            "pairing_same_allocation": {k: len(v) == 1 for k, v in sorted(allocations.items())},
            "offers_total": sum(c["metrics"]["offered"] for c in cells), "interpretation": LIMITS_NOTE}


def _source(ctx):
    return study.normal(ctx.settings["source_study"])


def evaluate(ctx):
    output = ctx.output
    frozen = study.verify_freeze(ctx)
    freeze_sha = file_hash(study.path(output, "freeze.json"))
    complete_path = study.path(output, "evaluation/complete.json")
    if complete_path.exists():
        return verify_evaluation(ctx)
    access = study.path(output, "evaluation/access.json")
    if access.exists():
        require(read_json(access).get("freeze_sha256") == freeze_sha, "Gold access receipt is bound to another freeze")
    else:
        fixed_write(access, {"freeze_sha256": freeze_sha, "cells": len(frozen["cells"]), "at": utc_now(),
                             "gold_source": ctx.p["gold"]["source_file"], "gold_sha256": ctx.p["gold"]["sha256"],
                             "purpose": "Retrospective scoring of actual returned labels in all 60 frozen cells; no selection or tuning",
                             "already_viewed": "These eligible-dev labels were evaluated in Study 4; not fresh confirmation"})
    gold = inputs.load_gold(_source(ctx), ctx.p, [r["sample_id"] for r in ctx.workload])
    preserved = study.path(output, "evaluation/source-gold.json")
    if not preserved.exists():  # exact pinned bytes, so an offline audit re-derives the subset itself
        data = study.path(_source(ctx), ctx.p["gold"]["source_file"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == ctx.p["gold"]["sha256"], "Study 4 gold changed while being preserved")
        with preserved.open("xb") as stream:
            stream.write(data)
    require(file_hash(preserved) == ctx.p["gold"]["sha256"], "Preserved Study 4 gold differs from its pinned SHA")
    fixed_write(study.path(output, "evaluation/gold.json"), gold)
    results = compute(ctx, gold)
    fixed_write(study.path(output, "evaluation/results.json"), results)
    fixed_write(complete_path, {"freeze_sha256": freeze_sha, "access_sha256": file_hash(access),
                                "gold_sha256": file_hash(study.path(output, "evaluation/gold.json")),
                                "results_sha256": file_hash(study.path(output, "evaluation/results.json")),
                                "source_gold_sha256": ctx.p["gold"]["sha256"],
                                "preserved_source_gold_sha256": file_hash(preserved)})
    return results


def verify_evaluation(ctx):
    output = ctx.output
    complete = read_json(study.path(output, "evaluation/complete.json"))
    names = {"freeze_sha256": "freeze.json", "access_sha256": "evaluation/access.json", "gold_sha256": "evaluation/gold.json",
             "results_sha256": "evaluation/results.json", "preserved_source_gold_sha256": "evaluation/source-gold.json"}
    require(all(complete.get(k) == file_hash(study.path(output, v)) for k, v in names.items())
            and complete.get("source_gold_sha256") == ctx.p["gold"]["sha256"] == complete.get("preserved_source_gold_sha256")
            and read_json(study.path(output, "evaluation/access.json")).get("freeze_sha256") == complete["freeze_sha256"],
            "Evaluation receipt does not bind its freeze, access, gold and results")
    return read_json(study.path(output, "evaluation/results.json"))


def _primary_check(ctx, gold, results):
    """A second, minimal path for the primary metric directly from raw rows."""
    limit = ctx.p["slo_ms"]
    by_id = {c["cell_id"]: c for c in results["cells"]}
    for cell in ctx.settings["order"]:
        receipt = ctx.json(f"cells/{cell['cell_id']}/accepted.json")
        count = 0
        for row in read_jsonl(study.path(study.path(ctx.output, receipt["attempt"]), "traces.jsonl")):
            if (row["rejected"] is False and row["status"] == "ok" and row["arrival_to_completion_ms"] <= limit
                    and row["result"]["prediction"] == gold[row["sample_id"]]):
                count += 1
        require(by_id[cell["cell_id"]]["metrics"]["timely_correct"][str(limit)] == count, "Primary metric differs on the minimal path")


def audit(ctx, *, offline=False, gold_source=None):
    """Re-verify the freeze and every cell from raw evidence, then recompute all metrics."""
    frozen = study.verify_freeze(ctx, offline=offline)
    results = verify_evaluation(ctx)
    gold = read_json(study.path(ctx.output, "evaluation/gold.json"))
    workload_ids = [r["sample_id"] for r in ctx.workload]
    require(set(gold) == set(workload_ids) and all(v in LABELS for v in gold.values()), "Returned gold subset differs")
    preserved = study.path(ctx.output, "evaluation/source-gold.json")
    require(file_hash(preserved) == ctx.p["gold"]["sha256"]
            and inputs.gold_subset(read_json(preserved), ctx.p, workload_ids) == gold,
            "Evaluated gold subset is not the exact eligible-dev subset of the pinned Study 4 gold")
    if not offline:
        require(gold == inputs.load_gold(_source(ctx), ctx.p, workload_ids), "Returned gold differs from the pinned source")
    if gold_source is not None:
        source = study.normal(gold_source)
        require(file_hash(source) == ctx.p["gold"]["sha256"], "Supplied gold source differs from the pinned SHA")
        full = read_json(source)
        require({sid: full[sid] for sid in workload_ids} == gold, "Returned gold differs from the supplied source")
    recomputed = compute(ctx, gold)
    require(fingerprint(recomputed) == fingerprint(results), "Recorded metrics differ from recomputation over raw traces")
    _primary_check(ctx, gold, recomputed)
    current = protocol.source_identity()["files"] == ctx.settings["source"]["files"]
    require(current, "Audit must run from the released source checkout recorded in study.json")
    value = {"status": "PASS", "freeze_sha256": file_hash(study.path(ctx.output, "freeze.json")),
             "results_sha256": file_hash(study.path(ctx.output, "evaluation/results.json")),
             "cells": len(frozen["cells"]), "offers": recomputed["offers_total"], "frozen_files": len(frozen["files"]),
             "gold_checked_against": "preserved pinned Study 4 gold bytes" + ("" if offline else " and the node source")
                                     + (" and a supplied source" if gold_source else ""),
             "current_checkout_matches_recorded_source": current, "offline": offline,
             "checks": ["supervision/stop receipts", "native ownership receipt", "plans and prerequisite hashes", "token reports",
                        "deployment flags and runtime identity", "warmup", "trace/schedule coverage", "raw responses",
                        "live gain recomputation", "ledger replay", "fallback journal", "summaries", "metrics", "primary metric minimal path"],
             "limit": "Returned evidence only; live node processes and model bytes are not re-observed by this audit"}
    if not offline:
        fixed_write(study.path(ctx.output, "audit.json"), {k: v for k, v in value.items() if k != "offline"})
    return value


def _fmt(stat, scale=1.0, digits=2):
    if stat is None or stat["median"] is None:
        return "—"
    return f"{scale * stat['median']:.{digits}f} [{scale * stat['min']:.{digits}f}, {scale * stat['max']:.{digits}f}]"


def report(ctx):
    results = verify_evaluation(ctx)
    p = ctx.p
    primary, secondary = p["slo_ms"], p["secondary_slo_ms"]
    lines = ["# StartLux-4B → StartLux-9B online budget and serving study", "",
             f"Protocol `{p['protocol']}` revision {p['revision']}. Each value is median [min, max] over {p['repeats']} repeats.", "",
             f"| Condition | Rate/s | Timely-correct/s @{primary} ms | @{secondary} ms | Accuracy among successful (%) | "
             "Correct/offer (%) | Rejected | Errors | Deadlines | 9B attempts/offer | p95 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results["summaries"]:
        m = row["metrics"]
        lines.append(f"| {row['condition']} | {row['rate']} | {_fmt(m[f'timely_correct_per_second_{primary}ms'])} | "
                     f"{_fmt(m[f'timely_correct_per_second_{secondary}ms'])} | {_fmt(m['accuracy_successful'], 100)} | "
                     f"{_fmt(m['correct_per_offer'], 100)} | {_fmt(m['client_rejected'], digits=0)} | {_fmt(m['errors'], digits=0)} | "
                     f"{_fmt(m['deadlines'], digits=0)} | {_fmt(m['startlux9b_attempts_per_offer'], digits=3)} | "
                     f"{_fmt(m['latency_p95_ms_admitted'], digits=0)} |")
    lines += ["", "Budget counts (pair conditions): see `evaluation/results.json` cells[].metrics.routing and ledger.", ""]
    lines += [f"- {note}" for note in results["interpretation"]]
    payload = "\n".join(lines) + "\n"
    target = study.path(ctx.output, "REPORT.md")
    if target.exists():
        require(target.read_text(encoding="utf-8") == payload, "Existing report differs from the fixed results")
        return target
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return target
