#!/usr/bin/env python3
"""Group the accepted 9B serving-cost receipts into the report tables (stdlib only).

Reads only the tracked receipts in evidence/, which are byte-identical copies of the audit outputs
accepted on 2026-11-10 (records/upgrade-service-acceptance-2026-11-10.json). It never opens a raw
bundle, calls a model or reruns the production evaluation.

  python3 -I reports/upgrade-9b-serving-2026-11-09/build_tables.py [--out DIR]

Writes summary.csv (deployment x rate x SLO), quality.csv and resources.csv to DIR, by default
the report folder. Before writing, it re-derives the medians, pass sets and totals from the 36 cells
and stops if they disagree with the accepted summaries or receipt.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
CONDITIONS = ("startlux4b", "startlux9b", "cascade")
RATES = (10, 20, 40, 80)
REPEATS = (0, 1, 2)
SLOS = {500: "primary", 1000: "secondary"}
DEPLOYMENT = {
    "startlux4b": ("StartLux-4B direct", "0.40"),
    "startlux9b": ("StartLux-9B direct", "0.50"),
    "cascade": ("StartLux-4B to Qwen3.5-9B cascade", "0.40 + 0.50"),
}
TIMELY_RULE = 0.99
WINDOW_SECONDS = 120


def require(condition, message):
    if not condition:
        raise SystemExit(f"build_tables: {message}")


def load(name):
    return json.loads((EVIDENCE / name).read_text())


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    recomputed = load("metrics/recomputed.json")
    receipt = load("metrics/audit-receipt.json")
    integrity = load("integrity-audit.json")
    require(receipt["status"] == "PASS" and receipt["discrepancies"] == [], "metrics receipt is not PASS")
    require(integrity["status"] == "PASS", "integrity receipt is not PASS")
    cells = {(c["condition"], c["rate"], c["repeat"]): c for c in recomputed["cells"]}
    require(len(recomputed["cells"]) == 36 and set(cells) == {
        (c, r, k) for c in CONDITIONS for r in RATES for k in REPEATS}, "expected exactly 36 cells")
    summaries = {(s["condition"], s["rate"]): s["metrics"] for s in recomputed["summaries"]}

    totals = {"offered": 0, "successful": 0, "rejected": 0}
    for cell in cells.values():
        p = cell["primary"]
        require(p["offered"] == p["successful"] + p["rejected"] and p["failed_admitted"] == 0,
                f'{cell["cell_id"]}: offered != successful + rejected')
        for key in totals:
            totals[key] += p[key]
    require(totals == {"offered": receipt["offered_total"], "successful": receipt["status_counts"]["ok"],
                       "rejected": receipt["status_counts"]["client_rejected"]}, "totals differ from the receipt")

    summary, passing = [], {c: {} for c in CONDITIONS}
    for slo, key in SLOS.items():
        for condition in CONDITIONS:
            passing[condition][key] = []
            for rate in RATES:
                group = [cells[(condition, rate, k)] for k in REPEATS]
                goodput = [g[key]["correct_goodput_per_second"] for g in group]
                timely = [g[key]["slo_fraction_all_offered"] for g in group]
                for g, value in zip(group, goodput):
                    require(math.isclose(value, g[key]["correct_met_slo"] / WINDOW_SECONDS, rel_tol=0, abs_tol=1e-12),
                            f'{g["cell_id"]}: goodput is not correct-and-timely / {WINDOW_SECONDS} s')
                saved = summaries[(condition, rate)][key + ".correct_goodput_per_second"]
                require((statistics.median(goodput), min(goodput), max(goodput)) == (saved["median"], saved["min"], saved["max"]),
                        f"{condition} {rate}/s {slo} ms: median/min/max differ from the accepted summary")
                passes = all(t >= TIMELY_RULE for t in timely)
                if passes:
                    passing[condition][key].append(rate)
                name, share = DEPLOYMENT[condition]
                row = {"condition": condition, "deployment": name, "gpu_memory_share": share,
                       "rate_per_second": rate, "slo_ms": slo, "repeats": len(group)}
                row.update({f"correct_goodput_r{k}": v for k, v in zip(REPEATS, goodput)})
                row.update(correct_goodput_median=statistics.median(goodput), correct_goodput_min=min(goodput),
                           correct_goodput_max=max(goodput))
                row.update({f"timely_fraction_r{k}": v for k, v in zip(REPEATS, timely)})
                row.update(timely_fraction_min=min(timely), passes_99pct_all_repeats=passes)
                row.update(
                    offered_total=sum(g["primary"]["offered"] for g in group),
                    successful_total=sum(g["primary"]["successful"] for g in group),
                    client_rejected_total=sum(g["primary"]["rejected"] for g in group),
                    correct_and_timely_total=sum(g[key]["correct_met_slo"] for g in group),
                    successful_per_second_median=statistics.median(g["primary"]["successful_per_offered_window_second"] for g in group),
                    latency_p95_ms_median=statistics.median(g["primary"]["latency_ms_all_admitted"]["p95"] for g in group),
                )
                if condition == "cascade":
                    deferral = summaries[(condition, rate)]["deferral_rate_routed"]
                    row.update(live_deferral_median=deferral["median"], live_deferral_min=deferral["min"],
                               live_deferral_max=deferral["max"])
                else:
                    row.update(live_deferral_median="", live_deferral_min="", live_deferral_max="")
                summary.append(row)
    require(passing == receipt["passing_rates_all_three_repeats_99pct"], "pass sets differ from the accepted receipt")

    quality = []
    for condition in CONDITIONS:
        live = receipt["live_history_comparison"][condition]
        variability = receipt["label_variability"][condition]
        name, share = DEPLOYMENT[condition]
        quality.append({
            "condition": condition, "deployment": name,
            "historical_full_pool_accuracy": receipt["historical_full_pool_accuracy"][condition],
            "successful_responses": live["successful_response_count"],
            "pooled_live_accuracy": live["pooled_live_accuracy"],
            "pooled_matched_historical_accuracy": live["pooled_matched_historical_accuracy"],
            "live_accuracy_cell_min": live["live_accuracy_per_cell_range"][0],
            "live_accuracy_cell_max": live["live_accuracy_per_cell_range"][1],
            "inputs_seen": variability["inputs_seen"],
            "inputs_with_more_than_one_live_label": variability["inputs_with_more_than_one_live_label"],
        })

    peaks = {c: [] for c in CONDITIONS}
    for cell in integrity["cell_inventory"]:
        peaks[cell["cell_id"].split("-")[1]].append(cell["gpu_peak_mib"] / 1024)
    require(all(len(v) == 12 for v in peaks.values()), "expected 12 sampled cells per deployment")
    resources = []
    for condition in CONDITIONS:
        roles = ["startlux4b", "qwen9b"] if condition == "cascade" else [condition]
        for role in roles:
            formal = [x for x in integrity["launches"] if x["condition"] == condition and x["role"] == role
                      and not x["scope"].startswith("preflight")]
            pre = [x for x in integrity["launches"] if x["condition"] == condition and x["role"] == role
                   and x["scope"].startswith("preflight")]
            require(len(formal) == 12 and len(pre) == 1, f"{condition}/{role}: unexpected launch count")
            ranges = integrity["formal_startup_kv_ranges"][f"{condition}:{role}"]
            kv = [x["kv_gib"][0] for x in formal]
            tokens = [x["kv_tokens"][0] for x in formal]
            require((min(kv), max(kv), min(tokens), max(tokens)) == (
                ranges["kv_gib_min"], ranges["kv_gib_max"], ranges["kv_tokens_min"], ranges["kv_tokens_max"]),
                f"{condition}/{role}: KV ranges differ from the integrity receipt")
            resources.append({
                "condition": condition, "deployment": DEPLOYMENT[condition][0], "role": role,
                "gpu_memory_share": "0.40" if role == "startlux4b" else "0.50",
                "formal_engine_starts": len(formal),
                "formal_kv_gib_min": min(kv), "formal_kv_gib_max": max(kv),
                "formal_kv_tokens_min": min(tokens), "formal_kv_tokens_max": max(tokens),
                "formal_startup_seconds_min": min(x["startup_seconds"] for x in formal),
                "formal_startup_seconds_max": max(x["startup_seconds"] for x in formal),
                "preflight_kv_gib": pre[0]["kv_gib"][0], "preflight_kv_tokens": pre[0]["kv_tokens"][0],
                "deployment_sampled_gpu_peak_gib_min": min(peaks[condition]),
                "deployment_sampled_gpu_peak_gib_max": max(peaks[condition]),
            })

    write(out / "summary.csv", summary)
    write(out / "quality.csv", quality)
    write(out / "resources.csv", resources)
    print(f"wrote summary.csv ({len(summary)} rows), quality.csv, resources.csv to {out}")


if __name__ == "__main__":
    main()
