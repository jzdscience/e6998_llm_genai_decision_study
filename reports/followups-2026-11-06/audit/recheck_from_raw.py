#!/usr/bin/env python3
"""Second, minimal recheck of both 11/6 returns, written separately from the main audits.

Reads only extracted raw evidence (per-request traces, prediction shards, sealed gold) with
the standard library. Never imports project code, fits, tunes or calls a model. Writes one
new JSON and refuses to overwrite an existing file.

  python3 -I recheck_from_raw.py \
    --service-root  EVIDENCE_ROOT/results/returned-service-complete-2026-11-06 \
    --clinc-gold    EVIDENCE_ROOT/results/returned-kev-2026-11-01/data/gold.json \
    --size-root     EVIDENCE_ROOT/results/returned-size-control-complete-2026-11-06 \
    --out           /tmp/followups-raw-recheck.json
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
import statistics

SLO_MS = 500.0
WINDOW_S = 180.0
CLINC_GOLD_SHA256 = "bc5ec4891caed470970115a41e2f6debe2a2967b3f5ff3496622a3d29311fc35"
ANLI_GOLD_SHA256 = "ed3525b721ed564a4cdf733e94ac381cfebb4ba06eb4a3a6844a1a0b37824f14"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def service(root, gold_path):
    if sha256(gold_path) != CLINC_GOLD_SHA256:
        raise SystemExit("CLINC gold SHA differs from the evaluated gold")
    gold = json.load(open(gold_path))
    evaluation = json.load(open(os.path.join(root, "evaluation.json")))
    saved = {c["cell_id"]: c for c in evaluation["cells"]}
    cells, mismatches = {}, []
    for cell_dir in sorted(glob.glob(os.path.join(root, "cells", "*"))):
        cell_id = os.path.basename(cell_dir)
        attempts = sorted(glob.glob(os.path.join(cell_dir, "attempt-*")))
        if len(attempts) != 1:
            raise SystemExit(f"{cell_id}: expected one attempt, found {len(attempts)}")
        c = collections.Counter()
        with open(os.path.join(attempts[0], "traces.jsonl")) as fh:
            for line in fh:
                r = json.loads(line)
                c["offered"] += 1
                if r["rejected"]:
                    c["rejected"] += 1
                    continue
                c["admitted"] += 1
                if r["status"] != "ok":
                    c["failed_admitted"] += 1
                    continue
                c["successful"] += 1
                timely = r["arrival_to_completion_ms"] <= SLO_MS
                correct = r["result"]["prediction"] == gold[r["sample_id"]]
                c["met_slo"] += timely
                c["correct"] += correct
                c["correct_met_slo"] += timely and correct
        cells[cell_id] = dict(c)
        for key in ("offered", "admitted", "correct", "correct_met_slo"):
            if c[key] != saved[cell_id][key]:
                mismatches.append([cell_id, key, c[key], saved[cell_id][key]])
    groups = collections.defaultdict(list)
    for cell_id, c in cells.items():
        _, condition, rate = cell_id.split("-")
        groups[(condition, int(rate.replace("rate", "")))].append(c)
    summary = []
    for (condition, rate), cs in sorted(groups.items()):
        summary.append({
            "condition": condition,
            "rate": rate,
            "cells": len(cs),
            "correct_goodput_per_second_median": statistics.median(x.get("correct_met_slo", 0) / WINDOW_S for x in cs),
            "successful_per_second_median": statistics.median(x.get("successful", 0) / WINDOW_S for x in cs),
            "offered": sum(x.get("offered", 0) for x in cs),
            "rejected": sum(x.get("rejected", 0) for x in cs),
            "failed_admitted": sum(x.get("failed_admitted", 0) for x in cs),
        })
    totals = collections.Counter()
    for c in cells.values():
        totals.update(c)
    return {"cells": len(cells), "mismatches_vs_saved_evaluation": mismatches,
            "totals": dict(totals), "condition_rate": summary}


def size(root, seed=6998, replicates=2000):
    gold_path = os.path.join(root, "sealed", "holdout_gold.json")
    if sha256(gold_path) != ANLI_GOLD_SHA256:
        raise SystemExit("ANLI gold SHA differs from the evaluated gold")
    gold = json.load(open(gold_path))
    examples = {}
    with open(os.path.join(root, "data", "examples.jsonl")) as fh:
        for line in fh:
            e = json.loads(line)
            examples[e["sample_id"]] = e
    predictions = {"startlux9b": {}}
    for shard in sorted(glob.glob(os.path.join(root, "predictions-job", "predictions", "startlux", "shard-*.json"))):
        for row in json.load(open(shard))["rows"]:
            if row["sample_id"] in gold:
                if row["sample_id"] in predictions["startlux9b"]:
                    raise SystemExit("duplicate holdout prediction")
                predictions["startlux9b"][row["sample_id"]] = row["prediction"]
    for name in ("startlux4b", "qwen9b", "qwen4b"):
        predictions[name] = {}
        with open(os.path.join(root, "references", f"{name}.jsonl")) as fh:
            for line in fh:
                row = json.loads(line)
                if row["sample_id"] in gold:
                    predictions[name][row["sample_id"]] = row["prediction"]
    ids = sorted(gold)
    for name, rows in predictions.items():
        if set(rows) != set(ids):
            raise SystemExit(f"{name}: prediction ids do not cover the holdout")
    correct = {name: [rows[i] == gold[i] for i in ids] for name, rows in predictions.items()}
    direct = {name: {"correct": sum(v), "n": len(v), "accuracy": sum(v) / len(v)} for name, v in correct.items()}
    comparisons = {}
    for ref in ("startlux4b", "qwen9b"):
        a, b = correct["startlux9b"], correct[ref]
        clusters = {}
        for sample_id, x, y in zip(ids, a, b):
            s = clusters.setdefault(examples[sample_id]["group_id"], [0, 0])
            s[0] += x - y
            s[1] += 1
        cl = list(clusters.values())
        rng = random.Random(seed)
        draws = []
        for _ in range(replicates):
            sample = [cl[rng.randrange(len(cl))] for _ in cl]
            draws.append(sum(d for d, _ in sample) / sum(n for _, n in sample))
        draws.sort()
        comparisons[f"startlux9b_minus_{ref}"] = {
            "both_correct": sum(x and y for x, y in zip(a, b)),
            "both_wrong": sum((not x) and (not y) for x, y in zip(a, b)),
            "candidate_only_correct": sum(x and not y for x, y in zip(a, b)),
            "reference_only_correct": sum(y and not x for x, y in zip(a, b)),
            "difference": sum(d for d, _ in cl) / sum(n for _, n in cl),
            "clusters": len(cl),
            "stdlib_percentile_interval_95": [draws[int(0.025 * replicates) - 1], draws[int(0.975 * replicates) - 1]],
            "interval_note": "Python random.Random, not the project's numpy generator; expect Monte Carlo differences from the saved interval",
        }
    return {"n": len(ids), "direct": direct, "comparisons": comparisons}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-root", required=True)
    ap.add_argument("--clinc-gold", required=True)
    ap.add_argument("--size-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    result = {
        "status": "PENDING",
        "service": service(a.service_root, a.clinc_gold),
        "size_control": size(a.size_root),
    }
    svc = result["service"]
    sc = result["size_control"]
    ok = (svc["cells"] == 60 and not svc["mismatches_vs_saved_evaluation"]
          and svc["totals"].get("offered") == 469804 and svc["totals"].get("failed_admitted", 0) == 0
          and sc["direct"]["startlux9b"]["correct"] == 2454)
    result["status"] = "PASS" if ok else "FAIL"
    with open(a.out, "x") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(result["status"], a.out)


if __name__ == "__main__":
    main()
