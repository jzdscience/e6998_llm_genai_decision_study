"""Owned subprocess for one bounded stage under the parent watchdog.

Kinds: a formal cell or the preflight concurrent check (open-loop schedule through
``run_cell``), or a serial stage (the preflight serial check or a cell's warmup).
"""
import argparse
import os
from pathlib import Path
import time

from ..common import StudyError, file_hash, fingerprint, read_json, write_jsonl
from ..service_study.measurement import run_cell
from ..service_study.resources import ResourceMonitor
from .client import call
from .identity import fixed_write, safe


SERIAL_FILES = ("plan.json", "rows.json")


def execute_serial(root, folder, plan):
    for kind in ("workload", "policy"):
        if file_hash(safe(root, plan[kind + "_file"])) != plan[kind + "_sha256"]:
            raise StudyError(f"Changed serial-stage {kind}")
    examples = read_json(safe(root, plan["workload_file"]))
    if [e["sample_id"] for e in examples] != plan["sample_ids"]:
        raise StudyError("Serial-stage inputs differ from the plan")
    policy = read_json(safe(root, plan["policy_file"]))["policy"] if plan["condition"] == "cascade" else None
    began, rows = time.perf_counter(), []
    for example in examples:
        rows.append(call(plan["condition"], plan["endpoints"], example, plan["request_timeout_seconds"], policy,
                         mode=plan["mode"]))
    fixed_write(folder / "rows.json", {"rows": rows, "elapsed_seconds": time.perf_counter() - began})
    record = {"plan_hash": fingerprint(plan), "files": {name: file_hash(folder / name) for name in SERIAL_FILES}}
    record["self_hash"] = fingerprint(record)
    fixed_write(folder / "complete.json", record)


def execute(root, folder):
    plan = read_json(folder / "plan.json")
    if plan.get("kind") == "serial":
        return execute_serial(root, folder, plan)
    p = plan["protocol"]
    for kind in ("schedule", "workload", "policy"):
        if file_hash(safe(root, plan[kind + "_file"])) != plan[kind + "_sha256"]:
            raise StudyError(f"Changed cell {kind}")
    schedule = read_json(safe(root, plan["schedule_file"]))
    examples = {e["sample_id"]: e for e in read_json(safe(root, plan["workload_file"]))}
    policy = read_json(safe(root, plan["policy_file"]))["policy"] if plan["condition"] == "cascade" else None
    if any(row["sample_id"] not in examples for row in schedule):
        raise StudyError("Scheduled sample missing from workload")

    def invoke(row, timeout):
        return call(plan["condition"], plan["endpoints"], examples[row["sample_id"]], timeout, policy,
                    mode=plan.get("mode", "formal"))

    with ResourceMonitor([os.getpid(), *plan["service_pids"]], folder / "resources.jsonl"):
        report, rows, events = run_cell(schedule, invoke, duration=p["duration_seconds"],
                                        workers=p["workers"], max_inflight=p["max_inflight"], slo_ms=p["slo_ms"],
                                        drain_seconds=p["drain_seconds"], request_timeout=p["request_timeout_seconds"],
                                        trace_path=folder / "traces.jsonl")
    write_jsonl(folder / "occupancy.jsonl", events)
    record = {"plan_hash": fingerprint(plan), "report": report,
              "files": {name: file_hash(folder / name) for name in
                        ("plan.json", "traces.jsonl", "occupancy.jsonl", "resources.jsonl")}}
    record["self_hash"] = fingerprint(record)
    fixed_write(folder / "complete.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    execute(args.root, args.attempt)


if __name__ == "__main__":
    main()
