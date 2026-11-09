"""Owned subprocess for one bounded measurement: a formal cell or the preflight concurrent check."""
import argparse
import os
from pathlib import Path

from ..common import StudyError, file_hash, fingerprint, read_json, write_jsonl
from ..service_study.measurement import run_cell
from ..service_study.resources import ResourceMonitor
from .client import call
from .identity import fixed_write, safe


def execute(root, folder):
    plan = read_json(folder / "plan.json")
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
