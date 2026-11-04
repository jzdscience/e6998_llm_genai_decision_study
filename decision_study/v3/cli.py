"""Explicit, fail-closed v3 operations. No implicit session extensions."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from ..common import StudyError, exclusive_run, utc_now, write_json
from .identity import PROJECT, ROLES, verify_release


def slurm_hours_left():
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job):
        raise StudyError("SLURM_JOB_ID is missing or invalid. Supply a verified remaining allocation with --session-hours-left NUMBER.")
    result = subprocess.run(["squeue", "--jobs", job, "--noheader", "--format=%i|%T|%L"],
                            text=True, capture_output=True, timeout=20, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise StudyError("Cannot identify exactly one current Slurm allocation")
    got_job, state, allowance = (part.strip() for part in lines[0].split("|"))
    if got_job != job or state != "RUNNING":
        raise StudyError("Current Slurm job is not RUNNING")
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", allowance)
    if not match:
        raise StudyError(f"Unrecognized Slurm time-left: {allowance!r}; no deadline guessed")
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    remaining = days * 86400 + hours * 3600 + minutes * 60 + seconds - 300
    if remaining <= 0:
        raise StudyError("Less than five minutes remain in this allocation")
    return remaining / 3600


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("status", "init-runtime", "start", "pipeline", "collect", "fit", "evaluate", "export"))
    p.add_argument("--v1-dir", type=Path, default=PROJECT / "results/solo-v1")
    p.add_argument("--output-dir", type=Path, default=PROJECT / "results/solo-v3")
    p.add_argument("--session-hours-left", type=float)
    p.add_argument("--new-allocation", action="store_true")
    p.add_argument("--role", choices=ROLES)
    p.add_argument("--phase", choices=("fit", "holdout"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.session_hours_left is not None and (not math.isfinite(args.session_hours_left) or args.session_hours_left <= 0):
        parser().error("--session-hours-left must be a finite positive number")
    output = args.output_dir.resolve()
    try:
        verify_release()
        from . import data, runtime, study
        with exclusive_run(output):
            try:
                if args.command in {"start", "init-runtime"}:
                    # Runtime initialize handles same-allocation idempotence without extending it.
                    hours = args.session_hours_left if args.session_hours_left is not None else slurm_hours_left()
                    init = runtime.initialize(args.v1_dir, output, session_hours_left=hours,
                                              new_allocation=args.new_allocation)
                    print(json.dumps(init, indent=2), flush=True)
                    if args.command == "init-runtime":
                        return 0
                if args.command in {"start", "pipeline"}:
                    for role in ROLES:
                        print(f"V3 FIT COLLECTION: {role}", flush=True)
                        study.run_job(args.v1_dir, output, role, "fit")
                        print("INTERMEDIATE BUNDLE:", study.export(output), flush=True)
                    study.make_freeze(output)
                    for role in ROLES:
                        print(f"V3 HOLDOUT COLLECTION: {role}", flush=True)
                        study.run_job(args.v1_dir, output, role, "holdout")
                        print("INTERMEDIATE BUNDLE:", study.export(output), flush=True)
                    study.evaluate(output)
                    print("FINAL BUNDLE:", study.export(output), flush=True)
                    print("V3_COMPLETE — collection, frozen-policy evaluation and export complete", flush=True)
                    return 0
                if args.command == "status":
                    result = study.status(output)
                elif args.command == "collect":
                    if args.role is None or args.phase is None:
                        raise StudyError("collect requires --role and --phase")
                    result = study.run_job(args.v1_dir, output, args.role, args.phase)
                    study.export(output)
                elif args.command == "fit":
                    result = study.make_freeze(output)
                elif args.command == "evaluate":
                    result = study.evaluate(output)
                elif args.command == "export":
                    result = {"bundle": str(study.export(output))}
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                write_json(output / "diagnostics/last_error.json", {
                    "created_at": utc_now(), "command": args.command, "exception_type": type(exc).__name__,
                    "error": str(exc), "traceback": traceback.format_exc(),
                    "subprocess_log": getattr(exc, "log_path", None)})
                print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
                try:
                    print("DIAGNOSTIC BUNDLE:", study.export(output), flush=True)
                except Exception as export_error:
                    print(f"Diagnostic export failed: {export_error}", file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
    except Exception as exc:
        print(f"BLOCKED before execution: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
