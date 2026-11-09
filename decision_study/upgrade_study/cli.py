"""Prepare, preflight and run the ANLI upgrade-versus-fallback serving study."""
import argparse
import json
from pathlib import Path
import signal
import sys
import traceback

from ..common import StudyError, exclusive_run, utc_now, write_json
from ..v3.cli import slurm_hours_left
from . import runtime, study
from .identity import safe

COMMANDS = ("prepare", "init-runtime", "preflight", "run", "start", "evaluate", "status", "export")


class Terminated(KeyboardInterrupt):
    """SIGTERM/SIGHUP from the session manager; unwinds through every cleanup stack."""


def _terminate(signum, frame):
    raise Terminated(f"received signal {signum}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--v1-dir", type=Path)
    p.add_argument("--v3-dir", type=Path)
    p.add_argument("--size-control-dir", type=Path)
    p.add_argument("--session-hours-left", type=float)
    p.add_argument("--new-allocation", action="store_true")
    p.add_argument("--resume-interrupted", action="store_true")
    p.add_argument("--retry-preflight", action="store_true",
                   help="Retry a condition whose earlier preflight failed for a transient cause; the failure stays preserved")
    return p


def _initialize(args, output):
    config = study.settings(output, check_source=True)
    hours = args.session_hours_left if args.session_hours_left is not None else slurm_hours_left()
    return runtime.initialize(config["v1_dir"], config["v3_dir"], config["size_control_dir"], output, hours,
                              new_allocation=args.new_allocation)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    needs_inputs = args.command in {"prepare", "start"}
    if needs_inputs and any(getattr(args, k) is None for k in ("v1_dir", "v3_dir", "size_control_dir")):
        p.error("prepare/start require --v1-dir --v3-dir --size-control-dir")
    output = args.output_dir.expanduser().absolute()
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _terminate)
    try:
        safe(output, ".active.lock")
        with exclusive_run(output):
            try:
                if needs_inputs:
                    result = study.prepare(args.v1_dir, args.v3_dir, args.size_control_dir, output)
                    if args.command == "prepare":
                        print(json.dumps(result, indent=2))
                        return 0
                    print(f"Prepared: {result['cells']} cells, {result['offers_planned']} offers; historical deferrals "
                          f"{result['historical_deferrals']}", flush=True)
                if args.command == "status":
                    print(json.dumps(study.status(output), indent=2))
                    return 0
                if args.command == "export":
                    print("BUNDLE:", study.export(output))
                    return 0
                if args.command == "evaluate":
                    study.evaluate(output)
                    print("FINAL BUNDLE:", study.export(output), flush=True)
                    return 0
                if args.command == "start" and (output / "evaluation.json").exists():
                    study.evaluate(output)
                    print("UPGRADE_STUDY_COMPLETE — verified existing evidence")
                    print("FINAL BUNDLE:", study.export(output))
                    return 0
                if args.command in {"init-runtime", "start"}:
                    result = _initialize(args, output)
                    if args.command == "init-runtime":
                        print(json.dumps(result, indent=2))
                        return 0
                if args.command in {"preflight", "start"}:
                    print("PREFLIGHT:", json.dumps(study.preflight(output, retry_failed=args.retry_preflight)), flush=True)
                    if args.command == "preflight":
                        print("PREFLIGHT BUNDLE:", study.export(output), flush=True)
                        return 0
                study.run(output, resume_interrupted=args.resume_interrupted)
                print("MEASUREMENT BUNDLE:", study.export(output), flush=True)
                study.evaluate(output)
                print("UPGRADE_STUDY_COMPLETE — 36 measured cells scored from actual responses", flush=True)
                print("FINAL BUNDLE:", study.export(output), flush=True)
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                write_json(safe(output, "diagnostics/last_error.json"), {"at": utc_now(), "command": args.command,
                           "error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()})
                print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
                try:
                    print("DIAGNOSTIC BUNDLE:", study.export(output), flush=True)
                except Exception as export_error:
                    print("Diagnostic export failed:", export_error, file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
    except StudyError as exc:
        print(f"BLOCKED before execution: {exc}", file=sys.stderr)
        return 2
