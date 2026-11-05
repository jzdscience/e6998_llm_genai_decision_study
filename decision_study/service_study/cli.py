"""Prepare or run the independently frozen CPU/GPU service study."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from ..common import StudyError, exclusive_run, utc_now, write_json
from ..v3.cli import slurm_hours_left
from . import runtime, study
from .identity import safe


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "init-runtime", "start", "pipeline", "evaluate", "status", "export"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--v1-dir", type=Path)
    p.add_argument("--bundle-dir", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--cpu-python", type=Path)
    p.add_argument("--session-hours-left", type=float)
    p.add_argument("--new-allocation", action="store_true")
    p.add_argument("--resume-interrupted", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command in {"prepare", "start"} and any(getattr(args, k) is None for k in ("v1_dir", "bundle_dir", "model_dir", "cpu_python")):
        p.error("prepare/start require --v1-dir --bundle-dir --model-dir --cpu-python")
    output = args.output_dir.expanduser().absolute()
    try:
        safe(output, ".active.lock")
        with exclusive_run(output):
            try:
                if args.command in {"prepare", "start"}:
                    result = study.prepare(args.v1_dir, args.bundle_dir, args.model_dir, args.cpu_python, output)
                    if args.command == "prepare":
                        print(json.dumps(result, indent=2)); return 0
                if args.command == "status":
                    print(json.dumps(study.status(output), indent=2)); return 0
                if args.command == "export":
                    print("BUNDLE:", study.export(output)); return 0
                if args.command in {"start", "init-runtime"}:
                    config = study.settings(output, check_source=True)
                    if args.command == "start" and (output/"evaluation.json").exists():
                        study.evaluate(output)
                        print("SERVICE_STUDY_COMPLETE — verified existing evidence")
                        print("FINAL BUNDLE:", study.export(output)); return 0
                    hours = args.session_hours_left if args.session_hours_left is not None else slurm_hours_left()
                    result = runtime.initialize(config["v1_dir"], output, hours, new_allocation=args.new_allocation)
                    if args.command == "init-runtime":
                        print(json.dumps(result, indent=2)); return 0
                if args.command in {"start", "pipeline"}:
                    study.run(output, resume_interrupted=args.resume_interrupted)
                    print("MEASUREMENT BUNDLE:", study.export(output), flush=True)
                    study.evaluate(output)
                    print("SERVICE_STUDY_COMPLETE —60 measured cells and actual-response quality", flush=True)
                    print("FINAL BUNDLE:", study.export(output), flush=True)
                elif args.command == "evaluate":
                    study.evaluate(output)
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
    except Exception as exc:
        print(f"BLOCKED before execution: {exc}", file=sys.stderr); return 2
