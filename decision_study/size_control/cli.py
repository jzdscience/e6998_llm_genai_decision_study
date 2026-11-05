"""Explicit StartLux9B preparation and node execution; no old sessions renewed."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from ..common import StudyError, exclusive_run, utc_now, write_json
from ..v3.cli import slurm_hours_left
from .identity import PROJECT, ROLE, checkout_identity
from . import data, runtime, study


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "status", "init-runtime", "download", "smoke", "start", "pipeline", "collect", "freeze", "evaluate", "export"))
    p.add_argument("--v1-dir", type=Path, required=True)
    p.add_argument("--v3-dir", type=Path, required=True, help="Completed original V3 results, not only prepared inputs")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--session-hours-left", type=float)
    p.add_argument("--new-allocation", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.session_hours_left is not None and (not math.isfinite(args.session_hours_left) or not 0 < args.session_hours_left <= 24):
        parser().error("--session-hours-left must be a finite number in (0,24]")
    # Check raw path symlinks before resolve can erase them.
    try:
        output = runtime._safe(args.output_dir, "")
        if any(output == Path(p).absolute() or Path(p).absolute() in output.parents or output in Path(p).absolute().parents
               for p in (args.v1_dir, args.v3_dir)):
            raise StudyError("Output must not overlap any original runtime or V3 evidence root")
        checkout = checkout_identity() if args.command not in {"status", "export"} else None
        with exclusive_run(output):
            try:
                if args.command in {"prepare", "start", "pipeline"}:
                    result = data.prepare(args.v3_dir, output, checkout=checkout)
                    if args.command == "prepare":
                        print(json.dumps(result, indent=2)); return 0
                if args.command in {"start", "pipeline"} and (output / "evaluation.json").exists():
                    # Complete immutable evidence is reusable independently of the
                    # current allocation. Validate before any Slurm/session action.
                    saved = study.status(output)
                    if not saved["collection_complete"] or not saved["evaluation_present"]:
                        raise StudyError("Saved evaluation lacks a complete validated prediction chain")
                    print("FINAL BUNDLE:", study.export(output), flush=True)
                    print("SIZE_CONTROL_COMPLETE — verified existing completed experiment reused", flush=True)
                    return 0
                if args.command in {"start", "init-runtime"}:
                    data.verify(output)
                    hours = args.session_hours_left if args.session_hours_left is not None else slurm_hours_left()
                    result = runtime.initialize(args.v1_dir, output, session_hours_left=hours, new_allocation=args.new_allocation)
                    if args.command == "init-runtime":
                        print(json.dumps(result, indent=2)); return 0
                if args.command in {"start", "pipeline"}:
                    study.run_job(args.v1_dir, output)
                    print("INTERMEDIATE BUNDLE:", study.export(output), flush=True)
                    study.evaluate(output)
                    print("FINAL BUNDLE:", study.export(output), flush=True)
                    print("SIZE_CONTROL_COMPLETE — 3200 new predictions and paired retrospective evaluation", flush=True)
                    return 0
                if args.command == "status": result = study.status(output)
                elif args.command == "download": result = runtime.ensure_download(output, ROLE)
                elif args.command == "smoke": result = study.run_job(args.v1_dir, output, smoke_only=True)
                elif args.command == "collect": result = study.run_job(args.v1_dir, output)
                elif args.command == "freeze": result = study.make_freeze(output)
                elif args.command == "evaluate": result = study.evaluate(output)
                elif args.command == "export": result = {"bundle": str(study.export(output))}
                print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
                if args.command in {"smoke", "collect", "evaluate"}:
                    print("BUNDLE:", study.export(output), flush=True)
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                write_json(output / "diagnostics/last_error.json", {"created_at": utc_now(), "command": args.command,
                    "exception_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                    "subprocess_log": getattr(exc, "log_path", None)})
                print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
                try: print("DIAGNOSTIC BUNDLE:", study.export(output), flush=True)
                except Exception as export_error: print(f"Diagnostic export failed: {export_error}", file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
    except Exception as exc:
        print(f"BLOCKED before execution: {exc}", file=sys.stderr); return 2
