"""Prepare, preflight, collect and score the 8 missing cells of the five-LLM direct-accuracy matrix."""
import argparse
import json
from pathlib import Path
import signal
import sys
import traceback

from ..common import StudyError, exclusive_run, utc_now, write_json
from ..v3.cli import slurm_hours_left
from . import data, runtime, study
from .identity import normal, safe

COMMANDS = ("prepare", "init-runtime", "preflight", "run", "freeze", "evaluate", "start", "status", "export", "audit")


class Terminated(KeyboardInterrupt):
    """SIGTERM/SIGHUP from the session manager; unwinds through every cleanup stack."""


def _terminate(signum, frame):
    raise Terminated(f"received signal {signum}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--v1-dir", type=Path, help="Original V1 runtime and results (results/solo-v1)")
    p.add_argument("--v2-dir", type=Path, help="Completed V2 study (results/solo-v2)")
    p.add_argument("--v3-dir", type=Path, help="Completed V3 study (results/solo-v3)")
    p.add_argument("--size-control-dir", type=Path, help="Completed StartLux-9B size control (results/aux-size-control-v1-r2)")
    p.add_argument("--session-hours-left", type=float,
                   help="Only when squeue is unavailable: the verified remaining allowance of THIS allocation")
    p.add_argument("--new-allocation", action="store_true", help="Acknowledge a genuinely different Slurm job")
    p.add_argument("--resume", action="store_true", help="Continue after an interrupted attempt; accepted rows are kept")
    p.add_argument("--retry-preflight", action="store_true",
                   help="Retry a role whose earlier preflight failed for a transient cause; the failure stays preserved")
    p.add_argument("--bundle", type=Path, help="audit: returned bundle ZIP, extracted into the new --output-dir first")
    p.add_argument("--report", type=Path, help="audit: write the JSON report here (outside the audited root)")
    return p


def _audit(args, output):
    """Read-only offline acceptance; never takes the run lock or writes inside the audited root."""
    from . import audit
    if args.bundle is not None:
        audit.extract_bundle(args.bundle, output)
    report = audit.audit(output)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report is not None:
        target = normal(args.report)
        if target == output or output in target.parents:
            raise StudyError("Write the audit report outside the audited root")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n")
    print(text)
    return 0


def _dirs(args):
    return data.source_dirs(args.v1_dir, args.v2_dir, args.v3_dir, args.size_control_dir)


def _initialize(args, output):
    value = study.settings(output, check_source=True)
    hours = args.session_hours_left if args.session_hours_left is not None else slurm_hours_left()
    return runtime.initialize(value["source_dirs"], output, hours, new_allocation=args.new_allocation)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    needs_inputs = args.command in {"prepare", "start"}
    if needs_inputs and any(getattr(args, k) is None for k in ("v1_dir", "v2_dir", "v3_dir", "size_control_dir")):
        p.error("prepare/start require --v1-dir --v2-dir --v3-dir --size-control-dir")
    output = normal(args.output_dir)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _terminate)
    try:
        safe(output, ".active.lock")
        if args.command == "audit":
            return _audit(args, output)
        if args.command == "status":
            result = study.status(output)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 1 if result["problems"] else 0
        with exclusive_run(output):
            try:
                if needs_inputs:
                    result = study.prepare(_dirs(args), output)
                    if args.command == "prepare":
                        print(json.dumps(result, indent=2))
                        return 0
                    print(f"Prepared: {result['cells']} cells, {result['new_predictions']} new test predictions planned; "
                          f"test gold unread", flush=True)
                if args.command == "export":
                    print("BUNDLE:", study.export(output))
                    return 0
                if args.command == "freeze":
                    print(json.dumps(study.make_freeze(output), indent=2))
                    return 0
                if args.command == "evaluate" or (args.command == "start" and safe(output, "evaluation.json").exists()):
                    study.evaluate(output)
                    print("DIRECT_MATRIX_COMPLETE — verified evaluation of all 20 cells", flush=True)
                    print("FINAL BUNDLE:", study.export(output), flush=True)
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
                study.run(output, resume=args.resume)
                print("COLLECTION BUNDLE:", study.export(output), flush=True)
                study.evaluate(output)
                print("DIRECT_MATRIX_COMPLETE — 20,400 new test predictions scored with the 12 accepted cells", flush=True)
                print("FINAL BUNDLE:", study.export(output), flush=True)
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                write_json(safe(output, "diagnostics/last_error.json"), {
                    "at": utc_now(), "command": args.command, "error": str(exc), "type": type(exc).__name__,
                    "traceback": traceback.format_exc(), "subprocess_log": getattr(exc, "log_path", None)})
                print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
                try:
                    print("DIAGNOSTIC BUNDLE:", study.export(output), flush=True)
                except Exception as export_error:  # noqa: BLE001
                    print("Diagnostic export failed:", export_error, file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
    except StudyError as exc:
        print(f"BLOCKED before execution: {exc}", file=sys.stderr)
        return 2
