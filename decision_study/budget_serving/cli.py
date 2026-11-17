"""Command line: ``start`` (the one resumable command), ``worker`` (supervised only), ``status``,
``audit`` (offline-capable), ``export`` and ``join`` (reassemble split archives)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from ..common import StudyError, exclusive_run, utc_now, write_json
from . import evaluation, export, pipeline, study, worker


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def parser():
    p = argparse.ArgumentParser(prog="python -m decision_study.budget_serving", description=__doc__)
    p.add_argument("command", choices=("start", "worker", "status", "audit", "export", "join"))
    p.add_argument("--runtime-root", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--attempt-dir", type=Path)
    p.add_argument("--retry-failed", action="store_true", help="Preserve a failed attempt and start a new one")
    p.add_argument("--gold-source", type=Path, help="audit: also check the returned gold subset against this Study 4 file")
    p.add_argument("--online", action="store_true", help="audit on the node: also re-check runtime sources and pinned gold")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--target", type=Path)
    return p


def _worker(args):
    try:
        if args.output_dir is None or args.attempt_dir is None:
            raise StudyError("worker requires --output-dir and --attempt-dir")
        worker.main(args.output_dir, args.attempt_dir)
        return 0
    except BaseException as exc:
        try:
            folder = study.normal(args.attempt_dir)
            if study.normal(args.output_dir) in folder.parents and folder.is_dir():
                write_json(study.path(folder, f"worker-error-{time.time_ns()}.json"),
                           {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc(), "at": utc_now()})
        except BaseException:
            pass
        print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 2


def main(argv=None):
    args = parser().parse_args(argv)
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _interrupt)
    if args.command == "worker":
        return _worker(args)
    if args.command == "join":
        if args.manifest is None or args.target is None:
            print("BLOCKED: join requires --manifest and --target", file=sys.stderr)
            return 2
        try:
            print(f"JOINED: {export.join(args.manifest, args.target)}")
            return 0
        except (StudyError, OSError, KeyError, ValueError) as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 2
    try:
        if args.output_dir is None:
            raise StudyError("--output-dir is required")
        if args.command == "start" and args.runtime_root is None:
            raise StudyError("start requires --runtime-root")
        output = study.validate_output(args.output_dir, args.runtime_root if args.command == "start" else None,
                                       existing=args.command != "start")
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    if args.command == "status":
        print(json.dumps(study.status(output), ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit":
        try:
            ctx = study.context(output, offline=not args.online)
            print(json.dumps(evaluation.audit(ctx, offline=not args.online, gold_source=args.gold_source), ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"AUDIT FAILED: {exc}", file=sys.stderr)
            return 2
    with exclusive_run(output):
        try:
            if args.command == "export":
                print(f"BUNDLE: {export.export(output, 'manual')}", flush=True)
                return 0
            result = pipeline.run(output, args.runtime_root, retry_failed=args.retry_failed)
            print(f"STUDY_{result}", flush=True)
            if result == "COMPLETE":
                try:  # optional Study 4 supplement; it can never change the serving result or exit code
                    supplement = pipeline.capture_embeddings(output)
                except Exception as exc:
                    supplement = {"status": "NOT_RECORDED", "error": str(exc)}
                print(f"OPTIONAL EMBEDDING SUPPLEMENT: {supplement.get('status')} (archive only; no text-head refit)", flush=True)
                bundle = export.export(output, "result")
                print(f"RESULT BUNDLE: {bundle}", flush=True)
                try:  # convenience listing after the complete result archive; never changes the outcome
                    print(f"TRANSFER MANIFEST: {pipeline.transfer_manifest(output, bundle, supplement)}", flush=True)
                except Exception as exc:
                    print(f"TRANSFER MANIFEST NOT WRITTEN: {exc}; download the RESULT BUNDLE above", file=sys.stderr, flush=True)
            else:
                print(f"CHECKPOINT BUNDLE: {export.export(output, 'checkpoint')}", flush=True)
                print("Run the same command again in a genuinely new Slurm allocation to resume.", flush=True)
            return 0
        except BaseException as exc:
            try:
                write_json(study.path(output, f"diagnostics/failure-{time.time_ns()}.json"),
                           {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc(), "at": utc_now()})
                print(f"DIAGNOSTIC BUNDLE: {export.export(output, 'diagnostic')}", flush=True)
            except BaseException as exporting:
                print(f"Diagnostic export failed: {exporting}", file=sys.stderr)
            print(f"BLOCKED: {exc}", file=sys.stderr, flush=True)
            return 2
