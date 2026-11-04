"""Explicit v2 entry point; the v1 CLI and frozen computation stay unchanged."""
import argparse
import json
from pathlib import Path
import traceback

from ..common import exclusive_run, utc_now, write_json
from .identity import verify_frozen_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "init-runtime", "run", "fit", "evaluate", "benchmark", "pipeline", "export", "status"):
        p = commands.add_parser(name)
        p.add_argument("--v1-dir", type=Path, default=Path("results/solo-v1"))
        p.add_argument("--output-dir", type=Path, default=Path("results/solo-v2"))
        if name == "prepare":
            p.add_argument("--raw-dir", type=Path, required=True)
        if name == "init-runtime":
            p.add_argument("--session-hours-left", type=float, required=True)
            p.add_argument("--new-allocation", action="store_true", help="Confirm this is a real newly obtained allocation, not an extension of an expired session")
        if name == "run":
            p.add_argument("--job", required=True, choices=["qwen_fit", "holdout_startlux", "holdout_qwen", "holdout_kev"])
    p = commands.add_parser("lab-build")
    p.add_argument("--v1-dir", type=Path, required=True)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p = commands.add_parser("lab")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    if args.command == "lab":
        from .lab.app import launch
        launch(args.bundle, port=args.port)
        return 0
    if args.command == "lab-build":
        from .lab.bundle import build
        print(build(args.v1_dir, args.benchmark_dir, args.bundle))
        return 0
    output = args.output_dir.resolve()
    source = args.v1_dir.resolve()
    if output == source or source in output.parents or output in source.parents:
        print("BLOCKED: v1 and v2 directories must be separate, without ancestor/descendant overlap.", flush=True)
        return 2
    try:
        verify_frozen_source()
        if args.command == "status":
            from .data_v2 import verify_fit
            info = {"data": verify_fit(output), "policies_frozen": (output/"policy_freeze.json").exists(),
                    "holdout_evaluated": (output/"evaluation.json").exists()}
        else:
            with exclusive_run(output):
                if args.command == "prepare":
                    from .data_v2 import prepare
                    info = prepare(args.v1_dir, args.raw_dir, output)
                elif args.command == "init-runtime":
                    from .runner_v2 import initialize_runtime
                    info = initialize_runtime(args.v1_dir, output, session_hours_left=args.session_hours_left,
                                              new_allocation=args.new_allocation)
                elif args.command == "export":
                    from .runner_v2 import export
                    info = {"bundle": str(export(output))}
                elif args.command == "fit":
                    from .study import fit
                    info = fit(args.v1_dir, output)
                elif args.command == "evaluate":
                    from .study import evaluate
                    info = evaluate(args.v1_dir, output)
                else:
                    from .runner_v2 import runtime_context, run_job, export
                    try:
                        stage = "benchmark" if args.command == "benchmark" else "prediction"
                        with runtime_context(args.v1_dir, output, stage) as (control, budget):
                            if args.command == "run":
                                info = run_job(args.v1_dir, output, args.job, control=control, budget=budget)
                            elif args.command == "benchmark":
                                from .openloop import run
                                info = run(args.v1_dir, output, control=control, budget=budget)
                            else:
                                from .study import fit, evaluate
                                from .openloop import run
                                info = {}
                                info["qwen_fit"] = run_job(args.v1_dir, output, "qwen_fit", control=control, budget=budget)
                                export(output)
                                info["freeze"] = fit(args.v1_dir, output)["freeze_hash"]
                                for job in ("holdout_startlux", "holdout_qwen"):
                                    info[job] = run_job(args.v1_dir, output, job, control=control, budget=budget)
                                    export(output)
                                info["openloop"] = run(args.v1_dir, output, control=control, budget=budget)
                                info["evaluation"] = evaluate(args.v1_dir, output)
                    except (Exception, KeyboardInterrupt) as exc:
                        write_json(output/"diagnostics/last_error.json", {"created_at": utc_now(), "stage": args.command,
                            "exception": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
                        raise
                    finally:
                        print("V2 RETURN BUNDLE:", export(output), flush=True)
        # Avoid printing whole datasets or gold. Commands return summaries/artifacts only.
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # A conflicting owner must not have its output changed by this process.
        if "Another command is using" not in str(exc):
            write_json(output/"diagnostics/last_error.json", {"created_at": utc_now(), "stage": args.command,
                        "exception": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        print(f"BLOCKED: {exc}", flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
