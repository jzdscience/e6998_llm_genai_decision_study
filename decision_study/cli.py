"""CLI used unchanged by Colab, VM scripts, and the notebook walkthroughs."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

from .common import StudyError, exclusive_run, set_status, write_json


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="decision-study", description="Frozen decision models: resumable experiment pipeline")
    p.add_argument("command", choices=["doctor","prepare","freeze","install","repair-python","run","evaluate","export","demo","self-test","session"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--models", nargs="+", choices=["kev","startlux","qwen"], default=["kev","startlux","qwen"])
    p.add_argument("--profile", choices=["primary","kev_no_fused","kev_eager","vllm019","reference"], default=None)
    p.add_argument("--job", choices=["smoke","kev_full","startlux_full","qwen_test","benchmark"], default="smoke")
    p.add_argument("--hourly-usd", type=float, default=0)
    p.add_argument("--cost-mode", choices=["metered","provided"], default="metered")
    p.add_argument("--session-hours-left", type=float, default=22)
    p.add_argument("--renew-session", action="store_true")
    p.add_argument("--managed-python-version", default="3.12.8",
                   help="Exact uv-managed CPython patch for explicit repair-python, default 3.12.8")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   help="Explicit vLLM fraction, frozen before smoke: .40 on48GB; .80 for24GB standalone")
    p.add_argument("--kind", choices=["pilot","experiment","benchmark"], default="experiment")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--random-reps", type=int, default=100)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--check-network", action="store_true", help="Bounded public dependency-host probes; no credentials are sent")
    p.add_argument("--sample-id")
    return p


def _prepare(args):
    from .data import prepare_data
    manifest = prepare_data(args.output_dir, cache_dir=args.cache_dir, download=not args.offline)
    set_status(args.output_dir,"data","complete",fingerprint=manifest["fingerprint"])
    return manifest


def _gpu_job(args):
    from . import runtime
    from .artifacts import export_bundle
    from .budget import Budget, ensure_session
    from .runner import run_model, smoke_model
    output, project = args.output_dir, args.project_dir
    # Early budget/device validation prevents an accidental download-only billed session.
    if args.cost_mode == "metered" and args.hourly_usd <= 0:
        raise StudyError("Set --hourly-usd to your actual VM hourly price or Colab compute-unit equivalent")
    runtime.doctor(output,require_gpu=True)
    ensure_session(output,args.session_hours_left)
    _prepare(args)
    models = args.models if args.job == "smoke" else ([args.job.split("_")[0]] if args.job != "benchmark" else ["kev","startlux","qwen"])
    kind = "pilot" if args.job == "smoke" else ("benchmark" if args.job == "benchmark" else "experiment")
    try:
        # Installs, cold starts, reference checks and tokenization are charged to pilot.
        with Budget(output,"pilot",args.hourly_usd,cost_mode=args.cost_mode) as budget:
            for model in models:
                if args.profile:
                    runtime.select_profile(output,model,args.profile)
                if args.gpu_memory_utilization is not None and model != "kev":
                    runtime.select_memory_fraction(output,model,args.gpu_memory_utilization)
                runtime.freeze_lock(project,output)
                runtime.install_environments(project,output,[model])
                budget.check()
                runtime.token_preflight(project,output,[model])
                smoke_model(project,output,model,budget=budget)
        if args.job == "smoke":
            result = {"passed_models":models}
        elif args.job == "benchmark":
            from .benchmark import run_benchmarks
            with Budget(output,"benchmark",args.hourly_usd,cost_mode=args.cost_mode) as budget:
                result = {"cells":len(run_benchmarks(project,output,budget=budget))}
        else:
            with Budget(output,"prediction",args.hourly_usd,cost_mode=args.cost_mode) as budget:
                result = run_model(project,output,models[0],budget=budget)
        set_status(output,f"run/{args.job}","complete")
        return result
    finally:
        path = export_bundle(output,kind)
        print(f"RESULT BUNDLE: {path}",flush=True)
        print("GPU services have been stopped. Shut down the cloud VM in the provider console to stop VM billing.",flush=True)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.project_dir = args.project_dir.expanduser().resolve()
    locks = ExitStack()
    acquired = False
    try:
        locks.enter_context(exclusive_run(args.output_dir))
        acquired = True
        if args.command == "doctor":
            from .runtime import doctor
            result = doctor(args.output_dir,require_gpu=args.require_gpu,check_network=args.check_network)
        elif args.command == "session":
            from .budget import ensure_session
            result = ensure_session(args.output_dir,args.session_hours_left,renew=args.renew_session)
        elif args.command == "prepare":
            result = _prepare(args)
        elif args.command == "freeze":
            from .runtime import freeze_lock
            result = freeze_lock(args.project_dir,args.output_dir)
        elif args.command == "install":
            from . import runtime
            from .budget import Budget, ensure_session
            ensure_session(args.output_dir,args.session_hours_left)
            with Budget(args.output_dir,"pilot",args.hourly_usd,cost_mode=args.cost_mode):
                if args.profile:
                    for model in args.models:
                        runtime.select_profile(args.output_dir,model,args.profile)
                if args.gpu_memory_utilization is not None:
                    for model in args.models:
                        if model != "kev":
                            runtime.select_memory_fraction(args.output_dir,model,args.gpu_memory_utilization)
                result = runtime.install_environments(args.project_dir,args.output_dir,args.models)
        elif args.command == "repair-python":
            from .runtime import repair_python
            from .budget import Budget, ensure_session
            ensure_session(args.output_dir,args.session_hours_left)
            with Budget(args.output_dir,"pilot",args.hourly_usd,cost_mode=args.cost_mode):
                result = repair_python(args.project_dir,args.output_dir,args.models,args.managed_python_version)
        elif args.command == "run":
            result = _gpu_job(args)
        elif args.command == "evaluate":
            from .runner import analyze
            result = analyze(args.output_dir,bootstrap_reps=args.bootstrap_reps,random_reps=args.random_reps)
            result = {"analysis_directory":str(args.output_dir/"analysis"),"status":"complete"}
        elif args.command == "export":
            from .artifacts import export_bundle
            result = {"bundle":str(export_bundle(args.output_dir,args.kind))}
        elif args.command == "demo":
            from .demo import replay
            result = replay(args.output_dir,args.models[0],args.sample_id)
        else:
            from .selftest import run_selftest
            result = run_selftest(args.output_dir)
        if args.command in {"prepare","freeze","install","repair-python","evaluate","session","self-test"}:
            set_status(args.output_dir,args.command,"complete")
        print(json.dumps(result,ensure_ascii=False,indent=2,default=str),flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if not acquired:
            print(f"BLOCKED: {exc}",file=sys.stderr)
            return 2
        stage = f"{args.command}/{args.job}" if args.command == "run" else args.command
        set_status(args.output_dir,stage,"blocked",error=f"{type(exc).__name__}: {exc}")
        diagnostic = {"stage":stage,"exception_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()}
        if getattr(exc, "log_path", None):
            diagnostic["subprocess_log"] = exc.log_path
        write_json(args.output_dir/"diagnostics"/"last_error.json",diagnostic)
        if args.command in {"run","install","repair-python","freeze"}:
            from .artifacts import export_bundle
            try:
                kind = "benchmark" if args.job == "benchmark" else "pilot"
                print(f"DIAGNOSTIC BUNDLE: {export_bundle(args.output_dir,kind)}",file=sys.stderr)
            except Exception as bundle_exc:
                print(f"Could not export diagnostic bundle: {bundle_exc}",file=sys.stderr)
        print(f"BLOCKED: {exc}\nDetails: {args.output_dir/'diagnostics'/'last_error.json'}",file=sys.stderr)
        return 130 if isinstance(exc,KeyboardInterrupt) else 2
    finally:
        locks.close()


if __name__ == "__main__":
    raise SystemExit(main())
