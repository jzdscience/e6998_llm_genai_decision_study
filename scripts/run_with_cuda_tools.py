#!/usr/bin/env python3
"""Prepare isolated matching CUDA tools before using the frozen study controller.

No model-environment mutation, session renewal, or change to model configuration.
Matching build wheels are installed into a separate checksummed target directory.
The child owns all Torch/CUDA allocations and exits before model startup.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from decision_study import runtime
from decision_study.artifacts import export_bundle
from decision_study.budget import Budget
from decision_study.common import exclusive_run, read_json, set_status, write_json

STAGE = "cuda-toolchain-preflight"
ENV_KEYS = {"CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC", "PATH", "MAX_JOBS",
            "FLASHINFER_NVCC_THREADS", "FLASHINFER_EXTRA_LDFLAGS"}


def preflight(output: Path) -> dict:
    # A recovery wrapper must never invent a fresh 22-hour session implicitly.
    if not (output / "runtime/session.json").is_file():
        raise RuntimeError("Recorded session missing; set the actual remaining allocation first.")
    lock = read_json(output / "runtime/lock.json")
    installed = read_json(output / "runtime/installations.json")
    if installed["lock_hash"] != lock["lock_hash"]:
        raise RuntimeError("Runtime installation lock mismatch")
    selected_path = output / "runtime/selected_profiles.json"
    selected = read_json(selected_path) if selected_path.exists() else {}
    if any(selected.get(model, "primary") != "primary" for model in ("startlux", "qwen")):
        raise RuntimeError("CUDA recovery requires the recorded primary vLLM profiles")
    name = "vllm-" + lock["packages"]["vllm_primary"]
    info = installed["environments"][name]
    python = Path(info["python"])
    if not python.is_absolute() or not os.access(python, os.X_OK):
        raise RuntimeError("Recorded vLLM Python is missing or not executable")
    if not os.access(python.parent / "ninja", os.X_OK):
        raise RuntimeError("Recorded vLLM Ninja is missing or not executable")
    stamp = time.time_ns()
    result_path = output / "runtime" / f"cuda-toolchain-{stamp}.json"
    log_name = f"cuda-toolchain-{stamp}.log"
    toolkit_target = runtime._work_dir(output) / "toolchains/cuda-13.0-flashinfer-curand10.4.0.35"
    with Budget(output, "pilot", cost_mode="provided") as budget:
        runtime._verify_environment(output, info)
        budget.check()
        print("Preparing isolated CUDA 13.0 tools; the model environment remains unchanged.", flush=True)
        print("Preflight log:", output / "runtime/logs" / log_name, flush=True)
        original_deadline = os.environ["DECISION_STUDY_DEADLINE_EPOCH"]
        os.environ["DECISION_STUDY_DEADLINE_EPOCH"] = str(min(float(original_deadline), time.time() + 1200))
        try:
            runtime._run([python, PROJECT / "scripts/prepare_cuda_toolkit.py",
                          "--target", toolkit_target,
                          "--requirements", PROJECT / "configs/cuda_build_toolkit_13_0_flashinfer.txt",
                          "--result", output / "runtime" / f"cuda-toolkit-setup-{stamp}.json"],
                         output, log_name, cwd=PROJECT)
            print("Checking matched compiler/headers, linker and FlashInfer sampling.", flush=True)
            runtime._run([python, PROJECT / "scripts/cuda_toolchain_preflight.py",
                          "--result", result_path,
                          "--work-dir", runtime._work_dir(output) / "cuda-toolchain-13.0-flashinfer",
                          "--toolkit-target", toolkit_target],
                         output, log_name, cwd=PROJECT)
            runtime._verify_environment(output, info)
        finally:
            os.environ["DECISION_STUDY_DEADLINE_EPOCH"] = original_deadline
        budget.check()
    result = read_json(result_path)
    if result.get("status") != "complete":
        raise RuntimeError(f"CUDA preflight did not pass; see {result_path}")
    environment = result.get("environment", {})
    if set(environment) != ENV_KEYS or not all(isinstance(v, str) for v in environment.values()):
        raise RuntimeError("Invalid preflight environment report")
    print("CUDA / FlashInfer preflight passed:", result_path, flush=True)
    return environment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", nargs="?", default="smoke",
                        choices=["smoke", "startlux_full", "qwen_test", "benchmark"])
    args = parser.parse_args(argv)
    output = Path(os.environ.get("OUTPUT_DIR", PROJECT / "results/solo-v1")).expanduser().resolve()
    os.chdir(PROJECT)
    try:
        with exclusive_run(output):
            try:
                environment = preflight(output)
                set_status(output, STAGE, "complete")
            except (Exception, KeyboardInterrupt) as exc:
                set_status(output, STAGE, "blocked", error=f"{type(exc).__name__}: {exc}")
                write_json(output / "diagnostics/last_error.json", {
                    "stage": STAGE, "exception_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "subprocess_log": getattr(exc, "log_path", None)})
                try:
                    print("DIAGNOSTIC BUNDLE:", export_bundle(output, "pilot"), flush=True)
                except Exception as bundle_exc:
                    print(f"Diagnostic export failed: {bundle_exc}", file=sys.stderr)
                print(f"BLOCKED: {exc}", file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 2
    except Exception as exc:
        # Never mutate an experiment owned by another controller.
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    # The GPU probe has exited and the lock is released before the normal CLI.
    env = dict(os.environ, **environment)
    env["PYTHON_BIN"] = os.path.abspath(sys.executable)
    env["OUTPUT_DIR"] = str(output)
    os.execvpe("bash", ["bash", str(PROJECT / "scripts/run_with_vllm_tools.sh"), args.job], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
