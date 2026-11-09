"""Single-model service launcher for the upgrade study.

This is the frozen V1 ``managed_services`` for one model, built from V1's own
helpers, with one difference: each vLLM engine command gets ``--max-num-seqs``
from the protocol. vLLM 0.31 refuses CUDA graph capture on these hybrid models
when ``max_num_seqs`` (default 256) exceeds the Mamba state blocks that fit in
the memory share; the node preflight of 2026-11-08 found 151 for StartLux-9B at
0.50. V1 passes no such flag and strips ``VLLM_*`` variables from the service
environment, so the command is extended here. V1's source is not edited.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time

from .. import runtime as frozen
from .identity import PROJECT

ENGINE_FLAG = "--max-num-seqs"


def engine_specs(project_dir, output_dir, model, profile, lock, installations, max_num_seqs):
    """V1's service specs for one model, with the sequence limit appended to its vLLM engine only."""
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs < 1:
        raise ValueError("max_num_seqs must be a positive integer")
    specs = frozen._service_specs(project_dir, output_dir, model, profile, lock, installations)
    engines = [spec for spec in specs if spec["name"] == f"{model}-engine"]
    if len(engines) != 1 or "vllm.entrypoints.openai.api_server" not in engines[0]["command"] or ENGINE_FLAG in engines[0]["command"]:
        raise RuntimeError(f"Expected exactly one vLLM engine for {model} without a sequence limit")
    engines[0]["command"] = [*engines[0]["command"], ENGINE_FLAG, str(max_num_seqs)]
    return specs


@contextmanager
def managed_service(output_dir, model, *, max_num_seqs, project_dir=PROJECT):
    """Start one model's owned process groups exactly as V1 does, plus the engine limit; always stop them."""
    if model not in frozen.MODELS or model == "kev":
        raise ValueError("Upgrade-study services are StartLux or Qwen")
    output_dir = Path(output_dir).resolve()
    frozen.doctor(output_dir, require_gpu=True)
    lock = frozen._read(output_dir / "runtime/lock.json")
    installations = frozen._read(output_dir / "runtime/installations.json")
    if installations["lock_hash"] != lock["lock_hash"]:
        raise RuntimeError("Runtime installation lock mismatch")
    processes, endpoints = [], {}
    metadata = {"started_at": frozen._now(), "models": {}, "processes": [], "max_num_seqs": max_num_seqs}
    try:
        selected_profile = frozen._resolve_profile(output_dir, model, None)
        env_name = frozen._environment_name(model, selected_profile, lock)
        frozen._verify_environment(output_dir, installations["environments"][env_name])
        for spec in engine_specs(project_dir, output_dir, model, selected_profile, lock, installations, max_num_seqs):
            frozen._check_port(spec["port"])
            log_path = output_dir / "runtime/logs" / f"{spec['name']}-{time.time_ns()}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
            try:
                process = subprocess.Popen(spec["command"], env=spec["env"], stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            finally:
                log.close()
            processes.append(process)
            metadata["processes"].append({"name": spec["name"], "pid": process.pid, "command": spec["command"],
                                          "log": str(log_path)})
            frozen._write(output_dir / "runtime/last_services.json", metadata)
            health = frozen._wait_ready(process, spec, log_path, lock["startup_timeout_seconds"])
            if model == "startlux" and spec["name"] == model:
                if health.get("prompt_attention") != "bidirectional" or health.get("defaults", {}).get("think_mode") != "never":
                    raise RuntimeError("StartLux health reports the wrong attention or thinking mode")
            metadata["models"][model] = {"health": health, "config": frozen.runtime_config(output_dir, model, selected_profile)}
        endpoints[model] = f"http://127.0.0.1:{lock['models'][model]['port']}"
        metadata["ready_at"] = frozen._now()
        frozen._write(output_dir / "runtime/last_services.json", metadata)
        yield endpoints
    finally:
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in reversed(processes):
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        metadata["stopped_at"] = frozen._now()
        frozen._write(output_dir / "runtime/last_services.json", metadata)
