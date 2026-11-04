"""Real HTTP cascades and wall-clock, single-device deployment measurements."""
from __future__ import annotations

import concurrent.futures
import csv
import threading
import subprocess
import time
from pathlib import Path

import numpy as np

from .adapters import ModelClient
from .analysis import apply_gate, rescale_probabilities, validate_calibration
from .common import StudyError, file_hash, fingerprint, read_json, set_status, utc_now, write_json, write_jsonl


def cascade_call(example: dict, local: ModelClient, fallback: ModelClient | None,
                 calibration: dict | None = None) -> dict:
    started = time.perf_counter()
    first = local.predict(example)
    accepted = first["status"] == "ok"
    confidence = None
    if fallback is not None:
        if calibration is None:
            raise StudyError("Live cascade requires a frozen task calibration")
        if accepted:
            p = np.array([[first["probabilities"][l] for l in example["labels"]]], dtype=np.float64)
            p = rescale_probabilities(p, calibration["temperature"]["temperature"])
            confidence = float(p.max())
            accepted = bool(apply_gate(np.array([confidence]), np.array([True]), calibration["gates"]["refitted"])[0])
        second = None if accepted else fallback.predict(example)
        final = first if accepted else second
    else:
        second, final = None, first
    return {
        "sample_id": example["sample_id"], "task": example["task"],
        "prediction": final["prediction"], "status": final["status"],
        "deferred": fallback is not None and not accepted, "local_accepted": accepted,
        "confidence": confidence, "latency_ms": 1000*(time.perf_counter()-started),
        "local": first, "fallback": second,
    }


class GPUMonitor:
    """Read-only process-independent GPU memory samples; no torch import required."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop.is_set():
            try:
                raw = subprocess.check_output([
                    "nvidia-smi", "-i", "0", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"
                ], text=True, timeout=5)
                self.samples.append({"at": time.time(), "gpus": [
                    {"uuid": fields[0].strip(), "used_mib": float(fields[1]), "total_mib": float(fields[2])}
                    for fields in (line.split(",") for line in raw.strip().splitlines())
                ]})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(.5)

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)

    def report(self):
        gpus = [g for sample in self.samples for g in sample["gpus"]]
        return {"samples": len(self.samples), "peak_used_mib": max((g["used_mib"] for g in gpus), default=None),
                "minimum_free_fraction": min(((g["total_mib"]-g["used_mib"])/g["total_mib"] for g in gpus), default=None),
                "device_uuids": sorted({g["uuid"] for g in gpus})}


def measure(examples: list[dict], gold: dict[str, str], local: ModelClient,
            fallback: ModelClient | None, calibration: dict | None, *, concurrency=1,
            min_seconds=30, budget=None, diagnostic_dir: Path | None = None) -> tuple[dict, list[dict]]:
    if not examples:
        raise StudyError("Empty performance workload")
    rows, cycles, failure = [], 0, None
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool, GPUMonitor() as monitor:
        begin = time.perf_counter()
        while cycles == 0 or time.perf_counter()-begin < min_seconds:
            try:
                if budget:
                    budget.check()
            except StudyError as exc:
                failure = str(exc)
                break
            futures = [pool.submit(cascade_call, e, local, fallback, calibration) for e in examples]
            cycle = []
            for future in futures:
                try:
                    cycle.append(future.result())
                except Exception as exc:
                    failure = str(exc)
                    for pending in futures:
                        pending.cancel()
                    break
            rows.extend({**r, "cycle": cycles, "correct": r["status"] == "ok" and r["prediction"] == gold[r["sample_id"]]} for r in cycle)
            cycles += 1
            if failure:
                break
            if sum(r["status"] != "ok" for r in rows) / len(rows) > .01:
                failure = "Performance cell exceeded 1% terminal failures; do not report failure-driven speedups"
                break
        elapsed = time.perf_counter()-begin
    n, success = len(rows), sum(r["status"] == "ok" for r in rows)
    if n == 0:
        if diagnostic_dir is not None:
            write_json(Path(diagnostic_dir)/f"empty-cell-{time.time_ns()}.json", {"failure":failure,"elapsed_seconds":elapsed})
        raise StudyError(failure or "No completed requests in performance cell")
    report = {
        "inputs": n, "unique_inputs": len(examples), "cycles": cycles, "elapsed_seconds": elapsed,
        "concurrency": concurrency, "load_model": "closed_loop", "precision": "bfloat16",
        "p50_ms": float(np.quantile([r["latency_ms"] for r in rows], .5)),
        "p95_ms": float(np.quantile([r["latency_ms"] for r in rows], .95)),
        "throughput_inputs_per_second": n / elapsed, "successful_inputs_per_second": success / elapsed,
        "failure_rate": 1-success/n, "accuracy": sum(r["correct"] for r in rows)/n,
        "fallback_fraction": sum(r["deferred"] for r in rows)/n,
        "local_failure_rate": sum(r["local"]["status"] != "ok" for r in rows)/n,
        "allocated_gpu_seconds_per_1000": 1000*elapsed/n,
        "reserved_gpu_count": 1, "memory": monitor.report(),
    }
    if failure:
        if diagnostic_dir is not None:
            prefix = Path(diagnostic_dir) / f"failed-cell-{time.time_ns()}"
            write_json(prefix.with_suffix(".json"), {**report, "failure": failure})
            write_jsonl(prefix.with_suffix(".trace.jsonl"), rows)
        raise StudyError(failure)
    return report, rows


def run_benchmarks(project_dir: Path, output_dir: Path, *, budget=None, repeats=3, min_seconds=30) -> list[dict]:
    from . import runtime
    from .data import load_examples, load_gold, select_benchmark
    from .runner import protocol_config, require_smoke
    output_dir = Path(output_dir)
    examples, gold = load_examples(output_dir), load_gold(output_dir)
    configs = {m: protocol_config(output_dir, m) for m in ("kev", "startlux", "qwen")}
    for model, config in configs.items():
        require_smoke(output_dir, model, config)
    calibrations = {m: read_json(output_dir / "calibration" / f"{m}.json") for m in ("kev", "startlux")}
    from .storage import load_predictions
    for model, cal in calibrations.items():
        validate_calibration(examples,gold,load_predictions(output_dir,model),cal,model)
        if cal["config_hash"] != fingerprint(configs[model]):
            raise StudyError("Benchmark calibration does not match this model/runtime/GPU")
    directory = output_dir / "benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    settings = {"config_hashes": {m: fingerprint(c) for m,c in configs.items()}, "repeats": repeats,
                "minimum_round_seconds": min_seconds, "calibration_hashes": {m:fingerprint(c) for m,c in calibrations.items()},
                "benchmark_source_hash": file_hash(Path(__file__))}
    settings_path = directory / "settings.json"
    if settings_path.exists() and read_json(settings_path) != settings:
        raise StudyError("Benchmark settings changed; use a new experiment directory")
    write_json(settings_path, settings)
    deployments = [("kev", None), ("startlux", None), ("qwen", None), ("kev", "qwen"), ("startlux", "qwen")]
    load_path = directory / "load_profile.json"
    # Select one common supported concurrency for all deployments using development inputs only.
    if load_path.exists():
        load = read_json(load_path)
        if load["settings_hash"] != fingerprint(settings):
            raise StudyError("Invalid benchmark load profile")
        parallel = load["concurrency"]
    else:
        trial_reports = []
        parallel = 1
        for candidate in (8, 4, 1):
            candidate_ok = True
            for model, fallback_model in deployments:
                with runtime.managed_services(project_dir, output_dir, [model] + ([fallback_model] if fallback_model else [])) as endpoints:
                    local = ModelClient(model, endpoints[model])
                    fallback = ModelClient("qwen", endpoints["qwen"]) if fallback_model else None
                    for task in ("clinc", "squad"):
                        dev = [e for e in examples if e["task"] == task and e["split"] == "development"]
                        # Entire dev pool includes longest input fixtures.
                        cal = calibrations[model]["tasks"][task] if fallback else None
                        try:
                            report, _ = measure(dev, gold, local, fallback, cal, concurrency=candidate, min_seconds=0, budget=budget, diagnostic_dir=directory/"diagnostics")
                            memory_ok = report["memory"]["minimum_free_fraction"] is not None and report["memory"]["minimum_free_fraction"] >= .10
                            candidate_ok &= memory_ok
                            trial_reports.append({"model":model,"fallback":fallback_model,"task":task,"concurrency":candidate,"report":report})
                        except StudyError as exc:
                            if "budget" in str(exc).lower():
                                raise
                            candidate_ok = False
                            trial_reports.append({"model":model,"task":task,"concurrency":candidate,"error":str(exc)})
                if not candidate_ok:
                    break
            if candidate_ok:
                parallel = candidate
                break
        else:
            write_json(directory / "load_failures.json", trial_reports)
            raise StudyError("No verified live deployment load profile, including concurrency 1")
        write_json(load_path, {"settings_hash": fingerprint(settings), "concurrency": parallel, "trials": trial_reports})
    modes = sorted({1, parallel})
    summaries = []
    set_status(output_dir, "benchmark", "running", concurrency=modes)
    for model, fallback_model in deployments:
        deployment = model + ("_to_qwen" if fallback_model else "_direct")
        with runtime.managed_services(project_dir, output_dir, [model] + ([fallback_model] if fallback_model else [])) as endpoints:
            local = ModelClient(model, endpoints[model])
            fallback = ModelClient("qwen", endpoints["qwen"]) if fallback_model else None
            for task in ("clinc", "squad"):
                workload = select_benchmark(examples, gold, task)
                cal = calibrations[model]["tasks"][task] if fallback else None
                dev = [e for e in examples if e["task"] == task and e["split"] == "development"]
                for e in dev:
                    if budget:
                        budget.check()
                    warm = cascade_call(e, local, fallback, cal)
                    if warm["status"] != "ok":
                        raise StudyError("Warm-up failed; inspect model runtime before timing")
                for concurrency in modes:
                    for repeat in range(repeats):
                        key = f"{task}-{deployment}-c{concurrency}-r{repeat}"
                        result_path = directory / f"{key}.json"
                        if result_path.exists():
                            report = read_json(result_path)
                            if report.get("settings_hash") != fingerprint(settings):
                                raise StudyError("Incompatible completed performance cell")
                            summaries.append(report)
                            continue
                        report, traces = measure(workload, gold, local, fallback, cal, concurrency=concurrency,
                                                 min_seconds=min_seconds, budget=budget, diagnostic_dir=directory/"diagnostics")
                        report.update(task=task, deployment=deployment, repeat=repeat, measured_at=utc_now(),
                                      settings_hash=fingerprint(settings), workload_ids=[e["sample_id"] for e in workload])
                        write_jsonl(directory / f"{key}.trace.jsonl", traces)
                        write_json(result_path, report)
                        summaries.append(report)
                        print(f"{key}: p50={report['p50_ms']:.1f}ms p95={report['p95_ms']:.1f}ms throughput={report['throughput_inputs_per_second']:.2f}/s", flush=True)
    write_json(directory / "summary.json", summaries)
    fields = ["task","deployment","repeat","concurrency","inputs","accuracy","fallback_fraction","p50_ms","p95_ms","throughput_inputs_per_second","successful_inputs_per_second","failure_rate","allocated_gpu_seconds_per_1000"]
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    plot_benchmarks(output_dir)
    set_status(output_dir, "benchmark", "complete", cells=len(summaries), concurrency=modes)
    return summaries


def plot_benchmarks(output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_json(Path(output_dir)/"benchmark"/"summary.json")
    for task in ("clinc", "squad"):
        selected = [r for r in rows if r["task"] == task and r["concurrency"] == 1]
        deployments = sorted({r["deployment"] for r in selected})
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for name in deployments:
            subset = [r for r in selected if r["deployment"] == name]
            x = np.median([r["allocated_gpu_seconds_per_1000"] for r in subset])
            y = np.median([r["accuracy"] for r in subset])*100
            axes[0].scatter(x, y, s=45)
            axes[0].annotate(name.replace("_", " "), (x,y), xytext=(4,5), textcoords="offset points", fontsize=8)
        axes[0].set(xlabel="Allocated GPU-seconds / 1,000 inputs", ylabel="Live accuracy (%)", title="Accuracy and allocated GPU time", ylim=(0,105), xlim=(0,None))
        x = np.arange(len(deployments))
        for offset, metric, color in [(-.18,"p50_ms","#3667a6"),(.18,"p95_ms","#c88328")]:
            vals = [np.median([r[metric] for r in selected if r["deployment"]==name]) for name in deployments]
            axes[1].bar(x+offset,vals,.36,label=metric.replace("_ms",""),color=color)
        axes[1].set(xticks=x, xticklabels=[d.replace("_", "\n") for d in deployments], ylabel="Client end-to-end latency (ms)",title="Serial latency; median across rounds")
        axes[1].legend()
        fig.suptitle(f"{task.upper()} — measured live calls, fixed 200-input workload")
        fig.savefig(Path(output_dir)/"benchmark"/f"{task}_performance.png",dpi=160)
        fig.savefig(Path(output_dir)/"benchmark"/f"{task}_performance.pdf")
        plt.close(fig)
        task_rows = [r for r in rows if r["task"] == task]
        modes = sorted({r["concurrency"] for r in task_rows})
        fig, ax = plt.subplots(figsize=(10,5),layout="constrained")
        width = .75/max(1,len(modes))
        for j, mode in enumerate(modes):
            rates = [np.median([r["successful_inputs_per_second"] for r in task_rows if r["deployment"]==d and r["concurrency"]==mode]) for d in deployments]
            ax.bar(x+(j-(len(modes)-1)/2)*width,rates,width,label=f"Concurrency {mode}",color=["#3667a6","#c88328"][j % 2])
        ax.set(xticks=x,xticklabels=[d.replace("_","\n") for d in deployments],ylabel="Successful inputs / second",title=f"{task.upper()} — real serving throughput, median across rounds")
        ax.legend()
        fig.savefig(Path(output_dir)/"benchmark"/f"{task}_throughput.png",dpi=160)
        fig.savefig(Path(output_dir)/"benchmark"/f"{task}_throughput.pdf")
        plt.close(fig)
