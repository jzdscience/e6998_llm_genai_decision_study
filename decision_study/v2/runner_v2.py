"""Reuse pinned model services with separate v2 journals and a writable control plane."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from .. import runtime
from ..adapters import ModelClient, decision_request, qwen_request
from ..budget import Budget, ensure_session
from ..common import StudyError, file_hash, fingerprint, read_json, utc_now, write_json
from ..data import load_examples as v1_examples, verify_data
from ..runner import predict_examples, protocol_config, require_smoke, _probability_comparison
from ..storage import PredictionStore
from . import data_v2
from .identity import PROJECT, job_root, protocol, source_identity, verify_frozen_source


def initialize_runtime(v1_dir, output_dir, *, session_hours_left, new_allocation=False):
    """Explicit operator action after obtaining an allocation; never invoked by prepare."""
    verify_frozen_source()
    if not new_allocation:
        raise StudyError("Use --new-allocation only after obtaining an actual code-server allocation")
    source, output = Path(v1_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise StudyError("v2 must use a separate output directory")
    verify_data(source)
    control = output / "runtime-control"
    names = ("lock.json", "installations.json", "doctor.json", "selected_profiles.json", "selected_memory.json")
    hashes = {}
    for name in names:
        src = source / "runtime" / name
        if not src.exists():
            continue
        dst = control / "runtime" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and name != "doctor.json" and file_hash(dst) != file_hash(src):
            raise StudyError(f"v2 runtime identity changed: {name}; use a new output directory")
        shutil.copyfile(src, dst)
        hashes[name] = file_hash(src)
    for name in ("lock.json", "installations.json", "doctor.json"):
        if name not in hashes:
            raise StudyError(f"Missing verified v1 runtime/{name}")
    # Reuse absolute installed model/environment paths; do not copy GBs of weights.
    work = runtime._work_dir(source)
    binding = {"v1_dir": str(source), "runtime_work_dir": str(work), "runtime_files": hashes}
    existing = control / "binding.json"
    if existing.exists():
        old = read_json(existing)
        if any(old[k] != binding[k] for k in ("v1_dir", "runtime_work_dir")):
            raise StudyError("Runtime paths changed; create a separate runtime-control directory")
    write_json(existing, binding)
    session = ensure_session(control, session_hours_left, renew=True)
    return {"runtime_control": str(control), "session": session, "notice": "Operator supplied allocation; no provider expiry detection"}


@contextmanager
def runtime_context(v1_dir, output_dir, stage="prediction", *, tool_preflight=True):
    source, output = Path(v1_dir).resolve(), Path(output_dir).resolve()
    control = output / "runtime-control"
    if not (control / "runtime/session.json").is_file():
        raise StudyError("Initialize the v2 runtime with the actual remaining allocation first")
    binding = read_json(control / "binding.json")
    if binding["v1_dir"] != str(source):
        raise StudyError("v1 source differs from the initialized runtime")
    for name, expected in binding["runtime_files"].items():
        if name != "doctor.json" and file_hash(control / "runtime" / name) != expected:
            raise StudyError(f"Runtime control identity changed: {name}")
    previous = dict(os.environ)
    try:
        os.environ["DECISION_STUDY_RUNTIME_DIR"] = binding["runtime_work_dir"]
        verify_frozen_source()
        if tool_preflight:
            from scripts.run_with_cuda_tools import preflight
            os.environ.update(preflight(control))
        with Budget(control, stage, cost_mode="provided") as budget:
            runtime.doctor(control, require_gpu=True)
            yield control, budget
    finally:
        # Context used only by the single CLI owner, never a threaded UI callback.
        os.environ.clear()
        os.environ.update(previous)


def verify_runtime_model(v1_dir, control, model):
    original = protocol_config(Path(v1_dir), model)
    require_smoke(Path(v1_dir), model, original)
    effective = runtime.runtime_config(control, model)
    if effective != original["runtime"]:
        raise StudyError(f"{model} effective runtime differs from v1; do not silently reuse its gate")
    return effective


def prediction_config(v1_dir, output_dir, job, control):
    manifest = data_v2.verify_fit(output_dir)
    model = "qwen" if job == "qwen_fit" else job.removeprefix("holdout_")
    examples = data_v2.load_examples(output_dir, role="qwen_fit" if job == "qwen_fit" else "holdout")
    renderer = qwen_request if model == "qwen" else lambda e: decision_request(e, model)
    return {"protocol": "solo-v2", "job": job, "model": model,
            "data_manifest_hash": fingerprint(manifest), "sources": source_identity(),
            "protocol_hash": fingerprint(protocol()), "runtime": verify_runtime_model(v1_dir, control, model),
            "request_hash": fingerprint([{ "id": e["sample_id"], "request": renderer(e)} for e in examples]),
            "expected_predictions": len(examples), "max_attempts": 2, "timeout_seconds": 120,
            "v1_config_hash": fingerprint(protocol_config(Path(v1_dir), model))}


def run_job(v1_dir, output_dir, job, *, control, budget):
    """No gold is loaded: collecting holdout predictions does not score the holdout."""
    config = prediction_config(v1_dir, output_dir, job, control)
    model = config["model"]
    examples = data_v2.load_examples(output_dir, role="qwen_fit" if job == "qwen_fit" else "holdout")
    jobdir = job_root(output_dir, job)
    store = PredictionStore(jobdir, model, config)
    ids = {e["sample_id"] for e in examples}
    if not set(store.rows) <= ids:
        raise StudyError("Unexpected IDs in v2 prediction journal")
    done = jobdir / "completion.json"
    if len(store.rows) == len(ids) and done.exists():
        receipt = read_json(done)
        if receipt.get("config_hash") != fingerprint(config):
            raise StudyError("Completed job identity mismatch")
        return receipt
    budget.check()
    tokens = runtime.token_preflight_examples(PROJECT, control, model, examples)
    if tokens.get("status") != "ok":
        raise StudyError("Exact tokenizer preflight failed; no silent truncation")
    # Recheck the known development cases on this allocation; no holdout labels.
    dev = v1_examples(Path(v1_dir), split="development")
    original_smoke = read_json(Path(v1_dir) / "smoke" / f"{model}.json")
    with runtime.managed_services(PROJECT, control, [model]) as endpoints:
        client = ModelClient(model, endpoints[model])
        current = []
        for example in dev:
            budget.check()
            current.append(client.predict(example))
        checks = _probability_comparison(dev, original_smoke["serial"], current)
        write_json(jobdir / "smoke.json", {"config_hash": fingerprint(config), "checks": checks,
                                         "rows": current, "created_at": utc_now()})
        if not all(v["passed"] for v in checks.values()):
            raise StudyError("New allocation serial smoke differs from v1; inspect before collecting")
        result = predict_examples(examples, client, store, budget=budget)
    if set(store.rows) != ids:
        raise StudyError("Job is not complete; resume with the identical configuration")
    receipt = {"job": job, "model": model, "count": len(ids), "config_hash": fingerprint(config),
               "failures": sum(r["status"] != "ok" for r in store.rows.values()),
               "created_at": utc_now(), "sampling": result, "scored_holdout": False}
    write_json(done, receipt)
    return receipt


def export(output_dir):
    """Small return bundle; never follows symlinks or includes weights/venvs/caches."""
    output = Path(output_dir).resolve()
    export_dir = output / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for parent, dirs, names in os.walk(output, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {"exports", "work", "__pycache__", ".git", "envs", "weights", "cache", "models"}
                   and not d.startswith((".venv", "venv"))
                   and not (Path(parent)/d).is_symlink()]
        for name in names:
            p = Path(parent)/name
            if p.is_symlink() or name in {".active.lock", "bundle_manifest.json"} or name.endswith(".tmp"):
                continue
            files.append(p)
    entries = {p.relative_to(output).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
               for p in sorted(files)}
    path = export_dir / "v2_bundle.zip"
    temp = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.relative_to(output).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps({"protocol": "solo-v2", "created_at": utc_now(),
                              "files": entries}, ensure_ascii=False, indent=2))
    temp.replace(path)
    return path
