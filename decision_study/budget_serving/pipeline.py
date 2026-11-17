"""The single resumable flow: verify inputs -> allocation and ownership -> tokens -> preflights -> 60 cells
-> freeze -> gold -> evaluate -> audit -> report. Every successful gate continues automatically."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import StudyError, canonical, file_hash, read_json, utc_now
from ..service_study.identity import fixed_write
from . import evaluation, runtime, study
from .protocol import DEPLOYMENTS, PROJECT

AMBIENT = ("DECISION_STUDY_DEADLINE_EPOCH", "DECISION_BUDGET_WORKER_OUTPUT", "DECISION_BUDGET_WORKER_ATTEMPT")
EMBEDDING_HELPER = PROJECT / "scripts/archive_learnability_embeddings.py"
EMBEDDING_TARGET = "exports/learnability-embeddings"
EMBEDDING_SECONDS = 3600


def ensure_selftest(output):
    """One native ownership self-test per actual allocation, bound to that allocation."""
    if study.path(output, "runtime/ownership-selftest.json").exists():
        try:
            return runtime.verify_native_selftest(output)
        except StudyError:
            pass  # belongs to an earlier allocation; a new receipt is required
    pointer = runtime.native_selftest(output)
    runtime.verify_native_selftest(output, pointer)
    return pointer


def _checkpoint(output, stage, info):
    study.stage_status(output, stage, "CHECKPOINT", **{k: v for k, v in info.items() if k != "stage"})
    return "CHECKPOINT"


def run(output, runtime_root, *, retry_failed=False):
    present = [name for name in AMBIENT if os.environ.get(name)]
    if present:
        raise StudyError(f"Refusing ambient worker/deadline variables in the controller: {present}")
    ctx = study.prepare(output, runtime_root)
    output = ctx.output
    if study.path(output, "evaluation/complete.json").exists():
        evaluation.audit(ctx)
        evaluation.report(ctx)
        study.stage_status(output, "study", "COMPLETE")
        return "COMPLETE"
    if not study.path(output, "freeze.json").exists():
        study.stage_status(output, "allocation", "RUNNING")
        runtime.initialize_allocation(output)
        ensure_selftest(output)
        study.stage_status(output, "tokens", "RUNNING")
        state, info = study.tokens_stage(ctx, retry_failed=retry_failed)
        if state == "CHECKPOINT":
            return _checkpoint(output, "tokens", info)
        for deployment in DEPLOYMENTS:
            study.stage_status(output, f"preflight-{deployment}", "RUNNING")
            state, info = study.preflight_stage(ctx, deployment, retry_failed=retry_failed)
            if state == "CHECKPOINT":
                return _checkpoint(output, f"preflight-{deployment}", info)
        order = ctx.settings["order"]
        for index, cell in enumerate(order, 1):
            study.stage_status(output, cell["cell_id"], "RUNNING", position=index, cells=len(order))
            state, info = study.cell_stage(ctx, cell, retry_failed=retry_failed)
            if state == "CHECKPOINT":
                return _checkpoint(output, cell["cell_id"], info)
            report = info["report"]
            print(f"{cell['cell_id']}: offered {report['offered']}, successful {report['successful']}, "
                  f"rejected {report['rejected']}, within {ctx.p['slo_ms']} ms {report['met_slo'][str(ctx.p['slo_ms'])]}",
                  flush=True)
        study.stage_status(output, "freeze", "RUNNING")
        study.freeze(ctx)
    study.stage_status(output, "evaluate", "RUNNING")
    evaluation.evaluate(ctx)
    evaluation.audit(ctx)
    evaluation.report(ctx)
    study.stage_status(output, "study", "COMPLETE")
    return "COMPLETE"


def capture_embeddings(output):
    """Optional Study 4 embedding supplement, run before the result archive so its record is inside.

    The helper is archive-only and nonblocking (INCOMPLETE exits 0); its large parts stay under
    exports/ and are listed by the transfer manifest. Nothing here changes the serving result.
    """
    if not EMBEDDING_HELPER.is_file():
        return {"status": "NOT_RUN", "reason": "helper absent in this checkout"}
    settings = read_json(study.path(output, "study.json"))
    source = study.normal(settings["source_study"])
    target = study.path(output, EMBEDDING_TARGET)
    if (target / "capture-receipt.json").exists():
        command = [sys.executable, "-I", "-B", str(EMBEDDING_HELPER), "--verify-only", "--output-dir", str(target)]
    else:
        command = [sys.executable, "-I", "-B", str(EMBEDDING_HELPER), "--source-dir", str(source), "--output-dir", str(target)]
    try:
        run = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=EMBEDDING_SECONDS)
        lines = [line for line in run.stdout.splitlines() if line.strip().startswith("{")]
        result = json.loads(lines[-1]) if lines else {}
        status = {"returncode": run.returncode, "helper_result": result, "status": result.get("status", "UNKNOWN"),
                  "stderr_tail": run.stderr[-4000:]}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        status = {"returncode": None, "status": "NOT_RECORDED", "error": f"{type(exc).__name__}: {exc}"}
    files = {}
    if target.is_dir():
        for item in sorted(target.iterdir()):
            if item.is_file() and not item.is_symlink():
                files[item.name] = {"sha256": file_hash(item), "bytes": item.stat().st_size}
    record = {"at": utc_now(), "helper": str(EMBEDDING_HELPER.relative_to(PROJECT)), "helper_sha256": file_hash(EMBEDDING_HELPER),
              "target": EMBEDDING_TARGET, "command": command, **status, "files": files, "full_refit_performed": False,
              "scope": "Optional archive of accepted Study 4 embedding bytes; the serving result does not depend on it"}
    study.path(output, "supplements").mkdir(exist_ok=True)
    fixed_write(study.path(output, f"supplements/embedding-capture-{time.time_ns()}.json"), record)
    return record


def _listing(path):
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}


def transfer_manifest(output, bundle, supplement):
    """One file naming everything to download: the result archive (or its parts) and the supplement."""
    bundle = Path(bundle)
    main = [_listing(bundle)]
    if bundle.name.endswith(".parts.json"):
        main += [_listing(bundle.parent / part["name"]) for part in read_json(bundle)["parts"]]
    target = study.path(output, EMBEDDING_TARGET)
    extra = [_listing(item) for item in sorted(target.iterdir()) if item.is_file() and not item.is_symlink()] if target.is_dir() else []
    value = {"schema_version": 1, "created_at": utc_now(), "exports_directory": str(study.path(output, "exports")),
             "main_result": {"entry": bundle.name, "files": main,
                             "scope": "Complete serving evidence and analysis; sufficient for the offline audit by itself"},
             "optional_supplement": {"directory": EMBEDDING_TARGET, "status": supplement.get("status"), "files": extra,
                                     "scope": "Study 4 embedding bytes only; a full text-head refit has NOT been performed"}}
    path = study.path(output, f"exports/TRANSFER-{time.time_ns()}.json")
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
    return path
