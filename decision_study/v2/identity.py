"""Evidence identities for the new protocol, independent of v1 run configuration."""
from pathlib import Path

from ..common import StudyError, file_hash, fingerprint, read_json

CORE_HASH = "9f25b6350da44fb22cb5f8a644e1cb68b707a82272a024c931701f84c701551f"
ANALYSIS_HASH = "4f011a1eac06f74553b4cbe68bf42a10dd5719950ba6603b005dff7e3789248e"
CORE_FILES = ("adapters.py", "runner.py", "runtime.py", "storage.py", "common.py")
PROJECT = Path(__file__).resolve().parents[2]


def verify_frozen_source():
    source = PROJECT / "decision_study"
    actual = fingerprint({name: file_hash(source / name) for name in CORE_FILES})
    if actual != CORE_HASH or file_hash(source / "analysis.py") != ANALYSIS_HASH:
        raise StudyError("Frozen v1 source changed; restore it before reusing the v1 runtime/evidence")
    return {"controller_source_hash": actual, "analysis_source_hash": ANALYSIS_HASH}


def source_identity():
    verify_frozen_source()
    names = ("identity.py", "data_v2.py", "runner_v2.py", "policies.py", "study.py", "openloop.py")
    return {"v1": verify_frozen_source(), "v2": {name: file_hash(Path(__file__).parent / name) for name in names}}


def protocol():
    return read_json(PROJECT / "configs/v2_protocol.json")


def job_root(output_dir, job):
    if job not in {"qwen_fit", "holdout_startlux", "holdout_qwen", "holdout_kev"}:
        raise StudyError(f"Unknown v2 prediction job: {job}")
    return Path(output_dir) / "predictions-jobs" / job
