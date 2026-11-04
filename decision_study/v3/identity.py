"""Bind every v3 job to the new protocol and frozen controller prerequisites."""
from pathlib import Path
import json

from ..common import StudyError, file_hash, fingerprint

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT / "configs/v3_protocol.json"
ROLES = ("startlux4b", "qwen4b", "qwen9b")


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def interface(role):
    if role not in ROLES:
        raise StudyError(f"Unknown v3 role: {role}")
    return protocol()["models"][role]["interface"]


def source_identity():
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [PROTOCOL_PATH]
    files += [PROJECT / "decision_study" / name for name in
              ("adapters.py", "runner.py", "runtime.py", "storage.py", "common.py", "budget.py")]
    files += [PROJECT / "scripts" / name for name in
              ("model_probe.py", "run_with_cuda_tools.py")]
    return {str(path.relative_to(PROJECT)): file_hash(path) for path in files}


def verify_release():
    from scripts.verify_v3_release import verify
    return verify(PROJECT)


def job_root(output_dir, role, phase):
    if role not in ROLES or phase not in ("fit", "holdout"):
        raise StudyError("Unknown v3 prediction job")
    return Path(output_dir) / "predictions-jobs" / f"{role}-{phase}"


def freeze_hash(payload):
    return fingerprint({k: v for k, v in payload.items() if k != "freeze_hash"})
