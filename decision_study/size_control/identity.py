"""Source and protocol bindings for an independent retrospective size control."""
from pathlib import Path
import subprocess
from ..common import StudyError, file_hash, read_json

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT / "configs/size_control_protocol.json"
ROLE = "startlux9b"


def protocol():
    return read_json(PROTOCOL_PATH)


def source_identity():
    files = list(Path(__file__).parent.glob("*.py")) + [PROTOCOL_PATH]
    files += list((PROJECT / "decision_study/v3").glob("*.py"))
    files += [PROJECT / "decision_study" / name for name in
              ("adapters.py", "runner.py", "runtime.py", "storage.py", "common.py", "budget.py", "runtime_compat.py")]
    files += [PROJECT / name for name in (
        "scripts/model_probe.py", "scripts/run_with_cuda_tools.py", "scripts/prepare_cuda_toolkit.py",
        "scripts/cuda_toolchain_preflight.py", "scripts/run_size_control.sh",
        "configs/runtime_defaults.json", "configs/cuda_build_toolkit_13_0_flashinfer.txt",
        "configs/v3_protocol.json")]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in sorted(files)}


def checkout_identity():
    result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"],
                            cwd=PROJECT, check=True, text=True, capture_output=True)
    if result.stdout.strip():
        raise StudyError("Use a clean committed checkout for node execution; preserve local changes")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, check=True,
                         text=True, capture_output=True).stdout.strip()
    return {"commit": sha, "clean": True, "source_files": source_identity()}
