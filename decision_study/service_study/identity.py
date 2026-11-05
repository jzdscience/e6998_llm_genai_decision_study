"""Fixed source/input identity without opening task labels during measurement."""
from pathlib import Path
import subprocess

from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT/"configs/service_study_protocol.json"
INPUTS = {
    "data/manifest.json": "7364419fcf52eaf691a40282c6027eec62a026cef0b41a6c9aa31eb06a593d4b",
    "data/examples.jsonl": "a0b699e5fe48e003ddcbb6cab751b0f088edb0b5d297ee46c250121bfdd10703",
}
GOLD_SHA256 = "bc5ec4891caed470970115a41e2f6debe2a2967b3f5ff3496622a3d29311fc35"


def safe(root, relative):
    root = Path(root).expanduser().absolute()
    if root.is_symlink():
        raise StudyError("Symlinked study root")
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise StudyError("Unsafe study path")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink():
            raise StudyError(f"Symlink in study path: {path}")
    return path


def fixed_write(path, value):
    """Idempotent write-once content, including plans and historical receipts."""
    path = Path(path)
    if path.is_symlink():
        raise StudyError(f"Refusing symlinked evidence: {path}")
    if path.exists():
        if read_json(path) != value:
            raise StudyError(f"Existing evidence differs; preserve and inspect: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value)+b"\n")


def source_identity(require_clean=True):
    git = lambda *args: subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()
    if require_clean and git("status", "--porcelain=v1", "--untracked-files=all"):
        raise StudyError("Run from a clean committed source checkout")
    paths = [*sorted((PROJECT/"decision_study/service_study").glob("*.py")),
             PROTOCOL, PROJECT/"configs/cheap_baseline_requirements.txt",
             PROJECT/"configs/runtime_defaults.json", PROJECT/"configs/cuda_build_toolkit_13_0_flashinfer.txt",
             *sorted((PROJECT/"decision_study/cheap_baselines").glob("*.py"))]
    paths += [PROJECT/name for name in (
        "decision_study/common.py", "decision_study/adapters.py", "decision_study/runtime.py",
        "decision_study/budget.py", "decision_study/v3/runtime.py", "decision_study/baselines/study.py",
        "decision_study/v3/cli.py", "decision_study/v3/identity.py",
        "scripts/model_probe.py", "scripts/prepare_cuda_toolkit.py", "scripts/cuda_toolchain_preflight.py",
        "scripts/run_service_study.sh", "scripts/run_followup_suite.sh")]
    return {"commit": git("rev-parse", "HEAD"), "files": {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}}


def load_inputs(v1_dir):
    hashes = {}
    for name, digest in INPUTS.items():
        actual = file_hash(safe(v1_dir, name))
        if actual != digest:
            raise StudyError(f"V1 input identity differs: {name}")
        hashes[name] = actual
    rows = read_jsonl(safe(v1_dir, "data/examples.jsonl"))
    main = sorted((r for r in rows if r["task"] == "clinc" and r["split"] == "test"), key=lambda e: e["sample_id"])
    warmup = sorted((r for r in rows if r["task"] == "clinc" and r["split"] == "development"), key=lambda e: e["sample_id"])
    if len(main) != 600 or len(warmup) != 64 or len({r["sample_id"] for r in main+warmup}) != 664:
        raise StudyError("Unexpected CLINC workload membership")
    return main, warmup, hashes


def protocol():
    value = read_json(PROTOCOL)
    if (value["protocol"] != "clinc-service-v1" or value["conditions"] != ["minilm400", "minilm1600", "startlux4b", "qwen4b"]
            or value["rates_per_second"] != [5, 25, 100] or value["duration_seconds"] != 180 or value["repeats"] != 5):
        raise StudyError("Protocol identity differs from this fixed release")
    return value
