"""Fixed protocol, cell inventory and source identity for the direct-accuracy matrix.

Nothing here opens a gold file, starts a model or inspects a GPU.
"""
import os
from pathlib import Path
import subprocess

from ..common import StudyError, file_hash, fingerprint, read_json
from ..service_study.identity import fixed_write, safe  # noqa: F401  (write-once helpers, re-exported)

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT / "configs/direct_matrix_protocol.json"
PROTOCOL_ID = "direct-matrix-v1"
OUTPUT_NAME = "aux-direct-matrix-v1"
ROLES = ("startlux9b", "qwen9b", "kev4b")
INTERFACE = {"startlux9b": "startlux", "qwen9b": "qwen", "kev4b": "kev"}
VLLM_ROLES = ("startlux9b", "qwen9b")
DATASETS = ("clinc", "squad1", "squad2", "anli")
DEV_SETS = ("clinc", "squad", "anli")
CELL_ORDER = ("startlux9b-clinc", "startlux9b-squad2", "qwen9b-clinc", "qwen9b-squad2",
              "kev4b-squad2", "kev4b-anli", "startlux9b-squad1", "qwen9b-squad1")
COUNTS = {"clinc": 600, "squad1": 2000, "squad2": 4000, "anli": 3200}
DEV_COUNTS = {"clinc": 64, "squad": 64, "anli": 90}
TOTAL_NEW = 20400
TOKEN_RESERVE = {"kev": 0, "startlux": 1, "qwen": 64}  # scripts/model_probe.py tokens()
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 64
GPU_SHARE = 0.8
MODELS = ("kev4b", "startlux4b", "startlux9b", "qwen4b", "qwen9b")
REFERENCE_CELLS = ("kev4b-clinc", "startlux4b-clinc", "qwen4b-clinc", "kev4b-squad1", "startlux4b-squad1",
                   "qwen4b-squad1", "startlux4b-squad2", "qwen4b-squad2", "startlux4b-anli", "qwen4b-anli",
                   "qwen9b-anli", "startlux9b-anli")
SOURCES = ("v1", "v2", "v3", "size_control")
REVISIONS = {"startlux9b": "18a17f734966b213b2f00fa9c915e8b9af2a9b20",
             "qwen9b": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
             "kev4b": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101"}


def normal(path):
    """Absolute path with ``..`` collapsed lexically, so overlap checks cannot be fooled; symlinks are refused elsewhere."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def split_cell(cell_id):
    role, _, dataset = cell_id.partition("-")
    if dataset not in DATASETS or role not in (*ROLES, *MODELS):
        raise StudyError(f"Unknown matrix cell: {cell_id}")
    return role, dataset


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def dev_set(dataset):
    return {"clinc": "clinc", "squad1": "squad", "squad2": "squad", "anli": "anli"}[dataset]


def protocol():
    """The protocol file, checked against the inventory this release implements."""
    value = read_json(PROTOCOL)
    if value.get("protocol") != PROTOCOL_ID or value.get("schema_version") != 1 or value.get("revision") != 1:
        raise StudyError("Direct-matrix protocol identity differs from this release")
    cells = value.get("cells", [])
    if [c.get("cell_id") for c in cells] != list(CELL_ORDER):
        raise StudyError("Direct-matrix cell order differs from this release")
    for cell in cells:
        role, dataset = split_cell(cell["cell_id"])
        if (cell.get("role") != role or cell.get("dataset") != dataset or cell.get("count") != COUNTS[dataset]
                or role not in ROLES):
            raise StudyError(f"Direct-matrix cell definition differs: {cell['cell_id']}")
    if sum(c["count"] for c in cells) != TOTAL_NEW or value.get("total_new_predictions") != TOTAL_NEW:
        raise StudyError("Direct-matrix prediction total differs from 20,400")
    if set(value.get("datasets", {})) != set(DATASETS):
        raise StudyError("Direct-matrix dataset inventory differs")
    for name, spec in value["datasets"].items():
        if (spec.get("count") != COUNTS[name] or spec.get("source") not in SOURCES
                or spec.get("development") != dev_set(name) or spec.get("cluster") not in {"sample_id", "group_id"}):
            raise StudyError(f"Direct-matrix dataset definition differs: {name}")
    if value["datasets"]["clinc"]["cluster"] != "sample_id" or any(
            value["datasets"][name]["cluster"] != "group_id" for name in ("squad1", "squad2", "anli")):
        raise StudyError("Direct-matrix cluster definitions differ")
    dev = value.get("development", {})
    if any(dev.get(name, {}).get("count") != count for name, count in DEV_COUNTS.items()):
        raise StudyError("Direct-matrix development sets differ")
    roles = value.get("roles", {})
    if set(roles) != set(ROLES):
        raise StudyError("Direct-matrix role inventory differs")
    for role in ROLES:
        spec = roles[role]
        if (spec.get("interface") != INTERFACE[role] or spec.get("revision") != REVISIONS[role]
                or spec.get("token_reserve") != TOKEN_RESERVE[INTERFACE[role]]):
            raise StudyError(f"Direct-matrix role definition differs: {role}")
        if role in VLLM_ROLES and (spec.get("max_num_seqs") != MAX_NUM_SEQS or spec.get("gpu_memory_utilization") != GPU_SHARE
                                   or spec.get("vllm_version") != "0.31.0"):
            raise StudyError(f"Direct-matrix vLLM engine settings differ: {role}")
    if roles["kev4b"].get("profile") != "primary" or roles["kev4b"].get("backend") != "torch":
        raise StudyError("Direct-matrix Kev profile differs from the accepted V1 primary profile")
    engine = value.get("engine", {})
    if engine.get("max_num_seqs") != MAX_NUM_SEQS or engine.get("applies_to") != list(VLLM_ROLES):
        raise StudyError("Direct-matrix engine sequence limit differs")
    refs = value.get("references", {})
    if set(REFERENCE_CELLS) - set(refs) or any(
            not isinstance(refs[c].get("accepted_correct"), int) or not _sha(refs[c].get("compact_predictions_hash"))
            or not _sha(refs[c].get("cell_predictions_hash")) for c in REFERENCE_CELLS):
        raise StudyError("Direct-matrix reference inventory or its pinned hashes differ")
    gold = value.get("gold", {})
    if set(gold) - {"access"} != set(DATASETS) or any(
            not _sha(gold[d].get("sha256")) or not _sha(gold[d].get("subset_fingerprint")) for d in DATASETS):
        raise StudyError("Direct-matrix gold inventory or its pinned hashes differ")
    e = value.get("evaluation", {})
    if e.get("models") != list(MODELS) or e.get("bootstrap_repetitions") != 2000 or e.get("bootstrap_seed") != 6998:
        raise StudyError("Direct-matrix evaluation settings differ")
    c = value.get("collection", {})
    if (c.get("concurrency") != 1 or c.get("max_attempts") != 2 or c.get("timeout_seconds") != 120
            or c.get("retry_delay_seconds") != 1 or c.get("preflight_longest_inputs") != 3):
        raise StudyError("Direct-matrix collection settings differ from the V1 client")
    return value


def cells():
    return [dict(c) for c in protocol()["cells"]]


SOURCE_FILES = (
    "configs/runtime_defaults.json", "configs/cuda_build_toolkit_13_0_flashinfer.txt",
    "configs/size_control_protocol.json", "configs/v2_protocol.json", "configs/v3_protocol.json",
    "decision_study/common.py", "decision_study/adapters.py", "decision_study/runner.py",
    "decision_study/storage.py", "decision_study/runtime.py", "decision_study/budget.py",
    "decision_study/runtime_compat.py", "decision_study/analysis.py",
    "decision_study/v2/identity.py", "decision_study/v2/data_v2.py", "decision_study/v2/study.py",
    "decision_study/v2/policies.py", "decision_study/v2/runner_v2.py", "decision_study/v2/openloop.py",
    "decision_study/v3/runtime.py", "decision_study/v3/study.py", "decision_study/v3/data.py",
    "decision_study/v3/identity.py", "decision_study/v3/policies.py", "decision_study/v3/stats.py",
    "decision_study/v3/cli.py",
    "decision_study/size_control/study.py", "decision_study/size_control/runtime.py",
    "decision_study/size_control/identity.py", "decision_study/size_control/data.py",
    "decision_study/upgrade_study/launcher.py", "decision_study/upgrade_study/runtime.py",
    "decision_study/upgrade_study/identity.py",
    "decision_study/service_study/identity.py",
    "scripts/model_probe.py", "scripts/prepare_cuda_toolkit.py", "scripts/cuda_toolchain_preflight.py",
    "scripts/run_direct_matrix.sh")


def source_files():
    paths = [*sorted((PROJECT / "decision_study/direct_matrix").glob("*.py")), PROTOCOL]
    paths += [PROJECT / name for name in SOURCE_FILES]
    return {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}


def source_identity(require_clean=True):
    """Commit and hashes of every file that shapes requests, runtimes, lifecycle or scoring."""
    git = lambda *args: subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()
    if require_clean and git("status", "--porcelain=v1", "--untracked-files=all"):
        raise StudyError("Run from a clean committed source checkout; preserve local changes elsewhere")
    return {"commit": git("rev-parse", "HEAD"), "files": source_files()}


def protocol_hash():
    return fingerprint(protocol())
