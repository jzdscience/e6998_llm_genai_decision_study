"""Fixed protocol, source, input, policy and history identities.

Nothing here opens the sealed ANLI test gold. Historical predictions are read
through the V3 and size-control verifiers so that their receipts are checked.
"""
import copy
from pathlib import Path
import subprocess

from ..common import StudyError, file_hash, fingerprint, read_json, read_jsonl
from ..service_study.identity import fixed_write, safe  # noqa: F401  (re-exported write-once helpers)

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL = PROJECT / "configs/upgrade_study_protocol.json"
PROTOCOL_ID = "anli-upgrade-service-v1"
CONDITIONS = ("startlux4b", "startlux9b", "cascade")
ROLES = ("startlux4b", "startlux9b", "qwen9b")
INTERFACE = {"startlux4b": "startlux", "startlux9b": "startlux", "qwen9b": "qwen"}
SERVICES = {"startlux4b": ("startlux4b",), "startlux9b": ("startlux9b",), "cascade": ("startlux4b", "qwen9b")}
V3_EXAMPLES_SHA256 = "a470ad0143224dd87906059651efc74b1c8342445dcfbffcea11768f37618544"
POLICY_FREEZE_SHA256 = "28efefe7814ef234e5ea96503386b577b1db34e6acafa90b57f421e5f39acf50"
GOLD_SHA256 = "ed3525b721ed564a4cdf733e94ac381cfebb4ba06eb4a3a6844a1a0b37824f14"
POLICY_THRESHOLD = 0.272907934835248
HISTORICAL_DEFERRALS = {"class_conditional": 766, "confidence": 743}
WORKLOAD_COUNT, WARMUP_COUNT = 3200, 90
MAX_NUM_SEQS = 64  # vLLM sequence limit for every engine (protocol revision 4)
LABELS = ("entailment", "neutral", "contradiction")


def protocol():
    value = read_json(PROTOCOL)
    fixed = {"protocol": PROTOCOL_ID, "conditions": list(CONDITIONS), "rates_per_second": [10, 20, 40, 80],
             "duration_seconds": 120, "repeats": 3, "workers": 16, "max_inflight": 64, "slo_ms": 500,
             "secondary_slo_ms": 1000, "request_timeout_seconds": 30, "drain_seconds": 60,
             "max_attempts_per_model_call": 1}
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise StudyError(f"Upgrade-study protocol differs from this release: {key}")
    if {k: list(v) for k, v in value.get("condition_services", {}).items()} != {k: list(v) for k, v in SERVICES.items()}:
        raise StudyError("Upgrade-study condition services differ from this release")
    policy = value.get("cascade_policy", {})
    if (policy.get("policy_freeze_sha256") != POLICY_FREEZE_SHA256 or policy.get("threshold") != POLICY_THRESHOLD
            or policy.get("path") != "policies.qwen9b.class_conditional"):
        raise StudyError("Upgrade-study cascade policy binding differs")
    if value.get("workload", {}).get("examples_sha256") != V3_EXAMPLES_SHA256 or value.get("gold", {}).get("sha256") != GOLD_SHA256:
        raise StudyError("Upgrade-study input or gold identity differs")
    engine = value.get("engine", {})
    if engine.get("max_num_seqs") != MAX_NUM_SEQS or engine.get("applies_to") != list(ROLES):
        raise StudyError("Upgrade-study engine sequence limit differs from this release")
    for role in ROLES:
        fraction = value["models"][role]["gpu_memory_utilization"]
        if isinstance(fraction, bool) or not 0.1 <= fraction <= 0.9:
            raise StudyError(f"Invalid GPU memory fraction for {role}")
    return value


def source_identity(require_clean=True):
    git = lambda *args: subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()
    if require_clean and git("status", "--porcelain=v1", "--untracked-files=all"):
        raise StudyError("Run from a clean committed source checkout")
    paths = [*sorted((PROJECT / "decision_study/upgrade_study").glob("*.py")), PROTOCOL]
    paths += [PROJECT / name for name in (
        "configs/runtime_defaults.json", "configs/cuda_build_toolkit_13_0_flashinfer.txt",
        "configs/size_control_protocol.json", "configs/v3_protocol.json",
        "decision_study/common.py", "decision_study/adapters.py", "decision_study/runtime.py",
        "decision_study/budget.py", "decision_study/runtime_compat.py",
        "decision_study/v3/runtime.py", "decision_study/v3/policies.py", "decision_study/v3/study.py",
        "decision_study/v3/data.py", "decision_study/v3/identity.py", "decision_study/v3/cli.py",
        "decision_study/size_control/study.py", "decision_study/size_control/runtime.py",
        "decision_study/size_control/identity.py", "decision_study/size_control/data.py",
        "decision_study/service_study/measurement.py", "decision_study/service_study/resources.py",
        "decision_study/service_study/identity.py",
        "scripts/model_probe.py", "scripts/prepare_cuda_toolkit.py", "scripts/cuda_toolchain_preflight.py",
        "scripts/run_upgrade_study.sh")]
    return {"commit": git("rev-parse", "HEAD"), "files": {str(p.relative_to(PROJECT)): file_hash(p) for p in paths}}


def load_inputs(v3_dir):
    path = safe(v3_dir, "data/examples.jsonl")
    if file_hash(path) != V3_EXAMPLES_SHA256:
        raise StudyError("V3 ANLI examples differ from the pinned file")
    rows = read_jsonl(path)
    workload = sorted((r for r in rows if r.get("task") == "anli" and r.get("split") == "holdout"), key=lambda r: r["sample_id"])
    warmup = sorted((r for r in rows if r.get("task") == "anli" and r.get("split") == "development"), key=lambda r: r["sample_id"])
    ids = [r["sample_id"] for r in workload + warmup]
    if len(workload) != WORKLOAD_COUNT or len(warmup) != WARMUP_COUNT or len(set(ids)) != len(ids):
        raise StudyError("Unexpected ANLI workload or warmup membership")
    if any(tuple(r.get("labels", ())) != LABELS for r in workload + warmup):
        raise StudyError("Unexpected ANLI label schema")
    return workload, warmup


def load_policy(v3_dir):
    """Return the complete frozen class_conditional policy object, unmodified."""
    path = safe(v3_dir, "policy_freeze.json")
    if file_hash(path) != POLICY_FREEZE_SHA256:
        raise StudyError("V3 policy freeze differs from the pinned file")
    freeze = read_json(path)
    policy = copy.deepcopy(freeze["policies"]["qwen9b"]["class_conditional"])
    decision = policy.get("decision", {})
    if (policy.get("family") != "class_conditional" or policy.get("fallback_model") != "qwen9b"
            or policy.get("local_model") != "startlux4b" or decision.get("mode") != "threshold"
            or decision.get("threshold") != POLICY_THRESHOLD or decision.get("comparison") != ">="):
        raise StudyError("Frozen V3 class_conditional policy differs from the study binding")
    return policy, {"source": "policy_freeze.json", "sha256": POLICY_FREEZE_SHA256,
                    "freeze_hash": freeze["freeze_hash"], "path": "policies.qwen9b.class_conditional",
                    "policy_fingerprint": fingerprint(policy)}


def load_history(v3_dir, size_control_dir, workload):
    """Verified historical holdout predictions plus the full-pool routing regression check."""
    from ..size_control import study as size_study
    from ..v3 import study as v3_study
    from ..v3.policies import route
    ids = {e["sample_id"] for e in workload}
    sources = {}
    local, local_done = v3_study.verified_job(v3_dir, "startlux4b", "holdout")
    fallback, fallback_done = v3_study.verified_job(v3_dir, "qwen9b", "holdout")
    nine, nine_done = size_study.verified_job(size_control_dir)
    for role, rows, done in (("startlux4b", local, local_done), ("qwen9b", fallback, fallback_done),
                             ("startlux9b", nine, nine_done)):
        if {r["sample_id"] for r in rows} != ids or len(rows) != len(ids):
            raise StudyError(f"Historical {role} predictions do not cover the workload exactly")
        sources[role] = {"predictions_hash": done["predictions_hash"], "count": len(rows),
                         "failures": sum(r["status"] != "ok" for r in rows)}
    freeze = read_json(safe(v3_dir, "policy_freeze.json"))
    by_id = {r["sample_id"]: r for r in local}
    decisions, counts = {}, {}
    for family in ("class_conditional", "confidence"):
        policy = freeze["policies"]["qwen9b"][family]
        chosen = {e["sample_id"]: route(policy, e, {**by_id[e["sample_id"]], "model": "startlux4b"}) for e in workload}
        counts[family] = sum(chosen.values())
        if family == "class_conditional":
            decisions = chosen
    if counts != HISTORICAL_DEFERRALS:
        raise StudyError(f"Historical routing regression failed: {counts}")
    keep = lambda r, probabilities: {"status": r["status"], "prediction": r["prediction"],
                                     **({"probabilities": r["probabilities"]} if probabilities else {})}
    history = {"startlux4b": {r["sample_id"]: {**keep(r, True), "deferred": decisions[r["sample_id"]]} for r in local},
               "startlux9b": {r["sample_id"]: keep(r, True) for r in nine},
               "qwen9b": {r["sample_id"]: keep(r, False) for r in fallback}}
    return history, {"sources": sources, "historical_deferrals": counts,
                     "note": "Full historical pool regression only; live deferral counts are measurements"}
