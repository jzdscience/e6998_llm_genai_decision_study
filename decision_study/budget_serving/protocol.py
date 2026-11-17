"""The fixed anli-budget-serving-v1 design and the exact source identity that executes it.

``load()`` accepts the machine-readable protocol only when every value this
implementation depends on equals the frozen design, and the separately maintained
runtime constants agree with it. A changed protocol therefore needs a new revision
and new code, never a silently adapted run.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import re
import subprocess

from ..common import StudyError, file_hash, read_json
from . import runtime

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/budget_serving_protocol.json"
PROTOCOL_ID = "anli-budget-serving-v1"
REVISION = 1
CONDITIONS = ("direct4b", "direct9b", "threshold", "budget", "random")
ROLES = ("startlux4b", "startlux9b")
FRONT, BACK = ROLES
PAIR_CONDITIONS = ("threshold", "budget", "random")
CAPPED = ("budget", "random")
SERVICES = {"direct4b": (FRONT,), "direct9b": (BACK,), "threshold": ROLES, "budget": ROLES, "random": ROLES}
DEPLOYMENTS = ("direct4b", "direct9b", "pair")
DEPLOYMENT_OF = {"direct4b": "direct4b", "direct9b": "direct9b", "threshold": "pair", "budget": "pair", "random": "pair"}
SMOKES = {"direct4b": ("direct4b",), "direct9b": ("direct9b",), "pair": PAIR_CONDITIONS}
# runtime.deployment() takes a condition; every pair condition maps to the same pair controls.
RUNTIME_CONDITION = {"direct4b": "direct4b", "direct9b": "direct9b", "pair": "threshold"}
SHARES = {"direct4b": {FRONT: 0.9}, "direct9b": {BACK: 0.9}, "pair": {FRONT: 0.4, BACK: 0.5}}
# Each derived runtime control has its own effective-configuration hash, so the raw
# tokenizer report is produced and bound once per control (direct4b, direct9b, pair4b, pair9b).
TOKEN_TARGETS = (("direct4b", FRONT), ("direct9b", BACK), ("threshold", FRONT), ("threshold", BACK))
LABELS = ("entailment", "neutral", "contradiction")
SOURCE_KEYS = ("data/examples.jsonl", "data/manifest.json", "fitting/models.json", "fitting/complete.json",
               "fitting/scores-eligible_dev.json", "freeze.json", "sealed/evaluation_gold.json", "evaluation/results.json")
# Watchdog constants that the protocol does not fix numerically (see RUN_BUDGET_SERVING_ZH.md).
TOKEN_REPORT_SECONDS = 1200   # ceiling for one raw tokenizer report (1,827 inputs, CPU tokenizer)
FLAT = {"protocol": PROTOCOL_ID, "revision": REVISION, "task": "anli", "conditions": list(CONDITIONS),
        "condition_services": {c: list(SERVICES[c]) for c in CONDITIONS},
        "memory_by_condition": {c: dict(SHARES[DEPLOYMENT_OF[c]]) for c in CONDITIONS},
        "rates_per_second": [10, 20, 40, 80], "duration_seconds": 120, "repeats": 3, "cell_count": 60,
        "workers": 16, "max_inflight": 64, "slo_ms": 500, "secondary_slo_ms": 1000,
        "request_timeout_seconds": 30, "drain_seconds": 60, "watchdog_grace_seconds": 5,
        "startup_timeout_seconds": 1200, "cleanup_margin_seconds": 180, "max_attempts_per_model_call": 1}


def require(condition, message):
    if not condition:
        raise StudyError(message)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate(value):
    """Fail closed unless the protocol is exactly the design implemented here."""
    require(isinstance(value, dict), "Budget-serving protocol must be a JSON object")
    for key, expected in FLAT.items():
        require(value.get(key) == expected, f"Protocol {key} differs from the implemented fixed design")
    require(_integer(value.get("seed")), "Protocol seed must be an integer")
    engine = value.get("engine", {})
    require(engine.get("max_num_seqs") == runtime.MAX_NUM_SEQS == 64 and engine.get("max_model_len") == runtime.MAX_MODEL_LEN == 8192
            and engine.get("precision") == "bfloat16" and engine.get("prefix_cache") is False and engine.get("truncate") is False,
            "Protocol engine settings differ from the fixed runtime")
    models = value.get("models", {})
    require(set(models) == set(ROLES), "Protocol model roles differ")
    for role in ROLES:
        model = models[role]
        require(model.get("interface") == "startlux" and model.get("revision") == runtime.REVISIONS[role]
                and model.get("repo") == runtime.REPOS[role]
                and (model.get("port"), model.get("engine_port")) == tuple(runtime.PORTS[role]),
                f"Protocol {role} identity differs from the runtime controls")
    layout = {key: share for key, (_, share) in runtime.CONTROL_LAYOUT.items()}
    require(layout == {"direct4b": 0.9, "direct9b": 0.9, "pair4b": 0.4, "pair9b": 0.5},
            "Runtime control memory shares differ from the protocol")
    require(runtime.STARTUP_SECONDS == value["startup_timeout_seconds"] and runtime.CLEANUP_MARGIN == value["cleanup_margin_seconds"],
            "Runtime startup/cleanup bounds differ from the protocol")
    require(tuple(runtime.CONDITIONS) == CONDITIONS, "Runtime condition names differ")
    budget = value.get("budget", {})
    require(budget.get("numerator") == 1 and budget.get("denominator") == 10 and budget.get("initial_reservations") == 0
            and budget.get("random_probability") == 0.1 and budget.get("applies_to") == list(CAPPED),
            "Protocol budget rule differs from the implemented causal cap")
    warmup = value.get("warmup", {})
    require(warmup.get("count") == 30 and warmup.get("budget_carryover") is False and warmup.get("formal_evidence") is False,
            "Protocol warmup differs")
    pre = value.get("preflight", {})
    concurrent, smoke = pre.get("concurrent", {}), pre.get("policy_smoke", {})
    require(pre.get("deployments") == list(DEPLOYMENTS) and pre.get("serial_count") == 90
            and (concurrent.get("rate_per_second"), concurrent.get("duration_seconds"), concurrent.get("workers"),
                 concurrent.get("max_inflight")) == (10, 30, 16, 64)
            and smoke.get("conditions") == list(CONDITIONS)
            and (smoke.get("rate_per_second"), smoke.get("duration_seconds")) == (10, 10),
            "Protocol preflight differs")
    policy = value.get("policy", {})
    require(policy.get("source_file") == "fitting/models.json" and policy.get("cell") == "startlux9b/n18000/front"
            and policy.get("selected_index") == 2 and policy.get("l2") == 0.01
            and policy.get("threshold") == 0.027109315183045934 and policy.get("comparison") == "gain >= threshold",
            "Protocol frozen policy differs")
    regression = policy.get("offline_regression", {})
    require(all(_integer(regression.get(k)) for k in ("selection_deferred", "selection_count",
                                                      "eligible_dev_threshold_deferred", "eligible_dev_count")),
            "Protocol offline regression values are missing")
    workload = value.get("workload", {})
    require(workload.get("source_file") == "data/examples.jsonl" and workload.get("split") == "eligible_dev"
            and workload.get("count") == 1737 and workload.get("premise_groups") == 1540 and workload.get("labels") == list(LABELS),
            "Protocol workload differs")
    files = value.get("source_files", {})
    require(set(files) == set(SOURCE_KEYS) and all(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v) for v in files.values()),
            "Protocol pinned source files differ")
    gold = value.get("gold", {})
    require(gold.get("source_file") == "sealed/evaluation_gold.json" and gold.get("sha256") == files["sealed/evaluation_gold.json"]
            and gold.get("source_count") == 3537, "Protocol gold binding differs")
    development = value.get("development", {})
    require(development.get("source_file") == "results/solo-v3/data/examples.jsonl"
            and development.get("function") == "decision_study.learnability_profile.study.inputs"
            and development.get("count") == 90 and isinstance(development.get("source_sha256"), str)
            and isinstance(development.get("rows_fingerprint"), str), "Protocol development inputs differ")
    study = value.get("source_study", {})
    require(study.get("protocol") == "anli-learnability-full-v1" and study.get("default_directory") == "results/aux-anli-learnability-v1-r2",
            "Protocol source study differs")
    require(value.get("automation", {}).get("default_output") == "results/aux-anli-budget-serving-v1",
            "Protocol default output differs")
    return value


def load():
    return validate(read_json(CONFIG))


def windows(p, kind):
    """(duration, drain) of every open-loop window in one supervised attempt."""
    pre = p["preflight"]
    if kind == "cell":
        return [(p["duration_seconds"], p["drain_seconds"])]
    if kind in DEPLOYMENTS:
        return ([(pre["concurrent"]["duration_seconds"], p["drain_seconds"])]
                + [(pre["policy_smoke"]["duration_seconds"], p["drain_seconds"])] * len(SMOKES[kind]))
    raise StudyError(f"Unknown attempt kind: {kind}")


def watchdog_seconds(p, kind, deployment):
    """Protocol ceiling: roles*startup + serial*timeout + windows/drains + grace + cleanup reserve."""
    if kind == "tokens":
        return len(TOKEN_TARGETS) * TOKEN_REPORT_SECONDS + p["watchdog_grace_seconds"] + p["cleanup_margin_seconds"]
    roles = len(SHARES[deployment])
    serial = p["warmup"]["count"] if kind == "cell" else p["preflight"]["serial_count"]
    value = (roles * p["startup_timeout_seconds"] + serial * p["request_timeout_seconds"]
             + sum(a + b for a, b in windows(p, "cell" if kind == "cell" else deployment))
             + p["watchdog_grace_seconds"] + p["cleanup_margin_seconds"])
    require(math.isfinite(value) and value > 0, "Invalid watchdog ceiling")
    return value


# ---------------------------------------------------------------------------------------
# Source identity: this package plus its static import closure inside the checkout, the
# helper scripts that the frozen runtime executes, the wrapper, and every config read.
EXTRA_FILES = ("scripts/run_budget_serving.sh", "scripts/model_probe.py", "scripts/cuda_toolchain_preflight.py",
               "configs/budget_serving_protocol.json", "configs/runtime_defaults.json",
               "configs/cuda_build_toolkit_13_0_flashinfer.txt")


def _module_file(name):
    base = PROJECT.joinpath(*name.split("."))
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def _imported(path, module):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                require(node.level - 1 < len(parts), f"Relative import escapes the package: {path}")
                parts = parts[:len(parts) - (node.level - 1)]
                prefix = ".".join(parts + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            names.append(prefix)
            names.extend(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def dependency_files():
    package = PROJECT / "decision_study/budget_serving"
    queue = ["decision_study.budget_serving", *(f"decision_study.budget_serving.{p.stem}" for p in sorted(package.glob("*.py")))]
    seen = {}
    while queue:
        name = queue.pop()
        if name in seen or not (name.split(".")[0] in {"decision_study", "scripts"}):
            continue
        path = _module_file(name)
        if path is None:
            continue
        seen[name] = path
        parts = name.split(".")
        queue.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        queue.extend(_imported(path, name))
    files = {p for p in seen.values()}
    files.update(PROJECT / name for name in EXTRA_FILES)
    # Frozen helpers read configuration dynamically (e.g. size_control.identity.protocol() inside
    # _verify_model_metadata); static imports cannot see that, so every config file is bound.
    files.update(p for p in (PROJECT / "configs").iterdir() if p.is_file() and p.suffix in {".json", ".txt"})
    for path in files:
        require(path.is_file() and not path.is_symlink(), f"Missing or symlinked source dependency: {path}")
    return sorted(files)


def source_identity():
    """File hashes are authoritative; commit/dirty state is truthful metadata only."""
    files = {str(p.relative_to(PROJECT)): file_hash(p) for p in dependency_files()}

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"files": files, "git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
            "note": "Exact content hashes of the executing source closure; commit fields are metadata, not a clean-release claim"}
