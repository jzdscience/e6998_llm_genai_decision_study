"""Per-cell configuration, the bound V1 client, row verification and the child-process tasks.

Formal predictions go through V1's ``ModelClient`` and ``predict_examples`` into a
V1 ``PredictionStore`` under ``cells/<cell>/``, so every cell has its own store,
journal and configuration even when two cells share a backend model.

CLINC inputs carry no ``input_hash``. Every new row instead binds
``example_sha256`` (canonical fingerprint of the saved input row) and
``request_sha256`` (fingerprint of the exact request body sent). The prepared
input files are canonical reserializations: original values and task text are
unchanged, and ``data/source_lines.json`` keeps each row's original line hash.
"""
import math
import time

from ..adapters import ModelClient, decision_request, parse_response, qwen_request
from ..common import StudyError, fingerprint, read_json, utc_now, write_json
from ..runner import _probability_comparison, predict_examples
from ..storage import PredictionStore, load_predictions
from . import data
from .identity import (INTERFACE, PROTOCOL_ID, TOKEN_RESERVE, VLLM_ROLES, MAX_NUM_SEQS, dev_set, protocol_hash, safe,
                       split_cell)


def request_body(example, model):
    """The exact body ModelClient sends; built from state, instructions, criteria and labels only."""
    return qwen_request(example, "qwen") if model == "qwen" else decision_request(example, model)


class MatrixClient(ModelClient):
    """V1 client; each row also binds its cell, deployment, saved input and exact request."""

    def __init__(self, model, endpoint, *, cell_id, deployment_id, purpose):
        super().__init__(model, endpoint)
        self.cell_id, self.deployment_id, self.purpose = cell_id, deployment_id, purpose

    def predict(self, example):
        row = super().predict(example)
        return {**row, "cell": self.cell_id, "deployment_id": self.deployment_id, "purpose": self.purpose,
                "example_sha256": fingerprint(example), "request_sha256": fingerprint(request_body(example, self.model))}


class MarginReached(StudyError):
    """The allocation stop margin arrived before the next request; resumable, never an error row."""


class MarginBudget:
    """Stand-in for V1's Budget in ``predict_examples``: refuses to start a request near the deadline."""

    def __init__(self, until_epoch):
        if isinstance(until_epoch, bool) or not isinstance(until_epoch, (int, float)) or not math.isfinite(until_epoch):
            raise StudyError("A finite stop time is required")
        self.until, self.projection = float(until_epoch), None

    def check(self):
        if time.time() >= self.until:
            raise MarginReached("Allocation stop margin reached before the next request; resume in a new allocation")

    def estimate_remaining(self, seconds_per_input, remaining):
        self.projection = {"remaining_inputs": remaining, "seconds_per_input": seconds_per_input,
                           "projected_remaining_seconds": seconds_per_input * remaining,
                           "seconds_before_stop_margin": max(0.0, self.until - time.time())}
        return self.projection


def cell_dir(output, cell_id):
    split_cell(cell_id)
    return safe(output, f"cells/{cell_id}")


def cell_config(output, cell_id, settings_value, runtime_value):
    """Everything that determines a cell's predictions; the PredictionStore refuses any change."""
    role, dataset = split_cell(cell_id)
    model = INTERFACE[role]
    examples = data.cell_examples(output, cell_id)
    dev = data.dev_examples(output, dev_set(dataset))
    return {"protocol": PROTOCOL_ID, "protocol_hash": protocol_hash(), "cell_id": cell_id, "role": role,
            "dataset": dataset, "model": model, "runtime": runtime_value,
            "engine": ({"max_num_seqs": MAX_NUM_SEQS, "launcher": "decision_study.upgrade_study.launcher.managed_service"}
                       if role in VLLM_ROLES else {"launcher": "decision_study.runtime.managed_services"}),
            "source_commit": settings_value["source"]["commit"], "sources": settings_value["source"]["files"],
            "manifest_hash": settings_value["manifest_hash"], "expected_predictions": len(examples),
            "inputs_hash": fingerprint(examples),
            "request_hash": fingerprint([{"id": e["sample_id"], "request": request_body(e, model)} for e in examples]),
            "development_set": dev_set(dataset), "development_hash": fingerprint(dev),
            "token_reserve": TOKEN_RESERVE[model], "max_attempts": 2, "timeout_seconds": 120,
            "retry_delay_seconds": 1, "prediction_concurrency": 1}


def check_rows(examples, rows, *, cell_id, model, purpose, config_hash=None, revision=None, deployments=None):
    """Exact IDs and bindings; a success must re-parse from its raw response, an error must carry no label."""
    expected = {e["sample_id"]: e for e in examples}
    by_id = {}
    for row in rows:
        if row.get("sample_id") in by_id:
            raise StudyError(f"Duplicate prediction for {row.get('sample_id')}")
        by_id[row.get("sample_id")] = row
    if set(by_id) != set(expected):
        missing, extra = len(set(expected) - set(by_id)), len(set(by_id) - set(expected))
        raise StudyError(f"Missing ({missing}) or unexpected ({extra}) prediction IDs")
    request_ids = set()
    for sid, row in by_id.items():
        e = expected[sid]
        if (row.get("model") != model or row.get("cell") != cell_id or row.get("purpose") != purpose
                or row.get("task") != e["task"] or row.get("split") != e["split"]):
            raise StudyError(f"Prediction identity mismatch for {sid}")
        if row.get("example_sha256") != fingerprint(e) or row.get("request_sha256") != fingerprint(request_body(e, model)):
            raise StudyError(f"Prediction is not bound to the saved input and exact request: {sid}")
        if config_hash is not None and row.get("config_hash") != config_hash:
            raise StudyError(f"Prediction configuration mismatch for {sid}")
        if revision is not None and row.get("model_revision") != revision:
            raise StudyError(f"Prediction checkpoint mismatch for {sid}")
        if deployments is not None and row.get("deployment_id") not in deployments:
            raise StudyError(f"Prediction from an unrecorded deployment: {sid}")
        attempts = row.get("attempts", [])
        if not isinstance(attempts, list) or len(attempts) not in (1, 2) or attempts[-1].get("status") != row.get("status"):
            raise StudyError(f"Missing or inconsistent request attempts for {sid}")
        for number, attempt in enumerate(attempts, 1):
            rid = attempt.get("request_id")
            if not isinstance(rid, str) or not rid or rid in request_ids:
                raise StudyError("Missing or duplicate request ID")
            request_ids.add(rid)
            spent = attempt.get("latency_ms")
            final = number == len(attempts)
            if (attempt.get("attempt") != number or attempt.get("status") not in {"ok", "error"}
                    or (not final and attempt["status"] != "error")
                    or (attempt["status"] == "error" and not isinstance(attempt.get("error"), dict))
                    or isinstance(spent, bool) or not isinstance(spent, (int, float)) or not math.isfinite(spent) or spent < 0):
                raise StudyError(f"Malformed request attempt record for {sid}")
        latency = row.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise StudyError(f"Invalid latency for {sid}")
        if row.get("status") == "ok":
            try:
                parsed = parse_response(row.get("raw_response"), e, model)
            except StudyError as exc:
                raise StudyError(f"A stored success does not parse from its raw response: {sid}") from exc
            if parsed != (row.get("prediction"), row.get("probabilities")):
                raise StudyError(f"Stored prediction differs from its raw response: {sid}")
        elif row.get("status") == "error":
            if row.get("prediction") is not None or row.get("probabilities") is not None or not row.get("error"):
                raise StudyError(f"An error row carries a prediction or lacks its error: {sid}")
        else:
            raise StudyError(f"Unknown terminal status for {sid}")
    return by_id


def serial_check(output, folder, plan):
    """Preflight child: send each listed input once; keep every raw response."""
    role, model = plan["role"], INTERFACE[plan["role"]]
    sets = {"development": lambda name: data.dev_examples(output, name),
            "longest": lambda name: data.formal_examples(output, name)}
    pool = {e["sample_id"]: e for e in sets[plan["purpose"]](plan["set"])}
    examples = [pool[sid] for sid in plan["sample_ids"]]
    client = MatrixClient(model, plan["endpoint"], cell_id=plan["label"], deployment_id=plan["deployment_id"],
                          purpose="preflight_" + plan["purpose"])
    budget = MarginBudget(plan["until_epoch"])
    rows = []
    for example in examples:
        budget.check()
        rows.append(client.predict(example))
    write_json(folder / "rows.json", rows)
    failures = sum(r["status"] != "ok" for r in rows)
    write_json(folder / "complete.json", {"plan_hash": fingerprint(plan), "count": len(rows), "failures": failures,
                                          "finished_at": utc_now()})
    return failures


def smoke(output, folder, plan, client, budget):
    dataset = split_cell(plan["cell_id"])[1]
    dev = data.dev_examples(output, dev_set(dataset))
    reference, current = [], []
    for target in (reference, current):
        for example in dev:
            budget.check()
            target.append(client.predict(example))
    value = {"cell_id": plan["cell_id"], "deployment_id": plan["deployment_id"], "config_hash": plan["config_hash"],
             "created_at": utc_now(), "reference": reference, "current": current,
             "scope": "Two serial development passes in the formal deployment; repeatability only, no labels read"}
    try:
        value["checks"] = _probability_comparison(dev, reference, current)
        value["passed"] = all(c["passed"] for c in value["checks"].values())
    except StudyError as exc:
        value.update(checks=None, passed=False, error=str(exc))
    write_json(folder / "smoke.json", value)
    if not value["passed"]:
        raise StudyError(f"Development smoke failed for {plan['cell_id']}: {value.get('error') or value['checks']}")
    return value


def collect(output, folder, plan, config):
    """Formal child: smoke, then resume the cell's store until every input has a terminal row."""
    if fingerprint(config) != plan["config_hash"]:
        raise StudyError("Cell configuration differs from the parent's plan")
    cell_id = plan["cell_id"]
    model = config["model"]
    examples = data.cell_examples(output, cell_id)
    budget = MarginBudget(plan["until_epoch"])
    client = MatrixClient(model, plan["endpoint"], cell_id=cell_id, deployment_id=plan["deployment_id"],
                          purpose="formal")
    smoke_client = MatrixClient(model, plan["endpoint"], cell_id=cell_id, deployment_id=plan["deployment_id"],
                                purpose="smoke")
    smoke(output, folder, plan, smoke_client, budget)
    directory = cell_dir(output, cell_id)
    store = PredictionStore(directory, model, config, spool_dir=directory / "spool")
    if not set(store.rows) <= {e["sample_id"] for e in examples}:
        raise StudyError("Unexpected persisted prediction IDs")
    before = len(store.rows)
    try:
        sampling = predict_examples(examples, client, store, budget=budget)
    except MarginReached as exc:
        write_json(folder / "partial.json", {"at": utc_now(), "persisted": len(store.rows), "before": before,
                                             "expected": len(examples), "reason": str(exc),
                                             "projection": budget.projection})
        raise
    write_json(folder / "complete.json", {"plan_hash": fingerprint(plan), "cell_id": cell_id,
                                          "deployment_id": plan["deployment_id"], "persisted": len(store.rows),
                                          "before": before, "expected": len(examples),
                                          "complete": len(store.rows) == len(examples), "sampling": sampling,
                                          "projection": budget.projection, "finished_at": utc_now()})
    return len(store.rows) == len(examples)


def stored_rows(output, cell_id, model):
    return load_predictions(cell_dir(output, cell_id), model)


def read_plan(folder):
    return read_json(folder / "plan.json")
