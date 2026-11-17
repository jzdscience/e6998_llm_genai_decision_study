"""One supervised worker: raw token reports, one preflight deployment, or one formal cell.

The parent supervisor bounds this whole process (imports, tokenization, startup,
warmup, measurement, shutdown). Request failures become trace data; an invariant
break, a failed engineering check or a failed warmup raises, which preserves the
evidence and makes the attempt fail. Gold is never available to this process.
"""
from __future__ import annotations

import os
import time

from ..common import StudyError, canonical, file_hash, fingerprint, read_json, utc_now
from ..service_study.identity import fixed_write
from ..service_study.resources import ResourceMonitor, hardware
from ..upgrade_study import runtime as old
from . import measurement, runtime, serve, study, verify
from .ledger import Ledger
from .protocol import CAPPED, PAIR_CONDITIONS, RUNTIME_CONDITION, SERVICES, SMOKES, TOKEN_TARGETS, require

OPEN_FILES = study.OPEN_FILES


def _deadline():
    value = os.environ.get("DECISION_STUDY_DEADLINE_EPOCH", "")
    try:
        deadline = float(value)
    except ValueError:
        raise StudyError("Worker has no supervisor deadline") from None
    require(deadline > 0, "Worker has no supervisor deadline")
    return deadline


def _complete(folder, plan, names, **extra):
    fixed_write(study.path(folder, "complete.json"), {
        "kind": plan["kind"], "plan_sha256": file_hash(study.path(folder, "plan.json")),
        "files": {name: file_hash(study.path(folder, name)) for name in sorted(names)}, "finished_at": utc_now(), **extra})


def _lines(target, rows):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sync(*handles):
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())


def _gpu_snapshot():
    try:
        return {"status": "ok", "processes": old.gpu_processes()}
    except StudyError as exc:
        return {"status": "unavailable", "error": str(exc)}


def check_runtime_identity(ctx, record, condition):
    """Stop before any request if this deployment is not the token-checked runtime identity."""
    for item in record.get("roles", []):
        if fingerprint(item.get("runtime", {})) != study.runtime_hash(ctx, condition, item["role"]):
            raise StudyError(f"{item['role']} runtime identity differs from the token/preflight identity "
                             "(GPU, driver or environment changed); preserve this output")


def open_stage(ctx, folder, prefix, schedule, make, *, enforced, window, pids, pair):
    p = ctx.p
    trace_path = study.path(folder, prefix + "traces.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("xb") as trace, study.path(folder, prefix + "ledger.jsonl").open("xb") as ledger_file, \
            study.path(folder, prefix + "fallback.jsonl").open("xb") as fallback_file:
        ledger = Ledger(numerator=p["budget"]["numerator"], denominator=p["budget"]["denominator"], enforced=enforced,
                        journal=ledger_file)
        journal = serve.FallbackJournal(fallback_file) if pair else None
        call = make(ledger, journal)
        with ResourceMonitor(pids, study.path(folder, prefix + "resources.jsonl")):
            rows, events, facts = measurement.run_open_loop(
                schedule, call, duration=window, workers=p["workers"], max_inflight=p["max_inflight"],
                drain_seconds=p["drain_seconds"], request_timeout=p["request_timeout_seconds"], ledger=ledger, trace=trace,
                journal=journal)
        _sync(trace, ledger_file, fallback_file)
    _lines(study.path(folder, prefix + "occupancy.jsonl"), events)
    fixed_write(study.path(folder, prefix + "facts.json"), facts)
    return rows, measurement.summarize(rows, duration=window, slo_ms=p["slo_ms"], secondary_slo_ms=p["secondary_slo_ms"])


def _serial(folder, relative, examples, call, p):
    target = study.path(folder, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as trace:
        rows = measurement.run_serial(examples, call, request_timeout=p["request_timeout_seconds"], trace=trace)
        _sync(trace)
    return verify.serial_summary(rows)


def _after_load(folder, endpoints, record):
    # Both roles name their processes startlux-engine/startlux, so facts stay grouped by role.
    logs = {item["role"]: old.engine_log_facts(item.get("processes", [])) for item in record.get("roles", [])}
    fixed_write(study.path(folder, "health.json"), {
        "roles": {role: old.health(endpoint) for role, endpoint in endpoints.items()},
        "engine_log": logs, "gpu_processes": _gpu_snapshot(), "at": utc_now()})


def tokens(ctx, folder, plan):
    receipts = {f"{condition}/{role}": runtime.token_check(ctx.output, condition, role, ctx.token_examples)
                for condition, role in TOKEN_TARGETS}
    fixed_write(study.path(folder, "tokens.json"), {"settings_sha256": ctx.settings_sha256,
                                                    "examples_fingerprint": fingerprint(ctx.token_examples), "receipts": receipts})
    _complete(folder, plan, {"plan.json", "tokens.json"})


def preflight(ctx, folder, plan):
    p, deployment = ctx.p, plan["deployment"]
    condition, pair = RUNTIME_CONDITION[deployment], deployment == "pair"
    mode = "forced_both" if pair else "direct"
    record = {}
    fixed_write(study.path(folder, "hardware.json"), hardware())
    try:
        with runtime.deployment(ctx.output, condition, sink=record) as endpoints:
            require(set(endpoints) == set(SERVICES[condition]), "Deployment endpoints differ from the fixed roles")
            check_runtime_identity(ctx, record, condition)
            fixed_write(study.path(folder, "deployment-open.json"), record)
            pids = [os.getpid(), *[proc["pid"] for proc in record.get("processes", [])]]
            serial = _serial(folder, "serial/rows.jsonl", ctx.development, serve.make_call(
                condition, mode=mode, endpoints=endpoints, examples=ctx.dev_examples,
                label="preflight_serial_forced_both" if pair else None), p)
            require(serial["failures"] == 0, f"Serial development check failed {serial['failures']}/{serial['count']}")
            _, report = open_stage(ctx, folder, "concurrent/", ctx.json("preflight/schedules/concurrent.json"),
                                   lambda ledger, journal: serve.make_call(condition, mode=mode, endpoints=endpoints,
                                                                           examples=ctx.dev_examples, journal=journal,
                                                                           label=study.PREFLIGHT_LABEL if pair else None),
                                   enforced=False, window=p["preflight"]["concurrent"]["duration_seconds"], pids=pids, pair=pair)
            require(report["successful"] == report["offered"], f"Concurrent check failures {report['status_counts']}")
            smoke_schedule = ctx.json("preflight/schedules/smoke.json")
            for smoke in SMOKES[deployment]:
                flags = ctx.json("preflight/flags/smoke.json")["flags"] if smoke == "random" else None

                def make(ledger, journal, smoke=smoke, flags=flags):
                    return serve.make_call(smoke, mode="policy" if pair else "direct", endpoints=endpoints,
                                           examples=ctx.dev_examples, policy=ctx.policy if pair else None, flags=flags,
                                           ledger=ledger if smoke in CAPPED else None, journal=journal)
                _, report = open_stage(ctx, folder, f"smoke-{smoke}/", smoke_schedule, make, enforced=smoke in CAPPED,
                                       window=p["preflight"]["policy_smoke"]["duration_seconds"], pids=pids, pair=pair)
                require(report["successful"] == report["offered"], f"Policy smoke {smoke} failures {report['status_counts']}")
            _after_load(folder, endpoints, record)
    finally:
        if not study.path(folder, "deployment.json").exists():
            fixed_write(study.path(folder, "deployment.json"), record)
    names = {"plan.json", "hardware.json", "deployment-open.json", "deployment.json", "serial/rows.jsonl", "health.json",
             *(f"concurrent/{n}" for n in OPEN_FILES), *(f"smoke-{c}/{n}" for c in SMOKES[deployment] for n in OPEN_FILES)}
    _complete(folder, plan, names)


def cell(ctx, folder, plan):
    p, cell_spec = ctx.p, plan["cell"]
    require(cell_spec in ctx.settings["order"], "Cell is not one of the 60 fixed cells")
    condition = cell_spec["condition"]
    pair = condition in PAIR_CONDITIONS
    schedule = ctx.json(cell_spec["schedule_file"])
    flags = ctx.json(cell_spec["flags_file"])["flags"] if condition == "random" else None
    record = {}
    fixed_write(study.path(folder, "hardware.json"), hardware())
    try:
        with runtime.deployment(ctx.output, condition, sink=record) as endpoints:
            require(set(endpoints) == set(SERVICES[condition]), "Deployment endpoints differ from the fixed roles")
            check_runtime_identity(ctx, record, condition)
            fixed_write(study.path(folder, "deployment-open.json"), record)
            pids = [os.getpid(), *[proc["pid"] for proc in record.get("processes", [])]]
            warm = _serial(folder, "warmup/rows.jsonl", ctx.development[:p["warmup"]["count"]], serve.make_call(
                condition, mode="forced_both" if pair else "direct", endpoints=endpoints, examples=ctx.dev_examples,
                label="warmup_forced_both" if pair else None), p)
            require(warm["failures"] == 0, f"Warmup failed {warm['failures']}/{warm['count']}; no formal measurement on this service")
            needed = p["duration_seconds"] + p["drain_seconds"] + p["watchdog_grace_seconds"] + p["cleanup_margin_seconds"]
            require(_deadline() - time.time() >= needed, "Insufficient worker allowance before the formal window")

            def make(ledger, journal):
                return serve.make_call(condition, mode="policy" if pair else "direct", endpoints=endpoints,
                                       examples=ctx.examples, policy=ctx.policy if pair else None, flags=flags,
                                       ledger=ledger if condition in CAPPED else None, journal=journal)
            _, report = open_stage(ctx, folder, "", schedule, make, enforced=condition in CAPPED,
                                   window=p["duration_seconds"], pids=pids, pair=pair)
            _after_load(folder, endpoints, record)
    finally:
        if not study.path(folder, "deployment.json").exists():
            fixed_write(study.path(folder, "deployment.json"), record)
    _complete(folder, plan, {"plan.json", "hardware.json", "deployment-open.json", "deployment.json", "warmup/rows.jsonl",
                             "health.json", *OPEN_FILES}, report=report)


def main(output, folder):
    output, folder = study.normal(output), study.normal(folder)
    require(os.environ.get("DECISION_BUDGET_WORKER_OUTPUT") == str(output)
            and os.environ.get("DECISION_BUDGET_WORKER_ATTEMPT") == str(folder),
            "Worker must be started by the budget-serving supervisor for this attempt")
    _deadline()
    ctx = study.context(output)
    plan = read_json(study.path(folder, "plan.json"))
    require(plan.get("attempt") == folder.name and output in folder.parents, "Worker plan does not name this attempt")
    require(runtime.session(output)["allocation"] == plan["allocation"], "Worker allocation differs from its plan")
    kind = plan.get("kind")
    expected = study.expected_plan(ctx, kind, deployment=plan.get("deployment"), cell=plan.get("cell"))
    require(all(plan.get(k) == v for k, v in expected.items()), "Worker plan differs from fixed settings and receipts")
    {"tokens": tokens, "preflight": preflight, "cell": cell}[kind](ctx, folder, plan)
