"""Controller: prepared identity, supervised attempts, raw-evidence acceptance, freeze and gold gate.

Successful gates continue automatically. Each token, preflight and cell attempt is one
supervised worker in its own immutable folder; an accepted receipt is re-derived from
the raw evidence whenever it is reused, so a rewritten receipt or hash is not enough.
A failed attempt blocks until ``--retry-failed``; an attempt interrupted by an ended
allocation is recovered by terminal-job proof and measured again in a new attempt.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import re
import sys
import time

from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl, utc_now, write_json
from ..service_study.identity import fixed_write
from ..service_study.measurement import make_schedule
from . import inputs, protocol, runtime, verify
from .measurement import summarize
from .protocol import (DEPLOYMENT_OF, DEPLOYMENTS, PROJECT, PROTOCOL_ID, REVISION, RUNTIME_CONDITION, SHARES, SMOKES,
                       TOKEN_TARGETS, require)

OUTPUT_PREFIX = "aux-anli-budget-serving-v1"
HISTORICAL = ("solo-v1", "solo-v3", "aux-size-control-v1-r2", "aux-anli-learnability-v1-r2")
FRESH_ALLOWED = {".active.lock", "diagnostics", "exports"}
ATTEMPT = re.compile(r"attempt-[0-9]{4}")
WORKER_MODULE = "decision_study.budget_serving"
OPEN_FILES = ("traces.jsonl", "ledger.jsonl", "fallback.jsonl", "occupancy.jsonl", "resources.jsonl", "facts.json")
PREFLIGHT_LABEL = "preflight_forced_both"


def normal(value):
    return runtime.normal(value)


def path(root, relative):
    return inputs.safe(root, relative)


def rel(output, target):
    return str(Path(target).relative_to(normal(output)))


def bytehash(value):
    """SHA-256 of the exact bytes fixed_write produces for value."""
    return hashlib.sha256(canonical(value) + b"\n").hexdigest()


# ------------------------------------------------------------------------------- identity
def validate_output(output, runtime_root=None, *, existing=False):
    """Run before any lock, write, diagnostic or export touches the supplied path."""
    output = normal(output)
    path(output, "study.json")
    require(not (output == PROJECT or output in PROJECT.parents or PROJECT in output.parents),
            "Study output must be outside the source checkout")
    if runtime_root is not None:
        root = normal(runtime_root)
        require(output.parent == root / "results" and output.name.startswith(OUTPUT_PREFIX),
                f"Use a new {OUTPUT_PREFIX} directory immediately under the runtime results root")
        for name in HISTORICAL:
            other = root / "results" / name
            require(not (output == other or output in other.parents or other in output.parents),
                    "Study output overlaps a historical result root")
    marker = path(output, "study.json")
    if marker.exists():
        value = read_json(marker).get("protocol", {})
        require(value.get("protocol") == PROTOCOL_ID and value.get("revision") == REVISION,
                "Output belongs to another protocol; preserve it")
    elif existing:
        raise StudyError("Output lacks its budget-serving study identity")
    elif output.exists():
        extra = {p.name for p in output.iterdir()} - FRESH_ALLOWED
        require(not extra, "Output contains unrelated files and no study identity; choose a new output")
    return output


def planned_order(p):
    """Per repeat: seeded rate order; per repeat/rate: seeded order of all five conditions."""
    order = []
    for repeat in range(p["repeats"]):
        rates = list(p["rates_per_second"])
        random.Random(f"{p['seed']}:rates:{repeat}").shuffle(rates)
        for rate in rates:
            conditions = list(p["conditions"])
            random.Random(f"{p['seed']}:conditions:{repeat}:{rate}").shuffle(conditions)
            for condition in conditions:
                order.append({"cell_id": f"r{repeat}-rate{rate:02d}-{condition}", "repeat": repeat, "rate": rate,
                              "condition": condition, "schedule_file": f"schedules/r{repeat}-rate{rate:02d}.json",
                              "flags_file": f"flags/r{repeat}-rate{rate:02d}.json" if condition == "random" else None})
    require(len(order) == p["cell_count"] and len({c["cell_id"] for c in order}) == len(order), "Fixed cell matrix differs")
    return order


def random_flags(p, label, count):
    """Preassigned Bernoulli flags from a stream separate from arrival, sample and order seeds."""
    rng = random.Random(f"{p['seed']}:random-candidates:{label}")
    probability = p["budget"]["random_probability"]
    return {"probability": probability, "seed_label": f"{p['seed']}:random-candidates:{label}",
            "flags": [int(rng.random() < probability) for _ in range(count)],
            "independence": "Seeded per offer before any measurement; no label, score or completion order is used"}


def seed_rules(p):
    return {"trace": f"service_study.measurement.make_schedule(workload, rate, duration, {p['seed']} + 1000*repeat + rate)",
            "preflight_concurrent": p["seed"] + 424242, "preflight_smoke": p["seed"] + 434343,
            "order": f"random.Random('{p['seed']}:rates:<repeat>') and '{p['seed']}:conditions:<repeat>:<rate>'",
            "random_flags": f"random.Random('{p['seed']}:random-candidates:<r<repeat>-rate<rate>|preflight-smoke>')"}


def derived_files(p, workload, development):
    """Schedules and random flags are pure functions of the anchored inputs and the protocol seed."""
    pre = p["preflight"]
    smoke = make_schedule(development, pre["policy_smoke"]["rate_per_second"], pre["policy_smoke"]["duration_seconds"],
                          p["seed"] + 434343)
    files = {"preflight/schedules/concurrent.json": make_schedule(
                 development, pre["concurrent"]["rate_per_second"], pre["concurrent"]["duration_seconds"], p["seed"] + 424242),
             "preflight/schedules/smoke.json": smoke, "preflight/flags/smoke.json": random_flags(p, "preflight-smoke", len(smoke))}
    for repeat in range(p["repeats"]):
        for rate in p["rates_per_second"]:
            schedule = make_schedule(workload, rate, p["duration_seconds"], p["seed"] + 1000 * repeat + rate)
            files[f"schedules/r{repeat}-rate{rate:02d}.json"] = schedule
            files[f"flags/r{repeat}-rate{rate:02d}.json"] = random_flags(p, f"r{repeat}-rate{rate}", len(schedule))
    return files


def build(runtime_root, p):
    """Verify pinned bytes and derive every immutable prepared file in memory."""
    root = normal(runtime_root)
    source = inputs.source_root(root, p)
    hashes = inputs.verify_source_files(source, p)
    workload = inputs.load_workload(source, p)
    development = inputs.load_development(root, p)
    policy = inputs.load_policy(source, p, workload)
    files = {"inputs/workload.json": workload, "inputs/development.json": development, "inputs/policy.json": policy,
             "inputs/source-binding.json": {"runtime_root": str(root), "source_study": str(source), "verified_sha256": hashes,
                                            "development_source_sha256": p["development"]["source_sha256"],
                                            "gold": "bytes hashed only; labels are parsed after the trace freeze"},
             **derived_files(p, workload, development)}
    settings = {"schema_version": 1, "protocol": p, "protocol_sha256": file_hash(protocol.CONFIG),
                "source": protocol.source_identity(), "runtime_root": str(root), "source_study": str(source),
                "files": {name: bytehash(value) for name, value in files.items()}, "order": planned_order(p),
                "seeds": seed_rules(p), "token_examples_fingerprint": fingerprint(workload + development),
                "scope": p["scope"]}
    return settings, files


def prepare(output, runtime_root):
    output = validate_output(output, runtime_root)
    p = protocol.load()
    settings, files = build(runtime_root, p)
    marker = path(output, "study.json")
    if marker.exists():
        previous = read_json(marker)
        # Commit/dirty metadata may change without changing any executing byte.
        settings["source"] = {**settings["source"], **{k: previous.get("source", {}).get(k) for k in ("git_commit", "git_dirty", "note")}}
        if previous != settings:
            if previous.get("source", {}).get("files") != settings["source"]["files"]:
                raise StudyError("Executing source changed after preparation; preserve this output and use a fresh output")
            raise StudyError("Prepared protocol, inputs or runtime root changed; preserve this output and use a fresh output")
    fixed_write(marker, settings)
    for name, value in files.items():
        fixed_write(path(output, name), value)
    runtime.prepare_controls(runtime_root, output)
    return context(output)


class Context:
    def __init__(self, output, settings, workload, development, policy):
        self.output, self.settings, self.p = output, settings, settings["protocol"]
        self.workload, self.development, self.policy = workload, development, policy
        self.examples = {e["sample_id"]: e for e in workload}
        self.dev_examples = {e["sample_id"]: e for e in development}
        self.settings_sha256 = file_hash(path(output, "study.json"))
        self.token_examples = workload + development

    def json(self, relative):
        return read_json(path(self.output, relative))


def context(output, *, offline=False):
    """Verified prepared state, anchored to this released checkout, protocol and the pinned Study 4
    fingerprints. Offline (returned bundle) skips only checks that need the node's runtime root."""
    output = normal(output)
    settings = read_json(path(output, "study.json"))
    p = protocol.validate(settings.get("protocol"))
    require(settings["protocol"] == protocol.load(), "Recorded protocol differs from this checkout's fixed protocol")
    require(settings["source"]["files"] == protocol.source_identity()["files"],
            "Recorded source differs from this checkout; preserve this output and use the released checkout or a fresh output")
    require(settings["order"] == planned_order(p) and settings["seeds"] == seed_rules(p), "Fixed order or seeds changed")
    expected = {"inputs/workload.json", "inputs/development.json", "inputs/policy.json", "inputs/source-binding.json",
                "preflight/schedules/concurrent.json", "preflight/schedules/smoke.json", "preflight/flags/smoke.json",
                *{c["schedule_file"] for c in settings["order"]}, *{c["schedule_file"].replace("schedules/", "flags/") for c in settings["order"]}}
    require(set(settings["files"]) == expected, "Prepared file inventory changed")
    for name, digest in settings["files"].items():
        target = path(output, name)
        require(target.is_file() and file_hash(target) == digest, f"Prepared file changed: {name}")
    workload = inputs.validate_workload(read_json(path(output, "inputs/workload.json")), p)
    development = read_json(path(output, "inputs/development.json"))
    require(len(development) == p["development"]["count"] and fingerprint(development) == p["development"]["rows_fingerprint"],
            "Prepared development inputs changed")
    policy = inputs.verify_policy_document(read_json(path(output, "inputs/policy.json")), p)
    require(settings["token_examples_fingerprint"] == fingerprint(workload + development), "Token input identity changed")
    for name, value in derived_files(p, workload, development).items():
        require(read_json(path(output, name)) == value, f"Prepared schedule or random flags differ from their seed derivation: {name}")
    return Context(output, settings, workload, development, policy)


# ------------------------------------------------------------------------------- attempts
def worker_command(output, folder):
    return [sys.executable, "-B", "-m", WORKER_MODULE, "worker", "--output-dir", str(normal(output)), "--attempt-dir", str(normal(folder))]


def attempt_folders(base):
    if not base.exists():
        return []
    folders = sorted(p for p in base.iterdir() if p.is_dir() and ATTEMPT.fullmatch(p.name))
    require(not any(p.is_symlink() for p in folders), "Symlinked attempt folder")
    return folders


def state_of(folder):
    if (folder / "rejected.json").exists() or (folder / "supervision-failure.json").exists():
        return "failed"
    if (folder / "supervised.json").exists():
        return "completed"
    if (folder / "supervision.json").exists():
        return "interrupted"
    return "unstarted"


def stage_base(output, kind, name):
    return path(output, f"stages/{name}" if kind != "cell" else f"cells/{name}")


def expected_plan(ctx, kind, *, deployment=None, cell=None):
    """Plan fields fixed by the prepared identity and every prerequisite receipt."""
    value = {"kind": kind, "settings_sha256": ctx.settings_sha256, "protocol": PROTOCOL_ID, "revision": REVISION}
    if kind == "tokens":
        value["examples_fingerprint"] = ctx.settings["token_examples_fingerprint"]
    if kind in {"preflight", "cell"}:
        value["tokens_sha256"] = file_hash(path(ctx.output, "preflight/tokens.json"))
    if kind == "preflight":
        value["deployment"] = deployment
    if kind == "cell":
        value["cell"] = cell
        value["preflight_sha256"] = {d: file_hash(path(ctx.output, f"preflight/{d}.json")) for d in DEPLOYMENTS}
    value["watchdog_seconds"] = protocol.watchdog_seconds(
        ctx.p, "tokens" if kind == "tokens" else "cell" if kind == "cell" else deployment,
        None if kind == "tokens" else DEPLOYMENT_OF[cell["condition"]] if kind == "cell" else deployment)
    return value


def run_stage(ctx, kind, name, fields, *, accepted, accept, retry_failed, label):
    """Reuse a verified acceptance, accept a finished attempt, or start one new bounded attempt."""
    receipt = accepted()
    if receipt is not None:
        return "ACCEPTED", receipt
    base = stage_base(ctx.output, kind, name)
    folders = attempt_folders(base)
    prior = []
    for folder in folders:
        state = state_of(folder)
        if state == "completed":
            return "ACCEPTED", accept(folder)
        if state == "failed" and not retry_failed:
            raise StudyError(f"{label} attempt {rel(ctx.output, folder)} failed; inspect its preserved evidence, "
                             "then rerun the same command with --retry-failed")
        if state in {"failed", "interrupted"}:
            # A retry flag is not stop evidence: a valid parent stop, or terminal-job proof from a new
            # allocation, must exist before another worker starts. No historical PID is signalled.
            runtime.recover_previous(ctx.output, folder)
        if state == "unstarted" and not (folder / "not-started.json").exists():
            fixed_write(folder / "not-started.json", {"reason": "controller stopped before supervision", "at": utc_now()})
        prior.append({"attempt": folder.name, "state": state})
    session = runtime.session(ctx.output)
    timeout = fields["watchdog_seconds"]
    allowance = session["deadline_epoch"] - time.time() - ctx.p["cleanup_margin_seconds"]
    if allowance < timeout:
        return "CHECKPOINT", {"stage": label, "remaining_seconds": allowance, "required_seconds": timeout}
    folder = base / f"attempt-{len(folders) + 1:04d}"
    folder.mkdir(parents=True)
    plan = {**fields, "stage": label, "attempt": folder.name, "prior_attempts": prior,
            "retry_failed_acknowledged": bool(retry_failed and any(x["state"] == "failed" for x in prior)),
            "worker_command": worker_command(ctx.output, folder), "allocation": session["allocation"], "created_at": utc_now()}
    fixed_write(folder / "plan.json", plan)
    result = runtime.supervise(ctx.output, folder, plan["worker_command"], timeout)
    if result.get("status") == "CHECKPOINT":
        fixed_write(folder / "not-started.json", {"reason": "checkpoint", **result, "at": utc_now()})
        return "CHECKPOINT", {"stage": label, **result}
    require(result.get("status") == "COMPLETE", "Supervisor returned neither COMPLETE nor CHECKPOINT")
    return "ACCEPTED", accept(folder)


def _accepting(output, folder, derive):
    try:
        return derive()
    except StudyError as exc:
        fixed_write(folder / "rejected.json", {"error": str(exc), "type": type(exc).__name__, "at": utc_now()})
        raise StudyError(f"{rel(output, folder)} is not accepted: {exc}") from exc


# ------------------------------------------------------------------------------- receipts
def verify_supervised(ctx, folder, plan, *, offline):
    """The runtime's acceptance of the lifecycle (on-time zero exit, bound parent cleanup, native
    ownership proof, no failure record), then this study's binding of that run to the plan."""
    runtime.verify_supervised(ctx.output, folder, offline=offline)
    sup = read_json(path(folder, "supervision.json"))
    done = read_json(path(folder, "supervised.json"))
    stop = read_json(path(folder, "parent-stop.json"))
    require(path(folder, "worker-process.json").is_file() and path(folder, "worker.log").is_file(), "Worker lifecycle files are missing")
    require(done.get("status") == "COMPLETE" and done.get("returncode") == 0 and done.get("plan_hash") == fingerprint(sup)
            and done.get("supervision_sha256") == file_hash(path(folder, "supervision.json"))
            and done.get("parent_stop_sha256") == file_hash(path(folder, "parent-stop.json"))
            and done.get("allocation") == sup.get("allocation") == plan["allocation"], "Supervision receipt does not bind this attempt")
    require(stop.get("plan_hash") == fingerprint(sup) and stop.get("gpu_checked") is True and stop.get("gpu_compute_processes") == []
            and "subreaper" in str(stop.get("ownership", "")), "Parent stop does not prove owned-process and GPU cleanup")
    for key in ("started_epoch", "deadline_epoch"):
        require(isinstance(sup.get(key), (int, float)) and not isinstance(sup.get(key), bool), "Supervision timing is invalid")
    require(sup["started_epoch"] <= done.get("exited_epoch", float("inf")) <= sup["deadline_epoch"] == done.get("deadline_epoch")
            and sup["deadline_epoch"] <= sup["started_epoch"] + plan["watchdog_seconds"] + 1e-6,
            "Worker exited after its watchdog or the watchdog differs from the plan")
    require(sup.get("command") == plan["worker_command"], "Supervised command differs from the plan")
    command = plan["worker_command"]
    require(command[1:6] == ["-B", "-m", WORKER_MODULE, "worker", "--output-dir"] and command[7] == "--attempt-dir"
            and Path(command[8]).name == folder.name and len(command) == 9, "Worker command is not this study's worker")
    if not offline:
        require(command == worker_command(ctx.output, folder), "Worker command differs from this checkout and attempt")
    require(sup.get("binding_hash") == fingerprint(read_json(path(ctx.output, "runtime/binding.json"))),
            "Supervised runtime binding differs")
    runtime.verify_native_selftest(ctx.output, sup.get("ownership_selftest"), allocation=sup["allocation"], offline=offline)
    return {"supervised_sha256": file_hash(path(folder, "supervised.json")), "parent_stop_sha256": done["parent_stop_sha256"]}


def verify_complete(folder, plan, names):
    complete = read_json(path(folder, "complete.json"))
    require(complete.get("kind") == plan["kind"] and complete.get("plan_sha256") == file_hash(path(folder, "plan.json"))
            and set(complete.get("files", {})) == set(names), "Worker completion receipt does not bind this plan")
    for name, digest in complete["files"].items():
        target = path(folder, name)
        require(target.is_file() and not target.is_symlink() and file_hash(target) == digest, f"Worker evidence changed: {name}")
    return complete


def check_plan(ctx, folder, expected):
    plan = read_json(path(folder, "plan.json"))
    require(all(plan.get(k) == v for k, v in expected.items()) and plan.get("attempt") == folder.name,
            "Attempt plan differs from the fixed settings and prerequisite receipts")
    # Every earlier attempt of this stage stays preserved with its recorded state; none may vanish.
    prior = plan.get("prior_attempts")
    require(isinstance(prior, list) and [x.get("attempt") for x in prior] == [f.name for f in attempt_folders(folder.parent)][:len(prior)]
            and [f.name for f in attempt_folders(folder.parent)] == [*(x["attempt"] for x in prior), folder.name]
            and all(state_of(folder.parent / x["attempt"]) == x.get("state") for x in prior),
            "Stage attempt history differs from the accepted plan (an earlier attempt is missing, extra or changed)")
    return plan


def token_receipt(ctx, folder, *, offline):
    plan = check_plan(ctx, folder, expected_plan(ctx, "tokens"))
    lifecycle = verify_supervised(ctx, folder, plan, offline=offline)
    complete = verify_complete(folder, plan, {"plan.json", "tokens.json"})
    tokens = read_json(path(folder, "tokens.json"))
    require(tokens.get("settings_sha256") == ctx.settings_sha256
            and tokens.get("examples_fingerprint") == fingerprint(ctx.token_examples)
            and set(tokens.get("receipts", {})) == {f"{c}/{r}" for c, r in TOKEN_TARGETS}, "Token stage output differs")
    for condition, role in TOKEN_TARGETS:
        runtime.verify_tokens(ctx.output, condition, role, ctx.token_examples, tokens["receipts"][f"{condition}/{role}"], offline=offline)
    return {"status": "PASS", "attempt": rel(ctx.output, folder), "settings_sha256": ctx.settings_sha256,
            "plan_sha256": complete["plan_sha256"], "tokens_sha256": file_hash(path(folder, "tokens.json")),
            "complete_sha256": file_hash(path(folder, "complete.json")), **lifecycle, "receipts": tokens["receipts"],
            "examples": len(ctx.token_examples), "scope": "Raw tokenizer reports for all 1,737 formal and 90 development inputs per runtime control"}


def runtime_hash(ctx, condition, role):
    receipt = ctx.json("preflight/tokens.json")["receipts"][f"{RUNTIME_CONDITION.get(DEPLOYMENT_OF[condition], condition)}/{role}"]
    return receipt["runtime_hash"]


def _flag(command, name):
    values = [command[i + 1] for i, part in enumerate(command[:-1]) if part == name]
    return values[0] if len(values) == 1 else None


def verify_deployment(ctx, record, condition, *, offline):
    """The runtime's reconstruction of the exact engine commands and identity, then this study's
    flag, share and token-bound runtime checks; returns the summary and the role endpoint map."""
    runtime.verify_engine_record(ctx.output, condition, record, offline=offline)
    deployment = DEPLOYMENT_OF.get(condition, condition)
    roles = SHARES[deployment]
    require(record.get("condition") == condition and [r.get("role") for r in record.get("roles", [])] == list(roles)
            and record.get("ready_at") and record.get("stopped_at"), "Deployment record is incomplete")
    summary = {}
    for item in record["roles"]:
        role = item["role"]
        port = ctx.p["models"][role]["port"]
        engines = [x for x in item.get("processes", []) if x.get("name") == "startlux-engine"]
        fronts = [x for x in item.get("processes", []) if x.get("name") == "startlux"]
        require(item.get("startup_failed") is False and item.get("endpoint") == f"http://127.0.0.1:{port}"
                and len(engines) == 1 and len(fronts) == 1, f"{role} did not start exactly one engine and frontend")
        command = engines[0].get("command", [])
        require(_flag(command, "--max-num-seqs") == "64" and _flag(command, "--max-model-len") == "8192"
                and _flag(command, "--gpu-memory-utilization") == str(roles[role]) and _flag(command, "--dtype") == "bfloat16",
                f"{role} engine command differs from the fixed share/sequence/length/precision")
        config = item.get("runtime", {})
        require(config.get("model_revision") == ctx.p["models"][role]["revision"] and config.get("max_model_len") == 8192
                and config.get("gpu_memory_utilization") == roles[role] and config.get("truncate") is False
                and config.get("prefix_cache") is False and config.get("precision") == "bfloat16",
                f"{role} effective runtime differs")
        require(fingerprint(config) == runtime_hash(ctx, condition, role), f"{role} runtime differs from its token-checked identity")
        health = item.get("health", {})
        require(health.get("prompt_attention") == "bidirectional" and health.get("defaults", {}).get("think_mode") == "never",
                f"{role} frontend attention/thinking differs")
        summary[role] = {"memory_fraction": roles[role], "startup_seconds": item.get("startup_seconds"),
                         "engine_log": item.get("engine_log"), "gpu_before_start": item.get("gpu_before_start")}
    return {"condition": condition, "roles": summary, "gpu_after_start": record.get("gpu_after_start"),
            "allocation": record.get("allocation"), "started_at": record.get("started_at"), "stopped_at": record.get("stopped_at"),
            "endpoints": {item["role"]: item["endpoint"] for item in record["roles"]}}


def _open_evidence(folder, prefix, schedule, *, condition, mode, label, examples, policy, flags, p, window, endpoints):
    rows = read_jsonl(path(folder, prefix + "traces.jsonl"))
    ledger_events = read_jsonl(path(folder, prefix + "ledger.jsonl"))
    journal = read_jsonl(path(folder, prefix + "fallback.jsonl"))
    require(read_jsonl(path(folder, prefix + "resources.jsonl")), "Open-loop stage has no resource samples")
    ledger = verify.verify_open_loop(rows, schedule, condition=condition, mode=mode, label=label, examples=examples, policy=policy,
                                     flags=flags, ledger_events=ledger_events, journal_events=journal, duration=window,
                                     drain_seconds=p["drain_seconds"], request_timeout=p["request_timeout_seconds"],
                                     endpoints=endpoints)
    report = summarize(rows, duration=window, slo_ms=p["slo_ms"], secondary_slo_ms=p["secondary_slo_ms"])
    facts = read_json(path(folder, prefix + "facts.json"))
    require(facts.get("ledger", {}).get("offered_received") == ledger["offered_received"]
            and facts["ledger"].get("reserved") == ledger["reserved"] and facts["ledger"].get("denied") == ledger["denied"],
            "Recorded ledger state differs from the replay")
    return rows, report, ledger


def preflight_receipt(ctx, deployment, folder, *, offline):
    p = ctx.p
    plan = check_plan(ctx, folder, expected_plan(ctx, "preflight", deployment=deployment))
    lifecycle = verify_supervised(ctx, folder, plan, offline=offline)
    names = {"plan.json", "hardware.json", "deployment-open.json", "deployment.json", "serial/rows.jsonl", "health.json",
             *(f"concurrent/{n}" for n in OPEN_FILES), *(f"smoke-{c}/{n}" for c in SMOKES[deployment] for n in OPEN_FILES)}
    complete = verify_complete(folder, plan, names)
    condition = RUNTIME_CONDITION[deployment]
    record = verify_deployment(ctx, read_json(path(folder, "deployment.json")), condition, offline=offline)
    endpoints = record["endpoints"]
    pair = deployment == "pair"
    mode = "forced_both" if pair else "direct"
    serial = verify.verify_serial(read_jsonl(path(folder, "serial/rows.jsonl")), ctx.development, condition=condition, mode=mode,
                                  label="preflight_serial_forced_both" if pair else None, request_timeout=p["request_timeout_seconds"],
                                  endpoints=endpoints)
    concurrent_rows, concurrent, concurrent_ledger = _open_evidence(
        folder, "concurrent/", ctx.json("preflight/schedules/concurrent.json"), condition=condition, mode=mode,
        label=PREFLIGHT_LABEL if pair else None, examples=ctx.dev_examples, policy=None, flags=None, p=p,
        window=p["preflight"]["concurrent"]["duration_seconds"], endpoints=endpoints)
    smokes = {}
    smoke_schedule = ctx.json("preflight/schedules/smoke.json")
    for smoke in SMOKES[deployment]:
        _, report, ledger = _open_evidence(
            folder, f"smoke-{smoke}/", smoke_schedule, condition=smoke, mode="policy" if pair else "direct", label=None,
            examples=ctx.dev_examples, policy=ctx.policy if pair else None,
            flags=ctx.json("preflight/flags/smoke.json")["flags"] if smoke == "random" else None, p=p,
            window=p["preflight"]["policy_smoke"]["duration_seconds"], endpoints=endpoints)
        smokes[smoke] = {"report": report, "ledger": ledger}
    health = read_json(path(folder, "health.json"))
    problems = []
    if serial["failures"] or serial["count"] != p["preflight"]["serial_count"]:
        problems.append(f"Serial development check: {serial['failures']} failures of {serial['count']}")
    for name, report in [("concurrent", concurrent), *((f"smoke {c}", v["report"]) for c, v in smokes.items())]:
        if report["successful"] != report["offered"]:
            problems.append(f"{name}: errors, deadlines or client rejections {report['status_counts']}")
    overlap = verify.role_overlap_seconds(concurrent_rows) if pair else None
    if pair and not overlap > 0:
        problems.append("The two roles never had overlapping in-flight HTTP calls in the concurrent check")
    if not health.get("roles") or not all(v.get("ok") for v in health["roles"].values()):
        problems.append("A service failed its health check after load")
    logs = health.get("engine_log") or {}
    if set(logs) != set(SHARES[deployment]):
        problems.append("Engine log facts are not recorded for every role")
    if any((facts or {}).get("failure_mentions") for role in logs.values() for facts in (role or {}).values()):
        problems.append("Engine logs report out-of-memory or engine failure before shutdown")
    return {"deployment": deployment, "status": "FAIL" if problems else "PASS", "problems": problems,
            "attempt": rel(ctx.output, folder), "settings_sha256": ctx.settings_sha256, "plan_sha256": complete["plan_sha256"],
            "tokens_sha256": plan["tokens_sha256"], "complete_sha256": file_hash(path(folder, "complete.json")), **lifecycle,
            "evidence": complete["files"], "service": record, "serial": serial,
            "concurrent": {"report": concurrent, "ledger": concurrent_ledger, "role_overlap_seconds": overlap,
                           "label": PREFLIGHT_LABEL if pair else "preflight_concurrent"},
            "smokes": smokes, "health": health,
            "gates": "Engineering only: no accuracy, goodput or minimum-transfer requirement"}


def cell_receipt(ctx, cell, folder, *, offline):
    p = ctx.p
    plan = check_plan(ctx, folder, expected_plan(ctx, "cell", cell=cell))
    lifecycle = verify_supervised(ctx, folder, plan, offline=offline)
    complete = verify_complete(folder, plan, {"plan.json", "hardware.json", "deployment-open.json", "deployment.json",
                                              "warmup/rows.jsonl", "health.json", *OPEN_FILES})
    condition = cell["condition"]
    record = verify_deployment(ctx, read_json(path(folder, "deployment.json")), condition, offline=offline)
    endpoints = record["endpoints"]
    pair = condition in protocol.PAIR_CONDITIONS
    count = p["warmup"]["count"]
    warmup = verify.verify_serial(read_jsonl(path(folder, "warmup/rows.jsonl")), ctx.development[:count], condition=condition,
                                  mode="forced_both" if pair else "direct", label="warmup_forced_both" if pair else None,
                                  request_timeout=p["request_timeout_seconds"], endpoints=endpoints)
    require(warmup["failures"] == 0 and warmup["count"] == count, "Warmup did not complete before measurement")
    flags = ctx.json(cell["flags_file"])["flags"] if condition == "random" else None
    rows, report, ledger = _open_evidence(folder, "", ctx.json(cell["schedule_file"]), condition=condition,
                                          mode="policy" if pair else "direct", label=None, examples=ctx.examples,
                                          policy=ctx.policy if pair else None, flags=flags, p=p, window=p["duration_seconds"],
                                          endpoints=endpoints)
    require(complete.get("report") == report, "Recorded cell summary differs from its raw trace")
    return {"cell_id": cell["cell_id"], "status": "ACCEPTED", "attempt": rel(ctx.output, folder), "cell": cell,
            "settings_sha256": ctx.settings_sha256, "plan_sha256": complete["plan_sha256"],
            "complete_sha256": file_hash(path(folder, "complete.json")), **lifecycle, "evidence": complete["files"],
            "report": report, "ledger": ledger, "warmup": warmup, "service": record, "allocation": plan["allocation"],
            "prior_attempts": plan["prior_attempts"]}


def _verified(ctx, receipt_path, derive):
    """A stored receipt counts only if raw evidence re-derives exactly the same receipt."""
    if not receipt_path.exists():
        return None
    stored = read_json(receipt_path)
    folder = path(ctx.output, stored.get("attempt", ""))
    require(folder.is_dir() and state_of(folder) == "completed", f"Receipt {rel(ctx.output, receipt_path)} names no completed attempt")
    require(derive(folder) == stored, f"Receipt {rel(ctx.output, receipt_path)} differs from its raw evidence")
    return stored


def accepted_tokens(ctx, *, offline=False):
    return _verified(ctx, path(ctx.output, "preflight/tokens.json"), lambda f: token_receipt(ctx, f, offline=offline))


def accepted_preflight(ctx, deployment, *, offline=False):
    stored = _verified(ctx, path(ctx.output, f"preflight/{deployment}.json"),
                       lambda f: preflight_receipt(ctx, deployment, f, offline=offline))
    require(stored is None or stored["status"] == "PASS", f"Preflight {deployment} receipt is not a PASS")
    return stored


def accepted_cell(ctx, cell, *, offline=False):
    return _verified(ctx, path(ctx.output, f"cells/{cell['cell_id']}/accepted.json"), lambda f: cell_receipt(ctx, cell, f, offline=offline))


def tokens_stage(ctx, *, retry_failed):
    def accept(folder):
        receipt = _accepting(ctx.output, folder, lambda: token_receipt(ctx, folder, offline=False))
        fixed_write(path(ctx.output, "preflight/tokens.json"), receipt)
        return receipt
    return run_stage(ctx, "tokens", "tokens", expected_plan(ctx, "tokens"), accepted=lambda: accepted_tokens(ctx),
                     accept=accept, retry_failed=retry_failed, label="tokens")


def preflight_stage(ctx, deployment, *, retry_failed):
    def accept(folder):
        receipt = _accepting(ctx.output, folder, lambda: preflight_receipt(ctx, deployment, folder, offline=False))
        if receipt["status"] != "PASS":
            fixed_write(folder / "rejected.json", receipt)
            raise StudyError(f"Preflight {deployment} failed: {'; '.join(receipt['problems'])}")
        fixed_write(path(ctx.output, f"preflight/{deployment}.json"), receipt)
        return receipt
    return run_stage(ctx, "preflight", f"preflight-{deployment}", expected_plan(ctx, "preflight", deployment=deployment),
                     accepted=lambda: accepted_preflight(ctx, deployment), accept=accept, retry_failed=retry_failed,
                     label=f"preflight-{deployment}")


def cell_stage(ctx, cell, *, retry_failed):
    def accept(folder):
        receipt = _accepting(ctx.output, folder, lambda: cell_receipt(ctx, cell, folder, offline=False))
        fixed_write(path(ctx.output, f"cells/{cell['cell_id']}/accepted.json"), receipt)
        return receipt
    return run_stage(ctx, "cell", cell["cell_id"], expected_plan(ctx, "cell", cell=cell),
                     accepted=lambda: accepted_cell(ctx, cell), accept=accept, retry_failed=retry_failed, label=cell["cell_id"])


# ------------------------------------------------------------------------------- freeze
MUTABLE = re.compile(r"^(status\.json|runtime/session\.json|runtime/ownership-selftest\.json|runtime/session-history/.*"
                     r"|runtime/allocation-checks/.*|runtime-controls/[^/]+/runtime/session\.json)$")
NEVER_EVIDENCE = {".active.lock", ".DS_Store"}  # the OS lock and Finder metadata, never study evidence
FREEZE_ROOTS = ("study.json", "inputs", "schedules", "flags", "preflight", "stages", "cells", "runtime", "runtime-controls",
                "ownership-selftest")


def evidence_files(output, roots=FREEZE_ROOTS):
    output = normal(output)
    files = {}
    for root in roots:
        base = path(output, root)
        if base.is_file():
            files[root] = file_hash(base)
            continue
        for directory, dirs, names in os.walk(base, followlinks=False):
            require(not any((Path(directory) / d).is_symlink() for d in dirs), "Symlinked evidence directory")
            dirs[:] = sorted(d for d in dirs if d not in {"__pycache__", "work"})
            for name in sorted(names):
                relative = rel(output, Path(directory) / name)
                target = path(output, relative)
                require(not target.is_symlink(), "Symlinked evidence file")
                if name not in NEVER_EVIDENCE and not MUTABLE.match(relative):
                    files[relative] = file_hash(target)
    return files


FREEZE_SCOPE = "Every raw trace, budget ledger, fallback journal, service record and verified stop before any gold label is parsed"
FREEZE_KEYS = {"status", "protocol", "revision", "settings_sha256", "source_files_fingerprint", "tokens_sha256",
               "preflight_sha256", "cells", "files", "gold_labels_parsed", "frozen_at", "scope", "self_hash"}


def freeze_bindings(ctx):
    """Every value a freeze must carry, recomputed from the receipts and the evidence tree on disk."""
    return {"status": "FROZEN", "protocol": PROTOCOL_ID, "revision": REVISION, "settings_sha256": ctx.settings_sha256,
            "source_files_fingerprint": fingerprint(ctx.settings["source"]["files"]),
            "tokens_sha256": file_hash(path(ctx.output, "preflight/tokens.json")),
            "preflight_sha256": {d: file_hash(path(ctx.output, f"preflight/{d}.json")) for d in DEPLOYMENTS},
            "cells": {c["cell_id"]: file_hash(path(ctx.output, f"cells/{c['cell_id']}/accepted.json")) for c in ctx.settings["order"]},
            "files": evidence_files(ctx.output), "gold_labels_parsed": False, "scope": FREEZE_SCOPE}


def freeze(ctx, *, offline=False):
    target = path(ctx.output, "freeze.json")
    if target.exists():
        return verify_freeze(ctx, offline=offline)
    require(not path(ctx.output, "evaluation").exists(), "Evaluation evidence exists without a freeze")
    require(accepted_tokens(ctx, offline=offline) is not None, "Token receipt missing at freeze")
    for deployment in DEPLOYMENTS:
        require(accepted_preflight(ctx, deployment, offline=offline) is not None, f"Preflight {deployment} missing at freeze")
    for cell in ctx.settings["order"]:
        require(accepted_cell(ctx, cell, offline=offline) is not None, "All 60 fixed cells must be accepted before the gold is read")
    value = {**freeze_bindings(ctx), "frozen_at": utc_now()}
    value["self_hash"] = fingerprint(value)
    fixed_write(target, value)
    return value


def verify_freeze(ctx, *, offline=False, deep=True):
    """The stored freeze must equal what the evidence on disk implies: exact inventory and digests
    (deliberately mutable allocation/status files excluded) and the actual receipt and source bindings."""
    value = read_json(path(ctx.output, "freeze.json"))
    require(isinstance(value, dict) and set(value) == FREEZE_KEYS and isinstance(value["frozen_at"], str)
            and value["self_hash"] == fingerprint({k: v for k, v in value.items() if k != "self_hash"}),
            "Freeze record fields or self hash differ")
    expected = freeze_bindings(ctx)
    files = expected.pop("files")
    differing = sorted(k for k, v in expected.items() if value[k] != v)
    require(not differing, f"Freeze bindings differ from the actual receipts and source: {differing}")
    frozen = value["files"]
    require(isinstance(frozen, dict) and frozen, "Freeze has no evidence inventory")
    missing = sorted(set(frozen) - set(files))
    unfrozen = sorted(set(files) - set(frozen))
    changed = sorted(name for name in set(files) & set(frozen) if files[name] != frozen[name])
    require(not (missing or unfrozen or changed),
            f"Frozen evidence inventory differs: missing {missing[:3]}, not frozen {unfrozen[:3]}, changed {changed[:3]}")
    if deep:
        require(accepted_tokens(ctx, offline=offline) is not None, "Frozen token receipt is missing")
        for deployment in DEPLOYMENTS:
            require(accepted_preflight(ctx, deployment, offline=offline) is not None, "Frozen preflight is missing")
        for cell in ctx.settings["order"]:
            require(accepted_cell(ctx, cell, offline=offline) is not None, f"Frozen cell {cell['cell_id']} no longer verifies")
    return value


def status(output):
    settings = read_json(path(output, "study.json"))
    order = settings["order"]
    done = [c["cell_id"] for c in order if path(output, f"cells/{c['cell_id']}/accepted.json").exists()]
    return {"protocol": PROTOCOL_ID, "cells_with_receipts": len(done), "cells_required": len(order),
            "tokens": path(output, "preflight/tokens.json").exists(),
            "preflight": {d: (read_json(path(output, f"preflight/{d}.json")).get("status")
                              if path(output, f"preflight/{d}.json").exists() else None) for d in DEPLOYMENTS},
            "frozen": path(output, "freeze.json").exists(), "evaluated": path(output, "evaluation/complete.json").exists(),
            "notice": "Receipt presence only; start/audit re-derive every receipt from raw evidence. No process liveness claim."}


def stage_status(output, stage, state, **details):
    write_json(path(output, "status.json"), {"stage": stage, "state": state, "at": utc_now(), **details})
    print(f"{stage}: {state}", flush=True)
