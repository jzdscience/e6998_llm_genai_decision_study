"""One offered request: a direct decision-model call or the two-stage cascade.

Both cascade stages share one monotonic deadline. The fallback receives only the
budget left after the front call and routing; an exhausted budget is recorded as
``deadline`` without a fallback call, and a fallback failure is the request's
failure. Each model call makes at most one HTTP attempt.
"""
import math
import time

from ..adapters import ModelClient
from ..common import StudyError
from ..v3.policies import local_information, route, score_policy
from .identity import INTERFACE

DIRECT = ("startlux4b", "startlux9b")
MODES = ("formal", "forced_fallback", "both_models")


def _stage(role, result, began_ms, ended_ms):
    if result.get("model") != INTERFACE[role]:
        raise StudyError(f"Response from {role} carries model {result.get('model')!r}")
    attempts = result.get("attempts") or []
    if len(attempts) > 1:
        raise StudyError("A model call made more than one HTTP attempt")
    return {"role": role, "sample_id": result["sample_id"], "interface": result["model"], "status": result["status"],
            "prediction": result["prediction"], "probabilities": result["probabilities"],
            "error": result["error"], "attempts": attempts,
            "request_ids": [a.get("request_id") for a in attempts],
            "latency_ms": result["latency_ms"], "started_ms": began_ms, "ended_ms": ended_ms,
            "raw_response": result["raw_response"], "token_counts": result.get("token_counts")}


def _done(status, prediction, source, stages, *, deferred=False, route_record=None, error=None, budget=None):
    return {"sample_id": stages[0]["sample_id"], "status": status, "prediction": prediction if status == "ok" else None, "final_source": source if status == "ok" else None,
            "deferred": deferred, "route": route_record, "stages": stages, "error": error, "budget_seconds": budget}


def call(condition, endpoints, example, timeout, policy=None, *, mode="formal", client=ModelClient, clock=time.monotonic):
    """Return a run_cell-compatible result with every stage preserved."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise StudyError("A positive finite request budget is required")
    if mode not in MODES:
        raise StudyError(f"Unknown call mode: {mode}")
    start = clock()
    deadline = start + timeout
    elapsed_ms = lambda: 1000 * (clock() - start)

    def invoke(role, budget):
        began = elapsed_ms()
        result = client(INTERFACE[role], endpoints[role], timeout=budget, max_attempts=1).predict(example)
        return result, _stage(role, result, began, elapsed_ms())

    if condition in DIRECT:
        if mode != "formal":
            raise StudyError("Direct conditions have no fallback mode")
        result, stage = invoke(condition, timeout)
        status = "ok" if result["status"] == "ok" else "error"
        return _done(status, result["prediction"], condition, [stage], budget=timeout,
                     error=None if status == "ok" else {"type": "model_call", "detail": result["error"]})
    if condition != "cascade":
        raise StudyError(f"Unknown condition: {condition}")
    if mode == "formal" and (not isinstance(policy, dict) or policy.get("family") != "class_conditional"):
        raise StudyError("The cascade needs the frozen class_conditional policy")

    front, front_stage = invoke("startlux4b", timeout)
    stages = [front_stage]
    route_record = {"mode": mode, "family": (policy or {}).get("family"),
                    "threshold": ((policy or {}).get("decision") or {}).get("threshold")}
    if mode == "formal" and front["status"] == "ok":
        # The service checkpoint was verified at startup; the response model name is
        # the interface ("startlux"), while the frozen router expects the role name.
        localrow = {**front, "model": "startlux4b"}
        try:
            local_information(example, localrow)
            score = score_policy(policy, example, localrow)
            deferred = route(policy, example, localrow)
        except StudyError as exc:
            route_record.update(reason="router_input", error=str(exc), deferred=False, fallback_called=False)
            return _done("error", None, None, stages, route_record=route_record, budget=timeout,
                         error={"type": "router_input", "message": str(exc)})
        route_record.update(score=score, deferred=deferred,
                            reason="score_at_or_above_threshold" if deferred else "score_below_threshold")
        if not deferred:
            route_record["fallback_called"] = False
            return _done("ok", front["prediction"], "startlux4b", stages, route_record=route_record, budget=timeout)
    elif mode == "formal":
        route_record.update(score=None, deferred=True, reason="front_failure")
    else:
        route_record.update(score=None, deferred=True, reason="preflight_forced_deferral" if mode == "forced_fallback" else "preflight_both_models")

    remaining = deadline - clock()
    if remaining <= 0:
        route_record["fallback_called"] = False
        return _done("deadline", None, None, stages, deferred=True, route_record=route_record, budget=timeout,
                     error={"type": "budget_exhausted_before_fallback"})
    back, back_stage = invoke("qwen9b", remaining)
    stages.append(back_stage)
    route_record["fallback_called"] = True
    if mode == "both_models" and front["status"] != "ok":
        return _done("error", None, None, stages, deferred=True, route_record=route_record, budget=timeout,
                     error={"type": "model_call", "detail": front["error"]})
    status = "ok" if back["status"] == "ok" else "error"
    return _done(status, back["prediction"], "qwen9b", stages, deferred=True, route_record=route_record, budget=timeout,
                 error=None if status == "ok" else {"type": "fallback_call", "detail": back["error"]})
