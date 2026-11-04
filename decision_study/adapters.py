"""Documented HTTP model interfaces with gold-free, shared task information."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from .common import StudyError


def decision_request(example: dict, model: str) -> dict:
    labels = example["labels"]
    body = {
        "state": example["state"],
        "questions": {"decision": {
            "type": "choice", "instructions": example["instructions"],
            "criteria": {label: example["criteria"][label] for label in labels},
        }},
    }
    if model == "startlux":
        body.update(think_mode="never", none_check=False, allow_none=False)
    return body


def qwen_messages(example: dict) -> list[dict]:
    choices = "\n".join(f"- {label}: {example['criteria'][label]}" for label in example["labels"])
    return [
        {"role": "system", "content": (
            "Complete the classification task using the supplied evidence and label definitions. "
            "Return only a JSON object with a single key, label. Do not include an explanation."
        )},
        {"role": "user", "content": (
            f"{example['instructions']}\n\nLabels:\n{choices}\n\nEvidence:\n{example['state']}"
        )},
    ]


def qwen_request(example: dict, served_name: str = "qwen") -> dict:
    return {
        "model": served_name, "messages": qwen_messages(example),
        "temperature": 0, "max_tokens": 64, "seed": 6998,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"json": {
            "type": "object", "properties": {"label": {"type": "string", "enum": example["labels"]}},
            "required": ["label"], "additionalProperties": False,
        }},
    }


def parse_response(raw: dict, example: dict, model: str) -> tuple[str, dict | None]:
    labels = example["labels"]
    if model == "qwen":
        try:
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("generation did not finish normally")
            answer = json.loads(choice["message"]["content"])
            if set(answer) != {"label"} or answer["label"] not in labels:
                raise ValueError("invalid JSON label or extra fields")
            return answer["label"], None
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StudyError(f"Invalid Qwen answer: {exc}") from exc
    try:
        answer = raw["answers"]["decision"]
        pred, probs = answer["choice"], answer["probabilities"]
        if pred not in labels or set(probs) != set(labels):
            raise ValueError("missing/extra labels or invalid choice")
        if any(isinstance(probs[label], bool) for label in labels):
            raise ValueError("boolean probability")
        values = [float(probs[label]) for label in labels]
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
            raise ValueError("non-finite or out-of-range probability")
        if max(values) - float(probs[pred]) > 1e-8:
            raise ValueError("returned choice does not maximize returned probabilities")
        total = math.fsum(values)
        if model == "kev":
            # The pinned official kev.api.round_prob serializes each coordinate
            # to four decimal places. K rounded entries can miss unit mass by
            # at most K * 0.00005; this is transport precision, not calibration.
            # Reject non-quantized or grossly malformed vectors before restoring
            # unit mass. ModelClient retains the untouched raw_response.
            if any(abs(p * 10_000 - round(p * 10_000)) > 1e-8 for p in values):
                raise ValueError("Kev probabilities do not match four-decimal serialization")
            if total <= 0 or abs(total - 1.0) > len(values) * 0.00005 + 1e-12:
                raise ValueError("Kev probability mass exceeds four-decimal rounding bound")
            values = [p / total for p in values]
        elif abs(total - 1.0) > 1e-5:
            raise ValueError("probabilities do not sum to one")
        # Preserve the server choice among rounded ties: it was selected from
        # unrounded logits and need not be the first serialized maximum.
        return pred, dict(zip(labels, values))
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyError(f"Invalid decision answer: {exc}") from exc


class ModelClient:
    def __init__(self, model: str, endpoint: str, *, timeout: float = 120,
                 max_attempts: int = 2, retry_delay: float = 1,
                 served_name: str = "qwen"):
        if model not in {"kev", "startlux", "qwen"}:
            raise ValueError(model)
        if max_attempts not in (1, 2):
            raise ValueError("Protocol permits at most one retry")
        self.model, self.endpoint = model, endpoint.rstrip("/")
        self.timeout, self.max_attempts = timeout, max_attempts
        self.retry_delay, self.served_name = retry_delay, served_name

    def predict(self, example: dict) -> dict[str, Any]:
        body = qwen_request(example, self.served_name) if self.model == "qwen" else decision_request(example, self.model)
        route = "/v1/chat/completions" if self.model == "qwen" else "/v1/systemone"
        attempts, started = [], time.perf_counter()
        raw, error, prediction, probabilities = None, None, None, None
        for attempt in range(1, self.max_attempts + 1):
            timeout = self.timeout
            deadline = os.environ.get("DECISION_STUDY_DEADLINE_EPOCH")
            if deadline:
                remaining = float(deadline)-time.time()
                if remaining <= 0:
                    raise StudyError("Monitored compute budget deadline reached before request")
                timeout = min(timeout,remaining)
            begin = time.perf_counter()
            request_id = str(uuid.uuid4())
            request = urllib.request.Request(
                self.endpoint + route, data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
                headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            )
            retryable = False
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                    server_request_id = response.headers.get("x-typesafe-request-id") or response.headers.get("x-request-id")
                prediction, probabilities = parse_response(raw, example, self.model)
                attempts.append({"attempt": attempt, "status": "ok", "request_id": request_id,
                                 "server_request_id": server_request_id,
                                 "latency_ms": 1000 * (time.perf_counter() - begin)})
                error = None
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
                error = {"type": "http", "code": exc.code, "message": detail}
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                error = {"type": "transport", "message": str(exc)}
                retryable = True
            except (StudyError, ValueError, KeyError, TypeError) as exc:
                error = {"type": "invalid_response", "message": str(exc)}
            attempts.append({"attempt": attempt, "status": "error", "request_id": request_id,
                             "error": error, "latency_ms": 1000 * (time.perf_counter() - begin)})
            if not retryable or attempt == self.max_attempts:
                break
            time.sleep(self.retry_delay)
        return {
            "sample_id": example["sample_id"], "task": example["task"], "split": example.get("split", "interactive"),
            "model": self.model, "status": "ok" if error is None else "error",
            "prediction": prediction if error is None else None,
            "probabilities": probabilities if error is None else None,
            "raw_response": raw, "error": error, "attempts": attempts,
            "latency_ms": 1000 * (time.perf_counter() - started),
            "token_counts": raw.get("usage", {}) if isinstance(raw, dict) else {},
        }
