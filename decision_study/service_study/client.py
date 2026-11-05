"""One actual HTTP attempt per offered request, preserving the observed response."""
import json
import math
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from ..adapters import ModelClient
from ..common import StudyError
from .cpu import ACCEPTED_MODELS, LABELS


def call_http(condition, endpoint, example, request_id, timeout):
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("", "/"):
        raise StudyError("Service measurements require the owned loopback endpoint")
    if condition in {"startlux4b", "qwen4b"}:
        model = {"startlux4b": "startlux", "qwen4b": "qwen"}[condition]
        return ModelClient(model, endpoint, timeout=timeout, max_attempts=1).predict(example)
    if condition not in ACCEPTED_MODELS:
        raise StudyError("Unknown service condition")
    started = time.perf_counter()
    request = urllib.request.Request(endpoint+"/predict", method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"text": example["state"], "request_id": request_id}).encode())
    raw = None
    result = {"sample_id": example["sample_id"], "prediction": None, "request_id": request_id}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("Oversized response")
        raw = json.loads(data)
        probabilities = raw["probabilities"]
        if (raw["request_id"] != request_id or raw["recipe"] != condition
                or raw["classifier_sha256"] != ACCEPTED_MODELS[condition]
                or raw["labels"] != list(LABELS) or raw["label"] not in LABELS
                or raw["predicted_label"] != raw["label"] or len(probabilities) != len(LABELS)
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                or abs(sum(probabilities)-1) > 1e-6
                or max(probabilities)-probabilities[LABELS.index(raw["label"])] > 1e-8):
            raise ValueError("Wrong classifier, class axis or invalid probabilities")
        result.update(status="ok", prediction=raw["label"], probabilities=dict(zip(LABELS, probabilities)))
    except urllib.error.HTTPError as exc:
        result.update(status="http_rejected" if exc.code == 429 else "error",
                      error={"http_code": exc.code, "detail": exc.read(4096).decode(errors="replace")})
    except (TimeoutError, ConnectionError, OSError) as exc:
        result.update(status="timeout" if isinstance(exc, TimeoutError) else "error", error=str(exc))
    except (ValueError, KeyError, TypeError) as exc:
        result.update(status="error", error={"type": "invalid_response", "message": str(exc)})
    result.update(raw_response=raw, attempts=1, latency_ms=1000*(time.perf_counter()-started))
    return result
