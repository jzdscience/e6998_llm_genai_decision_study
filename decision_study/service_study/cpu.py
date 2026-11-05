"""Local HTTP service for the two accepted frozen MiniLM classifiers.

Every request tokenizes and encodes its supplied text. Saved embeddings and task
predictions are verified at startup, never used as an inference lookup table.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import socket
import threading
import time
import uuid

from decision_study.cheap_baselines import encoder

RECIPES = ("minilm400", "minilm1600")
LABELS = ("account_blocked", "balance", "bill_balance", "bill_due", "freeze_account",
          "interest_rate", "min_payment", "order_checks", "pay_bill", "pin_change",
          "report_fraud", "routing", "spending_history", "transactions", "transfer", "oos")
ACCEPTED_RECEIPTS = {
    "freeze.json": "c233e62eea500b795e3c0b22d46f62ceec100f13958ab92f1f4e04f3dbe44986",
    "evaluation-access.json": "1da8a99a719b180f814ad30c218c2cbd491ccc5bae4872bebcc4f3cd42e7f755",
    "evaluation.json": "1935fa2fde23dc3810f7a222ecc7e7716c420f6b7b2e507ff22e43e1eb154a8b",
}
ACCEPTED_MODELS = {
    "minilm400": "103247d32c6bd123cdfe37e9cb2fe78ed0e44c43523e3f08ca6ec960f91a5663",
    "minilm1600": "4e8f16765aa0e196d768d8c7f8fd42e089cbadaf2e9261a92d6909b3efb84dc6",
}
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")}
THREAD_ENV["TOKENIZERS_PARALLELISM"] = "false"
MAX_BODY_BYTES = 65536
MAX_TEXT_CHARS = 16000


class ServiceError(RuntimeError):
    pass


class InputError(ValueError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(root, name):
    path = root / name
    if path.is_symlink() or Path(name).name != name or not path.is_file():
        raise ServiceError(f"Missing or unsafe baseline artifact: {name}")
    return path.read_bytes()


def _unseal(value):
    plain = {k: v for k, v in value.items() if k != "self_hash"}
    encoded = (json.dumps(plain, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    if value.get("self_hash") != _sha(encoded):
        raise ServiceError("Invalid baseline self hash")
    return value


def verify_bundle(bundle_dir, recipe):
    """Require the already-audited return, including all three frozen recipes."""
    if recipe not in RECIPES:
        raise ServiceError("Only accepted minilm400 and minilm1600 are service conditions")
    root = Path(bundle_dir).expanduser().absolute()
    if root.is_symlink():
        raise ServiceError("Baseline root cannot be a symlink")
    raw = {name: _read(root, name) for name in ACCEPTED_RECEIPTS}
    if any(_sha(raw[name]) != sha for name, sha in ACCEPTED_RECEIPTS.items()):
        raise ServiceError("Baseline return differs from the accepted completed run")
    freeze, access, evaluation = (_unseal(json.loads(raw[name])) for name in ACCEPTED_RECEIPTS)
    expected_files = {"protocol.json", "embeddings.npy", "embeddings.json"} | {
        f"{r}.{suffix}" for r in ("tfidf1600", *RECIPES)
        for suffix in ("model.json", "predictions.jsonl", "fit.json")}
    if (freeze.get("status") != "FROZEN" or evaluation.get("status") != "COMPLETE"
            or set(freeze.get("artifacts", {})) != expected_files
            or access.get("freeze_sha256") != ACCEPTED_RECEIPTS["freeze.json"]
            or evaluation.get("freeze_sha256") != ACCEPTED_RECEIPTS["freeze.json"]
            or evaluation.get("access_sha256") != ACCEPTED_RECEIPTS["evaluation-access.json"]):
        raise ServiceError("Baseline completion chain is incomplete")
    artifacts = {name: _read(root, name) for name in expected_files}
    if any(_sha(artifacts[name]) != sha for name, sha in freeze["artifacts"].items()):
        raise ServiceError("Changed baseline frozen artifact")
    protocol = json.loads(artifacts["protocol.json"])
    model = json.loads(artifacts[f"{recipe}.model.json"])
    if (_sha(artifacts[f"{recipe}.model.json"]) != ACCEPTED_MODELS[recipe]
            or model.get("kind") != "frozen_minilm_logistic_regression"
            or model.get("encoder") != encoder.ENCODER_SPEC
            or model.get("prediction_label_order") != list(LABELS)
            or model.get("embeddings_sha256") != freeze["artifacts"]["embeddings.npy"]
            or protocol.get("encoder") != encoder.ENCODER_SPEC
            or protocol.get("labels") != list(LABELS)):
        raise ServiceError("Baseline model/encoder/class identity mismatch")
    expected_encoder_source = freeze["source_identity"]["files"]["decision_study/cheap_baselines/encoder.py"]
    if _sha(Path(encoder.__file__).read_bytes()) != expected_encoder_source:
        raise ServiceError("Frozen encoder source differs from the accepted implementation")
    return protocol, model, {
        "recipe": recipe, "baseline_receipts": dict(ACCEPTED_RECEIPTS),
        "classifier_sha256": ACCEPTED_MODELS[recipe],
        "protocol_sha256": freeze["artifacts"]["protocol.json"],
        "encoder_source_sha256": expected_encoder_source,
        "original_fit_source_commit": freeze["source_identity"]["git_commit"],
    }


def environment(protocol):
    packages = {name: metadata.version(name) for name in protocol["environment"]}
    if packages != protocol["environment"]:
        raise ServiceError("Use the unchanged pinned cheap-baseline environment")
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "processor": platform.processor(),
            "packages": packages, "thread_environment": {k: os.environ.get(k) for k in THREAD_ENV}}


def validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) - {"text", "request_id"}:
        raise InputError("Expected only text and optional request_id")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise InputError(f"text must be a nonempty string of at most {MAX_TEXT_CHARS} characters")
    request_id = payload.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 256):
        raise InputError("request_id must be a string of at most 256 characters")
    return text, request_id


def classify_embedding(matrix, coefficients, intercept, classes):
    """Saved float64 multinomial LR, explicitly mapped to the study label axis."""
    import numpy as np
    if len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ServiceError("Classifier classes differ from the study")
    features = np.asarray(matrix, dtype=np.float64)
    if features.shape != (1, 384) or not np.isfinite(features).all():
        raise ServiceError("Invalid per-request embedding")
    logits = features @ coefficients.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    probability = probability[0, [classes.index(label) for label in LABELS]]
    if not np.isfinite(probability).all() or abs(float(probability.sum()) - 1) > 1e-8:
        raise ServiceError("Invalid classifier probabilities")
    return LABELS[int(probability.argmax())], probability.tolist()


class FrozenMiniLM:
    """Persistent encoder and one saved classifier; there is no prediction cache."""

    def __init__(self, bundle_dir, model_dir, recipe):
        started = time.perf_counter()
        # Set before importing numeric/tokenizer backends; also constrain loaded BLAS.
        os.environ.update(THREAD_ENV)
        protocol, model, identity = verify_bundle(bundle_dir, recipe)
        self.environment = environment(protocol)
        from threadpoolctl import threadpool_info, threadpool_limits
        import numpy as np
        self._thread_limits = threadpool_limits(limits=1)
        self.model_identity = encoder.verify(model_dir, protocol)
        root = Path(model_dir).expanduser().absolute()
        self.tokenizer = encoder._load_tokenizer(root / "tokenizer.json")
        self.session = encoder._load_session(root / "onnx/model.onnx", 1)
        if ({node.name for node in self.session.get_inputs()} != {"input_ids", "attention_mask", "token_type_ids"}
                or any(node.type != "tensor(int64)" for node in self.session.get_inputs())
                or [node.name for node in self.session.get_outputs()] != ["last_hidden_state"]):
            raise ServiceError("Unexpected frozen ONNX graph signature")
        self.coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        self.intercept = np.asarray(model["intercept"], dtype=np.float64)
        self.classes = model["classes"]
        if (self.coefficients.shape != (16, 384) or self.intercept.shape != (16,)
                or not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercept).all()
                or len(self.classes) != 16 or set(self.classes) != set(LABELS)):
            raise ServiceError("Invalid frozen classifier dimensions/numerics")
        self.identity = {**identity, "encoder": self.model_identity}
        self.environment["threadpools"] = threadpool_info()
        self.load_seconds = time.perf_counter() - started

    def predict(self, text):
        import numpy as np
        started = time.perf_counter()
        try:
            tokens, lengths = encoder._tokenize(self.tokenizer, [text], 256)
        except encoder.EncoderError as exc:
            raise InputError(str(exc)) from exc
        row = tokens[0]
        if len(row.type_ids) != lengths[0] or row.attention_mask != [1] * lengths[0]:
            raise ServiceError("Unexpected tokenizer output")
        ids = np.asarray([row.ids], dtype=np.int64)
        mask = np.ones_like(ids)
        types = np.asarray([row.type_ids], dtype=np.int64)
        hidden = self.session.run(["last_hidden_state"], {
            "input_ids": ids, "attention_mask": mask, "token_type_ids": types})[0]
        if hidden.shape != (1, lengths[0], 384):
            raise ServiceError("Invalid ONNX output dimensions")
        matrix = encoder.mean_pool(hidden, mask)
        encoded = time.perf_counter()
        label, probabilities = classify_embedding(matrix, self.coefficients, self.intercept, self.classes)
        completed = time.perf_counter()
        return {"label": label, "predicted_label": label, "probabilities": probabilities,
                "labels": list(LABELS), "recipe": self.identity["recipe"],
                "token_count": lengths[0], "truncated": False,
                "encoder_ms": (encoded - started) * 1000,
                "classifier_ms": (completed - encoded) * 1000,
                "service_ms": (completed - started) * 1000}


def resource_identity():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    limits = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            limits[name] = path.read_text().strip()
    return {"cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
            "allowed_cpu_affinity": affinity, "affinity_enforced": False,
            "cgroup_v2_root_limits": limits,
            "note": "One inference worker and numeric threads=1; no claim of exclusive CPU or enforced core affinity."}


class CPUService:
    def __init__(self, classifier, queue_capacity=64):
        if type(queue_capacity) is not int or not 0 <= queue_capacity <= 64:
            raise ServiceError("queue_capacity must be an integer between 0 and 64")
        self.classifier = classifier
        self.queue_capacity = queue_capacity
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minilm-inference")
        self.admission = threading.BoundedSemaphore(queue_capacity + 1)
        self.lock = threading.Lock()
        self.active = 0
        self.completed = 0
        self.rejected = 0
        self.failed = 0
        self.instance_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.resources = resource_identity()

    def submit(self, text, request_id):
        if not self.admission.acquire(blocking=False):
            with self.lock:
                self.rejected += 1
            return None
        queued_at = time.perf_counter()
        with self.lock:
            self.active += 1
        def work():
            dispatched_at = time.perf_counter()
            try:
                value = self.classifier.predict(text)
                value.update(request_id=request_id, queue_ms=(dispatched_at - queued_at) * 1000,
                             instance_id=self.instance_id,
                             classifier_sha256=self.classifier.identity["classifier_sha256"])
                with self.lock:
                    self.completed += 1
                return value
            except Exception:
                with self.lock:
                    self.failed += 1
                raise
            finally:
                with self.lock:
                    self.active -= 1
                self.admission.release()
        try:
            return self.executor.submit(work)
        except Exception:
            with self.lock:
                self.active -= 1
            self.admission.release()
            raise

    def health(self):
        with self.lock:
            counts = {"active_or_queued": self.active, "completed": self.completed,
                      "rejected": self.rejected, "failed": self.failed}
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_scale = 1 if platform.system() == "Darwin" else 1024
        return {"status": "ok", "service": "frozen_minilm_cpu", "instance_id": self.instance_id,
                "pid": os.getpid(), "started_at_utc": self.started_at,
                "identity": self.classifier.identity, "environment": self.classifier.environment,
                "resources": self.resources, "load_seconds": self.classifier.load_seconds,
                "batch_size": 1, "workers": 1, "queue_capacity": self.queue_capacity,
                "maximum_admitted": self.queue_capacity + 1, "prediction_cache": False,
                "numeric_threads": 1, "counts": counts,
                "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
                "process_peak_rss_bytes": int(usage.ru_maxrss * rss_scale)}

    def close(self):
        self.executor.shutdown(wait=True)


class CPUHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = True
    request_queue_size = 128

    def __init__(self, address, service):
        self.service = service
        self.maximum_handlers = service.queue_capacity + 8
        self.handler_slots = threading.BoundedSemaphore(self.maximum_handlers)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.handler_slots.acquire(blocking=False):
            try:
                request.settimeout(0.2)
                raw = b'{"error":"http_handlers_full"}'
                request.sendall(b"HTTP/1.0 429 Too Many Requests\r\nContent-Type: application/json\r\n"
                                + f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.handler_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.handler_slots.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # One response per connection, matching the client transport.

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_):
        pass

    def reply(self, code, value):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            # The admitted inference still finishes; client timeout remains in its trace.
            pass

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "unknown_path"})
        self.reply(200, {**self.server.service.health(),
                         "maximum_http_handlers": self.server.maximum_handlers})

    def do_POST(self):
        if self.path != "/predict":
            return self.reply(404, {"error": "unknown_path"})
        if self.headers.get("Transfer-Encoding"):
            return self.reply(400, {"error": "chunked_requests_unsupported"})
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self.reply(400, {"error": "missing_or_invalid_content_length"})
        if not 0 < length <= MAX_BODY_BYTES:
            return self.reply(413, {"error": "request_body_size"})
        if self.headers.get_content_type() != "application/json":
            return self.reply(415, {"error": "application_json_required"})
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InputError("Incomplete request body")
            text, request_id = validate_payload(json.loads(raw))
        except (InputError, ValueError, UnicodeDecodeError, socket.timeout) as exc:
            return self.reply(400, {"error": "invalid_input", "detail": str(exc)})
        future = self.server.service.submit(text, request_id)
        if future is None:
            return self.reply(429, {"error": "queue_full", "request_id": request_id})
        try:
            result = future.result()
        except InputError as exc:
            return self.reply(400, {"error": "invalid_input", "detail": str(exc), "request_id": request_id})
        except Exception as exc:
            return self.reply(500, {"error": "inference_failed", "detail": type(exc).__name__, "request_id": request_id})
        self.reply(200, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--queue-capacity", type=int, choices=(64,), default=64)
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536:
        parser.error("port must be between 1 and 65535")
    service = None
    server = None
    try:
        classifier = FrozenMiniLM(args.bundle_dir, args.model_dir, args.recipe)
        service = CPUService(classifier, args.queue_capacity)
        server = CPUHTTPServer((args.host, args.port), service)
        print(json.dumps({"ready": True, "url": f"http://{args.host}:{args.port}",
                          "health": service.health()}, ensure_ascii=False), flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except (ServiceError, encoder.EncoderError, OSError, ValueError, metadata.PackageNotFoundError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    finally:
        if server is not None:
            server.server_close()
        if service is not None:
            service.close()


if __name__ == "__main__":
    main()
