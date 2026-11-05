"""Software fixtures for resident CPU service and its frozen-artifact boundaries."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from decision_study.service_study import cpu


def serial(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def seal(value):
    return serial({**value, "self_hash": hashlib.sha256(serial(value)).hexdigest()})


@contextmanager
def bundle_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        protocol = {"encoder": cpu.encoder.ENCODER_SPEC, "labels": list(cpu.LABELS)}
        model = {"encoder": cpu.encoder.ENCODER_SPEC, "kind": "frozen_minilm_logistic_regression",
                 "prediction_label_order": list(cpu.LABELS),
                 "embeddings_sha256": hashlib.sha256(b"matrix").hexdigest()}
        artifacts = {"protocol.json": serial(protocol), "embeddings.json": b"{}", "embeddings.npy": b"matrix"}
        for recipe in ("tfidf1600", *cpu.RECIPES):
            artifacts.update({f"{recipe}.model.json": serial(model), f"{recipe}.fit.json": b"{}",
                              f"{recipe}.predictions.jsonl": b"{}\n"})
        for name, raw in artifacts.items():
            (root / name).write_bytes(raw)
        freeze = {"status": "FROZEN", "artifacts": {n: hashlib.sha256(b).hexdigest() for n, b in artifacts.items()},
                  "source_identity": {"git_commit": "fixture", "files": {
                      "decision_study/cheap_baselines/encoder.py": hashlib.sha256(Path(cpu.encoder.__file__).read_bytes()).hexdigest()}}}
        raw_freeze = seal(freeze)
        freeze_sha = hashlib.sha256(raw_freeze).hexdigest()
        raw_access = seal({"freeze_sha256": freeze_sha})
        access_sha = hashlib.sha256(raw_access).hexdigest()
        raw_eval = seal({"status": "COMPLETE", "freeze_sha256": freeze_sha, "access_sha256": access_sha})
        receipts = {"freeze.json": raw_freeze, "evaluation-access.json": raw_access, "evaluation.json": raw_eval}
        for n, b in receipts.items():
            (root / n).write_bytes(b)
        accepted = {n: hashlib.sha256(b).hexdigest() for n, b in receipts.items()}
        model_hash = hashlib.sha256(serial(model)).hexdigest()
        with patch.object(cpu, "ACCEPTED_RECEIPTS", accepted), patch.object(cpu, "ACCEPTED_MODELS", {r: model_hash for r in cpu.RECIPES}):
            yield root, accepted


class FrozenIdentityTests(unittest.TestCase):
    def test_accepted_complete_fixture_and_both_conditions(self):
        with bundle_fixture() as (root, _):
            for recipe in cpu.RECIPES:
                _, _, identity = cpu.verify_bundle(root, recipe)
                self.assertEqual(identity["recipe"], recipe)
            with self.assertRaises(cpu.ServiceError):
                cpu.verify_bundle(root, "new_fit")

    def test_changed_artifact_is_rejected_even_if_unneeded_for_online_prediction(self):
        with bundle_fixture() as (root, _):
            (root / "embeddings.json").write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(cpu.ServiceError, "frozen artifact"):
                cpu.verify_bundle(root, "minilm400")

    def test_receipt_identity_rejects_forged_new_return(self):
        with bundle_fixture() as (root, _):
            value = json.loads((root / "freeze.json").read_text())
            value["status"] = "different"
            (root / "freeze.json").write_bytes(serial(value))
            with self.assertRaisesRegex(cpu.ServiceError, "accepted completed"):
                cpu.verify_bundle(root, "minilm400")

    def test_chain_guard_independently_of_outer_receipt_pin(self):
        with bundle_fixture() as (root, accepted):
            value = json.loads((root / "evaluation.json").read_text())
            value.pop("self_hash")
            value["access_sha256"] = "wrong"
            raw = seal(value)
            (root / "evaluation.json").write_bytes(raw)
            accepted["evaluation.json"] = hashlib.sha256(raw).hexdigest()
            with self.assertRaisesRegex(cpu.ServiceError, "chain"):
                cpu.verify_bundle(root, "minilm400")

    def test_symlink_artifact_rejected(self):
        with bundle_fixture() as (root, _):
            (root / "embeddings.json").unlink()
            (root / "embeddings.json").symlink_to(root / "protocol.json")
            with self.assertRaisesRegex(cpu.ServiceError, "unsafe"):
                cpu.verify_bundle(root, "minilm400")


class InferenceTests(unittest.TestCase):
    def test_native_oos_probability_is_remapped_to_final_study_column(self):
        classes = sorted(cpu.LABELS)
        coefficients = np.zeros((16, 384), dtype=np.float64)
        intercept = np.zeros(16, dtype=np.float64)
        intercept[classes.index("oos")] = 5
        label, values = cpu.classify_embedding(np.zeros((1, 384)), coefficients, intercept, classes)
        self.assertEqual(label, "oos")
        self.assertGreater(values[-1], 0.9)
        self.assertAlmostEqual(sum(values), 1)

    def test_nonfinite_features_rejected_before_softmax(self):
        with self.assertRaises(cpu.ServiceError):
            cpu.classify_embedding(np.full((1, 384), np.nan), np.zeros((16, 384)), np.zeros(16), sorted(cpu.LABELS))

    def test_actual_request_path_encodes_every_repeat_and_rejects_overlength(self):
        from types import SimpleNamespace
        class Tokenizer:
            def __init__(self):
                self.calls = 0
            def encode_batch(self, texts, add_special_tokens):
                self.calls += 1
                return [SimpleNamespace(ids=[101] + [42] * len(t.split()) + [102],
                                        type_ids=[0] * (len(t.split()) + 2),
                                        attention_mask=[1] * (len(t.split()) + 2), overflowing=[]) for t in texts]
        class Session:
            def __init__(self):
                self.calls = 0
            def run(self, outputs, inputs):
                self.calls += 1
                return [np.ones((*inputs["input_ids"].shape, 384), dtype=np.float32)]
        classifier = cpu.FrozenMiniLM.__new__(cpu.FrozenMiniLM)
        classifier.tokenizer, classifier.session = Tokenizer(), Session()
        classifier.coefficients, classifier.intercept = np.zeros((16, 384)), np.zeros(16)
        classifier.classes, classifier.identity = sorted(cpu.LABELS), {"recipe": "minilm400"}
        for _ in range(2):
            result = classifier.predict("authored smoke input")
            self.assertEqual(result["token_count"], 5)
            self.assertFalse(result["truncated"])
            self.assertGreater(result["encoder_ms"], 0)
            self.assertGreaterEqual(result["service_ms"], result["encoder_ms"])
        self.assertEqual(classifier.session.calls, 2)
        self.assertEqual(classifier.tokenizer.calls, 2)
        with self.assertRaises(cpu.InputError):
            classifier.predict("word " * 255)
        self.assertEqual(classifier.session.calls, 2)

    def test_payload_rejects_unknown_fields_and_invalid_text(self):
        for payload in ({"text": "hi", "gold": "oos"}, {"text": ""}, {"text": []}, [],
                        {"text": "x" * (cpu.MAX_TEXT_CHARS + 1)}, {"text": "a", "request_id": 12}):
            with self.subTest(payload=repr(payload)[:60]), self.assertRaises(cpu.InputError):
                cpu.validate_payload(payload)


class FakeClassifier:
    identity = {"recipe": "minilm400", "classifier_sha256": "fixture"}
    environment = {"test_fixture": True}
    load_seconds = 0.0

    def predict(self, text):
        return {"label": "oos", "predicted_label": "oos", "recipe": "minilm400", "service_ms": 1.0}


class AdmissionTests(unittest.TestCase):
    def test_capacity_and_single_worker_are_real(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        class Blocking(FakeClassifier):
            def predict(self, text):
                calls.append(text)
                entered.set()
                if not release.wait(2):
                    raise RuntimeError("test timeout")
                return super().predict(text)
        service = cpu.CPUService(Blocking(), queue_capacity=1)
        try:
            first = service.submit("first", "1")
            self.assertTrue(entered.wait(1))
            second = service.submit("second", "2")
            self.assertIsNone(service.submit("third", "3"))
            self.assertEqual(calls, ["first"])
            self.assertEqual(service.health()["counts"]["active_or_queued"], 2)
            release.set()
            self.assertEqual(first.result(timeout=2)["request_id"], "1")
            self.assertEqual(second.result(timeout=2)["request_id"], "2")
            self.assertEqual(calls, ["first", "second"])
            self.assertEqual(service.health()["counts"]["rejected"], 1)
        finally:
            release.set()
            service.close()

    def test_failed_inference_releases_admission(self):
        class Failed(FakeClassifier):
            def predict(self, text):
                raise cpu.InputError("too long")
        service = cpu.CPUService(Failed(), queue_capacity=0)
        try:
            for _ in range(2):
                future = service.submit("bad", None)
                self.assertIsNotNone(future)
                with self.assertRaises(cpu.InputError):
                    future.result(timeout=2)
            self.assertEqual(service.health()["counts"]["failed"], 2)
            self.assertEqual(service.health()["counts"]["active_or_queued"], 0)
        finally:
            service.close()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.service = cpu.CPUService(FakeClassifier(), queue_capacity=0)
        self.server = cpu.CPUHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.service.close()

    def request(self, method, path, value=None):
        conn = HTTPConnection(*self.server.server_address, timeout=2)
        body = json.dumps(value) if value is not None else None
        conn.request(method, path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    def test_http_contract_and_input_gate(self):
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["workers"], 1)
        self.assertFalse(health["prediction_cache"])
        status, prediction = self.request("POST", "/predict", {"text": "authored request", "request_id": "r1"})
        self.assertEqual(status, 200)
        self.assertEqual(prediction["request_id"], "r1")
        self.assertEqual(prediction["classifier_sha256"], "fixture")
        self.assertGreaterEqual(prediction["queue_ms"], 0)
        self.assertEqual(self.request("POST", "/predict", {"text": "x", "gold": "oos"})[0], 400)
        self.assertEqual(self.request("GET", "/unknown")[0], 404)

    def test_http_handler_admission_is_independent_of_inference_admission(self):
        for _ in range(self.server.maximum_handlers):
            self.assertTrue(self.server.handler_slots.acquire(blocking=False))
        try:
            status, body = self.request("GET", "/health")
            self.assertEqual(status, 429)
            self.assertEqual(body["error"], "http_handlers_full")
        finally:
            for _ in range(self.server.maximum_handlers):
                self.server.handler_slots.release()


if __name__ == "__main__":
    unittest.main()
