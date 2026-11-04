"""Short CPU-only load tests; artificial clients, no HTTP or GPU calls."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from decision_study.common import StudyError, file_hash, fingerprint, write_json, write_jsonl
from decision_study.v2 import openloop


EXAMPLES = [{"sample_id": "software-fixture-1"}, {"sample_id": "software-fixture-2"}]


class PolicyCallTests(unittest.TestCase):
    def test_accept_skips_fallback_and_terminal_fallback_error_stays_error(self):
        local = Mock()
        fallback = Mock()
        local.predict.return_value = {"status": "ok", "prediction": "answerable"}
        with patch.object(openloop, "should_defer", return_value=False):
            result = openloop.call_policy(EXAMPLES[0], local, fallback, {"fixture": True})
        fallback.predict.assert_not_called()
        self.assertFalse(result["deferred"])
        local.predict.return_value = {"status": "error", "prediction": None}
        fallback.predict.return_value = {"status": "error", "prediction": None}
        with patch.object(openloop, "should_defer", side_effect=AssertionError("Failed local has no probability")):
            result = openloop.call_policy(EXAMPLES[0], local, fallback, {"fixture": True})
        self.assertTrue(result["deferred"])
        self.assertEqual(result["status"], "error")


class OpenArrivalTests(unittest.TestCase):
    def test_arrival_schedule_is_seeded_and_rejects_invalid_parameters(self):
        offsets = openloop.arrival_schedule(100, 1, 6998)
        self.assertEqual(offsets, openloop.arrival_schedule(100, 1, 6998))
        self.assertNotEqual(offsets, openloop.arrival_schedule(100, 1, 6999))
        self.assertTrue(all(0 < x < 1 for x in offsets))
        self.assertEqual(offsets, sorted(offsets))
        for rate, duration in ((0, 1), (1, 0), (float("nan"), 1), (1, float("inf"))):
            with self.assertRaises(ValueError):
                openloop.arrival_schedule(rate, duration, 0)

    def test_overload_counts_rejections_in_all_offered_denominator(self):
        gate = threading.Event()
        class Budget:
            def __init__(self):
                self.count = 0
            def check(self):
                self.count += 1
                if self.count == 10:
                    gate.set()
        def invoke(example):
            gate.wait(timeout=.5)
            return {"sample_id": example["sample_id"], "status": "ok", "prediction": "fixture"}
        with patch.object(openloop, "arrival_schedule", return_value=[0.0] * 10):
            report, rows, _ = openloop.run_cell(EXAMPLES, invoke, rate=100, duration=.02, seed=1,
                workers=1, max_inflight=1, slo_ms=1000, budget=Budget())
        self.assertEqual(report["offered"], 10)
        self.assertGreaterEqual(report["rejected"], 8)
        self.assertEqual(report["successful"] + report["failed"] + report["rejected"], 10)
        self.assertEqual([row["offer_id"] for row in rows], list(range(10)))
        self.assertEqual(report["slo_fraction_all_offered"], report["met_slo"] / 10)
        self.assertTrue(all(not row["meets_slo"] for row in rows if row["rejected"]))
        self.assertLess(report["goodput_per_second"], report["offered_rate_actual"])

    def test_queue_wait_and_failures_remain_in_latency_and_counts(self):
        calls = 0
        def invoke(example):
            nonlocal calls
            calls += 1
            time.sleep(.006)
            if calls == 2:
                raise RuntimeError("Artificial failure")
            return {"sample_id": example["sample_id"], "status": "ok", "prediction": "fixture"}
        with patch.object(openloop, "arrival_schedule", return_value=[0.0, 0.0, 0.0]):
            report, rows, _ = openloop.run_cell(EXAMPLES, invoke, rate=100, duration=.01, seed=1,
                workers=1, max_inflight=3, slo_ms=1000)
        self.assertEqual(report["offered"], 3)
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["successful"], 2)
        self.assertEqual(report["rejected"], 0)
        self.assertEqual(rows[1]["status"], "error")
        self.assertFalse(rows[1]["meets_slo"])
        self.assertGreater(rows[2]["worker_start_seconds"] - rows[2]["dispatch_seconds"], .005)
        self.assertGreater(rows[2]["arrival_to_completion_ms"], rows[2]["worker_ms"] + 5)
        self.assertGreater(report["drained_elapsed_seconds"], report["duration_seconds"])
        self.assertLess(report["occupancy_accounting_error_seconds"], 1e-7)

    def test_empty_arrival_window_has_defined_zero_denominators(self):
        with patch.object(openloop, "arrival_schedule", return_value=[]):
            report, rows, events = openloop.run_cell(EXAMPLES, Mock(side_effect=AssertionError("No request offered")),
                rate=.1, duration=.001, seed=1, workers=1, max_inflight=1)
        self.assertEqual(report["offered"], 0)
        self.assertEqual(report["goodput_per_second"], 0)
        self.assertIsNone(report["slo_fraction_all_offered"])
        self.assertIsNone(report["latency_ms_all_admitted"]["p95"])
        self.assertEqual(rows, [])
        self.assertEqual(events, [])


class ResumeEvidenceTests(unittest.TestCase):
    @staticmethod
    def complete(folder, cell="fixture-cell", settings_hash="fixture-settings"):
        write_jsonl(folder / f"{cell}-traces.jsonl", [{"offer_id": 0, "is_test_fixture": True}])
        write_jsonl(folder / f"{cell}-occupancy.jsonl", [{"at_seconds": 0, "active_workers": 1, "is_test_fixture": True}])
        report = {"settings_hash": settings_hash, "offered": 1, "is_test_fixture": True,
                  "evidence_hashes": {kind: file_hash(folder / f"{cell}-{kind}.jsonl") for kind in ("traces", "occupancy")}}
        report["report_hash"] = fingerprint(report)
        write_json(folder / f"{cell}.json", report)
        return report

    def test_completed_cell_requires_summary_identity_and_both_evidence_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertFalse(openloop._completed_cell(folder, "fixture-cell", "fixture-settings"))
            report = self.complete(folder)
            self.assertTrue(openloop._completed_cell(folder, "fixture-cell", "fixture-settings"))
            with self.assertRaises(StudyError):
                openloop._completed_cell(folder, "fixture-cell", "other-settings")
            write_json(folder / "fixture-cell.json", {**report, "offered": 99})
            with self.assertRaises(StudyError):
                openloop._completed_cell(folder, "fixture-cell", "fixture-settings")
            write_json(folder / "fixture-cell.json", report)
            (folder / "fixture-cell-traces.jsonl").write_text("changed\n")
            with self.assertRaisesRegex(StudyError, "traces evidence"):
                openloop._completed_cell(folder, "fixture-cell", "fixture-settings")
            self.complete(folder)
            (folder / "fixture-cell-occupancy.jsonl").unlink()
            with self.assertRaisesRegex(StudyError, "occupancy evidence"):
                openloop._completed_cell(folder, "fixture-cell", "fixture-settings")

    def test_full_and_partial_resume_validate_evidence_before_skipping(self):
        config = {"task": "squad", "unique_inputs": 1, "conditions": ["local_only"], "repeats": 1,
                  "arrival_rates_per_second": [10., 25.], "workers": 1, "max_inflight": 1,
                  "duration_seconds": .001, "slo_ms": 1000.}
        example = {"sample_id": "software-fixture", "task": "squad", "split": "validation"}
        policy = {"selected_policy": {}, "freeze_hash": "software-fixture"}
        client = Mock()
        client.predict.return_value = {"status": "ok", "prediction": "fixture"}
        monitor = Mock()
        monitor.report.return_value = {"is_test_fixture": True}
        @contextmanager
        def monitored():
            yield monitor
        @contextmanager
        def service(*args, **kwargs):
            yield {"startlux": "http://fixture.invalid"}
        measurement = ({"offered": 1, "met_slo": 1, "goodput_per_second": 1, "is_test_fixture": True},
                       [{"offer_id": 0, "is_test_fixture": True}], [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("decision_study.v2.study.load_freeze", return_value=policy), \
                 patch.object(openloop, "protocol", return_value={"openloop": config}), \
                 patch.object(openloop, "source_identity", return_value={"is_test_fixture": True}), \
                 patch.object(openloop, "load_examples", return_value=[example]), \
                 patch.object(openloop, "verify_runtime_model", return_value={"is_test_fixture": True}), \
                 patch.object(openloop.runtime, "token_preflight_examples", return_value={"status": "ok"}), \
                 patch.object(openloop.runtime, "managed_services", side_effect=service) as services, \
                 patch.object(openloop, "ModelClient", return_value=client), \
                 patch.object(openloop, "GPUMonitor", side_effect=monitored), \
                 patch.object(openloop, "run_cell", side_effect=lambda *a, **kw: (dict(measurement[0]), measurement[1], measurement[2])) as measure:
                first = openloop.run(root / "v1", root / "v2", control=root / "control", budget=Mock())
                second = openloop.run(root / "v1", root / "v2", control=root / "control", budget=Mock())
                self.assertEqual(first, second)
                self.assertEqual(services.call_count, 1)
                self.assertEqual(measure.call_count, 2)
                folder = root / "v2/openloop"
                missing = folder / "r0-local_only-rate10-traces.jsonl"
                original = missing.read_bytes()
                missing.unlink()
                with self.assertRaisesRegex(StudyError, "traces evidence"):
                    openloop.run(root / "v1", root / "v2", control=root / "control", budget=Mock())
                self.assertEqual(services.call_count, 1)
                missing.write_bytes(original)
                (folder / "r0-local_only-rate25.json").unlink()
                openloop.run(root / "v1", root / "v2", control=root / "control", budget=Mock())
                self.assertEqual(services.call_count, 2)
                self.assertEqual(measure.call_count, 3)
                config["slo_ms"] = 999
                with self.assertRaisesRegex(StudyError, "settings changed"):
                    openloop.run(root / "v1", root / "v2", control=root / "control", budget=Mock())
                self.assertEqual(services.call_count, 2)

    def test_incomplete_workload_rejected_before_any_services_or_tokenizer(self):
        config = {"task": "squad", "unique_inputs": 200}
        with tempfile.TemporaryDirectory() as temporary, \
             patch("decision_study.v2.study.load_freeze", return_value={"selected_policy": {}}), \
             patch.object(openloop, "protocol", return_value={"openloop": config}), \
             patch.object(openloop, "load_examples", return_value=[]), \
             patch.object(openloop.runtime, "managed_services") as services, \
             patch.object(openloop.runtime, "token_preflight_examples") as tokenize:
            with self.assertRaisesRegex(StudyError, "Incomplete validation workload"):
                openloop.run(Path("v1"), Path(temporary), control=Path("control"), budget=Mock())
            services.assert_not_called()
            tokenize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
