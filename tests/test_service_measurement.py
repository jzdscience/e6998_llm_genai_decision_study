"""Independent, short software checks for the paired service measurement.

All inputs and responses below are artificial. No model, gold file or GPU is
loaded, and these checks establish accounting behavior rather than performance.
"""
import json
from pathlib import Path
import tempfile
import time
import unittest

from decision_study.service_study import measurement
from decision_study.common import StudyError, file_hash, fingerprint, write_json, write_jsonl


EXAMPLES = [{"sample_id": f"fixture-{i}"} for i in range(3)]


def offers(count, arrival=0.0):
    return [{"offer_id": i, "sample_id": EXAMPLES[i % len(EXAMPLES)]["sample_id"],
             "arrival_seconds": arrival + i * 1e-6} for i in range(count)]


class PairedScheduleTests(unittest.TestCase):
    def test_schedule_is_reproducible_and_contains_only_bound_input_ids(self):
        first = measurement.make_schedule(EXAMPLES, rate=100, duration=1, seed=6998)
        same = measurement.make_schedule(EXAMPLES, rate=100, duration=1, seed=6998)
        other = measurement.make_schedule(EXAMPLES, rate=100, duration=1, seed=6999)
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertGreater(len(first), 10)
        self.assertEqual([r["offer_id"] for r in first], list(range(len(first))))
        self.assertTrue(all(r["sample_id"] in {e["sample_id"] for e in EXAMPLES}
                            for r in first))
        offsets = [r["arrival_seconds"] for r in first]
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(0 <= at < 1 for at in offsets))
        # No client condition can be part of this public scheduling interface.
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(same, sort_keys=True))

    def test_invalid_rate_and_window_are_rejected(self):
        for rate, duration in [(0, 1), (-1, 1), (float("nan"), 1),
                               (1, 0), (1, -1), (1, float("inf"))]:
            with self.subTest(rate=rate, duration=duration):
                with self.assertRaises((ValueError, RuntimeError)):
                    measurement.make_schedule(EXAMPLES, rate=rate, duration=duration, seed=0)


class OfferedAccountingTests(unittest.TestCase):
    def test_client_overload_preserves_every_offered_id_and_journal_row(self):
        def slow(row, timeout):
            time.sleep(.05)
            return {"status": "ok", "prediction": "fixture-label"}

        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            report, rows, events = measurement.run_cell(
                offers(12), slow, duration=.01, workers=1, max_inflight=1,
                slo_ms=1000, drain_seconds=1, request_timeout=.5, trace_path=trace)
            journal = [json.loads(line) for line in trace.read_text().splitlines() if line]
        self.assertEqual(report["offered"], 12)
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["offer_id"] for r in rows}, set(range(12)))
        self.assertEqual({r["offer_id"] for r in journal}, set(range(12)))
        self.assertGreaterEqual(report["rejected"], 10)
        self.assertEqual(report["successful"] + report["failed_admitted"] + report["rejected"], 12)
        self.assertEqual(report["slo_fraction_all_offered"], report["met_slo"] / 12)
        self.assertLess(report["goodput_per_second"], 12 / .01)
        self.assertTrue(events)
        self.assertTrue(all(row["status"] != "ok" for row in rows if row["rejected"]))

    def test_queue_wait_and_terminal_failure_remain_in_accounting(self):
        observed_timeouts = []

        def invoke(row, timeout):
            observed_timeouts.append(timeout)
            time.sleep(.012)
            if row["offer_id"] == 1:
                raise RuntimeError("Deliberate software fixture failure")
            return {"status": "ok", "prediction": "fixture-label"}

        report, rows, _ = measurement.run_cell(
            offers(3), invoke, duration=.01, workers=1, max_inflight=3,
            slo_ms=1000, drain_seconds=1, request_timeout=.5)
        by_id = {r["offer_id"]: r for r in rows}
        self.assertEqual(report["offered"], 3)
        self.assertEqual(report["successful"], 2)
        self.assertEqual(report["failed_admitted"], 1)
        self.assertEqual(report["rejected"], 0)
        self.assertEqual(by_id[1]["status"], "error")
        self.assertEqual(report["met_slo"], 2)
        self.assertGreater(by_id[2]["worker_start_seconds"] - by_id[2]["dispatch_seconds"], .015)
        self.assertGreater(by_id[2]["arrival_to_completion_ms"], by_id[2]["worker_ms"] + 15)
        self.assertEqual(len(observed_timeouts), 3)
        self.assertTrue(all(0 < timeout <= .5 for timeout in observed_timeouts))

    def test_empty_window_has_defined_no_request_denominators(self):
        def forbidden(*args):
            raise AssertionError("An empty offer schedule must not invoke a client")
        report, rows, events = measurement.run_cell(
            [], forbidden, duration=.001, workers=1, max_inflight=1,
            slo_ms=1000, drain_seconds=.1, request_timeout=.1)
        self.assertEqual(rows, [])
        self.assertEqual(events, [])
        self.assertEqual(report["offered"], 0)
        self.assertEqual(report["goodput_per_second"], 0)
        self.assertIsNone(report["slo_fraction_all_offered"])
        self.assertIsNone(report["latency_ms_all_admitted"]["p95"])


class CorrectGoodputTests(unittest.TestCase):
    def test_wrong_fast_and_correct_late_answers_do_not_count_as_correct_goodput(self):
        rows = []
        for i, (status, predicted, latency) in enumerate([
                ("ok", "right", 100), ("ok", "wrong", 100),
                ("ok", "right", 900), ("error", "right", 100)]):
            rows.append({"offer_id": i, "sample_id": "fixture", "arrival_seconds": i,
                         "dispatch_seconds": i, "rejected": False, "status": status,
                         "arrival_to_completion_ms": latency,
                         "result": {"prediction": predicted}})
        rows.append({"offer_id": 4, "sample_id": "fixture", "arrival_seconds": 4,
                     "dispatch_seconds": 4, "rejected": True, "status": "client_rejected",
                     "result": None})
        plain = measurement.summarize(rows, duration=10, slo_ms=500)
        self.assertNotIn("correct", plain)
        result = measurement.summarize(rows, duration=10, slo_ms=500, gold={"fixture": "right"})
        self.assertEqual(result["offered"], 5)
        self.assertEqual(result["successful"], 3)
        self.assertEqual(result["met_slo"], 2)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["correct_met_slo"], 1)
        self.assertEqual(result["correct_goodput_per_second"], .1)
        self.assertEqual(result["correct_fraction_all_offered"], .4)
        self.assertEqual(result["accuracy_successful"], 2 / 3)
        self.assertEqual(result["goodput_per_second"], .2)
        with self.assertRaisesRegex(StudyError, "Missing labels"):
            measurement.summarize(rows, duration=10, slo_ms=500, gold={})


class CompletedEvidenceTests(unittest.TestCase):
    @staticmethod
    def complete(folder):
        plan = {"is_software_fixture": True, "schedule": offers(1)}
        write_json(folder / "plan.json", plan)
        write_jsonl(folder / "traces.jsonl", [{"offer_id": 0, "is_software_fixture": True}])
        write_jsonl(folder / "occupancy.jsonl", [{"at_seconds": 0, "active_workers": 1}])
        write_jsonl(folder / "resources.jsonl", [{"is_software_fixture": True}])
        record = {"plan_hash": fingerprint(plan), "files": {
            name: file_hash(folder / name) for name in
            ("plan.json", "traces.jsonl", "occupancy.jsonl", "resources.jsonl")}}
        record["self_hash"] = fingerprint(record)
        write_json(folder / "complete.json", record)
        return record

    def test_resume_requires_matching_summary_and_each_raw_file(self):
        for name in ("plan.json", "traces.jsonl", "occupancy.jsonl", "resources.jsonl"):
            for action in ("remove", "change"):
                with self.subTest(name=name, action=action), tempfile.TemporaryDirectory() as temporary:
                    folder = Path(temporary)
                    self.assertIsNone(measurement.verify_cell(folder, "fixture"))
                    record = self.complete(folder)
                    self.assertEqual(measurement.verify_cell(folder, record["plan_hash"]), record)
                    target = folder / name
                    if action == "remove":
                        target.unlink()
                    else:
                        target.write_text("{}\n")
                    with self.assertRaisesRegex(StudyError, "evidence changed or missing"):
                        measurement.verify_cell(folder, record["plan_hash"])

    def test_changed_plan_identity_and_edited_complete_receipt_block_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            record = self.complete(folder)
            with self.assertRaisesRegex(StudyError, "identity mismatch"):
                measurement.verify_cell(folder, "another-plan")
            write_json(folder / "complete.json", {**record, "unexpected": True})
            with self.assertRaisesRegex(StudyError, "identity mismatch"):
                measurement.verify_cell(folder, record["plan_hash"])


if __name__ == "__main__":
    unittest.main()
