"""Independent integration checks using only visibly artificial service records."""
from contextlib import ExitStack, contextmanager
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from decision_study.common import StudyError, file_hash, fingerprint, read_json, write_json, write_jsonl
from decision_study.service_study import study


@contextmanager
def fixture(*, full_inventory=False):
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
        root = Path(temporary)
        output, v1 = root / "output", root / "v1"
        p = copy.deepcopy(study.protocol())
        if not full_inventory:
            p.update(conditions=["minilm400", "minilm1600"], rates_per_second=[5],
                     repeats=1, duration_seconds=1)
        rows = [{"sample_id": f"software-{i}", "state": f"Artificial input {i}",
                 "task": "software_fixture", "split": "test", "labels": ["yes", "no"]}
                for i in range(2)]
        warm = [{**rows[0], "sample_id": "software-warm", "split": "development"}]
        gold = {row["sample_id"]: "yes" for row in rows}
        write_json(v1 / "data/gold.json", gold)
        stack.enter_context(patch.object(study, "protocol", return_value=p))
        stack.enter_context(patch.object(study, "source_identity", return_value={"software_fixture": True}))
        stack.enter_context(patch.object(study, "load_inputs", return_value=(rows, warm, {"software_fixture": True})))
        stack.enter_context(patch.object(study, "verify_bundle", return_value=(None, None, {"software_fixture": True})))
        stack.enter_context(patch.object(study, "GOLD_SHA256", file_hash(v1 / "data/gold.json")))
        prepared = study.prepare(v1, root / "bundle", root / "encoder", root / "python", output)
        config = study.settings(output)
        yield root, output, v1, p, config, prepared


def complete(output, config, cell, *, label="yes", attempt=1):
    """Create a coherent artificial receipt, independently of child execution."""
    folder = output / "cells" / cell["cell_id"] / f"attempt-{attempt:04d}"
    folder.mkdir(parents=True)
    service = output / "services" / f"{cell['cell_id']}.json"
    write_json(service, {"software_fixture": True})
    warmup = output / "services" / f"{cell['cell_id']}-warmup.json"
    write_json(warmup, {"rows": [{"sample_id": row["sample_id"], "status": "ok",
                                  "prediction": "yes", "software_fixture": True}
                                 for row in read_json(output / "warmup.json")]})
    plan = {**cell, "settings_hash": fingerprint(config), "protocol": config["protocol"],
            "schedule_sha256": config["schedules"][cell["schedule_file"]],
            "workload_file": "workload.json", "workload_sha256": config["workload_sha256"],
            "service_record_file": str(service.relative_to(output)),
            "service_record_sha256": file_hash(service),
            "warmup_record_file": str(warmup.relative_to(output)),
            "warmup_record_sha256": file_hash(warmup),
            "allocation": {"hostname": "software-fixture", "slurm_job_id": "fixture"}}
    write_json(folder / "plan.json", plan)
    rows = [{**offer, "dispatch_seconds": offer["arrival_seconds"],
             "worker_start_seconds": offer["arrival_seconds"],
             "completion_seconds": offer["arrival_seconds"] + .1,
             "arrival_to_completion_ms": 100.0, "client_queue_ms": 0.0, "worker_ms": 100.0,
             "rejected": False, "status": "ok",
             "result": {"status": "ok", "prediction": label,
                        "raw_response": {"software_fixture": True, "label": label}}}
            for offer in read_json(output / cell["schedule_file"])]
    write_jsonl(folder / "traces.jsonl", rows)
    write_jsonl(folder / "occupancy.jsonl", [{"at_seconds": 0, "active_workers": 0}])
    write_jsonl(folder / "resources.jsonl", [{"software_fixture": True}])
    reseal(folder, config)
    return folder, rows


def reseal(folder, config, *, recompute=True):
    if recompute:
        report = study.summarize(read_jsonl(folder / "traces.jsonl"),
                                 config["protocol"]["duration_seconds"], config["protocol"]["slo_ms"])
    else:
        report = read_json(folder / "complete.json")["report"]
    record = {"plan_hash": fingerprint(read_json(folder / "plan.json")), "report": report,
              "files": {name: file_hash(folder / name) for name in
                        ("plan.json", "traces.jsonl", "occupancy.jsonl", "resources.jsonl")}}
    record["self_hash"] = fingerprint(record)
    write_json(folder / "complete.json", record)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class PreparationInventoryTests(unittest.TestCase):
    def test_formal_protocol_has_exact60_cells15_paired_schedules_and3_offer_hours(self):
        with fixture(full_inventory=True) as (_, output, _, p, config, prepared):
            self.assertEqual(prepared["cells"], 60)
            self.assertEqual(prepared["offer_window_hours"], 3)
            self.assertEqual(len(config["schedules"]), 15)
            self.assertEqual(len({c["cell_id"] for c in config["order"]}), 60)
            for repeat in range(5):
                for rate in (5, 25, 100):
                    selected = [c for c in config["order"] if c["repeat"] == repeat and c["rate"] == rate]
                    self.assertEqual({c["condition"] for c in selected}, set(p["conditions"]))
                    self.assertEqual(len({c["schedule_file"] for c in selected}), 1)
            self.assertFalse((output / "evaluation-access.json").exists())

    def test_settings_rejects_missing_or_duplicated_formal_cells(self):
        for edit in ("missing", "duplicate"):
            with self.subTest(edit=edit), fixture() as (_, output, _, _, config, _):
                bad = copy.deepcopy(config)
                if edit == "missing":
                    bad["order"].pop()
                else:
                    bad["order"][-1] = bad["order"][0]
                write_json(output / "settings.json", bad)
                with self.assertRaises(StudyError):
                    study.settings(output)


class CompleteSemanticTests(unittest.TestCase):
    def test_schedule_mismatch_is_rejected_even_under_fresh_hashes(self):
        with fixture() as (_, output, _, _, config, _):
            cell = config["order"][0]
            folder, rows = complete(output, config, cell)
            self.assertIsNotNone(study.completed(output, config, cell))
            rows[0]["sample_id"] = "another-artificial-input"
            write_jsonl(folder / "traces.jsonl", rows)
            reseal(folder, config)
            with self.assertRaisesRegex(StudyError, "exact offered schedule"):
                study.completed(output, config, cell)

    def test_summary_must_match_raw_trace_even_when_receipt_is_self_consistent(self):
        with fixture() as (_, output, _, _, config, _):
            cell = config["order"][0]
            folder, rows = complete(output, config, cell)
            rows[0]["status"] = "error"
            rows[0]["result"]["status"] = "error"
            write_jsonl(folder / "traces.jsonl", rows)
            reseal(folder, config, recompute=False)
            with self.assertRaisesRegex(StudyError, "summary differs"):
                study.completed(output, config, cell)

    def test_multiple_completed_attempts_never_select_a_result(self):
        with fixture() as (_, output, _, _, config, _):
            cell = config["order"][0]
            complete(output, config, cell)
            complete(output, config, cell, attempt=2)
            with self.assertRaisesRegex(StudyError, "Multiple completed attempts"):
                study.completed(output, config, cell)


class EvaluationBoundaryTests(unittest.TestCase):
    def test_incomplete_cells_cannot_open_gold_or_create_access(self):
        with fixture() as (_, output, v1, _, config, _):
            complete(output, config, config["order"][0])
            original = study.read_json
            def guard(path):
                if Path(path) == v1 / "data/gold.json":
                    raise AssertionError("Gold opened before all service cells completed")
                return original(path)
            with patch.object(study, "read_json", side_effect=guard):
                with self.assertRaisesRegex(StudyError, "must complete"):
                    study.evaluate(output)
            self.assertFalse((output / "evaluation-access.json").exists())

    def test_actual_response_predictions_are_scored_only_after_freeze_and_access(self):
        with fixture() as (_, output, v1, _, config, _):
            counts = {}
            for cell in config["order"]:
                label = "yes" if cell["condition"] == "minilm400" else "no"
                _, rows = complete(output, config, cell, label=label)
                counts[cell["condition"]] = len(rows)
            original = study.read_json
            gold_reads = []
            def guard(path):
                if Path(path) == v1 / "data/gold.json":
                    self.assertTrue((output / "freeze.json").is_file())
                    self.assertTrue((output / "evaluation-access.json").is_file())
                    gold_reads.append(path)
                return original(path)
            with patch.object(study, "read_json", side_effect=guard):
                evaluation = study.evaluate(output)
                again = study.evaluate(output)
            self.assertEqual(evaluation, again)
            self.assertEqual(len(gold_reads), 1)
            scored = {row["condition"]: row for row in evaluation["cells"]}
            self.assertEqual(scored["minilm400"]["correct"], counts["minilm400"])
            self.assertEqual(scored["minilm1600"]["correct"], 0)
            self.assertEqual(evaluation["status"], "COMPLETE")

    def test_existing_access_without_evaluation_blocks_silent_rescoring(self):
        with fixture() as (_, output, v1, _, config, _):
            for cell in config["order"]:
                complete(output, config, cell)
            study.freeze(output)
            write_json(output / "evaluation-access.json", {"software_fixture": True})
            original = study.read_json
            original_write = study.fixed_write
            def guard(path):
                if Path(path) == v1 / "data/gold.json":
                    raise AssertionError("Partial evaluation must not reopen labels")
                return original(path)
            def guard_write(path, value):
                if Path(path) == output / "evaluation-access.json":
                    raise AssertionError("Partial-access gate must stop before attempting a new access receipt")
                return original_write(path, value)
            with patch.object(study, "read_json", side_effect=guard), \
                    patch.object(study, "fixed_write", side_effect=guard_write):
                with self.assertRaisesRegex(StudyError, "previously started"):
                    study.evaluate(output)

    def test_collection_cannot_resume_after_freeze_if_completion_evidence_disappears(self):
        with fixture() as (_, output, _, _, config, _):
            folders = [complete(output, config, cell)[0] for cell in config["order"]]
            study.freeze(output)
            (folders[0] / "complete.json").unlink()
            with patch.object(study.runtime, "validate", side_effect=AssertionError("Runtime entered after freeze")):
                with self.assertRaises(StudyError):
                    study.run(output, resume_interrupted=True)

    def test_status_does_not_call_a_bogus_evaluation_complete(self):
        with fixture() as (_, output, _, _, config, _):
            for cell in config["order"]:
                complete(output, config, cell)
            write_json(output / "evaluation.json", {"status": "COMPLETE"})
            with self.assertRaises(StudyError):
                study.status(output)


class InterruptedResumeAndExportTests(unittest.TestCase):
    def test_watchdog_stops_owned_child_and_disqualifies_racing_completion(self):
        with fixture() as (_, output, _, p, config, _):
            folder = study._attempt(output, config["order"][0], resume_interrupted=False)
            process = Mock()
            process.poll.return_value = None
            racing = b'{"software_fixture":"racing completion"}\n'
            def stop(child):
                self.assertIs(child, process)
                (folder / "complete.json").write_bytes(racing)
            with patch.object(study.subprocess, "Popen", return_value=process), \
                    patch.object(study.runtime, "check_time"), \
                    patch.object(study.runtime, "stop", side_effect=stop) as stopped, \
                    patch.object(study.time, "monotonic", side_effect=[0, 1000]):
                with self.assertRaisesRegex(StudyError, "watchdog"):
                    study._run_process(output, folder, {"protocol": p})
            stopped.assert_called_once_with(process)
            self.assertFalse((folder / "complete.json").exists())
            self.assertEqual((folder / "unaccepted-complete.json").read_bytes(), racing)

    def test_retry_requires_explicit_acknowledgment_and_preserves_previous_bytes(self):
        with fixture() as (_, output, _, _, config, _):
            cell = config["order"][0]
            first = study._attempt(output, cell, resume_interrupted=False)
            (first / "partial.log").write_text("artificial interrupted evidence\n")
            old = {p.name: p.read_bytes() for p in first.iterdir()}
            with self.assertRaisesRegex(StudyError, "Interrupted attempt"):
                study._attempt(output, cell, resume_interrupted=False)
            second = study._attempt(output, cell, resume_interrupted=True)
            self.assertEqual(second.name, "attempt-0002")
            self.assertEqual(old, {p.name: p.read_bytes() for p in first.iterdir()})
            ack = read_json(second / "attempt.json")
            self.assertTrue(ack["explicit_resume_acknowledgment"])
            write_json(second / "complete.json", {"software_fixture": True})
            with self.assertRaisesRegex(StudyError, "Cannot rerun"):
                study._attempt(output, cell, resume_interrupted=True)

    def test_export_retains_interrupted_evidence_and_excludes_model_work_tree(self):
        with fixture() as (_, output, _, _, config, _):
            cell = config["order"][0]
            folder = study._attempt(output, cell, resume_interrupted=False)
            (folder / "partial.log").write_text("artificial interruption\n")
            (output / "runtime-work").mkdir()
            (output / "runtime-work/model.fake").write_bytes(b"not real model weights")
            archive = study.export(output)
            with zipfile.ZipFile(archive) as returned:
                self.assertIsNone(returned.testzip())
                self.assertIn(str((folder / "partial.log").relative_to(output)), returned.namelist())
                self.assertIn("settings.json", returned.namelist())
                self.assertFalse(any(name.startswith("runtime-work/") for name in returned.namelist()))


if __name__ == "__main__":
    unittest.main()
