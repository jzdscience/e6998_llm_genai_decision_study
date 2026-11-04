"""Local artificial runtime fixtures; no CUDA/model service is started."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from decision_study.common import StudyError, fingerprint, read_json, write_json
from decision_study.storage import load_predictions
from decision_study.v2 import identity, runner_v2


class RuntimeIdentityTests(unittest.TestCase):
    def test_frozen_core_and_analysis_edits_are_each_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            target = project / "decision_study"
            target.mkdir()
            for name in (*identity.CORE_FILES, "analysis.py"):
                shutil.copyfile(identity.PROJECT / "decision_study" / name, target / name)
            with patch.object(identity, "PROJECT", project):
                identity.verify_frozen_source()
                for name in ("runtime.py", "analysis.py"):
                    original = (target / name).read_bytes()
                    (target / name).write_bytes(original + b"\n# artificial mutation\n")
                    with self.assertRaisesRegex(StudyError, "Frozen v1 source changed"):
                        identity.verify_frozen_source()
                    (target / name).write_bytes(original)
                identity.verify_frozen_source()

    def test_job_path_rejects_unknown_and_traversal_names(self):
        for name in ("../../../escape", "startlux9b", "holdout_wrong"):
            with self.assertRaises(StudyError):
                identity.job_root(Path("output"), name)


class RuntimeLifecycleTests(unittest.TestCase):
    @staticmethod
    def source(root):
        source = root / "v1"
        for name in ("lock.json", "installations.json", "doctor.json"):
            write_json(source / "runtime" / name, {"fixture": name})
        return source

    def test_allocation_acknowledgment_and_separate_output_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            with patch.object(runner_v2, "ensure_session") as session, patch.object(runner_v2, "verify_data"):
                with self.assertRaisesRegex(StudyError, "new-allocation"):
                    runner_v2.initialize_runtime(source, root / "out", session_hours_left=1)
                session.assert_not_called()
                with self.assertRaisesRegex(StudyError, "separate output"):
                    runner_v2.initialize_runtime(source, source / "nested-v2", session_hours_left=1, new_allocation=True)
                self.assertFalse((source / "nested-v2").exists())

    def test_missing_session_cannot_implicitly_create_an_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(runner_v2, "Budget") as budget, patch.object(runner_v2.runtime, "doctor") as doctor:
                with self.assertRaisesRegex(StudyError, "actual remaining allocation"):
                    with runner_v2.runtime_context(root / "v1", root / "v2", tool_preflight=False):
                        self.fail("Should not enter an uninitialized runtime")
                budget.assert_not_called()
                doctor.assert_not_called()
            self.assertFalse((root / "v2").exists())

    def test_initialized_context_restores_environment_after_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            output = root / "v2"
            work = (root / "shared-installed-work").resolve()
            with patch.object(runner_v2, "verify_data"), patch.object(runner_v2.runtime, "_work_dir", return_value=work):
                runner_v2.initialize_runtime(source, output, session_hours_left=1, new_allocation=True)
            original_files = {p.name: p.read_bytes() for p in (source / "runtime").iterdir()}
            with patch.dict(os.environ, {"DECISION_STUDY_RUNTIME_DIR": "previous", "TEST_V2_SENTINEL": "preserved"}), patch.object(runner_v2.runtime, "doctor"):
                before = dict(os.environ)
                with self.assertRaisesRegex(RuntimeError, "deliberate test"):
                    with runner_v2.runtime_context(source, output, tool_preflight=False) as (control, budget):
                        self.assertEqual(os.environ["DECISION_STUDY_RUNTIME_DIR"], str(work))
                        budget.check()
                        os.environ["TEST_V2_TRANSIENT"] = "remove on exit"
                        raise RuntimeError("deliberate test")
                self.assertEqual(dict(os.environ), before)
            self.assertTrue(read_json(output / "runtime-control/budget.json")["sessions"][-1]["closed"])
            self.assertEqual(original_files, {p.name: p.read_bytes() for p in (source / "runtime").iterdir()})

    def test_changed_runtime_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            output = root / "v2"
            with patch.object(runner_v2, "verify_data"), patch.object(runner_v2.runtime, "_work_dir", return_value=root / "work"):
                runner_v2.initialize_runtime(source, output, session_hours_left=1, new_allocation=True)
            write_json(output / "runtime-control/runtime/lock.json", {"tampered": True})
            with patch.object(runner_v2.runtime, "doctor") as doctor:
                with self.assertRaisesRegex(StudyError, "identity changed"):
                    with runner_v2.runtime_context(source, output, tool_preflight=False):
                        self.fail("Changed identity must not enter runtime")
                doctor.assert_not_called()


class PredictionJobTests(unittest.TestCase):
    @staticmethod
    def example(sid, split="holdout"):
        return {"sample_id": sid, "task": "squad", "split": split, "group_id": "fixture-article",
                "state": "ARTIFICIAL software fixture", "instructions": "Classify", "labels": ["answerable", "unanswerable"],
                "criteria": {"answerable": "Supported", "unanswerable": "Unsupported"}}

    def test_prediction_configuration_never_opens_sealed_gold(self):
        example = self.example("holdout-1")
        with patch.object(runner_v2.data_v2, "verify", side_effect=AssertionError("sealed gold access")), \
             patch.object(runner_v2.data_v2, "verify_fit", return_value={"manifest_hash": "fixture"}), \
             patch.object(runner_v2.data_v2, "load_examples", return_value=[example]), \
             patch.object(runner_v2, "source_identity", return_value={"fixture": True}), \
             patch.object(runner_v2, "protocol", return_value={"fixture": True}), \
             patch.object(runner_v2, "verify_runtime_model", return_value={"fixture": True}), \
             patch.object(runner_v2, "protocol_config", return_value={"fixture": True}):
            result = runner_v2.prediction_config(Path("v1"), Path("v2"), "holdout_qwen", Path("control"))
        self.assertEqual(result["expected_predictions"], 1)

    def test_completed_job_resumes_without_model_calls_and_refuses_changed_config(self):
        examples = [self.example("fixture-1"), self.example("fixture-2")]
        dev = [self.example("fixture-development", "development")]
        config = {"protocol": "solo-v2", "job": "holdout_qwen", "model": "qwen", "runtime": {"model_revision": "fixture"}}
        calls = []
        class Client:
            model = "qwen"
            def predict(self, example):
                calls.append(example["sample_id"])
                return {"sample_id": example["sample_id"], "model": "qwen", "task": "squad", "split": example["split"],
                        "status": "ok", "prediction": "answerable", "probabilities": None, "attempts": [{}], "is_test_fixture": True}
        @contextmanager
        def services(*args, **kwargs):
            yield {"qwen": "http://fixture.invalid"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1, output = root / "v1", root / "out"
            write_json(v1 / "smoke/qwen.json", {"serial": [Client().predict(dev[0])]})
            calls.clear()
            with patch.object(runner_v2, "prediction_config", return_value=config), \
                 patch.object(runner_v2.data_v2, "load_examples", return_value=examples), \
                 patch.object(runner_v2, "v1_examples", return_value=dev), \
                 patch.object(runner_v2.runtime, "token_preflight_examples", return_value={"status": "ok"}) as tokenizer, \
                 patch.object(runner_v2.runtime, "managed_services", services), \
                 patch.object(runner_v2, "ModelClient", return_value=Client()):
                first = runner_v2.run_job(v1, output, "holdout_qwen", control=root / "control", budget=Mock())
                second = runner_v2.run_job(v1, output, "holdout_qwen", control=root / "control", budget=Mock())
                self.assertEqual(first, second)
                self.assertEqual(calls, [dev[0]["sample_id"], *[e["sample_id"] for e in examples]])
                self.assertEqual(tokenizer.call_count, 1)
                self.assertFalse(first["scored_holdout"])
            job = identity.job_root(output, "holdout_qwen")
            saved = load_predictions(job, "qwen")
            with patch.object(runner_v2, "prediction_config", return_value={**config, "changed": True}), \
                 patch.object(runner_v2.data_v2, "load_examples", return_value=examples):
                with self.assertRaisesRegex(StudyError, "Configuration changed"):
                    runner_v2.run_job(v1, output, "holdout_qwen", control=root / "control", budget=Mock())
            self.assertEqual(saved, load_predictions(job, "qwen"))


class ExportTests(unittest.TestCase):
    def test_export_excludes_runtime_payloads_symlinks_and_old_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            write_json(output / "data/fixture.json", {"software_test_fixture": True})
            for folder in ("work", "envs", ".venv", "weights", "cache", "__pycache__", ".git"):
                (output / folder).mkdir(parents=True)
                (output / folder / "should-not-export.bin").write_bytes(b"NOT AN ARTIFACT")
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.txt").write_text("DO NOT FOLLOW")
            (output / "linked-file.txt").symlink_to(outside / "secret.txt")
            (output / "linked-directory").symlink_to(outside, target_is_directory=True)
            write_json(output / "bundle_manifest.json", {"stale": True})
            (output / ".active.lock").write_text("")
            (output / "partial.tmp").write_text("partial")
            archive_path = runner_v2.export(output)
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                self.assertEqual(names.count("bundle_manifest.json"), 1)
                self.assertIn("data/fixture.json", names)
                self.assertFalse(any("should-not-export" in name or "linked-" in name for name in names))
                self.assertNotIn(".active.lock", names)
                self.assertNotIn("partial.tmp", names)
                manifest = json.loads(archive.read("bundle_manifest.json"))
                self.assertEqual(set(manifest["files"]), set(names) - {"bundle_manifest.json"})
                for name, spec in manifest["files"].items():
                    self.assertEqual(len(archive.read(name)), spec["bytes"])


if __name__ == "__main__":
    unittest.main()
