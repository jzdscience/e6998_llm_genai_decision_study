import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_study import runtime
from decision_study.common import read_json, write_json
from scripts import repair_startlux_reference as repair

PROJECT = Path(__file__).resolve().parents[1]


class StartLuxDependencyRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name).resolve()
        defaults = read_json(PROJECT / "configs/runtime_defaults.json")
        self.lock = {**defaults, "defaults_hash": runtime._hash(defaults), "lock_hash": runtime._hash(defaults)}
        write_json(self.out / "runtime/lock.json", self.lock)
        self.python = runtime._work_dir(self.out) / "envs/startlux-reference/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.touch()
        self.before = ("decision-study==0.1.0\npip==26.2.2\ntorch==2.15.0\ntransformers==5.8.1\n"
                       "startlux-decide @ file:///pinned/startlux\n")
        self.frozen = self.before
        self.resolved = self.out / "runtime/resolved/startlux-reference.txt"
        runtime._write_text(self.resolved, self.before)
        self.installations = {"lock_hash": self.lock["lock_hash"], "models": {"kev": {"revision": "locked"}},
                              "environments": {
                                  "kev": {"python": "/existing/kev/python", "freeze_sha256": "kev-before"},
                                  "vllm-0.31.0": {"python": "/existing/vllm/python", "freeze_sha256": "vllm-before"},
                                  "startlux-reference": {"python": str(self.python),
                                                         "freeze_sha256": repair._sha(self.before),
                                                         "python_version": "3.12.8", "managed_python": True}}}
        write_json(self.out / "runtime/installations.json", self.installations)
        # Existing Kev formal records must be permitted and preserved.
        write_json(self.out / "predictions/kev/config.json", {"kept": True})
        write_json(self.out / "predictions/kev/shard-000001.json", {"kept": "formal"})
        self.calls = []
        self.install_effect = None

    def tearDown(self):
        self.temp.cleanup()

    def fake_run(self, args, output, log, **kwargs):
        command = [str(x) for x in args]
        self.calls.append(command)
        self.assertEqual(Path(kwargs["cwd"]), PROJECT)
        if "freeze" in command:
            return self.frozen
        if "install" in command:
            self.assertIn("--no-deps", command)
            self.assertEqual(command[command.index("--only-binary") + 1], ":all:")
            self.assertIn("accelerate==1.15.0", command)
            self.assertIn("psutil==7.2.2", command)
            if self.install_effect:
                self.install_effect()
            self.frozen = self.before + "accelerate==1.15.0\npsutil==7.2.2\n"
        if "gpu-check" in command:
            write_json(Path(command[command.index("--result") + 1]),
                       {"status": "ok", "triton_build": {"status": "ok"}, "library_versions": {"torch": "2.15.0"}})
        return None

    def run_repair(self):
        with patch.object(runtime.shutil, "which", return_value="/fake/uv"), \
             patch.object(runtime, "_run", side_effect=self.fake_run):
            return repair.repair_reference(PROJECT, self.out)

    def test_additive_repair_preserves_kev_vllm_and_repeated_run_is_idempotent(self):
        protected_before = repair._protected(PROJECT, self.out, self.installations)
        write_json(self.out / "smoke/startlux.json", {"passed": True, "old": True})
        result = self.run_repair()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["already_complete"])
        updated = read_json(self.out / "runtime/installations.json")
        self.assertEqual(repair._protected(PROJECT, self.out, updated), protected_before)
        self.assertEqual(updated["environments"]["kev"], self.installations["environments"]["kev"])
        self.assertEqual(updated["environments"]["vllm-0.31.0"], self.installations["environments"]["vllm-0.31.0"])
        self.assertEqual(self.resolved.read_text(), self.frozen)
        self.assertEqual(updated["environments"]["startlux-reference"]["freeze_sha256"], repair._sha(self.frozen))
        self.assertFalse((self.out / "smoke/startlux.json").exists())
        marker = read_json(self.out / "runtime" / repair.MARKER)
        self.assertTrue((Path(marker["archive"]) / "smoke-startlux-before.json").exists())
        install_count = sum("install" in c for c in self.calls)
        self.assertTrue(self.run_repair()["already_complete"])
        self.assertEqual(sum("install" in c for c in self.calls), install_count)

    def test_startlux_formal_records_block_before_mutation(self):
        write_json(self.out / "predictions/startlux/config.json", {})
        with self.assertRaisesRegex(RuntimeError, "after formal predictions"):
            self.run_repair()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.out / "runtime" / repair.MARKER).exists())

    def test_completed_repair_allows_later_qwen_metadata_and_smoke(self):
        self.run_repair()
        installations = read_json(self.out / "runtime/installations.json")
        installations["models"]["qwen"] = {"model_path": "/now-downloaded/qwen"}
        installations["environments"]["vllm-0.31.0"]["gpu_probe"] = {"status": "ok"}
        write_json(self.out / "runtime/installations.json", installations)
        write_json(self.out / "smoke/qwen.json", {"passed": True})
        write_json(self.out / "smoke/startlux.json", {"passed": True, "after_repair": True})
        install_count = sum("install" in c for c in self.calls)
        self.assertTrue(self.run_repair()["already_complete"])
        self.assertEqual(sum("install" in c for c in self.calls), install_count)
        self.assertTrue((self.out / "smoke/startlux.json").exists())

    def test_partial_package_install_can_resume(self):
        def interrupt():
            self.frozen = self.before + "accelerate==1.15.0\n"
            raise RuntimeError("simulated package installation interruption")
        self.install_effect = interrupt
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.run_repair()
        self.assertEqual(read_json(self.out / "runtime" / repair.MARKER)["status"], "pending")
        self.assertEqual(self.resolved.read_text(), self.before)
        self.install_effect = None
        self.assertEqual(self.run_repair()["status"], "complete")

    def test_resume_commit_interrupted_after_freeze_write(self):
        original = repair.write_json
        def stop_metadata(path, value):
            if Path(path) == self.out / "runtime/installations.json":
                raise RuntimeError("interrupted metadata commit")
            return original(path, value)
        with patch.object(repair, "write_json", side_effect=stop_metadata):
            with self.assertRaisesRegex(RuntimeError, "metadata commit"):
                self.run_repair()
        self.assertEqual(self.resolved.read_text(), self.frozen)
        marker = read_json(self.out / "runtime" / repair.MARKER)
        self.assertEqual(marker["status"], "pending")
        install_count = sum("install" in c for c in self.calls)
        self.assertEqual(self.run_repair()["status"], "complete")
        self.assertEqual(sum("install" in c for c in self.calls), install_count)

    def test_resume_commit_interrupted_after_metadata_write(self):
        original = repair.write_json
        def stop_marker(path, value):
            if Path(path).name == repair.MARKER and value.get("status") == "complete":
                raise RuntimeError("interrupted final marker commit")
            return original(path, value)
        with patch.object(repair, "write_json", side_effect=stop_marker):
            with self.assertRaisesRegex(RuntimeError, "final marker"):
                self.run_repair()
        self.assertEqual(read_json(self.out / "runtime" / repair.MARKER)["status"], "pending")
        self.assertEqual(self.run_repair()["status"], "complete")

    def test_changed_old_pin_is_rejected_on_pending_resume(self):
        def interrupt():
            raise RuntimeError("interrupted")
        self.install_effect = interrupt
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_repair()
        self.frozen = self.before.replace("torch==2.15.0", "torch==2.16.0")
        self.install_effect = None
        with self.assertRaisesRegex(RuntimeError, "Existing dependency pins"):
            self.run_repair()
        self.assertEqual(self.resolved.read_text(), self.before)

    def test_foreign_path_and_symlink_refused(self):
        changed = copy.deepcopy(self.installations)
        changed["environments"]["startlux-reference"]["python"] = "/foreign/python"
        write_json(self.out / "runtime/installations.json", changed)
        with self.assertRaisesRegex(RuntimeError, "path missing, changed or symlinked"):
            self.run_repair()
        write_json(self.out / "runtime/installations.json", self.installations)
        env = self.python.parent.parent
        other = env.with_name("reference-real")
        env.rename(other)
        env.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "path missing, changed or symlinked"):
            self.run_repair()
        self.assertEqual(self.calls, [])

    def test_unrelated_env_metadata_change_is_refused_on_resume(self):
        def interrupt():
            raise RuntimeError("interrupted")
        self.install_effect = interrupt
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_repair()
        modified = read_json(self.out / "runtime/installations.json")
        modified["environments"]["vllm-0.31.0"]["freeze_sha256"] = "external-change"
        write_json(self.out / "runtime/installations.json", modified)
        self.install_effect = None
        with self.assertRaisesRegex(RuntimeError, "Unrelated model metadata"):
            self.run_repair()


if __name__ == "__main__":
    unittest.main()
