import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cleanup_legacy_cache.py"
spec = importlib.util.spec_from_file_location("cleanup_legacy_cache", SCRIPT)
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class LegacyCacheCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / "hub"
        self.cache.mkdir()
        self.keepers = self.cache / "models--Qwen--Qwen3.5-4B"
        self.keepers.mkdir()
        (self.keepers / "model.safetensors").write_bytes(b"keep")
        for name in cleanup.APPROVED_REPOSITORIES:
            target = self.cache / name
            target.mkdir()
            (target / "blob").write_bytes(b"old")

    def tearDown(self):
        self.temp.cleanup()

    def test_dry_run_changes_nothing_apply_removes_only_allowlisted_targets(self):
        plan = cleanup.cleanup_cache(self.cache)
        self.assertEqual({t["status"] for t in plan["targets"]}, {"planned"})
        self.assertTrue(all((self.cache / n / "blob").exists() for n in cleanup.APPROVED_REPOSITORIES))
        with patch.object(cleanup, "active_model_processes", return_value=[]):
            done = cleanup.cleanup_cache(self.cache, apply=True)
            again = cleanup.cleanup_cache(self.cache, apply=True)
        self.assertEqual({t["status"] for t in done["targets"]}, {"deleted"})
        self.assertEqual({t["status"] for t in again["targets"]}, {"missing"})
        self.assertEqual((self.keepers / "model.safetensors").read_bytes(), b"keep")

    def test_target_symlink_rejected_before_any_deletion(self):
        target = self.cache / cleanup.APPROVED_REPOSITORIES[-1]
        (target / "blob").unlink()
        target.rmdir()
        target.symlink_to(self.keepers, target_is_directory=True)
        with patch.object(cleanup, "active_model_processes", return_value=[]):
            with self.assertRaises(ValueError):
                cleanup.cleanup_cache(self.cache, apply=True)
        self.assertTrue((self.cache / cleanup.APPROVED_REPOSITORIES[0] / "blob").exists())
        self.assertEqual((self.keepers / "model.safetensors").read_bytes(), b"keep")

    def test_root_symlink_rejected_and_internal_snapshot_links_not_followed(self):
        alias = self.root / "alias"
        alias.mkdir()
        (alias / "hub").symlink_to(self.cache, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup.cleanup_cache(alias / "hub")
        external = self.root / "outside"
        external.mkdir()
        (external / "important").write_bytes(b"preserve")
        (self.cache / cleanup.APPROVED_REPOSITORIES[0] / "snapshot").symlink_to(external, target_is_directory=True)
        with patch.object(cleanup, "active_model_processes", return_value=[]):
            cleanup.cleanup_cache(self.cache, apply=True)
        self.assertEqual((external / "important").read_bytes(), b"preserve")

    def test_active_model_job_blocks_apply_and_missing_cache_is_idempotent(self):
        with patch.object(cleanup, "active_model_processes", return_value=[{"pid":123,"family":"Kev"}]):
            with self.assertRaises(RuntimeError):
                cleanup.cleanup_cache(self.cache, apply=True)
        self.assertTrue(all((self.cache / n / "blob").exists() for n in cleanup.APPROVED_REPOSITORIES))
        absent = self.root / "missing" / "hub"
        self.assertEqual({t["status"] for t in cleanup.cleanup_cache(absent, apply=True)["targets"]}, {"missing"})
        self.assertFalse(absent.exists())

    def test_process_scan_recognizes_model_jobs_without_exposing_arguments(self):
        output = (
            "31001 /opt/env/bin/python scripts/model_probe.py reference --model kev\n"
            "31002 /opt/env/bin/python -m vllm.entrypoints.openai.api_server --api-key secret\n"
            "31003 /bin/bash -c 'echo kev.serve'\n"
            "31004 /opt/env/bin/python other_analysis.py\n"
        )
        with patch.object(cleanup.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            processes = cleanup.active_model_processes()
        self.assertEqual({p["pid"] for p in processes}, {31001,31002})
        self.assertNotIn("secret", str(processes))


if __name__ == "__main__":
    unittest.main()
