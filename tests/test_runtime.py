import copy
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from decision_study import runtime

PROJECT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.defaults = json.loads((PROJECT / "configs/runtime_defaults.json").read_text())
        self.installed = {"models": {m: {"model_path": f"/models/{m}"} for m in runtime.MODELS},
                          "environments": {n: {"python": f"/env/{n}/bin/python"} for n in
                                           ["kev", "startlux-reference", "vllm-0.31.0", "vllm-0.19.1"]}}

    def tearDown(self):
        self.temp.cleanup()

    def test_requires_full_sha(self):
        for value in ("main", "139fdd94", "latest", None):
            with self.assertRaises(ValueError):
                runtime._valid_sha(value)
        self.assertEqual(runtime._valid_sha("a" * 40), "a" * 40)

    def test_port_check_allows_restart_after_server_active_close(self):
        # Active close leaves the server's accepted connection in TIME_WAIT even
        # though its listening socket has closed. This reproduces serial ->
        # concurrency smoke startup on a single fixed service port.
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                connection, _ = listener.accept()
                with connection:
                    connection.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
                # Peer closes only after observing the server's FIN.
        runtime._check_port(port)
        # The production server can immediately claim the same address too.
        with socket.socket() as replacement:
            replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            replacement.bind(("127.0.0.1", port))
            replacement.listen(1)

    def test_port_check_still_rejects_live_listeners(self):
        for address in ("127.0.0.1", "0.0.0.0"):
            with self.subTest(address=address), socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((address, 0))
                listener.listen(1)
                port = listener.getsockname()[1]
                with self.assertRaisesRegex(RuntimeError, "already occupied"):
                    runtime._check_port(port)
                # The probe sends no application data, and the unrelated
                # listener remains available for subsequent real requests.
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    self.assertEqual(connection.recv(1), b"")
                with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                    client.sendall(b"x")
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(2)
                        self.assertEqual(connection.recv(1), b"x")

    def test_lock_verifies_every_source_and_rejects_mutation(self):
        def fetch(url):
            return {"sha": url.rsplit("/", 1)[-1]}
        with patch.object(runtime, "_get_json", side_effect=fetch) as get:
            lock = runtime.freeze_lock(PROJECT, self.out)
            self.assertEqual(len(lock["verified_sources"]), 6)
            self.assertEqual(get.call_count, 6)
            get.reset_mock()
            runtime.freeze_lock(PROJECT, self.out)
            get.assert_not_called()
        lock["models"]["kev"]["revision"] = "b" * 40
        runtime._write(self.out / "runtime/lock.json", lock)
        with self.assertRaises(RuntimeError):
            runtime.freeze_lock(PROJECT, self.out)

    def test_source_mismatch_leaves_no_lock(self):
        with patch.object(runtime, "_get_json", return_value={"sha": "wrong"}):
            with self.assertRaises(RuntimeError):
                runtime.freeze_lock(PROJECT, self.out)
        self.assertFalse((self.out / "runtime/lock.json").exists())

    def test_api_quota_preserves_immutable_commit(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'[project]\nname="kev"\n'
        error = runtime.urllib.error.HTTPError("https://api.github.com/x", 403, "rate limit", {}, None)
        with patch.object(runtime, "_get_json", side_effect=error), patch.object(runtime.urllib.request, "urlopen", return_value=response) as fetch:
            result = runtime._verify_git_revision("org/repo", "a" * 40)
        self.assertEqual(result["sha"], "a" * 40)
        self.assertIn("/" + "a" * 40 + "/pyproject.toml", fetch.call_args.args[0])
        self.assertEqual(result["verification"], "pinned_raw_file")

    def test_per_model_profile_selection_and_prediction_lock(self):
        runtime.select_profile(self.out, "kev", "kev_eager")
        runtime.select_profile(self.out, "startlux", "vllm019")
        self.assertEqual(runtime._resolve_profile(self.out, "kev"), "kev_eager")
        self.assertEqual(runtime._resolve_profile(self.out, "startlux"), "vllm019")
        self.assertEqual(runtime._resolve_profile(self.out, "qwen"), "primary")
        runtime._write(self.out / "predictions/kev/config.json", {})
        with self.assertRaises(RuntimeError):
            runtime.select_profile(self.out, "kev", "primary")
        runtime.select_profile(self.out, "kev", "kev_eager")

    def test_explicit_memory_setting_changes_command_and_freezes(self):
        runtime.select_memory_fraction(self.out, "qwen", 0.8)
        cmd = runtime._service_specs(PROJECT, self.out, "qwen", "primary", self.defaults, self.installed)[0]["command"]
        self.assertEqual(cmd[cmd.index("--gpu-memory-utilization")+1], "0.8")
        runtime._write(self.out / "predictions/qwen/config.json", {})
        with self.assertRaises(RuntimeError):
            runtime.select_memory_fraction(self.out, "qwen", 0.4)
        with self.assertRaises(ValueError):
            runtime.select_memory_fraction(self.out, "kev", 0.8)

    def test_budget_deadline_clamps_runtime_timeout(self):
        with patch.object(runtime.time, "time", return_value=1000), patch.dict(os.environ, {"DECISION_STUDY_DEADLINE_EPOCH":"1010"}):
            self.assertEqual(runtime._timeout_seconds(1800), 10)
        with patch.object(runtime.time, "time", return_value=1011), patch.dict(os.environ, {"DECISION_STUDY_DEADLINE_EPOCH":"1010"}):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                runtime._timeout_seconds(1800)

    def test_long_running_operation_stops_group_at_budget_deadline(self):
        proc = MagicMock(pid=12345)
        proc.communicate.side_effect = runtime.subprocess.TimeoutExpired(["python", "probe"], 1)
        with patch.object(runtime.subprocess, "Popen", return_value=proc), patch.object(runtime, "_timeout_seconds", return_value=1), patch.object(runtime.os, "killpg") as stop:
            with self.assertRaisesRegex(RuntimeError, "budget"):
                runtime._run(["python", "probe"], self.out, "probe.log")
        stop.assert_called_once_with(12345, runtime.signal.SIGTERM)
        self.assertEqual(proc.communicate.call_args.kwargs["timeout"], 1)

    def test_failed_subprocess_points_to_its_actual_stderr_log(self):
        import sys
        with self.assertRaises(runtime.RuntimeCommandError) as caught:
            runtime._run([sys.executable, "-c", "import sys; print('fatal error: Python.h absent', file=sys.stderr); sys.exit(1)"],
                         self.out, "reference-kev.log")
        log = self.out / "runtime/logs/reference-kev.log"
        self.assertEqual(caught.exception.returncode, 1)
        self.assertIn(str(log), str(caught.exception))
        self.assertEqual(caught.exception.log_path, str(log))
        self.assertIn("fatal error: Python.h absent", log.read_text())

    def test_triton_build_probe_calls_driver_and_keeps_compiler_cause(self):
        import sys
        from types import SimpleNamespace
        from scripts.model_probe import triton_build_check
        get_target = MagicMock(return_value="cuda:89")
        modules = {"triton": SimpleNamespace(), "triton.runtime": SimpleNamespace(
            driver=SimpleNamespace(active=SimpleNamespace(get_current_target=get_target)))}
        with patch.dict(sys.modules, modules):
            result = triton_build_check()
            self.assertEqual(result["target"], "cuda:89")
            get_target.assert_called_once()
            failure = RuntimeError("compiler diagnostic")
            get_target.side_effect = failure
            with self.assertRaisesRegex(RuntimeError, "compiler/linker") as caught:
                triton_build_check()
            self.assertIs(caught.exception.__cause__, failure)

    def test_sanitized_per_process_controls(self):
        with patch.dict(os.environ, {"KEV_TEMPERATURE": "999", "STARTLUX_PREFIX_BIDIR_ALLOW_VERSION": "1",
                                     "VLLM_PLUGINS": "bad", "DECISION_STUDY_RUNTIME_DIR": str(self.out / "local")}):
            env = runtime._env(self.out)
            self.assertNotIn("KEV_TEMPERATURE", env)
            self.assertNotIn("STARTLUX_PREFIX_BIDIR_ALLOW_VERSION", env)
            self.assertNotIn("VLLM_PLUGINS", env)
            self.assertEqual(env["HF_HOME"], str((self.out / "local/cache/huggingface").resolve()))

    def test_instant_engine_attention_controls_and_independent_group(self):
        specs = runtime._service_specs(PROJECT, self.out, "startlux", "primary", self.defaults, self.installed)
        self.assertEqual(len(specs), 2)
        engine, frontend = specs
        self.assertTrue(engine["plugin_required"])
        self.assertEqual(engine["env"]["STARTLUX_PREFIX_BIDIR"], "1")
        for flag in ("--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--logprobs-mode", "--max-logprobs"):
            self.assertIn(flag, engine["command"])
        self.assertNotIn("--speculative-config", engine["command"])
        self.assertNotIn("--launch-vllm", frontend["command"])
        self.assertIn("--no-none-check", frontend["command"])
        self.assertNotIn("STARTLUX_PREFIX_BIDIR", frontend["env"])

    def test_qwen_does_not_enable_startlux_patch(self):
        spec = runtime._service_specs(PROJECT, self.out, "qwen", "primary", self.defaults, self.installed)[0]
        self.assertNotIn("STARTLUX_PREFIX_BIDIR", spec["env"])
        self.assertNotIn("--logprobs-mode", spec["command"])
        self.assertIn("--dtype", spec["command"])

    def test_explicit_fallbacks(self):
        eager = runtime._service_specs(PROJECT, self.out, "kev", "kev_eager", self.defaults, self.installed)[0]
        self.assertEqual(eager["env"]["KEV_FUSED"], "0")
        self.assertEqual(eager["env"]["KEV_CUDA_GRAPHS"], "0")
        self.assertEqual(eager["env"]["KEV_PREFIX_CACHE"], "0")
        self.assertIn("/__decision_study_missing_checkpoint__", eager["command"])
        reference = runtime._service_specs(PROJECT, self.out, "startlux", "reference", self.defaults, self.installed)
        self.assertEqual(len(reference), 1)
        self.assertIn("transformers", reference[0]["command"])
        fallback = runtime._service_specs(PROJECT, self.out, "qwen", "vllm019", self.defaults, self.installed)[0]
        self.assertIn("vllm-0.19.1", fallback["command"][0])

    def test_plugin_installed_is_not_active(self):
        process = MagicMock()
        process.poll.return_value = None
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.read.return_value = b'{}'
        log = self.out / "server.log"
        log.write_text("plugin installed\n")
        spec = {"name": "startlux", "url": "http://localhost:1", "health": "/health", "plugin_required": True}
        with patch.object(runtime.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "did not activate"):
                runtime._wait_ready(process, spec, log, 2)
        log.write_text("prefix-bidirectional prompt attention ON for /models/startlux\n")
        with patch.object(runtime.urllib.request, "urlopen", return_value=response):
            self.assertEqual(runtime._wait_ready(process, spec, log, 2), {})

    def test_cpu_doctor_works_without_cuda(self):
        with patch.object(runtime.subprocess, "run", side_effect=FileNotFoundError()):
            result = runtime.doctor(self.out)
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["gpu_ready"])
            with self.assertRaises(RuntimeError):
                runtime.doctor(self.out, require_gpu=True)

    def test_optional_network_check_does_not_block_offline_cpu(self):
        failure = {"checked": True, "metadata_endpoints_reachable": False, "endpoints": []}
        with patch.object(runtime.subprocess, "run", side_effect=FileNotFoundError()), patch.object(runtime, "connectivity_report", return_value=failure) as check:
            report = runtime.doctor(self.out, check_network=True)
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["network"]["metadata_endpoints_reachable"])
            check.assert_called_once()
            check.reset_mock()
            runtime.doctor(self.out)
            check.assert_not_called()

    def test_startup_failure_still_stops_owned_group(self):
        lock = {**self.defaults, "lock_hash": "test"}
        installs = {**self.installed, "lock_hash": "test"}
        runtime._write(self.out / "runtime/lock.json", lock)
        runtime._write(self.out / "runtime/installations.json", installs)
        proc = MagicMock(pid=12345)
        with patch.object(runtime, "doctor", return_value={"gpu": [{"memory_mb": 48000}]}), \
             patch.object(runtime, "_verify_environment"), patch.object(runtime, "_check_port"), \
             patch.object(runtime.subprocess, "Popen", return_value=proc), \
             patch.object(runtime, "_wait_ready", side_effect=RuntimeError("startup failed")), \
             patch.object(runtime.os, "killpg") as stop:
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                with runtime.managed_services(PROJECT, self.out, ["qwen"]):
                    self.fail("failed startup must never yield endpoints")
        stop.assert_called_once_with(12345, runtime.signal.SIGTERM)
        proc.wait.assert_called_once_with(timeout=15)
        metadata = runtime._read(self.out / "runtime/last_services.json")
        self.assertIn("stopped_at", metadata)

    def test_three_resident_models_rejected_before_gpu_work(self):
        with patch.object(runtime, "doctor") as doctor:
            with self.assertRaises(ValueError):
                with runtime.managed_services(PROJECT, self.out, ["kev", "startlux", "qwen"]):
                    self.fail("three models must not start")
            doctor.assert_not_called()

    def test_incompatible_cuda_stops_before_checkpoint_download(self):
        lock = {**self.defaults, "lock_hash": "test"}
        python = runtime._work_dir(self.out) / "envs/vllm-0.31.0/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        runtime._write(self.out / "runtime/installations.json", {
            "lock_hash": "test", "models": {}, "environments": {
                "vllm-0.31.0": {"python": str(python), "freeze_sha256": "test"}}})
        with patch.object(runtime, "doctor"), patch.object(runtime, "freeze_lock", return_value=lock), \
             patch.object(runtime.shutil, "which", return_value="/fake/uv"), \
             patch.object(runtime, "_check_python_headers"), \
             patch.object(runtime, "_verify_environment"), \
             patch.object(runtime, "_run", side_effect=RuntimeError("incompatible CUDA wheel")) as run:
            with self.assertRaisesRegex(RuntimeError, "incompatible CUDA"):
                runtime.install_environments(PROJECT, self.out, ["qwen"])
        run.assert_called_once()
        command = [str(x) for x in run.call_args.args[0]]
        self.assertIn("gpu-check", command)
        self.assertNotIn("download", command)

    def _repair_fixture(self):
        lock = {**self.defaults, "lock_hash": "test"}
        python = runtime._work_dir(self.out) / "envs/kev/bin/python"
        python.parent.mkdir(parents=True)
        python.touch()
        pins = "decision-study==0.1.0\npip==25.2\ntorch==2.8.0\ntransformers==5.17.0\nkev @ file:///old/pinned/kev\n"
        resolved = self.out / "runtime/resolved/kev.txt"
        resolved.parent.mkdir(parents=True)
        resolved.write_text(pins)
        runtime._write(self.out / "runtime/installations.json", {
            "lock_hash": "test", "models": {"kev": {"model_path": "/cached/kev"}},
            "environments": {"kev": {"python": str(python), "freeze_sha256": "old"}}})
        return lock, python, pins

    def _fake_install_run(self, pins, calls):
        def run(args, out, log_name, **kwargs):
            command = [str(x) for x in args]
            calls.append(command)
            if "venv" in command:
                python = Path(command[-1]) / "bin/python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.touch()
            if "freeze" in command:
                return pins
            if "gpu-check" in command:
                runtime._write(Path(command[command.index("--result") + 1]), {"status": "ok", "triton_build": "ok"})
            if "download" in command:
                self.fail("Python repair must not invoke checkpoint downloads")
            return ""
        return run

    def test_repair_uses_managed_python_preserves_pins_weights_and_data(self):
        lock, python, pins = self._repair_fixture()
        for relative in ("runtime/work/cache/huggingface/keep", "data/examples.jsonl", "runtime/session.json"):
            path = self.out / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("preserve me")
        calls = []
        headers = {"headers": ["/managed/include/Python.h"], "missing": [], "python_version": "3.12.8"}
        with patch.object(runtime, "doctor"), patch.object(runtime, "freeze_lock", return_value=lock), \
             patch.object(runtime.shutil, "which", return_value="/fake/uv"), \
             patch.object(runtime, "_checkout", return_value=Path("/locked/kev")), \
             patch.object(runtime, "_check_python_headers", return_value=headers), \
             patch.object(runtime, "_verify_environment"), \
             patch.object(runtime, "_run", side_effect=self._fake_install_run(pins, calls)):
            result = runtime.repair_python(PROJECT, self.out, ["kev"], "3.12.8")
        create = next(c for c in calls if "venv" in c)
        self.assertIn("--managed-python", create)
        self.assertIn("--clear", create)
        self.assertEqual(create[create.index("--python") + 1], "3.12.8")
        self.assertEqual(result["environments"], ["kev"])
        self.assertFalse(result["checkpoints_downloaded"])
        portable = (self.out / "runtime/resolved/kev-portable.txt").read_text()
        self.assertNotIn("decision-study", portable)
        self.assertNotIn("file:", portable)
        self.assertIn("torch==2.8.0", portable)
        self.assertIn("transformers==5.17.0", portable)
        marker = runtime._read(self.out / "runtime/repair-kev.json")
        self.assertEqual(marker["status"], "complete")
        self.assertEqual((Path(marker["archive"]) / "kev.txt").read_text(), pins)
        self.assertEqual(list(Path(marker["archive"]).rglob("python")), [])
        for relative in ("runtime/work/cache/huggingface/keep", "data/examples.jsonl", "runtime/session.json"):
            self.assertEqual((self.out / relative).read_text(), "preserve me")

    def test_pending_repair_forces_rebuild_after_interruption(self):
        lock, python, pins = self._repair_fixture()
        calls = []
        headers = {"headers": ["/managed/include/Python.h"], "missing": [], "python_version": "3.12.8"}
        with patch.object(runtime, "doctor"), patch.object(runtime, "freeze_lock", return_value=lock), \
             patch.object(runtime.shutil, "which", return_value="/fake/uv"), \
             patch.object(runtime, "_checkout", return_value=Path("/locked/kev")), \
             patch.object(runtime, "_check_python_headers", return_value=headers), \
             patch.object(runtime, "_verify_environment"):
            with patch.object(runtime, "_run", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.repair_python(PROJECT, self.out, ["kev"])
            self.assertEqual(runtime._read(self.out / "runtime/repair-kev.json")["status"], "pending")
            self.assertEqual((self.out / "runtime/resolved/kev.txt").read_text(), pins)
            with patch.object(runtime, "_run", side_effect=self._fake_install_run(pins, calls)):
                runtime.install_environments(PROJECT, self.out, ["kev"], download_models=False)
        self.assertIn("--clear", next(c for c in calls if "venv" in c))
        self.assertIsNone(runtime._pending_repair(self.out, "kev"))

    def test_missing_header_stops_before_heavy_packages(self):
        lock = {**self.defaults, "lock_hash": "test"}
        calls = []
        with patch.object(runtime, "doctor"), patch.object(runtime, "freeze_lock", return_value=lock), \
             patch.object(runtime.shutil, "which", return_value="/fake/uv"), \
             patch.object(runtime, "_checkout") as checkout, \
             patch.object(runtime, "_check_python_headers", side_effect=RuntimeError("Python.h missing")), \
             patch.object(runtime, "_run", side_effect=self._fake_install_run("", calls)):
            with self.assertRaisesRegex(RuntimeError, "Python.h"):
                runtime.install_environments(PROJECT, self.out, ["kev"])
        self.assertEqual(len(calls), 1)
        self.assertIn("--managed-python", calls[0])
        self.assertNotIn("--clear", calls[0])
        checkout.assert_not_called()

    def test_header_probe_preserves_actual_log_and_failure(self):
        failure = runtime.RuntimeCommandError(1, ["python", "-c", "check"], "/logs/headers.log")
        with patch.object(runtime, "_run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "Python.h") as caught:
                runtime._check_python_headers(self.out, "/env/python", "kev")
        self.assertEqual(caught.exception.log_path, "/logs/headers.log")
        self.assertIs(caught.exception.__cause__, failure)

    def test_normal_install_never_deletes_unregistered_existing_environment(self):
        lock = {**self.defaults, "lock_hash": "test"}
        keep = runtime._work_dir(self.out) / "envs/kev/important.txt"
        keep.parent.mkdir(parents=True)
        keep.write_text("keep")
        with patch.object(runtime, "doctor"), patch.object(runtime, "freeze_lock", return_value=lock), \
             patch.object(runtime.shutil, "which", return_value="/fake/uv"), patch.object(runtime, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "repair-python"):
                runtime.install_environments(PROJECT, self.out, ["kev"])
        run.assert_not_called()
        self.assertEqual(keep.read_text(), "keep")

    def test_repair_rejects_formal_predictions_in_shared_environment(self):
        lock = {**self.defaults, "lock_hash": "test"}
        runtime._write(self.out / "predictions/qwen/config.json", {})
        with patch.object(runtime, "freeze_lock", return_value=lock), patch.object(runtime, "_run") as run:
            with self.assertRaisesRegex(RuntimeError, "qwen.*formal predictions"):
                runtime.repair_python(PROJECT, self.out, ["startlux"])
        run.assert_not_called()
        self.assertFalse((self.out / "runtime/repair-vllm-0.31.0.json").exists())

    def test_repair_rejects_dependency_upgrade(self):
        original = "torch==2.8.0\ntransformers==5.17.0\ndecision_study==0.1.0\n"
        runtime._check_preserved_pins(original, "torch==2.8.0\ntransformers==5.17.0\n")
        with self.assertRaisesRegex(RuntimeError, "torch"):
            runtime._check_preserved_pins(original, "torch==2.9.0\ntransformers==5.17.0\n")

    def test_repair_cli_keeps_existing_session_deadline_and_company_budget(self):
        from decision_study.cli import main
        from decision_study.budget import ensure_session
        session = ensure_session(self.out, hours_left=1)
        def repair(project, output, models, version):
            self.assertEqual(models, ["kev"])
            self.assertEqual(version, "3.12.8")
            self.assertEqual(float(os.environ["DECISION_STUDY_DEADLINE_EPOCH"]), session["deadline_epoch"])
            return {"status": "complete"}
        with patch.object(runtime, "repair_python", side_effect=repair):
            status = main(["repair-python", "--models", "kev", "--managed-python-version", "3.12.8",
                           "--cost-mode", "provided", "--output-dir", str(self.out)])
        self.assertEqual(status, 0)
        self.assertEqual(runtime._read(self.out / "runtime/session.json")["deadline_epoch"], session["deadline_epoch"])

    def _git_build_fixture(self):
        path = self.out / "checkout"
        (path / "kev").mkdir(parents=True)
        (path / "kev/base.py").write_bytes(b"VALUE = 1\n")
        (path / "kev/file with spaces.py").write_bytes(b"SECOND = 2\n")
        def git(*args):
            return runtime.subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
        git("init", "-q")
        git("add", "kev")
        git("-c", "user.name=Runtime Test", "-c", "user.email=runtime@example.invalid", "commit", "-qm", "fixture")
        revision = git("rev-parse", "HEAD")
        target = runtime._work_dir(self.out) / "sources" / ("jaredpalmer--kev-" + revision[:12])
        target.parent.mkdir(parents=True)
        path.rename(target)
        copies = target / "build/lib/kev"
        copies.mkdir(parents=True)
        for original in (target / "kev").glob("*.py"):
            (copies / original.name).write_bytes(original.read_bytes())
        return target, revision

    def test_checkout_removes_only_identical_build_copies_and_handles_spaces(self):
        path, revision = self._git_build_fixture()
        self.assertEqual(runtime._checkout(self.out, "jaredpalmer/kev", revision), path)
        self.assertFalse((path / "build").exists())
        self.assertEqual((path / "kev/base.py").read_bytes(), b"VALUE = 1\n")
        self.assertEqual((path / "kev/file with spaces.py").read_bytes(), b"SECOND = 2\n")
        # Calling again is harmless, as is a second subsequent wheel build.
        runtime._checkout(self.out, "jaredpalmer/kev", revision)
        (path / "build/lib/kev").mkdir(parents=True)
        (path / "build/lib/kev/base.py").write_bytes(b"VALUE = 1\n")
        runtime._checkout(self.out, "jaredpalmer/kev", revision)
        self.assertFalse((path / "build").exists())

    def test_checkout_validates_all_paths_before_removing_any_copy(self):
        path, revision = self._git_build_fixture()
        changed = path / "build/lib/kev/file with spaces.py"
        changed.write_bytes(b"SECOND = 999\n")
        with self.assertRaisesRegex(RuntimeError, "differs from pinned HEAD"):
            runtime._checkout(self.out, "jaredpalmer/kev", revision)
        self.assertEqual((path / "build/lib/kev/base.py").read_bytes(), b"VALUE = 1\n")
        self.assertEqual(changed.read_bytes(), b"SECOND = 999\n")

    def test_checkout_preserves_tracked_changes_and_unknown_python(self):
        path, revision = self._git_build_fixture()
        unexpected = path / "build/lib/kev/unexpected.py"
        unexpected.write_bytes(b"UNKNOWN = True\n")
        with self.assertRaisesRegex(RuntimeError, "no regular tracked source"):
            runtime._checkout(self.out, "jaredpalmer/kev", revision)
        self.assertTrue((path / "build/lib/kev/base.py").exists())
        unexpected.unlink()
        (path / "kev/base.py").write_bytes(b"USER_EDIT = True\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected path"):
            runtime._checkout(self.out, "jaredpalmer/kev", revision)
        self.assertEqual((path / "kev/base.py").read_bytes(), b"USER_EDIT = True\n")
        self.assertTrue((path / "build/lib/kev/base.py").exists())

    def test_checkout_rejects_symlink_copy_and_symlink_ancestor(self):
        path, revision = self._git_build_fixture()
        candidate = path / "build/lib/kev/base.py"
        candidate.unlink()
        candidate.symlink_to(path / "kev/base.py")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            runtime._checkout(self.out, "jaredpalmer/kev", revision)
        self.assertTrue(candidate.is_symlink())
        candidate.unlink()
        (path / "build/lib/kev/file with spaces.py").unlink()
        (path / "build/lib/kev").rmdir()
        (path / "build/lib/kev").symlink_to(path / "kev", target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            runtime._remove_verified_kev_build_copies(self.out, path, "jaredpalmer/kev", revision,
                                                     "?? build/lib/kev/base.py\0")
        self.assertEqual((path / "kev/base.py").read_bytes(), b"VALUE = 1\n")

    def test_checkout_wrong_revision_never_removes_build_copies(self):
        path, revision = self._git_build_fixture()
        wrong = "f" * 40
        target = path.with_name("jaredpalmer--kev-" + wrong[:12])
        path.rename(target)
        with self.assertRaisesRegex(RuntimeError, "Source checkout changed"):
            runtime._checkout(self.out, "jaredpalmer/kev", wrong)
        self.assertTrue((target / "build/lib/kev/base.py").exists())


if __name__ == "__main__":
    unittest.main()
