"""Runtime recovery tool: guard tests with real processes, files, git and ZIPs, plus a rehearsal.

The rehearsal runs the real CLI in subprocesses against a temporary runtime root. Only external
boundaries are faked (tests/recover_node_runtime_fakes.py): uv, the environment interpreter, pip
inside the real toolkit installer, and Hugging Face downloads. Set RECOVER_SCRIPT to test a copy
(mutation checks). Nothing here touches a GPU, the network or the node.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("RECOVER_SCRIPT") or REPO / "scripts/recover_node_runtime.py")
FAKES = REPO / "tests/recover_node_runtime_fakes.py"
RUNNER = REPO / "tests/recover_node_runtime_runner.py"
COMMIT = "4b4e38ab9c040fe7eb02ae5075e87c595399d980"
EVIDENCE = Path("/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study/records/"
                "bundles/benchmark-complete-20261102-210735-72138989815c.zip")


def load(path, name=None):
    spec = importlib.util.spec_from_file_location(name or f"module_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load(SCRIPT, "recover_node_runtime_tests")
fakes = load(FAKES, "recover_node_runtime_fakes")


def real(path):
    return Path(os.path.realpath(path))


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie still answers kill(0); ask ps for its state
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(out) and not out.startswith("Z")


def kill_quietly(pid):
    for call in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
        try:
            call()
        except (ProcessLookupError, PermissionError):
            pass


def wait_until(predicate, limit=10.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def supervisor(**kwargs):
    options = {"niceness": 0, "poll": 0.01, "term_grace": 2.0, "kill_grace": 2.0}
    options.update(kwargs)
    return tool.Supervisor(**options)


# --------------------------------------------------------------------------- supervisor (real children)

def test_supervisor_captures_output_and_reaps(tmp_path):
    result = supervisor().run([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
                              env=dict(os.environ), timeout=30, capture=True, log_path=tmp_path / "c.log")
    assert (result.returncode, result.stdout, result.stderr.strip()) == (0, "out\n", "err")
    assert "COMMAND" in (tmp_path / "c.log").read_text() and "END outcome=exited" in (tmp_path / "c.log").read_text()


def test_timeout_stops_term_ignoring_grandchild_and_spares_unrelated(tmp_path):
    unrelated = subprocess.Popen(["sleep", "60"], start_new_session=True)
    marker = tmp_path / "grandchild.pid"
    script = f"(trap '' TERM; exec sleep 60) & echo $! > {marker}; wait"
    try:
        with pytest.raises(tool.ChildError, match="Exceeded"):
            supervisor(term_grace=0.5).run(["sh", "-c", script], env=dict(os.environ), timeout=1.0)
        grandchild = int(marker.read_text())
        assert wait_until(lambda: not alive(grandchild), 5), "TERM-ignoring grandchild survived"
        assert unrelated.poll() is None, "an unrelated process was signalled"
    finally:
        unrelated.kill()
        unrelated.wait()


def test_leader_exit_with_running_member_fails_and_member_is_stopped(tmp_path):
    marker = tmp_path / "member.pid"
    with pytest.raises(tool.ChildError, match="group members still running"):
        supervisor().run(["sh", "-c", f"sleep 60 & echo $! > {marker}"], env=dict(os.environ), timeout=30)
    assert wait_until(lambda: not alive(int(marker.read_text())), 5)


def test_exit_observed_after_deadline_is_not_accepted(monkeypatch):
    observed = []
    real_waitid = os.waitid

    def waitid(*args):
        info = real_waitid(*args)
        if info is not None:
            observed.append(True)
        return info

    base = time.monotonic()
    monkeypatch.setattr(tool.os, "waitid", waitid)
    runner = supervisor(clock=lambda: base + (1000 if observed else time.monotonic() - base))
    with pytest.raises(tool.ChildError, match="after the deadline"):
        runner.run([sys.executable, "-c", "pass"], env=dict(os.environ), timeout=60)


def test_unverified_stop_uses_bounded_reap_and_raises_stop_unverified():
    runner = supervisor(term_grace=0.05, kill_grace=0.2)
    runner._stop_group = lambda pgid: {"remaining": [pgid]}  # cleanup cannot prove the leader stopped
    started, leaders = time.monotonic(), []
    real_popen = tool.subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        leaders.append(process)
        return process

    tool.subprocess.Popen = popen
    try:
        with pytest.raises(tool.StopUnverified, match="not reaped within the bound"):
            runner.run([sys.executable, "-c", "import time; time.sleep(15)"], env=dict(os.environ), timeout=0.2)
        assert time.monotonic() - started < 10
    finally:
        tool.subprocess.Popen = real_popen
        for process in leaders:
            kill_quietly(process.pid)
            process.wait(timeout=10)


def test_group_listing_failure_still_kills_owned_group(monkeypatch):
    leaders = []
    real_popen = tool.subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        leaders.append(process)
        return process

    def failing_listing(pgid):
        raise OSError("listing unavailable")

    monkeypatch.setattr(tool.subprocess, "Popen", popen)
    monkeypatch.setattr(tool, "group_members", failing_listing)
    with pytest.raises(tool.StopUnverified, match="listing unavailable"):
        supervisor(term_grace=0.2, kill_grace=0.2).run([sys.executable, "-c", "import time; time.sleep(60)"],
                                                       env=dict(os.environ), timeout=0.2)
    assert leaders and leaders[0].returncode is not None, "owned leader was neither killed nor reaped"


def test_drain_error_still_reaps_and_closes_everything(monkeypatch, tmp_path):
    opened, selectors_made, leaders = [], [], []
    real_open, real_selector, real_popen = open, tool.selectors.DefaultSelector, tool.subprocess.Popen

    def tracked_open(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        opened.append(handle)
        return handle

    def tracked_selector():
        selector = real_selector()
        selectors_made.append(selector)
        return selector

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        leaders.append(process)
        return process

    def failing_drain(*args, **kwargs):
        raise RuntimeError("drain broke")

    monkeypatch.setattr(tool, "open", tracked_open, raising=False)
    monkeypatch.setattr(tool.selectors, "DefaultSelector", tracked_selector)
    monkeypatch.setattr(tool.subprocess, "Popen", popen)
    runner = supervisor()
    runner._drain = failing_drain
    with pytest.raises(tool.ChildError, match="drain broke"):
        runner.run([sys.executable, "-c", "pass"], env=dict(os.environ), timeout=30, log_path=tmp_path / "c.log")
    assert leaders[0].returncode is not None
    assert leaders[0].stdout.closed and leaders[0].stderr.closed and all(h.closed for h in opened)
    assert all(s.get_map() is None for s in selectors_made)  # closed selectors have no map


def _queue_sigint_during_cleanup(runner, verified):
    original = runner._cleanup

    def cleanup(*args, **kwargs):
        value = original(*args, **kwargs)
        if not verified:
            value.update(verified=False, remaining=None)
            value["errors"].append("cannot prove stop")
        os.kill(os.getpid(), signal.SIGINT)  # masked here, so it stays queued until the mask is restored
        return value

    runner._cleanup = cleanup


@pytest.fixture
def interrupt_handler():
    def handler(signum, frame):
        raise tool.Interrupted(f"signal {signum}")

    previous = signal.signal(signal.SIGINT, handler)
    yield
    signal.signal(signal.SIGINT, previous)


def test_queued_signal_cannot_hide_stop_unverified_and_state_stays_sticky(tmp_path, interrupt_handler):
    root = real(tmp_path)
    (root / "results").mkdir()
    layout = tool.Layout(root, root / "checkout", tool.load_pins(), pins_sha256=tool.PINS_SHA256)
    state = tool.State(layout).open()
    state.begin("env-vllm")
    runner = supervisor()
    _queue_sigint_during_cleanup(runner, verified=False)
    with pytest.raises(tool.StopUnverified) as caught:
        runner.run([sys.executable, "-c", "pass"], env=dict(os.environ), timeout=30)
    assert "signal received during cleanup: Interrupted" in str(caught.value)
    state.fail("env-vllm", caught.value)
    assert state.phase("env-vllm")["status"] == "stop-unverified" and state.data["stop_unverified"]
    state.close()
    with pytest.raises(tool.StopUnverified, match="Recovery is blocked"):
        tool.State(layout).open()


def test_queued_signal_is_delivered_after_a_verified_stop(interrupt_handler):
    leaders, real_popen = [], tool.subprocess.Popen

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        leaders.append(process)
        return process

    runner = supervisor()
    _queue_sigint_during_cleanup(runner, verified=True)
    tool.subprocess.Popen = popen
    try:
        with pytest.raises(tool.Interrupted):
            runner.run([sys.executable, "-c", "pass"], env=dict(os.environ), timeout=30)
    finally:
        tool.subprocess.Popen = real_popen
    assert leaders[0].returncode == 0 and leaders[0].stdout.closed


# --------------------------------------------------------------------------- pins, state, CLI with production pins

def test_actual_load_pins_binds_state_and_rejects_other_digests(tmp_path):
    pins = tool.load_pins()
    assert len(pins["v1_runtime_members"]) == 108 and "_sha256" not in pins
    root = real(tmp_path)
    (root / "results").mkdir()
    state = tool.State(tool.Layout(root, root / "checkout", pins, pins_sha256=tool.PINS_SHA256)).open()
    assert state.data["pins_sha256"] == tool.PINS_SHA256
    state.close()
    tool.State(tool.Layout(root, root / "checkout", pins, pins_sha256=tool.PINS_SHA256)).open().close()
    with pytest.raises(tool.RecoveryError, match="another tool, root or pin set"):
        tool.State(tool.Layout(root, root / "checkout", pins, pins_sha256="0" * 64)).open()
    with pytest.raises(tool.RecoveryError, match="digest is unknown"):
        tool.State(tool.Layout(root, root / "checkout", pins)).open()
    with pytest.raises(tool.RecoveryError, match="pins file changed"):
        tool.load_pins(expected_sha256="1" * 64)


def cli(args, env_extra=None, cwd=None):
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": os.environ.get("HOME", "/tmp"), "LC_ALL": "C",
           **(env_extra or {})}
    proc = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), *args], env=env, capture_output=True, text=True,
                          timeout=300, cwd=cwd)
    return proc.returncode, json.loads(proc.stdout)


def tree_listing(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


@pytest.mark.skipif(not EVIDENCE.is_file(), reason="Mac evidence archive not present")
def test_cli_plan_with_production_pins_and_real_archive_writes_nothing(tmp_path):
    root = real(tmp_path) / "decision-study"
    (root / "results/solo-v1/exports").mkdir(parents=True)
    (root / "results/solo-v1/exports/benchmark_bundle.zip").write_bytes(EVIDENCE.read_bytes())
    before = tree_listing(root)
    code, report = cli(["plan", "--runtime-root", str(root), "--matrix-checkout", str(REPO),
                        "--matrix-commit", COMMIT, "--uv", str(root / "missing-uv")])
    assert code == 1 and report["status"] == "BLOCKED" and report["tool"]["pins_sha256"] == tool.PINS_SHA256
    assert any("Archived paths name /home/zhanj289/e6998/decision-study" in b for b in report["blockers"])
    assert report["archive"]["members_to_restore"] == 108
    assert any("Study lock results/solo-v1: missing" == b for b in report["blockers"])
    assert tree_listing(root) == before


def test_cli_execute_fails_closed_before_any_write(tmp_path):
    root = real(tmp_path) / "decision-study"
    (root / "results").mkdir(parents=True)
    before = tree_listing(root)
    code, report = cli(["execute", "--runtime-root", str(root), "--matrix-checkout", str(REPO),
                        "--matrix-commit", COMMIT, "--uv", sys.executable], {"UV_INDEX_URL": "https://example.invalid"})
    assert code == 2 and report["status"] == "FAILED" and "UV_INDEX_URL" in report["error"]
    assert tree_listing(root) == before




# --------------------------------------------------------------------------- archive and path guards

def zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for info, data in entries:
            archive.writestr(info, data)
    return buffer.getvalue()


def member(name, mode=stat.S_IFREG | 0o640, compress=zipfile.ZIP_DEFLATED, flags=0):
    info = zipfile.ZipInfo(name)
    info.external_attr, info.compress_type = mode << 16, compress
    info.flag_bits |= flags
    return info


@pytest.mark.parametrize("entries, message", [
    ([(member("runtime/link", stat.S_IFLNK | 0o777), "target")], "symlink or special"),
    ([(member("runtime/fifo", stat.S_IFIFO | 0o640), "")], "symlink or special"),
    ([(member("../escape"), "x")], "Unsafe archive member path"),
    ([(member("runtime/../../escape"), "x")], "Unsafe archive member path"),
    ([(member("/absolute"), "x")], "Unsafe archive member name"),
    ([(member("runtime\\windows"), "x")], "Unsafe archive member name"),
    ([(member("runtime/a"), "x"), (member("runtime/a"), "y")], "duplicate"),
    ([(member("runtime/bz", compress=zipfile.ZIP_BZIP2), "x")], "unsupported compression"),
])
def test_archive_structure_guards(entries, message):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = zip_bytes(entries)
    with pytest.raises(tool.RecoveryError, match=message):
        tool.check_archive_structure(zipfile.ZipFile(io.BytesIO(data)))


def test_archive_encrypted_flag_is_refused():
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes([(member("runtime/enc"), "x")])))
    archive.infolist()[0].flag_bits |= 0x1  # zipfile resets flag bits when writing, so set it on the parsed entry
    with pytest.raises(tool.RecoveryError, match="encrypted"):
        tool.check_archive_structure(archive)


def test_archive_size_limit_and_member_pins(monkeypatch):
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes([(member("runtime/a.json"), "0123456789AB")])))
    monkeypatch.setitem(tool.ARCHIVE_LIMITS, "member_bytes", 10)
    with pytest.raises(tool.RecoveryError, match="too large"):
        tool.check_archive_structure(archive)
    good = {"size": 12, "sha256": hashlib.sha256(b"0123456789AB").hexdigest()}
    assert tool.read_member(archive, "runtime/a.json", good)[0] == b"0123456789AB"
    with pytest.raises(tool.RecoveryError, match="size differs"):
        tool.read_member(archive, "runtime/a.json", {**good, "size": 11})
    with pytest.raises(tool.RecoveryError, match="bytes differ"):
        tool.read_member(archive, "runtime/a.json", {**good, "sha256": "0" * 64})


@pytest.mark.skipif(not EVIDENCE.is_file(), reason="Mac evidence archive not present")
def test_real_archive_passes_guards_and_every_pinned_member_verifies():
    pins = tool.load_pins()
    archive = zipfile.ZipFile(io.BytesIO(EVIDENCE.read_bytes()))
    tool.check_archive_structure(archive)
    for name, pin in pins["v1_runtime_members"].items():
        tool.read_member(archive, name, pin)
    installations = tool.archived_json(archive, tool.Layout("/x", "/y", pins), "runtime/installations.json")
    assert str(tool.archived_root(installations)) == "/home/zhanj289/e6998/decision-study"
    assert not any(name in archive.namelist() for name in tool.ABSENT_MEMBERS[2:])


def test_lstat_chain_refuses_symlinked_component(tmp_path):
    root = real(tmp_path)
    (root / "elsewhere").mkdir()
    os.symlink(root / "elsewhere", root / "link")
    with pytest.raises(tool.RecoveryError, match="symlinked path component"):
        tool.lstat_chain(root, "link/file")
    assert tool.lstat_chain(root, "absent/file") is None


# --------------------------------------------------------------------------- locks, ownership, environment, adapter

def study_root(tmp_path):
    root = real(tmp_path) / "decision-study"
    for relative in tool.LOCKED_OUTPUTS:
        (root / relative).mkdir(parents=True)
        (root / relative / ".active.lock").touch()
    return root


def hold_lock(path):
    ready = Path(str(path) + ".held")
    code = ("import fcntl, sys, time, pathlib; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); "
            "pathlib.Path(sys.argv[2]).touch(); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", code, str(path), str(ready)])
    assert wait_until(ready.exists, 10)
    return holder


def test_study_locks_missing_held_and_free(tmp_path):
    root = study_root(tmp_path)
    layout = tool.Layout(root, root / "checkout", {})
    lock = root / tool.MATRIX_OUTPUT / ".active.lock"
    lock.unlink()
    with pytest.raises(tool.RecoveryError, match="mutual exclusion"):
        with tool.study_locks(layout):
            pass
    assert not lock.exists(), "a missing study lock must not be created"
    assert tool.probe_locks(layout)[tool.MATRIX_OUTPUT] == "missing"
    lock.touch()
    holder = hold_lock(root / tool.V3 / ".active.lock")
    try:
        assert tool.probe_locks(layout)[tool.V3] == "held"
        with pytest.raises(tool.RecoveryError, match="in use"):
            with tool.study_locks(layout):
                pass
    finally:
        holder.kill()
        holder.wait()
    with tool.study_locks(layout):
        assert tool.probe_locks(layout)[tool.SIZE] == "held"
    assert tool.probe_locks(layout)[tool.SIZE] == "free"


def test_state_refuses_preexisting_and_replaced_owned_trees(tmp_path):
    root = real(tmp_path)
    (root / "results/solo-v1/runtime").mkdir(parents=True)
    layout = tool.Layout(root, root / "checkout", tool.load_pins(), pins_sha256=tool.PINS_SHA256)
    with pytest.raises(tool.RecoveryError, match="not recovery-owned"):
        tool.State(layout).open()
    assert not (root / tool.STATE).exists()
    (root / "results/solo-v1/runtime").rmdir()
    state = tool.State(layout).open()
    owned = state.own(tool.V1_RUNTIME)
    state.close()
    owned.rmdir()
    owned.mkdir()  # same path, new inode: no longer the directory this recovery created
    state = tool.State(layout).open()
    with pytest.raises(tool.RecoveryError, match="disappeared or was replaced"):
        state.own(tool.V1_RUNTIME)
    with pytest.raises(tool.RecoveryError, match="not inside a recovery-owned tree"):
        state.check_owned(root / "results/solo-v3/runtime/binding.json")
    state.close()


def test_ambient_blockers_and_child_environment(monkeypatch):
    for key, value in {"UV_INDEX_URL": "x", "PIP_CONSTRAINT": "x", "PYTHONPATH": "x", "UV_NATIVE_TLS": "1",
                       "HF_HUB_CACHE": "x", "GIT_DIR": "x", "KEV_X": "x", "CUDA_VISIBLE_DEVICES": "0"}.items():
        monkeypatch.setenv(key, value)
    blocked = " ".join(tool.ambient_blockers())
    assert all(k in blocked for k in ("UV_INDEX_URL", "PIP_CONSTRAINT", "PYTHONPATH", "HF_HUB_CACHE"))
    assert "UV_NATIVE_TLS" not in blocked
    env = tool.child_env({"HF_HOME": "/h"})
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["HF_HOME"] == "/h" and env["UV_NATIVE_TLS"] == "1"
    assert not {"GIT_DIR", "KEV_X", "PYTHONPATH", "UV_INDEX_URL"} & set(env)
    assert tool.child_env(keep=("PYTHONPATH",))["PYTHONPATH"] == "x"


def test_frozen_run_adapter_is_scoped_and_supervised(tmp_path):
    import types
    frozen = types.ModuleType("fake_frozen")
    exec("import subprocess\n"
         "class RuntimeCommandError(subprocess.CalledProcessError):\n"
         "    def __init__(self, returncode, cmd, log_path, *, output=None, stderr=None):\n"
         "        super().__init__(returncode, cmd, output=output, stderr=stderr)\n"
         "        self.log_path = str(log_path)\n"
         "def _run(args, output_dir, log_name, *, cwd=None, capture=False):\n"
         "    raise AssertionError('the unsupervised frozen _run must not be called')\n", frozen.__dict__)
    original, captured = frozen._run, []
    with tool.frozen_run_adapter(frozen, supervisor(), tmp_path, {}, captured) as adapter:
        assert frozen._run is adapter
        assert frozen._run([sys.executable, "-c", "print('hi')"], "/unused", "x.log", capture=True) == "hi\n"
        with pytest.raises(frozen.RuntimeCommandError):
            frozen._run([sys.executable, "-c", "raise SystemExit(3)"], "/unused", "x.log", capture=True)
        with pytest.raises(tool.RecoveryError, match="already replaced"):
            with tool.frozen_run_adapter(frozen, supervisor(), tmp_path, {}, []):
                pass
    assert frozen._run is original
    with pytest.raises(ZeroDivisionError):
        with tool.frozen_run_adapter(frozen, supervisor(), tmp_path, {}, []):
            1 / 0
    assert frozen._run is original and (tmp_path / "x.log").exists()


def test_exclude_newer_rounds_up_to_whole_seconds():
    assert tool.exclude_newer("2026-10-30T20:15:33.170549+00:00") == "2026-10-30T20:15:34Z"
    assert tool.exclude_newer("2026-11-01T08:26:40+00:00") == "2026-11-01T08:26:40Z"
    with pytest.raises(tool.RecoveryError):
        tool.exclude_newer("2026-11-01T08:26:40")


# --------------------------------------------------------------------------- pinned sources (real git)

def git(*args, cwd=None):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *map(str, args)],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def make_repo(path, files, ignore="build/\n*.egg-info/\n"):
    path.mkdir(parents=True)
    for relative, text in {**files, ".gitignore": ignore}.items():
        (path / relative).parent.mkdir(parents=True, exist_ok=True)
        (path / relative).write_text(text)
    git("init", "-q", cwd=path)
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "pinned", cwd=path)
    return git("rev-parse", "HEAD", cwd=path)


def test_check_source_sees_ignored_extras_racy_edits_and_incomplete_clones(tmp_path):
    repo = real(tmp_path) / "repo"
    revision = make_repo(repo, {"pkg/__init__.py": "VALUE = 1\n", "pkg/core.py": "def f():\n    return 1\n"})
    sup = supervisor()
    assert tool.check_source(sup, repo, revision)["state"] == "pristine"
    (repo / "build/lib/pkg").mkdir(parents=True)
    (repo / "build/lib/pkg/core.py").write_text("def f():\n    return 2\n")  # ignored stale copy
    found = tool.check_source(sup, repo, revision)
    assert found["state"] == "extras" and "build/lib/pkg/core.py" in found["extras"]
    core = repo / "pkg/core.py"
    before = core.stat()
    core.write_text("def f():\n    return 9\n")  # same size; restore mtime so git's stat cache may call it clean
    os.utime(core, ns=(before.st_atime_ns, before.st_mtime_ns))
    found = tool.check_source(sup, repo, revision)
    assert found["state"] == "changed" and any("pkg/core.py" in p for p in found["problems"])
    assert tool.check_source(sup, repo, "0" * 40)["state"] == "changed"
    empty = real(tmp_path) / "empty"
    empty.mkdir()
    git("init", "-q", cwd=empty)
    assert tool.check_source(sup, empty, revision)["state"] == "incomplete"


# --------------------------------------------------------------------------- R6 gate on real restored metadata

@pytest.mark.skipif(not EVIDENCE.is_file(), reason="Mac evidence archive not present")
def test_failed_metadata_pin_spawns_nothing_derived_from_it(tmp_path, monkeypatch):
    pins = tool.load_pins()
    root = real(tmp_path)
    layout = tool.Layout(root, root / "checkout", pins, pins_sha256=tool.PINS_SHA256)
    archive = zipfile.ZipFile(io.BytesIO(EVIDENCE.read_bytes()))
    for name, pin in pins["v1_runtime_members"].items():
        target = layout.path(f"{tool.V1}/{name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tool.read_member(archive, name, pin)[0])
    assert tool.verify_members(layout) == []
    installations = layout.path(tool.V1_RUNTIME + "/installations.json")
    record = json.loads(installations.read_text())
    record["environments"]["kev"]["python"] = "/unexpected/copied/environment/bin/python"
    installations.write_text(json.dumps(record))
    (layout.path(tool.V1_RUNTIME) / "session.json").write_text("{}")
    spawned = []
    monkeypatch.setattr(tool.Supervisor, "run", lambda self, argv, **kw: spawned.append(argv))
    monkeypatch.setattr(tool, "check_headers", lambda *a, **k: spawned.append(a))
    report = tool.verify(layout, supervisor(), None, root / "verify", own_locks=True)
    assert report["status"] == "MATRIX_RUNTIME_NOT_READY" and spawned == []
    problems = " ".join(report["problems"])
    assert "installations.json" in problems and "session.json" in problems
    assert report["environments"].startswith("skipped") and report["matrix_output"].startswith("skipped")
    with pytest.raises(tool.RecoveryError, match="fail their pins"):
        tool.require_pins(layout)


# --------------------------------------------------------------------------- rehearsal world

QWEN, KEV, BASE, STARTLUX = ("c202236235762e1c871ad0ccb60c8ee5ba337b9a", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
                             "1001bb4d826a52d1f399e183466143f4da7b741b", "18a17f73" * 5)
TEMPERATURE = 2.406050072164233
SHARDS = {"model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-1.safetensors",
                                                                     "b": "model-2.safetensors"}}),
          "model-1.safetensors": "w1", "model-2.safetensors": "w2", "config.json": "{}"}


@pytest.fixture(scope="module")
def matrix_checkout(tmp_path_factory):
    target = real(tmp_path_factory.mktemp("matrix")) / "decision-study-src"
    git("clone", "--quiet", "--no-checkout", REPO, target)
    git("-C", target, "checkout", "--quiet", "--detach", COMMIT)
    return target


class World:
    pass


def build_world(base, checkout):
    w = World()
    w.base, w.checkout = real(base), checkout
    w.root, w.fake, w.home, w.tmp = (w.base / n for n in ("decision-study", "fake", "home", "tmp"))
    for path in (w.fake / "control", w.home, w.tmp):
        path.mkdir(parents=True)
    results, work = w.root / "results", w.root / "results/solo-v1/runtime/work"
    revisions = {}
    for repo, project, ignore in (("jaredpalmer/kev", "kev", "*.egg-info/\n"),
                                  ("StartLuxLabs/StartLux-Decision", "startlux-decide", "build/\n*.egg-info/\n")):
        source = w.base / "remote-src" / repo
        revisions[repo] = make_repo(source, {"pyproject.toml": f'[project]\nname = "{project}"\nversion = "0.0.1"\n',
                                             f"{project.replace('-', '_')}/__init__.py": "VALUE = 1\n"}, ignore)
        bare = w.base / "remotes" / f"{repo}.git"
        git("clone", "--quiet", "--bare", source, bare)
        git("-C", bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    (w.home / ".gitconfig").write_text(
        f'[url "file://{w.base}/remotes/"]\n\tinsteadOf = https://github.com/\n[protocol "file"]\n\tallow = always\n'
        "[init]\n\tdefaultBranch = main\n[advice]\n\tdetachedHead = false\n")
    w.sources = {repo: work / "sources" / f"{repo.replace('/', '--')}-{sha[:12]}" for repo, sha in revisions.items()}
    for relative in tool.LOCKED_OUTPUTS:
        (w.root / relative).mkdir(parents=True)
        (w.root / relative / ".active.lock").touch()
    (results / "solo-v2").mkdir()
    (results / "aux-direct-matrix-v1/status.json").write_text(json.dumps({"state": "blocked", "stages": {
        "preflight": {"state": "blocked", "last_error": "runtime missing"}}}))
    (w.root / "decision_study.egg-info").mkdir()
    (w.root / "decision_study.egg-info/PKG-INFO").write_text("Metadata-Version: 2.1\nName: decision-study\nVersion: 0.1.0\n")
    hub1, hub3 = work / "cache/huggingface/hub", results / "solo-v3/runtime-work/cache/huggingface/hub"
    kev_record = {"model": "kev", "model_path": str(hub1 / f"models--jaredpalmer--kev-4b/snapshots/{KEV}"),
                  "base_path": str(hub1 / f"models--Qwen--Qwen3.5-4B-Base/snapshots/{BASE}"), "revision": KEV,
                  "base_revision": BASE, "native_temperature": TEMPERATURE}
    qwen_record = {"model": "qwen", "model_path": str(hub3 / f"models--Qwen--Qwen3.5-9B/snapshots/{QWEN}"),
                   "base_path": None, "revision": QWEN, "base_revision": None, "native_temperature": None}
    controls = {
        "solo-v3/runtime/binding.json": {"work_dir": str(results / "solo-v3/runtime-work")},
        "solo-v3/runtime-controls/qwen9b/runtime/lock.json": {"models": {"qwen": {"repo": "Qwen/Qwen3.5-9B",
                                                                                  "revision": QWEN}}},
        "solo-v3/runtime-controls/qwen9b/runtime/installations.json": {"models": {"qwen": qwen_record}},
        "solo-v3/runtime-controls/qwen9b/runtime/download-qwen9b.json": qwen_record,
        "aux-size-control-v1-r2/runtime/binding.json": {"work_dir": str(results / "aux-size-control-v1-r2/runtime-work")},
        "aux-size-control-v1-r2/runtime-controls/startlux9b/runtime/lock.json": {"models": {}},
        "aux-size-control-v1-r2/runtime-controls/startlux9b/runtime/installations.json": {"models": {"startlux": {
            "model_path": str(results / "aux-size-control-v1-r2/runtime-work/cache/huggingface/hub/"
                              f"models--startlux-models--StartLux-Decision-v1.1-9B/snapshots/{STARTLUX}")}}},
    }
    for relative, value in controls.items():
        (results / relative).parent.mkdir(parents=True, exist_ok=True)
        (results / relative).write_text(json.dumps(value, indent=1) + "\n")
    fakes.make_snapshot(results / "aux-size-control-v1-r2/runtime-work/cache/huggingface/hub",
                        "startlux-models/StartLux-Decision-v1.1-9B", STARTLUX, SHARDS)
    freeze = {"kev": fakes.freeze_text(["decision-study==0.1.0", "flash-linear-attention==0.5.2", "huggingface_hub==2.4.0",
                                        f"kev @ file://{w.sources['jaredpalmer/kev']}", "pip==26.2.2",
                                        "setuptools==84.2.1"]),
              "vllm-0.31.0": fakes.freeze_text(["decision-study==0.1.0", "huggingface_hub==1.33.0", "pip==26.2.2",
                                                "setuptools==80.10.2", "vllm==0.31.0",
                                                f"startlux-decide @ file://{w.sources['StartLuxLabs/StartLux-Decision']}"])}
    created = {"kev": "2026-10-30T20:15:33.170549+00:00", "vllm-0.31.0": "2026-11-01T08:26:40.073053+00:00"}
    installations = {"lock_hash": "rehearsal", "models": {"kev": kev_record}, "environments": {
        name: {"created_at": created[name], "freeze_sha256": hashlib.sha256(text.encode()).hexdigest(),
               "managed_python": True, "python": str(work / f"envs/{name}/bin/python"), "python_version": "3.12.8"}
        for name, text in freeze.items()}}
    lock = {"lock_hash": "rehearsal", "verified_sources": [{"kind": "github", "repo": r, "sha": s} for r, s in revisions.items()],
            "models": {"kev": {"repo": "jaredpalmer/kev-4b", "revision": KEV, "base_repo": "Qwen/Qwen3.5-4B-Base",
                               "base_revision": BASE, "native_temperature": TEMPERATURE, "code_repo": "jaredpalmer/kev",
                               "code_revision": revisions["jaredpalmer/kev"]}}}
    restored = {"runtime/lock.json": json.dumps(lock), "runtime/installations.json": json.dumps(installations),
                "runtime/doctor.json": "{}", "runtime/download-kev.json": json.dumps(kev_record),
                "runtime/resolved/kev.txt": freeze["kev"], "runtime/resolved/vllm-0.31.0.txt": freeze["vllm-0.31.0"],
                "runtime/resolved/kev-portable.txt": fakes.portable_text(freeze["kev"]),
                tool.TOOLKIT_SETUP: json.dumps(toolkit_setup(checkout, work / "toolchains/cuda-13.0-flashinfer-curand10.4.0.35")),
                "runtime/logs/install.log": "archived log\n", "runtime/python-headers-kev.json": '{"archived": true}\n'}
    hidden = {"runtime/session.json": '{"pid": 1}', "runtime/last_services.json": '{"pids": [1]}'}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in {**restored, **hidden, "predictions/kev/config.json": "{}"}.items():
            archive.writestr(member(name), text)
    data = buffer.getvalue()
    (results / "solo-v1/exports").mkdir(parents=True)
    (results / "solo-v1/exports/benchmark_bundle.zip").write_bytes(data)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    entry = lambda text: {"size": len(text.encode()), "sha256": digest(text)}
    pins = {"derived_from": {"v1": {"archive": "rehearsal", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                             "v3": {"archive": "unused", "bytes": 0, "sha256": "0" * 64},
                             "size": {"archive": "unused", "bytes": 0, "sha256": "0" * 64}},
            "v1_runtime_members": {n: entry(t) for n, t in restored.items()},
            "v1_not_restored": {n: entry(t) for n, t in hidden.items()},
            "v3_controls": {r.split("/", 1)[1]: tool.sha256_file(results / r) for r in controls if r.startswith("solo-v3")},
            "size_controls": {r.split("/", 1)[1]: tool.sha256_file(results / r) for r in controls if r.startswith("aux-size")},
            "matrix_files": tool.load_pins()["matrix_files"]}
    w.pins = w.fake / "pins.json"
    w.pins.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    w.pins_sha = tool.sha256_file(w.pins)
    world_json = w.fake / "world.json"
    world_json.write_text(json.dumps({"dir": str(w.fake), "python": sys.executable, "fakes": str(FAKES), "hub": {
        "jaredpalmer/kev-4b": {"adapter_config.json": "{}", "adapter_model.safetensors": "adapter",
                               "kev_meta.json": json.dumps({"base": "Qwen/Qwen3.5-4B-Base", "base_revision": BASE,
                                                            "temperature": TEMPERATURE})},
        "Qwen/Qwen3.5-4B-Base": SHARDS, "Qwen/Qwen3.5-9B": SHARDS}}))
    w.uv = w.fake / "uv"
    w.uv.write_text(fakes.shim({"python": sys.executable, "fakes": str(FAKES), "path": str(world_json)}, "uv"))
    w.uv.chmod(0o750)
    w.protected = {p: p.read_bytes() for p in [*results.glob("solo-v3/**/*.json"), *results.glob("aux-*/**/*.json"),
                                              *results.glob("**/.active.lock")] if p.is_file()}
    return w


def toolkit_setup(checkout, target):
    """The archived setup identity this rehearsal's fake pip reproduces, computed by the pinned installer."""
    installer = load(checkout / "scripts/prepare_cuda_toolkit.py")
    requirements = checkout / tool.TOOLKIT_REQUIREMENTS
    manifest, wheels = installer._requirements(requirements)
    previous = os.umask(0o027)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"
            stage.mkdir()
            (stage / ".pip-work").mkdir()
            fakes.fake_pip_target(stage, requirements)
            tree = installer._tree(stage)
    finally:
        os.umask(previous)
    return {"status": "complete", "reused": True, "target": str(target), "requirements_sha256": manifest,
            "verification": {"wheels": wheels, "distributions": dict(installer.PINS), "tree_sha256": installer._digest(tree),
                             "installer_sha256": installer.file_hash(checkout / "scripts/prepare_cuda_toolkit.py"),
                             "requirements_sha256": manifest}}


def run_cli(w, command, mode="stub", env_extra=None):
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(w.home), "TMPDIR": str(w.tmp), "LC_ALL": "C",
           **(env_extra or {})}
    proc = subprocess.run([sys.executable, "-I", "-B", str(RUNNER), str(SCRIPT), str(w.pins), w.pins_sha, mode, command,
                           "--runtime-root", str(w.root), "--matrix-checkout", str(w.checkout), "--matrix-commit", COMMIT,
                           "--uv", str(w.uv)], env=env, capture_output=True, text=True, timeout=600)
    try:
        return proc.returncode, json.loads(proc.stdout)
    except ValueError:
        raise AssertionError(f"no JSON report (rc={proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")


def calls(w):
    path = w.fake / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def state_of(w):
    return json.loads((w.root / tool.STATE / "state.json").read_text())


@pytest.fixture
def world(tmp_path, matrix_checkout):
    return build_world(tmp_path, matrix_checkout)


# --------------------------------------------------------------------------- rehearsal: real CLI, fake external boundaries

def test_rehearsal_plan_execute_verify_end_to_end(world):
    w, work = world, world.root / "results/solo-v1/runtime/work"
    before = tree_listing(w.root)
    code, report = run_cli(w, "plan")
    assert (code, report["status"]) == (0, "READY_TO_EXECUTE"), report["blockers"]
    assert report["matrix_output"]["status"]["stages"]["preflight"]["last_error"] == "runtime missing"
    assert tree_listing(w.root) == before, "plan wrote below the runtime root"

    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (0, "MATRIX_RUNTIME_READY"), report.get("problems") or report
    assert all(state_of(w)["phases"][p]["status"] == "complete" for p in tool.PHASES)
    pins = json.loads(w.pins.read_text())
    for name, pin in pins["v1_runtime_members"].items():  # includes python-headers-kev.json: never rewritten
        assert hashlib.sha256((w.root / tool.V1 / name).read_bytes()).hexdigest() == pin["sha256"], name
    assert not any((w.root / tool.V1 / name).exists() for name in tool.ABSENT_MEMBERS)
    assert all(path.read_bytes() == data for path, data in w.protected.items())
    assert git("-C", w.checkout, "status", "--porcelain=v1", "--untracked-files=all", "--ignored") == ""
    log = calls(w)
    assert log and all(c["env"]["CUDA_VISIBLE_DEVICES"] == "" for c in log)
    freezes = [c for c in log if c["args"][1:] == ["-m", "pip", "freeze", "--all"]]
    assert freezes and all(str(w.root) in c["env"]["PYTHONPATH"].split(os.pathsep) for c in freezes)
    installs = [c["args"] for c in log if c["args"][1:3] == ["pip", "install"]]
    assert any("2026-10-30T20:15:34Z" in a and "flash-linear-attention==0.5.2" in a for a in installs)
    assert any("2026-11-01T08:26:41Z" in a and "vllm==0.31.0" in a for a in installs)
    portable = (w.root / tool.STATE / "inputs/kev-portable.txt").read_text()
    assert "decision-study" not in portable and portable == (w.root / tool.V1_RUNTIME / "resolved/kev-portable.txt").read_text()
    assert (work / "toolchains/cuda-13.0-flashinfer-curand10.4.0.35/.cuda-toolkit-receipt.json").is_file()

    code, report = run_cli(w, "verify", mode="real")  # frozen StartLux-9B metadata check and plans() unchanged
    assert code == 1 and len(report["problems"]) == 2, report["problems"]
    assert report["problems"][0].startswith("startlux9b:") and "plans()" in report["problems"][1]
    assert all(e["legacy_context"] for e in report["environments"].values()) and report["toolkit"]["tree_sha256"]
    assert {s["state"] for s in report["sources"].values()} == {"extras"}  # generated copies of the verified build
    kept = set(w.tmp.iterdir())

    code, report = run_cli(w, "verify")
    assert (code, report["status"], report["check_logs"]) == (0, "MATRIX_RUNTIME_READY", "removed")
    assert set(w.tmp.iterdir()) == kept

    count = len(calls(w))
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (0, "MATRIX_RUNTIME_READY")
    assert not [c for c in calls(w)[count:] if c["tool"] == "uv" and "install" in c["args"]], "a complete phase reran"


def uv_calls_since(w, count, word):
    return [c["args"] for c in calls(w)[count:] if c["tool"] == "uv" and word in c["args"]]


def test_rehearsal_freeze_mismatch_is_terminal_sticky_and_never_repinned(world):
    w = world
    (w.fake / "control/drift-vllm-0.31.0").touch()  # the fake installer resolves pip==26.2.3
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (3, "TERMINAL_MISMATCH") and "nothing re-pinned" in report["error"]
    record = state_of(w)["phases"]["env-vllm"]
    assert record["status"] == "failed" and record["terminal"] is True
    diffs = list((w.root / tool.STATE / "failures").glob("vllm-0.31.0-freeze-*/diff.txt"))
    assert diffs and "+pip==26.2.3" in diffs[0].read_text()
    resolved = w.root / tool.V1_RUNTIME / "resolved/vllm-0.31.0.txt"
    assert "pip==26.2.2" in resolved.read_text()  # archived target untouched
    count = len(calls(w))
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (3, "TERMINAL_MISMATCH") and not uv_calls_since(w, count, "venv")
    code, report = run_cli(w, "plan")
    assert report["status"] == "BLOCKED" and any("terminal mismatch" in b for b in report["blockers"])


def test_rehearsal_retry_preserves_unclean_source_and_rebuilds_owned_env(world):
    w = world
    (w.fake / "control/fail-install-once-kev").touch()
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (2, "FAILED") and state_of(w)["phases"]["env-kev"]["terminal"] is False
    kev = w.sources["jaredpalmer/kev"]
    (kev / "build/lib/kev").mkdir(parents=True)
    (kev / "build/lib/kev/__init__.py").write_text("VALUE = 'stale'\n")  # stale copy left by an attempt
    (kev / "kev.egg-info").mkdir()
    (kev / "kev.egg-info/PKG-INFO").write_text("ignored\n")
    count = len(calls(w))
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (0, "MATRIX_RUNTIME_READY"), report.get("error") or report.get("problems")
    preserved = state_of(w)["preserved"]
    assert len(preserved) == 1 and preserved[0]["phase"] == "env-kev"
    assert (Path(preserved[0]["to"]) / "build/lib/kev/__init__.py").read_text() == "VALUE = 'stale'\n"
    assert not (kev / "build/lib/kev/__init__.py").exists()
    venvs = uv_calls_since(w, count, "venv")
    assert len(venvs) == 1 and "--clear" in venvs[0] and venvs[0][-1].endswith("envs/kev")


def test_rehearsal_changed_tracked_source_blocks_retry_before_any_uv_call(world):
    w = world
    (w.fake / "control/fail-install-once-kev").touch()
    run_cli(w, "execute")
    edited = w.sources["jaredpalmer/kev"] / "kev/__init__.py"
    edited.write_text("VALUE = 2\n")
    count = len(calls(w))
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (2, "FAILED") and "Pinned source changed" in report["error"]
    assert [c for c in calls(w)[count:] if c["tool"] == "uv" and c["args"] != ["--version"]] == []
    assert edited.read_text() == "VALUE = 2\n" and "preserved" not in state_of(w)
    code, report = run_cli(w, "verify")
    assert report["status"] == "MATRIX_RUNTIME_NOT_READY"


def test_rehearsal_tampered_metadata_gates_resume_and_verify_without_spawning(world):
    w = world
    (w.fake / "control/fail-install-once-vllm-0.31.0").touch()
    run_cli(w, "execute")
    assert state_of(w)["phases"]["metadata"]["status"] == "complete"
    installations = w.root / tool.V1_RUNTIME / "installations.json"
    record = json.loads(installations.read_text())
    record["environments"]["vllm-0.31.0"]["python"] = str(w.base / "copied/bin/python")
    installations.write_text(json.dumps(record))
    count = len(calls(w))
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (2, "FAILED") and "fail their pins" in report["error"]
    code, report = run_cli(w, "verify")
    assert report["status"] == "MATRIX_RUNTIME_NOT_READY" and report["environments"].startswith("skipped")
    assert [c for c in calls(w)[count:] if c["args"] != ["--version"]] == [], "a metadata-derived program ran"


def test_rehearsal_blockers_stop_execute_before_any_write(world):
    w = world
    binding = w.root / "results/solo-v3/runtime/binding.json"
    original = binding.read_bytes()
    binding.write_text("{}\n")
    code, report = run_cli(w, "execute")
    assert code == 2 and "fail their pins" in report["error"] and not (w.root / tool.STATE).exists()
    binding.write_bytes(original)
    (w.root / tool.V1_RUNTIME).mkdir()
    code, report = run_cli(w, "execute")
    assert code == 2 and "not recovery-owned" in report["error"] and not (w.root / tool.STATE).exists()
    (w.root / tool.V1_RUNTIME).rmdir()
    lock = w.root / tool.MATRIX_OUTPUT / ".active.lock"
    lock.unlink()
    code, report = run_cli(w, "execute")
    assert code == 2 and "mutual exclusion" in report["error"]
    assert not (w.root / tool.STATE).exists() and not lock.exists()


def test_rehearsal_stop_unverified_record_blocks_execute_verify_and_plan(world):
    w = world
    layout = tool.Layout(w.root, w.checkout, tool.load_pins(w.pins, w.pins_sha), pins_sha256=w.pins_sha)
    state = tool.State(layout).open()
    state.begin("env-vllm")
    state.fail("env-vllm", tool.StopUnverified("simulated: owned group could not be proven stopped"))
    state.close()
    count = len(calls(w))
    code, report = run_cli(w, "execute")
    assert (code, report["status"]) == (4, "STOP_UNVERIFIED") and "Recovery is blocked" in report["error"]
    code, report = run_cli(w, "verify")
    assert report["status"] == "MATRIX_RUNTIME_NOT_READY" and any("Recovery is blocked" in p for p in report["problems"])
    code, report = run_cli(w, "plan")
    assert report["status"] == "BLOCKED" and any("Recovery is blocked" in b for b in report["blockers"])
    assert [c for c in calls(w)[count:] if c["args"] != ["--version"]] == []


def test_metadata_restore_never_overwrites_existing_content(world):
    w = world
    layout = tool.Layout(w.root, w.checkout, tool.load_pins(w.pins, w.pins_sha), pins_sha256=w.pins_sha)
    state = tool.State(layout).open()
    try:
        runtime = state.own(tool.V1_RUNTIME)
        (runtime / "doctor.json").write_text("not the archived bytes\n")
        with pytest.raises(tool.RecoveryError, match="never overwritten"):
            tool.Executor(layout, None, None, state, "uv").phase_metadata()
        assert (runtime / "doctor.json").read_text() == "not the archived bytes\n"
    finally:
        state.close()


def test_toolkit_phase_propagates_rejection_even_with_a_valid_target(tmp_path, monkeypatch):
    import types
    root = real(tmp_path)
    layout = tool.Layout(root, root / "checkout", {})
    target = layout.toolkit_dir()
    target.mkdir(parents=True)
    setup = root / "setup.json"
    setup.write_text(json.dumps({"status": "complete", "target": str(target)}))
    executor = object.__new__(tool.Executor)
    executor.layout, executor.matrix = layout, None
    executor.state = types.SimpleNamespace(dir=root / "state", check_owned=lambda path: path)
    executor.member = lambda name: setup
    executor.environment = lambda name: ({}, Path(sys.executable))

    def reject(*args, **kwargs):
        raise tool.ChildError("Exit observed after the deadline; result not accepted")

    executor.run = reject
    monkeypatch.setattr(tool, "toolkit_identity", lambda *args, **kwargs: {"would_accept": True})
    with pytest.raises(tool.ChildError, match="after the deadline"):
        executor.phase_toolkit()


def test_verify_refuses_tmpdir_inside_runtime_root_without_creating_anything(world):
    inside = world.root / "results/tmp"  # clean matrix checkout: without the check, verify would reach mkdtemp
    inside.mkdir()
    code, report = run_cli(world, "verify", env_extra={"TMPDIR": str(inside)})
    assert list(inside.iterdir()) == [], "verify created a directory inside the protected runtime root"
    assert code == 2 and "TMPDIR resolves inside" in report["error"]
