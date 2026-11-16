"""RHEL8 compatibility must retain kernel identity, not downgrade to numeric kill."""
import ctypes
import errno
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from decision_study.learnability_profile import pidfd as p
from decision_study.learnability_profile import study as s
from decision_study.common import StudyError


def unavailable(*args):
    raise OSError(errno.ENOSYS, "not implemented")


def test_callable_native_apis_take_priority(monkeypatch):
    events = []
    monkeypatch.setattr(p.os, "pidfd_open", lambda pid: events.append(("open", pid)) or 37, raising=False)
    monkeypatch.setattr(p.signal, "pidfd_send_signal", lambda fd, sig: events.append(("send", fd, sig)), raising=False)
    monkeypatch.setattr(p, "_syscall", lambda *args: pytest.fail("native APIs must take priority"))
    assert p.open_process(123) == 37
    p.send_signal(37, signal.SIGTERM)
    assert events == [("open", 123), ("send", 37, signal.SIGTERM)]


def test_missing_or_noncallable_apis_use_exact_raw_abi(monkeypatch):
    monkeypatch.setattr(p.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(p.signal, "pidfd_send_signal", None, raising=False)
    calls = []
    monkeypatch.setattr(p, "_syscall", lambda nr, *args: calls.append((nr, args)) or (37 if nr == 434 else 0))
    assert p.open_process(123) == 37
    p.send_signal(37, signal.SIGTERM)
    assert [row[0] for row in calls] == [434, 424]
    args = calls[0][1]
    assert [type(v) for v in args] == [ctypes.c_int, ctypes.c_uint]
    assert [v.value for v in args] == [123, 0]
    args = calls[1][1]
    assert [type(v) for v in args] == [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    assert [v.value for v in args] == [37, signal.SIGTERM, None, 0]


@pytest.mark.parametrize("platform,machine,pointer_bytes,long_bytes,int_bytes", [
    ("darwin", "x86_64", 8, 8, 4), ("linux", "aarch64", 8, 8, 4),
    ("linux", "x86_64", 4, 8, 4), ("linux", "x86_64", 8, 4, 4),
    ("linux", "x86_64", 8, 8, 8),
])
def test_raw_calls_refuse_other_abis_before_loading_libc(monkeypatch, platform, machine, pointer_bytes, long_bytes, int_bytes):
    monkeypatch.setattr(p.sys, "platform", platform)
    monkeypatch.setattr(p.platform, "machine", lambda: machine)
    original = ctypes.sizeof
    monkeypatch.setattr(p.ctypes, "sizeof", lambda t: pointer_bytes if t is ctypes.c_void_p else long_bytes if t is ctypes.c_long else int_bytes if t is ctypes.c_int else original(t))
    monkeypatch.setattr(p.ctypes, "CDLL", lambda *a, **k: pytest.fail("unsupported raw ABI reached libc"))
    with pytest.raises(OSError) as caught:
        p._syscall(434, ctypes.c_int(123), ctypes.c_uint(0))
    assert caught.value.errno == errno.ENOTSUP


def test_raw_syscall_sets_variadic_signature_and_clears_errno(monkeypatch):
    monkeypatch.setattr(p, "_require_raw_abi", lambda: None)
    class Call:
        def __call__(self, nr, *args):
            assert self.argtypes == [ctypes.c_long]
            assert self.restype is ctypes.c_long
            assert isinstance(nr, ctypes.c_long) and nr.value == 434
            assert ctypes.get_errno() == 0
            assert [a.value for a in args] == [123, 0]
            return 37
    call = Call()
    class Lib:
        syscall = call
    def load(name, *, use_errno):
        assert name is None and use_errno is True
        return Lib()
    monkeypatch.setattr(p.ctypes, "CDLL", load)
    ctypes.set_errno(errno.EPERM)
    assert p._syscall(434, ctypes.c_int(123), ctypes.c_uint(0)) == 37


@pytest.mark.parametrize("error,kind", [(errno.ESRCH, ProcessLookupError), (errno.EPERM, PermissionError), (errno.ENOSYS, OSError)])
def test_raw_errno_preserves_oserror_subclass(monkeypatch, error, kind):
    monkeypatch.setattr(p, "_require_raw_abi", lambda: None)
    class Call:
        def __call__(self, *args):
            ctypes.set_errno(error)
            return -1
    class Lib:
        syscall = Call()
    monkeypatch.setattr(p.ctypes, "CDLL", lambda *a, **k: Lib())
    with pytest.raises(kind) as caught:
        p._syscall(424, ctypes.c_int(37), ctypes.c_int(0), ctypes.c_void_p(), ctypes.c_uint(0))
    assert caught.value.errno == error


@pytest.mark.parametrize("native", [True, False])
def test_enosys_open_falls_back_to_cloexec_proc_descriptor_and_probes_it(monkeypatch, native):
    monkeypatch.setattr(p.os, "pidfd_open", unavailable if native else None, raising=False)
    monkeypatch.setattr(p, "_syscall", unavailable)
    opened, sent = [], []
    monkeypatch.setattr(p.os, "open", lambda path, flags: opened.append((path, flags)) or 37)
    monkeypatch.setattr(p, "send_signal", lambda fd, sig: sent.append((fd, sig)))
    assert p.open_process(123) == 37
    assert opened == [("/proc/123", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)]
    assert sent == [(37, 0)]


@pytest.mark.parametrize("error", [errno.EPERM, errno.EINVAL, errno.EMFILE, errno.ESRCH])
def test_open_does_not_hide_non_enosys_errors(monkeypatch, error):
    def fail(pid):
        raise OSError(error, "deliberate failure")
    monkeypatch.setattr(p.os, "pidfd_open", fail, raising=False)
    monkeypatch.setattr(p, "_proc_descriptor", lambda pid: pytest.fail("non-ENOSYS must not use another path"))
    with pytest.raises(OSError) as caught:
        p.open_process(123)
    assert caught.value.errno == error


@pytest.mark.parametrize("failure", [OSError(errno.ENOSYS, "send unavailable"), PermissionError(errno.EPERM, "denied"), KeyboardInterrupt()])
def test_failed_proc_signal_probe_closes_descriptor(monkeypatch, failure):
    monkeypatch.setattr(p.os, "pidfd_open", unavailable, raising=False)
    monkeypatch.setattr(p.os, "open", lambda *a: 37)
    closed = []
    monkeypatch.setattr(p.os, "close", closed.append)
    def send(fd, sig):
        assert (fd, sig) == (37, 0)
        raise failure
    monkeypatch.setattr(p, "send_signal", send)
    with pytest.raises(type(failure)):
        p.open_process(123)
    assert closed == [37]


@pytest.mark.parametrize("proc_present,kind", [(True, ProcessLookupError), (False, FileNotFoundError)])
def test_vanished_proc_target_is_not_confused_with_missing_procfs(monkeypatch, proc_present, kind):
    monkeypatch.setattr(p.os, "pidfd_open", unavailable, raising=False)
    def fail(*args):
        raise FileNotFoundError(errno.ENOENT, "missing")
    monkeypatch.setattr(p.os, "open", fail)
    monkeypatch.setattr(p.os.path, "isdir", lambda path: proc_present if path == "/proc/self" else False)
    with pytest.raises(kind) as caught:
        p.open_process(123)
    assert caught.value.errno == (errno.ESRCH if proc_present else errno.ENOENT)


@pytest.mark.parametrize("operation,args", [
    ("open_process", (0,)), ("open_process", (-1,)), ("open_process", (1 << 32,)),
    ("send_signal", (-1, 0)), ("send_signal", (1 << 32, 0)),
    ("send_signal", (37, -1)), ("send_signal", (37, (1 << 32) + 9)),
])
def test_values_cannot_wrap_to_different_c_int_targets(monkeypatch, operation, args):
    monkeypatch.setattr(p.os, "pidfd_open", lambda *a: pytest.fail("invalid PID reached native API"), raising=False)
    monkeypatch.setattr(p.signal, "pidfd_send_signal", lambda *a: pytest.fail("invalid signal reached native API"), raising=False)
    monkeypatch.setattr(p, "_syscall", lambda *a: pytest.fail("invalid argument reached raw syscall"))
    with pytest.raises(ValueError):
        getattr(p, operation)(*args)


@pytest.mark.parametrize("error", [None, OSError(errno.ENOSYS, "no pidfd sending")])
def test_self_probe_checks_self_and_always_closes(monkeypatch, error):
    opened, sent, closed = [], [], []
    monkeypatch.setattr(p, "open_process", lambda pid: opened.append(pid) or 37)
    def send(fd, sig):
        sent.append((fd, sig))
        if error:
            raise error
    monkeypatch.setattr(p, "send_signal", send)
    monkeypatch.setattr(p.os, "close", closed.append)
    if error:
        with pytest.raises(OSError): p.probe_self()
    else:
        p.probe_self()
    assert opened == [os.getpid()] and sent == [(37, 0)] and closed == [37]


def test_subreaper_uses_compatibility_gate_before_inventory(monkeypatch):
    monkeypatch.setattr(s.sys, "platform", "linux")
    def unavailable_probe():
        raise OSError(errno.ENOSYS, "kernel descriptor send unavailable")
    monkeypatch.setattr(s.pidfd, "probe_self", unavailable_probe)
    monkeypatch.setattr(s, "process_rows", lambda: pytest.fail("ownership inventory reached before kernel capability gate"))
    with pytest.raises(StudyError, match="stable kernel process-descriptor ownership") as caught:
        with s.subreaper():
            pytest.fail("unsupported kernel started an ownership scope")
    assert isinstance(caught.value.__cause__, OSError)


def _force_proc_backend(monkeypatch, backend):
    if backend == "raw_send":
        if platform.machine() != "x86_64" or ctypes.sizeof(ctypes.c_void_p) != 8:
            pytest.skip("Raw fallback is restricted to the x86_64 node ABI")
        monkeypatch.setattr(p.signal, "pidfd_send_signal", None, raising=False)
    elif not callable(getattr(p.signal, "pidfd_send_signal", None)):
        pytest.skip("This Python does not expose native pidfd sending")
    monkeypatch.setattr(p.os, "pidfd_open", unavailable, raising=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Requires real Linux proc-FD signal syscall")
@pytest.mark.parametrize("backend", ["native_send", "raw_send"])
def test_native_proc_descriptor_stays_bound_after_target_is_reaped(monkeypatch, backend):
    _force_proc_backend(monkeypatch, backend)
    target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    decoy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    fd = None
    try:
        fd = p.open_process(target.pid)
        assert os.get_inheritable(fd) is False
        p.send_signal(fd, 0)
        target.terminate(); target.wait(timeout=5)
        with pytest.raises(ProcessLookupError):
            p.send_signal(fd, 0)
        assert decoy.poll() is None
    finally:
        if fd is not None: os.close(fd)
        for child in (target, decoy):
            if child.poll() is None: child.kill()
            child.wait(timeout=5)


def _bounded_cleanup_of_owned_test_children(since):
    # Covers a readiness/assertion failure before the retained leaf FD exists.
    # Only current ancestry plus a stable descriptor authorizes each signal.
    import time
    until = time.monotonic()+3
    while time.monotonic() < until:
        owned = s.owned_descendants(since)  # Reaps direct zombies with WNOHANG.
        if not owned:
            return
        for pid in owned:
            s.signal_descendant(pid, since, signal.SIGKILL)
        time.sleep(.01)
    pytest.fail("Native fixture cleanup retained owned children after its deadline")


@pytest.mark.skipif(sys.platform != "linux", reason="Requires real Linux proc-FD and subreaper behavior")
@pytest.mark.parametrize("backend", ["native_send", "raw_send"])
def test_native_proc_backend_cleans_term_ignoring_orphan(monkeypatch, tmp_path, backend):
    _force_proc_backend(monkeypatch, backend)
    monkeypatch.setattr(s.old, "_launch_record", lambda *a: None)
    child_code = "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(30)"
    leader_code = """import subprocess,sys,pathlib,time
p=subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[3]],start_new_session=True)
pathlib.Path(sys.argv[1]).write_text(str(p.pid))
until=time.monotonic()+5
while not pathlib.Path(sys.argv[3]).is_file() and time.monotonic()<until: time.sleep(.01)
if not pathlib.Path(sys.argv[3]).is_file(): raise SystemExit('leaf did not become TERM-ignoring')
"""
    import time
    began = time.time()
    child_pid, child_fd = None, None
    with s.subreaper():
        leader = subprocess.Popen([sys.executable, "-c", leader_code, str(tmp_path/'pid'), child_code, str(tmp_path/'ready')], start_new_session=True)
        try:
            leader.wait(timeout=8)
            assert leader.returncode == 0
            child_pid = int((tmp_path/'pid').read_text())
            child_fd = p.open_process(child_pid)
            stopped = s.recover_stop(tmp_path, "qwen9b", {"started_epoch": began, "last_services_before": None},
                                     max_seconds=5, term_seconds=.1, check_gpu=False)
            assert {"pid": child_pid, "signal": "SIGKILL"} in stopped['signals']
            assert s.owned_descendants(began) == {}
        finally:
            if leader.poll() is None: leader.kill()
            leader.wait(timeout=5)
            if child_fd is not None:
                try:
                    p.send_signal(child_fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    os.close(child_fd)
            _bounded_cleanup_of_owned_test_children(began)


def test_signal_descendant_uses_compatibility_adapter(monkeypatch):
    events = []
    monkeypatch.setattr(s.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(s.signal, "pidfd_send_signal", None, raising=False)
    monkeypatch.setattr(s, "owned_descendants", lambda since: events.append(("ancestry", since)) or {123: {"pid": 123}})
    monkeypatch.setattr(s.pidfd, "open_process", lambda pid: events.append(("open", pid)) or 37)
    monkeypatch.setattr(s.pidfd, "send_signal", lambda fd, sig: events.append(("send", fd, sig)))
    original = os.close
    monkeypatch.setattr(s.os, "close", lambda fd: events.append(("close", fd)) if fd == 37 else original(fd))
    assert s.signal_descendant(123, 1.25, signal.SIGTERM) is True
    assert events == [("ancestry", 1.25), ("open", 123), ("ancestry", 1.25), ("send", 37, signal.SIGTERM), ("close", 37)]


def test_captured_rhel8_capabilities_take_proc_descriptor_then_raw_send(monkeypatch):
    """Replay the node's absent Python APIs + ENOSYS open + working raw send."""
    monkeypatch.setattr(p.os, "pidfd_open", None, raising=False)
    monkeypatch.setattr(p.signal, "pidfd_send_signal", None, raising=False)
    events = []
    def raw(number, *args):
        events.append(("syscall", number, [a.value for a in args]))
        if number == 434:
            raise OSError(errno.ENOSYS, "node observed: Function not implemented")
        assert number == 424
        return 0
    monkeypatch.setattr(p, "_syscall", raw)
    monkeypatch.setattr(p.os, "open", lambda path, flags: events.append(("open", path, flags)) or 37)
    fd = p.open_process(123)
    p.send_signal(fd, signal.SIGTERM)
    assert events == [
        ("syscall", 434, [123, 0]),
        ("open", "/proc/123", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC),
        ("syscall", 424, [37, 0, None, 0]),
        ("syscall", 424, [37, signal.SIGTERM, None, 0]),
    ]
