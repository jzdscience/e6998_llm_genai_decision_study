"""Stable Linux process descriptors, including RHEL8's proc-directory backend.

Some Python builds omit pidfd APIs and some kernels implement pidfd_send_signal
but not pidfd_open. Linux permits an open /proc/PID directory as the descriptor
for pidfd_send_signal. That still binds the kernel process identity: it never
falls back to signalling a numeric PID.

References: Linux pidfd_send_signal(2), and the upstream x86 syscall_64.tbl
(pidfd_send_signal=424, pidfd_open=434). Raw calls are limited to that ABI.
"""
import ctypes
import errno
import operator
import os
import platform
import signal
import sys


_INT_MAX = (1 << 31) - 1


def _integer(value, name, minimum=0):
    value = operator.index(value)
    if not minimum <= value <= _INT_MAX:
        raise ValueError(f"{name} is outside the supported C int range")
    return value


def _require_raw_abi():
    if (sys.platform != "linux" or platform.machine() != "x86_64"
            or ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8
            or ctypes.sizeof(ctypes.c_int) != 4):
        raise OSError(errno.ENOTSUP, "Raw pidfd calls require Linux x86_64 LP64")


def _syscall(number, *arguments):
    _require_raw_abi()
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.syscall
    call.argtypes = [ctypes.c_long]  # syscall is variadic; every remaining argument is explicitly typed.
    call.restype = ctypes.c_long
    ctypes.set_errno(0)
    result = call(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def send_signal(fd, sig):
    """Signal the process held by fd, never a recycled numeric PID."""
    fd, sig = _integer(fd, "fd"), _integer(sig, "signal")
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        native(fd, sig)
    else:
        _syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def _proc_descriptor(pid):
    try:
        fd = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except FileNotFoundError as exc:
        # A vanished target is normal during cleanup. A missing procfs itself
        # is an infrastructure error and must not masquerade as a stopped PID.
        if os.path.isdir("/proc/self"):
            raise ProcessLookupError(errno.ESRCH, "Process disappeared before its descriptor opened") from exc
        raise
    try:
        send_signal(fd, 0)  # Require actual kernel support for this descriptor type.
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_process(pid):
    """Open a CLOEXEC kernel descriptor; only ENOSYS permits the proc fallback."""
    pid = _integer(pid, "pid", minimum=1)
    native = getattr(os, "pidfd_open", None)
    try:
        if callable(native):
            return native(pid)
        # pidfd_open sets FD_CLOEXEC atomically, including this raw call.
        return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    except OSError as exc:
        if exc.errno != errno.ENOSYS:
            raise
    return _proc_descriptor(pid)


def probe_self():
    """Check real kernel capability before any ownership worker is created."""
    fd = open_process(os.getpid())
    try:
        send_signal(fd, 0)
    finally:
        os.close(fd)
