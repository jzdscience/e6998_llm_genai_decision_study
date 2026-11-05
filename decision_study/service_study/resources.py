"""Read-only process and device observations for matched service cells.

Linux /proc counters are retained in their original units. Missing observations
are explicit, and a numeric backend thread limit is not called CPU affinity.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import threading
import time


PROC = Path("/proc")
CGROUP = Path("/sys/fs/cgroup")


def _read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _sysconf(name):
    try:
        return os.sysconf(name)
    except (ValueError, OSError):
        return None


def _cgroup_limits():
    membership = _read(PROC / "self/cgroup")
    roots = [CGROUP]
    if membership:
        for line in membership.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[:2] == ["0", ""]:
                relative = Path(parts[2].lstrip("/"))
                if ".." not in relative.parts:
                    roots.insert(0, CGROUP / relative)
    result = {"membership": membership, "limits": {}}
    for root in dict.fromkeys(roots):
        for name in ("cpu.max", "cpuset.cpus.effective", "memory.max", "memory.high",
                     "memory.current", "memory.events"):
            value = _read(root / name)
            if value is not None:
                result["limits"][str(root / name)] = value
    return result


def hardware():
    """Capture host/allocation context without assuming an exclusive CPU node."""
    cpuinfo = _read(PROC / "cpuinfo")
    models = sorted({line.split(":", 1)[1].strip() for line in (cpuinfo or "").splitlines()
                     if line.startswith("model name") and ":" in line})
    status = _read(PROC / "self/status")
    allowed = next((line.split(":", 1)[1].strip() for line in (status or "").splitlines()
                    if line.startswith("Cpus_allowed_list:")), None)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "cpu_models": models,
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "controller_affinity": affinity, "cpus_allowed_list": allowed,
        "clock_ticks_per_second": _sysconf("SC_CLK_TCK"),
        "page_size_bytes": _sysconf("SC_PAGESIZE"), "cgroup": _cgroup_limits(),
        "slurm": {name: os.environ.get(name) for name in (
            "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
            "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS")},
        "note": "Thread limits do not establish CPU exclusivity. GPU samples cover the device, not only owned processes. No price or energy integral is inferred.",
    }


def _parse_stat(raw):
    """Parse /proc/PID/stat including command names containing spaces or ')'."""
    left, right = raw.find("("), raw.rfind(")")
    if left < 1 or right < left:
        raise ValueError("Malformed proc stat")
    pid = int(raw[:left].strip())
    fields = raw[right + 1:].split()
    if len(fields) < 22:
        raise ValueError("Incomplete proc stat")
    return {"pid": pid, "command": raw[left + 1:right], "state": fields[0],
            "ppid": int(fields[1]), "user_cpu_ticks": int(fields[11]),
            "system_cpu_ticks": int(fields[12]), "start_time_ticks": int(fields[19]),
            "virtual_bytes": int(fields[20]), "rss_pages": int(fields[21])}


def _scan_processes():
    if not PROC.is_dir():
        return {}, ["Linux /proc is unavailable"]
    table, errors = {}, []
    try:
        paths = list(PROC.iterdir())
    except OSError as exc:
        return {}, [f"proc inventory failed: {type(exc).__name__}"]
    for path in paths:
        if not path.name.isdigit():
            continue
        try:
            record = _parse_stat((path / "stat").read_text())
            table[record["pid"]] = record
        except (OSError, ValueError, UnicodeError):
            # Most are unrelated processes exiting during inventory. Keep a
            # count rather than labeling unobserved owned counters as zero.
            errors.append(path.name)
    return table, errors


def _number(value):
    try:
        return float(value.strip())
    except ValueError:
        return None


def _gpu_sample():
    fields = ("index", "uuid", "name", "memory.used", "memory.total", "utilization.gpu", "power.draw")
    command = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=2, check=True)
        rows = []
        for values in csv.reader(result.stdout.splitlines()):
            if len(values) != len(fields):
                raise ValueError("Unexpected GPU inventory columns")
            values = [value.strip() for value in values]
            rows.append({"index": int(values[0]), "uuid": values[1], "name": values[2],
                         "used_mib": _number(values[3]), "total_mib": _number(values[4]),
                         "utilization_percent": _number(values[5]), "power_watts": _number(values[6])})
        return {"status": "ok", "gpus": rows, "raw_csv": result.stdout,
                "query_seconds": time.monotonic() - started}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "gpus": None, "error": type(exc).__name__,
                "query_seconds": time.monotonic() - started}


class ResourceMonitor:
    """Journal one-second observations of roots plus their process descendants."""

    def __init__(self, owned_pids, path):
        self.roots = sorted(set(int(pid) for pid in owned_pids))
        if any(pid <= 0 for pid in self.roots):
            raise ValueError("Owned process IDs must be positive")
        self.path = Path(path)
        self.stop = threading.Event()
        self.thread = None
        self.handle = None
        self.identities = {}
        self.sample_count = 0
        self.errors = []

    def _sample(self):
        table, inventory_errors = _scan_processes()
        selected, unavailable = set(), []
        for pid in self.roots:
            row = table.get(pid)
            if row is None:
                unavailable.append({"pid": pid, "reason": "missing_or_unreadable"})
                continue
            start = row["start_time_ticks"]
            if pid in self.identities and self.identities[pid] != start:
                unavailable.append({"pid": pid, "reason": "pid_reused"})
                continue
            self.identities[pid] = start
            selected.add(pid)
        # Continue watching previously observed descendants after reparenting,
        # but only while their kernel process-start identity still matches.
        selected.update(pid for pid, start in self.identities.items()
                        if pid in table and table[pid]["start_time_ticks"] == start)
        while True:
            children = {pid for pid, row in table.items() if row["ppid"] in selected}
            fresh = children - selected
            if not fresh:
                break
            selected.update(fresh)
            self.identities.update({pid: table[pid]["start_time_ticks"] for pid in fresh})
        processes = [table[pid] for pid in sorted(selected)]
        page_size = _sysconf("SC_PAGESIZE")
        for row in processes:
            row["rss_bytes"] = row["rss_pages"] * page_size if page_size is not None else None
        gpu = _gpu_sample()
        value = {"sample_index": self.sample_count, "observed_at": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": time.monotonic() - self.started,
                 "owned_root_pids": self.roots, "processes": processes,
                 "unavailable_roots": unavailable, "proc_inventory_unreadable_count": len(inventory_errors),
                 "proc_inventory_error": inventory_errors[0] if not table and inventory_errors else None,
                 "clock_ticks_per_second": _sysconf("SC_CLK_TCK"), "page_size_bytes": page_size,
                 "gpu": gpu, "system_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
        self.handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        self.sample_count += 1

    def _loop(self):
        while not self.stop.wait(1):
            try:
                self._sample()
            except Exception as exc:
                self.errors.append({"type": type(exc).__name__, "message": str(exc)})
                self.stop.set()

    def __enter__(self):
        self.handle = self.path.open("x", encoding="utf-8")
        self.started = time.monotonic()
        try:
            self._sample()
        except BaseException:
            self.handle.close()
            raise
        self.thread = threading.Thread(target=self._loop, name="service-resource-monitor", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            # Do not close a handle still being written by the sampling thread.
            raise RuntimeError("Resource sampler failed to stop")
        try:
            self.handle.close()
        finally:
            if self.errors and exc_type is None:
                raise RuntimeError(f"Resource sampler failed: {self.errors}")
