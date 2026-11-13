"""CPU-only native Linux ownership gate, run before any GPU profile worker.

The supervised actor double-forks a separate-session, TERM-ignoring leaf. A sibling
decoy born AFTER the actor's start is deliberately named in stale launch metadata;
the actor must stop its actual descendants while that decoy stays alive. An outer
subreaper owns every self-test process for cleanup even if the actor fails.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..common import StudyError, file_hash, fingerprint, read_json, utc_now
from ..service_study.identity import fixed_write
from . import study as s

CHILD = r'''
import os,pathlib,signal,subprocess,sys,time
folder=pathlib.Path(sys.argv[1]); stage=sys.argv[2]; script=sys.argv[3]
(folder/(stage+'.pid')).write_text(str(os.getpid()))
if stage=='leaf':
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 (folder/'leaf.ready').write_text('ready')
 time.sleep(30)
elif stage=='middle':
 subprocess.Popen([sys.executable,'-c',script,str(folder),'leaf',script],start_new_session=True)
 until=time.monotonic()+5
 while not (folder/'leaf.ready').exists() and time.monotonic()<until: time.sleep(.01)
else:
 subprocess.Popen([sys.executable,'-c',script,str(folder),'middle',script],start_new_session=True)
 until=time.monotonic()+5
 while not (folder/'leaf.ready').exists() and time.monotonic()<until: time.sleep(.01)
 time.sleep(.3)
'''
DECOY = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"


def wait_file(path, seconds=10):
    until = time.monotonic()+seconds
    while time.monotonic() < until:
        if path.is_file():
            return
        time.sleep(.02)
    raise StudyError(f"Ownership self-test did not produce {path.name} on time")


def actor(folder):
    folder = s.normal(folder)
    began = time.time()
    fixed_write(folder/"actor-ready.json", {"pid": os.getpid(), "began_epoch": began})
    wait_file(folder/"decoy.json")
    decoy = read_json(folder/"decoy.json")
    if decoy["created_epoch"] <= began:
        raise StudyError("Self-test decoy must be born after this actor starts")
    # This process is not a child of the actor. The recorded numeric IDs are
    # intentionally not authority to stop it, even though it is newly born.
    fixed_write(s.safe(s.control(folder, "qwen9b"), "runtime/last_services.json"),
                {"started_at": utc_now(), "processes": [{"pid": decoy["pid"], "name": "qwen-engine"}]})
    plan = {"started_epoch": began, "last_services_before": None}
    worker = None
    with s.subreaper():
        try:
            worker = subprocess.Popen([sys.executable, "-c", CHILD, str(folder), "worker", CHILD], start_new_session=True)
            wait_file(folder/"leaf.ready")
            leaf = int((folder/"leaf.pid").read_text())
            worker.wait(timeout=10)
            if worker.returncode != 0 or leaf not in s.owned_descendants(began):
                raise StudyError("Self-test orphan was not adopted by the actor")
            stopped = s.recover_stop(folder, "qwen9b", plan, worker_pid=decoy["pid"],
                                     journal=folder/"actor-cleanup.jsonl", max_seconds=8, term_seconds=.1, check_gpu=False)
            if not any(r.get("pid") == leaf and r.get("signal") == "SIGKILL" for r in stopped["signals"]):
                raise StudyError("TERM-ignoring orphan was not stopped by the verified pidfd path")
            os.kill(decoy["pid"], 0)  # Observation only; the outer parent also checks its live Popen.
            if s.owned_descendants(began):
                raise StudyError("Self-test descendant tree is not empty")
            fixed_write(folder/"actor-result.json", {"status": "PASS", "leaf_pid": leaf, "decoy_pid": decoy["pid"],
                        "decoy_survived": True, "stop": stopped, "scope": "CPU process ownership only; no GPU calls"})
        finally:
            if worker is not None:
                s.stop_child(worker, began)
            s.recover_stop(folder, "qwen9b", plan, max_seconds=8, term_seconds=.1, check_gpu=False)


def run_selftest(output):
    settings, _ = s.context(output)
    sess = s.session(output)
    folder = s.safe(output, f"ownership-selftest/{time.time_ns()}")
    folder.mkdir(parents=True)
    plan = {"started_epoch": time.time(), "last_services_before": None}
    supervisor, decoy = None, None
    with s.subreaper():
        try:
            with (folder/"actor.log").open("x") as log:
                supervisor = subprocess.Popen([sys.executable, "-B", "-m", "decision_study.learnability_profile.ownership",
                                               "--actor", str(folder)], cwd=s.PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            wait_file(folder/"actor-ready.json")
            created = time.time()
            decoy = subprocess.Popen([sys.executable, "-c", DECOY], start_new_session=True)
            fixed_write(folder/"decoy.json", {"pid": decoy.pid, "created_epoch": created})
            supervisor.wait(timeout=25)
            if supervisor.returncode != 0 or decoy.poll() is not None:
                raise StudyError("Linux ownership self-test failed or signalled its unrelated decoy; no GPU started")
            result = read_json(folder/"actor-result.json")
            if result.get("status") != "PASS" or result.get("decoy_pid") != decoy.pid or result.get("decoy_survived") is not True:
                raise StudyError("Ownership actor receipt differs")
        finally:
            if supervisor is not None:
                s.stop_child(supervisor, plan["started_epoch"])
            stopped = s.recover_stop(folder, "qwen9b", plan, journal=folder/"outer-cleanup.jsonl",
                                     max_seconds=8, term_seconds=.1, check_gpu=False)
            fixed_write(folder/"outer-stop.json", stopped)
    s.session(output)
    receipt = {"status": "PASS", "settings_hash": fingerprint(settings), "allocation": sess["allocation"],
               "folder": str(folder.relative_to(s.normal(output))), "scope": "Native Linux pidfd/subreaper process ownership; no GPU/model validation",
               "files": {name: file_hash(folder/name) for name in
                         ("actor-ready.json", "decoy.json", "actor-result.json", "actor-cleanup.jsonl", "outer-cleanup.jsonl", "outer-stop.json")}}
    fixed_write(folder/"receipt.json", receipt)
    # Pointer only; each complete self-test directory and receipt is immutable.
    from ..common import write_json
    write_json(s.safe(output, "runtime/ownership-selftest.json"), {"file": str((folder/"receipt.json").relative_to(s.normal(output))),
                                                                 "sha256": file_hash(folder/"receipt.json")})
    return receipt


def require_receipt(output):
    settings, _ = s.context(output)
    sess = s.session(output)
    pointer = read_json(s.safe(output, "runtime/ownership-selftest.json"))
    verify_receipt(output, pointer, settings, sess["allocation"])
    return pointer


def verify_receipt(output, pointer, settings, allocation):
    """Verify one immutable receipt, including from an older accepted role's plan."""
    path = s.safe(output, pointer["file"])
    if file_hash(path) != pointer["sha256"]:
        raise StudyError("Native Linux ownership self-test receipt changed")
    value = read_json(path)
    if (value.get("status") != "PASS" or value.get("settings_hash") != fingerprint(settings)
            or value.get("allocation") != allocation or path.parent != s.safe(output, value["folder"])):
        raise StudyError("Native Linux ownership self-test is absent or belongs to other settings/allocation")
    names = {"actor-ready.json", "decoy.json", "actor-result.json", "actor-cleanup.jsonl", "outer-cleanup.jsonl", "outer-stop.json"}
    if set(value.get("files", {})) != names or any(file_hash(s.safe(path.parent, name)) != digest
                                                 for name, digest in value["files"].items()):
        raise StudyError("Native Linux ownership self-test evidence changed")
    actor_result = read_json(path.parent/"actor-result.json")
    if actor_result.get("status") != "PASS" or actor_result.get("decoy_survived") is not True:
        raise StudyError("Ownership self-test actor did not pass")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    args = parser.parse_args()
    actor(args.actor)


if __name__ == "__main__":
    main()
