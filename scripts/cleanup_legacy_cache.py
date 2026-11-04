#!/usr/bin/env python3
"""Remove only three user-approved legacy Hugging Face model caches.

Standard library only. Default is a read-only plan; --apply deletes the exact
allowlisted repository directories. Stop model/download jobs before applying.
The process check recognizes common CLIs, not every possible custom program.
No environment, current experiment model, or other cache is selected.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess


APPROVED_REPOSITORIES = (
    "models--mistralai--Mistral-7B-Instruct-v0.2",
    "models--google--gemma-7b-it",
    "models--meta-llama--Llama-2-13b-chat-hf",
)


def _checked_root(cache_dir: Path) -> Path:
    root = Path(cache_dir).expanduser().absolute()
    if ".." in root.parts or root.name != "hub":
        raise ValueError("Specify the actual Hugging Face hub directory, without '..'")
    # Do not resolve away a symlink before checking it.
    for component in [*reversed(root.parents), root]:
        if component.is_symlink():
            raise ValueError(f"Refusing a cache path through a symlink: {component}")
        if component.exists() and not component.is_dir():
            raise ValueError(f"Cache path component is not a directory: {component}")
    return root


def active_model_processes() -> list[dict]:
    """Inspect this user's common model/download CLI processes, without killing."""
    result = subprocess.run(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,args="],
        text=True, capture_output=True, check=True, timeout=10,
    )
    active = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
            continue
        try:
            argv = shlex.split(parts[1])
        except ValueError:
            argv = parts[1].split()
        if not argv:
            continue
        executable = Path(argv[0]).name
        is_python = executable.startswith("python")
        names = {Path(arg).name for arg in argv}
        family = None
        if executable in {"hf", "huggingface-cli"} and "download" in argv:
            family = "Hugging Face download"
        elif executable == "startlux-decide" or (is_python and "startlux-decide" in names):
            family = "StartLux"
        elif executable == "vllm" or executable.startswith("VLLM::"):
            family = "vLLM"
        elif is_python:
            if "model_probe.py" in names or ("decision_study" in argv and {"run", "install", "repair-python"}.intersection(argv)):
                family = "decision-study model job"
            elif "kev.serve" in argv:
                family = "Kev"
            elif any(arg.startswith("vllm.") for arg in argv):
                family = "vLLM"
            elif any(repo.removeprefix("models--").replace("--", "/") in arg
                     for repo in APPROVED_REPOSITORIES for arg in argv):
                family = "approved legacy model"
        if family:
            # Do not include full command lines: they may contain unrelated data.
            active.append({"pid": int(parts[0]), "family": family})
    return active


def cleanup_cache(cache_dir: Path, *, apply: bool = False) -> dict:
    root = _checked_root(cache_dir)
    report = {
        "mode": "apply" if apply else "dry_run",
        "cache_dir": str(root),
        "targets": [],
        "notice": "Only the three named repository directories are eligible. Filesystem free-space changes are not a measured quota reduction or guaranteed reclaimed bytes.",
    }
    if not root.exists():
        report["targets"] = [{"path": str(root / name), "status": "missing"} for name in APPROVED_REPOSITORIES]
        return report
    report["filesystem_free_bytes_before"] = shutil.disk_usage(root).free
    if apply:
        active = active_model_processes()
        if active:
            raise RuntimeError(f"Stop model/download jobs before cleanup; detected: {active}")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise RuntimeError("This Python/platform lacks fd-safe directory removal; no cleanup performed")
    # Anchor deletion to the validated directory descriptor. HF snapshot symlinks
    # inside a selected repository are unlinked, never traversed by rmtree.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, flags)
    try:
        for name in APPROVED_REPOSITORIES:
            try:
                info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                report["targets"].append({"path": str(root / name), "status": "missing"})
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Refusing non-directory or symlink target: {root / name}")
            report["targets"].append({"path": str(root / name), "status": "planned"})
        # Validate all targets before removing any of them.
        if apply:
            for target in report["targets"]:
                if target["status"] == "planned":
                    shutil.rmtree(Path(target["path"]).name, dir_fd=root_fd)
                    target["status"] = "deleted"
    finally:
        os.close(root_fd)
    report["filesystem_free_bytes_after"] = shutil.disk_usage(root).free
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--apply", action="store_true", help="Delete the three exact approved repositories; otherwise show a read-only plan")
    args = parser.parse_args(argv)
    try:
        result = cleanup_cache(args.cache_dir, apply=args.apply)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc),
                          "notice": "No non-allowlisted directory is selected. If a filesystem error occurred during deletion, rerunning the dry-run shows remaining targets."}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
