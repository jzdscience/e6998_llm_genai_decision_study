#!/usr/bin/env python3
"""Verify an isolated matching CUDA 13.0 toolkit and run FlashInfer sampling.

Run with the recorded vLLM interpreter, before loading model weights. This does
not install packages or alter their files. CUDA wheels are staged separately by
prepare_cuda_toolkit.py, never added to Python's module search path. Linker aliases live in --work-dir;
the result records the exact environment to reuse for the subsequent smoke.

Pinned source references:
https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.7.0.post1/flashinfer/jit/cpp_ext.py
https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.7.0.post1/flashinfer/jit/core.py
https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.7.0.post1/flashinfer/sampling.py
https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import traceback


PINS = {
    "nvidia-cuda-nvcc": "13.4.92",
    "nvidia-cuda-crt": "13.4.92",
    "nvidia-nvvm": "13.4.92",
    "nvidia-cuda-runtime": "13.0.96",
    "ninja": "1.13.2",
    "flashinfer-python": "0.7.0.post1",
    "torch": "2.13.0",
}
TOOLKIT_PINS = {"nvidia-cuda-nvcc": "13.0.88", "nvidia-cuda-crt": "13.0.88",
                "nvidia-nvvm": "13.0.88", "nvidia-cuda-runtime": "13.0.96",
                "nvidia-curand": "10.4.0.35"}
ENV_KEYS = (
    "CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC", "PATH", "MAX_JOBS",
    "FLASHINFER_NVCC_THREADS", "FLASHINFER_EXTRA_LDFLAGS",
)


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _compile_path(path: Path) -> None:
    # The pinned FlashInfer writer rejoins parsed flags without shell/Ninja
    # quoting. Reject unsupported paths rather than claiming shlex.quote fixes it.
    if not re.fullmatch(r"[A-Za-z0-9_./+-]+", str(path)):
        raise RuntimeError(
            f"FlashInfer's pinned build writer cannot safely quote this path: {path}. "
            "Use a toolkit/work directory without spaces or shell/Ninja metacharacters.")


def _toolkit_distributions(target: Path, prefix: Path) -> dict:
    if _inside(target, prefix) or _inside(prefix, target):
        raise RuntimeError("CUDA toolkit must be separate from the frozen model environment")
    distributions = {}
    for dist in metadata.distributions(path=[str(target)]):
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name in distributions:
            raise RuntimeError(f"Duplicate isolated toolkit distribution: {name}")
        distributions[name] = dist
    versions = {name: dist.version for name, dist in distributions.items()}
    if versions != TOOLKIT_PINS:
        raise RuntimeError(f"Isolated toolkit pins differ: {versions}")
    return distributions


def _package_file(package: str, relative: str, prefix: Path, distributions: dict) -> Path:
    dist = distributions[package]
    entries = [p for p in (dist.files or ()) if p.as_posix() == relative]
    if len(entries) != 1:
        raise RuntimeError(f"Expected one recorded {package} file: {relative}")
    path = Path(dist.locate_file(entries[0])).resolve(strict=True)
    if not _inside(path, prefix) or not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Invalid or external package file: {path}")
    return path


def _resolve_toolkit(prefix: Path, distributions: dict) -> tuple[Path, dict[str, str]]:
    required = {
        "nvcc": ("nvidia-cuda-nvcc", "bin/nvcc"),
        "nvcc_profile": ("nvidia-cuda-nvcc", "bin/nvcc.profile"),
        "ptxas": ("nvidia-cuda-nvcc", "bin/ptxas"),
        "cicc": ("nvidia-nvvm", "nvvm/bin/cicc"),
        "libdevice": ("nvidia-nvvm", "nvvm/libdevice/libdevice.10.bc"),
        "cuda_header": ("nvidia-cuda-runtime", "include/cuda.h"),
        "runtime_header": ("nvidia-cuda-runtime", "include/cuda_runtime.h"),
        "runtime_api_header": ("nvidia-cuda-runtime", "include/cuda_runtime_api.h"),
        "crt_header": ("nvidia-cuda-crt", "include/crt/host_config.h"),
        "cudart": ("nvidia-cuda-runtime", "lib/libcudart.so.13"),
        "cudadevrt": ("nvidia-cuda-runtime", "lib/libcudadevrt.a"),
        "curand_header": ("nvidia-curand", "include/curand.h"),
        "curand_kernel_header": ("nvidia-curand", "include/curand_kernel.h"),
        "curand_philox_header": ("nvidia-curand", "include/curand_philox4x32_x.h"),
    }
    files = {name: _package_file(pkg, f"nvidia/cu13/{rel}", prefix, distributions)
             for name, (pkg, rel) in required.items()}
    root = files["nvcc"].parent.parent
    for name, (_, relative) in required.items():
        if files[name] != (root / relative).resolve(strict=True):
            raise RuntimeError(f"CUDA wheel components do not share one toolkit: {name}")
    for name in ("nvcc", "ptxas", "cicc"):
        if not os.access(files[name], os.X_OK):
            raise RuntimeError(f"CUDA executable is not executable: {files[name]}")
    return root, {name: str(path) for name, path in files.items()}


def _driver_library() -> Path:
    # Resolve the library actually selected by the loader, not a toolkit stub.
    driver = ctypes.CDLL("libcuda.so.1")
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libcuda.so"):
            paths.add(Path(fields[5]).resolve(strict=True))
    if len(paths) != 1:
        raise RuntimeError(f"Cannot identify one loaded CUDA driver library: {sorted(map(str, paths))}")
    path = paths.pop()
    if not path.is_file() or "stubs" in path.parts or not hasattr(driver, "cuInit"):
        raise RuntimeError(f"Invalid CUDA driver library: {path}")
    return path


def _linker_view(work_dir: Path, cudart: Path, driver: Path) -> Path:
    view = work_dir / "linker"
    if work_dir.is_symlink() or view.is_symlink():
        raise RuntimeError("Refusing a symlinked toolchain work directory")
    view.mkdir(parents=True, exist_ok=True)
    for name, target in (("libcudart.so", cudart), ("libcuda.so", driver)):
        link = view / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve(strict=True) != target:
                raise RuntimeError(f"Existing linker alias differs; preserved: {link}")
        else:
            link.symlink_to(target)
    return view


def _command_version(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return (result.stdout + result.stderr).strip()


def _check_compiler_headers(nvcc_version: str, cuda_header: Path, runtime_header: Path) -> dict:
    releases = re.findall(r"release (\d+\.\d+),", nvcc_version)
    headers = re.findall(r"^\s*#\s*define\s+CUDA_VERSION\s+(\d+)\b", cuda_header.read_text(), re.M)
    runtime_headers = re.findall(r"^\s*#\s*define\s+CUDART_VERSION\s+(\d+)\b", runtime_header.read_text(), re.M)
    if releases != ["13.0"] or headers != ["13000"] or runtime_headers != ["13000"]:
        raise RuntimeError(f"CUDA compiler/header mismatch: nvcc={releases}, CUDA_VERSION={headers}, CUDART_VERSION={runtime_headers}; expected 13.0/13000/13000")
    return {"nvcc_release": releases[0], "CUDA_VERSION": int(headers[0]),
            "CUDART_VERSION": int(runtime_headers[0]), "matched": True}


def run_probe(work_dir: Path, report: dict, toolkit_target: Path) -> dict:
    if sys.platform != "linux" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Use the recorded Linux vLLM virtual-environment interpreter")
    prefix = Path(sys.prefix).resolve(strict=True)
    _compile_path(prefix)
    _compile_path(Path(sys.base_prefix).resolve(strict=True))
    _compile_path(work_dir)
    if toolkit_target.is_symlink():
        raise RuntimeError("Refusing a symlinked isolated toolkit target")
    toolkit_target = toolkit_target.resolve(strict=True)
    _compile_path(toolkit_target)
    report["python"] = sys.executable
    report["prefix"] = str(prefix)
    report["versions"] = {name: metadata.version(name) for name in PINS}
    if report["versions"] != PINS:
        raise RuntimeError(f"Installed versions differ from the incident's recorded pins: {report['versions']}")
    report["previous_environment"] = {key: os.environ.get(key) for key in ENV_KEYS if key != "PATH"}
    extras = {key: value for key, value in os.environ.items()
              if key.startswith("FLASHINFER_EXTRA_") and value.strip()}
    if extras:
        report["inherited_flashinfer_flags"] = extras
        raise RuntimeError("Unexpected inherited FLASHINFER_EXTRA_* flags; inspect them before compiling")
    distributions = _toolkit_distributions(toolkit_target, prefix)
    report["toolkit_target"] = str(toolkit_target)
    report["toolkit_versions"] = {name: dist.version for name, dist in distributions.items()}
    root, files = _resolve_toolkit(toolkit_target, distributions)
    _compile_path(root)
    report["toolkit_files"] = files
    driver = _driver_library()
    report["driver_library"] = str(driver)
    view = _linker_view(work_dir, Path(files["cudart"]), driver)
    report["linker_view"] = str(view)
    environment = {
        "CUDA_HOME": str(root), "CUDA_PATH": str(root),
        "FLASHINFER_NVCC": files["nvcc"],
        "PATH": os.pathsep.join((str(prefix / "bin"), str(root / "bin"), os.environ.get("PATH", ""))),
        "MAX_JOBS": "2", "FLASHINFER_NVCC_THREADS": "1",
        "FLASHINFER_EXTRA_LDFLAGS": "-L" + str(view),
    }
    os.environ.update(environment)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    report["environment"] = environment
    ninja = shutil.which("ninja")
    expected_ninja = prefix / "bin/ninja"
    if not ninja or Path(ninja).resolve() != expected_ninja.resolve(strict=True):
        raise RuntimeError("Ninja does not resolve to the recorded vLLM environment")
    cxx = shlex.split(os.environ.get("CXX", "c++"))
    if len(cxx) != 1 or not shutil.which(cxx[0]):
        raise RuntimeError("CXX must identify one available host compiler executable")
    if os.environ.get("CC"):
        cc = shlex.split(os.environ["CC"])
        if len(cc) != 1 or not shutil.which(cc[0]):
            raise RuntimeError("CC must identify one available host compiler executable")
    report["compiler_versions"] = {
        "nvcc": _command_version([files["nvcc"], "--version"]),
        "ninja": _command_version([ninja, "--version"]),
        "cxx": _command_version([cxx[0], "--version"]),
    }
    report["compiler_header_check"] = _check_compiler_headers(
        report["compiler_versions"]["nvcc"], Path(files["cuda_header"]), Path(files["runtime_api_header"]))
    # All discovery overrides are set before these modules cache CUDA paths.
    import torch
    import flashinfer
    from flashinfer.jit.cpp_ext import get_cuda_path, get_cuda_version

    if Path(get_cuda_path()).resolve() != root:
        raise RuntimeError("FlashInfer did not select the verified CUDA toolkit")
    report["torch_cuda"] = torch.version.cuda
    report["flashinfer_cuda"] = str(get_cuda_version())
    if torch.version.cuda != "13.0" or report["flashinfer_cuda"] != "13.0":
        raise RuntimeError("Unexpected CUDA runtime/compiler versions; inspect before loading models")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the recorded vLLM environment")
    report["gpu"] = {"name": torch.cuda.get_device_name(0),
                     "capability": list(torch.cuda.get_device_capability(0))}
    logits = torch.full((2, 128), -float("inf"), dtype=torch.float32, device="cuda:0")
    logits[0, 3], logits[1, 7] = 0, 0
    # Per-row tensors take the sampling module used by vLLM, avoiding scalar
    # top-k fast paths. One-hot mass gives exact, non-random expected tokens.
    top_k = torch.tensor([1, 1], dtype=torch.int32, device="cuda:0")
    top_p = torch.ones(2, dtype=torch.float32, device="cuda:0")
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, filter_apply_order="top_k_first", deterministic=True)
    torch.cuda.synchronize()
    observed = samples.cpu().tolist()
    if observed != [3, 7]:
        raise RuntimeError(f"FlashInfer one-hot sampling check failed: {observed}")
    report["sampling"] = {"status": "complete", "shape": [2, 128],
                          "expected": [3, 7], "observed": observed,
                          "cuda_synchronized": True}
    report["findings"] = [
        "Isolated CUDA 13.0 build wheels selected; model environment and source unchanged.",
        "External linker aliases point to the isolated CUDART and loaded NVIDIA driver.",
        "Compiler and header minor versions match; actual sampling JIT and execution passed.",
        "This small kernel check is not a passed Qwen model smoke or a performance measurement.",
    ]
    report["status"] = "complete"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--toolkit-target", type=Path, required=True)
    args = parser.parse_args(argv)
    result = args.result.expanduser().absolute()
    if result.exists() or result.is_symlink():
        raise RuntimeError(f"Result path already exists; use a unique path: {result}")
    report = {"schema_version": 1, "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        run_probe(args.work_dir.expanduser().absolute(), report, args.toolkit_target.expanduser().absolute())
    except BaseException as exc:
        report.update(status="blocked", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        result.parent.mkdir(parents=True, exist_ok=True)
        temporary = result.with_name(result.name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(result)
    print(json.dumps({"status": report["status"], "result": str(result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
