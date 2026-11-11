"""Fake external boundaries for the runtime-recovery rehearsal; everything else runs for real.

Faked: uv, the environment interpreter (its ``pip freeze --all`` imitates pip's egg-info discovery
through PYTHONPATH), pip inside the real toolkit installer, and Hugging Face downloads (a real cache
layout of content-addressed blobs and snapshot links). Real: git (file:// remotes), ZIP, files,
locks, processes, the frozen verifiers, scripts/prepare_cuda_toolkit.py and scripts/model_probe.py.

Shims call:  python -I recover_node_runtime_fakes.py WORLD.json {uv|python ENV_PYTHON} ARGS...
"""
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import types


def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def freeze_text(lines):
    """Order pip freeze uses for these inventories: case-insensitive canonical project name."""
    return "".join(line + "\n" for line in sorted(lines, key=lambda l: canon(re.split(r"[ =@]", l)[0])))


def portable_text(freeze):
    """The frozen _portable_requirements rule, restated for building the fake archive."""
    keep = [l for l in freeze.splitlines() if l.strip() and " @ file:" not in l
            and not re.match(r"(?i)^decision[-_.]study(?:\s|\[|[=<>!~@]|$)", l.strip())]
    return "\n".join(keep) + "\n"


def blob_id(name, data):
    if name.endswith(".safetensors"):
        return hashlib.sha256(data).hexdigest()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def make_snapshot(hub, repo, revision, files):
    repo_dir = Path(hub) / ("models--" + repo.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    (repo_dir / "blobs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    for name, text in sorted(files.items()):
        data = text.encode()
        blob = repo_dir / "blobs" / blob_id(name, data)
        if not blob.exists():
            blob.write_bytes(data)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            os.symlink(os.path.relpath(blob, link.parent), link)
    return str(snapshot)


def fake_pip_target(target, requirements):
    """What the pinned toolkit installer's pip --target would leave: packages plus dist-info."""
    target = Path(target)
    for line in Path(requirements).read_text().replace("\\\n", " ").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)\s+@\s+(\S+)", line)
        if not match:
            continue
        name, version = match[1], match[2].rsplit("/", 1)[1].split("-")[1]
        package = target / "nvidia" / canon(name).replace("-", "_")
        package.mkdir(parents=True, exist_ok=True)
        (package / "payload.bin").write_bytes(hashlib.sha256(name.encode()).digest())
        dist = target / f"{canon(name).replace('-', '_')}-{version}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def log_call(world, tool, args):
    keys = ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
            "UV_CONCURRENT_BUILDS", "MAX_JOBS", "PYTHONDONTWRITEBYTECODE")
    record = {"tool": tool, "args": args, "cwd": os.getcwd(), "env": {k: os.environ.get(k) for k in keys}}
    with open(Path(world["dir"]) / "calls.jsonl", "a") as handle:
        handle.write(json.dumps(record) + "\n")


def shim(world, kind):
    extra = ' "$0"' if kind == "python" else ""
    return f'#!/bin/sh\nexec {world["python"]} -I {world["fakes"]} {world["path"]} {kind}{extra} "$@"\n'


def uv(world, args):
    log_call(world, "uv", args)
    if args == ["--version"]:
        print("uv 0.0.0 (rehearsal fake)")
        return 0
    if args[:1] != ["--no-config"]:
        print("fake uv: expected --no-config", file=sys.stderr)
        return 2
    args = args[1:]
    control = Path(world["dir"]) / "control"
    if args[0] == "venv":
        env, clear = Path(args[-1]), "--clear" in args
        if args[1:5] != ["--managed-python", "--python", "3.12.8", "--seed"]:
            print(f"fake uv: unexpected venv arguments {args}", file=sys.stderr)
            return 2
        if env.exists():
            if not clear:
                print("fake uv: environment exists; use --clear", file=sys.stderr)
                return 2
            shutil.rmtree(env)
        base = Path(os.environ["UV_PYTHON_INSTALL_DIR"]) / "cpython-3.12.8-linux-x86_64-gnu"
        (base / "include/python3.12").mkdir(parents=True, exist_ok=True)
        (base / "include/python3.12/Python.h").write_text("/* fake */\n")
        (env / "bin").mkdir(parents=True)
        (env / "bin/python").write_text(shim(world, "python"))
        (env / "bin/python").chmod(0o750)
        (env / "pyvenv.cfg").write_text(f"home = {base}/bin\nversion_info = 3.12.8\n")
        (env / "fake-state.json").write_text(json.dumps({"base_prefix": str(base), "installed": {"pip": "pip==24.0"}}))
        return 0
    if args[:2] == ["pip", "check"]:
        print("All installed packages are compatible")
        return 0
    if args[:2] != ["pip", "install"]:
        print(f"fake uv: unsupported {args}", file=sys.stderr)
        return 2
    rest, options, specs = args[2:], {}, []
    while rest:
        item = rest.pop(0)
        if item in ("--python", "--exclude-newer", "-r"):
            options[item] = rest.pop(0)
        else:
            specs.append(item)
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", options.get("--exclude-newer", "")):
        print("fake uv: --exclude-newer missing", file=sys.stderr)
        return 2
    env = Path(options["--python"]).parents[1]
    once = control / f"fail-install-once-{env.name}"
    if once.exists():
        once.unlink()
        print("fake uv: simulated network failure", file=sys.stderr)
        return 1
    state = json.loads((env / "fake-state.json").read_text())
    pins = {}
    if "-r" in options:
        for line in Path(options["-r"]).read_text().splitlines():
            if line.strip():
                pins[canon(line.split("==")[0])] = line.strip()
    if (control / f"drift-{env.name}").exists():
        pins["pip"] = "pip==26.2.3"
    for spec in specs:
        local = re.fullmatch(r"(/[^\[]+)(?:\[(\w+)\])?", spec)
        if local:
            source = Path(local[1])
            name = re.search(r'^name = "([^"]+)"', (source / "pyproject.toml").read_text(), re.M)[1]
            state["installed"][canon(name)] = f"{name} @ file://{source}"
            (source / "build/lib").mkdir(parents=True, exist_ok=True)  # setuptools' in-tree build copy
            (source / "build/lib/marker.txt").write_text("built\n")
            egg = source / f"{canon(name).replace('-', '_')}.egg-info"  # ignored by the fake repos' .gitignore
            egg.mkdir(exist_ok=True)
            (egg / "PKG-INFO").write_text(f"Name: {name}\n")
        elif "==" in spec:
            state["installed"][canon(spec.split("==")[0])] = spec
        elif canon(spec) not in pins and canon(spec) not in state["installed"]:
            print(f"fake uv: unpinned requirement {spec}", file=sys.stderr)
            return 1
    state["installed"].update(pins)
    (env / "fake-state.json").write_text(json.dumps(state))
    return 0


def python(world, argv0, args):
    log_call(world, "python", [argv0, *args])
    env = Path(argv0).parents[1]
    state = json.loads((env / "fake-state.json").read_text())
    if args[:1] == ["-c"] and "Python.h" in args[1]:
        headers = [f"{state['base_prefix']}/include/python3.12/Python.h"]
        missing = [h for h in headers if not Path(h).is_file()]
        print(json.dumps({"python_version": "3.12.8", "executable": argv0, "base_prefix": state["base_prefix"],
                          "headers": headers, "missing": missing}))
        return 1 if missing else 0
    if args == ["-m", "pip", "freeze", "--all"]:
        lines = dict(state["installed"])
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
            for info in sorted(Path(entry).glob("*.egg-info/PKG-INFO")) if entry and Path(entry).is_dir() else []:
                text = info.read_text()
                name = re.search(r"^Name: (\S+)", text, re.M)[1]
                lines.setdefault(canon(name), f"{name}=={re.search(r'^Version: (\S+)', text, re.M)[1]}")
        sys.stdout.write(freeze_text(lines.values()))
        return 0
    if args and args[0].endswith("prepare_cuda_toolkit.py"):
        import platform
        import subprocess
        real_run = subprocess.run

        def fake_run(command, *a, **k):
            if "pip" in command and "--target" in command:
                fake_pip_target(command[command.index("--target") + 1], command[command.index("-r") + 1])
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, *a, **k)

        subprocess.run, sys.platform, platform.machine = fake_run, "linux", lambda: "x86_64"
        return run_script(args)
    if args and args[0].endswith("model_probe.py"):
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda repo, revision, allow_patterns=None: make_snapshot(
            Path(os.environ["HF_HOME"]) / "hub", repo, revision, world["hub"][repo])
        checkpoint = types.ModuleType("kev.checkpoint")

        class Checkpoint:
            def __init__(self, path):
                self.meta = types.SimpleNamespace(**json.loads((Path(path) / "kev_meta.json").read_text()))

        checkpoint.Checkpoint = Checkpoint
        sys.modules.update({"huggingface_hub": hub, "kev": types.ModuleType("kev"), "kev.checkpoint": checkpoint})
        return run_script(args)
    print(f"fake python: unsupported {args[:3]}", file=sys.stderr)
    return 2


def run_script(args):
    sys.argv = list(args)
    try:
        runpy.run_path(args[0], run_name="__main__")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    return 0


if __name__ == "__main__":
    world_path = Path(sys.argv[1])
    world = {**json.loads(world_path.read_text()), "path": str(world_path)}
    if sys.argv[2] == "uv":
        raise SystemExit(uv(world, sys.argv[3:]))
    raise SystemExit(python(world, sys.argv[3], sys.argv[4:]))
