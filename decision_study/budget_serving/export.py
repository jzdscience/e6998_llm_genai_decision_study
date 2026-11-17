"""Timestamped compact archives of the study output; large archives become hashed byte parts.

Included: prepared inputs and policy identity, protocol/source identity, every raw
trace, ledger and fallback journal, stage rows, service and lifecycle receipts (every
ownership self-test folder of every allocation), runtime controls with doctor and raw
token reports, logs, freeze, gold-access receipt, preserved pinned gold and analysis.
Excluded: weights, environments, caches (``runtime-work``) and earlier exports (the
optional embedding supplement lives under ``exports/`` and has its own parts). No
file covered by the freeze is omitted, so the archive stays fully auditable.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import time
import zipfile

from ..common import canonical, file_hash, read_json, utc_now
from .protocol import PROTOCOL_ID, REVISION, require
from . import study

PART_BYTES = 48 * 1024 * 1024
EXCLUDED_DIRS = {"runtime-work", "exports", "__pycache__", "work"}
KINDS = ("result", "checkpoint", "diagnostic", "manual")
PART = re.compile(r"(?P<archive>budget_serving_[a-z]+-[0-9]+\.zip)\.part(?P<index>[0-9]{4})")


def _members(output):
    output = study.normal(output)
    files = []
    for directory, dirs, names in os.walk(output, followlinks=False):
        require(not any((Path(directory) / d).is_symlink() for d in dirs), "Symlinked directory in study output")
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
        for name in sorted(names):
            target = Path(directory) / name
            relative = study.rel(output, target)
            study.path(output, relative)
            if name == ".active.lock" or target.is_symlink():
                continue
            files.append((relative, target))
    return files


def split(target):
    """Write <zip>.partNNNN byte parts and a parts manifest; the joined bytes equal the archive."""
    digest, size = file_hash(target), target.stat().st_size
    parts = []
    with target.open("rb") as stream:
        for index in range(1, 10000):
            block = stream.read(PART_BYTES)
            if not block:
                break
            name = f"{target.name}.part{index:04d}"
            with (target.parent / name).open("xb") as part:
                part.write(block)
                part.flush()
                os.fsync(part.fileno())
            parts.append({"name": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    joined = hashlib.sha256()
    for part in parts:
        joined.update((target.parent / part["name"]).read_bytes())
    require(joined.hexdigest() == digest, "Archive parts do not reproduce the archive")
    manifest = {"schema_version": 1, "archive": target.name, "archive_sha256": digest, "archive_bytes": size,
                "part_limit_bytes": PART_BYTES, "parts": parts,
                "join": f"python -m decision_study.budget_serving join --manifest {target.name}.parts.json --target {target.name}"}
    path = target.with_name(target.name + ".parts.json")
    with path.open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")
    target.unlink()
    return path


def _checked_parts(manifest_path, manifest):
    """Canonical names in numeric order, bounded sizes, regular files, totals matching the archive."""
    archive, parts = manifest.get("archive"), manifest.get("parts")
    require(isinstance(archive, str) and Path(archive).name == archive and isinstance(parts, list) and parts
            and manifest_path.name == f"{archive}.parts.json", "Parts manifest names no canonical archive")
    total, checked = 0, []
    for index, part in enumerate(parts, 1):
        name = part.get("name") if isinstance(part, dict) else None
        match = PART.fullmatch(name or "")
        require(match is not None and match["archive"] == archive and int(match["index"]) == index,
                "Part names are not the canonical numeric sequence")
        size = part.get("bytes")
        require(isinstance(size, int) and not isinstance(size, bool) and 0 < size <= PART_BYTES
                and (index == len(parts) or size == PART_BYTES), "Declared part sizes differ from the split rule")
        path = manifest_path.parent / name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == size, f"Part is missing or not a regular file: {name}")
        total += size
        checked.append((path, part))
    require(total == manifest.get("archive_bytes"), "Parts do not add up to the declared archive size")
    return checked


def join(manifest_path, target):
    """Rebuild the archive atomically; a failed join leaves no partial target behind."""
    manifest_path, target = study.normal(manifest_path), study.normal(target)
    require(not target.exists(), "Join target already exists")
    manifest = read_json(manifest_path)
    parts = _checked_parts(manifest_path, manifest)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.part")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as out:
            for path, part in parts:
                data = path.read_bytes()
                require(len(data) == part["bytes"] and hashlib.sha256(data).hexdigest() == part["sha256"],
                        f"Part differs: {part['name']}")
                digest.update(data)
                out.write(data)
            out.flush()
            os.fsync(out.fileno())
        require(digest.hexdigest() == manifest["archive_sha256"], "Joined archive differs from its manifest")
        os.link(temporary, target)  # no replacement of a concurrently created target
    finally:
        temporary.unlink(missing_ok=True)
    return target


def export(output, kind):
    require(kind in KINDS, "Unknown export kind")
    output = study.normal(output)
    files = _members(output)
    manifest = {"schema_version": 1, "kind": kind, "protocol": PROTOCOL_ID, "revision": REVISION, "created_at": utc_now(),
                "files": {relative: {"sha256": file_hash(target), "bytes": target.stat().st_size} for relative, target in files},
                "excluded": sorted(EXCLUDED_DIRS),
                "scope": "Raw serving evidence and receipts; no weights, environments or caches"}
    target = study.path(output, f"exports/budget_serving_{kind}-{time.time_ns()}.zip")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for relative, source in files:
            archive.write(source, relative)
        archive.writestr("BUNDLE_MANIFEST.json", canonical(manifest) + b"\n")
    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        require(names == {r for r, _ in files} | {"BUNDLE_MANIFEST.json"} and archive.testzip() is None, "Archive verification failed")
    if target.stat().st_size > PART_BYTES:
        return split(target)
    return target
