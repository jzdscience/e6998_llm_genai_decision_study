#!/usr/bin/env python3
"""Preserve the 176 accepted Study 4 arrays, separately from serving evidence.

Standard-library only. The pinned historical freeze is the trust anchor; this
script does not call the historical source-context check, refit a router, infer a
model or replace any historical file. Default failures are reported as INCOMPLETE
with exit zero because this is an optional supplement. Use --strict for a gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time
import zipfile

PINNED_FREEZE_SHA256 = "a1b4d4e2b17ce3f57cda7de7d8c1190f21e89601da04ab2b084647b901f6a967"
ACCEPTED_BUNDLE_SHA256 = "08d88006855aef956bac8722293e24f333afd6cb4249f400dd9393cdf9649874"
EXPECTED_ARRAYS = 176
EXPECTED_ROWS = 22437
DIMENSION = 1536
CHUNK_ROWS = 128
PART_BYTES = 48 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
PREFIX = "collection/embeddings/"
ARCHIVE = "learnability_embeddings.zip"
SCOPE = "Accepted embedding bytes only; full text-head refit has NOT been performed."


class CaptureError(RuntimeError):
    """Evidence is missing, changed, unsafe or inconsistent."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise CaptureError(message)


def root_path(value):
    path = Path(value).expanduser().absolute()
    require(not path.is_symlink(), f"Symlinked root is not allowed: {path}")
    return path.resolve()


def member_path(root, name):
    require(isinstance(name, str) and bool(name) and "\\" not in name, "Invalid member path")
    pure = PurePosixPath(name)
    require(not pure.is_absolute() and pure.as_posix() == name and not {"..", "."}.intersection(pure.parts),
            "Unsafe member path")
    path = root
    for part in pure.parts:
        path = path / part
        require(not path.is_symlink(), f"Symlinked evidence path: {name}")
    require(root in path.parents, "Member escapes root")
    return path


def read_json_bytes(data):
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise CaptureError("Invalid JSON evidence") from exc


def read_small(path, limit=2 * 1024 * 1024):
    require(path.is_file() and not path.is_symlink(), f"Missing regular evidence file: {path}")
    require(path.stat().st_size <= limit, f"Oversized metadata: {path}")
    return path.read_bytes()


def expected_members(freeze_bytes):
    require(digest(freeze_bytes) == PINNED_FREEZE_SHA256, "Historical freeze differs from accepted SHA-256")
    freeze = read_json_bytes(freeze_bytes)
    require(freeze.get("models_and_scores_frozen_before_evaluation_gold") is True, "Invalid historical freeze")
    files = freeze.get("files")
    require(isinstance(files, dict), "Missing frozen file inventory")
    selected = {name: value for name, value in files.items() if name.startswith(PREFIX)}
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in selected.values()),
            "Invalid frozen hashes")
    arrays = sorted(name for name in selected if name.endswith(".npy"))
    receipts = sorted(name for name in selected if name.endswith("/receipt.json"))
    require(len(arrays) == EXPECTED_ARRAYS and len(receipts) == EXPECTED_ARRAYS, "Frozen embedding count differs")
    require(set(selected) == set(arrays + receipts + [PREFIX + "identity.json", PREFIX + "complete.json"]),
            "Unexpected frozen embedding member")
    require(all(re.fullmatch(r"collection/embeddings/chunks/[0-9]{8}/vectors-[0-9]+\.npy", name) for name in arrays),
            "Unexpected embedding array path")
    return {"freeze.json": PINNED_FREEZE_SHA256, **selected}


def verify_metadata(reader, members):
    identity = read_json_bytes(reader(PREFIX + "identity.json"))
    complete = read_json_bytes(reader(PREFIX + "complete.json"))
    identity_hash = digest(canonical(identity))
    require(identity.get("encoding", {}).get("dimension") == DIMENSION, "Embedding dimension differs")
    require(complete.get("identity_hash") == identity_hash and complete.get("count") == EXPECTED_ROWS
            and complete.get("dimension") == DIMENSION, "Embedding completion identity differs")
    ids, referenced = [], set()
    for start in range(0, EXPECTED_ROWS, CHUNK_ROWS):
        folder = f"{PREFIX}chunks/{start:08d}"
        name = folder + "/receipt.json"
        require(name in members, "Missing expected embedding receipt")
        receipt = read_json_bytes(reader(name))
        count = min(CHUNK_ROWS, EXPECTED_ROWS - start)
        vectors = receipt.get("vectors")
        require(isinstance(vectors, str) and re.fullmatch(r"vectors-[0-9]+\.npy", vectors), "Unsafe receipt vector path")
        array_name = folder + "/" + vectors
        require(receipt.get("identity_hash") == identity_hash and receipt.get("start") == start
                and receipt.get("count") == count and array_name in members
                and receipt.get("sha256") == members[array_name], "Embedding receipt identity differs")
        sample_ids = receipt.get("ids")
        require(isinstance(sample_ids, list) and len(sample_ids) == count
                and all(isinstance(value, str) and value for value in sample_ids), "Invalid embedding IDs")
        ids.extend(sample_ids)
        referenced.add(array_name)
    require(len(ids) == len(set(ids)) == EXPECTED_ROWS, "Embedding IDs are incomplete or repeated")
    require(referenced == {name for name in members if name.endswith(".npy")}, "Unreferenced frozen embedding array")


def verify_source(source):
    freeze = read_small(member_path(source, "freeze.json"))
    members = expected_members(freeze)
    sizes = {}
    for name, expected in sorted(members.items()):
        path = member_path(source, name)
        require(path.is_file(), f"Missing frozen evidence: {name}")
        require(stat.S_ISREG(path.stat().st_mode), f"Non-regular evidence: {name}")
        require(file_digest(path) == expected, f"Frozen evidence changed: {name}")
        sizes[name] = path.stat().st_size
    require(sum(sizes.values()) <= MAX_TOTAL_BYTES, "Supplement exceeds its size bound")
    verify_metadata(lambda name: read_small(member_path(source, name)), members)
    return members, sizes


def publish_bytes(output, name, data):
    """Create an immutable file atomically; reuse equal bytes, never overwrite."""
    target = member_path(output, name)
    if target.exists():
        require(target.is_file() and target.stat().st_size == len(data) and file_digest(target) == digest(data),
                f"Existing capture file differs; preserve it and choose a new output: {name}")
        return
    with tempfile.NamedTemporaryFile(prefix=".capture-", dir=output, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == len(data)
                    and file_digest(target) == digest(data), f"Concurrent capture differs: {name}")
    finally:
        temporary.unlink(missing_ok=True)


def archive_source(source, target, members):
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name, expected in sorted(members.items()):
            path = member_path(source, name)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info._compresslevel = 6
            hashed = hashlib.sha256()
            with path.open("rb") as incoming, archive.open(info, "w", force_zip64=True) as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    hashed.update(block)
                    outgoing.write(block)
            member_path(source, name)
            require(hashed.hexdigest() == expected, f"Evidence changed while archiving: {name}")


def verify_capture(output, *, require_receipt=True):
    """Verify parts AND every archived member against the fixed historical freeze."""
    output = root_path(output)
    manifest = read_json_bytes(read_small(member_path(output, "manifest.json")))
    require(isinstance(manifest, dict), "Capture manifest must be an object")
    require(manifest.get("schema_version") == 1 and manifest.get("archive") == ARCHIVE
            and manifest.get("freeze_sha256") == PINNED_FREEZE_SHA256
            and manifest.get("full_refit_performed") is False
            and manifest.get("accepted_bundle_sha256") == ACCEPTED_BUNDLE_SHA256
            and manifest.get("array_count") == EXPECTED_ARRAYS and manifest.get("row_count") == EXPECTED_ROWS
            and manifest.get("dimension") == DIMENSION and manifest.get("part_limit_bytes") == PART_BYTES
            and manifest.get("scope") == SCOPE, "Capture manifest identity differs")
    parts = manifest.get("parts")
    require(isinstance(parts, list) and 0 < len(parts) <= 1024
            and all(isinstance(part, dict) for part in parts), "Invalid capture parts")
    require({p.name for p in output.glob(ARCHIVE + ".part*")} == {p.get("name") for p in parts},
            "Unlisted or missing archive parts")
    total = 0
    archive_hash = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="embedding-verify-") as temp:
        joined = Path(temp) / ARCHIVE
        with joined.open("wb") as stream:
            for index, part in enumerate(parts, 1):
                name = f"{ARCHIVE}.part{index:04d}"
                require(part.get("name") == name, "Parts are missing, reordered or unsafe")
                path = member_path(output, name)
                require(path.is_file() and 0 < path.stat().st_size <= PART_BYTES, "Invalid capture part size")
                require(path.stat().st_size == part.get("bytes") and file_digest(path) == part.get("sha256"),
                        f"Capture part changed: {name}")
                total += path.stat().st_size
                require(total <= MAX_TOTAL_BYTES, "Capture parts exceed size bound")
                with path.open("rb") as incoming:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        archive_hash.update(block)
                        stream.write(block)
        require(total == manifest.get("archive_bytes") and archive_hash.hexdigest() == manifest.get("archive_sha256"),
                "Reassembled archive identity differs")
        with zipfile.ZipFile(joined) as archive:
            infos = archive.infolist()
            require(len(infos) == len({item.filename for item in infos}), "Duplicate ZIP member")
            require(all(stat.S_IFMT(item.external_attr >> 16) == stat.S_IFREG for item in infos), "Non-regular ZIP member")
            require(all(item.file_size <= 2 * 1024 * 1024 for item in infos), "Oversized ZIP member")
            members = expected_members(archive.read("freeze.json"))
            require(set(archive.namelist()) == set(members), "Archive member inventory differs")
            expected_inventory = {}
            for name, expected in sorted(members.items()):
                member_path(output, name)  # Validate archived relative paths without extracting.
                data = archive.read(name)
                require(digest(data) == expected, f"Archived evidence changed: {name}")
                expected_inventory[name] = {"bytes": len(data), "sha256": expected}
            require(manifest.get("members") == expected_inventory, "Capture member index differs")
            verify_metadata(archive.read, members)
    if require_receipt:
        receipt = read_json_bytes(read_small(member_path(output, "capture-receipt.json")))
        require(receipt == {"status": "PASS_ARCHIVED_ONLY", "scope": SCOPE,
                "freeze_sha256": PINNED_FREEZE_SHA256,
                "manifest_sha256": digest(canonical(manifest) + b"\n"),
                "array_count": EXPECTED_ARRAYS, "parts": len(parts), "full_refit_performed": False},
                "Capture completion receipt differs")
    return manifest


def capture(source_dir, output_dir):
    source, output = root_path(source_dir), root_path(output_dir)
    require(source != output and source not in output.parents and output not in source.parents,
            "Capture output must not overlap the historical study")
    require(source.is_dir(), "Historical study root is missing")
    members, sizes = verify_source(source)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(path.is_symlink() for path in output.iterdir()), "Symlinked capture output member")
    with tempfile.TemporaryDirectory(prefix=".capture-build-", dir=output) as temp:
        archive = Path(temp) / ARCHIVE
        archive_source(source, archive, members)
        parts = []
        with archive.open("rb") as stream:
            for index in range(1, 1025):
                block = stream.read(PART_BYTES)
                if not block:
                    break
                name = f"{ARCHIVE}.part{index:04d}"
                publish_bytes(output, name, block)
                parts.append({"name": name, "bytes": len(block), "sha256": digest(block)})
            else:
                raise CaptureError("Too many archive parts")
        manifest = {"schema_version": 1, "scope": SCOPE, "accepted_bundle_sha256": ACCEPTED_BUNDLE_SHA256,
                    "freeze_sha256": PINNED_FREEZE_SHA256, "array_count": EXPECTED_ARRAYS,
                    "row_count": EXPECTED_ROWS, "dimension": DIMENSION, "full_refit_performed": False,
                    "archive": ARCHIVE, "archive_sha256": file_digest(archive), "archive_bytes": archive.stat().st_size,
                    "part_limit_bytes": PART_BYTES, "parts": parts,
                    "members": {name: {"bytes": sizes[name], "sha256": value} for name, value in sorted(members.items())}}
        publish_bytes(output, "manifest.json", canonical(manifest) + b"\n")
    verified = verify_capture(output, require_receipt=False)
    instructions = ("Study 4 embedding supplement\n\n" + SCOPE + "\n"
                    "Keep every .partNNNN file, manifest.json and capture-receipt.json together.\n"
                    "These parts are one ZIP byte stream, not individually extractable ZIP files.\n"
                    "Verify with the released archive_learnability_embeddings.py:\n"
                    "  python3 archive_learnability_embeddings.py --verify-only --output-dir THIS_DIRECTORY --strict\n"
                    "After verification, join in numeric filename order:\n"
                    "  cat learnability_embeddings.zip.part* > learnability_embeddings.zip\n"
                    "The ZIP paths overlay the accepted Study 4 compact bundle, without replacing different bytes.\n"
                    "No models, environments, inference caches or new fitting results are included.\n")
    publish_bytes(output, "README.txt", instructions.encode())
    receipt = {"status": "PASS_ARCHIVED_ONLY", "scope": SCOPE, "freeze_sha256": PINNED_FREEZE_SHA256,
               "manifest_sha256": digest(canonical(verified) + b"\n"), "array_count": EXPECTED_ARRAYS,
               "parts": len(verified["parts"]), "full_refit_performed": False}
    publish_bytes(output, "capture-receipt.json", canonical(receipt) + b"\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return nonzero if the optional supplement is incomplete")
    args = parser.parse_args(argv)
    if not args.verify_only and args.source_dir is None:
        parser.error("--source-dir is required unless --verify-only is used")
    try:
        if args.verify_only:
            manifest = verify_capture(args.output_dir)
            result = {"status": "PASS_ARCHIVED_ONLY", "array_count": manifest["array_count"],
                      "full_refit_performed": False, "scope": SCOPE}
        else:
            result = capture(args.source_dir, args.output_dir)
    except (CaptureError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        result = {"status": "INCOMPLETE", "scope": SCOPE, "error": str(exc), "full_refit_performed": False}
        # Never replace a prior PASS receipt; failures have separate immutable names.
        try:
            output = root_path(args.output_dir)
            source = root_path(args.source_dir) if args.source_dir else None
            require(source is None or (source != output and source not in output.parents and output not in source.parents),
                    "Unsafe failure receipt path")
            output.mkdir(parents=True, exist_ok=True)
            publish_bytes(output, f"incomplete-{time.time_ns()}.json", canonical(result) + b"\n")
        except (CaptureError, OSError):
            pass
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if args.strict else 0
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
