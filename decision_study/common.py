"""Portable artifact IO. No GPU dependencies are imported by the controller."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from contextlib import contextmanager


class StudyError(RuntimeError):
    """A failed protocol or runtime check, with a recoverable diagnostic."""


@contextmanager
def exclusive_run(output_dir: Path):
    """OS-released lock: a crashed session cannot leave a permanent busy marker."""
    import fcntl
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".active.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyError("Another command is using this OUTPUT_DIR. Run one job at a time.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise StudyError(f"Invalid JSONL {path}:{number}; do not silently discard records") from exc
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def set_status(output_dir: Path, stage: str, state: str, **details: Any) -> dict:
    path = Path(output_dir) / "status.json"
    status = read_json(path) if path.exists() else {"schema_version": 1, "stages": {}}
    status["updated_at"] = utc_now()
    status["stages"][stage] = {"state": state, "updated_at": utc_now(), **details}
    states = [r["state"] for r in status["stages"].values()]
    status["state"] = "blocked" if "blocked" in states else "partial"
    # Only the orchestrator may declare all required experiment stages complete.
    write_json(path, status)
    return status
