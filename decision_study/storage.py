"""Append-only local journal and atomic, checksummed persistent result shards."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .common import StudyError, canonical, fingerprint, read_json, write_json


class PredictionStore:
    def __init__(self, output_dir: Path, model: str, config: dict, *, spool_dir: Path | None = None,
                 shard_size: int = 25):
        self.output_dir, self.model, self.config = Path(output_dir), model, config
        self.config_hash = fingerprint(config)
        self.directory = self.output_dir / "predictions" / model
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        config_path = self.directory / "config.json"
        if config_path.exists() and read_json(config_path) != config:
            raise StudyError(f"Configuration changed for {model}. Use a NEW OUTPUT_DIR; old results are preserved.")
        if not config_path.exists():
            write_json(config_path, config)
        self.rows: dict[str, dict] = {}
        self.pending: list[dict] = []
        self.sequence = 0
        for path in sorted(self.directory.glob("shard-*.json")):
            shard = read_json(path)
            if shard.get("config_hash") != self.config_hash or fingerprint(shard["rows"]) != shard.get("sha256"):
                raise StudyError(f"Corrupt or mismatched prediction shard: {path}")
            if shard.get("count") != len(shard["rows"]):
                raise StudyError(f"Incorrect shard count: {path}")
            for row in shard["rows"]:
                self._validate(row)
                if row["sample_id"] in self.rows:
                    raise StudyError(f"Duplicate completed sample: {row['sample_id']}")
                self.rows[row["sample_id"]] = row
            self.sequence = max(self.sequence, int(path.stem.split("-")[-1]))
        spool_root = Path(spool_dir) if spool_dir else Path(tempfile.gettempdir()) / "decision-study-spool"
        journal_dir = spool_root / fingerprint(str(self.output_dir.resolve()))[:16] / self.config_hash
        journal_dir.mkdir(parents=True, exist_ok=True)
        self.journal = journal_dir / f"{model}.jsonl"
        self._recover_journal()

    def _validate(self, row: dict) -> None:
        if row.get("model") != self.model or row.get("config_hash") != self.config_hash:
            raise StudyError("Prediction identity does not match store")
        if row.get("status") not in {"ok", "error"} or not row.get("sample_id"):
            raise StudyError("Only completed, explicitly classified results may be persisted")

    def _recover_journal(self) -> None:
        if not self.journal.exists():
            return
        content = self.journal.read_bytes()
        good_offset = 0
        lines = content.splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if i == len(lines) - 1 and not line.endswith(b"\n"):
                    with self.journal.open("r+b") as stream:
                        stream.truncate(good_offset)
                    write_json(self.directory / "journal_recovery.json", {
                        "discarded_incomplete_tail_bytes": len(content) - good_offset,
                        "reason": "process interruption before journal record was complete",
                    })
                    break
                raise StudyError("Journal corruption before final incomplete line") from exc
            good_offset += len(line)
            self._validate(row)
            sample_id = row["sample_id"]
            if sample_id in self.rows:
                if self.rows[sample_id] != row:
                    raise StudyError(f"Conflicting completed predictions for {sample_id}")
                continue
            self.rows[sample_id] = row
            self.pending.append(row)
        self.sync()

    def append(self, row: dict) -> None:
        self._validate(row)
        sample_id = row["sample_id"]
        if sample_id in self.rows:
            raise StudyError(f"Refusing to replace terminal result {sample_id}")
        with self.journal.open("ab") as stream:
            stream.write(canonical(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows[sample_id] = row
        self.pending.append(row)
        if len(self.pending) >= self.shard_size:
            self.sync()

    def sync(self) -> None:
        while self.pending:
            rows = self.pending[:self.shard_size]
            self.sequence += 1
            write_json(self.directory / f"shard-{self.sequence:06d}.json", {
                "config_hash": self.config_hash, "count": len(rows), "rows": rows, "sha256": fingerprint(rows),
            })
            del self.pending[:len(rows)]
        write_json(self.directory / "index.json", {
            "model": self.model, "config_hash": self.config_hash, "terminal": len(self.rows),
            "successful": sum(r["status"] == "ok" for r in self.rows.values()),
            "failed": sum(r["status"] == "error" for r in self.rows.values()), "shards": self.sequence,
        })
        # Persistent shards are durable; old local journal entries can now be discarded.
        if self.journal.exists():
            with self.journal.open("wb") as stream:
                stream.flush()
                os.fsync(stream.fileno())


def load_predictions(output_dir: Path, model: str) -> list[dict]:
    directory = Path(output_dir) / "predictions" / model
    config_path = directory / "config.json"
    if not config_path.exists():
        return []
    config_hash = fingerprint(read_json(config_path))
    found: dict[str, dict] = {}
    for path in sorted(directory.glob("shard-*.json")):
        shard = read_json(path)
        if shard.get("config_hash") != config_hash or fingerprint(shard["rows"]) != shard.get("sha256"):
            raise StudyError(f"Invalid shard {path}")
        if shard.get("count") != len(shard["rows"]):
            raise StudyError(f"Invalid count {path}")
        for row in shard["rows"]:
            if (row.get("config_hash") != config_hash or row.get("model") != model or row["sample_id"] in found
                    or row.get("status") not in {"ok","error"}):
                raise StudyError(f"Mixed identity or duplicate in {path}")
            found[row["sample_id"]] = row
    return list(found.values())
