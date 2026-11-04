#!/usr/bin/env python3
"""Read-only verification of the additive v2 release and its v1 prerequisites."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]


def verify(project=PROJECT):
    manifest = json.loads((project / "V2_RELEASE_MANIFEST.json").read_text())
    checked = {}
    for group in ("files", "required_existing_files"):
        for name, expected in manifest[group].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe release member: {name}")
            path = project / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing regular file: {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"File differs from release: {name}. Preserve the existing file and inspect the difference.")
        checked[group] = len(manifest[group])
    return {"status": "verified", "release": manifest["release"], "checked": checked,
            "notice": "Read-only file verification. No allocation renewal, model calls or new v2 results."}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
