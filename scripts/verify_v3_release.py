#!/usr/bin/env python3
"""Verify an additive v3 delivery without modifying any existing scientific files."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPORTER = "decision_study/artifacts.py"
CANONICAL_EXPORTER = "4a24522738ef5a75e926bb0b757da60cf11a6cea86d7acf1a1a0f18454c93338"
AUDITED_EXPORTERS = {
    "b0586c1bad4ea6ba645e96daaf845f00763dad758d40a5c547680f04d8281831",
    "d7c87342e501ae346cabc397d3afbc8199274efce3c06f940e15c75e0c2d9dd2",
}


def verify(project=PROJECT):
    project = Path(project)
    manifest = json.loads((project / "V3_RELEASE_MANIFEST.json").read_text())
    errors, accepted, counts = [], [], {}
    for group in ("files", "required_existing_files"):
        for name, expected in manifest[group].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe release member")
            path = project / relative
            if any(p.is_symlink() for p in (path, *list(path.parents)[:len(relative.parts)-1])) or not path.is_file():
                errors.append(f"Missing regular file: {name}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual == expected:
                continue
            if group == "required_existing_files" and name == EXPORTER and expected == CANONICAL_EXPORTER and actual in AUDITED_EXPORTERS:
                accepted.append({"file": name, "sha256": actual, "reason": "Previously audited exporter-only variant, imported by CUDA preflight; v3 uses its own exporter"})
            else:
                errors.append(f"File differs from v3 release: {name} (expected {expected}, got {actual})")
        counts[group] = len(manifest[group])
    if errors:
        raise ValueError("\n".join(errors) + "\nPreserve existing files; inspect differences before running.")
    return {"status": "verified", "release": manifest["release"], "checked": counts,
            "compatible_exporters": accepted, "notice": "File verification only; GPU feasibility and accuracy await real execution."}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except Exception as exc:
        print(f"BLOCKED: {exc}")
        raise SystemExit(2)
