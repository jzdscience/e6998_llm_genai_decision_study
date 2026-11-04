#!/usr/bin/env python3
"""Read-only verification of the additive v2 release and its v1 prerequisites."""
from pathlib import Path
import hashlib
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]

# Only exporter-only variants from already shipped v1 archives are compatible.
# v2 imports this module through the CUDA preflight but uses its own exporter.
EXPORTER = "decision_study/artifacts.py"
CANONICAL_EXPORTER = "4a24522738ef5a75e926bb0b757da60cf11a6cea86d7acf1a1a0f18454c93338"
AUDITED_EXPORTERS = {
    "b0586c1bad4ea6ba645e96daaf845f00763dad758d40a5c547680f04d8281831": "original v1 bundle",
    "d7c87342e501ae346cabc397d3afbc8199274efce3c06f940e15c75e0c2d9dd2": "v1 timestamp export patch",
}


def regular_file(project, name):
    relative = Path(name)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe release member: {name}")
    path = project
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink release member: {name}")
    if not path.is_file():
        raise ValueError(f"Missing regular file: {name}")
    return path


def verify(project=PROJECT):
    project = Path(project).resolve()
    manifest = json.loads((project / "V2_RELEASE_MANIFEST.json").read_text())
    compatibility = manifest.get("compatible_existing_files", {})
    for name, hashes in compatibility.items():
        if (name != EXPORTER
                or manifest["required_existing_files"].get(name) != CANONICAL_EXPORTER
                or not isinstance(hashes, list)
                or any(value not in AUDITED_EXPORTERS for value in hashes)):
            raise ValueError(f"Unaudited release compatibility rule: {name}")
    checked = {}
    accepted = []
    differences = []
    for group in ("files", "required_existing_files"):
        for name, expected in manifest[group].items():
            try:
                path = regular_file(project, name)
            except ValueError as exc:
                differences.append(str(exc))
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                if group == "required_existing_files" and actual in compatibility.get(name, []):
                    accepted.append({"file": name, "actual_sha256": actual,
                                     "canonical_sha256": expected,
                                     "source": AUDITED_EXPORTERS[actual]})
                else:
                    differences.append(f"File differs from release: {name}\n"
                                       f"  expected: {expected}\n  actual:   {actual}")
        checked[group] = len(manifest[group])
    if differences:
        raise ValueError("\n".join(differences) +
                         "\nPreserve existing files and inspect these differences; do not overwrite scientific code.")
    return {"status": "verified", "release": manifest["release"], "checked": checked,
            "compatible_existing_files": accepted,
            "notice": "Read-only file verification. No allocation renewal, model calls or new v2 results."}


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2))
    except Exception as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
