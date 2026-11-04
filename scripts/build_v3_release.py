#!/usr/bin/env python3
"""Build an additive v3 package after data preparation and local validation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
V2_RELEASE_SHA256 = "a8aba344b370f95dd2e2d86ca4eec6428d7f0604a4deb230de5d8a050251559d"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def build(output, prepared, release):
    if sha(release.read_bytes()) != V2_RELEASE_SHA256:
        raise ValueError("Original v2 release identity differs")
    with zipfile.ZipFile(release) as archive:
        required = json.loads(archive.read("decision-study/V2_RELEASE_MANIFEST.json"))["required_existing_files"]
    for name, digest in required.items():
        if sha((PROJECT / name).read_bytes()) != digest:
            raise ValueError(f"Local frozen dependency differs: {name}")
    files = {}
    for directory, pattern in [("decision_study/v3", "*.py"), ("tests", "test_v3*.py")]:
        for p in sorted((PROJECT / directory).glob(pattern)):
            files[p.relative_to(PROJECT).as_posix()] = p.read_bytes()
    for name in ("configs/v3_protocol.json", "docs/V3_PROTOCOL_ZH.md", "docs/RUN_V3_ZH.md",
                 "scripts/run_v3.sh", "scripts/verify_v3_release.py", "scripts/build_v3_release.py"):
        files[name] = (PROJECT / name).read_bytes()
    for name in ("data/examples.jsonl", "data/fit_gold.json", "data/manifest.json", "sealed/holdout_gold.json"):
        files["results/solo-v3/" + name] = (prepared / name).read_bytes()
    manifest = {"schema_version": 1, "release": "decision-study-v3-anli-2026-11-03",
                "protocol": "solo-v3-anli", "required_existing_files": required,
                "files": {name: sha(payload) for name, payload in sorted(files.items())},
                "notice": "Additive independent v3 files only. No v1/v2 science or runtime directories overwritten. GPU execution and model results not yet verified."}
    (PROJECT / "V3_RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files["V3_RELEASE_MANIFEST.json"] = (PROJECT / "V3_RELEASE_MANIFEST.json").read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a published archive. Use a new candidate filename when revising.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo("decision-study/" + name, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return {"archive": str(output), "bytes": output.stat().st_size,
            "sha256": sha(output.read_bytes()), "members": len(files), "required_existing_files": len(required)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepared", type=Path, default=PROJECT / "results/solo-v3-prepared")
    p.add_argument("--v2-release", type=Path, default=PROJECT.parent / "decision-study-v2-ready.zip")
    args = p.parse_args()
    print(json.dumps(build(args.output, args.prepared, args.v2_release), indent=2))
