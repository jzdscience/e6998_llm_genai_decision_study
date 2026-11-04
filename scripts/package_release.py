#!/usr/bin/env python3
"""Build a source-only handoff archive, excluding research results and caches."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
target = root.parent / "decision-study-bundle.zip"
allowed_roots = {"decision_study","tests","scripts","configs","docs","notebooks"}
allowed_files = {"README.md","VALIDATION.md","pyproject.toml",".gitignore"}
files = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()
         and (p.relative_to(root).parts[0] in allowed_roots or p.relative_to(root).as_posix() in allowed_files)
         and not any(part in {"__pycache__",".ipynb_checkpoints",".pytest_cache"} for part in p.relative_to(root).parts)
         and p.suffix not in {".pyc",".pyo"}]
checksums = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path in files:
        archive.write(path,"decision-study/"+path.relative_to(root).as_posix())
    archive.writestr("decision-study/SOURCE_SHA256.json",json.dumps(checksums,indent=2))
print(json.dumps({"archive":str(target),"source_files":len(files),"bytes":target.stat().st_size,
                  "sha256":hashlib.sha256(target.read_bytes()).hexdigest()},indent=2))
