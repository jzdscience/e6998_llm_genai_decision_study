"""CPU-only, read-only acceptance of a returned direct-matrix bundle.

Everything is checked from the bundle and this checkout alone: settings and source
file hashes, prepared inputs and the pinned per-ID historical references, the
runtime binding (against pinned accepted configurations), every raw tokenizer and
preflight stage file, each cell's stored rows, smoke, attempt acceptance and
deployment stop evidence, the freeze and access receipts, and finally a re-score
of all 20 cells from the exported gold subsets (pinned fingerprints) compared with
``evaluation.json``. Node paths, model snapshots, Slurm and network are never used,
and nothing inside the bundle root is written.
"""
from collections import Counter
from pathlib import PurePosixPath
import hashlib
import json
import zipfile

from ..common import StudyError, file_hash, read_json, utc_now
from . import data, runtime, study
from .data import checked
from .identity import CELL_ORDER, PROTOCOL_ID, REFERENCE_CELLS, ROLES, normal, protocol, protocol_hash


def extract_bundle(bundle, destination):
    """Extract into a new directory after checking member names; then verify every manifest hash."""
    bundle, destination = normal(bundle), normal(destination)
    if destination.exists():
        raise StudyError(f"Extraction target already exists: {destination}")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        for name in names:
            parts = PurePosixPath(name).parts
            info = archive.getinfo(name)
            if (name.startswith("/") or ".." in parts or not parts or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise StudyError(f"Unsafe bundle member: {name}")
        if len(set(names)) != len(names) or "bundle_manifest.json" not in names:
            raise StudyError("Bundle has duplicate members or no manifest")
        manifest = json.loads(archive.read("bundle_manifest.json"))
        expected = set(manifest.get("files", {})) | {"bundle_manifest.json"}
        if set(names) != expected:
            raise StudyError("Bundle members differ from its manifest")
        destination.mkdir(parents=True)
        for name in names:
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = archive.read(name)
            if name != "bundle_manifest.json":
                entry = manifest["files"][name]
                if hashlib.sha256(payload).hexdigest() != entry["sha256"] or len(payload) != entry["bytes"]:
                    raise StudyError(f"Bundle member differs from its manifest: {name}")
            target.write_bytes(payload)
    return destination, manifest


def audit(root):
    """Return a PASS report or raise StudyError naming the first broken link."""
    root = normal(root)
    checks = []
    value = study.settings(root)
    checks.append("settings: protocol hash and source file hashes equal this checkout; prepared inputs verified")
    _, binding = runtime.verify_binding_offline(root, value)
    ctx = study.context(root, binding=binding)
    checks.append("runtime binding: controls and pinned accepted configurations verified offline")
    for role in ROLES:
        study.verify_role_evidence(root, ctx, role)
    checks.append("raw tokenizer reports and exact preflight stage inventories verified for all roles")
    completions = study.all_completions(root, ctx)
    checks.append("8 completion receipts recomputed from rows, smoke, attempt acceptance and stop evidence")
    freeze = study.make_freeze(root, require_existing=True, offline=True)
    access = checked(root / "evaluation-access.json", "access_hash")
    payload = study.access_payload(root, freeze, completions)
    if {k: access.get(k) for k in payload} != payload or study._epoch(access["created_at"]) < study._epoch(freeze["created_at"]):
        raise StudyError("Evaluation access is not bound to this freeze and evidence")
    checks.append("freeze and evaluation-access receipts bind the completions and references")
    gold = data.load_sealed_gold(root)
    evaluation = checked(root / "evaluation.json", "evaluation_hash")
    pins = protocol()["gold"]
    gold_files = {name: {"source_sha256": pins[name]["sha256"], "subset_sha256": file_hash(root / f"sealed/{name}-gold.json"),
                         "count": len(values), "label_counts": dict(sorted(Counter(values.values()).items()))}
                  for name, values in gold.items()}
    body = study.result_body(root, freeze, access, completions, gold, gold_files)
    stored = {k: v for k, v in evaluation.items() if k not in ("created_at", "evaluation_hash")}
    if body != stored:
        differing = sorted(k for k in set(body) | set(stored) if body.get(k) != stored.get(k))
        raise StudyError(f"Re-scored evaluation differs from evaluation.json in {differing}")
    checks.append("all 20 cells and 40 paired intervals re-scored from the pinned gold subsets equal evaluation.json")
    p = protocol()
    matrix = body["matrix"]
    return {"status": "PASS", "protocol": PROTOCOL_ID, "protocol_hash": protocol_hash(), "audited_at": utc_now(),
            "root": str(root), "source_commit": value["source"]["commit"], "evaluation_hash": evaluation["evaluation_hash"],
            "freeze_hash": freeze["freeze_hash"], "checks": checks,
            "cells": {cell: {"correct": m["correct"], "n": m["n"], "failures": m["failures"],
                             "kind": m["source"]["kind"]} for cell, m in matrix.items()},
            "references_match_accepted": all(matrix[c]["correct"] == p["references"][c]["accepted_correct"]
                                             for c in REFERENCE_CELLS),
            "new_cells": list(CELL_ORDER),
            "notice": "Offline acceptance from the bundle alone; no node path, snapshot, scheduler or model was used"}
