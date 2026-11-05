"""Reuse exact completed V3 inputs/references without reading old test gold early."""
from collections import Counter
from datetime import datetime
from pathlib import Path
from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl, write_json, write_jsonl, utc_now
from ..v3 import data as old_data, study as old_study
from .identity import protocol, source_identity
from .runtime import _safe


def seal(payload, key):
    return {**payload, key: fingerprint(payload)}


def checked(path, key):
    value = read_json(path)
    if value.get(key) != fingerprint({k: v for k, v in value.items() if k != key}):
        raise StudyError(f"Artifact self hash changed: {path}")
    return value


def _source_public(v3_dir):
    v3 = Path(v3_dir).absolute()
    pins = protocol()["data"]["source_sha256"]
    for name, expected in pins.items():
        if name == "sealed/holdout_gold.json":
            continue
        if file_hash(_safe(v3, name)) != expected:
            raise StudyError(f"Original V3 evidence differs from accepted return: {name}")
    manifest = old_data.verify_fit(v3)
    rows = old_data.load_examples(v3)
    selected = [row for row in rows if row["split"] in {"development", "holdout"}]
    p = protocol()["data"]
    if (Counter(row["split"] for row in selected) != {"development": p["development_count"], "holdout": p["formal_count"]}
            or Counter(row["round"] for row in selected if row["split"] == "holdout") != p["round_counts"]):
        raise StudyError("V3 size control inputs changed")
    references, provenance = {}, {}
    for role in p["reference_roles"]:
        predictions, done = old_study.verified_job(v3, role, "holdout")
        references[role] = sorted(predictions, key=lambda row: row["sample_id"])
        directory = v3 / "predictions-jobs" / (role + "-holdout")
        config = read_json(directory / "predictions" / old_study.interface(role) / "config.json")
        provenance[role] = {"completion": done, "config": config, "rows_hash": fingerprint(references[role])}
    return v3, selected, manifest, references, provenance


def prepare(v3_dir, output, *, checkout=None):
    """Read public input and previously accepted predictions; never old sealed gold."""
    output = Path(output).absolute()
    v3 = Path(v3_dir).absolute()
    if output == v3 or v3 in output.parents or output in v3.parents:
        raise StudyError("Size control and historical V3 directories must not overlap")
    target = _safe(output, "data/manifest.json")
    if target.exists():
        manifest = verify(output)
        if manifest["source_v3_dir"] != str(v3):
            raise StudyError("Prepared source directory differs; do not rebind existing evidence")
        return manifest
    if _safe(output, "freeze.json").exists() or _safe(output, "evaluation-access.json").exists():
        raise StudyError("Missing prepared manifest in an existing run; preserve evidence")
    v3, rows, old_manifest, references, provenance = _source_public(v3)
    payloads = {"data/examples.jsonl": b"".join(canonical(row) + b"\n" for row in rows)}
    for role, predictions in references.items():
        payloads[f"references/{role}.jsonl"] = b"".join(canonical(row) + b"\n" for row in predictions)
    provenance = {"original_manifest": old_manifest, "original_evaluation": read_json(v3 / "evaluation.json"),
                  "original_freeze": read_json(v3 / "policy_freeze.json"),
                  "original_access": read_json(v3 / "holdout_access.json"), "jobs": provenance}
    payloads["references/provenance.json"] = canonical(provenance) + b"\n"
    import hashlib
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    manifest = seal({"schema_version": 1, "protocol": protocol()["protocol"], "protocol_hash": fingerprint(protocol()),
                    "source_v3_dir": str(v3), "source_files": source_identity(), "checkout": checkout,
                    "files": hashes, "count": len(rows), "created_at": utc_now(),
                    "scope": protocol()["scope"], "old_gold_read": False}, "manifest_hash")
    for name, payload in payloads.items():
        path = _safe(output, name)
        if path.exists() and path.read_bytes() != payload:
            raise StudyError(f"Existing partial preparation differs: {name}")
    for name, payload in payloads.items():
        path = _safe(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
    write_json(target, manifest)
    return verify(output)


def verify(output):
    manifest = checked(_safe(output, "data/manifest.json"), "manifest_hash")
    expected = {"data/examples.jsonl", "references/provenance.json"} | {
        f"references/{role}.jsonl" for role in protocol()["data"]["reference_roles"]}
    if (manifest.get("protocol") != protocol()["protocol"] or manifest.get("protocol_hash") != fingerprint(protocol())
            or manifest.get("source_files") != source_identity() or set(manifest.get("files", {})) != expected):
        raise StudyError("Prepared protocol/source/inventory changed")
    for name, sha in manifest["files"].items():
        if file_hash(_safe(output, name)) != sha:
            raise StudyError(f"Prepared evidence changed: {name}")
    rows = read_jsonl(_safe(output, "data/examples.jsonl"))
    p = protocol()["data"]
    if (len(rows) != manifest["count"] or len({r["sample_id"] for r in rows}) != len(rows)
            or Counter(r["split"] for r in rows) != {"development": p["development_count"], "holdout": p["formal_count"]}):
        raise StudyError("Prepared example coverage changed")
    holdout = [e for e in rows if e["split"] == "holdout"]
    if Counter(e["round"] for e in holdout) != p["round_counts"]:
        raise StudyError("Prepared ANLI round membership changed")
    provenance = read_json(_safe(output, "references/provenance.json"))
    for role in p["reference_roles"]:
        predictions = read_jsonl(_safe(output, f"references/{role}.jsonl"))
        job = provenance["jobs"][role]
        if fingerprint(sorted(predictions, key=lambda r: r["sample_id"])) != job["completion"]["predictions_hash"]:
            raise StudyError("Historical prediction identity changed")
        old_study._check_rows(holdout, predictions, old_study.interface(role), fingerprint(job["config"]),
                              job["config"]["runtime"]["model_revision"])
    return manifest


def examples(output, role):
    verify(output)
    return [e for e in read_jsonl(_safe(output, "data/examples.jsonl")) if e["split"] == role]


def references(output):
    verify(output)
    return {role: read_jsonl(_safe(output, f"references/{role}.jsonl"))
            for role in protocol()["data"]["reference_roles"]}


def gold_for_evaluation(output, freeze, access):
    """Only called after the complete prediction/freeze/access chain is verified."""
    manifest = verify(output)
    if (checked(_safe(output, "freeze.json"), "freeze_hash") != freeze
            or checked(_safe(output, "evaluation-access.json"), "access_hash") != access
            or access.get("freeze_hash") != freeze["freeze_hash"]
            or freeze.get("manifest_hash") != manifest["manifest_hash"]):
        raise StudyError("No matching size-control freeze/access authorization")
    frozen_at = datetime.fromisoformat(freeze["created_at"])
    accessed_at = datetime.fromisoformat(access["created_at"])
    if frozen_at.tzinfo is None or accessed_at.tzinfo is None or accessed_at < frozen_at:
        raise StudyError("Evaluation access chronology differs from the original freeze")
    target = _safe(output, "sealed/holdout_gold.json")
    expected = protocol()["data"]["source_sha256"]["sealed/holdout_gold.json"]
    source = target if target.exists() else _safe(manifest["source_v3_dir"], "sealed/holdout_gold.json")
    if file_hash(source) != expected:
        raise StudyError("Original ANLI gold differs from the accepted fixed pool")
    gold = read_json(source)
    if set(gold) != {e["sample_id"] for e in examples(output, "holdout")} or not set(gold.values()) <= set(old_data.LABELS):
        raise StudyError("Original ANLI gold membership differs")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return gold
