#!/usr/bin/env python3
"""Input-only local ANLI exposure census. Never selects a formal split or reads gold fields."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import unicodedata
import zipfile

SOURCE_SHA256 = "e5c058f2bb4e6190b0651badca2c590e45db95248f8b28cde674615ee40820bf"
LEDGER_SHA256 = "2c61052dfccc1d2bc4a7adf5a27d54d219a8c418886be151bd7eac181410d606"
COUNTS = {"R1": {"train": 16946, "dev": 1000, "test": 1000},
          "R2": {"train": 45460, "dev": 1000, "test": 1000},
          "R3": {"train": 100459, "dev": 1200, "test": 1200}}
PRUNE = {".git", "node_modules", "cache", "envs", "weights", "model", "models",
         "__pycache__"}
TEXT_EXTENSIONS = {".json", ".jsonl", ".csv", ".md", ".py", ".sh", ".txt", ".log", ".yaml", ".yml", ".ipynb"}
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
ANLI_REF = re.compile(rb"anli:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
SAMPLE = re.compile(rb'"sample_id"\s*:\s*"(anli:[^"\\]+)"')
LIMIT = 100 * 1024 * 1024
DELIMITER = "\n\nHypothesis:\n"


class AuditError(ValueError):
    pass


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_bounded(path):
    with Path(path).open("rb") as handle:
        payload = handle.read(LIMIT + 1)
    if len(payload) > LIMIT:
        raise AuditError(f"Text/input exceeds audit bound: {path}")
    return payload


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def hashes(premise, hypothesis):
    return (digest(normalized(premise).encode()),
            digest((normalized(premise) + "\0" + normalized(hypothesis)).encode()))


def safe_relative(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or not relative or "\\" in relative:
        raise AuditError(f"Unsafe relative path: {relative}")
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise AuditError(f"Path escapes root: {relative}")
    return path


def zip_members(archive):
    entries = archive.infolist()
    names = [x.filename for x in entries]
    if len(names) != len(set(names)):
        raise AuditError("Duplicate ZIP member")
    for info in entries:
        p = PurePosixPath(info.filename)
        if p.is_absolute() or ".." in p.parts or "\\" in info.filename:
            raise AuditError("Unsafe ZIP member")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise AuditError("Symlink ZIP member")
        if info.file_size > LIMIT:
            raise AuditError(f"ZIP member exceeds audit bound: {info.filename}")
    return entries


def input_projection(pairs):
    # JSON decoding must process the containing record, but label/reason/verifier
    # fields are immediately discarded and never accessed by the audit logic.
    return {key: value for key, value in pairs if key in {"uid", "context", "hypothesis"}}


def read_source(path, expected_sha=SOURCE_SHA256, counts=COUNTS):
    data = read_bounded(path)
    if digest(data) != expected_sha:
        raise AuditError("Official source archive SHA differs")
    result, members = {}, {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        zip_members(archive)
        for rd, splits in counts.items():
            for split, count in splits.items():
                name = f"anli_v1.0/{rd}/{split}.jsonl"
                payload = archive.read(name)
                lines = [line for line in payload.splitlines() if line.strip()]
                if len(lines) != count:
                    raise AuditError(f"Source count differs: {name}")
                members[name] = {"sha256": digest(payload), "count": count}
                for line in lines:
                    row = json.loads(line, object_pairs_hook=input_projection)
                    if any(not isinstance(row.get(k), str) or not row[k].strip()
                           for k in ("uid", "context", "hypothesis")):
                        raise AuditError("Invalid source input schema")
                    sid = "anli:" + row["uid"]
                    if sid in result:
                        raise AuditError("Duplicate source UID")
                    ph, ih = hashes(row["context"], row["hypothesis"])
                    result[sid] = {"sample_id": sid, "round": rd, "source_split": split,
                                   "premise_hash": ph, "input_hash": ih}
    return result, members


def verify_input_rows(payload, source, schema):
    ids = set()
    for line in payload.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("sample_id")
        if sid not in source:
            raise AuditError("Unknown historical scientific sample_id")
        if sid in ids:
            raise AuditError("Duplicate historical sample_id")
        if schema == "v3":
            state = row.get("state")
            if not isinstance(state, str) or not state.startswith("Premise:\n") or state.count(DELIMITER) != 1:
                raise AuditError("Unknown historical input schema")
            premise, hypothesis = state[len("Premise:\n"):].split(DELIMITER)
            if row.get("task") != "anli":
                raise AuditError("Historical task differs")
        elif schema == "cpu":
            premise, hypothesis = row.get("field_a"), row.get("field_b")
            if not isinstance(premise, str) or not isinstance(hypothesis, str):
                raise AuditError("Unknown CPU historical input schema")
        else:
            raise AuditError("Unknown ledger input schema")
        ph, ih = hashes(premise, hypothesis)
        expected = source[sid]
        if (ph, ih) != (expected["premise_hash"], expected["input_hash"]):
            raise AuditError("Historical input differs from pinned source")
        if row.get("input_hash") != ih or row.get("premise_hash", row.get("first_text_hash")) != ph:
            raise AuditError("Historical serialized input hash differs")
        ids.add(sid)
    return ids


def census_ids(payload, source, location, scientific=False, by_uuid=None):
    if by_uuid is None:
        by_uuid = {sid[5:].encode(): sid for sid in source}
    found = {by_uuid[x] for x in UUID.findall(payload) if x in by_uuid}
    for raw in SAMPLE.findall(payload):
        sid = raw.decode("utf-8")
        if sid in source:
            found.add(sid)
        elif UUID.fullmatch(raw[5:]) or scientific:
            raise AuditError(f"Unknown scientific sample_id in {location}")
    if scientific:
        for uid in ANLI_REF.findall(payload):
            if uid not in by_uuid:
                raise AuditError(f"Unknown scientific ANLI reference in {location}")
    return found


def iter_files(root, excluded_paths=(), skipped_symlinks=None):
    excluded = {p.resolve() for p in excluded_paths}
    for parent, dirs, files in os.walk(root, followlinks=False):
        keep = []
        for directory in dirs:
            path = Path(parent) / directory
            if directory in PRUNE or directory.startswith(".venv") or path.resolve() in excluded:
                continue
            if path.is_symlink():
                if skipped_symlinks is not None:
                    skipped_symlinks.append(str(path))
                continue
            keep.append(directory)
        dirs[:] = sorted(keep)
        for name in sorted(files):
            path = Path(parent) / name
            if path.suffix.lower() not in TEXT_EXTENSIONS | {".zip"}:
                continue
            if path.is_symlink():
                if skipped_symlinks is not None:
                    skipped_symlinks.append(str(path))
                continue
            if path.stat().st_size > LIMIT and path.suffix.lower() != ".zip":
                raise AuditError(f"Text file exceeds audit bound: {path}")
            yield path


def jsonl(path, rows):
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def git_blobs(root):
    """Reachable historical text blobs across all local refs, once per object."""
    if not (root / ".git").exists():
        return
    proc = subprocess.run(["git", "-C", str(root), "rev-list", "--objects", "--all"],
                          capture_output=True, check=True, timeout=60)
    for line in proc.stdout.decode().splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        oid, name = parts
        path = PurePosixPath(name)
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in PRUNE or part.startswith(".venv") for part in path.parts):
            continue
        size = int(subprocess.check_output(["git", "-C", str(root), "cat-file", "-s", oid], timeout=30))
        if size > LIMIT:
            raise AuditError(f"Historical Git blob exceeds audit bound: {oid}")
        payload = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", oid], timeout=30)
        yield oid, name, payload


def audit(repo_root, evidence_root, ledger_path, output_dir):
    script_identity = file_hash(__file__)
    roots = {"repo": Path(repo_root).resolve(), "evidence": Path(evidence_root).resolve()}
    output = Path(output_dir).resolve()
    if output.exists() or any(root.is_relative_to(output) for root in roots.values()):
        raise AuditError("Output must be a new directory and cannot contain an input root")
    if file_hash(ledger_path) != LEDGER_SHA256:
        raise AuditError("Exposure ledger differs from the reviewed pin")
    ledger = json.loads(Path(ledger_path).read_text())
    if ledger.get("schema_version") != 1:
        raise AuditError("Unknown ledger schema")
    src = ledger["source"]
    source_path = safe_relative(roots[src["root"]], src["path"])
    source, members = read_source(source_path)
    by_uuid = {sid[5:].encode(): sid for sid in source}
    uses, inventory = defaultdict(set), []
    for entry in ledger["historical_inputs"]:
        location = f"{entry['root']}:{entry['path']}"
        path = safe_relative(roots[entry["root"]], entry["path"])
        payload = read_bounded(path)
        if digest(payload) != entry["sha256"]:
            raise AuditError(f"Historical input SHA differs: {location}")
        ids = verify_input_rows(payload, source, entry["schema"])
        if len(ids) != entry["count"]:
            raise AuditError(f"Historical count differs: {location}")
        for sid in ids:
            uses[sid].add(location + ":" + entry["use"])
    # Archive identity and canonical input-member pins are independent checks.
    for entry in ledger["archives"]:
        path = safe_relative(roots[entry["root"]], entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise AuditError(f"Historical archive SHA differs: {entry['path']}")
        with zipfile.ZipFile(path) as archive:
            zip_members(archive)
            for item in entry.get("input_members", []):
                payload = archive.read(item["path"])
                if digest(payload) != item["sha256"]:
                    raise AuditError("Historical ZIP input member SHA differs")
                verify_input_rows(payload, source, item["schema"])
        if file_hash(path) != entry["sha256"]:
            raise AuditError("Historical archive changed during verification")
    scan_exclusions = []
    for item in ledger["scan_exclusions"]:
        scan_exclusions.append(safe_relative(roots[item["root"]], item["path"]))
    if any(output.is_relative_to(root) for root in roots.values()) and not any(output.is_relative_to(path) for path in scan_exclusions):
        raise AuditError("Output inside a scanned root must be under a reviewed audit-only exclusion")
    skipped_symlinks = []
    for root_name, root in roots.items():
        for path in iter_files(root, scan_exclusions, skipped_symlinks):
            location = f"{root_name}:{path.relative_to(root).as_posix()}"
            if path.suffix.lower() == ".zip":
                sha = file_hash(path)
                if sha == SOURCE_SHA256:
                    inventory.append({"location": location, "sha256": sha, "category": "raw_source_audit_only", "source_uid_count": 0})
                    continue
                with zipfile.ZipFile(path) as archive:
                    infos = zip_members(archive)
                    if any(x.filename.startswith("anli_v1.0/") for x in infos):
                        raise AuditError("Unpinned raw ANLI archive")
                    for info in infos:
                        p = PurePosixPath(info.filename)
                        if p.suffix.lower() not in TEXT_EXTENSIONS:
                            continue
                        member_location = location + "!" + info.filename
                        payload = archive.read(info)
                        ids = census_ids(payload, source, member_location, scientific=p.suffix.lower() in {'.json', '.jsonl'}, by_uuid=by_uuid)
                        for sid in ids:
                            uses[sid].add(member_location)
                        inventory.append({"location": member_location, "sha256": digest(payload), "archive_sha256": sha,
                                          "category": "historical_archive_text", "source_uid_count": len(ids)})
                if file_hash(path) != sha:
                    raise AuditError(f"Archive changed during census: {location}")
            else:
                payload = read_bounded(path)
                sha = digest(payload)
                ids = census_ids(payload, source, location, scientific=path.suffix.lower() in {'.json', '.jsonl'}, by_uuid=by_uuid)
                for sid in ids:
                    uses[sid].add(location)
                inventory.append({"location": location, "sha256": sha, "category": "historical_local_text", "source_uid_count": len(ids)})
        for oid, name, payload in git_blobs(root):
            location = f"{root_name}:git:{oid}:{name}"
            ids = census_ids(payload, source, location, scientific=PurePosixPath(name).suffix.lower() in {'.json', '.jsonl'}, by_uuid=by_uuid)
            for sid in ids:
                uses[sid].add(location)
            inventory.append({"location": location, "sha256": digest(payload), "category": "historical_git_blob", "source_uid_count": len(ids)})
    historical = [source[sid] for sid in sorted(uses)]
    historical_ph = {x["premise_hash"] for x in historical}
    historical_ih = {x["input_hash"] for x in historical}
    dev, exclusions = [], []
    for row in source.values():
        if row["source_split"] != "dev":
            continue
        reasons = []
        if row["sample_id"] in uses:
            reasons.append("historical_sample_id")
        if row["premise_hash"] in historical_ph:
            reasons.append("historical_premise")
        if row["input_hash"] in historical_ih:
            reasons.append("historical_input")
        if reasons:
            exclusions.append({**row, "reasons": reasons})
        else:
            dev.append(row)
    dev_ph = {x["premise_hash"] for x in dev}
    test_ph = {x["premise_hash"] for x in source.values() if x["source_split"] == "test"}
    fitting = [x for x in source.values() if x["source_split"] == "train" and x["premise_hash"] not in test_ph | dev_ph]
    fresh = [x for x in fitting if x["premise_hash"] not in historical_ph and x["input_hash"] not in historical_ih]
    summarize = lambda rows: {"count": len(rows), "premise_count": len({x['premise_hash'] for x in rows}),
                              "round_counts": dict(sorted(Counter(x['round'] for x in rows).items()))}
    report = {"schema_version": 1, "status": "BLOCKED_UNRESOLVED_LOCAL_SYMLINKS" if skipped_symlinks else "LOCAL_UID_CENSUS_COMPLETE_UNKEYED_AND_EXTERNAL_USAGE_UNCONFIRMED",
              "source_sha256": SOURCE_SHA256, "source_members": members, "ledger_sha256": LEDGER_SHA256,
              "script_sha256": script_identity, "historical": summarize(historical), "candidate_dev": summarize(dev),
              "dev_exclusion_reasons": dict(Counter(r for x in exclusions for r in x['reasons'])),
              "fitting_candidates_allow_historical_train_reuse": summarize(fitting), "fresh_train_pool": summarize(fresh),
              "scanned_entries": len(inventory), "raw_source_audit_only_entries": sum(x['category']=='raw_source_audit_only' for x in inventory),
              "unresolved_symlinks": skipped_symlinks,
              "scan_exclusions": [{**item, "classification": "audit_only_not_historical_model_use"} for item in ledger['scan_exclusions']],
              "pruned_directory_names": sorted(PRUNE),
              "pruned_directory_prefixes": [".venv"],
              "notes": ["No label values, performance or class proportions are used or emitted.",
                        "Counts precede input-duplicate and conflicting-label resolution; no formal split selected.",
                        "Conservative exposure includes recognized source UIDs mentioned in local prose or fixtures.",
                        "Census is UID-based. UID-less copied input/few-shot text is unresolved; canonical pinned scientific input rows are separately verified by normalized hashes.",
                        "Non-UUID synthetic sample IDs outside canonical scientific inputs are not source observations.",
                        "Historical gold/sealed files are scanned for input IDs only; their label values are never interpreted or emitted.",
                        "Pruned environment/cache/weight directories and explicitly excluded current-work directory are listed by policy.",
                        "Outside-local teammate/node usage is unconfirmed. This is not a declaration of globally unseen data.",
                        "Snapshot inventory is stale after historical files change; rerun before freezing data."]}
    if file_hash(__file__) != script_identity:
        raise AuditError("Audit source changed during execution; rerun the final source")
    output.mkdir(parents=True)
    jsonl(output / "history.jsonl", ({**source[sid], "locations": sorted(uses[sid])} for sid in sorted(uses)))
    jsonl(output / "inventory.jsonl", inventory)
    jsonl(output / "candidate-dev.jsonl", sorted(dev, key=lambda x: x['sample_id']))
    jsonl(output / "dev-exclusions.jsonl", sorted(exclusions, key=lambda x: x['sample_id']))
    jsonl(output / "fitting-candidates.jsonl", sorted(fitting, key=lambda x: x['sample_id']))
    jsonl(output / "fresh-train-pool.jsonl", sorted(fresh, key=lambda x: x['sample_id']))
    report['output_sha256'] = {p.name: file_hash(p) for p in sorted(output.iterdir())}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    ledger = args.ledger or args.repo_root / "records/anli-exposure-inventory-2026-11-12.json"
    try:
        report = audit(args.repo_root, args.evidence_root, ledger, args.output_dir)
    except (AuditError, OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: report[k] for k in ('status', 'historical', 'candidate_dev', 'fresh_train_pool')}, indent=2))
    return 1 if report['status'].startswith('BLOCKED') else 0


if __name__ == "__main__":
    raise SystemExit(main())
