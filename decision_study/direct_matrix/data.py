"""Exact saved inputs and verified historical cells; test gold opens only after the freeze.

``prepare`` reads each source ``data/examples.jsonl`` directly after checking its
pinned SHA-256. It never calls the V1 loader, because that loader hashes every
manifest file including ``data/gold.json``. The historical verifiers used for the
12 reused cells read inputs, predictions, receipts and (V2, V3) calibration and
validation labels only, never a test-gold file.

The prepared ``data/*.jsonl`` files are canonical reserializations of the selected
rows: every original value and the task text are kept, and ``data/source_lines.json``
records each row's source line number and the SHA-256 of its original line bytes.
The subset files are therefore not byte copies of the source lines.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from ..adapters import parse_response
from ..common import StudyError, canonical, file_hash, fingerprint, read_json, read_jsonl, utc_now, write_json
from ..storage import load_predictions
from .identity import (CELL_ORDER, DATASETS, DEV_SETS, PROJECT, REFERENCE_CELLS, dev_set, fixed_write, normal, protocol,
                       protocol_hash, safe, source_files, split_cell)

FORBIDDEN_FIELDS = {"gold", "label", "answers", "answer", "is_impossible", "target"}
REQUEST_FIELDS = ("state", "instructions", "criteria", "labels")


def seal(payload, key):
    return {**payload, key: fingerprint(payload)}


def checked(path, key):
    value = read_json(path)
    if not isinstance(value, dict) or value.get(key) != fingerprint({k: v for k, v in value.items() if k != key}):
        raise StudyError(f"Artifact self hash changed: {path}")
    return value


def _within(path, root):
    path, root = normal(path), normal(root)
    return path == root or root in path.parents


def source_dirs(v1_dir, v2_dir, v3_dir, size_control_dir):
    dirs = {"v1": v1_dir, "v2": v2_dir, "v3": v3_dir, "size_control": size_control_dir}
    return {name: str(normal(path)) for name, path in dirs.items()}


def check_output_location(output, dirs):
    output = normal(output)
    for root in [*map(Path, dirs.values()), PROJECT]:
        if _within(output, root) or _within(root, output):
            raise StudyError(f"Use an independent output directory that overlaps no input or source root: {root}")


def _source_rows(directory, spec):
    """Rows of one pinned source file, in file order, with their exact line hashes."""
    path = safe(directory, spec["file"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["file_sha256"]:
        raise StudyError(f"Source inputs differ from the accepted file: {path}")
    rows = []
    for number, line in enumerate(data.split(b"\n"), 1):
        if line.strip():
            rows.append((number, hashlib.sha256(line).hexdigest(), json.loads(line)))
    return rows


def _select(rows, selector):
    return [(n, h, r) for n, h, r in rows if all(r.get(k) == v for k, v in selector.items())]


def _check_set(name, selected, spec, labels=None):
    rows = [r for _, _, r in selected]
    ids = [r["sample_id"] for r in rows]
    if len(rows) != spec["count"] or len(set(ids)) != len(ids):
        raise StudyError(f"{name}: expected {spec['count']} unique inputs, found {len(rows)}")
    if fingerprint(sorted(ids)) != spec["ids_fingerprint"]:
        raise StudyError(f"{name}: sample-ID membership differs from the protocol")
    if fingerprint(sorted(rows, key=lambda r: r["sample_id"])) != spec["rows_fingerprint"]:
        raise StudyError(f"{name}: saved input rows differ from the protocol")
    for row in rows:
        if set(row) & FORBIDDEN_FIELDS:
            raise StudyError(f"{name}: an input row carries a gold-like field")
        if labels is not None and list(row["labels"]) != labels:
            raise StudyError(f"{name}: label order differs from the protocol")
        # The request builders order criteria by the saved label order, so only membership must agree.
        if (set(row["criteria"]) != set(row["labels"]) or len(set(row["labels"])) != len(row["labels"])
                or not isinstance(row.get("state"), str) or not row["state"]):
            raise StudyError(f"{name}: criteria/labels/state are malformed")
    return rows


def _historical_rows(examples, rows, model, *, config_hash=None, revision=None):
    """Exact ID coverage and parse of the raw response for every reused prediction."""
    expected = {e["sample_id"]: e for e in examples}
    by_id = {}
    for row in rows:
        if row["sample_id"] in expected:
            if row["sample_id"] in by_id:
                raise StudyError("Duplicate historical prediction")
            by_id[row["sample_id"]] = row
    if set(by_id) != set(expected):
        raise StudyError("Historical predictions do not cover the cell exactly")
    for sid, row in by_id.items():
        e = expected[sid]
        if row.get("model") != model or row.get("task") != e["task"] or row.get("split") != e["split"]:
            raise StudyError("Historical prediction identity differs")
        if config_hash is not None and row.get("config_hash") != config_hash:
            raise StudyError("Historical prediction configuration differs")
        if revision is not None and row.get("model_revision") != revision:
            raise StudyError("Historical prediction checkpoint differs")
        if row.get("status") == "ok":
            if parse_response(row["raw_response"], e, model) != (row.get("prediction"), row.get("probabilities")):
                raise StudyError("Historical prediction differs from its raw response")
        elif row.get("status") != "error" or row.get("prediction") is not None:
            raise StudyError("Historical prediction status is not terminal")
    return [by_id[e["sample_id"]] for e in examples]


def _references(dirs, formal):
    """Verify the 12 reused cells with their own studies' verifiers and the protocol pins."""
    from ..size_control import study as size_study
    from ..v2 import data_v2, study as v2_study
    from ..v3 import study as v3_study
    pins = protocol()["references"]
    v1, v2, v3, size = (Path(dirs[k]) for k in ("v1", "v2", "v3", "size_control"))
    compact, provenance = {}, {}
    v1_cache = {}
    for cell in REFERENCE_CELLS:
        pin = pins[cell]
        role, dataset = split_cell(cell)
        examples = formal[dataset]
        model = pin["model"]
        evidence = {"study": pin["study"], "source_dir": dirs[pin["source"]], "config_hash": pin["config_hash"]}
        if pin["study"] == "v1":
            if model not in v1_cache:
                config = read_json(safe(v1, f"predictions/{model}/config.json"))
                if fingerprint(config) != pin["config_hash"]:
                    raise StudyError(f"V1 {model} configuration differs from the accepted run")
                job = load_predictions(v1, model)
                if fingerprint(sorted(job, key=lambda r: r["sample_id"])) != pin["job_predictions_hash"]:
                    raise StudyError(f"V1 {model} predictions differ from the accepted run")
                v1_cache[model] = (config, job)
            config, job = v1_cache[model]
            rows = _historical_rows(examples, job, model, config_hash=pin["config_hash"],
                                    revision=config["runtime"]["model_revision"])
            evidence.update(verifier="decision_study.storage.load_predictions + pinned config/job hashes",
                            job_predictions_hash=pin["job_predictions_hash"])
        elif pin["study"] == "v2":
            manifest = data_v2.verify_fit(v2)
            index = v2_study._job_predictions(v1, v2, manifest, examples, pin["job"], model)
            rows = _historical_rows(examples, list(index.values()), model, config_hash=pin["config_hash"])
            evidence.update(verifier="decision_study.v2.study._job_predictions", job=pin["job"])
        elif pin["study"] == "v3":
            found, done = v3_study.verified_job(v3, pin["role"], "holdout")
            if done["config_hash"] != pin["config_hash"]:
                raise StudyError(f"V3 {pin['role']} completion differs from the accepted run")
            rows = _historical_rows(examples, found, model, config_hash=pin["config_hash"])
            evidence.update(verifier="decision_study.v3.study.verified_job", role=pin["role"],
                            completion_predictions_hash=done["predictions_hash"])
        else:
            found, done = size_study.verified_job(size)
            if done["config_hash"] != pin["config_hash"] or done["completion_hash"] != pin["completion_hash"]:
                raise StudyError("Size-control completion differs from the accepted run")
            rows = _historical_rows(examples, found, model, config_hash=pin["config_hash"])
            evidence.update(verifier="decision_study.size_control.study.verified_job",
                            completion_hash=done["completion_hash"])
        full_hash = fingerprint(sorted(rows, key=lambda r: r["sample_id"]))
        if full_hash != pin["cell_predictions_hash"]:
            raise StudyError(f"Historical cell {cell} differs from its accepted predictions")
        compact[cell] = [{"sample_id": r["sample_id"], "status": r["status"], "prediction": r["prediction"]} for r in rows]
        if fingerprint(compact[cell]) != pin["compact_predictions_hash"]:
            raise StudyError(f"Historical cell {cell}: per-ID view differs from its pinned compact hash")
        evidence.update(cell_predictions_hash=full_hash, count=len(rows),
                        failures=sum(r["status"] != "ok" for r in rows), compact_hash=fingerprint(compact[cell]))
        provenance[cell] = evidence
    return compact, provenance


def build(dirs):
    """Selected inputs, development sets, references and provenance, all in memory."""
    p = protocol()
    sources = {}
    for name, spec in p["datasets"].items():
        key = (spec["source"], spec["file"], spec["file_sha256"])
        if key not in sources:
            sources[key] = _source_rows(dirs[spec["source"]], spec)
    by_source = {spec["source"]: sources[(spec["source"], spec["file"], spec["file_sha256"])]
                 for spec in p["datasets"].values()}
    formal, dev, lines = {}, {}, {}
    for name in DATASETS:
        spec = p["datasets"][name]
        selected = _select(by_source[spec["source"]], spec["selector"])
        formal[name] = _check_set(name, selected, spec, spec["labels"])
        lines[name] = [{"sample_id": r["sample_id"], "line": n, "line_sha256": h} for n, h, r in selected]
    for name in DEV_SETS:
        spec = p["development"][name]
        selected = _select(by_source[spec["source"]], spec["selector"])
        dev[name] = _check_set(f"development/{name}", selected, spec)
        lines["dev-" + name] = [{"sample_id": r["sample_id"], "line": n, "line_sha256": h} for n, h, r in selected]
    for name in DATASETS:
        d = dev[dev_set(name)]
        if ({r["sample_id"] for r in d} & {r["sample_id"] for r in formal[name]}
                or {r["group_id"] for r in d} & {r["group_id"] for r in formal[name]}):
            raise StudyError(f"Development inputs overlap the formal {name} inputs")
        if any((r["instructions"], r["criteria"], r["labels"]) != (formal[name][0]["instructions"], formal[name][0]["criteria"],
                                                                   formal[name][0]["labels"]) for r in d + formal[name]):
            raise StudyError(f"Task text differs between {name} formal and development inputs")
    compact, provenance = _references(dirs, formal)
    return formal, dev, lines, compact, provenance


def _payloads(formal, dev, lines, compact, provenance):
    lines_of = lambda rows: b"".join(canonical(row) + b"\n" for row in rows)
    payloads = {f"data/formal/{name}.jsonl": lines_of(rows) for name, rows in formal.items()}
    payloads.update({f"data/dev/{name}.jsonl": lines_of(rows) for name, rows in dev.items()})
    payloads["data/source_lines.json"] = canonical(lines) + b"\n"
    payloads.update({f"references/{cell}.jsonl": lines_of(rows) for cell, rows in compact.items()})
    payloads["references/provenance.json"] = canonical(provenance) + b"\n"
    return payloads


def prepare(dirs, output, *, source=None):
    """Write the exact inputs once; a second call verifies and changes nothing."""
    output = normal(output)
    check_output_location(output, dirs)
    target = safe(output, "data/manifest.json")
    if target.exists():
        manifest = verify(output)
        if manifest["source_dirs"] != dirs:
            raise StudyError("Prepared source directories differ; do not rebind existing evidence")
        return manifest
    if any(safe(output, name).exists() for name in ("freeze.json", "evaluation-access.json", "cells")):
        raise StudyError("Missing prepared manifest in an existing run; preserve the evidence")
    formal, dev, lines, compact, provenance = build(dirs)
    payloads = _payloads(formal, dev, lines, compact, provenance)
    for name, payload in payloads.items():
        path = safe(output, name)
        if path.exists() and path.read_bytes() != payload:
            raise StudyError(f"Existing partial preparation differs: {name}")
    for name, payload in payloads.items():
        path = safe(output, name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(payload)
            tmp.replace(path)
    manifest = seal({"schema_version": 1, "protocol": protocol()["protocol"], "protocol_hash": protocol_hash(),
                     "source_dirs": dirs, "source_files": source_files(), "source": source,
                     "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
                     "counts": {name: len(rows) for name, rows in formal.items()},
                     "development_counts": {name: len(rows) for name, rows in dev.items()},
                     "test_gold_opened": False, "created_at": utc_now(),
                     "notice": protocol()["scope"]}, "manifest_hash")
    write_json(target, manifest)
    return verify(output)


def verify(output):
    """Prepared inputs and references against the manifest and the protocol pins."""
    output = normal(output)
    manifest = checked(safe(output, "data/manifest.json"), "manifest_hash")
    p = protocol()
    expected = ({f"data/formal/{n}.jsonl" for n in DATASETS} | {f"data/dev/{n}.jsonl" for n in DEV_SETS}
                | {"data/source_lines.json", "references/provenance.json"} | {f"references/{c}.jsonl" for c in REFERENCE_CELLS})
    if (manifest.get("protocol") != p["protocol"] or manifest.get("protocol_hash") != protocol_hash()
            or manifest.get("source_files") != source_files() or set(manifest.get("files", {})) != expected):
        raise StudyError("Prepared protocol, source files or inventory changed")
    for name, sha in manifest["files"].items():
        if file_hash(safe(output, name)) != sha:
            raise StudyError(f"Prepared evidence changed: {name}")
    for name in DATASETS:
        rows = read_jsonl(safe(output, f"data/formal/{name}.jsonl"))
        spec = p["datasets"][name]
        _check_set(name, [(0, "", r) for r in rows], spec, spec["labels"])
    for name in DEV_SETS:
        rows = read_jsonl(safe(output, f"data/dev/{name}.jsonl"))
        _check_set(f"development/{name}", [(0, "", r) for r in rows], p["development"][name])
    provenance = read_json(safe(output, "references/provenance.json"))
    for cell in REFERENCE_CELLS:
        rows = read_jsonl(safe(output, f"references/{cell}.jsonl"))
        _, dataset = split_cell(cell)
        # The protocol pin, derived from the verified original rows, is what binds each per-ID label;
        # the provenance copy is only a convenience and is checked against it.
        if (fingerprint(rows) != p["references"][cell]["compact_predictions_hash"]
                or provenance[cell]["compact_hash"] != p["references"][cell]["compact_predictions_hash"]
                or provenance[cell]["cell_predictions_hash"] != p["references"][cell]["cell_predictions_hash"]
                or [r["sample_id"] for r in rows] != [r["sample_id"] for r in read_jsonl(safe(output, f"data/formal/{dataset}.jsonl"))]):
            raise StudyError(f"Prepared reference cell changed: {cell}")
    return manifest


def formal_examples(output, dataset):
    if dataset not in DATASETS:
        raise StudyError(f"Unknown dataset {dataset}")
    return read_jsonl(safe(output, f"data/formal/{dataset}.jsonl"))


def dev_examples(output, name):
    if name not in DEV_SETS:
        raise StudyError(f"Unknown development set {name}")
    return read_jsonl(safe(output, f"data/dev/{name}.jsonl"))


def cell_examples(output, cell_id):
    _, dataset = split_cell(cell_id)
    return formal_examples(output, dataset)


def references(output):
    return {cell: read_jsonl(safe(output, f"references/{cell}.jsonl")) for cell in REFERENCE_CELLS}


def gold_for_evaluation(output, freeze, access):
    """The first and only test-gold read; requires the verified freeze and access chain."""
    manifest = verify(output)
    if (checked(safe(output, "freeze.json"), "freeze_hash") != freeze
            or checked(safe(output, "evaluation-access.json"), "access_hash") != access
            or access.get("freeze_hash") != freeze["freeze_hash"]
            or freeze.get("manifest_hash") != manifest["manifest_hash"]):
        raise StudyError("No matching freeze/evaluation-access authorization for test gold")
    frozen_at, accessed_at = (datetime.fromisoformat(x["created_at"]) for x in (freeze, access))
    if frozen_at.tzinfo is None or accessed_at.tzinfo is None or accessed_at < frozen_at:
        raise StudyError("Evaluation access predates the freeze")
    p = protocol()
    gold, files = {}, {}
    for name in DATASETS:
        spec = p["gold"][name]
        path = safe(manifest["source_dirs"][spec["source"]], spec["file"])
        if file_hash(path) != spec["sha256"]:
            raise StudyError(f"Gold for {name} differs from the accepted file")
        values = read_json(path)
        ids = [e["sample_id"] for e in formal_examples(output, name)]
        labels = set(p["datasets"][name]["labels"])
        if any(values.get(sid) not in labels for sid in ids):
            raise StudyError(f"Gold for {name} does not cover the cell with valid labels")
        gold[name] = {sid: values[sid] for sid in ids}
        if fingerprint(gold[name]) != spec["subset_fingerprint"]:
            raise StudyError(f"Gold subset for {name} differs from its pinned fingerprint")
        target = safe(output, f"sealed/{name}-gold.json")
        fixed_write(target, gold[name])
        files[name] = {"source_sha256": spec["sha256"], "subset_sha256": file_hash(target), "count": len(ids),
                       "label_counts": dict(sorted(Counter(gold[name].values()).items()))}
    return gold, files


def load_sealed_gold(output):
    """Offline audit: the exported per-ID gold subsets, each checked against its protocol pin."""
    p = protocol()
    gold = {}
    for name in DATASETS:
        path = safe(output, f"sealed/{name}-gold.json")
        if not path.is_file():
            raise StudyError(f"Exported gold subset for {name} is missing")
        values = read_json(path)
        ids = [e["sample_id"] for e in formal_examples(output, name)]
        labels = set(p["datasets"][name]["labels"])
        if (fingerprint(values) != p["gold"][name]["subset_fingerprint"] or set(values) != set(ids)
                or any(values[sid] not in labels for sid in ids)):
            raise StudyError(f"Exported gold subset for {name} differs from its pinned fingerprint or inputs")
        gold[name] = values
    return gold
