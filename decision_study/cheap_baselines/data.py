"""Pinned, portable training-only CLINC inputs for the expanded cheap baselines.

Preparation may parse the public all-splits source to extract official training
rows. Fitting loads only these prepared rows and the pinned project input files;
neither path opens the project's V1 gold file.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import unicodedata
import urllib.request

LABELS = ("account_blocked", "balance", "bill_balance", "bill_due", "freeze_account",
          "interest_rate", "min_payment", "order_checks", "pay_bill", "pin_change",
          "report_fraud", "routing", "spending_history", "transactions", "transfer", "oos")
COUNTS = {"calibration": 400, "validation": 400, "test": 600, "stress": 850}
RAW_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
RAW_PINS = {
    "data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "domains.json": "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a",
}
RAW_URLS = {name: f"https://raw.githubusercontent.com/clinc/oos-eval/{RAW_REVISION}/data/{name}"
            for name in RAW_PINS}
SOURCE_PINS = {
    "v1_manifest_sha256": "7364419fcf52eaf691a40282c6027eec62a026cef0b41a6c9aa31eb06a593d4b",
    "v1_examples_sha256": "a0b699e5fe48e003ddcbb6cab751b0f088edb0b5d297ee46c250121bfdd10703",
    "v1_gold_sha256": "bc5ec4891caed470970115a41e2f6debe2a2967b3f5ff3496622a3d29311fc35",
    "v2_examples_sha256": "f29fac859f0324e7ed7d4cf54bdc674472a27451695ee9fff545d635d23bb25b",
    "v2_fit_gold_sha256": "e44283e1775f8085c09ee9770e98038d527e4810a7e50626cc4240ab76078e9b",
}
PREPARED_PINS = {
    "train1600.jsonl": "cec43d033880b75b77933bed0b03d345a36321a63f105f53e279851bd654f468",
    "manifest.json": "abcb276997e62d2d53dce4a3cbe745d25d78b3462c31524b1687bba23cc5899b",
}


class DataError(RuntimeError):
    """Preserve an identity conflict instead of silently repairing input data."""


def _require(condition, message):
    if not condition:
        raise DataError(message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _norm(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _root(path, *, exists=True):
    path = Path(path).expanduser().absolute()
    _require(not path.is_symlink(), f"Symlink root: {path}")
    _require(not exists or path.is_dir(), f"Missing input directory: {path}")
    return path.resolve()


def _safe(root, name):
    relative = Path(name)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe input path")
    path = root
    for part in relative.parts:
        path /= part
        _require(not path.is_symlink(), f"Symlink input: {path}")
    return path


def _read(root, name, expected):
    path = _safe(root, name)
    _require(path.is_file(), f"Missing file: {path}")
    content = path.read_bytes()
    _require(_hash(content) == expected, f"SHA256 mismatch: {path}")
    return content


def _rows(content):
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _disjoint(output, inputs):
    for other in inputs:
        _require(output != other and output not in other.parents and other not in output.parents,
                 "Prepared output must be separate from raw and project input roots")


def _project(v1_dir, v2_dir):
    v1, v2 = _root(v1_dir), _root(v2_dir)
    manifest = json.loads(_read(v1, "data/manifest.json", SOURCE_PINS["v1_manifest_sha256"]))
    _require(manifest.get("protocol") == "solo-v1" and
             manifest["files"].get("examples.jsonl") == SOURCE_PINS["v1_examples_sha256"] and
             manifest["files"].get("gold.json") == SOURCE_PINS["v1_gold_sha256"],
             "V1 manifest identity differs")
    original = _rows(_read(v1, "data/examples.jsonl", SOURCE_PINS["v1_examples_sha256"]))
    clinc = [row for row in original if row.get("task") == "clinc"]
    _require(Counter(row["split"] for row in clinc) == Counter({**COUNTS, "development": 64}),
             "CLINC role counts differ")
    by_id, normalized = {}, set()
    for row in clinc:
        sid, normalized_text = row["sample_id"], _norm(row["state"])
        _require(sid not in by_id and normalized_text and normalized_text not in normalized and
                 row.get("labels") == list(LABELS) and not
                 any(key in row for key in ("gold", "label", "answers", "is_impossible")),
                 "Duplicate, invalid, or label-leaking CLINC project input")
        by_id[sid] = row
        normalized.add(normalized_text)
    roles = {name: sorted((r for r in clinc if r["split"] == name), key=lambda r: r["sample_id"])
             for name in COUNTS}
    v2rows = _rows(_read(v2, "data/examples.jsonl", SOURCE_PINS["v2_examples_sha256"]))
    fitting = [r for r in v2rows if r.get("task") == "clinc"]
    _require(Counter(r["split"] for r in fitting) == Counter(calibration=400, validation=400) and
             len({r["sample_id"] for r in fitting}) == 800 and
             all(by_id.get(r["sample_id"]) == r for r in fitting), "V2 fitting rows differ from V1")
    fit_gold = json.loads(_read(v2, "data/fit_gold.json", SOURCE_PINS["v2_fit_gold_sha256"]))
    expected = Counter({label: 20 for label in LABELS[:-1]})
    expected["oos"] = 100
    for name in ("calibration", "validation"):
        _require(Counter(fit_gold[r["sample_id"]] for r in roles[name]) == expected,
                 "Fitting role class counts differ")
    return {"roles": roles, "predict": roles["validation"] + roles["test"] + roles["stress"],
            "fit_gold": fit_gold, "development": [r for r in clinc if r["split"] == "development"],
            "v1": v1, "v2": v2}


def _validate_training(training, project):
    _require(len(training) == 1600 and training == sorted(training, key=lambda r: r["sample_id"]),
             "Training pool size or order differs")
    by_id, normalized = {}, set()
    for row in training:
        _require(set(row) == {"sample_id", "state", "label", "source_split"}, "Training schema differs")
        sid, text, label, source = (row[k] for k in ("sample_id", "state", "label", "source_split"))
        _require(source in ("train", "oos_train") and label in LABELS and
                 (label == "oos") == (source == "oos_train"), "Unexpected training source/label")
        norm = _norm(text)
        _require(norm and norm not in normalized and sid not in by_id, "Duplicate training input")
        _require(sid.startswith(f"clinc:{source}:") and sid.endswith(":" + _hash(text.encode())[:16]),
                 "Training sample ID differs from source/text")
        by_id[sid] = row
        normalized.add(norm)
    _require(Counter(r["label"] for r in training) == Counter({label: 100 for label in LABELS}),
             "Official training class counts differ")
    calibration = []
    for row in project["roles"]["calibration"]:
        sid = row["sample_id"]
        _require(sid in by_id and by_id[sid]["state"] == row["state"] and
                 by_id[sid]["source_split"] == row["source_split"] and
                 by_id[sid]["label"] == project["fit_gold"][sid], "Calibration is not exact training subset")
        calibration.append(by_id[sid])
    _require(not ({r["sample_id"] for r in project["predict"]} & by_id.keys()) and
             not ({_norm(r["state"]) for r in project["predict"]} & normalized),
             "Official training overlaps validation/test/stress")
    included_dev = [r for r in project["development"] if r["sample_id"] in by_id]
    excluded_dev = [r for r in project["development"] if r["sample_id"] not in by_id]
    _require(len(included_dev) == 60 and len(excluded_dev) == 4 and
             all(by_id[r["sample_id"]]["state"] == r["state"] for r in included_dev) and
             all(r["sample_id"].startswith("clinc:authored-oos:") and _norm(r["state"]) not in normalized
                 for r in excluded_dev), "Engineering development overlap differs")
    return calibration


def _prepared_content(raw_dir, project):
    raw = _root(raw_dir)
    source = json.loads(_read(raw, "data_full.json", RAW_PINS["data_full.json"]))
    domains = json.loads(_read(raw, "domains.json", RAW_PINS["domains.json"]))
    _require(sorted(domains.get("banking", [])) == list(LABELS[:-1]), "Banking domain differs")
    _require(len(source["train"]) == 15000 and len(source["oos_train"]) == 100,
             "Official source training sizes differ")
    training = []
    for split in ("train", "oos_train"):
        for index, (text, label) in enumerate(source[split]):
            _require(split != "oos_train" or label == "oos", "Unexpected OOS training label")
            if label not in LABELS:
                continue
            training.append({"sample_id": f"clinc:{split}:{index}:{_hash(text.encode())[:16]}",
                             "state": text, "label": label, "source_split": split})
    training.sort(key=lambda r: r["sample_id"])
    _validate_training(training, project)
    train_bytes = b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
                           for r in training)
    manifest = {"schema_version": 1, "data_id": "clinc-cheap-baselines-v1", "label_order": list(LABELS),
                "raw_revision": RAW_REVISION, "raw_sources": RAW_PINS, "project_sources": SOURCE_PINS,
                "files": {"train1600.jsonl": _hash(train_bytes)}, "training_count": 1600,
                "training_label_counts": {label: 100 for label in LABELS},
                "audit": {"calibration_subset_count": 400, "engineering_development_included": 60,
                          "authored_development_excluded": 4, "training_normalized_unique": 1600,
                          "evaluation_id_overlap": 0, "evaluation_normalized_text_overlap": 0,
                          "prediction_counts": {k: COUNTS[k] for k in ("validation", "test", "stress")}},
                "boundary": "Preparation parses the public all-splits source; fitting reads only the extracted training artifact. Test/stress are previously viewed retrospective data."}
    return {"train1600.jsonl": train_bytes, "manifest.json": _bytes(manifest)}


def prepare(raw_dir, v1_dir, v2_dir, output_dir):
    """Extract the fixed training-only artifact; never read/hash V1 gold."""
    project = _project(v1_dir, v2_dir)
    raw, output = _root(raw_dir), _root(output_dir, exists=False)
    _disjoint(output, (raw, project["v1"], project["v2"]))
    contents = _prepared_content(raw, project)
    for name, content in contents.items():
        _require(_hash(content) == PREPARED_PINS[name], f"Prepared {name} differs from released data")
    output.mkdir(parents=True, exist_ok=True)
    present = {p.name for p in output.iterdir()}
    _require(not present or present == set(contents), "Partial/nonempty prepared directory; preserve it")
    for name, content in contents.items():
        path = _safe(output, name)
        if path.exists():
            _require(path.read_bytes() == content, f"Existing prepared file differs: {path}")
        else:
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
    return json.loads(contents["manifest.json"])


def load(prepared_dir, v1_dir, v2_dir):
    """Load pinned training/evaluation inputs without raw data or V1 gold access."""
    prepared = _root(prepared_dir)
    project = _project(v1_dir, v2_dir)
    _disjoint(prepared, (project["v1"], project["v2"]))
    manifest = json.loads(_read(prepared, "manifest.json", PREPARED_PINS["manifest.json"]))
    _require(manifest.get("data_id") == "clinc-cheap-baselines-v1" and
             manifest["project_sources"] == SOURCE_PINS and manifest["raw_sources"] == RAW_PINS and
             manifest["label_order"] == list(LABELS) and
             manifest["files"] == {"train1600.jsonl": PREPARED_PINS["train1600.jsonl"]},
             "Prepared manifest identity differs")
    training = _rows(_read(prepared, "train1600.jsonl", PREPARED_PINS["train1600.jsonl"]))
    calibration = _validate_training(training, project)
    return {"training": {"400": calibration, "1600": training}, "roles": project["roles"],
            "predict": project["predict"], "fit_gold": project["fit_gold"],
            "sources": dict(SOURCE_PINS), "prepared_hashes": dict(PREPARED_PINS),
            "input_identity": {"project_sources": dict(SOURCE_PINS), "prepared_files": dict(PREPARED_PINS)},
            "input_roots": {"prepared": str(prepared), "v1": str(project["v1"]), "v2": str(project["v2"])}}


def download(raw_dir):
    """Fetch only two pinned official files; conflicting existing bytes block."""
    raw = _root(raw_dir, exists=False)
    raw.mkdir(parents=True, exist_ok=True)
    _require({p.name for p in raw.iterdir()} <= set(RAW_PINS), "Raw download directory contains unrelated files")
    # Check every existing file before any request or write.
    for name, expected in RAW_PINS.items():
        path = _safe(raw, name)
        if path.exists():
            _read(raw, name, expected)
    for name, expected in RAW_PINS.items():
        path = _safe(raw, name)
        if path.exists():
            continue
        request = urllib.request.Request(RAW_URLS[name], headers={"User-Agent": "decision-study-cheap-baselines/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read(10_000_001)
        _require(len(content) <= 10_000_000 and _hash(content) == expected,
                 f"Downloaded source differs: {name}")
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    return {name: {"url": RAW_URLS[name], "sha256": expected} for name, expected in RAW_PINS.items()}
