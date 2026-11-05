"""Frozen 400-label CLINC TF-IDF baseline, with separate fitting and label access."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unicodedata
import warnings
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT / "configs/clinc_tfidf400_protocol.json"
LABELS = ("account_blocked", "balance", "bill_balance", "bill_due", "freeze_account",
          "interest_rate", "min_payment", "order_checks", "pay_bill", "pin_change",
          "report_fraud", "routing", "spending_history", "transactions", "transfer", "oos")
COUNTS = {"calibration": 400, "validation": 400, "test": 600, "stress": 850}
SOURCE_PINS = {
    "v1_manifest_sha256": "7364419fcf52eaf691a40282c6027eec62a026cef0b41a6c9aa31eb06a593d4b",
    "v1_examples_sha256": "a0b699e5fe48e003ddcbb6cab751b0f088edb0b5d297ee46c250121bfdd10703",
    "v1_gold_sha256": "bc5ec4891caed470970115a41e2f6debe2a2967b3f5ff3496622a3d29311fc35",
    "v2_examples_sha256": "f29fac859f0324e7ed7d4cf54bdc674472a27451695ee9fff545d635d23bb25b",
    "v2_fit_gold_sha256": "e44283e1775f8085c09ee9770e98038d527e4810a7e50626cc4240ab76078e9b",
}
PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl", "narwhals", "cloudpickle")
FIT_FILES = ("protocol.json", "model.json", "predictions.jsonl", "fit.json")


class BaselineError(RuntimeError):
    """An identity or phase boundary failed; preserve the run for inspection."""


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return _hash(Path(path).read_bytes())


def _seal(value):
    return {**value, "self_hash": _hash(_bytes(value))}


def _unseal(value):
    content = {k: v for k, v in value.items() if k != "self_hash"}
    if value.get("self_hash") != _hash(_bytes(content)):
        raise BaselineError("Artifact self hash changed")
    return value


def _write(path, value):
    with path.open("xb") as handle:
        handle.write(_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _safe(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BaselineError(f"Unsafe path: {relative}")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise BaselineError(f"Symlink path: {path}")
    return path


def _root(path, *, exists=True):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or (exists and not path.is_dir()):
        raise BaselineError(f"Expected a non-symlink directory: {path}")
    return path.resolve()


@contextmanager
def _locked(output, inputs):
    output = _root(output, exists=False)
    for source in inputs:
        source = _root(source)
        if output == source or output in source.parents or source in output.parents:
            raise BaselineError("Output must be separate from both input roots")
    source_root = PROJECT.resolve()
    if output == source_root or output in source_root.parents or source_root in output.parents:
        raise BaselineError("Output must be separate from the source checkout")
    output.mkdir(parents=True, exist_ok=True)
    lock = _safe(output, ".baseline.lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise BaselineError(f"Run is locked: {lock}; inspect the owner, do not auto-delete") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()) + "\n")
        yield output
    finally:
        lock.unlink()


def load_protocol(path=None):
    protocol = json.loads(Path(path or PROTOCOL_PATH).read_text())
    if (protocol.get("schema_version") != 1 or protocol.get("experiment_id") != "clinc-tfidf400-v1"
            or protocol.get("labels") != list(LABELS) or protocol.get("counts") != COUNTS
            or protocol.get("sources") != SOURCE_PINS):
        raise BaselineError("Protocol/source pins differ from the fixed CLINC400 baseline")
    return protocol


def environment_identity():
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "threads": 1,
            "packages": {name: metadata.version(name) for name in PACKAGES}}


def _source_identity():
    names = [f"decision_study/baselines/{name}" for name in ("__init__.py", "__main__.py", "study.py")]
    result = {"files": {name: sha256(PROJECT / name) for name in names}}
    command = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    result["git_commit"] = command.stdout.strip() if command.returncode == 0 else None
    status = subprocess.run(["git", "--no-optional-locks", "-C", str(PROJECT), "status",
                             "--porcelain=v1", "--untracked-files=all"],
                            capture_output=True, text=True, timeout=10)
    result["git_available"] = command.returncode == status.returncode == 0
    result["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    return result


def _require_clean_checkout():
    identity = _source_identity()
    if not identity["git_available"] or identity["git_dirty"]:
        raise BaselineError("Formal baseline CLI requires a clean Git checkout; commit the reviewed recipe first")


def _verified(root, name, expected):
    path = _safe(root, name)
    if not path.is_file():
        raise BaselineError(f"Missing source file: {path}")
    data = path.read_bytes()
    if _hash(data) != expected:
        raise BaselineError(f"Source SHA256 mismatch: {path}")
    return data


def _rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _distribution(rows, gold, banking, oos):
    expected = Counter({label: banking for label in LABELS[:-1]})
    expected["oos"] = oos
    if Counter(gold[row["sample_id"]] for row in rows) != expected:
        raise BaselineError("Label counts differ from frozen banking/OOS quotas")


def _context(v1_dir, v2_dir, protocol_path):
    v1, v2 = _root(v1_dir), _root(v2_dir)
    protocol = load_protocol(protocol_path)
    environment = environment_identity()
    if environment["packages"] != protocol["environment"]:
        raise BaselineError("Package versions differ from the baseline protocol; use its isolated environment")
    pins = protocol["sources"]
    manifest = json.loads(_verified(v1, "data/manifest.json", pins["v1_manifest_sha256"]))
    if (manifest.get("protocol") != "solo-v1"
            or manifest["files"].get("examples.jsonl") != pins["v1_examples_sha256"]
            or manifest["files"].get("gold.json") != pins["v1_gold_sha256"]):
        raise BaselineError("V1 manifest does not bind the expected source files")
    original = _rows(_verified(v1, "data/examples.jsonl", pins["v1_examples_sha256"]))
    selected = [row for row in original if row.get("task") == "clinc" and row.get("split") in COUNTS]
    if Counter(row["split"] for row in selected) != Counter(COUNTS):
        raise BaselineError("CLINC input counts differ from the fixed protocol")
    by_id, normalized = {}, set()
    for row in selected:
        sid = row["sample_id"]
        text = row["state"]
        norm = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if (sid in by_id or norm in normalized or not norm or row.get("labels") != list(LABELS)
                or any(key in row for key in ("gold", "label", "answers", "is_impossible"))):
            raise BaselineError("Duplicate, invalid, or label-leaking CLINC input")
        by_id[sid] = row
        normalized.add(norm)
    v2_rows = _rows(_verified(v2, "data/examples.jsonl", pins["v2_examples_sha256"]))
    fitting = [row for row in v2_rows if row.get("task") == "clinc"]
    if (Counter(row["split"] for row in fitting) != Counter({"calibration": 400, "validation": 400})
            or len({row["sample_id"] for row in fitting}) != 800
            or any(by_id.get(row["sample_id"]) != row for row in fitting)):
        raise BaselineError("V2 fitting inputs are not the exact V1 calibration/validation records")
    fit_gold = json.loads(_verified(v2, "data/fit_gold.json", pins["v2_fit_gold_sha256"]))
    role = {name: sorted((row for row in selected if row["split"] == name), key=lambda r: r["sample_id"])
            for name in COUNTS}
    for name in ("calibration", "validation"):
        _distribution(role[name], fit_gold, 20, 100)
    predict = role["validation"] + role["test"] + role["stress"]
    return {"protocol": protocol, "protocol_sha256": sha256(protocol_path or PROTOCOL_PATH),
            "environment": environment, "source_identity": _source_identity(), "v1": v1,
            "v2": v2, "input_roots": {"v1": str(v1), "v2": str(v2)},
            "roles": role, "fit_gold": fit_gold, "predict": predict}


def _fit_model(train_texts, train_labels, predict_texts, protocol):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    options = dict(protocol["vectorizer"])
    options["ngram_range"] = tuple(options["ngram_range"])
    vectorizer = TfidfVectorizer(**options)
    classifier = LogisticRegression(**protocol["classifier"])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        started = time.perf_counter()
        matrix = vectorizer.fit_transform(train_texts)
        try:
            classifier.fit(matrix, train_labels)
        except ConvergenceWarning as exc:
            raise BaselineError("Classifier did not converge; preserve this failed run") from exc
        fit_seconds = time.perf_counter() - started
        if set(classifier.classes_) != set(LABELS):
            raise BaselineError("Classifier class set differs from the fixed labels")
        started = time.perf_counter()
        probabilities = classifier.predict_proba(vectorizer.transform(predict_texts))
        if probabilities.shape != (len(predict_texts), len(LABELS)):
            raise BaselineError("Classifier probability shape differs from the workload/labels")
        columns = [list(classifier.classes_).index(label) for label in LABELS]
        probabilities = probabilities[:, columns]
        predict_seconds = time.perf_counter() - started
    model = {"kind": "tfidf_logistic_regression", "vocabulary": vectorizer.vocabulary_,
             "idf": vectorizer.idf_.tolist(), "coefficients": classifier.coef_.tolist(),
             "intercept": classifier.intercept_.tolist(), "classes": classifier.classes_.tolist(),
             "prediction_label_order": list(LABELS), "iterations": classifier.n_iter_.tolist()}
    # Serialization is also a finite-value check for all exported fitted parameters.
    _bytes(model)
    if not np.isfinite(probabilities).all():
        raise BaselineError("Non-finite prediction probabilities")
    return model, probabilities.tolist(), {"fit_seconds": fit_seconds,
            "batch_predict_seconds": predict_seconds, "prediction_count": len(predict_texts),
            "scope": protocol["interpretation"]["timing"], "threads": 1}


def _predictions(context, probabilities):
    if len(probabilities) != len(context["predict"]):
        raise BaselineError("Prediction count mismatch")
    result = []
    for row, values in zip(context["predict"], probabilities):
        values = [float(value) for value in values]
        if (len(values) != len(LABELS) or any(not math.isfinite(x) or x < 0 or x > 1 for x in values)
                or abs(sum(values) - 1.0) > 1e-8):
            raise BaselineError("Invalid probability vector")
        result.append({"sample_id": row["sample_id"], "split": row["split"],
                       "predicted_label": LABELS[max(range(len(LABELS)), key=values.__getitem__)],
                       "probabilities": values})
    return result


def _check_freeze(output, context):
    freeze_path = _safe(output, "freeze.json")
    if not freeze_path.is_file():
        raise BaselineError("Missing freeze.json; evaluation never fits a missing model")
    freeze = _unseal(json.loads(freeze_path.read_text()))
    for name in ("protocol_sha256", "environment", "source_identity", "input_roots"):
        if freeze.get(name) != context[name]:
            raise BaselineError(f"Frozen {name} changed; do not refit this run")
    if freeze.get("sources") != context["protocol"]["sources"] or set(freeze.get("artifacts", {})) != set(FIT_FILES):
        raise BaselineError("Frozen sources/artifact inventory changed")
    for name, expected in freeze["artifacts"].items():
        if sha256(_safe(output, name)) != expected:
            raise BaselineError(f"Frozen artifact changed: {name}")
    if json.loads(_safe(output, "protocol.json").read_text()) != context["protocol"]:
        raise BaselineError("Saved protocol differs from the active protocol")
    predictions = _rows(_safe(output, "predictions.jsonl").read_bytes())
    verified = _predictions(context, [row["probabilities"] for row in predictions])
    if predictions != verified:
        raise BaselineError("Prediction IDs/order/labels differ from the frozen workload")
    train_ids = [row["sample_id"] for row in context["roles"]["calibration"]]
    if freeze.get("training_ids") != train_ids:
        raise BaselineError("Frozen training membership changed")
    return freeze, predictions


def fit(v1_dir, v2_dir, output_dir, *, protocol_path=None):
    context = _context(v1_dir, v2_dir, protocol_path)
    with _locked(output_dir, (v1_dir, v2_dir)) as output:
        if _safe(output, "freeze.json").exists():
            return _check_freeze(output, context)[0]
        if any(path.name != ".baseline.lock" for path in output.iterdir()):
            raise BaselineError("Partial or nonempty output without a valid freeze; preserve it and choose a new output")
        training = context["roles"]["calibration"]
        labels = [context["fit_gold"][row["sample_id"]] for row in training]
        model, probabilities, timing = _fit_model([row["state"] for row in training], labels,
                                               [row["state"] for row in context["predict"]], context["protocol"])
        predictions = _predictions(context, probabilities)
        _write(output / "protocol.json", context["protocol"])
        _write(output / "model.json", model)
        _write(output / "fit.json", {"training_count": 400, "training_label_counts": dict(Counter(labels)),
                                    "feature_count": len(model["vocabulary"]), "model_bytes": len(_bytes(model)),
                                    "timing": timing, "interpretation": context["protocol"]["interpretation"]})
        with (output / "predictions.jsonl").open("xb") as handle:
            for row in predictions:
                handle.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        freeze = _seal({"status": "FROZEN", "protocol_sha256": context["protocol_sha256"],
                        "source_identity": context["source_identity"], "environment": context["environment"],
                        "input_roots": context["input_roots"],
                        "sources": context["protocol"]["sources"], "label_order": list(LABELS),
                        "training_ids": [row["sample_id"] for row in training],
                        "artifacts": {name: sha256(output / name) for name in FIT_FILES}})
        _write(output / "freeze.json", freeze)
        return freeze


def _metrics(rows, gold):
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for row in rows:
        matrix[LABELS.index(gold[row["sample_id"]])][LABELS.index(row["predicted_label"]) ] += 1
    support = [sum(row) for row in matrix]
    predicted = [sum(row[i] for row in matrix) for i in range(len(LABELS))]
    f1 = [2 * matrix[i][i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0.0
          for i in range(len(LABELS))]
    tp = matrix[-1][-1]
    return {"count": len(rows), "accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / len(rows),
            "macro_f1": sum(f1) / len(LABELS), "macro_f1_label_count": len(LABELS),
            "banking_accuracy": sum(matrix[i][i] for i in range(len(LABELS) - 1)) / sum(support[:-1]) if sum(support[:-1]) else None,
            "oos": {"precision": tp / predicted[-1] if predicted[-1] else 0.0,
                    "recall": tp / support[-1] if support[-1] else 0.0, "f1": f1[-1]},
            "confusion_matrix": matrix, "label_order": list(LABELS)}


def evaluate(v1_dir, v2_dir, output_dir, *, protocol_path=None):
    context = _context(v1_dir, v2_dir, protocol_path)
    with _locked(output_dir, (v1_dir, v2_dir)) as output:
        freeze, predictions = _check_freeze(output, context)
        freeze_hash = sha256(output / "freeze.json")
        access_path, evaluation_path = _safe(output, "evaluation-access.json"), _safe(output, "evaluation.json")
        expected_gold = context["protocol"]["sources"]["v1_gold_sha256"]
        if evaluation_path.exists():
            access = _unseal(json.loads(access_path.read_text()))
            evaluation = _unseal(json.loads(evaluation_path.read_text()))
            if (access.get("freeze_sha256") != freeze_hash or evaluation.get("freeze_sha256") != freeze_hash
                    or evaluation.get("access_sha256") != sha256(access_path)
                    or evaluation.get("gold_sha256") != expected_gold):
                raise BaselineError("Evaluation identity changed")
            _verified(context["v1"], "data/gold.json", expected_gold)
            return evaluation
        if access_path.exists():
            raise BaselineError("Prior evaluation access without completion; preserve this partial run")
        _write(access_path, _seal({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "freeze_sha256": freeze_hash, "gold_sha256": expected_gold,
                                  "purpose": context["protocol"]["interpretation"]["evaluation"]}))
        # This is deliberately the first V1 gold-file read in the lifecycle.
        gold = json.loads(_verified(context["v1"], "data/gold.json", expected_gold))
        for name in ("calibration", "validation"):
            if any(gold[row["sample_id"]] != context["fit_gold"][row["sample_id"]] for row in context["roles"][name]):
                raise BaselineError("V1 and V2 fitting labels differ")
        _distribution(context["roles"]["test"], gold, 30, 150)
        _distribution(context["roles"]["stress"], gold, 0, 850)
        groups = {name: [row for row in predictions if row["split"] == name] for name in ("validation", "test", "stress")}
        mixed = [row for row in groups["test"] if gold[row["sample_id"]] != "oos"] + groups["stress"]
        metrics = {"validation": _metrics(groups["validation"], gold), "main_test": _metrics(groups["test"], gold),
                   "oos_only_stress": {"count": len(groups["stress"]), "oos_recall": _metrics(groups["stress"], gold)["oos"]["recall"]},
                   "historical_mixed_stress": {**_metrics(mixed, gold), "overlapping_main_banking": 450}}
        evaluation = _seal({"status": "COMPLETE", "freeze_sha256": freeze_hash,
                            "access_sha256": sha256(access_path), "gold_sha256": expected_gold,
                            "metrics": metrics, "interpretation": context["protocol"]["interpretation"]})
        _write(evaluation_path, evaluation)
        return evaluation


def _bundle_unlocked(output):
    names = [*FIT_FILES, "freeze.json", "evaluation-access.json", "evaluation.json"]
    contents = {name: _safe(output, name).read_bytes() for name in names}
    freeze = _unseal(json.loads(contents["freeze.json"]))
    evaluation = _unseal(json.loads(contents["evaluation.json"]))
    access = _unseal(json.loads(contents["evaluation-access.json"]))
    if (set(freeze.get("artifacts", {})) != set(FIT_FILES)
            or any(_hash(contents[name]) != expected for name, expected in freeze["artifacts"].items())
            or evaluation.get("status") != "COMPLETE"
            or evaluation.get("freeze_sha256") != _hash(contents["freeze.json"])
            or access.get("freeze_sha256") != _hash(contents["freeze.json"])
            or evaluation.get("access_sha256") != _hash(contents["evaluation-access.json"])
            or evaluation.get("gold_sha256") != freeze["sources"]["v1_gold_sha256"]):
        raise BaselineError("Cannot export changed or incomplete evaluation artifacts")
    export = _safe(output, "exports/light_baseline_bundle.zip")
    export.parent.mkdir(parents=True, exist_ok=True)
    if export.exists():
        with zipfile.ZipFile(export) as archive:
            if set(archive.namelist()) != set(names) or any(archive.read(name) != data for name, data in contents.items()):
                raise BaselineError("Existing export differs; preserve it")
        return export
    with zipfile.ZipFile(export, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return export


def bundle(output_dir):
    with _locked(_root(output_dir), ()) as output:
        return _bundle_unlocked(output)


def run(v1_dir, v2_dir, output_dir, *, protocol_path=None):
    fit(v1_dir, v2_dir, output_dir, protocol_path=protocol_path)
    evaluation = evaluate(v1_dir, v2_dir, output_dir, protocol_path=protocol_path)
    return {"evaluation": evaluation, "bundle": str(bundle(output_dir))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "fit", "evaluate"))
    parser.add_argument("--v1-dir", required=True, type=Path)
    parser.add_argument("--v2-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _require_clean_checkout()
        result = globals()[args.action](args.v1_dir, args.v2_dir, args.output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (BaselineError, OSError, ValueError, KeyError, TypeError, metadata.PackageNotFoundError,
            subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0
