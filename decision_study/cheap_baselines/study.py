"""Freeze three fixed classifiers before any retrospective test-label scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings
import zipfile

from . import data, encoder

PROJECT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT / "configs/clinc_cheap_protocol.json"
RECIPES = ("tfidf1600", "minilm400", "minilm1600")
LABELS = ("account_blocked", "balance", "bill_balance", "bill_due", "freeze_account",
          "interest_rate", "min_payment", "order_checks", "pay_bill", "pin_change",
          "report_fraud", "routing", "spending_history", "transactions", "transfer", "oos")
FIT_FILES = ("protocol.json", "embeddings.npy", "embeddings.json", *(
    f"{recipe}.{suffix}" for recipe in RECIPES
    for suffix in ("model.json", "predictions.jsonl", "fit.json")))


class BaselineError(RuntimeError):
    pass


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def sha256(path):
    return _hash(Path(path).read_bytes())


def _seal(value):
    return {**value, "self_hash": _hash(_bytes(value))}


def _unseal(value):
    if value.get("self_hash") != _hash(_bytes({k: v for k, v in value.items() if k != "self_hash"})):
        raise BaselineError("Changed artifact self hash")
    return value


def _write(path, value):
    _write_bytes(path, _bytes(value))


def _write_bytes(path, value):
    if path.is_symlink():
        raise BaselineError(f"Symlink output: {path}")
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _safe(root, name):
    path = root / name
    if Path(name).name != name or path.is_symlink():
        raise BaselineError(f"Unsafe artifact path: {path}")
    return path


def _source_identity():
    names = [f"decision_study/cheap_baselines/{name}" for name in
             ("__init__.py", "__main__.py", "data.py", "encoder.py", "study.py")]
    names += ["configs/cheap_baseline_requirements.txt", "scripts/run_cheap_baselines.sh",
              "scripts/run_cheap_baseline_suite.sh"]
    head = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=10)
    status = subprocess.run(["git", "--no-optional-locks", "-C", str(PROJECT), "status",
                             "--porcelain=v1", "--untracked-files=all"],
                            capture_output=True, text=True, timeout=10)
    return {"files": {name: sha256(PROJECT / name) for name in names},
            "git_commit": head.stdout.strip() if head.returncode == 0 else None,
            "git_available": head.returncode == status.returncode == 0,
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}


def environment_identity(protocol):
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "processor": platform.processor(),
            "threads": 1, "packages": {name: metadata.version(name) for name in protocol["environment"]}}


def load_protocol(path=None):
    protocol = json.loads(Path(path or PROTOCOL_PATH).read_text())
    if (protocol.get("schema_version") != 1 or protocol.get("experiment_id") != "clinc-cheap-v1"
            or protocol.get("recipes") != list(RECIPES) or protocol.get("labels") != list(LABELS)):
        raise BaselineError("Not the fixed CLINC cheap-baseline protocol")
    return protocol


def _context(v1_dir, v2_dir, prepared_dir, model_dir, protocol_path=None):
    protocol = load_protocol(protocol_path)
    context = data.load(prepared_dir, v1_dir, v2_dir)
    if context["prepared_hashes"] != protocol["prepared"]:
        raise BaselineError("Prepared files differ from the released protocol")
    if context["sources"] != protocol["sources"]:
        raise BaselineError("Input sources differ from the released protocol")
    environment = environment_identity(protocol)
    if environment["packages"] != protocol["environment"]:
        raise BaselineError("Use the isolated pinned cheap-baseline environment")
    model_dir = Path(model_dir).expanduser().absolute()
    if model_dir.is_symlink():
        raise BaselineError("Model cache root cannot be a symlink")
    model_dir = model_dir.resolve()
    context.update(protocol=protocol, protocol_sha256=sha256(protocol_path or PROTOCOL_PATH),
                   environment=environment, source_identity=_source_identity(),
                   model_identity=encoder.verify(model_dir, protocol), model_dir=model_dir)
    context["input_roots"] = {**context["input_roots"], "model": str(model_dir)}
    return context


@contextmanager
def _locked(output_dir, inputs):
    output = Path(output_dir).expanduser().absolute()
    if output.is_symlink():
        raise BaselineError("Output cannot be a symlink")
    output = output.resolve()
    for value in [PROJECT, *inputs]:
        source = Path(value).expanduser().resolve()
        if source == output or source in output.parents or output in source.parents:
            raise BaselineError("Output must be separate from source, inputs and model cache")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".baseline.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise BaselineError(f"Output is locked: {lock}; inspect its owner") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()) + "\n")
        yield output
    finally:
        lock.unlink()


def _classifier(train_matrix, labels, prediction_matrix, protocol):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    classifier = LogisticRegression(**protocol["classifier"])
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        started = time.perf_counter()
        try:
            classifier.fit(train_matrix, labels)
        except ConvergenceWarning as exc:
            raise BaselineError("Classifier did not converge; preserve this run") from exc
        fit_seconds = time.perf_counter() - started
        classes = list(classifier.classes_)
        if set(classes) != set(LABELS):
            raise BaselineError("Classifier class set changed")
        started = time.perf_counter()
        probabilities = classifier.predict_proba(prediction_matrix)
        probabilities = probabilities[:, [classes.index(label) for label in LABELS]]
        predict_seconds = time.perf_counter() - started
    if not np.isfinite(probabilities).all():
        raise BaselineError("Non-finite classifier probabilities")
    model = {"coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
             "classes": classes, "prediction_label_order": list(LABELS),
             "iterations": classifier.n_iter_.tolist()}
    _bytes(model)
    return model, probabilities.tolist(), {"fit_seconds": fit_seconds,
                                           "batch_predict_seconds": predict_seconds}


def _fit_tfidf(train_texts, train_labels, predict_texts, protocol):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from threadpoolctl import threadpool_limits
    options = dict(protocol["vectorizer"])
    options["ngram_range"] = tuple(options["ngram_range"])
    vectorizer = TfidfVectorizer(**options)
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        train = vectorizer.fit_transform(train_texts)
        feature_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predict = vectorizer.transform(predict_texts)
        feature_predict_seconds = time.perf_counter() - started
        model, probabilities, timing = _classifier(train, train_labels, predict, protocol)
    model.update(kind="tfidf_logistic_regression", vocabulary=vectorizer.vocabulary_,
                 idf=vectorizer.idf_.tolist())
    timing.update(feature_fit_seconds=feature_fit_seconds, feature_predict_seconds=feature_predict_seconds)
    return model, probabilities, timing


def _fit_dense(train_matrix, labels, predict_matrix, protocol):
    # Fixed float64 LR input; the frozen encoder artifact remains float32.
    import numpy as np
    model, probabilities, timing = _classifier(np.asarray(train_matrix, dtype=np.float64), labels,
                                               np.asarray(predict_matrix, dtype=np.float64), protocol)
    model.update(kind="frozen_minilm_logistic_regression")
    return model, probabilities, timing


def _predictions(context, probabilities):
    if len(probabilities) != len(context["predict"]):
        raise BaselineError("Prediction count differs from workload")
    rows = []
    for example, probability in zip(context["predict"], probabilities):
        values = [float(x) for x in probability]
        if (len(values) != len(LABELS) or any(not math.isfinite(x) or x < 0 or x > 1 for x in values)
                or abs(sum(values) - 1) > 1e-8):
            raise BaselineError("Invalid probability vector")
        rows.append({"sample_id": example["sample_id"], "split": example["split"],
                     "probabilities": values,
                     "predicted_label": LABELS[max(range(len(LABELS)), key=values.__getitem__)]})
    return rows


def _check_freeze(output, context):
    import numpy as np
    freeze = _unseal(json.loads(_safe(output, "freeze.json").read_bytes()))
    for name in ("protocol_sha256", "source_identity", "environment", "input_roots", "model_identity", "sources", "prepared_hashes"):
        if freeze.get(name) != context[name]:
            raise BaselineError(f"Frozen {name} changed")
    if freeze.get("status") != "FROZEN" or set(freeze.get("artifacts", {})) != set(FIT_FILES):
        raise BaselineError("Incomplete three-recipe freeze")
    for name, expected in freeze["artifacts"].items():
        if sha256(_safe(output, name)) != expected:
            raise BaselineError(f"Changed frozen artifact: {name}")
    if json.loads(_safe(output, "protocol.json").read_bytes()) != context["protocol"]:
        raise BaselineError("Saved protocol changed")
    all_rows = context["training"]["1600"] + context["predict"]
    feature_meta = json.loads(_safe(output, "embeddings.json").read_bytes())
    expected_input = [{"sample_id": row["sample_id"], "text_sha256": _hash(row["state"].encode())} for row in all_rows]
    if (feature_meta["inputs"] != expected_input or feature_meta["model_identity"] != context["model_identity"]
            or feature_meta["matrix_sha256"] != sha256(output / "embeddings.npy")):
        raise BaselineError("Encoder feature identity changed")
    matrix = np.load(output / "embeddings.npy", allow_pickle=False)
    _check_matrix(matrix, len(all_rows), context["protocol"])
    result = {}
    for recipe in RECIPES:
        train = context["training"]["400" if recipe == "minilm400" else "1600"]
        if freeze["training_ids"].get(recipe) != [row["sample_id"] for row in train]:
            raise BaselineError(f"Training membership changed: {recipe}")
        rows = [json.loads(line) for line in (output / f"{recipe}.predictions.jsonl").read_bytes().splitlines()]
        if rows != _predictions(context, [row["probabilities"] for row in rows]):
            raise BaselineError(f"Prediction ID/order/class mismatch: {recipe}")
        result[recipe] = rows
    return freeze, result


def _check_matrix(matrix, count, protocol):
    import numpy as np
    if (matrix.shape != (count, protocol["encoder"]["embedding_dimension"]) or matrix.dtype != np.float32
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=2e-5, rtol=0)):
        raise BaselineError("Embedding shape, dtype, finite values or normalization invalid")


def fit(v1_dir, v2_dir, prepared_dir, model_dir, output_dir, *, protocol_path=None):
    import numpy as np
    context = _context(v1_dir, v2_dir, prepared_dir, model_dir, protocol_path)
    with _locked(output_dir, (v1_dir, v2_dir, prepared_dir, model_dir)) as output:
        if (output / "freeze.json").exists():
            return _check_freeze(output, context)[0]
        if any(path.name != ".baseline.lock" for path in output.iterdir()):
            raise BaselineError("Partial output without suite freeze; preserve it and choose a new output")
        protocol = context["protocol"]
        _write(output / "protocol.json", protocol)
        all_rows = context["training"]["1600"] + context["predict"]
        matrix, feature_timing = encoder.encode([r["state"] for r in all_rows], context["model_dir"], protocol)
        _check_matrix(matrix, len(all_rows), protocol)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        _write_bytes(output / "embeddings.npy", buffer.getvalue())
        feature_meta = {"inputs": [{"sample_id": row["sample_id"], "text_sha256": _hash(row["state"].encode())} for row in all_rows],
                        "model_identity": context["model_identity"], "matrix_sha256": sha256(output / "embeddings.npy"),
                        "encoding": feature_timing,
                        "scope": "Shared frozen encoder for 1600 train + 1850 evaluation inputs; charge this work explicitly, not zero encoder cost per classifier."}
        _write(output / "embeddings.json", feature_meta)
        lookup = {row["sample_id"]: i for i, row in enumerate(all_rows)}
        ntrain = len(context["training"]["1600"])
        training_ids = {}
        for recipe in RECIPES:
            train = context["training"]["400" if recipe == "minilm400" else "1600"]
            labels = [r["label"] for r in train]
            if recipe == "tfidf1600":
                model, probabilities, timing = _fit_tfidf([r["state"] for r in train], labels,
                                                          [r["state"] for r in context["predict"]], protocol)
            else:
                model, probabilities, timing = _fit_dense(matrix[[lookup[r["sample_id"]] for r in train]], labels,
                                                          matrix[ntrain:], protocol)
                model["embeddings_sha256"] = feature_meta["matrix_sha256"]
                model["encoder"] = protocol["encoder"]
            predictions = _predictions(context, probabilities)
            _write(output / f"{recipe}.model.json", model)
            _write_bytes(output / f"{recipe}.predictions.jsonl", b"".join(
                (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode() for row in predictions))
            _write(output / f"{recipe}.fit.json", {"recipe": recipe, "training_count": len(train),
                   "training_label_counts": dict(Counter(labels)), "prediction_count": len(predictions),
                   "timing": timing, "shared_encoder_metadata": "embeddings.json" if recipe.startswith("minilm") else None,
                   "model_bytes": len(_bytes(model)), "interpretation": protocol["interpretation"]})
            training_ids[recipe] = [row["sample_id"] for row in train]
        freeze = _seal({"status": "FROZEN", **{name: context[name] for name in
                        ("protocol_sha256", "source_identity", "environment", "input_roots", "model_identity", "sources", "prepared_hashes")},
                        "training_ids": training_ids, "artifacts": {name: sha256(output / name) for name in FIT_FILES}})
        _write(output / "freeze.json", freeze)
        return _check_freeze(output, context)[0]


def _metrics(rows, gold):
    pairs = Counter((gold[r["sample_id"]], r["predicted_label"]) for r in rows)
    support = Counter(gold[r["sample_id"]] for r in rows)
    predicted = Counter(r["predicted_label"] for r in rows)
    f1 = {label: 2 * pairs[label, label] / (support[label] + predicted[label])
          if support[label] + predicted[label] else 0. for label in LABELS}
    bank_count = sum(support[l] for l in LABELS[:-1])
    return {"count": len(rows), "accuracy": sum(pairs[l, l] for l in LABELS) / len(rows),
            "macro_f1": sum(f1.values()) / len(LABELS), "macro_f1_label_count": len(LABELS),
            "banking_accuracy": sum(pairs[l, l] for l in LABELS[:-1]) / bank_count if bank_count else None,
            "oos": {"precision": pairs["oos", "oos"] / predicted["oos"] if predicted["oos"] else 0.,
                    "recall": pairs["oos", "oos"] / support["oos"] if support["oos"] else 0., "f1": f1["oos"]},
            "label_order": list(LABELS), "confusion_matrix": [[pairs[a, b] for b in LABELS] for a in LABELS]}


def _verify_evaluation(output, context):
    access = _unseal(json.loads(_safe(output, "evaluation-access.json").read_bytes()))
    evaluation = _unseal(json.loads(_safe(output, "evaluation.json").read_bytes()))
    for value in (access, evaluation):
        if value["freeze_sha256"] != sha256(output / "freeze.json") or value["gold_sha256"] != context["sources"]["v1_gold_sha256"]:
            raise BaselineError("Evaluation source identity changed")
    if (evaluation["status"] != "COMPLETE" or evaluation["access_sha256"] != sha256(output / "evaluation-access.json")
            or set(evaluation["metrics"]) != set(RECIPES)):
        raise BaselineError("Incomplete or changed evaluation")
    return evaluation


def evaluate(v1_dir, v2_dir, prepared_dir, model_dir, output_dir, *, protocol_path=None):
    context = _context(v1_dir, v2_dir, prepared_dir, model_dir, protocol_path)
    with _locked(output_dir, (v1_dir, v2_dir, prepared_dir, model_dir)) as output:
        freeze, predictions = _check_freeze(output, context)
        access_path, evaluation_path = output / "evaluation-access.json", output / "evaluation.json"
        if evaluation_path.exists():
            result = _verify_evaluation(output, context)
            if sha256(Path(v1_dir) / "data/gold.json") != context["sources"]["v1_gold_sha256"]:
                raise BaselineError("Gold changed")
            return result
        if access_path.exists():
            raise BaselineError("Prior access without completed evaluation; preserve the partial run")
        _write(access_path, _seal({"started_at_utc": datetime.now(timezone.utc).isoformat(),
               "freeze_sha256": sha256(output / "freeze.json"), "gold_sha256": context["sources"]["v1_gold_sha256"],
               "purpose": context["protocol"]["interpretation"]["evaluation"]}))
        # First test-label file access in the fit/evaluate lifecycle; prepare is a separate declared phase.
        raw = (Path(v1_dir) / "data/gold.json").read_bytes()
        if _hash(raw) != context["sources"]["v1_gold_sha256"]:
            raise BaselineError("Gold changed")
        gold = json.loads(raw)
        for split, bank, oos in (("calibration", 20, 100), ("validation", 20, 100), ("test", 30, 150), ("stress", 0, 850)):
            rows = context["roles"][split]
            if Counter(gold[r["sample_id"]] for r in rows) != Counter({**{l: bank for l in LABELS[:-1]}, "oos": oos}):
                raise BaselineError("Gold class quotas changed")
        if any(gold[r["sample_id"]] != r["label"] for r in context["training"]["400"]):
            raise BaselineError("Fitting label alignment changed")
        metrics = {}
        for recipe, rows in predictions.items():
            groups = {split: [r for r in rows if r["split"] == split] for split in ("validation", "test", "stress")}
            mixed = [r for r in groups["test"] if gold[r["sample_id"]] != "oos"] + groups["stress"]
            metrics[recipe] = {"validation": _metrics(groups["validation"], gold), "main_test": _metrics(groups["test"], gold),
                               "oos_only_stress": {"count": 850, "oos_recall": _metrics(groups["stress"], gold)["oos"]["recall"]},
                               "historical_mixed_stress": {**_metrics(mixed, gold), "overlapping_main_banking": 450}}
        evaluation = _seal({"status": "COMPLETE", "freeze_sha256": sha256(output / "freeze.json"),
                           "access_sha256": sha256(access_path), "gold_sha256": context["sources"]["v1_gold_sha256"],
                           "metrics": metrics, "interpretation": context["protocol"]["interpretation"]})
        _write(evaluation_path, evaluation)
        return evaluation


def bundle(output_dir):
    with _locked(output_dir, ()) as output:
        names = [*FIT_FILES, "freeze.json", "evaluation-access.json", "evaluation.json"]
        contents = {name: _safe(output, name).read_bytes() for name in names}
        freeze = _unseal(json.loads(contents["freeze.json"]))
        if set(freeze["artifacts"]) != set(FIT_FILES) or any(_hash(contents[n]) != h for n, h in freeze["artifacts"].items()):
            raise BaselineError("Cannot export incomplete or changed fitted artifacts")
        _verify_evaluation(output, freeze)
        exports = output / "exports"
        if exports.is_symlink():
            raise BaselineError("Symlink export directory")
        exports.mkdir(exist_ok=True)
        path = exports / "cheap_baseline_bundle.zip"
        if path.is_symlink():
            raise BaselineError("Symlink export")
        if path.exists():
            with zipfile.ZipFile(path) as archive:
                if len(archive.namelist()) != len(names) or set(archive.namelist()) != set(names) or any(archive.read(n) != b for n, b in contents.items()):
                    raise BaselineError("Existing export differs; preserve it")
        else:
            with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in contents.items():
                    archive.writestr(name, value)
        return path


def run(v1_dir, v2_dir, prepared_dir, model_dir, output_dir, *, protocol_path=None):
    fit(v1_dir, v2_dir, prepared_dir, model_dir, output_dir, protocol_path=protocol_path)
    result = evaluate(v1_dir, v2_dir, prepared_dir, model_dir, output_dir, protocol_path=protocol_path)
    return {"evaluation": result, "bundle": str(bundle(output_dir))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("raw-dir", "v1-dir", "v2-dir", "prepared-dir"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    prepare_parser.add_argument("--download", action="store_true")
    download_parser = sub.add_parser("download-model")
    download_parser.add_argument("--model-dir", required=True, type=Path)
    for action in ("preflight", "fit", "evaluate", "run"):
        action_parser = sub.add_parser(action)
        for name in ("v1-dir", "v2-dir", "prepared-dir", "model-dir"):
            action_parser.add_argument(f"--{name}", type=Path, required=True)
        if action != "preflight":
            action_parser.add_argument("--output-dir", type=Path, required=True)
    bundle_parser = sub.add_parser("bundle")
    bundle_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        identity = _source_identity()
        if not identity["git_available"] or identity["git_dirty"]:
            raise BaselineError("Requires a clean committed checkout")
        if args.action == "prepare":
            if args.download:
                data.download(args.raw_dir)
            result = data.prepare(args.raw_dir, args.v1_dir, args.v2_dir, args.prepared_dir)
        elif args.action == "download-model":
            result = encoder.download(args.model_dir, load_protocol())
        elif args.action == "bundle":
            result = {"bundle": str(bundle(args.output_dir))}
        elif args.action == "preflight":
            context = _context(args.v1_dir, args.v2_dir, args.prepared_dir, args.model_dir)
            result = {"status": "READY", "training_counts": {k: len(v) for k, v in context["training"].items()},
                      "predictions_per_recipe": len(context["predict"]), "model_identity": context["model_identity"]}
        else:
            result = globals()[args.action](args.v1_dir, args.v2_dir, args.prepared_dir, args.model_dir, args.output_dir)
        if args.action == "fit":
            result = {"status": "FROZEN", "recipes": list(RECIPES), "output_dir": str(args.output_dir)}
        elif args.action in ("evaluate", "run"):
            evaluation = result["evaluation"] if args.action == "run" else result
            result = {"status": evaluation["status"],
                      "main_test": {name: {k: metric["main_test"][k] for k in ("count", "accuracy", "macro_f1", "banking_accuracy")}
                                    for name, metric in evaluation["metrics"].items()},
                      **({"bundle": result["bundle"]} if args.action == "run" else {})}
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    except (BaselineError, data.DataError, encoder.EncoderError, OSError, ValueError, KeyError, TypeError,
            metadata.PackageNotFoundError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0
