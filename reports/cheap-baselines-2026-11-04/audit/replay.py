"""Read-only mathematical replay of three saved classifier parameter sets.

No production project imports, fitting, test-label reads or ONNX session execution.
The tokenizer pass verifies only stored token counts. Saved features remain intact.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import unicodedata
import zipfile
from importlib import metadata
import numpy as np
from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
for name in ("bundle", "repo", "artifact-root", "prepared-dir", "output"):
    parser.add_argument(f"--{name}", type=Path, required=True)
parser.add_argument("--tokenizer", type=Path)
args = parser.parse_args()
tokenizer_path = args.tokenizer or args.repo / ".validation/cheap-baselines-2026-11-04/model/tokenizer.json"
COMMIT = "1b70899bbcf15ee686df04b36e0c333a2fc21e5c"
EXPECTED_ARCHIVE = "7f00178c7bd1d86eeeb121929131447675d0053c10072f687f88d7e3ae77b2be"
RECIPES = ("tfidf1600", "minilm400", "minilm1600")
LABELS = ["account_blocked", "balance", "bill_balance", "bill_due", "freeze_account", "interest_rate",
          "min_payment", "order_checks", "pay_bill", "pin_change", "report_fraud", "routing",
          "spending_history", "transactions", "transfer", "oos"]
sha = lambda raw: hashlib.sha256(raw).hexdigest()
canonical = lambda value: (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
rows = lambda raw: [json.loads(line) for line in raw.splitlines() if line.strip()]
norm = lambda text: " ".join(unicodedata.normalize("NFKC", text).casefold().split())
assert sha(args.bundle.read_bytes()) == EXPECTED_ARCHIVE
fit_names = {"protocol.json", "embeddings.npy", "embeddings.json"} | {
    f"{recipe}.{suffix}" for recipe in RECIPES for suffix in ("model.json", "predictions.jsonl", "fit.json")}
with zipfile.ZipFile(args.bundle) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)) == 15
    assert set(names) == fit_names | {"freeze.json", "evaluation-access.json", "evaluation.json"}
    assert sum(i.file_size for i in archive.infolist()) < 20_000_000
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        assert not path.is_absolute() and ".." not in path.parts and not stat.S_ISLNK(item.external_attr >> 16)
    assert archive.testzip() is None
    raw = {name: archive.read(name) for name in names}
documents = {name: json.loads(value) for name, value in raw.items() if name.endswith(".json")}
freeze, access, evaluation = [documents[n] for n in ("freeze.json", "evaluation-access.json", "evaluation.json")]
for value in (freeze, access, evaluation):
    assert value["self_hash"] == sha(canonical({k: v for k, v in value.items() if k != "self_hash"}))
assert freeze["status"] == "FROZEN" and evaluation["status"] == "COMPLETE"
assert set(freeze["artifacts"]) == fit_names
for name, expected in freeze["artifacts"].items():
    assert sha(raw[name]) == expected, name
assert access["freeze_sha256"] == evaluation["freeze_sha256"] == sha(raw["freeze.json"])
assert evaluation["access_sha256"] == sha(raw["evaluation-access.json"])
assert access["gold_sha256"] == evaluation["gold_sha256"] == freeze["sources"]["v1_gold_sha256"]
assert freeze["source_identity"]["git_commit"] == COMMIT
assert freeze["source_identity"]["git_available"] is True and freeze["source_identity"]["git_dirty"] is False
for name, expected in freeze["source_identity"]["files"].items():
    saved = subprocess.check_output(["git", "-C", str(args.repo), "show", f"{COMMIT}:{name}"])
    assert sha(saved) == expected, name
protocol_raw = subprocess.check_output(["git", "-C", str(args.repo), "show", f"{COMMIT}:configs/clinc_cheap_protocol.json"])
protocol = documents["protocol.json"]
assert sha(protocol_raw) == freeze["protocol_sha256"] and json.loads(protocol_raw) == protocol
assert protocol["recipes"] == list(RECIPES) and protocol["labels"] == LABELS
assert freeze["environment"]["packages"] == protocol["environment"]
assert freeze["sources"] == protocol["sources"] and freeze["prepared_hashes"] == protocol["prepared"]

v1 = args.artifact_root / "results/returned-experiment-2026-11-02-101054"
v2 = args.artifact_root / "results/solo-v2-prepared"
source_paths = {"v1_examples_sha256": v1 / "data/examples.jsonl", "v1_manifest_sha256": v1 / "data/manifest.json",
                "v2_examples_sha256": v2 / "data/examples.jsonl", "v2_fit_gold_sha256": v2 / "data/fit_gold.json"}
sources = {key: path.read_bytes() for key, path in source_paths.items()}
for key, value in sources.items():
    assert sha(value) == protocol["sources"][key]
prepared = {name: (args.prepared_dir / name).read_bytes() for name in protocol["prepared"]}
for name, value in prepared.items():
    assert sha(value) == protocol["prepared"][name]
training = rows(prepared["train1600.jsonl"])
manifest = json.loads(prepared["manifest.json"])
assert manifest["files"]["train1600.jsonl"] == sha(prepared["train1600.jsonl"])
assert manifest["project_sources"] == protocol["sources"]
assert len(training) == 1600 and Counter(r["label"] for r in training) == Counter({label: 100 for label in LABELS})
assert training == sorted(training, key=lambda row: row["sample_id"])
examples = rows(sources["v1_examples_sha256"])
roles = {split: sorted((r for r in examples if r["task"] == "clinc" and r["split"] == split), key=lambda r: r["sample_id"])
         for split in ("calibration", "validation", "test", "stress")}
assert {name: len(group) for name, group in roles.items()} == {"calibration": 400, "validation": 400, "test": 600, "stress": 850}
prediction_inputs = roles["validation"] + roles["test"] + roles["stress"]
all_inputs = training + prediction_inputs
assert len({r["sample_id"] for r in all_inputs}) == 3450
assert len({norm(r["state"]) for r in all_inputs}) == 3450
train_lookup = {r["sample_id"]: r for r in training}
cal_ids = [r["sample_id"] for r in roles["calibration"]]
assert len(set(cal_ids)) == 400 and set(cal_ids) <= train_lookup.keys()
fit_gold = json.loads(sources["v2_fit_gold_sha256"])
for row in roles["calibration"]:
    assert row["state"] == train_lookup[row["sample_id"]]["state"]
    assert fit_gold[row["sample_id"]] == train_lookup[row["sample_id"]]["label"]
assert set(freeze["training_ids"]) == set(RECIPES)
for recipe in RECIPES:
    expected = cal_ids if recipe == "minilm400" else [r["sample_id"] for r in training]
    assert freeze["training_ids"][recipe] == expected

feature = documents["embeddings.json"]
assert feature["inputs"] == [{"sample_id": r["sample_id"], "text_sha256": sha(r["state"].encode())} for r in all_inputs]
assert feature["matrix_sha256"] == sha(raw["embeddings.npy"])
assert feature["model_identity"] == freeze["model_identity"] == feature["encoding"]["identity"]
assert feature["model_identity"]["files"] == protocol["encoder"]["files"]
matrix = np.load(io.BytesIO(raw["embeddings.npy"]), allow_pickle=False)
assert matrix.shape == (3450, 384) and matrix.dtype == np.float32 and np.isfinite(matrix).all()
norm_error = float(np.abs(np.linalg.norm(matrix, axis=1) - 1.).max())
assert norm_error <= 2e-5
tokenizer_bytes = tokenizer_path.read_bytes()
assert sha(tokenizer_bytes) == protocol["encoder"]["files"]["tokenizer.json"]["sha256"]
assert len(tokenizer_bytes) == protocol["encoder"]["files"]["tokenizer.json"]["bytes"]
assert metadata.version("tokenizers") == protocol["environment"]["tokenizers"]
tokenizer = Tokenizer.from_file(str(tokenizer_path))
tokenizer.no_truncation()
tokenizer.no_padding()
encoded = tokenizer.encode_batch([r["state"] for r in all_inputs], add_special_tokens=True)
lengths = [len(row.ids) for row in encoded]
assert lengths == feature["encoding"]["token_lengths"]
assert max(lengths) == feature["encoding"]["max_tokens_including_special"] == 25
assert all(0 < n <= protocol["encoder"]["max_length"] for n in lengths)
assert not any(row.overflowing for row in encoded)
assert feature["encoding"]["truncated_count"] == 0
assert feature["encoding"]["count"] == 3450 and feature["encoding"]["dimension"] == 384
assert feature["encoding"]["dtype"] == "float32" and feature["encoding"]["batch_size"] == 32
assert feature["encoding"]["threads"] == 1

model_results = {}
for recipe in RECIPES:
    model = documents[f"{recipe}.model.json"]
    fit = documents[f"{recipe}.fit.json"]
    predictions = rows(raw[f"{recipe}.predictions.jsonl"])
    assert [r["sample_id"] for r in predictions] == [r["sample_id"] for r in prediction_inputs]
    assert [r["split"] for r in predictions] == [r["split"] for r in prediction_inputs]
    assert len(predictions) == len({r["sample_id"] for r in predictions}) == 1850
    train_ids = freeze["training_ids"][recipe]
    assert fit["training_count"] == len(train_ids) and fit["prediction_count"] == 1850
    assert fit["training_label_counts"] == Counter(train_lookup[sid]["label"] for sid in train_ids)
    assert fit["model_bytes"] == len(raw[f"{recipe}.model.json"])
    classes = model["classes"]
    assert len(classes) == 16 and set(classes) == set(LABELS) and model["prediction_label_order"] == LABELS
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    assert intercept.shape == (16,) and np.isfinite(intercept).all() and np.isfinite(coefficients).all()
    assert model["iterations"] and all(0 < n < protocol["classifier"]["max_iter"] for n in model["iterations"])
    detail = {}
    if recipe == "tfidf1600":
        assert model["kind"] == "tfidf_logistic_regression"
        assert protocol["vectorizer"] == {"analyzer": "word", "ngram_range": [1, 2], "lowercase": True,
            "strip_accents": None, "token_pattern": r"(?u)\b\w\w+\b", "stop_words": None,
            "min_df": 1, "max_df": 1.0, "max_features": None, "binary": False, "norm": "l2",
            "use_idf": True, "smooth_idf": True, "sublinear_tf": False}
        tokenize = re.compile(protocol["vectorizer"]["token_pattern"]).findall
        def terms(text):
            tokens = tokenize(text.lower())
            return tokens + [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        frequencies = Counter()
        for row in training:
            frequencies.update(set(terms(row["state"])))
        vocabulary = model["vocabulary"]
        assert vocabulary == {term: index for index, term in enumerate(sorted(frequencies))}
        idf = np.asarray(model["idf"], dtype=np.float64)
        expected_idf = np.log((1. + len(training)) / (np.asarray([frequencies[t] for t in sorted(frequencies)]) + 1.)) + 1.
        assert idf.shape == expected_idf.shape and np.isfinite(idf).all()
        idf_error = float(np.max(np.abs(idf - expected_idf)))
        assert idf_error <= 1e-14
        features = np.zeros((1850, len(vocabulary)), dtype=np.float64)
        for index, row in enumerate(prediction_inputs):
            for term, count in Counter(terms(row["state"])).items():
                if term in vocabulary:
                    column = vocabulary[term]
                    features[index, column] = count * idf[column]
            length = float(np.linalg.norm(features[index]))
            if length:
                features[index] /= length
        detail = {"vocabulary_exact_training_match": True, "idf_max_absolute_error": idf_error,
                  "zero_feature_rows": int((np.linalg.norm(features, axis=1) == 0).sum())}
    else:
        assert model["kind"] == "frozen_minilm_logistic_regression"
        assert model["embeddings_sha256"] == sha(raw["embeddings.npy"])
        assert model["encoder"] == protocol["encoder"]
        features = np.asarray(matrix[1600:], dtype=np.float64)
    assert coefficients.shape == (16, features.shape[1])
    logits = features @ coefficients.T + intercept
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    remap = [classes.index(label) for label in LABELS]
    probabilities = probabilities[:, remap]
    saved = np.asarray([r["probabilities"] for r in predictions], dtype=np.float64)
    assert saved.shape == (1850, 16) and np.isfinite(saved).all() and (saved >= 0).all() and (saved <= 1).all()
    assert np.allclose(saved.sum(axis=1), 1., rtol=0, atol=1e-12)
    error = np.abs(probabilities - saved)
    assert np.allclose(probabilities, saved, rtol=0, atol=1e-12)
    predicted_labels = [LABELS[index] for index in probabilities.argmax(axis=1)]
    label_matches = sum(a == b["predicted_label"] for a, b in zip(predicted_labels, predictions))
    assert label_matches == 1850
    model_results[recipe] = {"prediction_count": 1850, "label_matches": label_matches,
        "max_probability_absolute_error": float(error.max()), "mean_probability_absolute_error": float(error.mean()),
        "feature_count": features.shape[1], "coefficient_shape": list(coefficients.shape),
        "iterations": model["iterations"], "training_count": len(train_ids), "class_column_remap": remap, **detail}

result = {"status": "PASS", "audit_kind": "Independent saved-parameter replay; no fitting, ONNX execution or test-label reads",
    "archive_sha256": EXPECTED_ARCHIVE, "source_commit": COMMIT, "source_hash_chain_verified": True,
    "source_protocol_sha256": sha(protocol_raw), "exported_protocol_sha256": sha(raw["protocol.json"]),
    "protocol_content_identical_despite_serialization": True, "prepared_hashes": freeze["prepared_hashes"],
    "embedding_shape": list(matrix.shape), "embedding_dtype": str(matrix.dtype),
    "embedding_max_unit_norm_error": norm_error, "feature_ids_text_hashes_order_verified": True,
    "full_training_count": 1600, "calibration_exact_subset_count": 400, "all_feature_ids_and_normalized_inputs_unique": True,
    "token_lengths_exactly_match": True, "token_count": len(lengths), "max_tokens_including_special": max(lengths),
    "truncated_count": 0, "tokenizer_sha256": sha(tokenizer_bytes), "models": model_results,
    "total_labels_replayed": sum(r["label_matches"] for r in model_results.values()), "probability_atol": 1e-12,
    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__, "tokenizers": metadata.version("tokenizers")},
    "limits": ["Uses archived embedding values; encoder hidden-state provenance needs separate replay.",
               "Does not rerun or prove classifier optimization history; checks the declared saved-parameter prediction mapping.",
               "Integrity hashes are not signatures or forensic proof of historical label reads.",
               "No elapsed-time remeasurement, model selection or new task-quality evaluation is performed."]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_bytes(canonical(result))
print(json.dumps({"status": result["status"], "labels": result["total_labels_replayed"], "models": model_results}, indent=2))
