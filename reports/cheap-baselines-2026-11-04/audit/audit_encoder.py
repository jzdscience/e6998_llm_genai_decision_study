"""Independently replay a frozen CPU encoder; never fit or read scoring gold."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import zipfile

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode()


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "repo", "prepared-dir", "examples", "model-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    commit = "1b70899bbcf15ee686df04b36e0c333a2fc21e5c"
    bundle_hash = "7f00178c7bd1d86eeeb121929131447675d0053c10072f687f88d7e3ae77b2be"
    assert sha(args.bundle.read_bytes()) == bundle_hash
    with zipfile.ZipFile(args.bundle) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == 15
        assert sum(i.file_size for i in archive.infolist()) < 30_000_000
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
        assert archive.testzip() is None
        raw = {name: archive.read(name) for name in names}
    docs = {name: json.loads(value) for name, value in raw.items() if name.endswith(".json")}
    freeze, access, evaluation = [docs[name] for name in
                                  ("freeze.json", "evaluation-access.json", "evaluation.json")]
    assert freeze["status"] == "FROZEN" and evaluation["status"] == "COMPLETE"
    for document in (freeze, access, evaluation):
        assert document["self_hash"] == sha(canonical({k: v for k, v in document.items() if k != "self_hash"}))
    assert set(freeze["artifacts"]) == set(names) - {"freeze.json", "evaluation-access.json", "evaluation.json"}
    for name, expected in freeze["artifacts"].items():
        assert sha(raw[name]) == expected
    assert access["freeze_sha256"] == evaluation["freeze_sha256"] == sha(raw["freeze.json"])
    assert evaluation["access_sha256"] == sha(raw["evaluation-access.json"])
    assert access["gold_sha256"] == evaluation["gold_sha256"] == freeze["sources"]["v1_gold_sha256"]
    assert freeze["source_identity"]["git_commit"] == commit
    assert freeze["source_identity"]["git_available"] is True and freeze["source_identity"]["git_dirty"] is False
    for name, expected in freeze["source_identity"]["files"].items():
        released = subprocess.check_output(["git", "-C", str(args.repo), "show", f"{commit}:{name}"])
        assert sha(released) == expected
    protocol_raw = subprocess.check_output(["git", "-C", str(args.repo), "show", f"{commit}:configs/clinc_cheap_protocol.json"])
    protocol = docs["protocol.json"]
    assert sha(protocol_raw) == freeze["protocol_sha256"]
    assert json.loads(protocol_raw) == protocol
    assert protocol["environment"] == freeze["environment"]["packages"]
    spec = protocol["encoder"]
    assert spec["repository"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert spec["revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert spec["max_length"] == 256 and spec["batch_size"] == 32 and spec["embedding_dimension"] == 384
    assert spec["threads"] == 1 and spec["provider"] == "CPUExecutionProvider"
    model_identity = {"repository": spec["repository"], "revision": spec["revision"],
                      "provider": spec["provider"], "files": spec["files"]}
    assert freeze["model_identity"] == model_identity
    assert len(spec["files"]) == 8
    for name, expected in spec["files"].items():
        assert file_identity(args.model_dir / name) == expected
    model_config = json.loads((args.model_dir / "config.json").read_text())
    assert model_config["hidden_size"] == 384 and model_config["pad_token_id"] == 0
    pooling = json.loads((args.model_dir / "1_Pooling/config.json").read_text())
    assert pooling == {"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
                       "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
                       "pooling_mode_mean_sqrt_len_tokens": False}
    assert json.loads((args.model_dir / "sentence_bert_config.json").read_text())["max_seq_length"] == 256
    modules = json.loads((args.model_dir / "modules.json").read_text())
    assert [m["type"] for m in modules] == ["sentence_transformers.models.Transformer",
                                           "sentence_transformers.models.Pooling", "sentence_transformers.models.Normalize"]
    for name, expected in freeze["prepared_hashes"].items():
        assert sha((args.prepared_dir / name).read_bytes()) == expected
    assert sha(args.examples.read_bytes()) == freeze["sources"]["v1_examples_sha256"]
    training = [json.loads(line) for line in (args.prepared_dir / "train1600.jsonl").read_text().splitlines()]
    examples = [json.loads(line) for line in args.examples.read_text().splitlines()]
    roles = {split: sorted((r for r in examples if r["task"] == "clinc" and r["split"] == split), key=lambda r: r["sample_id"])
             for split in ("validation", "test", "stress")}
    assert len(training) == 1600 and training == sorted(training, key=lambda r: r["sample_id"])
    assert {k: len(v) for k, v in roles.items()} == {"validation": 400, "test": 600, "stress": 850}
    ordered = training + roles["validation"] + roles["test"] + roles["stress"]
    assert len(ordered) == 3450 and len({r["sample_id"] for r in ordered}) == 3450
    feature = docs["embeddings.json"]
    assert feature["inputs"] == [{"sample_id": r["sample_id"], "text_sha256": sha(r["state"].encode())} for r in ordered]
    assert feature["model_identity"] == feature["encoding"]["identity"] == model_identity
    assert feature["matrix_sha256"] == sha(raw["embeddings.npy"])
    saved = np.load(io.BytesIO(raw["embeddings.npy"]), allow_pickle=False)
    assert saved.dtype == np.float32 and saved.shape == (3450, 384) and np.isfinite(saved).all()
    assert np.allclose(np.linalg.norm(saved, axis=1), 1., atol=1e-5, rtol=0)

    tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    tokens = tokenizer.encode_batch([r["state"] for r in ordered], add_special_tokens=True)
    lengths = [len(t.ids) for t in tokens]
    assert lengths == feature["encoding"]["token_lengths"]
    assert max(lengths) == feature["encoding"]["max_tokens_including_special"] == 25
    assert all(0 < n <= 256 for n in lengths) and not any(t.overflowing for t in tokens)
    assert feature["encoding"]["truncated_count"] == 0
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.model_dir / "onnx/model.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert {x.name: x.type for x in session.get_inputs()} == {k: "tensor(int64)" for k in ("input_ids", "attention_mask", "token_type_ids")}
    assert [x.name for x in session.get_outputs()] == ["last_hidden_state"]
    replay = np.empty_like(saved)
    # Predeclared audit tolerance for cross-platform float32 replay, not a task metric.
    tolerance = 1e-5
    for start in range(0, len(tokens), 32):
        batch = tokens[start:start + 32]
        width = max(len(t.ids) for t in batch)
        feeds = {k: np.zeros((len(batch), width), dtype=np.int64)
                 for k in ("input_ids", "attention_mask", "token_type_ids")}
        for j, token in enumerate(batch):
            n = len(token.ids)
            assert token.attention_mask == [1] * n
            feeds["input_ids"][j, :n] = token.ids
            feeds["attention_mask"][j, :n] = 1
            feeds["token_type_ids"][j, :n] = token.type_ids
        hidden = session.run(["last_hidden_state"], feeds)[0]
        assert hidden.shape == (len(batch), width, 384) and np.isfinite(hidden).all()
        mask = feeds["attention_mask"].astype(np.float32)
        pooled = np.einsum("btd,bt->bd", hidden, mask, optimize=False) / mask.sum(axis=1)[:, None]
        unit = pooled / np.sqrt((pooled * pooled).sum(axis=1))[:, None]
        replay[start:start + len(batch)] = unit
    assert np.isfinite(replay).all()
    errors = np.abs(saved.astype(np.float64) - replay.astype(np.float64))
    row_errors = errors.max(axis=1)
    s64, r64 = saved.astype(np.float64), replay.astype(np.float64)
    cosine = (s64 * r64).sum(axis=1) / (np.linalg.norm(s64, axis=1) * np.linalg.norm(r64, axis=1))
    assert float(row_errors.max()) <= tolerance
    classifier_replay = {}
    for name in ("minilm400", "minilm1600"):
        model = docs[f"{name}.model.json"]
        labels, classes = protocol["labels"], model["classes"]
        assert model["prediction_label_order"] == labels and set(classes) == set(labels)
        coefficient, intercept = np.array(model["coefficients"]), np.array(model["intercept"])
        assert coefficient.shape == (16, 384) and intercept.shape == (16,)
        logits = r64[1600:] @ coefficient.T + intercept
        exp = np.exp(logits - logits.max(axis=1)[:, None])
        probability = exp / exp.sum(axis=1)[:, None]
        probability = probability[:, [classes.index(label) for label in labels]]
        pred = [json.loads(line) for line in raw[f"{name}.predictions.jsonl"].splitlines()]
        assert [r["sample_id"] for r in ordered[1600:]] == [r["sample_id"] for r in pred]
        saved_probability = np.array([r["probabilities"] for r in pred])
        replay_labels = [labels[i] for i in probability.argmax(axis=1)]
        label_changes = sum(a != b["predicted_label"] for a, b in zip(replay_labels, pred))
        assert label_changes == 0, (name, label_changes)
        classifier_replay[name] = {"count": len(pred), "label_changes": label_changes,
                                   "maximum_probability_absolute_error": float(np.abs(probability - saved_probability).max())}
    encoder_timing = {k: v for k, v in feature["encoding"].items() if k.endswith("_seconds")}
    components = sum(v for k, v in encoder_timing.items() if k != "total_seconds")
    assert all(v >= 0 for v in encoder_timing.values()) and components <= encoder_timing["total_seconds"]
    classifier_timings = {}
    for name in ("tfidf1600", "minilm400", "minilm1600"):
        fit = docs[f"{name}.fit.json"]
        timing = fit["timing"]
        assert all(v >= 0 for v in timing.values())
        classifier_timings[name] = {"training_count": fit["training_count"], "prediction_count": fit["prediction_count"],
                                    "reported_seconds": timing, "shared_encoder_metadata": fit["shared_encoder_metadata"]}
        if name == "tfidf1600":
            classifier_timings[name]["training_feature_and_classifier_seconds_sum"] = timing["feature_fit_seconds"] + timing["fit_seconds"]
            classifier_timings[name]["prediction_feature_and_classifier_seconds_sum"] = timing["feature_predict_seconds"] + timing["batch_predict_seconds"]
    result = {"status": "PASS", "audit_kind": "Independent fixed ONNX replay; no fitting, new protocol or gold-file scoring",
              "archive_sha256": bundle_hash, "source_commit": commit,
              "model_identity": model_identity, "rows": 3450, "dimension": 384,
              "all_input_ids_and_text_hashes_match": True, "all_token_lengths_match": True,
              "max_tokens_including_special": max(lengths), "truncated_count": 0,
              "saved_embedding_sha256": sha(raw["embeddings.npy"]),
              "cross_platform_replay": {"max_absolute_error": float(errors.max()), "mean_absolute_error": float(errors.mean()),
                                        "max_row_l2_difference": float(np.linalg.norm(s64 - r64, axis=1).max()),
                                        "minimum_cosine_similarity": float(cosine.min()), "float32_absolute_tolerance": tolerance,
                                        "rows_over_tolerance": int((row_errors > tolerance).sum()),
                                        "saved_max_unit_norm_error": float(np.abs(np.linalg.norm(s64, axis=1) - 1).max()),
                                        "replay_max_unit_norm_error": float(np.abs(np.linalg.norm(r64, axis=1) - 1).max()),
                                        "byte_identical": bool(saved.tobytes() == replay.tobytes())},
              "pooling_replay_implementation": "Independent float32 einsum masked sum, token-count division and L2 normalization; observed small differences may combine platform/backend and reduction-order effects, not an isolated hardware causal comparison.",
              "classifiers_using_replayed_encoder": classifier_replay,
              "audit_environment": {"python": platform.python_version(), "system": platform.system(),
                                    "machine": platform.machine(), "provider": session.get_providers(), "threads": 1,
                                    "packages": {k: metadata.version(k) for k in ("numpy", "onnxruntime", "tokenizers")}},
              "reported_node_environment": freeze["environment"],
              "reported_encoder_timing_seconds": encoder_timing,
              "encoder_component_seconds_sum": components,
              "encoder_unassigned_clock_overhead_seconds": encoder_timing["total_seconds"] - components,
              "reported_classifier_timings": classifier_timings,
              "timing_review": [
                  "Encoder total is one 3450-input CPU pass, including verification/tokenizer/session loads; inference_pool includes batch padding, ONNX calls, pooling and concatenation.",
                  "Encoder work covers1600 training and1850 evaluation rows, shared between two heads. It is not1850-only inference latency and must not be omitted from either deployment cost discussion.",
                  "Classifier fit/predict clocks exclude encoder, input verification, object construction, array float64 conversion, serialization, process startup and HTTP/queue overhead.",
                  "TFIDF feature fit/transform clocks can be added to corresponding classifier clocks as arithmetic component sums, not independently measured end-to-end times.",
                  "Stored Linux elapsed times cannot be independently remeasured from this bundle. This cross-platform replay is numerical validation, not a new performance benchmark or measured GPU speedup.",
                  "Recorded processor is only x86_64; no detailed CPU model, repeat trials or service traces are present."],
              "limits": "Hashes are integrity checks, not signatures or historical OS access traces. Original embeddings/predictions remain unchanged; replay matrix is not substituted into published task metrics."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result))
    print(json.dumps({k: result[k] for k in ("status", "cross_platform_replay", "classifiers_using_replayed_encoder")}, indent=2))


if __name__ == "__main__":
    main()
