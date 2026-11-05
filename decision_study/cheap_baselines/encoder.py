"""Fixed public MiniLM ONNX encoder; CPU-only, no training or silent truncation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request


MODEL_FILES = {
    "onnx/model.onnx": {"sha256": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452", "bytes": 90405214},
    "tokenizer.json": {"sha256": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037", "bytes": 466247},
    "config.json": {"sha256": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41", "bytes": 612},
    "tokenizer_config.json": {"sha256": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b", "bytes": 350},
    "special_tokens_map.json": {"sha256": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3", "bytes": 112},
    "sentence_bert_config.json": {"sha256": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7", "bytes": 53},
    "1_Pooling/config.json": {"sha256": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23", "bytes": 190},
    "modules.json": {"sha256": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf", "bytes": 349},
}
ENCODER_SPEC = {
    "repository": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "backend": "onnxruntime", "provider": "CPUExecutionProvider", "dtype": "float32",
    "embedding_dimension": 384, "pooling": "attention_mask_mean_then_l2",
    "max_length": 256, "batch_size": 32, "threads": 1, "files": MODEL_FILES,
}


class EncoderError(RuntimeError):
    """A fixed encoder identity, token boundary or numeric invariant failed."""


def _spec(protocol):
    spec = protocol.get("encoder")
    if spec != ENCODER_SPEC:
        raise EncoderError("Encoder settings differ from the fixed MiniLM CPU recipe")
    return spec


def _path(root, relative):
    path = root
    for part in Path(relative).parts:
        if part in ("..", "/"):
            raise EncoderError("Unsafe encoder path")
        path = path / part
        if path.is_symlink():
            raise EncoderError(f"Symlink in encoder path: {path}")
    return path


def _root(model_dir):
    root = Path(model_dir).expanduser().absolute()
    if root.is_symlink():
        raise EncoderError("Encoder root must not be a symlink")
    return root


def _file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(data)
            digest.update(data)
    return {"sha256": digest.hexdigest(), "bytes": size}


def verify(model_dir, protocol):
    """Verify immutable downloaded bytes and the publisher's pooling settings."""
    spec, root = _spec(protocol), _root(model_dir)
    for name, expected in spec["files"].items():
        path = _path(root, name)
        if not path.is_file() or _file_identity(path) != expected:
            raise EncoderError(f"Missing or changed encoder file: {path}")
    read = lambda name: json.loads(_path(root, name).read_text())
    config, pooling = read("config.json"), read("1_Pooling/config.json")
    if (config["hidden_size"] != 384 or config["pad_token_id"] != 0
            or read("sentence_bert_config.json")["max_seq_length"] != 256
            or pooling != {"word_embedding_dimension": 384, "pooling_mode_cls_token": False,
                           "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
                           "pooling_mode_mean_sqrt_len_tokens": False}
            or [m["type"] for m in read("modules.json")] != [
                "sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
                "sentence_transformers.models.Normalize"]):
        raise EncoderError("Publisher encoder/pooling configuration changed")
    return {"repository": spec["repository"], "revision": spec["revision"],
            "files": spec["files"], "provider": spec["provider"]}


def download(model_dir, protocol):
    """Download pinned public files only; never replace a conflicting final file."""
    spec, root = _spec(protocol), _root(model_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = _path(root, ".download.lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise EncoderError(f"Encoder download locked: {lock}; inspect its owner") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()) + "\n")
        for name, expected in spec["files"].items():
            target = _path(root, name)
            if target.exists():
                if not target.is_file() or _file_identity(target) != expected:
                    raise EncoderError(f"Existing encoder file conflicts: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{name}"
            fd, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
            temporary = Path(temporary)
            try:
                with os.fdopen(fd, "wb") as handle, urllib.request.urlopen(url, timeout=60) as source:
                    size = 0
                    for data in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(data)
                        if size > expected["bytes"]:
                            raise EncoderError(f"Downloaded encoder file is oversized: {name}")
                        handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                if _file_identity(temporary) != expected:
                    raise EncoderError(f"Downloaded encoder hash/size mismatch: {name}")
                # Exclusive hard-link installation prevents overwriting concurrent files.
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return verify(root, protocol)
    finally:
        lock.unlink()


def _load_tokenizer(path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def _tokenize(tokenizer, texts, max_length):
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise EncoderError("Encoder expects a nonempty list of nonempty strings")
    tokens = tokenizer.encode_batch(list(texts), add_special_tokens=True)
    lengths = [len(row.ids) for row in tokens]
    if len(tokens) != len(texts) or any(n == 0 or n > max_length for n in lengths):
        raise EncoderError(f"Input exceeds frozen {max_length}-token limit; no truncation permitted")
    if any(row.overflowing for row in tokens):
        raise EncoderError("Tokenizer produced overflow segments; no truncation permitted")
    return tokens, lengths


def check_tokens(texts, model_dir, protocol):
    """Read-only token coverage preflight; does not create an inference session."""
    spec, root = _spec(protocol), _root(model_dir)
    identity = verify(root, protocol)
    _, lengths = _tokenize(_load_tokenizer(_path(root, "tokenizer.json")), texts, spec["max_length"])
    return {"identity": identity, "count": len(texts), "token_lengths": lengths,
            "max_tokens_including_special": max(lengths), "truncated_count": 0}


def _load_session(path, threads):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise EncoderError("The frozen encoder must use CPUExecutionProvider exclusively")
    return session


def mean_pool(token_embeddings, attention_mask):
    """Publisher masked-mean pooling followed by float32 L2 normalization."""
    import numpy as np
    values = np.asarray(token_embeddings, dtype=np.float32)
    mask = np.asarray(attention_mask)
    if (values.ndim != 3 or mask.shape != values.shape[:2]
            or not np.isfinite(values).all() or not np.isin(mask, (0, 1)).all()
            or not (mask.sum(axis=1) > 0).all()):
        raise EncoderError("Invalid token embeddings or attention mask")
    mask = mask.astype(np.float32)[..., None]
    pooled = (values * mask).sum(axis=1) / mask.sum(axis=1)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 0).any():
        raise EncoderError("Non-finite or zero-norm pooled embedding")
    result = pooled / norm
    if not np.isfinite(result).all():
        raise EncoderError("Non-finite normalized embedding")
    return np.ascontiguousarray(result, dtype=np.float32)


def encode(texts, model_dir, protocol):
    """Encode unmodified texts in fixed batches; reject any input over 256 tokens."""
    import numpy as np
    started = time.perf_counter()
    spec, root = _spec(protocol), _root(model_dir)
    identity = verify(root, protocol)
    verification_seconds = time.perf_counter() - started
    tick = time.perf_counter()
    tokenizer = _load_tokenizer(_path(root, "tokenizer.json"))
    tokenizer_load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    tokens, token_lengths = _tokenize(tokenizer, texts, spec["max_length"])
    tokenize_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    session = _load_session(_path(root, "onnx/model.onnx"), spec["threads"])
    expected_inputs = {"input_ids", "attention_mask", "token_type_ids"}
    if ({node.name for node in session.get_inputs()} != expected_inputs
            or any(node.type != "tensor(int64)" for node in session.get_inputs())
            or [node.name for node in session.get_outputs()] != ["last_hidden_state"]):
        raise EncoderError("ONNX input/output signature differs from the fixed encoder")
    session_load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    blocks = []
    for offset in range(0, len(tokens), spec["batch_size"]):
        batch = tokens[offset:offset + spec["batch_size"]]
        width = max(len(row.ids) for row in batch)
        ids = np.zeros((len(batch), width), dtype=np.int64)
        mask = np.zeros_like(ids)
        types = np.zeros_like(ids)
        for index, row in enumerate(batch):
            length = len(row.ids)
            if len(row.type_ids) != length or row.attention_mask != [1] * length:
                raise EncoderError("Unexpected unpadded tokenizer output")
            ids[index, :length] = row.ids
            mask[index, :length] = 1
            types[index, :length] = row.type_ids
        hidden = session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask,
                                                   "token_type_ids": types})[0]
        if hidden.shape != (len(batch), width, spec["embedding_dimension"]):
            raise EncoderError("ONNX hidden-state dimensions differ from frozen encoder")
        blocks.append(mean_pool(hidden, mask))
    matrix = np.concatenate(blocks, axis=0)
    inference_pool_seconds = time.perf_counter() - tick
    if (matrix.shape != (len(texts), spec["embedding_dimension"])
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0, atol=1e-5)):
        raise EncoderError("Invalid final embedding matrix")
    return matrix, {"identity": identity, "count": len(texts), "dimension": spec["embedding_dimension"],
                    "dtype": str(matrix.dtype), "max_tokens_including_special": max(token_lengths),
                    "token_lengths": token_lengths, "truncated_count": 0,
                    "batch_size": spec["batch_size"], "threads": spec["threads"],
                    "verification_seconds": verification_seconds,
                    "tokenizer_load_seconds": tokenizer_load_seconds,
                    "tokenize_seconds": tokenize_seconds, "session_load_seconds": session_load_seconds,
                    "inference_pool_seconds": inference_pool_seconds,
                    "total_seconds": time.perf_counter() - started,
                    "timing_scope": "One CPU embedding pass, including identity verification and loads; component clocks separate tokenization/session/padding/inference/pooling. No HTTP or classifier timing."}
