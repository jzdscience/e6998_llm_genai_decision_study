#!/usr/bin/env python3
"""Run inside the isolated model environment, never in the controller kernel.

Uses pinned upstream renderers/readouts; it does not approximate token lengths.
All heavyweight imports are inside the selected operation.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import sysconfig
import time

# Import the controller's exact task rendering without installing its dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def examples_at(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def request_for(example, model):
    from decision_study.adapters import decision_request
    return decision_request(example, model)


def download(model, out, lock):
    from huggingface_hub import snapshot_download
    cfg = lock["models"][model]
    patterns = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja", "*.model"]
    base_path = None
    if model == "kev":
        base_path = snapshot_download(cfg["base_repo"], revision=cfg["base_revision"], allow_patterns=patterns)
    path = snapshot_download(cfg["repo"], revision=cfg["revision"], allow_patterns=patterns)
    if model == "kev":
        from kev.checkpoint import Checkpoint
        checkpoint = Checkpoint(path)
        if (checkpoint.meta.base, checkpoint.meta.base_revision) != (cfg["base_repo"], cfg["base_revision"]):
            raise RuntimeError("Kev checkpoint base model/revision differs from the experiment lock")
        if not math.isclose(checkpoint.meta.temperature, cfg["native_temperature"], abs_tol=1e-10):
            raise RuntimeError("Kev published temperature differs from lock")
    elif model == "startlux":
        from startlux_decide.modelcfg import load_model_folder
        folder = load_model_folder(path)
        if not folder.bidirectional or folder.kind != "instant":
            raise RuntimeError("The pinned StartLux checkpoint is not the expected Instant bidirectional model")
        if not math.isclose(folder.temperature("choice"), cfg["native_temperature"], abs_tol=1e-10):
            raise RuntimeError("StartLux published choice temperature differs from lock")
    return {"model": model, "model_path": path, "base_path": base_path, "revision": cfg["revision"],
            "base_revision": cfg.get("base_revision"), "native_temperature": cfg.get("native_temperature")}


def triton_build_check():
    """Initialize the real CUDA helper; a Torch matmul alone never tests its C build."""
    from triton.runtime import driver
    include = sysconfig.get_path("include")
    try:
        target = driver.active.get_current_target()
    except Exception as exc:
        raise RuntimeError(
            "Triton CUDA helper initialization failed after Torch BF16 passed. "
            "Inspect the compiler/linker stderr above (missing headers, libcuda, or write failure); "
            f"Python include directory: {include}. Do not infer the cause from exit status alone."
        ) from exc
    return {"status": "ok", "target": str(target), "python_include": include,
            "scope": "Triton CUDA driver helper initialized; not a model-kernel correctness test"}


def gpu_check():
    """Validate wheel/driver/BF16 compatibility before any checkpoint download."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Installed Torch cannot access CUDA; check driver/wheel compatibility before downloading checkpoints")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/runtime does not support the frozen BF16 experiment protocol")
    matrix = torch.ones((16, 16), device="cuda:0", dtype=torch.bfloat16)
    product = matrix @ matrix
    torch.cuda.synchronize()
    if product.dtype != torch.bfloat16 or not bool((product == 16).all().item()):
        raise RuntimeError("BF16 CUDA matrix smoke check failed")
    triton_report = triton_build_check()
    versions = {}
    for distribution in ("torch", "transformers", "vllm", "flash-linear-attention", "causal-conv1d"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {"status": "ok", "cuda_available": True, "bf16_matmul_passed": True,
            "gpu_name": torch.cuda.get_device_name(0), "compute_capability": list(capability),
            "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "library_versions": versions, "checkpoint_downloaded": False,
            "triton_build": triton_report,
            "scope": "Tiny CUDA operation and Triton driver helper; model attention/readout is checked separately"}


def _renderers(model, model_path, lock):
    if model == "kev":
        from kev.checkpoint import Checkpoint
        from kev.model import encode, load_tokenizer
        from kev.api import SystemOneRequest, to_record
        ck = Checkpoint(model_path)
        tok = load_tokenizer(ck.meta.base, revision=ck.meta.base_revision)
        def render(e):
            req = SystemOneRequest.model_validate(request_for(e, model))
            record, _ = to_record(req)
            enc = encode(tok, record, max_state=lock["max_model_len"], max_branch=lock["max_model_len"],
                         strict=True, option_isolation=ck.meta.option_isolation)
            return enc["ids"], enc
        return tok, render
    if model == "startlux":
        from startlux_decide.modelcfg import load_tokenizer
        from startlux_decide.prompt import check_tokenizer, from_systemone, messages, plain_prompt
        tok = load_tokenizer(model_path)
        check_tokenizer(tok)
        def render(e):
            req = request_for(e, model)
            row = from_systemone(req["state"], req["questions"]["decision"], allow_none=False)
            prompt = plain_prompt(tok, messages(row))
            return tok.encode(prompt, add_special_tokens=False), prompt
        return tok, render
    from transformers import AutoTokenizer
    from decision_study.adapters import qwen_messages
    tok = AutoTokenizer.from_pretrained(model_path)
    def render(e):
        prompt = tok.apply_chat_template(qwen_messages(e), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tok.encode(prompt, add_special_tokens=False), prompt
    return tok, render


def tokens(model, examples, model_path, lock):
    _, render = _renderers(model, model_path, lock)
    reserve = lock["max_output_tokens"] if model == "qwen" else 1 if model == "startlux" else 0
    rows, overlong = [], []
    for example in examples:
        try:
            ids, _ = render(example)
            row = {"sample_id": example["sample_id"], "input_tokens": len(ids), "reserved_output_tokens": reserve,
                   "token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
            if len(ids) + reserve > lock["max_model_len"]:
                overlong.append({**row, "error": "context_limit_exceeded"})
            rows.append(row)
        except ValueError as exc:
            overlong.append({"sample_id": example["sample_id"], "error": str(exc)})
    return {"status": "blocked" if overlong else "ok", "model": model, "model_revision": lock["models"][model]["revision"],
            "max_model_len": lock["max_model_len"], "n_examples": len(examples), "rows": rows, "overlong": overlong,
            "renderer": "official", "truncated": False}


def probability_row(example, model, values, start, raw=None):
    values = [float(v) for v in values]
    if len(values) != len(example["labels"]) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid reference probability vector")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise ValueError(f"Reference probabilities sum to {total}")
    winner = max(range(len(values)), key=values.__getitem__)
    return {"sample_id": example["sample_id"], "model": model, "status": "ok",
            "prediction": example["labels"][winner], "probabilities": dict(zip(example["labels"], values)),
            "latency_ms": 1000 * (time.perf_counter() - start), "attempts": [], "error": None,
            "raw_response": raw or {"reference": True}, "token_counts": {}}


def reference_kev(examples, model_path, lock):
    import torch
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions
    checkpoint = Checkpoint(model_path)
    tok, model = checkpoint.load("cuda", LoadOptions(dtype=torch.bfloat16, backend="torch", merge=True, cuda_graphs=False, fused=False))
    results = []
    for example in examples:
        req = SystemOneRequest.model_validate(request_for(example, "kev"))
        record, _ = to_record(req)
        enc = model.encode(tok, record, max_state=lock["max_model_len"], max_branch=lock["max_model_len"], strict=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            values = model.probs(enc)[0].tolist()
        torch.cuda.synchronize()
        row = probability_row(example, "kev", values, start, {"reference": "official-eager-bf16", "native_temperature": model.head.temperature})
        row["token_counts"] = {"input_tokens": len(enc["ids"])}
        results.append(row)
    return results


async def reference_startlux(examples, model_path, lock):
    import torch
    from startlux_decide.modelcfg import load_model_folder
    from startlux_decide.prompt import check_tokenizer
    from startlux_decide.backends.transformers import TransformersBackend
    folder = load_model_folder(model_path)
    tok, render = _renderers("startlux", model_path, lock)
    letter_ids = check_tokenizer(tok)
    engine = TransformersBackend(model_path, tok, bidirectional=True, device="cuda", dtype="bfloat16")
    await engine.start()
    results = []
    try:
        for example in examples:
            ids, prompt = render(example)
            torch.cuda.synchronize()
            start = time.perf_counter()
            logits = await engine.readout(prompt, letter_ids[:len(example["labels"])])
            scaled = [float(v) / folder.temperature("choice") for v in logits]
            max_logit = max(scaled)
            exp = [math.exp(v - max_logit) for v in scaled]
            values = [v / sum(exp) for v in exp]
            torch.cuda.synchronize()
            row = probability_row(example, "startlux", values, start, {"reference": "official-transformers-bf16", "engine": engine.describe(), "native_temperature": folder.temperature("choice")})
            row["token_counts"] = {"input_tokens": len(ids)}
            results.append(row)
    finally:
        await engine.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "tokens", "reference", "gpu-check"))
    parser.add_argument("--model", choices=("kev", "startlux", "qwen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    lock = read(args.output_dir / "runtime/lock.json")
    if args.action == "gpu-check":
        result = gpu_check()
    elif args.action == "download":
        result = download(args.model, args.output_dir, lock)
    else:
        installations = read(args.output_dir / "runtime/installations.json")
        model_path = installations["models"][args.model]["model_path"]
        examples = examples_at(args.examples or args.output_dir / "data/examples.jsonl")
        if args.action == "tokens":
            result = tokens(args.model, examples, model_path, lock)
        elif args.model == "kev":
            result = {"predictions": reference_kev(examples, model_path, lock)}
        elif args.model == "startlux":
            result = {"predictions": asyncio.run(reference_startlux(examples, model_path, lock))}
        else:
            raise ValueError("Qwen has no probability-reference operation")
    write(args.result, result)


if __name__ == "__main__":
    main()
