# GPU runtime contract

This implementation uses upstream code and published checkpoints pinned on 2026-10-30. CPU tests verify command construction, lock immutability, and process checks. A GPU installation and official-reference probability comparison are still required before formal prediction. No GPU execution was performed while authoring this package.

## Environments and pins

`configs/runtime_defaults.json` is the immutable runtime specification. `freeze_lock` verifies full Git/Hugging Face commit IDs and writes `runtime/lock.json`; it refuses mutable refs and never resolves a replacement model silently. Kev's adapter uses **Qwen3.5-4B-Base**, independently of the Qwen3.5-4B instruct fallback. Its head metadata pins base revision `1001bb4d826a52d1f399e183466143f4da7b741b` and published temperature 2.406050072164233. StartLux choice temperature is 1.1103.

The controller does not import Torch. Linux Python 3.12 environments are isolated:

- `kev`: official Kev source and serving extra. Torch/Transformers follow the pinned upstream requirements; the full resolved environment is saved after installation. Primary fused path adds flash-linear-attention 0.5.2.
- `startlux-reference`: official StartLux `[transformers]` extra, including Transformers 5.8.x. It serves as the independent numerical reference and an explicitly selected slow fallback.
- `vllm-0.31.0`: official StartLux plugin and vLLM 0.31.0. Qwen and StartLux use separate processes even though this environment is shared.
- `vllm-0.19.1`: created only when `profile=vllm019` is selected explicitly.

New environments explicitly use `uv venv --managed-python` and verify `Python.h`
before installing GPU packages. A discovered system Python executable alone is
insufficient for Triton's native extension build. Existing environments are never
silently cleared; an incomplete or header-less environment requires the explicit
`repair-python` command. That command preserves exact resolved dependency pins,
archives bounded metadata/logs, records a resumable repair marker before replacing
the selected venv, and runs the GPU/Triton checks without downloading checkpoints.
It rejects repairs that would change an environment used by formal predictions.
See [the company Kev recovery steps](RECOVER_KEV_ZH.md).

All models are BF16. T4 is rejected for this protocol; no automatic FP16/4-bit conversion is performed. `doctor` checks NVIDIA driver visibility and compute capability. The actual installed Torch CUDA compatibility is checked when the services start, not inferred solely from the GPU name.

## Colab persistence

Set this **before installation or runtime calls**:

```python
import os
os.environ['DECISION_STUDY_RUNTIME_DIR'] = '/content/decision-study-runtime'
```

Keep `OUTPUT_DIR` on mounted Drive. Metadata, logs, manifests, and dependency resolution live under `OUTPUT_DIR`; environments, Git checkouts, and the Hugging Face model cache live on local runtime storage. A reset re-creates missing environments and re-downloads the pinned weights. Do not copy virtual environments or weights to Drive or include them in result bundles.

On a VM the default working storage is `OUTPUT_DIR/runtime/work`. The resolved dependency files are retained in `OUTPUT_DIR/runtime/resolved`. Subsequent installation reuses their version pins. Local source package URLs are replaced by the same pinned repository checkout when rebuilding on a different machine.

## Explicit install and preflight

From the package directory:

```bash
python scripts/bootstrap_models.py --output-dir /path/to/results --models kev startlux qwen
```

The installer uses `uv`, installs pinned code, validates each isolated environment with a tiny BF16 CUDA matrix multiplication **before downloading checkpoint weights**, and then validates Kev base metadata and StartLux Instant metadata. It saves dependency freezes and `gpu-check-ENV.json` with Torch/CUDA/library versions. This catches a company VM's incompatible driver/wheel combination before multi-gigabyte transfers. The tiny CUDA test does not claim model inference succeeded; official-reference comparisons remain mandatory. Installation logs are `runtime/logs/install.log` and `downloads.log`.

`doctor(..., check_network=True)` optionally probes bounded public metadata endpoints for Hugging Face, GitHub, raw GitHub and PyPI. Failures are recorded without blocking offline CPU analysis. A successful probe does not guarantee that a corporate proxy permits every package CDN, model shard or Git download. No root privileges or containers are required; environments and caches remain in user-writable storage.

The controller's token preflight invokes `scripts/model_probe.py` inside the relevant model environment. It reads the actual prepared inputs and calls:

- Kev: official `SystemOneRequest`, `to_record`, `encode(strict=True)` and checkpoint tokenizer/base revision.
- StartLux: official `from_systemone`, `messages`, `plain_prompt`, `check_tokenizer`.
- Qwen: the controller's exact `qwen_messages`, actual checkpoint chat template with `enable_thinking=False`.

The GPU preflight also initializes Triton's actual CUDA driver helper before
checkpoint downloads. A successful Torch BF16 matrix multiplication alone does
not test the native C extension build required by Triton/FLA. A failed compiler
or linker invocation must be diagnosed from its stderr, not its exit code.
Subprocess errors now include the exact log path in both the top-level message
and `diagnostics/last_error.json` (`subprocess_log`).

Counts include all wrappers and option descriptions. Qwen reserves 64 output tokens. No `truncation=True` is used, and no failing sample is replaced by a shorter one. Every record includes a hash of the actual token IDs. Oversized inputs produce `tokens-MODEL.json` with status `blocked`.

## Serving and ownership

`managed_services` binds localhost only and refuses occupied ports. It starts only the requested model or one decision model plus Qwen. It never attaches to an unidentified existing service. Every subprocess has an owned process group, which is terminated in `finally`, including failures during startup.

StartLux vLLM and its TypeSafe frontend are separately managed; this prevents an engine in a detached process group from surviving frontend termination. The engine must log `prefix-bidirectional prompt attention ON`, not merely plugin registration. The pinned plugin additionally refuses unsupported workers and incompatible attention settings. The frontend must report bidirectional attention and thinking `never` in health metadata.

Other fixed controls:

- Instant: prefix cache, chunked prefill, MTP/speculative decoding, images and internal none option disabled; processed readout logprobs over up to 26 option letters.
- Kev: merged LoRA; cache/date preprocessing/truncation disabled. Published temperature is not overridden. `/v1/models` must report the expected local checkpoint and base.
- Qwen: no StartLux plugin activation; BF16, prefix cache/chunked prefill off; greedy schema-constrained JSON configured by the adapter.
- Serving cap: 8,192 tokens. Each vLLM process initially reserves 0.40 of GPU memory.

The official Kev server does not expose a comparable GPU-fraction argument. The runtime starts Kev first, then Qwen, and leaves actual peak-memory/concurrency validation to the smoke and benchmark checks. Pairing requires the planned GPU class (at least 44,000 MiB reported for a nominal 48 GB device). An environment is not a proof that two models fit.

For **standalone 24 GB Colab predictions**, explicitly select a vLLM reservation of 0.80 before smoke using `--gpu-memory-utilization 0.8`. The 0.40 paired-deployment default leaves only about 9.6 GB on a 24 GB device and may not fit a 4B model's runtime overhead. The chosen fraction is saved per model in `runtime/selected_memory.json` and included in the computation identity. Kev has no corresponding fraction setting. Paired live deployment rejects vLLM reservations above 0.40. Moving from a Colab 0.80 run to the fixed 48 GB 0.40 deployment requires a new matched run and gates; it never reuses an incompatible configuration silently.

`runtime/last_services.json` records effective configuration, startup metadata, exact launch commands, logs, and shutdown time. No complete environment-variable dump or credential is saved.

## Reference checks and fallback profiles

Reference prediction **must be run while model services are stopped**, so the reference does not create a hidden third resident model. Kev uses the official eager BF16 loader with graphs/fused off; StartLux uses the official Transformers bidirectional reference and applies the published choice temperature once. Both return full class probabilities in the input label order.

Available explicit profiles:

| Profile | Change |
|---|---|
| `primary` | Kev fused + graphs; StartLux/Qwen vLLM 0.31.0 |
| `kev_no_fused` | Kev fused off, graphs on |
| `kev_eager` | Kev fused and graphs off |
| `vllm019` | StartLux/Qwen use the officially supported vLLM 0.19.1 |
| `reference` | StartLux Transformers reference; Qwen remains vLLM 0.31.0 |

Profiles are not attempted silently. A changed computation gets a different effective configuration hash and requires affected calibration/validation/test predictions to be regenerated. StartLux reference is a correctness-oriented implementation; do not interpret its throughput as the best available StartLux serving performance.

Explicit profiles persist per model in `runtime/selected_profiles.json`; an omitted profile uses that saved selection. Profile or memory-reservation changes are refused once formal prediction configuration/shards exist. Package freezes are verified again before running a service or probe, so a manual dependency upgrade cannot masquerade as the recorded environment.

The controller budget supplies `DECISION_STUDY_DEADLINE_EPOCH`. Runtime install/download/reference subprocesses inherit the remaining deadline and have a two-hour maximum per operation. On timeout, the runtime terminates their entire owned process group. Service startup also obeys the remaining deadline. This stops monitored work; it does not terminate the cloud VM or observe the provider's bill.

## Sources

- [Kev checkpoint loader](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/checkpoint.py)
- [Kev serving](https://github.com/jaredpalmer/kev/blob/2074cc44256200bb37540103fa4572901faa10f3/kev/serve.py)
- [StartLux official rendering](https://github.com/StartLuxLabs/StartLux-Decision/blob/a99977d9332decc9fb27a2488b4a83b533b6cd5f/startlux_decide/prompt.py)
- [StartLux vLLM plugin](https://github.com/StartLuxLabs/StartLux-Decision/blob/a99977d9332decc9fb27a2488b4a83b533b6cd5f/docs/vllm_bidirectional.md)
- [StartLux engines](https://github.com/StartLuxLabs/StartLux-Decision/blob/a99977d9332decc9fb27a2488b4a83b533b6cd5f/docs/engines.md)
- [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B)
