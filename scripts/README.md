# Scripts

Files marked **bound** are hashed by `SOURCE_SHA256.json` or a V2/V3 release manifest; keep their
path and bytes. Node commands run from a pinned checkout with the existing runtime root (see the
top-level README).

## Local, CPU only

| Script | Purpose |
|---|---|
| `reproduce_cpu.sh` | All CPU checks: tests, Lab evidence, follow-up audits, figures, paper build |
| `run_lab.sh` | Start the Decision Lab at `http://127.0.0.1:7860` (`--port` to change) |

## GPU node: experiment entry points

| Script | Study |
|---|---|
| `run_with_cuda_tools.sh` and its helper `run_with_cuda_tools.py` (helper **bound**) | Study 1 jobs (`smoke`, `startlux_full`, `qwen_test`, `benchmark`) with the isolated CUDA 13.0 toolchain |
| `run_with_vllm_tools.sh` | Earlier Study 1 launcher that only puts the installed build tools on `PATH` |
| `run_v2.sh` **bound** | Study 2: `init-runtime`, `pipeline`, `benchmark`, `evaluate`, `export` |
| `run_v3.sh` **bound** | Study 3: `start` (resumable) |
| `run_light_baseline.sh` | TF-IDF-400 CLINC baseline |
| `run_cheap_baselines.sh`, `run_cheap_baseline_suite.sh` | TF-IDF-1600 and MiniLM baselines; the suite also prepares the pinned CPU environment |
| `run_service_study.sh`, `run_size_control.sh` | CPU/GPU service study and StartLux-9B size control, one at a time |
| `run_followup_suite.sh` | Both follow-ups in sequence; reads the Slurm time left (`--new-allocation` after a new session) |
| `run_vm.sh` **bound** | Original Study 1 batch entry: all jobs, then evaluation (`JOB=all bash scripts/run_vm.sh`) |
| `prepare_node_checkout.py` | Connect a pinned Git checkout to the existing runtime root and verify the prepared data |

## Release identity and runtime tooling

| Script | Purpose |
|---|---|
| `verify_v2_release.py`, `verify_v3_release.py` **bound** | Read-only check of the node's files against the V2/V3 release manifests |
| `build_v3_release.py` **bound** | Builds the additive V3 package and its manifest |
| `package_release.py` **bound** | Source-only handoff archive for V1 (uses `SOURCE_SHA256.json`, `VALIDATION.md`) |
| `bootstrap_models.py` **bound** | Install environments and download pinned weights on a Linux CUDA machine |
| `model_probe.py` **bound** | Probe run inside the model environment |
| `cuda_toolchain_preflight.py` **bound** | Verify the isolated CUDA 13.0 toolkit with a FlashInfer sampling compile |
| `prepare_cuda_toolkit.py` **bound** | Install the hash-pinned CUDA toolkit into an isolated directory |
| `build_notebooks.py` **bound** | Build the two thin V1 notebooks in `notebooks/` |

## Historical repairs (Study 1, 31 Oct to 1 Nov)

| Script | Purpose |
|---|---|
| `recover_kev.sh` **bound**, `cleanup_legacy_cache.py` **bound** | Kev `Python.h` incident and removal of three legacy caches |
| `recover_startlux.sh`, `repair_startlux_reference.py` | Missing reference-only dependencies for StartLux; receipt in `records/patches/` |
