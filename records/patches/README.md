# Patch receipts, 2026-11-01 to 2026-11-02

Each JSON lists the files and SHA-256 hashes of one small repair package that was uploaded to
the GPU node during the Study 1 runs. The files sat at the repository root until 2026-11-07;
nothing in the code reads them. Paths inside each receipt describe the package layout at the
time and may name documents that now live in `docs/history/`.

| Receipt | Problem it fixed | Runbook |
|---|---|---|
| `STARTLUX_REPAIR_SHA256.json` | StartLux reference check lacked Accelerate | `docs/history/RECOVER_STARTLUX_ZH.md` |
| `QWEN_TOOLS_PATCH_SHA256.json` | vLLM launch path did not contain `ninja` | `docs/history/RECOVER_QWEN_TOOLS_ZH.md` |
| `QWEN_CUDA_PATCH_SHA256.json` | sampling JIT pointed at a missing `nvcc` | `docs/history/RECOVER_QWEN_CUDA_ZH.md` |
| `QWEN_CUDA_V2_PATCH_SHA256.json` | CUDA 13.4 compiler against 13.0 headers; isolated 13.0 toolchain | `docs/history/RECOVER_QWEN_CUDA_V2_ZH.md` |
| `QWEN_CUDA_V3_PATCH_SHA256.json` | missing `curand.h` for the sampler | `docs/history/RECOVER_QWEN_CUDA_V3_ZH.md` |

The packages themselves are archived outside Git in
`project/solo-experiments-2026-10-30/archive/` with their own SHA-256 list.
