# Troubleshooting

Failures met while running this project, with the cause and the fix that worked. Node commands
assume the runtime root `<root>` (on our node `/home/<user>/e6998/decision-study`) and its
controller environment `<root>/.venv-controller`.

## On the GPU node

### The code-server session disappears while a run is loading models

Slurm killed the job for host memory (`sacct` shows `OUT_OF_MEMORY`); GPU memory was not the
limit. Request at least 64 GB of host RAM for the allocation; our runs used 100 GB. To confirm:
`sacct -j <jobid> --units=G --format=JobID,State%20,ExitCode,ReqMem,MaxRSS,Elapsed`.
Completed predictions survive; rerun the same command in the new allocation.

### `BLOCKED: Session deadline reached` (Study 1 controller)

The controller stops at the session end it recorded. In a new allocation run
`.venv-controller/bin/python -m decision_study session --renew-session --session-hours-left <H> --output-dir results/solo-v1`
with the remaining hours the platform shows, minus a margin.

### `BLOCKED: Renewal requires an explicitly acknowledged different Slurm job` (follow-up suite)

The suite reads the Slurm time left by itself and refuses to switch jobs silently. After opening a
new session, add `--new-allocation`. Study 2 uses
`bash scripts/run_v2.sh init-runtime --new-allocation --session-hours-left <H> ...`.

### `argument --session-hours-left: invalid float value: 'bash scripts/...'`

A `read -rp` prompt captured the next pasted command. Type the number directly on the command line.

### vLLM cannot find `ninja` while compiling

The environment had ninja, but the launch path did not include it. Start Qwen through
`bash scripts/run_with_vllm_tools.sh smoke`.

### Qwen sampling JIT fails: missing `nvcc`, CUDA 13.4 compiler against 13.0 headers, or missing `curand.h`

The node's default CUDA did not match the PyTorch/vLLM build. `bash scripts/run_with_cuda_tools.sh <job>`
prepares an isolated CUDA 13.0 toolchain with cuRAND (about 160 MB on first use) and leaves the
model environment unchanged. History: `docs/history/RECOVER_QWEN_CUDA*_ZH.md`.

### `BLOCKED: Model dependencies changed after installation` from a fresh Git checkout

The new checkout no longer exposed the old runtime's `decision-study==0.1.0` metadata, so the
dependency hash changed by one line. Fixed in commit `8b5370d` (`decision_study/runtime_compat.py`).
Do not edit the expected hashes.

### `BLOCKED: Could not verify that port 18120 is unused: timed out` (service study, CPU cells)

Check `ss -ltnp 'sport = :18120'` and `pgrep -a -u "$USER" -f 'decision_study|vllm|startlux'`. If both
print nothing, rerun the same command: completed cells are verified and skipped. If a process is
listed, report it before stopping anything.

### `BLOCKED: File differs from release: decision_study/artifacts.py` (V2 release check)

An older, already audited `artifacts.py` was present. The release verifier now accepts that
recorded version; preserve the file and rerun `scripts/verify_v2_release.py`.

### A result bundle is too large to upload

Copy the ZIP directly to the analysis machine. If only chat upload is available, split it with
`split_for_chat.py` and rebuild with `join_chunks.py` (in `transfer-tools/` beside this
repository; 19 MiB parts with per-part and whole-file SHA-256).

## On macOS

### Reads hang or time out on files under `~/Desktop` or `~/Documents`

Typical messages are `git status` failing with `mmap failed: Operation timed out`, or a ZIP read
that stalls. iCloud has evicted the file contents and left placeholders, and `ls -lO` shows
`dataless`. In Finder, choose Download Now or Keep Downloaded for the folder, or restore the files
from their archived ZIPs. Archive evidence by SHA-256 as soon as it arrives, because a file synced
from another Mac under the same name can replace the local copy first.

### `Unable to create '.git/index.lock': File exists`

Make sure no git process is running, then remove `.git/index.lock`. If `.git/index` itself is
missing, rebuild it from the last commit with `git reset --mixed`; the working tree is untouched,
and anything that was staged has to be staged again; `records/git-recovery-2026-11-04.json` documents one such recovery.

### A search in zsh reports zero matches for everything

An unquoted glob in a flag (`grep -rl --include=*.py ...`) makes zsh stop with
`no matches found` before grep runs. Quote the pattern: `--include='*.py'`.

### The Decision Lab port is taken

`bash scripts/run_lab.sh --port 7861`.

### The paper does not compile on BasicTeX (missing `phvb`, `pcrr8t` or `inconsolata`)

`paper/main.tex` uses the non-review ACL mode and falls back to Latin Modern typewriter;
see `paper/README.md`.
