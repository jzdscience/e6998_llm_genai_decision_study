# 节点运行时恢复手册（direct-matrix-v1，2026-11-11）

`results/solo-v1/runtime/` 与 `results/solo-v3/runtime-work/` 在节点上被删除后，用
`scripts/recover_node_runtime.py` 按存档证据恢复 direct-matrix-v1 实际读取的部分。最初发布前，工具只在 Mac 上用
真实进程、真实 git/ZIP/文件系统和伪造的 uv、环境解释器、Hugging Face 下载做过演练；发布时节点上的安装、
下载、GPU 和模型行为都还没有运行过，11/11 的节点结果见下一节。Claude Code 以 max effort 实现，Codex 独立验收：
43 项测试通过，21 个有意破坏的版本均被测试识别，另有两项独立破坏测试确认了实际的错误安装调用与目录写入。
发布验收记录见 [`records/runtime-recovery-2026-11-11-acceptance.json`](../records/runtime-recovery-2026-11-11-acceptance.json)。
详细设计、探针和原始测试日志保留在本机 `.validation/runtime-recovery-2026-11-11/`，不随 Git 发布。

## 11/11 节点结果与共享 blob 修复（待节点只读复核）

节点用 `d1633be` 运行了 plan 与 execute（UTC 2026-11-12T00:53）。七个阶段全部完成。两个环境的 freeze 哈希、
CUDA 工具链 tree 哈希、矩阵 settings 都通过，冻结的 `plans()` 也已算出。最终 verify 只报了 16 条
`does not resolve to a blob of its repository`，涉及 kev4b-adapter、kev4b-base、qwen9b 和从未改动的 startlux9b
中的大文件（safetensors 分片、`head.pt`、`tokenizer.json`）。execute 以退出码 1 结束，命令链里的 `set -e`
使随后的矩阵 preflight 没有运行。

原因在 huggingface_hub 1.32 引入的共享 blob 存储。用 Xet 下载的文件只在缓存根下存一份，位置是
`blobs/<xet[:2]>/<xet>`，仓库内的 `blobs/<etag>` 改为指向它的相对链接。节点上确认的一条链是
`snapshots/…/model-00001-of-00004.safetensors -> ../../blobs/6de4c4ab… -> ../../blobs/76/76981d8e…`，
只有最后一个文件是普通文件。该布局要求存储目录中有内容为 `1` 后接换行（字节 `31 0a`）的标记文件 `.huggingface-shared-blobs`，
节点上的标记和全部文件内容还需由修复后的 verify 确认。存档环境里的
huggingface_hub 1.33.0（vLLM）和 2.4.0（kev）写入规则相同。Xet 名是另一种哈希，与文件的 SHA-256 不同。
原验证器要求仓库 blob 本身是普通文件，所以拒绝了这些链接，16 个大文件也就还没有做内容哈希。

修复只改动 `verify_snapshot`，并新增 `hub_blob_payload`。仓库 blob 是普通文件时沿用原规则。仓库 blob 是链接时，
只接受 huggingface_hub 写出的那一种形式：目标字面上就是 `../../blobs/<两位十六进制>/<64 位小写十六进制>`，
两位前缀等于哈希开头，仓库目录名形如 `models--…`，存储带版本 1 标记，终点是普通文件，从缓存根往下经过的目录
都是真实目录。内容仍以仓库 blob 名（Hub ETag）为准，流式计算 SHA-256 或 Git blob SHA-1，Xet 名不参与比较。
其他缓存或任意目标、多余的链接跳转、悬空或循环链接、指向缓存外的目录链接都会被拒绝，报错末尾的括号里写明原因。
仓库自己的 `blobs` 目录如果是链接，现在也会被拒绝，原版本会跟随这种链接；huggingface_hub 不会创建这种目录。
verify 不写缓存，不重新下载，pin、TOOL 常量、恢复状态和阶段记录保持原样。

本机（Mac）验证：新增的 31 项测试与原有 43 项共 74 项通过；同一组新测试对原脚本运行时，共享缓存上的 CLI 演练
得到与节点同类的报错；5 个有意破坏的版本都被识别。16 个大文件在节点上的内容哈希和 GPU 预检要等节点复核，
步骤见第 8 节。Codex 独立重跑 74 项测试、执行 18 项缓存探针并复核 5 项破坏测试的失败原因后，接受本次修复进入节点验证，
详见[新增验收记录](../records/runtime-cache-layout-2026-11-11-acceptance.json)。
共享存储说明参见 [Hugging Face 官方缓存文档](https://huggingface.co/docs/hub/local-cache)。

## 恢复内容

- V1 运行时元数据：存档 `results/solo-v1/exports/benchmark_bundle.zip`（SHA-256 `72138989815c…`）中
  `runtime/` 下的 108 个成员，逐个按 `configs/runtime_recovery_2026_11_11_pins.json` 校验字节后硬链接到位，
  已有文件一律不覆盖。`session.json`、`last_services.json`（历史会话与 PID）不恢复，
  `selected_profiles.json`、`selected_memory.json` 在 V1 中本来就不存在，恢复后也保持不存在。
- 两个环境 `vllm-0.31.0` 与 `kev`：uv 管理的 CPython 3.12.8，安装命令沿用存档 `install.log` 中的原始命令，
  再加上存档冻结清单里的精确版本。只有在冻结的 `legacy_environment` 中由冻结的 `_verify_environment`
  算出的 `pip freeze --all` 哈希等于存档值时才算完成；不一致即记录差异并终止，永不改写目标版本。
- kev 与 StartLux-Decision 源码：按 V1 `_checkout` 的命令固定在原始提交和原始路径。
- CUDA 13.0 工具链：用矩阵检出中固定的 `scripts/prepare_cuda_toolkit.py` 安装，再用其 `validate_target`
  校验，并与存档收据比较 tree、wheel、发行版、安装器和需求文件的身份。
- 权重：Kev LoRA `139fdd94…` 与 Qwen3.5-4B-Base `1001bb4d…`（V1 缓存），Qwen3.5-9B `c2022362…`
  （新建的 `solo-v3/runtime-work`）。下载结果记录必须与存档记录完全相同。
- StartLux-9B（`aux-size-control-v1-r2` 下，19G）只读校验，不下载也不修改。

`startlux-reference` 环境、StartLux-4B 与 Qwen3.5-4B 权重不恢复，因为 direct-matrix 不读取它们；
`installations.json` 仍按原字节列出它们。工具报告的就绪状态是 `MATRIX_RUNTIME_READY`，只针对矩阵。

## 写入范围

工具自己的写入只发生在三处：新建的 `results/solo-v1/runtime/`、新建的 `results/solo-v3/runtime-work/`，以及
状态目录 `results/runtime-recovery-2026-11-11/`（状态、日志、收据、安装输入、失败证据、被保留的旧尝试）。
它不写矩阵输出 `results/aux-direct-matrix-v1/`、V3 与 size-control 的控制文件、其他 V1 文件，也不写任何检出。
uv 缓存和托管 Python 按 V1 的位置放在 `solo-v1/runtime/work/cache/uv` 与 `work/python`。

子进程照常使用系统临时目录和各自的用户缓存：uv、pip、setuptools 构建（存档日志里有 `Building llguidance==1.7.6`，
这类构建可能写 `~/.cargo` 等缓存）和 huggingface_hub 都可能如此，工具不对它们做沙箱隔离。
`verify` 只在 `TMPDIR` 下建一个临时检查目录；开始前会确认 `TMPDIR` 不在运行时根目录或任何检出之内，READY 时删除该目录，
NOT READY 时保留并在报告中给出路径。

## 0. 先记录磁盘与矩阵状态

这一步回答此前尚未提供的磁盘与 `last_error` 信息；`plan` 的输出也会包含磁盘余量、锁状态和矩阵输出的
`status.json` 摘要。

```bash
ROOT=/home/zhanj289/e6998/decision-study
df -h "$ROOT"
quota -s 2>/dev/null || true
python3 -m json.tool "$ROOT/results/aux-direct-matrix-v1/status.json" | head -60
```

磁盘估算（含 uv 缓存）：env-vllm 约 30 GiB，env-kev 约 20 GiB，toolkit 约 6 GiB，Kev 权重约 14 GiB，
Qwen3.5-9B 约 24 GiB，合计约 97 GiB。`statvfs` 看不到配额，配额请用 `quota` 自行确认。

## 1. 准备独立的恢复检出，矩阵检出保持 4b4e38a

矩阵检出 `/home/zhanj289/e6998/decision-study-src` 必须停在
`4b4e38ab9c040fe7eb02ae5075e87c595399d980` 且干净，因为矩阵输出的 `settings.json` 绑定了它。
恢复工具放在另一个检出里运行。发布分支是 `runtime-recovery-2026-11-11`；运行前将其 HEAD 与发布消息中的
完整提交 SHA 核对。发布消息提供固定提交的完整命令。

```bash
SRC=/home/zhanj289/e6998/decision-study-src
REC=/home/zhanj289/e6998/decision-study-recovery
git -C "$SRC" rev-parse HEAD          # 应为 4b4e38ab9c040fe7eb02ae5075e87c595399d980
git -C "$SRC" status --porcelain=v1 --untracked-files=all   # 应无输出
git clone --branch runtime-recovery-2026-11-11 --single-branch https://github.com/jzdscience/e6998_llm_genai_decision_study.git "$REC"
git -C "$REC" checkout --detach origin/runtime-recovery-2026-11-11
git -C "$REC" rev-parse HEAD
sha256sum "$REC/configs/runtime_recovery_2026_11_11_pins.json"
# 应为 7f3e27c095ead75991eef9a822695d211bb7fe649506687bead57806810c4e56
```

工具运行前会自己再查一遍矩阵检出：HEAD、`git status --porcelain=v1 --untracked-files=all`、
assume-unchanged/skip-worktree 标记，以及它导入或执行的 17 个文件的 SHA-256。

## 2. 通用变量

在公司 code-server 的计算节点终端运行，需要能访问 PyPI、GitHub 与 Hugging Face。工具不调用 GPU（子进程的
`CUDA_VISIBLE_DEVICES` 为空）；耗时可能达数小时，可放在当前 allocation 内的 `tmux` 或 `screen` 中，
但它们不能延长 allocation 的时限。

```bash
ROOT=/home/zhanj289/e6998/decision-study
SRC=/home/zhanj289/e6998/decision-study-src
REC=/home/zhanj289/e6998/decision-study-recovery
PY="$ROOT/.venv-controller/bin/python"
OUT="$HOME/runtime-recovery-2026-11-11"     # 输出 JSON 放在运行时根目录之外
mkdir -p "$OUT"
recover() {  # 用法: recover plan|execute|verify
  env -u PYTHONPATH -u PYTHONSTARTUP "$PY" -I -B "$REC/scripts/recover_node_runtime.py" "$1" --runtime-root "$ROOT" --matrix-checkout "$SRC" \
    --matrix-commit 4b4e38ab9c040fe7eb02ae5075e87c595399d980
}
```

`--uv` 默认是原始安装器 `$ROOT/.venv-controller/bin/uv`。`PYTHONPATH`、`PYTHONHOME`、`UV_*`（`UV_NATIVE_TLS`、
`UV_HTTP_TIMEOUT`、`UV_HTTP_RETRIES`、`UV_NO_PROGRESS` 除外）、`PIP_*`、`HF_ENDPOINT`、`HF_HUB_CACHE`、
`HF_HUB_OFFLINE`、`DECISION_STUDY_RUNTIME_DIR` 等会改变解析结果的变量一旦设置，工具会报阻断；按报告逐个 `unset`。
uv 以 `--no-config` 运行，不读取 `uv.toml` 或 `pyproject.toml` 中的 uv 配置。

## 3. plan（只读）

```bash
recover plan | tee "$OUT/plan.json"; echo "exit=${PIPESTATUS[0]}"
```

`READY_TO_EXECUTE` 退出码为 0，`BLOCKED` 为 1。`blockers` 列出所有阻断项，例如：存档哈希不符、根目录与存档绝对路径不一致、
原项目 `decision_study.egg-info`（0.1.0）缺失、四个研究输出（solo-v1、solo-v3、aux-size-control-v1-r2、
aux-direct-matrix-v1）的 `.active.lock` 缺失或被占用、V3/size-control 控制文件与 pin 不符、恢复目标在没有恢复状态时已经存在。
`.active.lock` 由 `common.exclusive_run` 创建且从不删除，用过的输出都应有它；缺失时工具拒绝执行，因为此时无法与研究命令互斥。

## 4. execute（唯一会写入的命令）

```bash
recover execute | tee "$OUT/execute-$(date -u +%Y%m%dT%H%M%SZ).json"; echo "exit=${PIPESTATUS[0]}"
```

执行期间持有上述四个 `.active.lock`（打开时不创建）和状态目录锁。阶段依次为 metadata、sources、env-vllm、toolkit、
env-kev、weights-kev、weights-qwen9b，每个阶段开始前重新核对 V1 元数据与 V3/size-control 控制文件的 pin，
每次安装源码前重新核对源码的 HEAD、每个跟踪文件的原始 blob 以及未跟踪和被忽略的文件。全部完成后自动运行一次完整 verify。

安装期间终端可能较长时间没有输出。可在另一个终端查看
`results/runtime-recovery-2026-11-11/state.json` 中的当前阶段，以及 `logs/` 下对应阶段的日志。

| 退出码 | 状态 | 处理 |
|---|---|---|
| 0 | `MATRIX_RUNTIME_READY` | 进入第 6 步 preflight |
| 1 | `MATRIX_RUNTIME_NOT_READY` | 阶段已完成但最终校验未过；把 `problems` 发回复核 |
| 2 | `FAILED` | 普通失败（网络、超时、进程组已确认停止）；排除原因后重新运行同一条 execute 即从中断处续跑 |
| 3 | `TERMINAL_MISMATCH` | 重建结果与存档身份不符（例如 freeze 哈希）；不要重试，把 `failures/` 下的差异发回 |
| 4 | `STOP_UNVERIFIED` | 某个子进程组无法确认已停止；先用 `ps -eo pid,pgid,stat,lstart,args` 检查残留进程，之后 execute 与 verify 都会拒绝继续，需人工复核 |
| 130 | `INTERRUPTED` | 被信号中断且子进程组已确认停止；重新运行 execute 续跑 |

续跑规则：中断的环境只在恢复自己创建的目录内用 `uv venv --clear` 重建；源码检出若跟踪内容被改动，原地保留并失败；
若是恢复自己创建的未完成检出或带有额外文件（例如旧的构建副本），整个检出移到
`results/runtime-recovery-2026-11-11/preserved/` 下再重新克隆，不删除任何内容。下载由 huggingface_hub 续传；
工具链安装器原子发布，失败的暂存目录保留在 `work/toolchains/` 下。

## 5. verify（只读）

```bash
recover verify | tee "$OUT/verify.json"; echo "exit=${PIPESTATUS[0]}"
```

`MATRIX_RUNTIME_READY`（退出码 0）表示以下内容全部通过：矩阵检出身份；恢复的 108 个元数据字节与 pin 一致、四个缺省记录不存在、
旁边没有多余文件；两个环境在冻结的 legacy 上下文中报告 3.12.8、头文件齐全且 freeze 哈希一致；工具链身份；两个源码检出的
HEAD 与跟踪内容；每个快照文件都链接到本仓库的 blob（普通文件，或按“11/11 节点结果”一节所述形式指向同一缓存共享存储的
链接），其内容的 SHA-256 或 Git blob SHA-1 等于 blob 名即 Hub ETag，分片齐全，StartLux-9B 另经冻结的 `_verify_model_metadata`；
V3/size-control 控制文件未变；冻结的 `direct_matrix.runtime.plans()` 对矩阵输出计算成功，若已有 `runtime/binding.json` 则二者相等；
已有 `settings.json` 通过冻结的 `study.settings(check_source=True)`。任一 pin 不符时，工具不会运行或读取由该文件推出的程序、
源码或快照路径，直接给出 NOT READY。

## 6. 用原始包装脚本做 preflight

READY 只说明文件与身份已恢复，GPU、CUDA 与模型行为仍未测试。在实际 GPU allocation 内，从未改动的矩阵检出运行：

```bash
cd "$SRC"
bash scripts/run_direct_matrix.sh "$ROOT" --preflight-only --retry-preflight
```

`--retry-preflight` 允许重试这次环境缺失导致的失败预检。只有在矩阵输出已存有 allocation 且当前 Slurm 作业确实不同的情况下才加
`--new-allocation`；包装脚本读取实际的 Slurm 剩余时间。
完成后把 `$ROOT/results/aux-direct-matrix-v1/exports/direct_matrix_bundle.zip` 交回复核。

## 7. 交回的材料

```bash
tar -czf "$OUT/runtime-recovery-state.tgz" -C "$ROOT/results" runtime-recovery-2026-11-11
ls -l "$OUT"
```

请交回 `$OUT` 下的 plan/execute/verify JSON 与上面的压缩包（状态、日志、收据、安装输入、失败证据，不含环境与权重）。

## 8. 共享 blob 修复后：在现有恢复检出中只读 verify

11/11 的 execute 已把七个阶段全部记录为 complete，因此继续时把现有恢复检出 `$REC` 更新到修复提交，再运行一次只读
`verify` 即可。矩阵检出保持 `4b4e38a`，恢复状态、环境、权重和缓存都保持原样。

```bash
ROOT=/home/zhanj289/e6998/decision-study
SRC=/home/zhanj289/e6998/decision-study-src
REC=/home/zhanj289/e6998/decision-study-recovery
OUT="$HOME/runtime-recovery-2026-11-11"
mkdir -p "$OUT"
git -C "$SRC" rev-parse HEAD                                  # 应为 4b4e38ab9c040fe7eb02ae5075e87c595399d980
git -C "$SRC" status --porcelain=v1 --untracked-files=all     # 应无输出
git -C "$REC" status --porcelain=v1 --untracked-files=all     # 应无输出
git -C "$REC" fetch origin runtime-recovery-2026-11-11
git -C "$REC" checkout --detach FIX_SHA                       # FIX_SHA 换成发布消息中的完整提交 SHA
git -C "$REC" rev-parse HEAD
sha256sum "$REC/configs/runtime_recovery_2026_11_11_pins.json"
# 仍应为 7f3e27c095ead75991eef9a822695d211bb7fe649506687bead57806810c4e56
env -u PYTHONPATH -u PYTHONSTARTUP "$ROOT/.venv-controller/bin/python" -I -B \
  "$REC/scripts/recover_node_runtime.py" verify --runtime-root "$ROOT" --matrix-checkout "$SRC" \
  --matrix-commit 4b4e38ab9c040fe7eb02ae5075e87c595399d980 | tee "$OUT/verify-shared-blobs.json"; echo "exit=${PIPESTATUS[0]}"
```

- verify 会对四个快照的全部文件重新做流式哈希，耗时取决于磁盘吞吐，可能需要数分钟。它只在 `TMPDIR` 下建临时检查目录，
  READY 时删除。报告里的 `tool.script_sha256` 应等于发布消息给出的值，`tool.pins_sha256` 仍为 `7f3e27c0…`。
- 退出码 0（`MATRIX_RUNTIME_READY`）后，在实际 GPU allocation 内按第 6 步从矩阵检出运行 preflight。
- 退出码 1 时把 JSON 发回。`blob content does not match its name` 表示该文件的字节与 ETag 不符；括号原因含
  `unsupported link`、`symlinked path component`、`missing or not a regular file` 或 `not a version-1` 时，表示节点缓存
  与 huggingface_hub 的共享布局有出入。两种情况都保留现场发回复核，缓存保持原样，不删除也不重新下载。

## 已知限制

- 节点 11/11 运行了 plan 与 execute：七个阶段完成，freeze 与工具链身份一致（见“11/11 节点结果”一节）。共享 blob 修复后的
  verify 与 GPU 预检还没有在节点运行。存档只记录了 Python 版本号，节点上 uv 为 3.12.8 选用的 python-build-standalone
  构建无法与原始构建比对。
- 工具链 tree 的逐字节相等只能在节点上确认；工具在 umask 027 下运行，与存档文件权限一致。
- 子进程清理以进程组为范围：用 `setsid` 或 `setpgid` 离开进程组的后代不在范围内。Linux 上通过 `/proc` 枚举进程组，
  这一分支只在 Linux 上运行，Mac 测试走的是 `ps` 分支。
- 恢复的元数据是 2026-11-02T21:07Z 导出时的状态，之后对 V1 运行时的改动无法恢复；direct-matrix 不读取这些改动。
