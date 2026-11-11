# 在公司节点补齐直接准确率矩阵

2026-11-10。规则见[协议](DIRECT_MATRIX_PROTOCOL_ZH.md)，本机软件与论文已通过
[Codex 独立验收](DIRECT_MATRIX_ACCEPTANCE_2026-11-10_ZH.md)。交付分支是 `matrix-completion-2026-11-10`；
使用交付消息中的完整 40 位 release SHA 固定节点版本。GPU 上 8 个新单元都还没有运行。

## 位置

| 位置 | 用途 |
|---|---|
| `/home/zhanj289/e6998/decision-study-src` | 独立的干净源码 checkout，固定在 release SHA；命令都在这里执行 |
| `/home/zhanj289/e6998/decision-study` | 原运行根，保持原位、不改写；读取其中的 `.venv-controller` 和四个已有结果目录 |
| `/home/zhanj289/e6998/decision-study/results/aux-direct-matrix-v1` | 本研究唯一写入的新输出目录 |

脚本默认读取运行根下的 `results/solo-v1`、`results/solo-v2`、`results/solo-v3` 和 `results/aux-size-control-v1-r2`，
全部只读。环境不重建，权重不下载：StartLux-9B 与 Qwen3.5-9B 用规模对照和 V3 已下载的快照，Kev-4B 用 V1 的 kev
环境、LoRA 和 Hugging Face 缓存（离线读取）。快照或缓存文件缺失时程序停止，不替换版本。新的编译缓存等运行文件
写在新输出目录的 `runtime-work/` 下。

## 运行前

1. Codex 已出具最终验收，维护者已提交并推送，交付消息给出 release SHA。
2. 节点源码 checkout 按 [NODE_GIT_ZH.md](NODE_GIT_ZH.md) 的“后续更新”取到这个 SHA。下面这段把那里的
   `STUDY_COMMIT` 固定为交付的 SHA，占位符没有替换或 SHA 不完整时直接停止：

   ```bash
   (
     set -eu
     RELEASE_SHA=REPLACE_WITH_THE_DELIVERED_40_CHARACTER_SHA
     case "$RELEASE_SHA" in *[!0-9a-f]*) echo "STOP: set RELEASE_SHA to the delivered SHA"; exit 2 ;; esac
     test "${#RELEASE_SHA}" -eq 40 || { echo "STOP: RELEASE_SHA must have 40 characters"; exit 2; }
     cd /home/zhanj289/e6998/decision-study-src
     test "$(git remote get-url origin)" = "https://github.com/jzdscience/e6998_llm_genai_decision_study.git" || {
       echo "STOP: origin differs; inspect this checkout."; exit 2; }
     test -z "$(git status --porcelain)" || { echo "STOP: local changes exist."; git status --short; exit 2; }
     git fetch origin
     git checkout --detach "$RELEASE_SHA"
     /home/zhanj289/e6998/decision-study/.venv-controller/bin/python scripts/prepare_node_checkout.py \
       --runtime-root /home/zhanj289/e6998/decision-study \
       --expected-commit "$RELEASE_SHA"
   )
   ```

3. 在已获分配的 Slurm job 内的 code-server 终端执行。GPU 上没有其他 compute 进程，也没有别的实验命令在跑。
   控制器每次启动模型前只读查询 GPU 进程，发现不属于本次部署的进程就停止，不向它发信号。
4. 整个研究期间不更新、不切换、不修改这个 checkout，换 allocation 恢复时也一样。settings 绑定了 commit 和源码
   哈希，checkout 一变，后续命令就拒绝继续。

## 默认运行：一条命令

```bash
cd /home/zhanj289/e6998/decision-study-src
bash scripts/run_direct_matrix.sh /home/zhanj289/e6998/decision-study
```

脚本先确认当前目录是干净的独立 checkout（不能是运行根），找到 `.venv-controller`，打印源码 commit，然后执行
`start`：

1. prepare：在 CPU 上准备精确输入，并用各自的核验函数读取 12 个历史单元；不读测试 gold。
2. init-runtime：绑定三个角色的运行控制和当前 allocation。剩余时间由 squeue 按 `SLURM_JOB_ID` 读取（job 必须是
   RUNNING，再减 5 分钟余量）；脚本不接受小时数变量，也不沿用旧 session 的截止时间。
3. preflight：三个角色各启动一次，内容见协议第 7 节。三个角色都 PASS 后才开始正式采集；任一角色 FAIL，命令就
   停止并导出诊断包，此时没有任何正式预测。
4. run：按固定顺序采集 8 个单元，共 5 次部署，每次部署结束都核验停止。
5. 采集完成后先导出一次（`COLLECTION BUNDLE`），再冻结、第一次读取 gold、评价，最后导出最终包（`FINAL BUNDLE`）。

成功时终端依次出现 `DIRECT_MATRIX_COMPLETE` 和 `DIRECT_MATRIX_FINISHED`。回传：

```text
/home/zhanj289/e6998/decision-study/results/aux-direct-matrix-v1/exports/direct_matrix_bundle.zip
```

包里有准备输入、清单、原始回答、收据和日志，不含权重、环境和缓存。研究完成后再次运行同一命令，只会核验已有
结果并重新导出，`evaluation.json` 保持原字节。

## 可选：只做预检

```bash
bash scripts/run_direct_matrix.sh /home/zhanj289/e6998/decision-study --preflight-only
```

执行 prepare、init-runtime 和 preflight，导出 `PREFLIGHT BUNDLE` 后打印 `PREFLIGHT_COMPLETE`，不做正式采集。
需要先请审核看兼容性时用它，例如 KV 容量、引擎参数、token 长度和开发集回答。之后的默认命令会核验并复用这些
PASS 收据；如果默认命令在另一个 Slurm job 中运行，要加 `--new-allocation`。

## 中断与恢复

先看终端末尾和诊断包，再选择下表的命令，都在同一个源码 checkout 中执行。恢复总是在同一个输出目录
`results/aux-direct-matrix-v1` 里继续，脚本固定使用这个路径；旧运行根和四个输入目录始终只读。只有下文列出的、需要
协议另作决定的情况才换新的输出目录；恢复也从不指向旧的运行目录。

| 情况 | 命令 |
|---|---|
| 同一个 job 内，某次尝试失败或被中断 | `bash scripts/run_direct_matrix.sh /home/zhanj289/e6998/decision-study --resume` |
| 换了新的 Slurm job | `bash scripts/run_direct_matrix.sh /home/zhanj289/e6998/decision-study --new-allocation` |
| 换了新的 Slurm job，且有失败或被中断的尝试 | `bash scripts/run_direct_matrix.sh /home/zhanj289/e6998/decision-study --new-allocation --resume` |
| 某个角色的预检因临时原因失败（偶发 I/O、端口） | 上面相应的命令再加 `--retry-preflight` |

- 存在未被接受的尝试时，普通重跑会被拒绝，并提示加 `--resume`。时间余量用完时正在采集的那次尝试也属于这种情况，
  所以在新 job 里通常要用 `--new-allocation --resume`。`--resume` 为每个这样的尝试写下 `recovery.json`：已写入
  分片的行保留，已 fsync 进日志的行先复制备份、再由原 `PredictionStore` 折叠进分片，这些输入都不会再请求一次。
  只有剩下的输入在新部署中继续采集。
- `--new-allocation` 只用于真正不同的 Slurm job。在同一个 job 里加这个开关会被拒绝；它从不延长旧截止时间，旧
  session 记录保存在 `runtime/session-history/`。
- 换 job 后，控制器会被动复查上一次未确认停止的部署，不发信号、不取消任何 job。在同一主机上，进程表里不能再有
  记录的进程组成员；在另一台主机上，必须由 `sacct` 证明上一个 job 已处于终止状态（COMPLETED、CANCELLED、
  TIMEOUT、FAILED 等）。job 仍在运行、查不到或记账不可用时命令停止：等旧 job 结束，或由它的所有者正常结束，
  然后重跑。
- 尝试中有在 watchdog 截止之后才记录的行，这些行不能被接受，程序停止并保留现场，需要按协议另作决定。
- 预检因 OOM、KV 不足或引擎退出而失败时不要重试，需要新的协议修订和新的输出目录；当前输出原样保留。
- 两次 allocation 之间 GPU 驱动若有变化，有效配置随之改变，控制器停止，同样需要新的输出目录。
- 已有 `evaluation-access.json` 而缺少 `evaluation.json` 时程序停止，保留现场等待检查。

## 单独查看状态或导出

脚本只封装默认运行和 `--preflight-only`。没有其他命令在使用这个输出目录时，可以执行：

```bash
cd /home/zhanj289/e6998/decision-study-src
PY=/home/zhanj289/e6998/decision-study/.venv-controller/bin/python
OUT=/home/zhanj289/e6998/decision-study/results/aux-direct-matrix-v1
"$PY" -m decision_study.direct_matrix status --output-dir "$OUT"
"$PY" -m decision_study.direct_matrix export --output-dir "$OUT"
```

`status` 只读，核验已有收据，不读 gold，不访问模型、GPU 或调度器，发现问题时返回 1。`export` 导出当前状态，
可能只含部分进度。

## 失败时回传什么

控制器开始执行后出现的 `BLOCKED` 会写 `diagnostics/last_error.json`，并尝试导出诊断包，回传
`exports/direct_matrix_bundle.zip` 和终端最后几十行即可。脚本自身的检查（checkout 不干净、在运行根执行、找不到
`.venv-controller`）和 `BLOCKED before execution`（例如另一个命令正占用这个输出目录）不导出诊断包，按提示处理后
重跑。
不要删除、移动或修改输出目录中的任何文件。

## 回传后在 Mac 上的验收

1. 计算回传 ZIP 的 SHA-256，按 SHA 命名归档到外部证据根的 `records/bundles/`，登记进
   `records/BUNDLE_INDEX.json`，状态写为待验收。
2. 在与节点相同 commit 的 checkout 中运行离线审计（命令见协议第 11 节），解压到新目录，报告写在该目录之外。
3. 这个审计重放的是生产代码的计算，Codex 再用独立脚本复算。
4. 两项都通过之后，才把 8 个新值写进论文 Table 2 和各处文档。

## 本机验证范围

本机只做了 CPU 上的测试、注入错误和 CLI 演练，具体见协议第 13 节；它们使用假模型服务，标签永远不是结果。真实
模型行为、0.8 份额加 `--max-num-seqs 64` 的 9B 引擎启动、Kev 的离线加载与内核编译、真实 token 长度、每个单元
的耗时以及 `sacct` 的可用性，都要由节点预检和正式运行回答。
