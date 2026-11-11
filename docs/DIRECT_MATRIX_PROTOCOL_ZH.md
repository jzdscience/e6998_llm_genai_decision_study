# 直接准确率矩阵补全协议（direct-matrix-v1）

2026-11-10。Ju 当天授权补齐五个 LLM 的直接准确率矩阵（Codex 建议，Claude Code 实现）。机器可读版本是
`configs/direct_matrix_protocol.json`（revision 1，`status: frozen_before_collection`），实现在
`decision_study/direct_matrix/`，节点入口是 `scripts/run_direct_matrix.sh`，运行步骤见
[RUN_DIRECT_MATRIX_ZH.md](RUN_DIRECT_MATRIX_ZH.md)。

当前状态：软件、测试、论文与运行说明已通过 [Codex 本机验收](DIRECT_MATRIX_ACCEPTANCE_2026-11-10_ZH.md)，
通过 `matrix-completion-2026-11-10` 分支交付；节点使用交付消息中的固定 SHA。
节点 GPU 上 8 个新单元都没有运行（0/8），没有任何新的模型准确率。本文只描述已经实现的规则。

## 1. 范围

论文 Table 2 把 Kev-4B、StartLux-4B、StartLux-9B、Qwen3.5-4B、Qwen3.5-9B 放在四个评价集上。20 格中 12 格有已验收
结果，另外 8 格从未测量，表中记为 NM。本研究只补这 8 格：沿用原模型、提示与标签，在固定测试输入上逐条串行预测，
再和 12 个历史单元一起评分。

全部正式输入都属于早先研究已经评分过的测试集，所以这是回顾性扩展。新单元只作描述，不能确认任何政策、路由收益或
一般优越性；Study 1–3 原有的主比较和主/次要标记保持不变。本研究不拟合、不调参、不校准、不做路由或模型选择，
不测服务容量，不训练 CPU 分类器，也不引入新数据。

## 2. 20 个单元

| 评价集（n） | Kev-4B | StartLux-4B | StartLux-9B | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|---|---|---|
| CLINC main test（600） | 历史 V1 | 历史 V1 | 新，第 1 组 | 历史 V1 | 新，第 1 组 |
| SQuAD 2.0 S1 test（2,000） | 历史 V1 | 历史 V1 | 新，第 3 组 | 历史 V1 | 新，第 3 组 |
| SQuAD 2.0 S2 holdout（4,000） | 新，第 2 组 | 历史 V2 | 新，第 1 组 | 历史 V2 | 新，第 1 组 |
| ANLI test（3,200） | 新，第 2 组 | 历史 V3 | 历史规模对照 | 历史 V3 | 历史 V3 |

两个 SQuAD 列都是 SQuAD 2.0 可答性二分类。S1 是 Study 1 的测试集，取自官方 dev 的 35 篇文章、2,000 题；S2 是
Study 2 的 holdout，取自此前未用过的 57 篇 train 文章、4,000 题。两列对应两个研究划分，与 SQuAD v1/v2 版本无关。

8 个新单元合计 20,400 条测试预测。第 1 组 9,200 条（两个 9B 角色 × CLINC 600 与 S2 4,000），第 2 组 7,200 条
（Kev-4B × S2 4,000 与 ANLI 3,200），第 3 组 4,000 条（两个 9B 角色 × S1 2,000）。12 个历史单元的 28,600 条预测
原样复用，矩阵完成后共 20 个单元、49,000 条测试预测，其中新采集的是 20,400 条。

## 3. 采集顺序

顺序在任何新结果出现前固定，之后不按结果调整。同一角色的相邻单元共用一次部署，正式采集共启动 5 次模型：

1. StartLux-9B：CLINC 600，然后 S2 4,000
2. Qwen3.5-9B：CLINC 600，然后 S2 4,000
3. Kev-4B：S2 4,000，然后 ANLI 3,200
4. StartLux-9B：S1 2,000
5. Qwen3.5-9B：S1 2,000

预检阶段另外每个角色启动一次，共 3 次。

## 4. 输入与开发集

正式输入直接读取已验收的准备文件，先核对 SHA-256：CLINC 与 S1 来自 V1 的 `data/examples.jsonl`（`a0b699e5…`），
S2 来自 V2（`f29fac85…`），ANLI 来自 V3（`a470ad01…`）。prepare 按 task/split 选出各组后核对条数、排序后的
sample ID 指纹、整行指纹和标签顺序，带 gold 类字段的行一律拒绝。它不调用 V1 的数据加载器，因为那个加载器会对
`data/gold.json` 求哈希。准备出的 `data/formal/*.jsonl` 是规范化重新序列化的文件，保留全部原始值和任务文本；
`data/source_lines.json` 记录每行的原始行号与原始行字节的 SHA-256。所以这些文件与原始行的字节并不相同。

开发集沿用既有设置：V1 CLINC 64 条、V1 SQuAD 64 条（V2 的 smoke 也用这组）、V3 ANLI 90 条。prepare 检查它们的
sample ID 和 group ID 与每个正式划分都不相交，任务说明、criteria 和标签与正式输入一致。开发集的回答只在两遍之间
比较可重复性，从不和标签比较。

## 5. 模型、提示与运行设置

模型、提示、标签、温度和接口全部沿用已验收的设置，不在任何 holdout 上调整。

| 角色 | 检查点 | 只读复用的来源 | 运行方式 |
|---|---|---|---|
| StartLux-9B | StartLux-Decision-v1.1-9B-Instant `18a17f73…`，原生温度 1.1014 | 规模对照 `aux-size-control-v1-r2` 的角色 lock 与快照，V1 的 vLLM 0.31 环境 | 新的单模型 vLLM 引擎，显存份额 0.8，`--max-num-seqs 64`；think_mode=never，none_check 与 allow_none 为 false |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` `c2022362…` | V3 `solo-v3` 的角色 lock 与快照，同一 vLLM 环境 | 新的单模型 vLLM 引擎，0.8，`--max-num-seqs 64`；温度 0，seed 6998，关闭 thinking，JSON schema 的 label 字段限定为保存的标签，最多 64 个输出 token |
| Kev-4B | `jaredpalmer/kev-4b` `139fdd94…`，base Qwen3.5-4B-Base `1001bb4d…`，代码 `2074cc44…`，原生温度 2.406050072164233 | V1 `solo-v1` 的 kev 环境、LoRA 与 base 快照 | V1 已验收的 PyTorch primary profile（合并 LoRA、fused kernel、CUDA graphs），经 V1 原启动器；base 从 V1 的 Hugging Face 缓存以 `HF_HUB_OFFLINE=1` 读取 |

- 请求体由 V1 的 `decision_request` 与 `qwen_request` 生成，criteria 按保存的标签顺序排列。Kev 与 StartLux 的
  SystemOne choice 请求由同一份保存状态、说明、criteria 和标签顺序构成，StartLux 另加上表中的三个字段，与 V1 相同。
  两者返回完整标签分布（Kev 的概率保留四位小数），Qwen 只返回一个受约束的标签。
- 上下文上限 8,192 token，输出预留沿用 `scripts/model_probe.py`：Kev 0、StartLux 1、Qwen 64。输入一律不截断，
  超长就停止。
- 每个角色的有效运行配置必须与同一检查点此前那次已验收运行逐项相同，GPU 条目除外：Kev 对照 V1，Qwen3.5-9B
  对照 V3，StartLux-9B 对照规模对照，三份配置的哈希固定在协议里。GPU 必须是一张至少 44,000 MiB 的 L40S。
- 0.8 是 V3 和规模对照单独运行 9B 时的份额，那两次没有序列上限。本研究保留 0.8，再加上服务研究为这类混合
  Mamba 模型验证过的 `--max-num-seqs 64`。请求串行发送，这个上限不会被用满；它写进每个单元的配置。已有的配置
  文件都不改。
- Kev 的 API（`kev/api.py`，`2074cc44`）接受 1–255 个命名 criteria，ANLI 的 3 标签和 SQuAD 的 2 标签问题都在
  schema 范围内。真实渲染长度与 GPU 行为由节点的 token 检查和预检确认。

## 6. 采集规则

- 并发 1，每条输入最多 2 次尝试，超时 120 秒，重试间隔 1 秒。请求经 V1 的 `ModelClient` 与 `predict_examples`
  写入每个单元自己的 V1 `PredictionStore`（`cells/<单元>/`）。
- 每行记录单元、部署、保存输入的指纹（`example_sha256`）和实际请求体的指纹（`request_sha256`）。成功行必须能从
  原始响应重新解析出同一预测。
- 两次尝试都失败的输入记为终止错误，留在完整分母里按答错计，不重试也不替换。沿用 V1 规则：前 50 条之后终止
  错误超过 1% 时该单元停止，等待检查。
- 每个单元在同一部署内先把它的开发集串行回答两遍，用 V1 的可重复性阈值核对：argmax 一致率至少 0.98；概率模型
  的平均最大概率差不超过 0.01、p95 不超过 0.05。开发集出现失败或未过阈值时该单元停止。这一步不读标签。
- 每个采集阶段在自有子进程中运行，受父进程的绝对截止时间约束。父进程只在子进程按时以 0 退出后写
  `accepted.json`；其他情况下该次尝试保留现场，只有经显式 `--resume` 核对并写下 `recovery.json` 后才能计入。
- allocation 剩余不足 1,800 秒时不启动新部署；截止前 900 秒起不再发新请求；watchdog 在截止前 600 秒结束子进程。
  三个值是保守设定，真实单元耗时要到节点才知道。
- 单元完成需要同时满足：每条输入都有终止行；每个贡献过行的部署都有经核验的停止记录；每次尝试都已接受或已经
  `--resume` 核对；smoke 通过。

## 7. 节点预检

每个角色一次部署，依次执行：

1. tokenizer 报告：用官方渲染器为该角色的全部正式与开发输入计数，输入加预留不超过 8,192，不截断。
2. health 与身份检查。
3. 把该角色服务的每个开发集串行回答一遍。
4. 每个单元按精确 token 数取最长的 3 条正式输入，只检查回答格式。这些回答作为预检证据保存，不算正式预测，
   也不评分。
5. 再查一次 health，然后停止部署并核验没有残留进程。

PASS 收据每次被复用时，都会按冻结输入和 token 收据重建应有的阶段与 sample ID，并核对其中列出的全部原始文件。
FAIL 的证据原样保留；只有确认是临时原因时才用 `--retry-preflight` 重试。

## 8. 复用的 12 个历史单元

| 单元 | 来源 | 已验收正确数 |
|---|---|---|
| Kev-4B / StartLux-4B / Qwen3.5-4B × CLINC | V1 | 465 / 561 / 528（n = 600） |
| Kev-4B / StartLux-4B / Qwen3.5-4B × S1 | V1 | 1,627 / 1,825 / 1,720（n = 2,000） |
| StartLux-4B / Qwen3.5-4B × S2 | V2 | 3,671 / 3,438（n = 4,000） |
| StartLux-4B / Qwen3.5-4B / Qwen3.5-9B × ANLI | V3 | 2,374 / 1,773 / 2,119（n = 3,200） |
| StartLux-9B × ANLI | 规模对照 | 2,454（n = 3,200） |

每个历史单元由原研究自己的核验函数读取：V1 用固定的配置与预测哈希，并逐行从原始响应重新解析；V2 用
`v2.study._job_predictions`；V3 与规模对照用各自的 `verified_job`。读取后再对照协议里固定的整行哈希和逐 ID 精简
哈希。精简哈希固定了每个 ID 的状态与预测，单个标签被改动时，即使总正确数不变也会被拒绝。评价时必须精确重现
上表的正确数，否则停止。这些核验函数只读输入、预测、收据和 V2/V3 的 calibration/validation 标签，不读测试 gold。

## 9. 冻结与 gold

8 个单元全部完成且停止核验通过后才写 `freeze.json`，它绑定 8 份完成收据和 12 个历史单元。接着写
`evaluation-access.json`，之后才第一次读取测试 gold。gold 文件必须符合协议固定的 SHA-256，逐 ID 子集还要符合
固定指纹。评价时把子集导出为 `sealed/<评价集>-gold.json`，离线审计只凭回传包就能重新评分。冻结之后才记录的预测
一律拒绝。

## 10. 评价与统计

- 准确率 = 正确数 / n，n 是固定的完整分母，终止错误按答错计。缺少任何一条预测时评价停止，缺失的单元从不按 0 计。
- 每个评价集内 5 个模型两两比较，在相同 sample ID 上计算配对准确率差，共 10 × 4 = 40 个区间。方法沿用 V2/V3：
  配对百分位整群 bootstrap，2,000 次，种子 6998，每次抽样计算按题加权的合并差值。整群单位是 CLINC 的每条输入
  （600 个）、SQuAD 的文章（S1 35 篇，S2 57 篇）和 ANLI 的规范化 premise（2,854 个）。
- Study 1 的 CLINC 区间在类别内分层重采样，本研究按单条输入重采样，所以矩阵给出的 CLINC 区间可能与论文现有的
  Study 1 区间略有差别。论文中已报告的数字以原研究为准。
- 40 个区间都是描述性的，不做多重比较校正。即使各模型之间没有真实差异，也预计约有 2 个区间因偶然排除 0。
  其中没有主比较或确认性比较。
- 结果不支持等效性、架构因果、路由收益、服务性能或一般优越性方面的结论，因为这些输入此前都已被查看过。

## 11. 离线验收

回传包可以在 Mac 上只凭包本身验收，不访问节点路径、模型快照、Slurm 或网络。审计要在与节点相同 commit 的
checkout 中运行，因为 settings 绑定了相关源码文件的哈希，文件不同就会拒绝：

```bash
.venv-lab/bin/python -m decision_study.direct_matrix audit \
  --bundle /path/to/direct_matrix_bundle.zip \
  --output-dir /path/to/new-extraction-dir \
  --report /path/outside/that/dir/direct-matrix-audit.json
```

`--output-dir` 必须是尚不存在的新目录。命令先检查 ZIP 成员：没有绝对路径、`..` 或符号链接，没有重复，成员集合
与 `bundle_manifest.json` 一致。逐个核对 SHA-256 和字节数后解压，然后依次核验 settings 的协议哈希与源码哈希、准备
输入、导出的运行控制与固定的已验收配置、原始 tokenizer 报告与预检阶段、8 份完成收据（从行、smoke、尝试接受与
停止证据重算）、freeze 与 access 收据，最后用导出的 gold 子集重算 20 个单元和 40 个区间并与 `evaluation.json`
比较。`--report` 必须写在被审计目录之外。

这个审计调用生产代码自己的评分和统计函数，作用是在回传证据上重放同一套计算。独立算法的复算仍由 Codex 按以往
惯例另做。

## 12. 与论文的关系

节点结果回传并通过验收之前，论文 Table 2 的 8 格保持 NM。验收之后，8 个新值以回顾性标记写入 Table 2，12 个历史
值不变，40 个描述性区间可以放进附录；Study 1–3 的主比较与主/次要标记不变。

## 13. 本机已验证与未验证

本机验证（Mac，无 GPU，记录在 `.validation/matrix-completion-2026-11-10/claude-handoff.md`）：

1. `tests/test_direct_matrix.py` 的 43 项在工作树上通过，用时 222.85 秒；未变异的隔离副本上同样 43 项通过，
   用时 243.88 秒。之后补强的测试中，前 5 项针对性测试（含完整的中途 SIGKILL CLI 演练）通过，用时 110 秒。
2. 17 处单一守卫的注入错误各在隔离副本中运行，全部按预期原因失败，逐项日志在同目录的 `mutation/`。
3. CLI 演练在真实 V1/V2/V3/规模对照证据上依次执行 prepare、init-runtime、preflight、start（第 53 条正式行之后
   SIGKILL）、被拒绝的普通重跑、`--new-allocation`、`--resume`、全部 8 个单元、冻结、评价、导出和离线审计。被
   替换的只有服务启动、nvidia-smi、硬件 doctor、CUDA 工具链检查、tokenizer 运行、节点上的快照元数据读取、旧工作
   目录作用域和模型 HTTP 服务。演练中的标签来自假服务，永远不是结果。
4. Codex 的独立审阅提出 M1–M6、S1、S2。M1–M5、S1、S2 在其复查的版本中关闭，M6（`--resume` 时折叠已写入
   日志的行）的修复通过了针对性探针。
5. 全仓 pytest 第一次运行为 505 passed、11 skipped、8 failed。8 个失败都是演练输出被放在主 checkout 内，触发了
   生产代码的目录重叠保护；保护逻辑正确，夹具已改到临时目录。之后实现会话在最终代码上单独重跑全仓：514 passed、
   11 skipped、64 subtests，exit 0，用时 346.76 秒，含全部 44 项 direct-matrix 测试（同目录 `final-pytest.log`）。
   11 个 skip 是原有的可选证据与可选依赖跳过。

尚未验证、要由节点预检和正式运行回答的事项：真实模型行为；两个 9B 引擎在 0.8 份额加 `--max-num-seqs 64` 下能否
启动；Kev 经 V1 缓存离线加载 base 以及内核编译；真实 token 长度、回答格式、延迟和每个单元的耗时；`sacct` 能否用于
跨节点的停止核验。两次 allocation 之间若 GPU 驱动变化，有效配置随之改变，控制器会停止，需要新的输出目录。
