# 一条命令运行 StartLux 4B→9B 在线预算与服务研究

2026-11-16 软件交付稿。协议为 `anli-budget-serving-v1` revision 1，设计与数值以
[协议说明](BUDGET_SERVING_PROTOCOL_ZH.md) 和 [budget_serving_protocol.json](../configs/budget_serving_protocol.json) 为准。
本机只做过无 GPU 的模拟测试，节点同驻留、真实延迟和正式结果都还没有发生。

## 运行与续跑

在节点上按交付消息里的完整提交 SHA 固定一个独立源码 checkout，再在实际 L40S allocation 内运行：

```bash
cd /home/zhanj289/e6998/<budget-serving 源码目录>
bash scripts/run_budget_serving.sh /home/zhanj289/e6998/decision-study
```

第一个参数是旧运行根，包装脚本拒绝把源码目录放在运行根里面。控制器解释器固定为
`<运行根>/.venv-controller/bin/python`；脚本清除 `PYTHONHOME`、`PYTHONSTARTUP`、用户 site、
`HOURS_LEFT` 与任何 `DECISION_STUDY_DEADLINE_EPOCH`，把 `PYTHONPATH` 设为本 checkout，并先确认
`decision_study` 确实从这个 checkout 导入。剩余时间只从当前 Slurm job 读取，没有手填小时数的入口。

新输出根默认是：

```text
/home/zhanj289/e6998/decision-study/results/aux-anli-budget-serving-v1
```

程序依次完成以下步骤，成功的门槛自动进入下一步：

1. 核对 Study 4 固定文件的 SHA（含 gold 的字节哈希，此时不解析标签），从 `data/examples.jsonl`
   取 1,737 条 eligible-dev，核对 membership 与 1,540 个 premise 组，取 `startlux9b/n18000/front`
   的冻结 head 与阈值 0.027109315183045934，并复核历史回归值 87/900 与 438/1,737。
   开发输入用 `learnability_profile.study.inputs` 读取的 90 条 V3 train 来源行。
2. 写入不可变的 `study.json`、输入副本、12 份 Poisson/完整循环轨迹、随机条件的预置 Bernoulli(0.10) 标记，
   以及 60 个单元的固定顺序；随后由 runtime 生成四个运行控制目录。
3. 读取实际 Slurm job 与剩余时间，运行原生进程归属自检（每个 allocation 一次）。
4. 一个受监督 worker 为 direct4b、direct9b、pair4b、pair9b 四个控制各生成一份原始 tokenizer 报告，
   覆盖 1,737 + 90 条输入且无截断。
5. 三个部署预检：direct4b、direct9b、pair（4B 0.40 + 9B 0.50）。每个部署 90 条串行开发输入、
   30 秒 10 req/s 并发检查（pair 每条强制两个角色，标为 `preflight_forced_both`），再对该部署的
   条件各跑 10 秒 10 req/s 的真实政策烟测（pair 依次为 threshold、budget、random），每次烟测使用新账本。
   通过标准只看工程有效性：回答有效、无错误与超时、预算不变量成立、负载后健康、停止已核验。
   pair 另要求两个角色的 HTTP 调用在客户端观测到的 in-flight 时间区间有重叠，这一项只说明请求同时在途，
   GPU 内核是否并行没有观测。准确率与 goodput 都不作通行标准。
6. 60 个正式单元按固定顺序执行。每个单元重新启动服务、串行预热 30 条开发输入（pair 两个角色都调用）、
   用新账本测量 120 秒窗口并最多 drain 60 秒，然后停止服务并由父进程核验归属与 GPU 清理。
7. 60 个单元全部接受后写 `freeze.json`，再写绑定该 freeze 的 gold 访问收据，按 1,737 个 ID 读取 gold，
   并把固定 SHA 的 Study 4 gold 原字节保存为 `evaluation/source-gold.json`；随后计算评价、
   从原始证据重新审计、生成 `REPORT.md`。
8. 调用可选的 embedding 补存（见下文），再导出结果包，最后写一份 `exports/TRANSFER-*.json`
   列出需要下载的全部文件及其 SHA。

屏幕最后出现 `STUDY_COMPLETE`、`RESULT BUNDLE: ...` 和 `TRANSFER MANIFEST: ...` 才表示整个研究完成。

每个请求的 30 秒共享期限覆盖整个 offer：模型阶段按时返回、但整条 offer 在期限之后才完成时，记为
`offer_after_deadline`，原始阶段回答保留为证据。500/1000 ms 的 goodput 本来就按完整到达-完成时间计算。

`STUDY_CHECKPOINT` 表示剩余时间不足以安全开始下一个 attempt，已保存进度并导出 `CHECKPOINT BUNDLE`。
开启真正的新 allocation 后运行同一条命令即可。程序会确认 job ID 确实不同、用 `sacct` 证明旧 job 已终止，
再重新做本 allocation 的进程归属自检并继续。同一个 job 内的中断 attempt 不会被信号或替换，程序会拒绝继续，
需要等该 allocation 结束后在新 allocation 续跑。

真正的技术失败显示 `BLOCKED` 和 `DIAGNOSTIC BUNDLE: ...`，失败 attempt 的全部证据原样保留。
检查诊断后，若判断可以在同一科学身份下重试，显式加参数：

```bash
bash scripts/run_budget_serving.sh /home/zhanj289/e6998/decision-study --retry-failed
```

重试会新建 `attempt-000N`，旧 attempt 继续留在原处，接受收据里记录全部先前 attempt 及其状态。
开始新 attempt 前，失败 attempt 必须已有有效的父进程停止收据；缺少时只能在旧 job 结束后的新 allocation
中凭终止证明续跑。重试不会放宽任何哈希、输入、gold 时序或停止检查。源码、协议或运行根变化后，程序要求换新的输出根
（例如 `--output-dir .../aux-anli-budget-serving-v1-r2`），原输出保持不动。

进度可随时只读查看：

```bash
<运行根>/.venv-controller/bin/python -B -m decision_study.budget_serving status \
  --output-dir /home/zhanj289/e6998/decision-study/results/aux-anli-budget-serving-v1
```

## 每个 attempt 的时间上限

上限按协议公式计算，角色数×1,200 + 串行条数×30 + 全部窗口与 drain + 5 秒 grace + 180 秒清理保留。
启动一个 attempt 前，allocation 剩余时间须不少于“上限 + 180 秒”，否则保存 checkpoint。
这些数值是最坏情况的安全上限，实际耗时以节点记录为准。

| attempt | 上限（秒） |
|---|---:|
| tokens（4 份报告，每份按 1,200 秒计） | 4,985 |
| direct4b / direct9b 预检 | 4,245 |
| pair 预检 | 5,585 |
| direct 正式单元 | 2,465 |
| pair 正式单元 | 3,665 |

每份 tokenizer 报告的上限固定为 1,200 秒（代码常数 `protocol.TOKEN_REPORT_SECONDS`），四份合计再加 5 秒 grace 和 180 秒清理保留。

## 证据位置

```text
study.json                       协议、源码身份、输入哈希、固定顺序与种子规则
inputs/                          1,737 条 workload、90 条开发输入、冻结 policy 文档、来源绑定
schedules/ flags/                每个 repeat/rate 的共同轨迹与 random 预置标记
preflight/tokens.json            token 接受收据；preflight/<部署>.json 为 PASS 收据
stages/<阶段>/attempt-NNNN/      token 与预检 worker 的计划、原始行、账本、服务记录、停止收据
cells/<单元>/attempt-NNNN/       traces.jsonl、ledger.jsonl、fallback.jsonl、warmup、资源采样、服务与停止收据
cells/<单元>/accepted.json       从原始证据重新推导出的接受收据
freeze.json                      gold 解析前全部证据的哈希
evaluation/ audit.json REPORT.md gold 访问收据、Study 4 gold 原字节、1,737 条标签子集、评价、审计与报告
supplements/                     可选 embedding 补存的调用记录（状态、命令、文件 SHA）
exports/                         结果包或分片、TRANSFER 清单、learnability-embeddings/ 补存目录
```

每个 offer 在 trace 中恰好一行，被客户端拒绝、失败、超时的 offer 都保留。账本按单调序号记录 offer、
reserve、deny；fallback.jsonl 在客户端真正开始 9B HTTP 的时刻记一次尝试，已扣许可但期限先到、没有发出
HTTP 的情况则在客户端返回后记为未使用。报告分别给出许可数（reservation）、进入后级的次数（invocation）
和实际 9B HTTP 尝试数，direct9b 的 9B 调用也按实际尝试计数。
审计核对因果次序：许可在本 offer 的 4B 回答之后，9B 开始在许可和尝试记录之后。
任何已存收据在复用、freeze 和审计时都要由原始证据重新推导出完全相同的内容，单独改写哈希无法通过。

## 回传验收

结果包包含原始 trace、账本、阶段记录、运行控制与 token 报告、生命周期与停止收据、全部日志、freeze 和评价；
freeze 覆盖的文件全部打包，不按大小省略。权重、环境和 `runtime-work` 缓存不进包。
压缩包超过 48 MiB 时拆成 `.partNNNN` 字节分片，屏幕上的主路径是 `*.parts.json` 清单。
`join` 只接受规范命名、按编号排列、大小符合拆分规则的普通文件，先写临时文件，核对整体 SHA 后才生成目标，
失败不会留下半成品。拼回后在安全的新目录解压审计：

```bash
python3 -B -m decision_study.budget_serving join --manifest /path/budget_serving_result-XXXX.zip.parts.json \
  --target /path/budget_serving_result-XXXX.zip
python3 -B -m decision_study.budget_serving audit --output-dir /absolute/path/to/extracted-bundle
```

审计必须在 `study.json` 记录的同一发布 checkout 中运行：源码闭包、全部 configs 与协议文件要和记录逐字节一致，
工作负载、冻结 policy 单元和 policy 要与固定的 Study 4 指纹一致，轨迹和随机标记要能由种子重新生成。
随后重新核对 freeze、每个单元的计划、监督与停止收据、部署命令、预热、原始回答（含失败路径）、端点与角色、
实时 gain 重算、账本重放、fallback 日志与因果次序、汇总与全部指标。gold 子集由包内保存的 Study 4 gold
原字节按固定 SHA 重新推导；`--gold-source` 可另外与节点外的原文件比对。审计不重新观察节点进程或模型字节。

## 可选的 Study 4 embedding 补存

研究完成、导出结果包之前，程序以 `python -I -B scripts/archive_learnability_embeddings.py` 把 176 份
Study 4 embedding 补存到 `<输出根>/exports/learnability-embeddings/`，按助手实际输出的 JSON 显示
`PASS_ARCHIVED_ONLY` 或 `INCOMPLETE`，调用记录写入 `supplements/` 并随结果包导出；补存分片本身留在
`exports/` 下，由 TRANSFER 清单列出。补存只保存原 embedding 字节，完整 text-head 重拟合尚未进行。
补存缺失或失败不改变研究结果与退出码。

## 实验前固定的执行约定

以下约定在正式运行前确定，代码按此实现，运行中不再调整。

- pair 预检通过 runtime 的 `threshold` 条件启动服务，三个 pair 条件共用 pair4b/pair9b 控制；
  烟测在同一部署内依次跑 threshold、budget、random 三条政策，各用新账本。
- token 报告共四份，每个运行控制一份，因为各控制的有效配置哈希不同；每份上限 1,200 秒。
- 种子推导（协议 `seed` 为 20261116）：
  - 正式轨迹：`service_study.measurement.make_schedule(workload, rate, 120, 20261116 + 1000×repeat + rate)`，
    到达间隔用这个整数种子，完整循环的样本打乱用它加 1,000,000；
  - 预检并发轨迹 `make_schedule(development, 10, 30, 20261116 + 424242)`，烟测轨迹
    `make_schedule(development, 10, 10, 20261116 + 434343)`；
  - 顺序：`random.Random("20261116:rates:<repeat>")` 打乱负载，`random.Random("20261116:conditions:<repeat>:<rate>")`
    打乱五个条件；
  - random 标记：`random.Random("20261116:random-candidates:r<repeat>-rate<rate>")` 按 offer 编号逐个取
    `rng.random() < 0.1`，烟测用 `20261116:random-candidates:preflight-smoke`。这条字符串种子流与整数轨迹种子分开，
    在测量前写入 `flags/`，与标签、分数和完成次序无关。
- `router_input`：4B 回答通过冻结 adapters 解析、但冻结 `front_features` 拒绝打分时，控制器把该 offer 记为
  `router_input` 错误，不申请许可、不调用 9B，原始 4B 回答保留在 trace 中。
- HTTP 200 正文无法使用（JSON `null`、非 JSON、非 UTF-8、超过 1 MiB、答案字段不合格）时，阶段记为
  `invalid_response` 并写明 `phase`（`answer`、`json`、`utf8`、`size`），作为失败调用计入全部分母。
  审计按记录的 phase 重放：保留的正文必须与该失败一致，能解析成有效回答的正文不能记为失败。
- 驱动身份固定：各角色的有效运行配置含 GPU 型号与驱动版本，并与 token 报告绑定。驱动不同的节点不能混入
  已开始的研究，程序也不自动放宽；这种情况下单元在发出任何请求前失败并保留证据，需要回到运行身份相同的节点续跑，
  否则另立协议 revision 和新输出。
- attempt 启动余量按完整 watchdog 上限计算（pair 单元约 64 分钟、pair 预检约 96 分钟），allocation 末段
  可能闲置，这一保守取舍已接受。
- 离线审计使用节点运行时的同一份已发布源码，即 `study.json` 记录的源码闭包。
- freeze 必须完整：记录的证据清单要与证据目录逐项一致（只排除 session、自检指针、status 等按设计可变的文件），
  token、三个预检、60 个单元接受收据和源码指纹要与实际 SHA 相符；每个阶段的 attempt 目录要与接受计划记录的
  先前 attempt 一一对应，失败或中断的 attempt 不能被删除。

## 需要注意的边界

- 每个部署启动后先比较各角色的有效运行配置（含 GPU 型号与驱动）与 token 报告绑定的配置，
  不一致时按上面的驱动约定处理。
- 1,737 条输入在 Study 4 已经评价过；本研究给出回顾性的系统取舍，三轮重复只报告 median/min/max
  和逐格配对差。direct 0.90 与 pair 0.40+0.50 的显存份额不同，比较对象是部署方式。
- 本机整条命令模拟使用 10 个单元（5 条件 × 2 负载 × 1 轮）、40 条合成输入、约 1 秒的窗口，以及伪造的 Slurm、
  进程归属、GPU 和模型服务边界；它覆盖完整命令、失败重试、checkpoint、换 job 续跑、离线审计与篡改检查。
  正式协议仍是 60 个单元、1,737 条输入、120 秒窗口。模拟不能替代节点上的真实启动、同驻留和测量。
