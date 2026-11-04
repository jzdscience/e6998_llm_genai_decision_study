# v2 冻结实验协议

本协议对应 [configs/v2_protocol.json](../configs/v2_protocol.json) 和独立的
`decision_study/v2/` 入口。它在新增模型预测与 holdout 评价前确定。当前数据已经准备，
CPU Decision Lab 可展示已完成的 v1；**v2 GPU 采集、开放到达测量和新 holdout 结果尚未运行验收**。
代码存在、软件夹具通过或能生成报告，都不等于真实模型实验已完成。

## 研究范围与主问题

保持 StartLux-Decision-v1.1-4B-Instant 和 Qwen3.5-4B 的 v1 模型版本、原生接口、
精度、提示及运行环境，研究固定在线转交政策是否改善新 SQuAD 文章上的准确率，
并测量该政策的真实服务代价。语言模型权重不训练。

主比较是验证集选择出的 v2 政策相对验证集冻结的置信度政策，在新 holdout 上的
**配对准确率差**。两者都以验证集 10% 转交预算选择在线阈值。该预算不意味着
holdout 恰好转交 10%；实际转交率必须报告，不能在测试时用 top-k 重选政策。

本轮不加入 ANLI、9B 后级、三模型一致性路由、转交后仲裁或实时网页推理。
Kev 的原 v1 结果继续保留；代码有独立的可选 `holdout_kev` 采集入口，但默认流水线
只采集 StartLux/Qwen 的新 holdout。CLINC 拟合区记录仅保留为探索数据，不产生新的
CLINC 验证性结论。

## 数据与目标总体

| 数据角色 | 数量与来源 | 用途 |
|---|---|---|
| 原 calibration/validation | 合计 2,800 条，逐行复制原 v1 输入和标签 | 补齐 Qwen 预测；原 StartLux 预测直接复用 |
| SQuAD calibration | 1,000 条，包含于上述 2,800 条 | 仅在此拟合路由模型 |
| SQuAD validation | 1,000 条，包含于上述 2,800 条 | 固定在线阈值、选择候选政策 |
| 新 SQuAD holdout | 4,000 条，answerable/unanswerable 各 2,000 | 冻结后一次主要评价 |
| 开放到达性能池 | 原 SQuAD validation 中按固定哈希选出的 200 个唯一输入 | 测量固定在线政策；不接触 holdout 标签 |

新 holdout 来自固定版本 SQuAD2 原 train 中，v1 任意 split 都没有使用过的 57 篇文章。
准备阶段排除与 v1 重复的样本 ID、规范化输入或 context，再按
`sha256(6998-solo-v2|holdout|sample_id)` 在两个 gold 类别内分别选择。
原始文本保持不变，精确分词检查必须在模型运行前完成；不允许超长时静默截断或换例。

实际 manifest 记录了 14,157 条候选问题，选择的 4,000 条覆盖 57 篇文章。
未使用文章的原分配池为 development 50 篇、calibration 5 篇、validation 2 篇；
它们没有被 v1 实际选成输入。候选 answerable 出现在 57 篇文章，unanswerable 仅在
33 篇文章，不能把 4,000 个问题视作 4,000 个独立统计单位。

目标是预先规定的类别平衡、问题等权准确率。该数据是**本项目未用的 train 来源文章**，
既不保证与原 dev/test 同分布，也不证明基础模型预训练未见。不能将本次结果不加限定
地归因为在原测试总体上的提升。

模型输入与标签分开存储：

- `data/examples.jsonl`：2,800 条 fitting 输入和 4,000 条 holdout 输入，不含 gold。
- `data/fit_gold.json`：仅 fitting 标签。
- `sealed/holdout_gold.json`：仅新 holdout 标签。
- `data/manifest.json`：来源、选择规则、角色、文件哈希、协议和准备代码身份。

封存是研究流程控制，不是加密或操作系统访问权限。拟合与状态查询使用不打开 sealed gold
的验证入口；评价必须先验证 policy freeze。准备阶段为构造平衡样本和校验数据会读取标签，
但不产生任何模型评价或基于预测挑选样本。

## 路由政策

路由阶段只读取当前输入与已经执行的 StartLux 输出。不会读取 Qwen 的本次输出、gold
或未来输入批次，因此可以逐请求运行。

| 政策 | 拟合/分数 |
|---|---|
| `confidence` | `1 − released 最大概率`；无需拟合正确性模型 |
| `class_conditional` | 用最大概率的 logit 和预测类别拟合前级正确性；分数为 `1 − P̂(StartLux正确)` |
| `gain` | 用上述特征加 `log1p(输入字符数)` 分别拟合两模型正确性；分数为 `P̂(Qwen正确) − P̂(StartLux正确)` |
| `original_v1` | 原 v1 refitted 温度与风险 gate 原样保留，仅作次要描述性参照 |

两个拟合候选使用低容量的正则化 logistic 模型。特征标准化统计量只来自 calibration，
截距不正则化，固定 L2 为 0.01；全同类别目标使用 Beta(1,1) 平滑常数模型。
这里两端正确性都从标签拟合，不把校准后的最大概率自动当作给定类别/长度条件下的正确率。

`class_conditional` 与 `gain` 在 calibration 拟合后，分别在 validation 按转交分数
从高到低选择不超过预算的完整同分组。validation 上级联正确数较高者被选为主候选，
相同时选 `class_conditional`。置信度基线使用相同预算规则。

运行失败必须转交；若强制转交已经超过预算，记录其造成的预算偏离。
成功请求的固定阈值比较为 `score >= threshold`，并具有明确的全转交/全接受状态。
测试时原样使用这个在线政策，不按 holdout 的分数重新计算精确配额。

除主要 10% 预算外，5% 和 20% 作为预定描述性敏感性分析；原 v1 温度/风险门槛不重拟合。

## 主要终点、区间与解释

`policy_freeze.json` 保存选定政策、基线、各预算候选、validation 结果，以及数据、代码、
原校准与运行身份。只有身份一致、预测完整并通过相应采集验收记录的结果才能评价。

主要终点为：

`新 holdout 上 accuracy(selected) − accuracy(confidence)`

按文章进行 2,000 次配对 percentile bootstrap，报告 95% 区间。每次重采样文章及其
全部已选问题，再计算问题等权差值，保留文章内相关性。随机种子固定为 6998。
不承诺 4,000 条样本必能检出 0.6pp 或其他指定大小的效果。

同时报告两种政策各自的实际转交数、救回、误伤、净正确数、最终失败数和直接模型准确率。
直接模型、原 gate、其他预算比较均为描述性结果，不对多组未校正比较逐项声称显著。
区间包含零表示当前证据不足以确认改善，不能自动写成等价或证明没有收益。

首次评价记录 `holdout_access.json` 和评价收据。后续允许对**相同冻结身份和输入**做幂等
核验，不允许在查看 holdout 后修改政策、样本或配置再产生替代结果。

## 真实开放到达测量

先补齐 Qwen fitting 预测，在 CPU 拟合并冻结政策，然后才测量选定政策。
性能池来自原 validation，质量 holdout 继续封存；最后才执行主要评价。

固定 4 个条件、3 个到达率、3 轮，共 **36 个测量单元**：

| 条件 | 实际执行 |
|---|---|
| `local_only` | 只驻留并调用 StartLux |
| `resident_idle` | StartLux + Qwen 驻留，但请求只调用 StartLux |
| `selected_cascade` | 两模型驻留，按已冻结的选定政策真实转交 |
| `qwen_only` | 只驻留并调用 Qwen |

到达过程为固定种子的 Poisson，速率 10/25/40 请求每秒；每单元有 30 秒 offered window，
8 个 worker，最多 64 个 active+queued 请求，SLO 为 arrival-to-completion ≤500ms。
同一速率/轮次在各条件下使用匹配的到达时间表，每轮条件顺序按固定种子打乱。
36×30 秒仅为 offered window 合计 18 分钟，不包含启动、预热、drain 或其他采集时间。

记录 scheduled arrival、实际 dispatch、worker start、completion、拒绝/失败、worker
占用事件与 GPU 内存采样。满队列直接记录 rejected，不通过阻塞产生器来偷偷降低 offered load。
所有 offered 请求都计入 SLO 分母，迟到完成和拒绝不算满足 SLO。

主要系统指标 `goodput = 成功且满足 SLO 的请求数 / offered window 秒数`；另外分别报告
以 offered window 和完整 drain 时间为分母的成功吞吐。p50/p95/p99 使用已接纳请求的
arrival-to-completion 时延，包含失败的已接纳请求；拒绝数量另报。

模型启动和预热单独记录。worker 占用事件与逐请求执行时间做计量核对，但该核对不用于
声称 Little 定律已经识别“GPU 竞争”或“容量损失”的原因。相同转交率也不意味着政策性能相同，
因此只为实际测过的选定政策报告系统结果。

## 工程与完成标准

v2 使用独立的输出、数据、预测 job、控制面和入口。原 v1 的 controller、analysis、
benchmark 及工具链文件保持原身份；不能通过修改旧哈希绕过检查。
`runtime-control/` 是可写运行控制面，复用原已安装权重与环境路径，不复制或重装大模型。

默认 GPU 主采集为 2,800 条 Qwen fitting、4,000 条 StartLux holdout、4,000 条 Qwen
holdout，合计 10,800 次输入预测，另有开发样例 smoke、预热和性能请求。所有 GPU 成功、
失败、耗时与内存结论都要等真实回传后核验。

本轮完成需要三项预测 job 的匹配收据、有效 policy freeze、36 个带原始 trace/占用记录
的性能单元，以及封存身份一致的 `evaluation.json`。仅交付 ZIP、通过 CPU mock 或出现
某个输出文件不构成远端成功证明。操作与恢复见 [CONTINUE_V2_ZH.md](CONTINUE_V2_ZH.md)。

