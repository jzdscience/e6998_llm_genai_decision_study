# 冻结实验结果与验收记录

2026-11-02 · `solo-v1` · Kev-4B、StartLux-Decision-v1.1-4B-Instant、Qwen3.5-4B

**在本次冻结的数据、提示和运行配置下，StartLux 的直接分类准确率最高。** Qwen 明显优于 Kev，但作为 StartLux 的后级只带来很小的点估计增益，尚无可靠的准确率提升证据。前级置信度能够改善相同转交数量下的路由选择；验证集上选择的 5% 风险阈值却没有普遍迁移到测试集。温度重标定改善概率指标的效果也因任务和模型而异。

正式预测、冻结校准和离线分析已经完成。本包没有端到端 benchmark，不能据此声称级联更快、更省显存或成本更低。

## 1. 数据、结果与复核范围

CLINC 是 15 个银行意图加 `oos` 的 16 类分类；SQuAD 是根据给定段落判断问题是否可回答的二分类，表内准确率不是抽取式问答的 EM/F1。

| 数据用途 | CLINC | SQuAD | 合计 |
|---|---:|---:|---:|
| 校准集，只拟合温度 | 400 | 1,000 | 1,400 |
| 验证集，只选择接收阈值 | 400 | 1,000 | 1,400 |
| 主测试集 | 600 | 2,000 | 2,600 |
| 额外 stress 样本 | 850 | 0 | 850 |
| 开发验收样本，不进入正式结果 | 64 | 64 | 128 |

Kev 和 StartLux 各产生 6,250 条正式预测，即 1,400 校准 + 1,400 验证 + 2,600 测试 + 850 stress。Qwen 只预测测试和 stress，共 3,450 条。三者合计 **15,950 条**，均为首次请求成功，没有终止错误或重试。共有 6,378 个输入，包含上述 128 个开发样本。

独立验收核对了 768 个清单文件、638 个预测分片及既有 Kev 受保护文件。指标复核从原始预测重新计算三个工作负载的分类、概率、级联、曲线、2,000 次 paired bootstrap、配额匹配对照及 60 个错误示例，**51/51 项检查通过，未发现差异**。此次复核使用已有校准和阈值，没有重新拟合或调整测试集阈值。

## 2. 主测试集的直接预测

准确率区间是 95% paired percentile bootstrap 区间，使用 2,000 次重采样。CLINC 按真实类别分层抽样；SQuAD 以 35 个文章为聚类单位重采样。宏 F1 是分类指标。

| 主测试集 | 模型 | 正确数 / 总数 | 准确率 | 95% CI | 宏 F1 |
|---|---|---:|---:|---:|---:|
| CLINC | Kev | 465 / 600 | 77.50% | 75.00–80.17% | 0.7651 |
| CLINC | **StartLux** | **561 / 600** | **93.50%** | **91.50–95.17%** | **0.9327** |
| CLINC | Qwen | 528 / 600 | 88.00% | 85.67–90.17% | 0.8714 |
| SQuAD | Kev | 1,627 / 2,000 | 81.35% | 78.38–84.18% | 0.8115 |
| SQuAD | **StartLux** | **1,825 / 2,000** | **91.25%** | **88.64–93.35%** | **0.9122** |
| SQuAD | Qwen | 1,720 / 2,000 | 86.00% | 83.32–88.40% | 0.8600 |

应使用配对差异区间比较模型，而不是根据上表边际区间是否重叠推断显著性。StartLux 相对 Qwen 的准确率差异在 CLINC 为 **+5.50 个百分点，95% CI +3.67 至 +7.33**；在 SQuAD 为 **+5.25 个百分点，95% CI +3.59 至 +6.82**。

CLINC 的差距主要来自银行意图分类。三个模型在主测试集的 OOS 召回率都为 148/150，即 98.67%；但 Kev、StartLux、Qwen 将银行请求错误判为 OOS 的数量分别为 **117、20、36**。在同一批 450 条银行请求上，分类准确率分别为 70.44%、91.78%、84.44%。因此不能仅凭 Kev 的高 OOS 召回率认为其分类能力与另外两者接近。

## 3. 冻结阈值下的级联与选择性风险

级联先使用 Kev 或 StartLux 的置信度决定是否接收；未接收的样本使用已经实际获得的 Qwen 预测。下表是对真实预测的离线组合。`coverage` 是由前级接受的比例；`selective risk` 是**前级接受样本中的错误比例**，不是级联系统的总体错误率。

`released` 直接使用发布概率；`refitted` 使用校准集拟合的额外温度。两者的阈值均在独立验证集选择后冻结。

| 主测试集 / 前级 | 概率条件 | 接受数 / 总数（coverage） | 接受错误数 / 接受数（risk） | Qwen 转交数 | 级联准确率 | 相对前级净增加正确数 |
|---|---|---:|---:|---:|---:|---:|
| CLINC / Kev | released | 390/600（65.00%） | 26/390（6.67%） | 210 | 86.17% | +52（挽救 67 − 误伤 15） |
| CLINC / Kev | refitted | 385/600（64.17%） | 29/385（7.53%） | 215 | 85.67% | +49（65 − 16） |
| CLINC / StartLux | 两条件相同 | 599/600（99.83%） | 38/599（6.34%） | 1 | 93.50% | 0（0 − 0） |
| SQuAD / Kev | 两条件相同 | 708/2,000（35.40%） | 31/708（4.38%） | 1,292 | 85.85% | +90（184 − 94） |
| SQuAD / StartLux | 两条件相同 | 1,875/2,000（93.75%） | 118/1,875（6.29%） | 125 | 91.50% | +5（29 − 24） |

“挽救”表示前级错误而 Qwen 正确，且该样本被转交；“误伤”表示前级原本正确，转交后 Qwen 错误。

- **Kev 级联改善了 Kev，但没有超过直接 Qwen。** CLINC released 相对 Kev 增加 8.67 个百分点，却比直接 Qwen 低 1.83 个百分点；后者的配对 95% CI 为 −3.17 至 −0.67。SQuAD 级联比 Kev 高 4.50 个百分点，与 Qwen 的差异为 −0.15 个百分点，95% CI −0.512 至 +0.151。
- **StartLux 的级联收益很有限。** CLINC 只转交一条，两者都预测错误，因此准确率不变。SQuAD 增加 5 个正确样本，提升 0.25 个百分点；配对 95% CI 为 **−0.293 至 +0.825 个百分点**，不能据此确认优于直接 StartLux。
- **5% 是验证集经验约束，未构成测试风险保证。** 主测试上，CLINC 两个前级以及 SQuAD StartLux 的接受风险都超过 5%。Kev/SQuAD 的点估计为 4.38%，但风险区间为 2.57–6.67%，仍不能宣称未来风险受控于 5%。StartLux/CLINC 验证集接受全部 400 条、错误仅 5 条，测试集接受风险却升至 6.34%，直接展示了验证风险不能自动迁移。

### 相同转交配额下的对照

随机对照与 oracle 均使用**完全相同的 Qwen 转交数量**；本次没有必须转交的运行失败。Oracle 使用真实标签排序转交收益，是事后上界。下表使用随机路由的解析期望；`summary.csv` 使用 100 次随机路由的均值，两者有小幅 Monte Carlo 差异。

| 主测试集 / 条件 | 转交数 | 随机路由期望准确率 | 冻结置信度路由 | 同配额 oracle |
|---|---:|---:|---:|---:|
| CLINC / Kev released | 210 | 81.18% | 86.17% | 91.00% |
| CLINC / Kev refitted | 215 | 81.26% | 85.67% | 91.00% |
| CLINC / StartLux | 1 | 93.49% | 93.50% | 93.67% |
| SQuAD / Kev | 1,292 | 84.35% | 85.85% | 90.85% |
| SQuAD / StartLux | 125 | 90.92% | 91.50% | 94.55% |

置信度路由在这些固定配额下优于随机路由的点期望，但与 oracle 仍有距离。这不等同于相对最佳直接模型有收益，也不说明真实执行成本降低。所有级联输入都需要先运行前级，转交样本还要运行 Qwen。

## 4. 概率质量与温度重标定

只有 Kev 和 StartLux 提供完整概率向量。Qwen 只返回标签，不能从本结果计算其 NLL、Brier 或 ECE。下表均为主测试集；数值越低越好。ECE 使用 15 个等宽 top-label 置信度分箱，Brier 对每个样本的所有类别平方误差求和后取均值。

| 任务 / 模型 | 拟合温度 | NLL：released → refitted | Brier：released → refitted | ECE：released → refitted |
|---|---:|---:|---:|---:|
| CLINC / Kev | 0.8306 | 0.6188 → 0.6271 | 0.3050 → 0.3125 | 0.0577 → 0.0556 |
| CLINC / StartLux | 0.8309 | 0.2455 → 0.2597 | 0.1041 → 0.1048 | 0.0265 → 0.0248 |
| SQuAD / Kev | 1.2949 | 0.4315 → 0.4223 | 0.2741 → 0.2707 | 0.0417 → 0.0214 |
| SQuAD / StartLux | 0.9584 | 0.2161 → 0.2166 | 0.1276 → 0.1278 | 0.0133 → 0.0157 |

StartLux 的 released 概率在两个任务上都明显优于 Kev。额外重标定在 Kev/SQuAD 上同时改善三项指标；在 CLINC 上，两模型的 ECE 略降而 NLL、Brier 上升；StartLux/SQuAD 的三项指标略有变差。因此不能把 ECE 的单项改善概括为概率质量全面改善。当前分析未给这些概率指标提供置信区间，也未计算 ICI，细小差异应按描述性结果解读。

重标定并未自动改善路由。SQuAD 是二分类，标量温度单调变换保持置信度排序，本次两个前级的接受集合均完全不变。StartLux/CLINC 接受集合也未变。Kev/CLINC 有 11 条接受状态改变，接受总数减少 5 条，级联正确数减少 3 条。不能在看到测试结果后重新选择温度或阈值以补救这些结果。

## 5. CLINC stress 与 OOS 占比

主测试工作负载由 450 条银行请求和 150 条 OOS 构成，OOS 占 25%。Stress 工作负载复用这 **450 条银行请求**，另加 850 条 stress OOS，共 1,300 条，OOS 占 65.38%。两个工作负载重叠，不能按两个独立测试集计算证据量；它们同时改变 OOS 占比和 OOS 样本组成。

| CLINC 指标 | 主测试 | Stress |
|---|---:|---:|
| Kev 直接准确率 | 77.50% | 89.62% |
| StartLux 直接准确率 | 93.50% | 96.62% |
| Qwen 直接准确率 | 88.00% | 94.08% |
| Kev released coverage / risk | 65.00% / 6.67% | 82.62% / 2.51% |
| StartLux released coverage / risk | 99.83% / 6.34% | 99.92% / 3.31% |

Stress 上的指标更好不表示模型变强。银行子集完全相同、预测也相同；更多容易识别的 OOS 改变了聚合准确率、coverage 和接受风险。Kev 的宏 F1 仅从 0.7651 到 0.7790，而直接准确率提高了 12.12 个百分点，说明只看总体准确率会掩盖银行分类的弱点。

归档的 stress 差异区间在重采样中共享银行样本，正确保留了这种重叠。`clinc_prevalence` 图另在同一条件池上重加权，用于展示固定条件表现下的占比敏感性；它不证明真实部署中的纯因果占比效应。

## 6. 错误示例与解释边界

错误示例文件每个任务最多 30 条，按模型、互补错误类别及 OOS 分桶后确定性抽取。它适合解释错误机制，不能用桶内样例数量估计错误发生频率。

- CLINC 的 `make 1234 the pin on my savings account`，标签为 `pin_change`。Kev 判为 `oos`，重标定置信度约 0.885，仍被前级接受；Qwen 的正确答案没有进入最终决策。这说明“后级能够修正”还需要路由确实找到该样本。
- `look up the current auto loan rate at chase bank` 的冻结标签为 `oos`，而 StartLux 与 Qwen 均判为 `interest_rate`，StartLux 重标定置信度约 0.996。该例提示源数据意图边界与自然语言相似性之间存在困难；本分析保留原始标签，不据此宣布标签错误。
- SQuAD 的一些失败涉及角色组合、否定或问题前提变化。例如段落分别列出 Attorney General 和 Obama campaign advisor，问题将两种身份合并；Kev 与 Qwen 都预测可回答。即便两模型共享语义相关线索，后级也未必纠正前级。

结论适用于当前冻结任务、提示、模型版本及采样数据。公开训练/开发数据暴露不完全可知；已核验项目样本划分、SQuAD文章隔离及温度/阈值的拟合来源，这不能证明预训练未见过数据。Bootstrap 条件化于已选定模型、温度和阈值，不包含重新训练、重新选门限的变异，也没有多重比较校正。

## 7. 证据与复现

- [原始实验归档](../../records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip)，SHA256 `be748917f2f2184e3a7d3468b4fa222e7d1b77c0abbba863fce64f395c1840e7`。
- [完整性与模型验收](../../.validation/formal-experiment-2026-11-02/audit.json)及其[验证脚本](../../.validation/formal-experiment-2026-11-02/verify.py)。
- [独立指标复算收据](../../.validation/formal-experiment-2026-11-02/independent-metrics.json)及[可重复运行的复算脚本](../../.validation/formal-experiment-2026-11-02/check_independent_metrics.py)。
- [全部指标和区间](../../results/returned-experiment-2026-11-02-101054/analysis/metrics.json)、[汇总表](../../results/returned-experiment-2026-11-02-101054/analysis/summary.csv)、[概率指标](../../results/returned-experiment-2026-11-02-101054/analysis/probability_metrics.csv)、[错误示例](../../results/returned-experiment-2026-11-02-101054/analysis/error_cases.jsonl)。

在项目根目录使用已具备 NumPy 的 controller Python 运行下列命令。指定新的结果路径可保留原审计收据。

```bash
python .validation/formal-experiment-2026-11-02/check_independent_metrics.py \
  --archive records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip \
  --result /tmp/formal-metrics-recheck.json
```

## 8. 报告或演示建议采用的现有图

优先使用归档已有版本，使每张图承担一个明确论点。

| 图 | 适合说明的内容 | 配套口径 |
|---|---|---|
| [CLINC accuracy/complementarity](../../results/returned-experiment-2026-11-02-101054/analysis/figures/clinc_accuracy_complementarity.png) | StartLux 的直接优势及与 Qwen 的有限互补性 | Qwen 优于 Kev，但没有优于 StartLux |
| [SQuAD routing](../../results/returned-experiment-2026-11-02-101054/analysis/figures/squad_routing.png) | 冻结路由、同配额随机对照和 oracle | 强调 StartLux 的净增益仅 5 条，提升区间跨零 |
| [CLINC calibration/risk](../../results/returned-experiment-2026-11-02-101054/analysis/figures/clinc_calibration_risk.png) | 验证集阈值与测试接受风险 | 5% 是经验目标，不能当测试风险保证 |
| [SQuAD probability quality](../../results/returned-experiment-2026-11-02-101054/analysis/figures/squad_probability_quality.png) | Kev 重标定改善概率指标但不改变接受集合 | 概率校准与路由收益分开解释 |
| [CLINC prevalence](../../results/returned-experiment-2026-11-02-101054/analysis/figures/clinc_prevalence.png) | OOS 占比如何改变聚合指标 | 与真实 stress 的组成变化区分，不能当独立复制 |

另有[四例CPU历史结果回放](DEMO_REPLAY_ZH.md)，现有CLI已逐例实际验证，覆盖纠错收益、
转交损害和高置信度错误。样例利用已知结果选取，只供说明，不代表无偏评估或实时推理。
