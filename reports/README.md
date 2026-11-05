# 实验结果入口

V1–V3 已完成，共 **44,450 条正式模型预测**。下面是已验收的历史报告，保留原内容与
SHA 清单；原始预测、审计脚本和大包位于 [外部证据目录](../docs/ARTIFACTS_ZH.md)。

| 证据 | 范围 | 主要结论 |
|---|---|---|
| [V1 质量](formal-2026-11-02/RESULTS_ZH.md) | 15,950 条预测；CLINC 与 SQuAD 可答性 | StartLux 主测试分别为 93.50% 和 91.25%；级联收益有限 |
| [V1 性能](benchmark-2026-11-02/BENCHMARK_ZH.md) | 60 个测量单元、60,200 次请求 | 含双模型驻留和并发 8 的实际测量 |
| [V2](v2-complete-2026-11-03/RESULTS_ZH.md) | 10,800 条预测；4,000 题 holdout；36 个性能单元 | gain−confidence −0.225pp，95% CI [−0.908,+0.478]pp |
| [V3](v3-complete-2026-11-03/RESULTS_ZH.md) | 17,700 条预测；3,200 题 ANLI holdout | positive_gain9B−confidence9B +0.594pp，95% CI [−0.528,+1.674]pp |
| [CLINC 轻量基线](light-baseline-2026-11-04/RESULTS_ZH.md) | 400 个训练标签；另计 1,850 条 CPU 分类器预测 | TF-IDF＋LR main 73.83%、banking 66.00%；回顾性对照，独立重放通过 |

两个主比较区间均跨零，未显示主方法改善。V3 positive_gain 冻结为零转交，与直接
StartLux 一致；实验仍完整采集了后级预测。Qwen9B 已做顺序质量实验，未测同卡共驻留性能。

## 仓库内可以直接浏览的图表

- [V1 质量图](formal-2026-11-02/figures/clinc_accuracy_complementarity.png)、
  [SQuAD 路由图](formal-2026-11-02/figures/squad_routing.png)。
- [V1 吞吐](benchmark-2026-11-02/figures/throughput.png)、
  [性能汇总 CSV](benchmark-2026-11-02/summary.csv)。
- [V2 主比较](v2-complete-2026-11-03/primary_difference.png)、
  [V2 SLO goodput](v2-complete-2026-11-03/slo_goodput.png)。
- [V3 准确率](v3-complete-2026-11-03/accuracy.png)、
  [V3 预算与主比较](v3-complete-2026-11-03/routing_budget_and_primary.png)。

历史报告里指向 `results/`、`.validation/` 和原始 ZIP 的链接需要额外取得 artifact；
本入口没有把它们当作 clone 自带文件。原始报告的 SHA 清单保持不变，当前复制清单见
[仓库整合记录](../records/repo-adoption-2026-11-03.json)。

CPU Decision Lab 已整合 V1–V3，启动见 [LAB_CURRENT_ZH.md](../docs/LAB_CURRENT_ZH.md)。
V2/V3 便携包保存在 `decision-lab-2026-11-04/`，原 V1 回放包保持不变。
