# 回传核验范围与复算入口

这里保留归档完整性、固定代码官方审计、独立统计、独立拟合和生命周期的精选收据。原始 ZIP、gold、完整日志和逐样本预测不进入 Git。[正式验收](../../../records/anli-learnability-2026-11-15-acceptance.json)将这些收据及原包用 SHA 关联。

外部证据根及约定见 [ARTIFACTS_ZH.md](../../../docs/ARTIFACTS_ZH.md)。本轮对应：

- 原包：`records/bundles/anli-learnability-complete-20261115-08d88006855a.zip`
- 解压：`results/returned-learnability-complete-2026-11-15`
- 独立脚本和完整审计输出：`records/audits/anli-learnability-2026-11-15/`

官方复算需要固定提交 `41dfb7547698691634fa0e5a4b8f87824436fc8f` 的独立 checkout 和 NumPy 环境。从该 checkout 运行下面的命令，替换解压路径；不要用来源不明的包内代码执行审计。

```bash
python -B -m decision_study.learnability audit --output-dir /absolute/path/to/returned-learnability-complete-2026-11-15
```

独立审计脚本没有导入项目评价／拟合函数。原始运行位置为源码仓库忽略的 `.validation/anli-learnability-return-2026-11-15-08d88006855a/`；脚本保留本机路径和文件 SHA，归档副本不是免配置的通用 CLI。统计脚本仅需 NumPy 和同级 `extracted/`；拟合审计另需固定 ANLI 源 ZIP、固定 tokenizer 与发布 checkout；生命周期审计另需该发布 checkout。换机器重跑时先配置这些位置，保留原始审计脚本副本并记录调整。

`results.json`（上一级）逐字节复制自原包 `evaluation/results.json`，`NODE_REPORT.md` 同样保持节点原字节。图仅读取汇总值，不重拟合或重新选择配置；本机以 Matplotlib 3.11.2 生成，命令从仓库根执行：

```bash
python reports/learnability-2026-11-15/plot_results.py \
  --results reports/learnability-2026-11-15/results.json \
  --output reports/learnability-2026-11-15
```

**验收边界：** 精简包省略 176 份 embedding 数组。我们独立复算了 front-only 拟合结果的目标函数、梯度与标准化，并重放候选选择、阈值和评价分数，从冻结 text 分数重算了结果，但没有从文本向量独立重放 text-head 拟合／选择。完整重现需节点保留的矩阵或按冻结实现和环境重建后核验。停止记录是包内证据，本机审计没有直接观察当前 GPU。串行预测不是在线服务／SLO 实验。
