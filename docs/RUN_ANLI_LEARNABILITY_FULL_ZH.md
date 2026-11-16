# 一条命令运行完整 ANLI 实验

当前为 2026-11-15 RHEL8 兼容修复版。科学协议不变，模型权重及原环境复用；
发布说明、内核实测和新输出边界见 [RHEL8 运行说明](RUN_ANLI_LEARNABILITY_RHEL8_ZH.md)。
方案与固定方法见 [完整协议](ANLI_LEARNABILITY_FULL_ZH.md)。

## 运行与续跑

在独立源码目录 `decision-study-learnability-rhel8` 中，按交付的完整提交 SHA 固定 checkout 后运行：

```bash
cd /home/zhanj289/e6998/decision-study-learnability-rhel8
bash scripts/run_learnability.sh /home/zhanj289/e6998/decision-study
```

实际 L40S / 100 GB RAM allocation 中运行。新输出根为：

```text
/home/zhanj289/e6998/decision-study/results/aux-anli-learnability-v1-r2
```

原 `c3c8b65` 与 `aux-anli-learnability-v1` 的失败记录保留。不要手工替换旧源码、改写 source hash，
或把旧尝试收据复制进新输出。本版重新生成相同固定输入，只复用已核验的 ANLI 源 ZIP 和原模型/环境。

程序自动执行数据准备、Linux 进程清理自检、三个模型旧开发预检、67,311 条正式预测、冻结 MiniLM 特征、
路由器拟合与选择、冻结全部参数和评价分数、读取评价标签、计算报告和打包。成功阶段无需人工交接。
预检或完整性检查失败会保留诊断并停止，不能跳过检查。

默认资产：GPU controller 为运行根 `.venv-controller/bin/python`；CPU 优先复用相邻
`decision-study-src/.venv-cheap-baselines/bin/python`，其次运行根同名环境，核对固定旧包版本；
MiniLM 在 `results/aux-clinc-cheap-v1-inputs/model` 逐文件核对 SHA；三个 LLM、CUDA、vLLM 按旧绑定读取。
ANLI 源 ZIP 复用固定 SHA 缓存，否则只下载固定官方 URL，并设独立时限与 SHA 校验。

最终显示 `STUDY_COMPLETE` 才表示整套实验完成；下载最后打印的 `learnability_bundle-*.zip`。
`REPORT.md`、`evaluation/results.json` 位于同一输出根。

`STUDY_CHECKPOINT` 表示实际 allocation 余量不足且已保存进度。开启真实新 allocation 后运行同一条命令，
程序核实 Slurm 余量与旧 job 终止状态，无需 HOURS_LEFT 或手工改 session。
真正失败显示 `BLOCKED`，调查后可显式重试：

```bash
bash scripts/run_learnability.sh /home/zhanj289/e6998/decision-study --retry-failed
```

这不跳过哈希、输入、gold 时序或停止检查。此次 c3c8b65 的 API 存在性失败没有启动模型，
新输出的首次执行不需要该选项。

## 回传验收

紧凑包省略大特征矩阵，保留其哈希/重建身份；原矩阵留在节点 `collection/embeddings/`。
输入、原始预测、拟合参数、冻结路由分数、评价标签和停止证据保留。使用相同固定源码，在安全解压的新目录验收：

```bash
python3 -B -m decision_study.learnability audit --output-dir /absolute/path/to/extracted-bundle
```

该审计从冻结分数重算指标与区间；不重新观察节点 GPU，不把省略的特征矩阵声明为已重新验算。
