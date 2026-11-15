# 一条命令运行完整 ANLI 实验

这份手册取代此前只运行 90 条开发预检再人工回传的步骤。内部检查通过后自动继续，最后只需回传一个结果包。
方案与固定方法见 [完整协议](ANLI_LEARNABILITY_FULL_ZH.md)。

在固定提交的独立源码 checkout、实际 L40S / 100 GB RAM code-server allocation 中运行：

```bash
bash scripts/run_learnability.sh /home/zhanj289/e6998/decision-study
```

程序自动执行数据准备、Linux 进程清理自检、三个模型旧开发预检、67,311 条正式预测、冻结 MiniLM 特征、
路由器拟合与选择、冻结全部参数和评价分数、读取评价标签、计算报告和打包。无需每一步停下回传或等待人工批准。
预检或数据完整性检查失败时会保存诊断并停止，不能跳过检查。

默认复用以下已有资产，不修改其绑定或安装环境：

- GPU controller：`/home/zhanj289/e6998/decision-study/.venv-controller/bin/python`。
- CPU 环境优先：`/home/zhanj289/e6998/decision-study-src/.venv-cheap-baselines/bin/python`，其次运行根下同名环境；核对完整旧包版本。
- MiniLM：运行根下 `results/aux-clinc-cheap-v1-inputs/model`，逐文件核对固定 SHA。
- 三个 LLM、CUDA 与 vLLM：已恢复的 V1/V3/size-control 绑定与缓存。
- ANLI 源 ZIP：优先复用已核验缓存，否则只下载固定官方 URL 并核验 SHA，有独立下载时限。

新输出全部在：

```text
/home/zhanj289/e6998/decision-study/results/aux-anli-learnability-v1
```

终端最终显示 `STUDY_COMPLETE` 才代表整套实验完成。然后下载最后打印的 `learnability_bundle-*.zip`。
`REPORT.md` 和 `evaluation/results.json` 在同一输出根内。

若显示 `STUDY_CHECKPOINT`，说明当前 allocation 余量不足，进度已保存。开启新的实际 allocation 后，在同一源码目录运行同一条命令即可。
无需填写 HOURS_LEFT 或手动修改 session；程序核实真实 Slurm 余量与旧 job 终止状态。

真正的失败显示 `BLOCKED` 并导出诊断。调查该失败后，可明确重试：

```bash
bash scripts/run_learnability.sh /home/zhanj289/e6998/decision-study --retry-failed
```

如果曾使用此参数处理失败，后续续跑也可以保留它；该参数不会跳过哈希、输入、gold 时序或进程停止检查。
不要删除输出目录、改收据或覆盖旧源码来解除阻断。

回传包省略大的特征矩阵，但含它们的哈希与重建身份；原始矩阵继续保存在节点 `collection/embeddings/`。
其余输入、原始预测、拟合参数、冻结路由分数、评价标签和停止证据均保留，用于独立验收。
使用同一个固定源码提交，在安全解压的新目录上运行：

```bash
python3 -B -m decision_study.learnability audit --output-dir /absolute/path/to/extracted-bundle
```

该审计重算冻结分数对应的结果和区间，不重新观察节点 GPU，也不把缺省的特征矩阵声明为已重新验算。
