# 数据、结果与源码的边界

11/5新增两项独立实验的准备记录与运行入口见
[followups-preparation-2026-11-05.json](../records/followups-preparation-2026-11-05.json)和
[RUN_FOLLOWUPS_ZH.md](RUN_FOLLOWUPS_ZH.md)。11/5启动兼容修复后，node新输出分别为旧运行根下
`results/aux-clinc-service-v1-r2`与`results/aux-size-control-v1-r2`；原无后缀失败根保留，已作为
diagnostic条目加入索引。旧`solo-v1`、完整`solo-v3`、已验收`aux-clinc-cheap-v1`与MiniLM模型均
原位只读复用。

两项结果已于11/6回传、11/7验收。外部证据根新增
`records/bundles/service-complete-20261106-d212c130c707.zip`、
`records/bundles/size-control-complete-20261106-f427f7762bac.zip`，解压于
`results/returned-service-complete-2026-11-06`与`results/returned-size-control-complete-2026-11-06`，
两个条目已加入[BUNDLE_INDEX.json](../records/BUNDLE_INDEX.json)。11/6的审计产物在本仓库忽略的
`.validation/followups-return-2026-11-06/`，脚本与精选收据复制在
[reports/followups-2026-11-06/](../reports/followups-2026-11-06/RESULTS_ZH.md)。重跑服务审计还需要
V1的CLINC gold，位于外部根`results/returned-kev-2026-11-01/data/gold.json`。

11/7起，外部根的旧日志`docs/EXPERIMENT_LOG.md`、`docs/CURRENT_STATE.md`和V1的`analysis/metrics.json`
在仓库内各有一份逐字节副本（`docs/history/*_PRE_REPO_ZH.md`、`reports/formal-2026-11-02/analysis/`）；
外部根旧的`.venv-lab`环境已移入废纸篓。证据根旁的旧补丁ZIP、`.sha256`和散落的notebook/runbook副本移到
`project/solo-experiments-2026-10-30/archive/`，带大小与SHA清单；证据根`decision-study/`未变。

2026-11-03。本仓库保存源码、协议和精选报告。原始结果包、准备数据、审计产物和环境
没有搬入 Git；clone 后不会自动得到这些文件。

## 当前证据位置

本机原始证据根为：

```text
/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study
```

此路径是本机存储位置，不是别人 clone 后应有的目录。换电脑时把需要的 artifact 复制到
自己的存储根，保留相对路径并校验 SHA。服务器现有约 89 GB 的运行树仍在原位，本次没有
下载权重、复制虚拟环境或移动服务器的 results。

[BUNDLE_INDEX.json](../records/BUNDLE_INDEX.json) 的 `archive` 和 `extracted_directory`
均相对**外部证据根**，不是本源码仓库。每个结果包的 SHA、字节数和原导出时间均保留。
完整早期恢复记录仍在外部根的 `docs/EXPERIMENT_LOG.md`。

仓库中的历史报告保持原字节，引用的 `../../results/`、`../../.validation/`、
`../../records/bundles/` 路径需要外部 artifact；这些引用不表示文件已加入 Git。
各报告的原始 `SHA256.json` 也按外部证据树解释，不是当前 checkout 的完整文件清单。
[报告入口](../reports/README.md) 提供本仓库可直接浏览的汇总与图表。

## 干净 clone 的本地开发

在仓库根目录用 Python 3.11–3.13 建独立环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m decision_study --help
.venv/bin/python -m decision_study.v2 --help
.venv/bin/python -m decision_study.v3 --help
```

部分测试需要原始 V1 artifact 或可选 Gradio，缺少时会跳过。HTTP 软件测试需要允许
临时本地测试端口，不调用真实模型。安装 Lab 及启动方式见 [LAB_CURRENT_ZH.md](LAB_CURRENT_ZH.md)；已跟踪的
V1 和 V2/V3 便携证据包可用于三轮展示，不需要原始运行目录。

## 恢复 V2/V3 prepared 数据

原发布校验器要求以下八个文件在源码根下，忽略规则已排除整个 `results/`：

```text
results/solo-v2/data/examples.jsonl
results/solo-v2/data/fit_gold.json
results/solo-v2/data/manifest.json
results/solo-v2/sealed/holdout_gold.json
results/solo-v3/data/examples.jsonl
results/solo-v3/data/fit_gold.json
results/solo-v3/data/manifest.json
results/solo-v3/sealed/holdout_gold.json
```

从原始发布 ZIP 中仅恢复这八个文件，或从外部证据根的 `results/solo-v2-prepared/`、
`results/solo-v3-prepared/` 取得对应文件，逐项与仓库 release manifest 的 SHA 比较。
目标已经存在且不同就停止检查，不覆盖。原发布 ZIP 位于上述证据根的上一级：

| 发布包 | SHA256 |
|---|---|
| `decision-study-v2-ready.zip` | `a8aba344b370f95dd2e2d86ca4eec6428d7f0604a4deb230de5d8a050251559d` |
| `decision-study-v3-ready.zip` | `8fdd6f8dbbff4f9daac2cd342ebc0a2a67ac77f1f5d8a7b805862fdfb2237fb9` |

恢复数据后，在当前源码根执行：

```bash
.venv/bin/python scripts/verify_v2_release.py
.venv/bin/python scripts/verify_v3_release.py
```

本次仅在临时验证树恢复八个文件并通过两项校验；当前源码目录仍不含 prepared 数据。
不要把历史发布 ZIP 整包覆盖到当前 Git 工作区，也不要用 clone 后缺数据为由放宽校验。
prepared 文件足以做发布文件核验，不能代替模型安装、真实 runtime 或公司 allocation。

旧实验运行元数据绑定服务器绝对路径，不能通过搬运结果目录和修改 binding 恢复运行。
下一轮应冻结新的代码与运行身份，具体入口按对应协议执行。

## CLINC 轻量基线的新增输入

`clinc-cheap-v1` 不要求从本机手工上传训练数据。其独立入口按固定官方revision下载两份
原始CLINC文件，校验SHA后提取训练专用数据；再下载八份固定MiniLM文件并核验。
具体路径与命令见 [RUN_CHEAP_BASELINES_ZH.md](RUN_CHEAP_BASELINES_ZH.md)。
这些输入留在旧runtime的新 `results/aux-clinc-cheap-v1-inputs/` 下；训练输出在并列的
`results/aux-clinc-cheap-v1/`，不进入Git。完整prepared数据约255KB，编码器约90.9MB；
prepare阶段与fit阶段的标签访问边界见 [协议](CHEAP_BASELINES_PROTOCOL_ZH.md)。

本轮回传已验收。原ZIP和解压目录按 [BUNDLE_INDEX.json](../records/BUNDLE_INDEX.json)
记录保存在外部证据根；两份训练专用prepared另存外部根的 `results/cheap-baselines-prepared/`。
本机编码器缓存仍在源码仓库忽略的 `.validation/cheap-baselines-2026-11-04/model/`，
新clone需按协议恢复权重，不把该本机缓存路径视作仓库自带。
独立审计脚本及完整结果见 [回传报告](../reports/cheap-baselines-2026-11-04/RESULTS_ZH.md)。
