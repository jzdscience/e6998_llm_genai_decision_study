# Decision Study（中文说明）

研究何时直接使用 decision model、何时转交给生成式模型值得其质量与服务成本。
V1、V2、V3 三轮共 44,450 条正式模型预测；后续的 StartLux9B 规模对照另有 3,200 条，CLINC 轻量
基线有 7,400 条 CPU 分类器预测，CPU/GPU 服务测量有 469,804 次请求，9B 服务研究有 162,801 次 offered
请求。服务请求重复使用已评分的输入，和预测条数分开计。以上均已验收。
English overview: [README.md](README.md)。

- 第一次打开先看[入门指南](docs/START_HERE_ZH.md)，里面有阅读顺序、演示启动命令和讲稿入口
- [课程报告初版（ACL 双栏）](paper/main.pdf)，源码与编译说明见 [paper/README.md](paper/README.md)
- [当前进度与下一步](docs/CURRENT_STATE.md)、[实验与排障日志](docs/EXPERIMENT_LOG.md)
- [文档索引](docs/README.md)、[常见故障](docs/TROUBLESHOOTING.md)
- [结果与图表](reports/README.md)
- [数据、环境与结果包的取得方式](docs/ARTIFACTS_ZH.md)
- [源码仓库和服务器协作](docs/REPO_MANAGEMENT_ZH.md)、[node 首次配置与 Git 更新命令](docs/NODE_GIT_ZH.md)
- [CPU Decision Lab](docs/LAB_CURRENT_ZH.md)（V1–V3 加 follow-up 页，运行 `bash scripts/run_lab.sh`）
- [演示大纲与讲稿](docs/PRESENTATION_ZH.md)
- [直接准确率矩阵补全的协议](docs/DIRECT_MATRIX_PROTOCOL_ZH.md)与[节点运行说明](docs/RUN_DIRECT_MATRIX_ZH.md)（本机已准备，GPU 尚未运行）

| 阶段 | 任务与内容 | 状态 |
|---|---|---|
| V1 | CLINC / SQuAD 可答性，Kev4B / StartLux4B / Qwen4B，校准、级联与性能 | 已验收 |
| V2 | 新 SQuAD holdout，gain / confidence / class-conditional 与开放到达测量 | 已验收 |
| V3 | ANLI，StartLux4B 前级、Qwen4B / Qwen9B 后级 | 已验收；顺序质量实验 |
| CLINC 轻量基线 | TF-IDF / MiniLM 加 LR，400 与 1,600 个训练标签 | 已验收；回顾性对照 |
| 服务测量 | MiniLM400/1600（CPU）、StartLux4B、Qwen4B，5/25/100 请求/秒，500 ms SLO | 已验收 |
| 规模对照 | StartLux9B 在 V3 的 ANLI test 上直接预测 | 已验收；回顾性对照 |
| 9B 服务研究 | StartLux4B（0.40）、StartLux9B（0.50）、4B→Qwen9B 级联（0.40 + 0.50），ANLI，10/20/40/80 请求/秒，500/1,000 ms SLO | 已验收（11/10）；回顾性，[报告](reports/upgrade-9b-serving-2026-11-09/RESULTS_ZH.md) |
| 直接准确率矩阵补全 | 论文 Table 2 的 8 个未测格：StartLux9B 与 Qwen9B × CLINC 和两个 SQuAD 划分，Kev4B × S2 holdout 与 ANLI；20,400 条测试预测，复用 12 个已验收单元 | [本机验收通过](docs/DIRECT_MATRIX_ACCEPTANCE_2026-11-10_ZH.md)；节点 GPU 0/8，没有新准确率；[协议](docs/DIRECT_MATRIX_PROTOCOL_ZH.md)、[运行说明](docs/RUN_DIRECT_MATRIX_ZH.md) |

V2/V3 的预定主比较均未显示统计上的改善。V3 positive_gain 选择零转交，与直接 StartLux
相同；实验仍采集了完整后级预测。9B 服务研究中，直接 9B 在四档已测负载下的中位正确 goodput 都高于
冻结级联，80 请求/秒时直接 4B 最高；三种部署的显存份额不同，结论限于这组固定部署。各项结论及限制见
原始结果报告和课程报告。

## 本地环境与复现

使用 Python 3.11–3.13。主环境装演示和测试依赖：

```bash
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[lab,test]'
bash scripts/reproduce_cpu.sh
```

`reproduce_cpu.sh` 依次运行 CPU 测试、Lab 便携证据检查、11/6 两项回传的审计重跑、9B 服务研究的
两项审计与 Lab follow-up 证据重建、9B 报告表格与图、论文图和论文 PDF，全部写入临时目录，不改动已
跟踪文件。审计重跑需要外部证据根，路径不同就设置 `EVIDENCE_ROOT`；CLINC 基线测试需要独立的
`.venv-cheap-baselines`（按 `configs/cheap_baseline_requirements.txt` 安装），缺少时脚本会提示跳过。
`--strict` 把任何跳过（包括缺少原始输入）都算失败，`--only audits-9b` 等只运行指定步骤。

Git 保存源码、协议、精选报告及证据索引；模型、环境和原始运行目录不进 Git。
clone 后如需运行发布校验器，还须恢复固定 prepared 数据，见 [artifact 指南](docs/ARTIFACTS_ZH.md)。
GPU 服务器使用固定代码版本，现有运行目录与环境保持原位。

V1/V2/V3 科学源码与发布身份保留；被发布清单或 `SOURCE_SHA256.json` 绑定的文件保持原路径。
11/7 把已不再使用的恢复记录和规划文档移到 `docs/history/`，五份补丁收据移到
`records/patches/`，见[文档索引](docs/README.md)。已看过的 holdout 不再用于选方法并宣称
确认性改善。
