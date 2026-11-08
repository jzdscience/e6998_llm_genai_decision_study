# Decision Study（中文说明）

研究何时直接使用 decision model、何时转交给生成式模型值得其质量与服务成本。
V1、V2、V3 三轮共 44,450 条正式模型预测；后续的 StartLux9B 规模对照另有 3,200 条，CLINC 轻量
基线有 7,400 条 CPU 分类器预测，CPU/GPU 服务测量有 469,804 次请求。以上均已验收。
English overview: [README.md](README.md)。

- [课程报告初版（ACL 双栏）](paper/main.pdf)，源码与编译说明见 [paper/README.md](paper/README.md)
- [当前进度与下一步](docs/CURRENT_STATE.md)、[实验与排障日志](docs/EXPERIMENT_LOG.md)
- [文档索引](docs/README.md)、[常见故障](docs/TROUBLESHOOTING.md)
- [结果与图表](reports/README.md)
- [数据、环境与结果包的取得方式](docs/ARTIFACTS_ZH.md)
- [源码仓库和服务器协作](docs/REPO_MANAGEMENT_ZH.md)、[node 首次配置与 Git 更新命令](docs/NODE_GIT_ZH.md)
- [CPU Decision Lab](docs/LAB_CURRENT_ZH.md)（已接入 V1–V3，运行 `bash scripts/run_lab.sh`）

| 阶段 | 任务与内容 | 状态 |
|---|---|---|
| V1 | CLINC / SQuAD 可答性，Kev4B / StartLux4B / Qwen4B，校准、级联与性能 | 已验收 |
| V2 | 新 SQuAD holdout，gain / confidence / class-conditional 与开放到达测量 | 已验收 |
| V3 | ANLI，StartLux4B 前级、Qwen4B / Qwen9B 后级 | 已验收；顺序质量实验 |
| CLINC 轻量基线 | TF-IDF / MiniLM 加 LR，400 与 1,600 个训练标签 | 已验收；回顾性对照 |
| 服务测量 | MiniLM400/1600（CPU）、StartLux4B、Qwen4B，5/25/100 请求/秒，500 ms SLO | 已验收 |
| 规模对照 | StartLux9B 在 V3 的 ANLI test 上直接预测 | 已验收；回顾性对照 |

V2/V3 的预定主比较均未显示统计上的改善。V3 positive_gain 选择零转交，与直接 StartLux
相同；实验仍采集了完整后级预测。各项结论及限制见原始结果报告和课程报告。

## 本地环境与复现

使用 Python 3.11–3.13。主环境装演示和测试依赖：

```bash
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[lab,test]'
bash scripts/reproduce_cpu.sh
```

`reproduce_cpu.sh` 依次运行 CPU 测试、Lab 便携证据检查、11/6 两项回传的审计重跑、论文图和
论文 PDF，全部写入临时目录，不改动已跟踪文件。审计重跑需要外部证据根，路径不同就设置
`EVIDENCE_ROOT`；CLINC 基线测试需要独立的 `.venv-cheap-baselines`（按
`configs/cheap_baseline_requirements.txt` 安装），缺少时脚本会提示跳过。

Git 保存源码、协议、精选报告及证据索引；模型、环境和原始运行目录不进 Git。
clone 后如需运行发布校验器，还须恢复固定 prepared 数据，见 [artifact 指南](docs/ARTIFACTS_ZH.md)。
GPU 服务器使用固定代码版本，现有运行目录与环境保持原位。

V1/V2/V3 科学源码与发布身份保留；被发布清单或 `SOURCE_SHA256.json` 绑定的文件保持原路径。
11/7 把已不再使用的恢复记录和规划文档移到 `docs/history/`，五份补丁收据移到
`records/patches/`，见[文档索引](docs/README.md)。已看过的 holdout 不再用于选方法并宣称
确认性改善。
