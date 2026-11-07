# Decision Study

研究何时直接使用 decision model、何时转交给生成式模型值得其质量与服务成本。
V1、V2、V3 三轮共 44,450 条正式模型预测；后续的 StartLux9B 规模对照另有 3,200 条，CLINC 轻量
基线有 7,400 条 CPU 分类器预测，CPU/GPU 服务测量有 469,804 次请求。以上均已验收。

- [当前进度与下一步](docs/CURRENT_STATE.md)
- [结果与图表](reports/README.md)
- [源码仓库和服务器协作](docs/REPO_MANAGEMENT_ZH.md)
- [node 首次配置与 Git 更新命令](docs/NODE_GIT_ZH.md)
- [数据、环境与结果包的取得方式](docs/ARTIFACTS_ZH.md)
- [原计划与后续扩展对照](docs/POST_V3_SCOPE_ZH.md)
- [CPU Decision Lab](docs/LAB_CURRENT_ZH.md)（已接入 V1–V3，运行 `bash scripts/run_lab.sh`）
- [CLINC 轻量基线结果](reports/cheap-baselines-2026-11-04/RESULTS_ZH.md)、
  [服务测量与 StartLux9B 规模对照结果](reports/followups-2026-11-06/RESULTS_ZH.md)

| 阶段 | 任务与内容 | 状态 |
|---|---|---|
| V1 | CLINC / SQuAD 可答性，Kev4B / StartLux4B / Qwen4B，校准、级联与性能 | 已验收 |
| V2 | 新 SQuAD holdout，gain / confidence / class-conditional 与开放到达测量 | 已验收 |
| V3 | ANLI，StartLux4B 前级、Qwen4B / Qwen9B 后级 | 已验收；顺序质量实验 |
| CLINC 轻量基线 | TF-IDF / MiniLM 加 LR，400 与 1,600 个训练标签 | 已验收；回顾性对照 |
| 服务测量 | MiniLM400/1600（CPU）、StartLux4B、Qwen4B，5/25/100 请求/秒，500 ms SLO | 已验收 |
| 规模对照 | StartLux9B 在 V3 的 ANLI test 上直接预测 | 已验收；回顾性对照 |

V2/V3 的预定主比较均未显示统计上的改善。V3 positive_gain 选择零转交，与直接 StartLux
相同；实验仍采集了完整后级预测。各项结论及限制见原始结果报告。

## 本地开发

使用 Python 3.11–3.13，在本目录建立自己的环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Git 保存源码、协议、精选报告及证据索引；不保存模型、环境或原始运行目录。
clone 后如需运行发布校验器，还须恢复固定 prepared 数据，见 [artifact 指南](docs/ARTIFACTS_ZH.md)。
GPU 服务器使用固定代码版本，现有运行目录与环境保持原位。

V1/V2/V3 科学源码与发布身份保留。新的研究和前端改动从本仓库继续开发，已看过的 holdout
不再用于选方法并宣称确认性改善。后续扩展尚未冻结为 V4。

早期安装与恢复文档保留用于历史复现；其中的旧 deadline、报错状态和“尚未运行”描述
不代表当前进度。[最初 V1 README](docs/LEGACY_V1_README.md) 的原内容另行保留。
