# Decision Study — 单人实验包

冻结模型：Kev-4B v1.0、StartLux-Decision-v1.1-4B-Instant、Qwen3.5-4B。
任务：CLINC banking＋OOS，以及 SQuAD 2.0 给定段落可答性。

已经遇到公司环境的 `Python.h` 缺失和旧缓存配额问题时，直接按
[Kev 修复说明](docs/RECOVER_KEV_ZH.md)运行恢复脚本；无需重下本项目权重。

**推荐主环境：公司 code-server，一张 L40S 级别 48 GB GPU。**
两个 notebook 和终端脚本调用同一套 Python 实现。第一次先完成 `smoke`，
再运行正式预测；本包没有包含真实模型实验结果。

## 开始

先读 [中文操作说明](docs/QUICKSTART_ZH.md)。在有 Python 3.11–3.13、Git、
NVIDIA driver 的 Linux code-server 中解压代码包，进入本目录：

```bash
python -m decision_study doctor --check-network --output-dir ./results/solo-v1
python -m pip install -e . uv
python -m decision_study run --job smoke --cost-mode provided \
  --session-hours-left 22 --output-dir ./results/solo-v1
```

`OUTPUT_DIR` 应放在跨 session 保留的目录。模型、源码和隔离 Python 环境默认
放在 `OUTPUT_DIR/runtime/work`；也可设置 `DECISION_STUDY_RUNTIME_DIR` 指向其他
保留目录。Colab notebook 使用本地临时目录放大型缓存、Drive 保存结果。

公司 GPFS home 已确认是 **100 GB 软配额 / 150 GB 硬上限**，且与其他文件共享。
继续模型安装前按中文操作说明检查占用，并确认大型缓存可用的存储目录；
不能把 150 GB 硬上限当作项目可用预算。已安装的环境不要直接移动。

`doctor` 从源码目录运行时只使用 Python 标准库，适合先确认公司环境的网络与
GPU。`--check-network` 检查少量公开元数据入口；成功不代表所有模型大文件/CDN
必然可达。公司持久目录及四个元数据入口已确认可用；完整模型运行仍需 smoke 验收。

公司提供算力时用 `--cost-mode provided`，只统计运行时间，不编造美元费用。
租用云 VM 时改用 `--cost-mode metered --hourly-usd 实际单价`。

22 小时是**用户指定的剩余时长预算**，并不是读取公司的到期时间。第一次创建
deadline，后续 JOB 复用同一个截止时间，不会每次重置。临近会话结束会停止模型
工作并保存结果；续开会话后执行：

```bash
python -m decision_study session --renew-session --session-hours-left 22 \
  --output-dir ./results/solo-v1
```

如果 session 已运行一段时间，传入真实剩余时间并留出保存余量。会话被平台直接
终止时，重新运行相同 JOB 会恢复已持久化的 shard。

## 正式运行

```bash
python -m decision_study run --job kev_full --cost-mode provided --output-dir ./results/solo-v1
python -m decision_study run --job startlux_full --cost-mode provided --output-dir ./results/solo-v1
python -m decision_study run --job qwen_test --cost-mode provided --output-dir ./results/solo-v1
python -m decision_study evaluate --output-dir ./results/solo-v1
python -m decision_study export --kind experiment --output-dir ./results/solo-v1
```

在相同硬件上做真实性能测量：

```bash
python -m decision_study run --job benchmark --cost-mode provided --output-dir ./results/solo-v1
```

每次运行自动输出 `exports/pilot_bundle.zip`、`experiment_bundle.zip` 或
`benchmark_bundle.zip`。部分结果和阻塞原因会明确标记；中途的实验包并不表示
三个模型都已完成。

终端批量入口：`JOB=all COST_MODE=provided bash scripts/run_vm.sh`。
`all` 完成正式预测和分析；真实性能使用单独的 `JOB=benchmark`，便于先检查质量结果。

## Notebook

- [Run_Experiments.ipynb](notebooks/Run_Experiments.ipynb)：code-server、Colab、VM 共用入口。
- [Analyze_and_Demo.ipynb](notebooks/Analyze_and_Demo.ipynb)：结果包检查、CPU 统计、图表、回放和可选实时 demo。

不想启动 GPU 时设置 `EXECUTE_GPU=False`。分析 notebook 可在普通 CPU 上运行；
实时 demo 需要原来已安装模型、已验收且配置匹配的 GPU 环境。

## 实验边界

| 项目 | 固定规则 |
|---|---|
| 基础推理 | Kev 6,250；StartLux 6,250；Qwen 3,450 |
| Calibration / validation | 严格分离；SQuAD 按文章分池 |
| SQuAD 标签冲突 | 相同可见输入有相反标签时整组隔离，保留日志 |
| Temperature | 发布概率上的额外正标量，仅 calibration 拟合 |
| Gate | validation 经验错误 ≤5%、接受 ≥50；无可行解则全转交 |
| Test | gate 冻结；不按测试结果重调提示或阈值 |
| Inference failures | 留在分母，本地失败强制转交，最终失败计错 |
| Probability metrics | 有效完整分布上计算，同时报告失败和有效数量 |
| Uncertainty | 2,000 次配对 bootstrap；SQuAD 文章 cluster，CLINC 类别分层 |
| Timing | 真实请求，包含实际 fallback；缓存分析不充当推理计时 |
| Resume | 配置、GPU 类型、版本一致才继续；失败终态也不无限重试 |

细节见 [数据协议](docs/data_protocol.md)、[指标协议](docs/metrics.md)、
[运行环境](docs/runtime.md)。模型上游数据暴露仍是研究限制：CLINC 参与过部分模型
开发评估，SQuAD/Qwen 的未披露暴露不能视为不存在。

## 验证与限制

本机已验证数据下载／哈希／划分、CPU 单元测试、人工测试夹具的完整统计绘图流程，
以及 notebook 的 CPU 路径。测试夹具输出显著标注 `SOFTWARE TEST FIXTURE`，
不代表任何模型准确率或速度。详见 [验证记录](VALIDATION.md)。

**尚未在 GPU 上执行模型安装、参考推理或性能实验。** 公司驱动、外网／镜像访问、
CUDA wheel 与可用显存由 `doctor`、安装后的 BF16 GPU probe 和 `smoke` 验收。
发现问题时先返回 `pilot_bundle.zip`，不要修改版本锁或跳过双向注意力检查。

开发验证：

```bash
python -m pip install -e '.[test,notebook]'
python -m pytest -q
python -m decision_study self-test --output-dir ./outputs/software-check
```

所有原始数据与上游模型采用其各自许可证。源数据出处和版本信息保留在 manifest
及文档中。此代码包不上传数据、不启动外部云实例、不发送邮件或消息。
