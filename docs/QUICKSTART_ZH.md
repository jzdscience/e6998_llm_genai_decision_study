# 从代码包到可回传结果

先检查机器与网络，返回初步 `pilot_bundle.zip`；网络通过后再跑模型小样本
验收 `smoke`，更新同一个 pilot 包。验收通过后再跑全量。
正式实验采用三个冻结模型、两个任务，共 15,950 条基础预测；后续校准、
路由和统计复用这些结果。真实性能需要额外运行。

若已有 `fatal error: Python.h: No such file or directory`，使用
[Kev 专项恢复步骤](RECOVER_KEV_ZH.md)，无需从头下载模型。

## 当前进度：Kev 串行验收已通过，开始正式预测

2026-10-30 最新公司 pilot 已通过 Kev 的 BF16、Triton 和 128 条串行接口检查。
更新包修复服务重启时端口探测的 TIME_WAIT 误报，并明确标记未通过的并发验证。
将 `decision-study-kev-ready.zip` 上传到 `/home/zhanj289/e6998/` 后执行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-kev-ready.zip
cd decision-study
source .venv-controller/bin/activate
python -m decision_study run --job kev_full \
  --cost-mode provided --output-dir ./results/solo-v1
```

该命令复用环境和权重，因代码 hash 更新而重新做 smoke，然后完成 2,800 条
calibration/validation、冻结温度/gate，再预测 3,450 条 test/stress。
结果每 25 条持久化；中断后重跑相同命令恢复。结束后返回
`results/solo-v1/exports/experiment_bundle.zip`，仅 Kev 完成时 partial 属于预期。
此前三个旧模型缓存已经清理；无需再次执行 repair-python。
原 session 截止时间继续有效，只有真实新开 session 后才按下文续期。
完成 Kev 后再检查 home 当前配额占用，安排需要新增环境的 StartLux。

## 首选：公司 code-server + 单张 L40S

Notebook 默认针对公司提供的 L40S 48 GB、24 小时可续期 allocation。
先将代码 ZIP 上传至 code-server 并解压，在终端进入项目根目录。
使用公司允许的 Python 3.11–3.13 环境，不需要 sudo 或修改平台 CUDA。
你已确认工作目录会在会话间保留；把代码和 `results/solo-v1` 放在该持久目录。
首次运行先用 Python 标准库检查环境和依赖源的可达性，
不先安装依赖、下载数据或模型（本次公司环境已经完成这一步）：

```bash
python -m decision_study doctor --check-network --output-dir ./results/solo-v1
python -m decision_study export --kind pilot --output-dir ./results/solo-v1
```

先返回生成的 `results/solo-v1/exports/pilot_bundle.zip`。这个初步包包含机器与网络
检查结果，明确标为部分完成，不表示模型已通过 smoke。网络检查只发送有界的公开
依赖源请求，不发送公司凭据。确认可访问官方依赖源后，再执行：

```bash
python -m pip install -e . uv
python -m decision_study prepare --output-dir ./results/solo-v1
python -m decision_study run --job smoke --cost-mode provided --hourly-usd 0 \
  --session-hours-left 22 --gpu-memory-utilization 0.40 \
  --project-dir . --output-dir ./results/solo-v1
python -m decision_study export --kind pilot --output-dir ./results/solo-v1
```

若偏好 notebook，打开 `Run_Experiments.ipynb`，保留以下默认值即可：

```python
PLATFORM = "code-server"
COST_MODE = "provided"
HOURLY_USD = 0.0
GPU_MEMORY_UTILIZATION = 0.40
SESSION_HOURS_LEFT = 22.0
NEW_SESSION = False
```

`provided` 表示算力由公司分配，记录 GPU 时间而不虚构美元费用；不是无限算力。
新的 24 小时 allocation 默认只使用未来 22 小时，给导出、备份和续期留两小时。
如果 session 已经使用了一段时间，填写不超过实际剩余时间的值。
`runtime/session.json` 保存绝对截止时间，重新运行命令不会自动延长它。

**只有平台实际确认续期或重新分配后**，运行：

```bash
python -m decision_study session --renew-session --session-hours-left 22 \
  --output-dir ./results/solo-v1
```

Notebook 对应设置为 `NEW_SESSION=True`；执行一次后改回 `False`。
这只记录新的运行窗口，不会调用公司平台续期，也不能让已到期 allocation 继续运行。

公司网络需允许下载官方数据、Hugging Face 权重、GitHub 源码与 Python 依赖。
2026-10-30 的公司 doctor 回传已确认 L40S 和四个元数据入口可达；这不保证
大模型分片下载一定成功。
使用你已确认会保留的工作目录作为 `OUTPUT_DIR`，每阶段仍导出并下载结果 ZIP。
模型环境/缓存若也位于同一持久目录，可减少后续安装；若个别缓存丢失，按锁定
版本重建。尚未通过 pilot 前，不承诺一次 24 小时内跑完全部实验。

### 公司 home 配额：先检查，再继续安装

用户收到的 2026-10-30 平台邮件明确：`/home/zhanj289` 的 block soft quota 是
100 GB，hard quota 是 150 GB；邮件时占用 100.9 GB，软配额宽限期一周。
150 GB 不是可以长期使用的配额，也不是实验包独享的剩余空间。
新的 GPU/code-server session 不会清零 home 存储占用。

超软配额时先停止安装/下载。不要删除整个结果目录，也不要根据下载日志里的
MiB 数字或各子目录独立 `du` 的相加结果判断实际占用。下面都是只读检查：

```bash
du -x -h --max-depth=1 /home/zhanj289 | sort -h
du -x -h --max-depth=2 /home/zhanj289/e6998/decision-study/results/solo-v1/runtime/work | sort -h
df -h /home/zhanj289 /tmp
```

`df` 显示文件系统空间，不能替代用户 quota；`du` 用来定位大目录，不能完全
去重 copy-on-write 共享块，不能把两个有重叠范围的输出再相加。
GPFS 配额以平台工具/邮件为准。

优先从公司存储指南确认你获分配的 scratch、project 或节点本地目录，以及配额、
session 间保留情况和自动清理期限。大型环境和权重可以由
`DECISION_STUDY_RUNTIME_DIR` 指向该目录，小型 `OUTPUT_DIR` 保留在持久存储。
路径必须实际确认；不要假定 `/tmp` 或任意 `/scratch` 可用。

现有安装记录、Python 环境入口和依赖 freeze 包含绝对路径。已经安装后不要
直接 `mv` 环境或只改变量就继续正式预测；先保存日志/版本记录，制定重建与清理
顺序。正式预测开始前可以重跑 smoke；正式预测开始后还须验证配置一致性。
未经占用报告确认，不自动删除用户的其他项目、Conda 环境或共享缓存。

## 第一次使用 Colab

1. 在 Colab 上传并打开 `notebooks/Run_Experiments.ipynb`。
2. 选择 GPU runtime。优先支持 BF16 的 L4/A100；最终双模型 cascade 建议固定
   48 GB 云 GPU。T4 不能当成相同 BF16 配置的替代品。
3. 将参数设为 `PLATFORM="colab"`、`COST_MODE="metered"`、`JOB="smoke"`，填写正数
   `HOURLY_USD`：云 VM 用实际小时价；Colab 用当前 GPU 的 compute-unit 消耗
   换算为美元/小时的估计。估计会写入预算记录，不代表服务商账单。
   48 GB 主方案使用 `GPU_MEMORY_UTILIZATION=0.40`；24 GB Colab 只做单模型
   预测时，可在第一次 smoke 前改为 `0.80`。不要把该设置用于同卡双模型 cascade。
4. 点击“全部运行”。提示时上传 `decision-study-bundle.zip`，再授权挂载自己的
   Google Drive。默认结果目录是 `MyDrive/decision-study/solo-v1`。
5. 等待安装和 smoke，保存最终的 `pilot_bundle.zip`。初次下载模型、编译内核可能
   明显慢于后续推理；不要因为安装时间长就跳过正确性验收。

前五分钟的工作是选择 GPU、填写参数并上传代码；这里不承诺五分钟内装完模型。
GPU 兼容性、显存需求与真实费用以 pilot 为准。

模型权重、源码 checkout 和隔离环境放在 Colab 本地
`/content/decision-study-runtime`；小型结果与运行锁保存在 Drive。
请勿把几十 GB 的模型环境复制进 Drive。
Notebook 会一并安装 `uv`，它负责创建相互隔离的模型环境。

## 普通云 VM / Jupyter

推荐规格是支持 BF16 的 NVIDIA 48 GB 单卡、64 GB 系统内存和约 150 GB 磁盘。
准备 NVIDIA driver、Python 3.11–3.13、Git。无需预先把 Kev、StartLux 与 Qwen
的冲突依赖装进一个环境。

解压代码包，进入项目目录打开同一个 notebook：

```python
PLATFORM = "vm"
JOB = "smoke"
COST_MODE = "metered"
HOURLY_USD = 你的实际小时单价
PACKAGE_DIR = "/完整路径/decision-study"
OUTPUT_DIR = "/持久磁盘/decision-study-results/solo-v1"
```

也可使用项目根目录提供的 VM 启动脚本，或直接调用 CLI：

```bash
python -m pip install -e . uv
python -m decision_study doctor --output-dir ./results/solo-v1
python -m decision_study prepare --output-dir ./results/solo-v1
python -m decision_study run --job smoke --cost-mode metered --hourly-usd 1.00 \
  --session-hours-left 22 \
  --gpu-memory-utilization 0.40 --project-dir . --output-dir ./results/solo-v1
python -m decision_study export --kind pilot --output-dir ./results/solo-v1
```

其中 `1.00` 只是命令格式示例，运行前替换为实际单价。不要将示例数字当报价。

## 验收通过后的运行顺序

所有任务使用同一个 `OUTPUT_DIR`。每次修改 `JOB` 并重跑 notebook 即可：

| JOB | 工作 | 回传结果 |
|---|---|---|
| `smoke` | 环境、token 长度、输出协议、参考路径与服务路径检查 | `pilot_bundle.zip` |
| `kev_full` | Kev 校准/验证 → 冻结温度与 gate → 测试 | `experiment_bundle.zip` |
| `startlux_full` | StartLux 同样流程 | 更新后的 `experiment_bundle.zip` |
| `qwen_test` | Qwen 主测试与额外 OOS，无无用的校准推理 | 完整 `experiment_bundle.zip` |
| `benchmark` | 固定 GPU 上真实 direct/cascade、串行与并发计时 | `benchmark_bundle.zip` |

单个模型完成后的实验包仍可能是部分结果，`status.json` 会明确说明。
完整结果需要 Kev、StartLux、Qwen 三个正式任务都完成。

质量预测每次只驻留一个模型；live cascade 才同时驻留 decision+Qwen。
原生概率/额外温度、随机路由和 oracle 等准确率分析不重复调用 GPU。

## 中断、错误与费用

- 重跑相同 JOB、相同输出目录会恢复；不要删除记录后反复重试失败样本。
- 输出以小 shard 持久化。中断可能重跑尚未同步的最后一个 shard；不会把缺失
  样本伪装成完成。
- 若 Colab 换了 GPU 型号，或你改了模型、精度、后端、prompt，配置检查会停止。
  先返回诊断，不要修改 hash 来强行合并。
- 从 24 GB Colab 迁移到 48 GB VM 时使用新的 `OUTPUT_DIR`，重新运行 smoke 和
  正式质量预测，再做该配置的 live benchmark。旧结果保留作独立运行，不混合。
- 缺少双向注意力插件、非法概率、token 超限等会阻止正式运行；低准确率和
  零本地覆盖率本身是研究结果。
- 阶段预算估算达到限额时停止并导出。预算工具不能读取所有服务商实际费用，
  也不能自动终止云平台计费。
- 保存结果后停止 VM，或断开并删除不再使用的 Colab runtime。只关闭浏览器
  或结束 Python 并不保证停止付费。

如果只想在 CPU 上检查目录、数据与包安装，把 `EXECUTE_GPU=False`。
这不会产生模型预测；不能把 CPU 检查包当作通过 GPU 验收。

## 返回结果与 demo

把完整 ZIP 返回，而不是只发 accuracy 截图。包中保留 sample ID、预测分布、
错误、配置、清单、温度、gate 和必要计时记录，后续才能复算配对统计与错例。
代码包和结果包不包含模型权重或你的云账户凭据。

在 CPU 环境打开 `Analyze_and_Demo.ipynb`，加载同一代码包和结果 ZIP：

1. 验证压缩包校验和并检查缺失阶段。
2. 在完整结果上复算统计和图表。
3. 回放已记录的真实决策；界面明确标为 `RECORDED REPLAY`。
4. 只有需要现场真实调用时才设置 `RUN_LIVE=True`，并回到原来已验收的 GPU
   runtime，保留其模型环境和权重，将 `OUTPUT_DIR` 指向原实验目录。新的 CPU
   机器仅导入结果 ZIP 后可以分析/回放，不能据此直接启动 live 模型。

首次导入结果 ZIP 使用新目录。之后复算同一结果时设置 `USE_EXISTING_OUTPUT=True`
并保留该目录路径，避免把更新后的分析状态覆盖成旧 ZIP 的状态。
在原 GPU 上运行 live demo 也使用 `USE_EXISTING_OUTPUT=True`，不重新导入 ZIP。

SQuAD live demo 必须同时给完整 passage 和 question；它只判断给定段落是否
支持答案，不执行检索或生成答案。实时与回放模式都复用正式实验的 adapter 和 gate。

第一轮拿到数据后再决定文章重点：模型能接受多少请求、Qwen 是否真正补救错误、
转交开销是否抵消收益。无需预设 decision 模型或 cascade 必然获胜。
