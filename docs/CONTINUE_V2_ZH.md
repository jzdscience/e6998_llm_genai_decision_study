# 公司服务器运行 v2

本包交付的是可恢复的 v2 代码、预先准备的数据和 CPU Decision Lab。
**v2 的真实 GPU 采集、性能测量和 holdout 评价尚未运行验收。** 原 v1 已完成，保持不变。
本轮执行范围见 [V2_PROTOCOL_ZH.md](V2_PROTOCOL_ZH.md)，不包含 9B、ANLI 或实时网页。

交付文件为 [decision-study-v2-ready.zip](../../decision-study-v2-ready.zip)。ZIP 内以
`decision-study/` 为顶层目录，含独立 v2 模块、脚本、文档及准备好的
`results/solo-v2/data/`、`results/solo-v2/sealed/`。公司机器无需再次下载/抽样 SQuAD，
也无需安装 Gradio 或改变既有模型环境。

## 先确认真实新 allocation

旧记录的 session deadline 已过期。只有实际获得新的 code-server/GPU allocation 后，
才能初始化 v2 的运行窗口。下方 `--new-allocation` 表示操作员确认，不会替平台创建作业，
也不会自动探测到期时间。请填写平台显示的**实际剩余小时数**。

将 ZIP 放到公司机器 `/home/zhanj289/e6998/` 后运行：

```bash
cd /home/zhanj289/e6998
unzip -n decision-study-v2-ready.zip
cd decision-study
.venv-controller/bin/python scripts/verify_v2_release.py
```

`unzip -n` 保留已有文件。若机器上有内容不同的旧 v2 文件，发布核验必须拒绝继续；
不要改用 `unzip -o` 覆盖，也不要删除既有结果或伪造哈希。保留报错和文件身份，核对后
再决定如何建立另一份工作副本。校验还检查依赖的 v1 冻结源文件与工具链入口。

核验成功且已有真实新 allocation 后：

```bash
read -rp "平台显示的实际剩余小时数: " HOURS_LEFT
bash scripts/run_v2.sh init-runtime \
  --new-allocation --session-hours-left "$HOURS_LEFT" \
  --v1-dir results/solo-v1 --output-dir results/solo-v2

bash scripts/run_v2.sh status \
  --v1-dir results/solo-v1 --output-dir results/solo-v2
```

初始化写入独立的 `results/solo-v2/runtime-control/`，复用 v1 已验收的权重和环境路径。
它不修改原 v1 session、prediction、calibration 或 analysis，也不重新安装 CUDA 工具链。
后续 GPU 命令在启动时执行 v3 工具环境 preflight。

## 推荐的一次流水线

```bash
bash scripts/run_v2.sh pipeline \
  --v1-dir results/solo-v1 --output-dir results/solo-v2
```

执行顺序固定为：

1. Qwen 对原 calibration/validation 的 2,800 条 fitting 预测。
2. CPU 拟合，validation 选择并冻结在线政策。
3. StartLux 和 Qwen 各对 4,000 条新 SQuAD holdout 预测，此时不评分。
4. 在原 validation 性能池上执行 36 个真实开放到达单元，使用刚冻结的选定政策。
5. 验证 freeze 与预测身份，首次打开 holdout 标签，计算主比较及文章簇区间。
6. 导出结果包。

预测 job 会在新的 allocation 上用既有 development 样例做 serial 一致性检查，
并检查精确 tokenizer 长度。异常不会触发静默换模型、改精度或截断数据。

该流程会多次启动/停止模型。36 个性能单元的 offered window 合计 18 分钟，完整会话
还包括权重加载、smoke、主采集、预热和 drain；这不是总耗时承诺。不要因为看到已保存
的部分结果就认定全部完成。

## 中断后的分步恢复

同一个仍有效的 allocation 内，不需要重复初始化。先查看状态，然后重新运行相同
`pipeline` 即可复用配置匹配的完成 job/性能单元。预测分片采用持久 journal 恢复；
配置或源文件变化会拒绝混合，而不是覆盖旧结果。

需要单独运行某一步时，命令如下。保持相同的目录和发布版本。

```bash
# 1. 补齐 Qwen fitting；已完成且身份匹配时返回既有收据
bash scripts/run_v2.sh run --job qwen_fit \
  --v1-dir results/solo-v1 --output-dir results/solo-v2

# 2. CPU 拟合与冻结；不需要模型服务
bash scripts/run_v2.sh fit \
  --v1-dir results/solo-v1 --output-dir results/solo-v2

# 3. 新 holdout 只采集，不评分
bash scripts/run_v2.sh run --job holdout_startlux \
  --v1-dir results/solo-v1 --output-dir results/solo-v2
bash scripts/run_v2.sh run --job holdout_qwen \
  --v1-dir results/solo-v1 --output-dir results/solo-v2

# 4. 使用已冻结政策真实测量；匹配且完整的测量单元会跳过
bash scripts/run_v2.sh benchmark \
  --v1-dir results/solo-v1 --output-dir results/solo-v2

# 5. CPU 评价；必须已有有效 freeze 和两个完整 holdout job
bash scripts/run_v2.sh evaluate \
  --v1-dir results/solo-v1 --output-dir results/solo-v2

# 6. 随时导出已完成部分或诊断，导出不等于全部完成
bash scripts/run_v2.sh export \
  --v1-dir results/solo-v1 --output-dir results/solo-v2
```

如 allocation 已经结束，先在平台获得真实新 allocation，再用新的实际剩余小时数运行
`init-runtime --new-allocation`。输出仍可使用 `results/solo-v2`，已完成身份匹配的
结果会保留。不要把 expired window 的续期当作修复异常的通用操作。

冻结后 `fit` 只核验并返回现有政策；相同身份的 `evaluate` 可以幂等核验，
不再挑选政策或重新抽样。查看 holdout 后改变政策/样本不属于恢复，必须另立前瞻协议。

## 结果与需要回传的文件

主要文件位于：

| 文件/目录 | 作用 |
|---|---|
| `predictions-jobs/qwen_fit/` | 2,800 条预测、smoke 与 completion 收据 |
| `predictions-jobs/holdout_startlux/` | 4,000 条 StartLux holdout 预测 |
| `predictions-jobs/holdout_qwen/` | 4,000 条 Qwen holdout 预测 |
| `policy_freeze.json` | validation 选定政策、阈值、拟合与源代码身份 |
| `openloop/` | 36 个单元及原始 trace、占用事件、测量设置 |
| `holdout_access.json` | 首次评价使用的 freeze/manifest 身份 |
| `evaluation.json` | 主比较、95%文章簇区间及描述性指标 |
| `diagnostics/last_error.json` | 最近失败的阶段和 traceback；旧错误不自动代表新阶段失败 |
| `exports/v2_bundle.zip` | 需要下载回传的结果/诊断包 |

回传公司机器上的：

```text
/home/zhanj289/e6998/decision-study/results/solo-v2/exports/v2_bundle.zip
```

流水线在主要采集完成后以及正常/异常退出时尝试导出。即使失败，也先回传已有包；
如完整 code-server allocation 被平台直接终止，可在下次 allocation 运行 `export` 收集
持久目录中的证据。无需打包模型权重、虚拟环境或整个 home。

回传后核对文件哈希、逐 job 样本身份与失败数、36 个性能单元的配置/trace、freeze 绑定
及评价分母，才能确认真实完成。此说明不提前宣称新准确率、goodput 或 GPU 内存结果。

## CPU 演示独立运行

当前本机可用的 Decision Lab 展示 v1 真实结果，不依赖上述 GPU 作业。完整说明见
[LAB.md](LAB.md)。公司 controller 和模型环境不需要安装网页依赖。

```bash
.venv-lab/bin/python -B -m decision_study.v2 lab \
  --bundle reports/decision-lab-2026-11-02 --port 7860
```

这条命令用于已经安装独立 `.venv-lab` 的本机；当前页面不把尚未运行的 v2 结果伪装为完成，
也没有实时模型推理按钮。
