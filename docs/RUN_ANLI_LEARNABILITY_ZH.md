# ANLI 可学习互补性：第一阶段

本阶段运行本机输入使用审计，以及公司节点上的旧开发集 profile。它不训练新 router，也不评价新 holdout。
正式研究仍按[执行协议草案](ANLI_LEARNABILITY_PROTOCOL_ZH.md)的门槛推进。

## 本机审计

在本机源码仓库根目录运行，输出必须是一个不存在的新目录：

```bash
python3 scripts/audit_anli_exposure.py \
  --evidence-root /Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study \
  --output-dir .validation/anli-routing-phase1-2026-11-12/audit-runs/recheck-01
```

脚本固定官方源 SHA、六个历史输入文件和八个接受结果包的身份，扫描本机文本／ZIP 以及本地 Git 可达历史，
对 UID 做保守使用登记。旧 gold/sealed 文件只检索输入 ID，不解释或输出标签。原始源 ZIP 单列为来源审计，
本阶段审计工作目录单列为 AUDIT_ONLY，避免重跑时把候选清单误认成模型使用。

输出含来源与历史文件清单、使用位置、dev 排除原因、候选 dev、可复用训练候选及新 train 来源候选。
它们只有 ID／输入哈希，没有原文、gold 或模型成绩。候选数先于标签冲突与重复输入处理；不能直接当作已冻结训练样本。

该脚本的成功状态明确是 **UID census**，不单独证明 UID-less few-shot 没有使用，更不证明外部机器或上游模型未见。
正式放行还要合并独立文本检查和外部使用声明。修改任何历史文件后重新盘点，不能复用旧 inventory 冒充当前状态。

## 公司节点预检

运行入口是 `scripts/run_learnability_profile.sh`，具体源码身份与验证范围见[软件验收记录](../records/anli-learnability-preparation-2026-11-12/software-acceptance.json)。
源码必须放在独立 checkout；旧运行根保持 `/home/zhanj289/e6998/decision-study`。

进入已固定提交的新源码目录，在**实际 GPU allocation 的 code-server 终端**运行：

```bash
bash scripts/run_learnability_profile.sh /home/zhanj289/e6998/decision-study
```

脚本读取实际 Slurm 剩余时间，使用旧 controller、冻结的 vLLM 环境与三个已有 checkpoint，不重新安装或下载模型。
只运行旧 V3 的 90 条 official-train development 输入，依次启动 StartLux4B、Qwen9B、StartLux9B。
单模型显存份额 0.8，cap64，最大长度 8192，客户端并发 1。

Linux 的进程所有权自检先于任何模型启动。自检使用无 GPU 的临时子进程，检查多层、独立 session 的孤儿进程回收，
并确认无关诱饵进程没有被停止；不能建立可靠进程身份时直接阻断。Mac 软件测试不代替这个节点检查。
需要节点 Python／Linux 提供 `pidfd_open` 和 `pidfd_send_signal`；缺少时会停止并导出诊断，不能跳过该检查。
沿用已经恢复的 L40S、100 GB RAM allocation 即可，建议实际剩余时间至少四小时，便于保留启动与清理余量。

新输出为：

```text
/home/zhanj289/e6998/decision-study/results/aux-anli-learnability-profile-v1
```

完整成功或失败都会尝试输出带时间戳的 `learnability_profile_bundle-*.zip`，位于该根的 `exports/`。
下载终端最后打印的那一个包；不要复用旧时间戳的包。包内包含开发输入、原始响应、token、日志与资源证据，
不含权重、环境或新评价标签，不加入公开 Git。

回传后在与节点相同源码提交的 checkout 下，可对安全解压的新目录做只读验收：

```bash
python3 -B -m decision_study.learnability_profile audit --output-dir /absolute/path/to/extracted-profile
```

验收重解析原始回答、复算请求延迟汇总，核对 token／启动／退出／停止证据及源码身份；它不能重新观察远端显存或权重字节。
启动耗时只检查保存的单调时钟差值一致性，不把该记录当作独立外部测量。

## 中断与续跑

同一个 allocation 内正常重跑默认命令会核验并跳过已经接受的角色。失败或中断的尝试会保留；检查原因后显式重试：

```bash
bash scripts/run_learnability_profile.sh /home/zhanj289/e6998/decision-study --retry-failed
```

只有实际开启不同 Slurm allocation 后，才增加 `--new-allocation`：

```bash
bash scripts/run_learnability_profile.sh /home/zhanj289/e6998/decision-study --new-allocation --retry-failed
```

换 allocation 时会核实旧 job 已终止；无法核实则阻断，不靠修改 session 文件放行。
源码、输入、环境或模型身份改变也会阻断，应保留现有输出先调查，不能清空收据或改哈希强行续跑。

## 怎样解释结果

该 profile 证明三个角色能在当前节点配置下完成旧开发输入，提供 token 范围、启动和串行采集成本及停止证据。
它不证明正式 18,000 条拟合池所需时间，也不提供 SLO 或新准确率结论。正式成本估计还要检查新 train 候选的长度组成；
需要代表性采样时，只从已分配的 fitting/selection 块中取得。

回传验收完成后，才能固定正式数据分组、训练与选择预算、特征和统计规则，并交付正式采集入口。
