# V3：ANLI 上的后级规模与可选转交

2026-11-03，先于本轮模型预测与评价制定。机器可读协议为
[v3_protocol.json](../configs/v3_protocol.json)。本轮是独立新实验，不修改已完成 V1/V2
的模型、源码、政策或评价。本文记录设计与完成标准，不把本地软件测试当成 GPU 结果。

## 研究问题与唯一主比较

V2 的固定预算 gain 在独立 holdout 没有显示优于 confidence；它有时会为用满预算而
转交预计净收益为负的请求。本轮检验在一个新任务上，**只在预计正收益时转交**是否优于
相同 validation 预算上限的 confidence 政策。同时保留 4B/9B 后级对照，观察规模增加
是否带来直接质量与可利用的互补性。预先固定 Qwen9B 为主要后级，不根据 validation
或 test 的胜负改换主要后级。

唯一主终点为 ANLI 全部官方 test 上：

`accuracy(positive_gain, Qwen9B 后级) − accuracy(confidence, Qwen9B 后级)`

它比较两种冻结在线政策，目标转交预算均为 validation 的 10% 上限。positive_gain
可以不用满预算；实际调用量必须另报。不是严格等实际调用量或等美元成本的比较。
区间跨零只表示未确认改善，不证明等效或政策无效。若没有正收益请求，返回不转交政策
本身就是合法结果；不能为得到非零转交或正结果继续改阈值。

## 新数据与隔离

本地固定 SQuAD2 train 的 442 篇文章与 dev 的 35 篇已全部被 V1/V2 使用。剩余问题
不能再称为新文章 holdout。因此本轮使用项目尚未运行的 ANLI 三分类推断，覆盖 R1/R2/R3，
不按结果挑 round。公开任务不保证模型预训练未见；本项目未用和模型从未见过须分开。

| 角色 | 来源和数量 | 用途 |
|---|---|---|
| development | 官方 train 每 round×类别 10，共 90 | 精确 token 检查与 serial smoke，无质量停止规则 |
| calibration | 官方 train 每 round×类别 200，共 1,800 | 两端正确性模型拟合 |
| validation | 官方 train 每 round×类别 100，共 900 | 冻结转交阈值，不选主模型/主政策 |
| holdout | 官方 test 原样保留 3,200，R1/R2/R3 为 1,000/1,000/1,200 | 首次主要评价 |

原 ZIP SHA256 固定为 `e5c058f2bb4e6190b0651badca2c590e45db95248f8b28cde674615ee40820bf`。
数据审核发现官方 train/dev 与 test 存在重复 premise，不能仅依靠官方 split 名称防止
材料泄漏。先保留 test 全部 UID；从 train 拟合候选中排除 test 的规范化 premise/input。
冲突标签的重复训练 input 整组排除；同标签重复保留 UID 字典序首项。剩余 premise
按固定 seed 的哈希分配到 calibration/validation/development，比例为 60/30/10，再在
各角色按 round×类别定额选题；不临时补池。四角色之间 premise 和完整 input 不重叠。
官方 dev 不进入本轮训练、选择或测试，只做来源审核。所有排除和数量均记入 manifest。

holdout 有 2,854 个规范化 premise 组，跨 round 的同 premise 共享 group_id。
输入只有 premise、hypothesis 和固定类别定义，严禁包含 `reason`、gold 或攻击元数据。
cal/val/dev 标签与 holdout 标签分开保存；准备阶段为预定分层及验证标签合法性会读取
源标签，不计算模型得分。采集、fit、status 不解析 sealed gold。封存是流程约束，不是加密。

## 模型和推理身份

| 角色 | checkpoint | 固定 revision | 显存份额 |
|---|---|---|---:|
| startlux4b | StartLux-Decision-v1.1-4B-Instant | `416088f77cfce09633906a3675c7cd2e5dabcf10` | 0.40 |
| qwen4b | Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 0.40 |
| qwen9b | Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | 0.80 |

沿用已验收 vLLM 0.31.0 环境、BF16、8,192 token 上限、禁止静默截断。Qwen 输出为
受约束 label JSON，temperature=0、seed=6998、thinking=false、最多 64 tokens；StartLux
使用原生 choice、think_mode=never。精确 tokenizer 预检必须覆盖该 job 全部输入与开发题。
出现超长样本、依赖不符或资源不足即阻断，不换样本、量化、降精度或悄悄换模型。

三个模型**顺序独立运行**。复用现有 4B 权重与已安装环境；9B 下载到独立 v3 cache，
不重装环境或写入旧模型目录。0.80 是新 9B 单独运行的预定 reservation，并不是已证实
能装下或能与 4B 同时驻留。9B checkpoint 本身约 19.3GB BF16/F32 参数数据，下载与
缓存余量、GPU/host 内存仍须实际检查；100GB RAM 不等于增加显存。

每次启动用 90 个 development 输入做两次 serial 响应检查，保留原响应、概率与失败记录。
这验证同模型重复执行和接口，不是另一个实现的正确性证明。正式预测单并发、最多两次
尝试；新输入超过 50 后终止错误率大于 1% 则停机保留证据。该规则只看运行错误，不看
holdout 准确率。不能把开发 smoke 的通过描述为 9B 确认优于 4B。

## 政策

两种后级分别拟合以下四个预定政策，不进行主候选择优。

| 政策 | 评分与转交 |
|---|---|
| confidence | `1 − StartLux最大概率`，按完整同分组冻结 validation 10% 上限 |
| class_conditional | 基于置信度 logit 与预测类别 one-hot 拟合前级正确性，按预计错误分数转交 |
| fixed_gain | 两个正确性模型之差，按完整同分组用满不超过预算的最大可行集合 |
| positive_gain | 与 fixed_gain 共用拟合 head，但只允许分数严格大于零的请求进入阈值选择；可不转交 |

gain 特征为 released 最大概率的 logit、三类预测 one-hot、`log1p(state 字符数)`。
标准化与两个正确性 logistic head 仅使用 calibration；固定 L2=0.01，截距不惩罚。
单一目标类别时使用 Beta(1,1) 平滑常数模型。validation 只决定冻结门槛；test 不做 top-k
或重新算配额。前级运行失败强制转交，另报造成的预算偏离；失败不被删除或算成正确。

政策 freeze 绑定协议、源码、数据及三种 fitting 预测。三个 fit jobs 共 8,100 次正式预测
全部验证后才冻结，然后三种 holdout jobs 共 9,600 次预测。合计 **17,700 次正式预测**，
另有开发 smoke；中断只补未持久化输入，不覆盖已有终止结果。

## 评价与停止边界

在全部六个 job 的配置、样本、shard、原响应、smoke、completion 校验通过后，先记录
绑定 freeze 的首次访问凭据，再读取 sealed gold。主差值使用 2,000 次按 premise 组
配对 percentile bootstrap，种子 6998；重采样整组全部题目，计算问题等权 pooled 差值。
跨 round 同组一起采样，R3 1,200 题自然占更大问题权重；不是三个 round 的等权平均。

两种后级的直接准确率、其余固定政策、逐 round 表、实际转交、救回/误伤、失败与 oracle
互补性为描述性结果，不对多组未校正比较声称显著。oracle 使用 gold，不能作为可部署政策。
首次结果后只允许相同身份的幂等复算，不允许改政策或数据再覆盖结果。

本轮只记录采集请求的串行时延、设备和资源来源，**不生成真实级联的性能结论**。
两端单独时延相加不是同卡级联测速。只有在开发区验证 9B 同驻留可行后，才可另立下一阶段
配置和较长 open-arrival 测量协议，不将 V2 gain 的速度嫁接到新政策。

完成需六份有效 completion、policy freeze、首次 access、evaluation 和评价收据；中间
导出的同名 `v3_bundle.zip` 不是完成证明。V1/V2 结果继续原样保留，无论本轮结果方向如何。

## 资料

- [ANLI 官方仓库与原始数据链接](https://github.com/facebookresearch/anli)。
- [Qwen3.5-9B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)及[本次固定版本](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)。
- 运行与回传见 [RUN_V3_ZH.md](RUN_V3_ZH.md)。
