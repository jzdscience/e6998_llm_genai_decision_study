# 当前状态

最后核对：2026-11-05，America/Los_Angeles。证据内的原始 UTC 时间不改写。

## 最新交付：两项收尾实验已准备，待 node 执行

用户于11/5授权完成CPU/GPU同口径服务测量与StartLux9B规模对照。本机已经实现两个独立
namespace、冻结配置、源码/数据校验、可恢复运行、评价与导出；没有远端GPU执行或新质量
结果。联合入口见 [RUN_FOLLOWUPS_ZH.md](RUN_FOLLOWUPS_ZH.md)。

- `clinc-service-v1`：MiniLM400/1600、StartLux4B、Qwen4B；同一CLINC600输入和配对到达表，
  5/25/100请求每秒，180秒×5重复，共60cells、3小时offered窗口、469,804次计划请求。
  每cell新服务/预热，真实每请求编码，报告实际HTTP输出的及时且正确goodput；不由缓存准确率
  推算。保留全部失败、中断、原始trace与资源样本，CPU单线程不等于独占CPU或零GPU allocation成本。
- `size-control-anli-v1`：新增StartLux9B在原ANLI3200上的顺序采集，与原三模型配对。
  固定官方revision和9B原生温度1.1014；90条dev两遍smoke，3290输入token检查，无新路由。
- 两项使用独立输出/session；实际Slurm剩余时间自动查询，不能续同一allocation。联合入口先
  服务实验再9B；完成阶段先校验证据再跳过，不要求为导出已完成结果启动GPU或新session。

全仓CPU回归418 passed、9 skipped、64 subtests；新增Bash联合入口3项针对性检查通过。
真实公开输入准备在gold-read guard下通过；两个CPU分类头的人工文本HTTP smoke通过。
实际L40S显存、服务性能与9B准确率仍待运行和回传验收。软件验证/源码身份见
[准备记录](../records/followups-preparation-2026-11-05.json)。旧V1–V3及CPU基线结果保持原字节。

## 源码与证据位置

`decision_study_repo` 已作为当前源码工作仓库。用户导入基线为 `main` 上的
`fa113e5f0594fa35164ce4321332d143a9242295`，核对时工作区干净，共 112 个已跟踪文件。
该基线已经包含 V2 和 V3 源码，并非只有 V3 之前的代码。本次整理保留原提交，补充文档、
忽略规则、精选报告和证据索引。用户已提交并推送为
`4d573513f7990b8149ef22bd48be59043f980d65`，2026-11-04 通过远端查询核实。

原始结果及审计保留在外部旧项目树，不随本次源码整理搬移。位置、SHA 和恢复边界见
[ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)，报告入口见 [reports/README.md](../reports/README.md)，
整合记录见 [repo-adoption-2026-11-03.json](../records/repo-adoption-2026-11-03.json)。

## 实验已完成到哪里

| 阶段 | 已验收范围 | 主要结论与边界 |
|---|---|---|
| V1 | 15,950 条正式预测；60 个性能单元、60,200 次请求 | CLINC、SQuAD 可答性及三种 4B 模型比较完成；质量与性能采用各自固定样本口径 |
| V2 | 10,800 条正式预测；36/36 性能单元；4,000 题最终评价 | gain−confidence 为 −0.225pp，95% CI [−0.908,+0.478]pp，未显示主方法改善 |
| V3 | 17,700 条正式预测；六组 completion、freeze 和 3,200 题 ANLI 评价 | positive_gain9B−confidence9B 为 +0.594pp，95% CI [−0.528,+1.674]pp；主政策零转交，与直接 StartLux 相同 |
| CLINC 轻量基线 | 400 标签 TF-IDF＋LR；1,850 条 CPU 预测；独立参数/指标重放通过 | main 73.83%、banking 66.00%；mixed stress 87.23%，回顾性监督对照 |
| CLINC 完整预算与 embedding | TF-IDF1600、MiniLM400/1600；5,550 条 CPU 预测；四项独立审计通过 | MiniLM1600 main 93.17%、banking 95.11%、main OOS recall 87.33%；与 StartLux 总正确数差2条，错误分布不同 |

合计 **44,450 条正式模型预测**，不含 smoke 或性能请求。预定 V1/V2/V3 均无需为补齐
流程而重跑 GPU。V3 独立采集了 Qwen 输出；政策零转交不表示实验没有调用后级模型。
V3 验证了 Qwen9B 单模型顺序运行，没有验证 9B 共驻留级联性能。
上面的 44,450 条仅计 V1–V3 的 LLM 预测，两批轻量基线 7,400 条 CPU 分类器预测另计；
这些模型重复预测同一评价池，不能按预测条数累计独立样本量。

## 本次仓库核对

现有 V2/V3 科学源码与发布身份一致；`artifacts.py` 属于已经审计并允许的 `b0586c…`
历史导出器版本。没有为统一目录而替换它或修改科学源码。

临时验证树恢复精确的八份 V2/V3 prepared 文件后，通过原发布校验器：V2 为 30 个发布
文件和 22 个依赖，V3 为 25 个发布文件和 22 个依赖。此结论不表示本源码目录已经包含
数据或 GPU 环境。本次 CPU 测试及最终核对记录以整合记录为准；本机验证不代替 GPU 执行。

最终 CPU 验证为 233 项通过、9 项跳过、7 个子测试通过；跳过项为 1 项官方源缓存检查和
8 项需要原始 V1 证据的 Lab 集成检查。三个 CLI 入口正常，已跟踪 Lab 包的 hash/schema
核验通过。112 个原有文件中只改 README 与 `.gitignore`，其余 110 个原字节不变。

**Git 已于 2026-11-04 恢复并验证。** 159 个 loose object 全部可读且对象 ID 校验一致，
检查时 `.git` 无 dataless 文件。缺失的 index 从已保存、校验通过的 112 项原记录恢复；
恢复前后 182 个工作文件内容未变。原仓库的 `status`、`diff`、`fsck --full` 和
`add --dry-run --all` 均通过；31 条 dangling blob 提示已记录并保留对象。
未修改提交历史，未实际 add、commit 或 push。详情见
[git-recovery-2026-11-04.json](../records/git-recovery-2026-11-04.json)。
这次验证不代表云端文件同步的保留下载策略已改变。

## V1–V3 演示已接通（2026-11-04）

新增独立 `decision_study/lab/`，复用冻结 V1 Lab，并接入 V2/V3 的主比较、政策质量、
validation/holdout实际转交、12+12个解释案例和真实V2性能。原科学源码、冻结参数、原报告
及原V1 Lab均未改变。V3无共驻留性能测量，页面明确显示未测，不合成速度。

新便携包 `reports/decision-lab-2026-11-04/` 约174 KB；制作时核验原包、访问收据与冻结
身份，再重放全部既有holdout政策并与原评价比对。查看只读便携JSON，不依赖外部原始目录。
V2主政策表是validation10%目标；confidence实际validation转交9.6%、holdout11%。
原V1 gate另标选择池；V3confidence为10%→23.219%，positive_gain零转交仍是本地结果。

本机独立 `.venv-lab` 已安装，启动：`bash scripts/run_lab.sh`。
完整说明见 [LAB_CURRENT_ZH.md](LAB_CURRENT_ZH.md)，验收记录见
[lab-2026-11-04.json](../records/lab-2026-11-04.json)。本次没有模型调用或GPU操作。

## 接下来做什么

用户已授权通过 Git 交付 node 源码。新增 `scripts/prepare_node_checkout.py` 和
[NODE_GIT_ZH.md](NODE_GIT_ZH.md)，采用独立 `decision-study-src` clone，保留旧运行根。
本机已验证八份真实 prepared 数据的首次复制与重复配置、原 V2/V3 verifier、三个 CLI
帮助入口及 14 项针对性测试。用户随后回传公司 code-server 输出，node 在
`3dc18faa6c20edf8dc0c6824489f5b8297625186` 打印 READY，8 copied / 0 reused，
源码和旧运行根位置与计划一致。源码配置已完成；这不代表 GPU/session 或新实验已执行。
本机准备记录与远端终端验收分开保留：
[本机验证](../records/node-git-preparation-2026-11-04.json)、
[node 验收](../records/node-git-acceptance-2026-11-04.json)。

第一批轻量分类基线已实现于独立 `decision_study/baselines/`，方案
`clinc-tfidf400-v1` 固定使用 400 个 calibration 标签训练 TF-IDF＋LogisticRegression。
validation 仅诊断，600 条 main test 与 stress 保留回顾性解释；850 条 OOS 与 1,300 条
混合 stress 分开报告。依赖安装于独立 `.venv-baselines`，原 controller/GPU 环境不变。
准备阶段 27 项人工软件 fixture 测试与真实输入预检通过；随后用户已在 node 完成真实运行。
回传绑定源码 `797aff1edfe17d7daaf93551e6cfc236035b6945`，7 个包成员与身份链核验通过，
全部指标独立复算一致，1,850 条保存参数重放标签一致，最大概率差 4.44e−16。
主测试 443/600=73.83%，银行子集 297/450=66.00%；136 条银行请求误判 OOS。
原始 ZIP 已按 SHA 保存，真实回传验收见
[报告](../reports/light-baseline-2026-11-04/RESULTS_ZH.md)和
[验收记录](../records/light-baseline-acceptance-2026-11-04.json)。
运行入口见 [RUN_LIGHT_BASELINE_ZH.md](RUN_LIGHT_BASELINE_ZH.md)，
历史准备边界保留于 [准备记录](../records/light-baseline-preparation-2026-11-04.json)。

`clinc-cheap-v1` 已实现于独立 `decision_study/cheap_baselines/`，固定三项
`tfidf1600`、`minilm400`、`minilm1600`，使用同一 MiniLM ONNX 编码器和固定 LR 配方。
真实训练池整理及输入长度预检通过，最长25tokens；人工句子的真实CPU encoder smoke
通过。新38项与原baseline27项测试通过，9项隔离mutation检出。24项运行依赖单独固定，
Linux/Python3.11 wheel解析通过。上述准备记录与后续回传验收分开保存。

用户已在node完成三项实验，回传绑定 `1b70899bbcf15ee686df04b36e0c333a2fc21e5c`；
15成员及全局freeze/access/evaluation身份通过，5,550条预测与全部指标独立复算一致。
保存参数重放全部标签一致；独立ONNX重放3,450行最大特征差2.98e−7，两个保存分类头
的全部标签仍一致。保留原矩阵和报告，没有重新训练或按test调整参数。
TF-IDF1600、MiniLM400、MiniLM1600的main准确率分别为84.50%、90.67%、93.17%。
MiniLM1600在银行子集比StartLux多对15条、OOS少对17条；总体接近不代表等效或相同行为。
两种特征400→1600均main改善而mixed下降；训练量与OOS先验变化未分离。
原计划中的轻量分类基线已补齐，结果见
[完整报告](../reports/cheap-baselines-2026-11-04/RESULTS_ZH.md)与
[验收记录](../records/cheap-baselines-acceptance-2026-11-04.json)。
运行入口与阶段边界见 [运行说明](RUN_CHEAP_BASELINES_ZH.md)、
[方案](CHEAP_BASELINES_PROTOCOL_ZH.md)、[准备记录](../records/cheap-baselines-preparation-2026-11-04.json)。

1. 浏览三轮演示并准备课程讲述，突出“何时转交值得”、负结果及真实服务成本。
2. TF-IDF400/1600与MiniLM400/1600已完成；StartLux9B与同口径CPU/GPU服务测量已经实现。
   当前需按联合入口在公司node启动，回传两份最终结果包后分别验收；尚无这两项的实测结论。
   每轮单独固定数据角色与评价范围；
   不能用已看的 holdout 选优后称确认结果。
3. 结合最终证据完成课程报告、演示排练和固定版本交付。Git已恢复，可继续检查差异并
   整理提交；CPU演示与本机开发环境保持原位。
4. 稳健性/live仍是可选待办；预算控制和语义router是新增候选，不替代原计划事项。

范围对照见 [POST_V3_SCOPE_ZH.md](POST_V3_SCOPE_ZH.md)。11/5新增两个独立auxiliary协议，
没有将其称为新路由V4；本机未启动GPU，历史结果包不证明当前allocation仍在运行。
