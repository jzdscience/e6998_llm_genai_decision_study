# 当前状态

最后核对：2026-11-04，America/Los_Angeles。证据内的原始 UTC 时间不改写。

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

合计 **44,450 条正式模型预测**，不含 smoke 或性能请求。预定 V1/V2/V3 均无需为补齐
流程而重跑 GPU。V3 独立采集了 Qwen 输出；政策零转交不表示实验没有调用后级模型。
V3 验证了 Qwen9B 单模型顺序运行，没有验证 9B 共驻留级联性能。

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
27 项人工软件 fixture 测试通过，真实输入身份与角色预检通过；未进行真实拟合/评分。
运行入口见 [RUN_LIGHT_BASELINE_ZH.md](RUN_LIGHT_BASELINE_ZH.md)，
验收边界见 [准备记录](../records/light-baseline-preparation-2026-11-04.json)。

1. 浏览三轮演示并准备课程讲述，突出“何时转交值得”、负结果及真实服务成本。
2. 在 node 执行并回传这批 400 标签 TF-IDF 基线，再安排完整 1,600 标签预算、冻结
   embedding 分类头、StartLux9B 与加强性能测量。每轮单独固定数据角色与评价范围；
   不能用已看的 holdout 选优后称确认结果。
3. 结合最终证据完成课程报告、演示排练和固定版本交付。Git已恢复，可继续检查差异并
   整理提交；CPU演示与本机开发环境保持原位。
4. 稳健性/live仍是可选待办；预算控制和语义router是新增候选，不替代原计划事项。

范围对照见 [POST_V3_SCOPE_ZH.md](POST_V3_SCOPE_ZH.md)。未冻结V4协议、未启动新GPU
实验，历史结果包不证明当前allocation仍在运行。
