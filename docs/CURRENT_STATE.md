# 当前状态

最后核对：2026-11-08，America/Los_Angeles。证据内的原始 UTC 时间不改写。

## 11/8：9B 服务成本实验设计稿（待审，未实现）

用户请 Claude 设计并实施 review 建议的 9B 服务成本实验，回答“升级前级还是加后级更划算”。设计稿见
[UPGRADE_STUDY_PROTOCOL_ZH.md](UPGRADE_STUDY_PROTOCOL_ZH.md)，机器可读草案为
`configs/upgrade_study_protocol.json`（`status: draft_for_review_not_frozen`）。在 ANLI 3,200 题上比较
StartLux-4B 直接、StartLux-9B 直接、StartLux-4B→Qwen3.5-9B 冻结 class_conditional 级联三种部署，速率
10/20/40/80、每单元 120 秒、3 轮、每单元重启服务，主指标为 500 ms SLO 下的 correct goodput。

设计依据已核实：引擎日志显示 9B 权重 16.8 GiB，V1 双模型启动器每模型 0.40（18.0 GiB）的上限放不下 9B，
因此级联按 0.40+0.50 依次启动两个单模型服务；用 `decision_study.v3.policies.route` 和冻结政策重放 V3 保存的
StartLux-4B 测试集预测，class_conditional 正好转交 766 题、confidence 743 题，与 V3 一致。预计节点时长
4–5 小时。等 Codex 审核设计后再实现；没有写实验代码、启动 GPU 或运行模型。

## 11/8：按 review 更正论文与报告

R1–R5 已逐条处理。论文 `paper/main.tex` 改了低负载 SLO 的表述（Qwen 25/s 五轮及时率 98.85%，r0 一轮
p95 674.5 ms、12 次拒绝、245 次晚返回），把“半个 CPU 核”改为观测用量（约 92 CPU 秒/180 秒，0.51 核当量，
affinity 允许 16 个逻辑 CPU），标签变化改为“观察到、原因未分离”，显存改为 18.9 GiB；related work 区分
生成前的路由（RouteLLM、Hybrid LLM）与生成后的级联；结论和摘要限定在已评估的冻结政策与已测负载；
Table 3 补上 coverage 区间（取自 V1 `analysis/metrics.json`）。Study 2 低负载句改为“无拒绝、每轮至少
91% 及时”。README 同步。成员贡献仍为橙色拟议文本。

已冻结的 follow-up 报告不改字节，新增
[ERRATA_2026-11-08_ZH.md](../reports/followups-2026-11-06/ERRATA_2026-11-08_ZH.md)。`scripts/reproduce_cpu.sh`
增加 `--strict`（跳过即失败）、显式检查 raw recheck 的 `status`，并只在 `paper/requirements-figures.txt`
固定的 Matplotlib 3.11.2 下做图的逐字节比较。本机 `--strict` 全部通过；把证据根指向不存在的路径时按预期
返回 1。论文 10 页，无未定义引用和 overfull。没有模型调用或 GPU 操作。

## 11/8：Claude 接手成果的独立 review

复核 `8b5370d` 到 `c938300` 的分析、论文与仓库整理。核心计数、配对区间、两份最终 ZIP、
28 项报告 SHA 均与原审计一致；科学源码、配置、测试和 Git 内 V2/V3 受绑定文件保持身份。
Lab、五个审计命令、12 项收据比较、三张论文图和临时论文编译通过。主测试初次的 25 个失败
来自沙箱 socket bind 限制，相关五个文件允许本地端口后 70 passed；基线环境 65 passed。
干净源码树使用已有依赖通过 Lab/图复现，全新依赖环境尚未验证。

论文可继续使用，但应修正低负载全部满足 SLO、半核部署、batching 因果归因和 related-work
分类等表述，并补 coverage 区间、确认真实成员贡献。新 baseline/service/size 结果尚未接入
现有 Lab。建议 Claude 继续实施，Codex 做独立科学与复现验收；新增 GPU 实验不是交付前提。
具体证据、位置与分工见 [11/8 审阅清单](REVIEW_2026-11-08_ZH.md)。这是 review 与建议，
本轮未修改论文、实验或前端，未 commit/push。下文保留 11/7 及以前的历史状态。

## 11/7：外部证据根瘦身与备份

外部证据根`project/solo-experiments-2026-10-30/decision-study/`继续保留：原始结果包、解压目录、
原始审计收据和早期日志都只在这里，仓库按设计不收这些文件。其中旧环境`.venv-lab`（489 MB）已移入
废纸篓，证据根从2.6 GB降到2.2 GB；清空废纸篓后才真正释放空间。

外部根的旧日志`docs/EXPERIMENT_LOG.md`与`docs/CURRENT_STATE.md`复制为
`docs/history/*_PRE_REPO_ZH.md`，V1的`analysis/metrics.json`复制到`reports/formal-2026-11-02/analysis/`，
三份都与原件逐字节一致；`metrics.json`也与V1原始包内的同名成员一致，是论文Table 3的来源。

核对备份时发现`startlux-repaired-pilot-20261101-172548`（4 MB）已按SHA归档，却不在任何索引里。
315个清单哈希、CRC和解压目录都核对通过后，已补入`records/BUNDLE_INDEX.json`，共21个条目。
全部21个原始结果包另复制到`~/coms6998-evidence-backup/`（iCloud同步范围外，289 MB），附
`SHA256SUMS`，`shasum -c`全部通过。这份备份仍在同一台Mac上，需要用户再拷到移动硬盘或私有网盘；
包内有节点主机名与路径，不宜公开。

## 11/7：仓库整理、英文入口与 CPU 一键复现

12份已不再使用的恢复记录和规划文档移到`docs/history/`，5份补丁收据移到`records/patches/`
（附说明）。被`SOURCE_SHA256.json`或V2/V3发布清单绑定的文档与脚本保持原路径和字节；科学源码、
configs、报告和外部证据根都没有改动。50处markdown链接随之改写，全仓断链从60处降到52处（旧V1 README
里的根相对链接一并修好），新增文件没有断链。剩余断链指向Git外的证据目录，与原来一致。

新增英文`README.md`（原中文入口改名为`README_ZH.md`并更新）、[文档索引](README.md)（标出受绑定
的文件）、[常见故障](TROUBLESHOOTING.md)、`scripts/README.md`和`scripts/reproduce_cpu.sh`。复现脚本
依次运行CPU测试（主环境365 passed、11 skipped，基线环境65 passed，合计430）、Lab证据检查、11/6
两项回传的审计重跑（12项比较一致：10项逐字节，两份metrics-audit分别排除archive路径和numpy版本字段）、论文
三张图（逐字节一致）和论文临时编译，约1分钟，不改动已跟踪文件。论文图不再写入CreationDate，
附录C加入该脚本。

仓库外，`project/solo-experiments-2026-10-30/`顶层的21个旧补丁ZIP、`.sha256`和notebook/runbook
副本移入`archive/`，附大小与SHA清单；证据根`decision-study/`未动。桌面上两份已按SHA归档的结果包和
0字节占位文件仍保留，是否删除由用户决定。提交在分支`repo-refactor-2026-11-07`（基于
`paper-draft-2026-11-07`），尚未push。

## 11/7：课程报告初版（ACL 双栏）

`paper/`新增报告初版，按三个研究问题组织。proposal的主终点（test coverage与selective risk）单列
一表，正文覆盖V1–V3、CLINC轻量基线、服务测量与StartLux9B规模对照，附TA反馈对照表和复现附录。
BasicTeX上pdflatex加bibtex编译通过：10页，无未定义引用、无overfull、无Type 3字体。三张数据图由
`paper/figures/make_figures.py`从已跟踪的验收数据生成，数字来源列在`paper/README.md`。作者沿用
proposal的三人；Weiyu与Yizhi的贡献句是对剩余交付的拟议分工，以橙色标注，需本人完成并确认后定稿。
算力致谢的写法待定。提交在分支`paper-draft-2026-11-07`。

## 11/7：服务测量与 StartLux9B 规模对照回传验收

用户11/6把最终两包下载到本机：`COMS6998/service_study_bundle.zip`（156,888,964字节，SHA
`d212c130…`）和`size_control_bundle.zip`（4,984,446字节，SHA `f427f776…`）。11/6晚的Codex会话
完成了SHA命名归档、解压校验、全部审计和出图，审计全部PASS，但没有写报告、更新状态或提交，
也没有输出最终回复。11/7在Claude Code中接手。两份归档与桌面原文件SHA一致；四个审计脚本
从仓库位置重跑，14份输出13份逐字节一致，另1份只差archive路径字段；新写的标准库复算从
原始trace和预测分片重新计数，与包内评价完全一致。

- `clinc-service-v1`：60单元、469,804次计划请求，390,571次成功、79,233次client拒绝、0失败。
  100请求/秒时及时且正确输出MiniLM1600 93.43/秒、MiniLM400 90.92/秒、StartLux4B 0.46/秒、
  Qwen4B 0.09/秒；StartLux4B每秒仍完成64.71个、Qwen4B 47.48个，几乎全部晚于500 ms。25请求/秒时
  四者为22.01–23.52/秒。47单元在allocation 66103994，13单元在66376166，只有r3的5请求/秒一组
  跨allocation配对。
- `size-control-anli-v1`：StartLux9B在V3的3,200题ANLI test上2,454题正确（76.69%），比StartLux4B
  +2.50pp，95% CI [1.34, 3.81]pp；比Qwen9B +10.47pp [8.78, 12.29]pp。属于回顾性直接模型对照，
  没有拟合或路由，9B服务性能未测。health的prefix_cache字段与启动配置不一致，与V3相同，已披露。

11/6曾发现的80 MB服务中间包（SHA `165134a1…`，31个完成单元）没有按SHA归档就被同名最终包经
iCloud覆盖，只剩[元数据记录](../records/service-local-transfer-2026-11-06.json)；最终包含全部60个
单元。报告见[followups-2026-11-06/RESULTS_ZH.md](../reports/followups-2026-11-06/RESULTS_ZH.md)，
验收记录见[followups-acceptance-2026-11-06.json](../records/followups-acceptance-2026-11-06.json)，
两个条目已加入[BUNDLE_INDEX.json](../records/BUNDLE_INDEX.json)。本次没有调用模型、启动GPU或
修改节点checkout；节点仍是`8b5370d`。下面各节保留11/6及之前的历史状态。

## 11/6范围讨论：暂缓新包分析，优先整合交付

用户要求暂不分析新包，先评估剩余工作。本轮只读已验收报告、源码和本地课程PDF；Lab
便携证据check通过，未调用模型。建议先做CLINC基线页、真实MiniLM CPU输入演示、整合报告
与干净clone复现；研究扩展至多选在线预算控制、OOS训练比例消融、标签稳健性之一。
详见[11/6后续建议](history/NEXT_STEPS_2026-11-06_ZH.md)。这些是待选方案，未启动新实验或改动
现有前端；新服务/9B结果待用户恢复分析后验收。节点运行仍保留原固定提交。

## 最新交付障碍：150 MB 新包可通过本机路径读取

11/7更新：该障碍已解除，用户随后把最终包下载到本机原路径，验收见本页顶部。以下为当时记录。

后续澄清：用户在另一台Mac远程使用Codex，不能让agent所在Mac访问code-server；新文件
尚未同步到agent。改为在用户手边Mac的系统Terminal分块，再通过聊天逐份上传。独立
`../transfer-tools/split_for_chat.py`将原ZIP字节按19 MiB封装为小ZIP，每片带完整文件与
分片SHA、序号/总数；`join_chunks.py`重组校验。原80 MB包正序/逆序重组相同SHA，缺片、
重复、篡改和已有目标均正确拒绝。最新150 MB包仍待接收，node保持原提交不变。
见[分块准备记录](../records/service-chunk-transfer-2026-11-06.json)。下面的直读建议因两台
电脑的访问边界而不适用当前情况，仅保留其初查经过。

用户已确认旧端口检查没有监听/匹配进程；之后旧session结束并新开session，已说明需用
`--new-allocation`继续同一提交及结果根。用户11/6报告新服务ZIP约150 MB无法聊天上传，
尚未收到新包，不能据体积确认最终完成。

本机发现`/Users/zhanj289/Desktop/COMS6998/service_study_bundle.zip`，79,908,436字节。
目录与少量JSON检查显示413成员、31份完成收据，包含前次端口错误，无freeze/evaluation。
这只是旧包初查，未做完整CRC、cell身份和trace验收。保留旧包，让用户把最新150 MB包
以不同文件名下载到Mac，再由本机路径直接读取，无需上传、分块或修改node源码。
见[本机文件记录](../records/service-local-transfer-2026-11-06.json)。以下记录保留历史状态。

## 最新节点进展：正式服务测量已开始，下一次 CPU 启动端口探测超时

用户回传`8b5370d`终端输出，20条不同cell完成行包含CPU和GPU条件，说明上一处依赖
校验已通过。该片段不代表精确总进度；目前未收到本轮新诊断ZIP，尚未验收最终结果。
随后`18120`的一秒TCP连接探测超时；该端口属于MiniLM CPU服务。源码确认错误在下一
服务启动、warmup及attempt创建之前；超时本身不能判定端口空闲、残留进程或内存问题。

保持node为干净`8b5370d`及原`-r2`输出。确认当前端口/进程后，同allocation原命令续跑
会核验并跳过已完成cell；此错误本身不需`--resume-interrupted`。已完成的低goodput组
保留；日志goodput只计500ms内成功返回，不含正确性评价，也不是全部成功吞吐量。
当前端口状态与实际续跑待用户返回；没有修改科学源码、端口或终止远端进程。
见[操作员回传记录](../records/service-port-interruption-2026-11-05.json)。以下保留前次交付状态。

## 最新故障与修复：Git checkout 的环境元数据发现

用户回传首次服务诊断包（原UTC17:13:22/PST09:13:22），提交`81e390c`。第一条件
StartLux4B在`pip freeze --all`哈希检查阻断，尚无service或cell产物，9B未进入。
原202行清单与新201行清单只差`decision-study==0.1.0`；其他字节完全一致，恢复该原行
可精确重现原SHA。该差异与旧项目egg-info不再被发现一致；原节点当时的Python搜索路径
未随包记录，不能仅凭这个包确定其具体路径配置。

新增兼容上下文固定新checkout为cwd和第一导入位置，并在子进程PYTHONPATH的第二位置
加入原项目根，发现其已有egg-info。只切cwd不足以保证pip能发现该元数据，显式路径使
修复不依赖继承的相对PYTHONPATH。继续调用未修改的GPU环境完整SHA校验；不安装、
过滤依赖或替换expected SHA。两项辅助runtime均接入，覆盖CUDA预检、下载、token检查及
服务启动；新联合入口使用`aux-clinc-service-v1-r2`和`aux-size-control-v1-r2`，原失败根保留。
此次仍是修复交付，实际node续跑和新结果待回传。诊断见
[原包核验](../records/service-startup-diagnosis-2026-11-05.json)，运行见
[联合入口](RUN_FOLLOWUPS_ZH.md)。后面的最初准备记录保留其历史状态。

修复后全仓CPU回归430 passed、9 skipped、64 subtests；9项新增兼容测试包含真实pip
完整哈希恢复、真实版本变化拒绝和子进程源码来源；12项隔离mutation全部检出。
本机pip25.0.1/Python3.13验证不能代替节点pip26.2.2/Python3.12.8和实际GPU启动。
修复与原文件保护核验见[修复记录](../records/followups-runtime-repair-2026-11-05.json)。

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
| CPU/GPU 服务测量 | 4 个直接服务 × 5/25/100 请求/秒 × 5 轮；60 单元、469,804 次请求；审计重跑与原始 trace 复算一致 | 100 请求/秒时及时且正确输出 MiniLM1600 93.43/秒、StartLux4B 0.46/秒、Qwen4B 0.09/秒；GPU 服务约每秒完成 65 与 47 个 |
| StartLux9B 规模对照 | 3,200 条 ANLI 正式预测；与 V3 三个模型逐题配对 | 76.69%，比 StartLux4B +2.50pp，95% CI [1.34, 3.81]pp；回顾性，未测 9B 服务性能 |

V1–V3 合计 **44,450 条正式模型预测**，加上 StartLux9B 的 3,200 条共 47,650 条，不含 smoke 或
性能请求。预定 V1/V2/V3 均无需为补齐流程而重跑 GPU。V3 独立采集了 Qwen 输出；政策零转交
不表示实验没有调用后级模型。V3 验证了 Qwen9B 单模型顺序运行，没有验证 9B 共驻留级联性能。
两批轻量基线 7,400 条 CPU 分类器预测和服务测量的 469,804 次请求另计；这些模型重复预测同一
评价池，不能按预测条数累计独立样本量。

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
2. TF-IDF400/1600、MiniLM400/1600、StartLux9B规模对照与同口径CPU/GPU服务测量均已完成并验收
   （11/7）。每轮单独固定数据角色与评价范围；不能用已看的 holdout 选优后称确认结果。
   若报告需要比较"换更大前级"与"加生成式后级"的服务成本，还缺StartLux9B的服务测量。
3. 结合最终证据完成课程报告、演示排练和固定版本交付。Git已恢复，可继续检查差异并
   整理提交；CPU演示与本机开发环境保持原位。
4. 稳健性/live仍是可选待办；预算控制和语义router是新增候选，不替代原计划事项。

范围对照见 [POST_V3_SCOPE_ZH.md](history/POST_V3_SCOPE_ZH.md)。11/5新增两个独立auxiliary协议，
没有将其称为新路由V4；本机未启动GPU，历史结果包不证明当前allocation仍在运行。
