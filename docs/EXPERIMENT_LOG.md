# 项目日志

此日志从源码仓库接管开始。此前完整恢复、发布和回传验收记录保留于外部旧项目树的
`docs/EXPERIMENT_LOG.md`，访问位置见 [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)；仓库内另有逐字节副本
[history/EXPERIMENT_LOG_PRE_REPO_ZH.md](history/EXPERIMENT_LOG_PRE_REPO_ZH.md)。历史文件不因
仓库整理而改写；当前状态以 [CURRENT_STATE.md](CURRENT_STATE.md) 及其证据为准。

## 2026-11-03：接管用户建立的源码仓库

用户提供 `decision_study_repo`。核对其原始 `main` 提交
`fa113e5f0594fa35164ce4321332d143a9242295`，工作区干净，共 112 个已跟踪文件；
其中已经有 V2/V3 源码、协议和发布清单。Git remote 已配置，本次没有提交或推送。

现有 V2/V3 科学源码与发布身份一致；历史 `artifacts.py` 为允许的 `b0586c…` 导出器版本。
在临时验证树恢复八份精确 prepared 文件后，原 V2/V3 发布校验器分别通过 30+22 和
25+22 项文件检查。本次未修改科学源码、冻结配置、模型结果或已有报告内容。

本次补充当前状态、仓库使用说明、artifact 入口、忽略规则、精选原始报告与整合记录。
新仓库作为后续源码入口；原始结果包、数据、审计和历史环境留在外部旧树。CPU 测试和
最终校验明细见 [repo-adoption-2026-11-03.json](../records/repo-adoption-2026-11-03.json)，
不将本机测试称为远端 GPU 验收。

V1/V2/V3 已完成共 44,450 条正式预测。下一步优先接入 V2/V3 的 CPU Lab 展示和报告；
原计划未完成项与新增候选保持分列。本次没有冻结 V4 或启动新的 GPU 实验。

最终完整 CPU 测试为 233 passed、9 skipped、7 subtests passed；初次失败仅由沙箱禁止
测试端口引起，允许临时本地端口后通过。三个 CLI 从本仓库导入，便携 Lab 包 hash/schema
通过；49 份报告/历史文档副本保持原字节，四个最终结果原包重新计算 SHA 和大小一致。
忽略规则的正反检查通过。原有 112 文件仅 README 与 `.gitignore` 改变。

Git 工作文件已可读，但 84 个 `.git` 文件仍为 dataless；完整 diff 两次出现 mmap timeout。
后台读取扫描已停止；Finder 操作被 Mac 锁屏阻断。没有重建 Git 对象、改写历史或声称
仓库完整性检查通过，需用户完整下载文件夹后验证 `git diff` / `git fsck --full`。

## 2026-11-04：恢复项目主线，CPU Lab 已整合 V1–V3

用户希望继续推进项目。本次直接完成 V2/V3 结果展示，Git云端占位问题不作为本机开发
的前置条件；没有启动新GPU实验，也没有要求重跑已经完成的V1–V3。

- 新增独立 `decision_study/lab/`、启动器 `scripts/run_lab.sh` 和说明
  `docs/LAB_CURRENT_ZH.md`；复用冻结V1界面并保留其案例、阈值探索与性能回调。
  新仓库自己的 `.venv-lab` 已安装，Gradio6.31.0；只监听localhost，公开分享关闭。
- 从原始完整V2/V3包和受保护报告核验消费文件，先检查freeze/evaluation/access身份，再
  读取已评价holdout。重放所有表内政策，与原完整评价的正确数、转交、救回和误伤一致。
  不调用fit/evaluate或写新标签访问收据。两个独立构建输出逐字节一致。
- 新便携JSON174,060字节，各轮12个按已知结果选出的解释案例；明确不代表总体分布。
  所有总体指标来自完整评价池。查看无需原始结果目录、模型、公司allocation或网络服务。
- V2主表用validation10%目标，confidence实际validation9.6%、holdout11%；原V1 gate
  选择池另标。V3将Qwen4B/9B政策名称完整区分，零转交政策与实际后级采集不混淆。
  V2性能只属于已测gain级联；V3共驻留性能显示未测，不用串行时延代替。
- 全CPU测试255项通过、9项因缺可选原始缓存/证据跳过、7个子测试通过。移动图调整后
  22项新测试再次通过。分别移除哈希核验、冻结路由核验、HTML转义、政策选择、实际预算
  显示、后级规模标签的六个隔离mutation均被检出，原源码没有残留变异。
- 真实浏览器验证七项场景，包括V2负载切换、同一V3案例confidence救回与positive_gain
  保留本地错误、嵌入V1 sandbox、390px布局和移动菜单；无页面异常或外部请求。
  修正了移动端CI图裁切；桌面及移动截图已目视检查。
- 原129项快照中仅报告入口README更新，原科学源码/配置/脚本/测试/报告产物不变。
  `pyproject.toml` 只新增Lab可选依赖和命令入口。没有commit、push或修复Git对象下载。
- 验收记录 `records/lab-2026-11-04.json`，细节及截图 `.validation/lab-2026-11-04/`。
  下一步回到原扩展计划，先补轻量分类基线，再安排StartLux9B与加强性能测量的独立协议。
  已见holdout不能重新充当方法选优后的确认集；没有自动取消旧待办或冻结V4。
- 按claudeception评估，已验证的便携冻结证据、预算选择池与模型标签边界补入现有
  `sealed-evaluation-boundaries` 技能；不创建重复技能或把界面测试说成模型实测。

## 2026-11-04：Git 对象读取与缺失 index 恢复

用户遇到 mmap timeout，随后 `git add --all` 报 index.lock 已存在。重新检查时 index
和锁均已不存在，未发现 Git 进程。中断的对象替换方案没有执行；旧的 83 个对象已可读，
另有 31 个对象为 dataless，其中 6 个不能从当前源码或远端推断内容，因此未进行替换。

限时读取最终成功取得全部 159 个 loose object；逐个解压并验证对象 ID，保留独立字节
备份。所有原对象留在原位，包括 31 个 dangling blob。将已保存的有效 112 项 index
恢复到原仓库，原字节备份保留在 `.git/recovery-2026-11-04/original-index.lock`。

原仓库直接执行 status、diff、fsck --full、add --dry-run --all 均返回 0，没有借用临时
clone 的对象库或 index 完成最终验收。恢复时 182 个工作文件 SHA 未变（不含 .git 和
.DS_Store）；未实际 add、commit、push 或改动科学源码。检查时锁不存在，dataless 为 0。
验收记录见 `records/git-recovery-2026-11-04.json`。同步策略未修改，不能保证未来不再
出现占位文件。本次已验证经验补入现有 `macos-dataless-copy-timeouts` 技能。

## 2026-11-04：Git 交付与 node 源码准备

用户已提交并推送 `4d573513f7990b8149ef22bd48be59043f980d65`，本次从远端核实一致，
并授权继续用 Git 更新 node。公司端只能通过用户的 code-server 操作，本机 SSH 探测未能
建立连接；没有远端文件变更或 GPU 操作。

新增 `scripts/prepare_node_checkout.py`、针对性测试和 `docs/NODE_GIT_ZH.md`。新源码
clone 放在旧运行根旁边，使用原 controller，保留模型、环境与 runtime 的绝对路径。
脚本要求指定完整 commit、干净工作区、独立根目录和正确导入位置，只按发布清单复制
八份缺失 prepared 文件，已有冲突在复制前退出。原科学源码、清单和校验器不变。

14 项针对性测试通过。在临时 Git clone 上，使用原始八份 prepared 数据完成首次配置
和重复运行：8 copied/0 reused 与 0 copied/8 reused；每次均运行原 V2/V3 verifier 和
三个 CLI 帮助入口。模拟旧运行根九项文件字节不变，源码 checkout 保持干净。没有安装
或升级旧环境依赖。文件身份、测试及边界见 `records/node-git-preparation-2026-11-04.json`。

源码同步是下一轮的准备步骤；轻量分类基线、StartLux9B 和加强性能测量尚未实现或冻结为
新协议，不将旧 V3 pipeline 当作新实验启动。node 实际配置等待用户回传 READY 输出。

## 2026-11-04：node Git 配置回传验收

用户回传 code-server 完整成功输出，固定提交为
`3dc18faa6c20edf8dc0c6824489f5b8297625186`，READY、8 copied / 0 reused。
新源码根为 `/home/zhanj289/e6998/decision-study-src`，旧运行根仍为
`/home/zhanj289/e6998/decision-study`。按该提交的 READY 执行路径，版本、导入位置、
八份 prepared 文件、两个 release verifier 和三个 CLI 帮助检查均已通过。
终端证据另存 `records/node-git-acceptance-2026-11-04.json`，不改写先前的本机验证收据。
未据此宣称 GPU 可用、allocation 续期或新实验运行；接下来准备独立的 CLINC 轻量分类基线。

## 2026-11-04：交付第一批 CLINC 400 标签 CPU 基线

新增独立 `decision_study/baselines/`、机器可读 `clinc-tfidf400-v1` 方案、隔离环境依赖、
launcher、运行说明与测试，未修改已冻结科学源码、V1/V2/V3 release manifest 或旧报告。
本机仅新建 `.venv-baselines`，原 Lab/controller/GPU 环境不升级。

TF-IDF 词表/IDF 与 multinomial LR 只使用原 calibration 400 条；一套固定参数，validation
不选参。训练读取 V2 fitting 标签，不读取或哈希包含 test 标签的 V1 gold；全部模型与
1,850 条预测先冻结，评价验证身份并记录访问后才打开 V1 gold。已有 test/stress 已见，
新结果定位为预算明确的回顾性监督基线，不称未见确认结果。

本机原始输入预检通过，CLINC 角色为 400/400/600/850，V2 fitting 行与 V1 一致。软件
测试使用完整规模的人工 fixture，27 项通过；原 Lab 环境的 14 项 node 配置测试通过，
新 baseline 测试模块因可选依赖不存在跳过 1 项。没有训练真实模型或计算真实评价结果。
400 标签以外的官方 1,600 标签方案、冻结 embedding、StartLux9B 和加强系统测量仍待办。

质量口径显式区分 main600、OOS-only850 与 mixed-stress1300（共享 main 的450条banking）。
CPU 成本记录为一次1,850条批量预测，不冒充原 GPU HTTP/SLO 测量。软件验收与源码 SHA
见 `records/light-baseline-preparation-2026-11-04.json`；实际运行待 node 回传新 bundle。

## 2026-11-04：400 标签 TF-IDF 回传完成并独立验收

用户回传 `light_baseline_bundle.zip`，SHA256
`7fcd28785a06bf235ee46ca829e4d839cf88743894c5b965cc3e04fcb215414f`，531,979 字节。
源提交为 `797aff1edfe17d7daaf93551e6cfc236035b6945`。评价访问原时间为
2026-11-05T06:41:05.568966+00:00，即本地 11/4 22:41；未改写原证据时间。

7 个成员、self-hash、freeze/access/evaluation 链、三份源码、协议、五个输入 SHA 通过。
400 个训练 ID 与 calibration 精确一致；validation400、test600、stress850 共1,850条预测
完整。独立计数器复算所有指标和混淆矩阵，全部一致。V1 三个原对照的 641 份 config/shard
从原 formal ZIP 校验并按 ID 比较，没有重跑模型或复用性能请求冒充独立样本。

独立从保存 JSON 重构 TF-IDF/logits/softmax/列重排，1,850 条标签一致，概率最大差
4.44e−16；2,184 维词表与训练文本吻合，IDF 误差为0，原模型记录18次迭代。没有重新拟合
或生成新的评价访问收据。包内训练52.10ms、一次合并预测18.46ms仅是CPU内部计时，
不与GPU服务相除计算速度比。

主测试443/600=73.83%，银行297/450=66.00%；157个错误包括136个银行误拒、17个业务间
错分、4个OOS漏拒。mixed stress1134/1300=87.23%，复用相同银行子集；850条OOS-only
recall为98.47%。400标签监督条件与原提示式LLM不同，结果仍是已见数据上的回顾性比较。

原包、解压与三项独立审计归档于外部证据根；源码仓库新增
`reports/light-baseline-2026-11-04/`、验收记录与 bundle 索引，更新当前状态和原范围对照。
准备记录和既有V1–V3报告保持原字节。未改科学源码/配置，未启动新GPU实验。
下一批仍为完整1,600标签及冻结embedding基线，其后StartLux9B和加强性能测量。

## 2026-11-04：交付完整标签预算与冻结 MiniLM 分类器

用户要求继续。本次新增独立 `decision_study/cheap_baselines/`，固定TF-IDF1600、
MiniLM400、MiniLM1600三个条件及统一评分流程。原400标签baseline、V1/V2/V3源码、
配置、发布清单和已验收结果保持不变；不是重跑旧pipeline或新V4路由实验。

官方完整训练1600包含calibration400及原engineering-dev的60条银行文本，不含4条
authored OOS；与validation400/test600/stress850的ID和规范化文本均零重叠。数据准备
从固定revision原始文件提取训练专用文件，两次字节相同；fit/preflight不读raw或V1 gold。
训练预算指实际传入LR的标签，400/1600同时改变OOS先验，不声称纯样本量效应。

MiniLM固定官方revision及八份文件SHA，使用未量化ONNX、maskedmean+L2、CPU1线程、
batch32，256token上限拒绝截断。真实3450输入仅做tokenize预检，max25；人工33句
encoder smoke验证批次/padding、CPU签名、长文本阻断和离线缓存复用。没有对真实
实验数据执行embedding、拟合或评分。编码器成本和LR成本单列，不能省略共享编码成本。

新38项人工fixture测试加原baseline27项通过，9项隔离mutation被检出；原node配置
测试单独运行。新环境固定24项CPU运行依赖，pip check通过，Linux/Python3.11全量
binary wheel解析通过。处理了跨目录launcher导入旧包风险：入口先cd到新源码根；
模型根在resolve前拒绝symlink。依赖锁生成时排除了当前目录可发现的本项目egg-info，
最终依赖清单不含decision-study这个本地源码包。

新增一个suite入口，下载/准备数据与权重、预检、三项全局freeze、评价和导出均可一次
执行。完整结果可核验复用；无freeze的部分结果和已有access但缺评价的结果会保留并
停止，不自动删证据重跑。实际node运行等待回传 `cheap_baseline_bundle.zip`。
准备验收见 `records/cheap-baselines-preparation-2026-11-04.json`；真实结果尚未产生。

## 2026-11-04：完整预算与 MiniLM 回传验收

用户回传 `cheap_baseline_bundle.zip`，6,793,864字节，SHA256
`7f00178c7bd1d86eeeb121929131447675d0053c10072f687f88d7e3ae77b2be`。绑定源码
`1b70899bbcf15ee686df04b36e0c333a2fc21e5c`，原评价访问时间2026-11-05T07:33:50.228579+00:00，
即本地11/4 23:33。三条件共同freeze与最终评价完整。

15个成员及全hash链、8份发布源码/脚本/依赖文件、5个项目输入、2份prepared和8份实际
encoder文件通过；3×1850条预测、训练成员与全部指标独立复算一致。原TF-IDF400与V1
原始ZIP核验后逐ID对照。保存模型参数重放5550标签一致，TF-IDF词表4717维、IDF误差0。
独立ONNX重放3450行最大差2.98e−7，两个MiniLM分类头全部标签仍一致；原矩阵未覆盖。
微差可同时来自平台/backend及独立pooling归约顺序，不声称单独硬件原因。

Main结果：TF-IDF1600为507/600=84.50%，MiniLM400为544/600=90.67%，MiniLM1600为
559/600=93.17%。StartLux历史结果561/600=93.50%；MiniLM1600银行多对15、OOS少对17。
两者互相独自正确26/28，不能把总数差2解释成只两条预测不同，也没有等效性确认。
MiniLM400→1600 main+15但mixed−41；TF-IDF400→1600 main+64但mixed−141。
训练量与OOS先验共同变化，未将交换归因于单一原因，未据test调参或生成新政策。

包内共享编码3450条14.968秒；两个LR头1850条预测约1.7ms不含编码。只说明CPU路径已
运行，不与GPU服务相除宣称实测速比；原Linux耗时未被离线重测。
两批轻量分类基线已完成，另计7400条CPU预测；V1–V3正式LLM预测仍为44,450条。

原包、解压、prepared及审计归档于外部证据根；仓库新增
`reports/cheap-baselines-2026-11-04/`与`records/cheap-baselines-acceptance-2026-11-04.json`，
更新索引、当前状态和原范围。旧科学源码、准备记录、原报告不变。下一步保留StartLux9B
与加强系统测量，后者可考虑加入MiniLM CPU服务，但本次没有启动或冻结新实验。

## 2026-11-05：交付服务性能与 StartLux9B 两项独立实验

用户明确授权完成CPU/GPU同口径服务测量与StartLux9B规模对照。本机已实现两套独立
namespace、固定协议、node入口与文档，旧科学源码/配置/原报告/冻结参数保持原字节。
公司端只能由用户code-server启动，本次没有远端GPU执行或新正式模型结果。

服务协议固定MiniLM400/1600、StartLux4B、Qwen4B四直接部署；同CLINC main600，公共
Poisson schedule，5/25/100每秒，180秒×5，共60cells、3小时offered窗口、469804计划请求。
CPU每请求真实编码，保留保存分类头与类别轴；实际HTTP标签在全部测量freeze后评价，
计算及时且正确goodput，不将旧准确率乘吞吐量。各cell重新启动服务/64条dev预热，防止
超时后后端残留工作跨rate；启动/预热不混入正式窗口，原始资源及失败请求均保存。

StartLux9B只补原ANLI3200，复用三模型9600条原预测；90条dev两遍smoke，3290输入exact-token。
固定revision18a17f734966b213b2f00fa9c915e8b9af2a9b20、BF16、原生温度1.1014、0.8显存份额。
新runtime/session/cache和新评价链独立；已完成阶段先验收再跳过，避免为了导出重新初始化
allocation。实际单卡加载可行性仍待node。两阶段均阻断已有GPU工作，不杀其他进程。

全仓CPU回归418 passed、9 skipped、64 subtests；随后联合Bash入口3项测试通过，验证首次
无可选参数、路径含空格、恢复flag分流和前阶段失败不继续。修复本机Bash3.2中set-u展开
空数组报unbound variable的问题；改为始终非空的完整命令参数数组。
实际输入prepare在gold读取guard下通过且重复字节一致；真实CPU模型仅用人工文本做HTTP
smoke。实际child→HTTP→trace→receipt软件fixture及watchdog退出竞争检查通过。
关键身份/预算/评价/调度隔离mutations被独立断言检出，生产源码无变异残留。

准备记录 `records/followups-preparation-2026-11-05.json`；联合运行说明
`docs/RUN_FOLLOWUPS_ZH.md`。下一步用户node运行后回传service_study_bundle.zip和
size_control_bundle.zip，分别验收后再写实测结论与整合前端。不把软件准备标成实验完成。

## 2026-11-05：服务启动依赖清单阻断与 Git checkout 兼容修复

用户在提交81e390c启动联合入口后回传诊断包，SHA256
`cbdfea79daee12f6074e2b7e719d9acc7033dedc79eb59620ecce3ce4cc5a287`。
28个成员及源码/输入/到达表身份通过，原包按SHA归档。第一计划cell的StartLux4B在
CUDA预检中的完整pip-freeze哈希检查阻断，服务与正式cell未启动，9B阶段尚未进入。
不是由该记录证明的OOM或模型加载失败；不能将诊断ZIP算作完成结果。

原202行清单与回传201行仅差`decision-study==0.1.0`，其他字节完全相同，补入原行
可精确重现原SHA。旧项目元数据未被发现与该差异一致，但节点历史Python搜索路径
没有记录，不能断言其具体配置。本机真实pip表明只切cwd不足以保证发现旧egg-info。

新增独立runtime_compat上下文，让新checkout作为cwd与第一导入位置，并显式把原项目根
作为子进程搜索路径第二项，复用真实现存egg-info。旧完整哈希校验未改；不创建元数据、
重装依赖、过滤清单或改expected SHA。兼容入口覆盖两项新辅助runtime；原V1五个科学
文件、两个新实验协议、CUDA配置及原准备收据逐字节保持。联合入口使用新的两个-r2根，
保留首次失败根，避免新源码与旧准备身份混用；无正式cell中断，无需resume-interrupted。

9项新增测试与12项隔离mutation通过；最终全仓430 passed、9 skipped、64 subtests。
真实隔离pip测试在没有继承PYTHONPATH时恢复精确hash，改变已安装fixture版本仍被拒绝；
子进程使用新源码，嵌套/异常恢复正确。独立源码复核覆盖vLLM、StartLux入口和model/CUDA
脚本。实际节点pip与GPU启动仍待用户执行，未记任何新质量/性能结果或续期allocation。

诊断见`records/service-startup-diagnosis-2026-11-05.json`，修复验证见
`records/followups-runtime-repair-2026-11-05.json`，执行入口仍为`docs/RUN_FOLLOWUPS_ZH.md`。

## 2026-11-05：节点完成多组服务测量后，CPU 端口启动检查阻断

用户粘贴`8b5370d`运行输出，包含20条不同cell完成行；尚无本轮诊断包，不能据片段声称
精确完成总数或全部测量通过。CPU/GPU均已产生完成行，前次依赖校验阻断已越过。
最后报`Could not verify that port 18120 is unused: timed out`。

独立源码复核确认18120为CPU端口，错误来自一秒连接检查，早于bind检查、下一CPU进程、
warmup和attempt创建。已完成cell在原提交、原输出下续跑会逐项验证并跳过；本次错误
单独不要求resume-interrupted。请求当前ss/进程输出，尚不能确定是残留监听还是瞬时问题。
没有盲目换端口、改冻结源码、杀进程或让用户重做完成的cell。

日志及时goodput使用500ms SLO和180秒窗口；低值可能伴随晚返回、拒绝或失败，须看trace
区分。全部及时的CPU100rps组只证明这些已观察负载，不代表最大容量或质量结论。保留所有
低goodput组，待诊断与最终artifact核验。记录`records/service-port-interruption-2026-11-05.json`。

## 2026-11-06：新结果包上传体积限制与本机直接读取

用户随后回传ss只有标题、pgrep无输出；之后明确新开session，已给出同提交同输出加
`--new-allocation`的续跑命令。11/6用户报告service ZIP约150 MB无法上传，尚不能确认完整性。
只读查找发现本机旧`COMS6998/service_study_bundle.zip`为79,908,436字节。ZIP目录/选定JSON
初查见413成员、31份完成收据、前次端口错误，无freeze和evaluation；没有据此声称31组已
独立验收。文件SHA及检查范围见`records/service-local-transfer-2026-11-06.json`。

采用更简单的本机文件路径交付：新包下载到Mac并另取文件名，不覆盖旧包，无需上传聊天
或改动正在使用的节点checkout。待新文件就位后再执行完整安全归档、校验与结果审计。

用户随后明确其远程操作另一台Mac，agent主机不能访问code-server，新文件未同步。
因此不能继续要求在agent主机重新下载。独立于Git实验checkout准备原ZIP字节分块工具，
每片19 MiB、ZIP_STORED外壳、原文件与分片SHA和序号/总数。原79,908,436字节包5片
正序及逆序输入都精确还原原SHA；缺片、重复、篡改和已存在目标4项异常检查通过。
给用户可粘贴到手边Mac系统Terminal的sender；接收后由agent合并。此为传输准备，
最新150 MB包与完整结果仍未收到。见`records/service-chunk-transfer-2026-11-06.json`。

## 2026-11-06：暂缓新包分析期间的项目取舍

用户要求先不分析新包，查看项目还可推进哪些工作。仅复核已有计划、已验收报告、Lab
源码和本地14页课程交付PDF。独立Lab check通过V1–V3便携证据、3450条V1输入，0模型调用。
当前页面已经有三轮证据、案例与政策探索；轻量分类基线尚在报告中，README的基线状态
仍滞后于验收记录。未将本轮当成前端从零建设或新增模型运行。

建议优先Lab基线整合、真实MiniLM CPU现场输入、整合报告/演示和干净clone复现/CI。
若追加研究，在线预算控制为深度方向；CPU训练量/OOS先验消融和标签稳健性为替代，
最多选一条。已查看holdout保留探索边界，预算规则本身达标不当成研究胜出。
建议与验收条件写入`docs/NEXT_STEPS_2026-11-06_ZH.md`；未启动任何新候选或分析新结果包，
没有修改现有实验、前端、模型、配置、历史报告或节点提交。

## 2026-11-07：服务测量与 StartLux9B 规模对照回传验收

用户11/6把最终两包下载到本机原路径。服务包156,888,964字节、776成员，SHA
`d212c130c707a30fc9b99be759e0ec4386000b21fca0792dbe2867b619e0998b`；规模对照包4,984,446字节、
166成员，SHA `f427f7762bac745d9316cb02195c09f98ab551794de1da047509be7f9fe7d1ed`，两者均绑定提交
`8b5370d`。11/6晚的Codex会话已按SHA归档到外部证据根`records/bundles/`、完成解压校验与全部审计并
出图，审计全部PASS；该轮没有写报告、更新状态或提交，也没有输出最终回复。

11/7在Claude Code中接手，未调用模型、未启动GPU、未改节点checkout。两份归档、桌面原文件与报告目录
中的评价副本SHA一致。四个审计脚本逐字节复制到`reports/followups-2026-11-06/audit/`后从仓库位置重跑，
14份输出中13份与原收据逐字节一致，`service/metrics-audit.json`只差archive路径字段；README中的命令块
按原样执行，结果相同。新写的`recheck_from_raw.py`只用标准库，用V1 CLINC gold从原始trace重新计数，
60单元的offered、admitted、correct、及时且正确四项与包内评价全部一致；9B从原始分片和sealed gold
重新评分，2,454/3,200及四格配对计数一致，标准库bootstrap区间[1.28, 3.75]pp与保存的[1.34, 3.81]pp
差别来自随机数生成器。

服务测量：469,804次计划请求，390,571次成功、79,233次client拒绝、0失败。100请求/秒时及时且正确输出
MiniLM1600 93.43/秒、MiniLM400 90.92/秒、StartLux4B 0.46/秒、Qwen4B 0.09/秒；GPU服务每秒仍完成
64.71与47.48个，几乎全部晚于500 ms。25请求/秒时四者为22.01–23.52/秒。GPU批处理下StartLux有2条、
Qwen有8条输入出现过不同标签，MiniLM 117,451次响应与11/4保存预测完全一致。47单元在allocation
66103994、13单元在66376166，只有r3-rate5一组跨allocation配对。

规模对照：StartLux9B 76.69%，比StartLux4B +2.50pp [1.34, 3.81]，比Qwen9B +10.47pp [8.78, 12.29]；
三个round均高于4B。回顾性直接模型对照，9B服务性能未测。health的prefix_cache字段与启动配置不一致，
与V3相同，保留并披露。

11/6在本机发现的80 MB服务中间包（SHA `165134a1…`）没有按SHA归档，之后被同名最终包经iCloud
覆盖，只剩`records/service-local-transfer-2026-11-06.json`中的元数据；最终包含全部60个单元。

产出：`reports/followups-2026-11-06/RESULTS_ZH.md`、`audit/`与`evidence/`、`SHA256.json`，
`records/followups-acceptance-2026-11-06.json`，`records/BUNDLE_INDEX.json`新增两个条目。下一步把两组
结果接入课程报告与Decision Lab；若报告需要9B前级的服务成本，需另立协议在节点运行。

## 2026-11-07：课程报告初版

在`paper/`写出ACL双栏初版，按RQ1直接替代、RQ2转交、RQ3服务约束组织；V1/V2/V3在文中称Study 1–3。
所有数字取自已验收报告、`reports/decision-lab-2026-11-04/study.json`、`reports/followups-2026-11-06/`
和Study 1的`analysis/metrics.json`；V1 selective risk的全部95%区间从后者补齐。成文时逐项核对数字，
改正了两处初稿笔误（SQuAD oracle差距3.05pp、CLINC两模型独对26与28题）和一处运行环境描述
（Kev使用官方PyTorch服务栈，StartLux与Qwen使用vLLM 0.31.0）。2026年的四篇引用按arXiv与Hugging Face
页面核对作者与题名。没有新的模型调用或实验。下一步由组员按拟议分工补文献综述、报告与复现教程。

## 2026-11-07：仓库整理与 CPU 复现入口

用户同意"整理外层、冻结核心"的方案。先用引号正确的grep与引用分析确认：V2/V3发布清单绑定
`CONTINUE_V2_ZH.md`、`LAB.md`、`V2_PROTOCOL_ZH.md`、`RUN_V3_ZH.md`、`V3_PROTOCOL_ZH.md`和9个脚本，
`SOURCE_SHA256.json`绑定V1的5份文档和7个脚本，这些都不移动。根目录5份补丁收据只出现在
`records/repo-adoption-2026-11-03.json`的历史清单里，代码不读取，移到`records/patches/`。12份历史/
规划文档的冻结引用同样只有该清单，移到`docs/history/`。没有改写任何records或已冻结报告。

链接改写脚本按原位置解析每个相对链接，再按新位置重算；对旧V1 README中原本断开的根相对链接
改为指向实际文件。改写后断链60→52，新文件0处。正文中以纯文本提到旧路径的历史日志条目保持原样。

`scripts/reproduce_cpu.sh`在本机全部通过：测试365+65、Lab check、13份审计收据、3张图逐字节一致、
论文编译。比较中发现size metrics-audit记录了运行时的numpy版本（2.5.4对原2.3.5），其余值完全相同，
脚本对该字段与service metrics-audit的archive路径各自排除后比较。

仓库外的旧补丁包移入`project/solo-experiments-2026-10-30/archive/`，移动后逐个复核大小与SHA。
没有模型调用、GPU操作或push。

## 2026-11-07：证据根瘦身、早期日志入库与原始包备份

用户确认保留外部证据根并做三项处理。外部根旧`.venv-lab`经核对没有进程或文件引用，移入废纸篓
（`~/.Trash/decision-study-old-venv-lab-20261107-174517`）。旧日志与旧状态页、V1 `analysis/metrics.json`
逐字节复制进仓库，SHA分别为`39adf7bc…`、`38cac56a…`、`a6922bee…`，最后一项与V1原包成员一致。

备份前用索引核对`records/bundles/`的21个文件，发现`startlux-repaired-pilot-20261101-172548-c597b063335b.zip`
未入索引。该包为11/1的诊断pilot：Kev 6,250条完成，StartLux reference探针在run/smoke失败，StartLux与Qwen
尚未开始。核验后按时间顺序补入索引。21个包复制到`~/coms6998-evidence-backup/`，大小与SHA逐一对照索引，
`shasum -a 256 -c SHA256SUMS`全部OK。本机没有外接盘或已同步的个人网盘，异地副本留待用户完成。

## 2026-11-08：独立复核 Claude 接手的分析、论文与重构

用户请求审查接手成果并建议后续分工。以 `c938300` 对比 `8b5370d`，并核对原始审计、
完整论文、相关报告、源码、课程交付 PDF 与核心引用。科学源码/配置/测试没有变化，
Git 内 V2/V3 发布身份一致；最终 ZIP、28 项报告 SHA、原 18 项索引内容保持。

现有环境主测试初次342 passed、25 failed、11 skipped；25个失败全为沙箱禁止socket bind，
相关五文件允许本地端口后70 passed。基线环境65 passed。五个审计命令、12项收据比较、
Lab检查、三张图逐字节复现和临时论文编译通过。原shell因首次测试失败返回1，未称其整体
exit0。干净git archive源码树使用已有依赖也通过Lab与出图，全新依赖安装仍待验证。

发现论文与新报告把中位数扩展成所有低负载轮次满足SLO：Qwen25的r0 p95约674.50ms，
12拒绝、245晚返回，r2另2晚返回。其他应改项包括CPU时间被称为半核部署、batching因果
归因、RouteLLM/Hybrid调用前路由被归入文本前级级联、部分结论范围、coverage区间遗漏。
未发现需推翻主要数据或重跑实验的证据。

新增 `docs/REVIEW_2026-11-08_ZH.md` 并更新状态，建议Claude继续实施、Codex独立验收；
优先论文更正、Lab接入新结果、干净环境复现与presentation/demo。未修改论文、已冻结报告、
科学源码或原始结果，未启动实验、commit或push。

## 2026-11-08：按 Codex review 更正论文、报告勘误与复现脚本

逐条对照原始记录核实 review 的 R1–R5，均成立。`cells.csv` 显示 Qwen 25/s 的 r0 有 4,559 次计划请求、
12 次拒绝、245 次晚返回、p95 674.5 ms，r2 另有 2 次晚返回，五轮 22,332/22,591 及时；其余三个服务在
5/25/s 全部及时。V2 在 10/25/s 无拒绝，最低一轮及时率 91.4%（local_only 10/s 的 dispatch 停顿）。
MiniLM 100/s 约 92 CPU 秒/180 秒，affinity 为 16 个逻辑 CPU；启动中位数 0.76 秒，最长 5.54 秒
（r3-minilm1600-rate100）；显存峰值 19,315 与 19,179 MiB。

论文改写相应句子并补 Table 3 的 coverage 区间；冻结报告另立勘误文件，原 `RESULTS_ZH.md` 与 `SHA256.json`
不动。11/7 日志中“GPU批处理下……出现过不同标签”一句以勘误 E4 为准：变化是观察结果，原因未分离。
复现脚本加入 `--strict`、raw recheck `status` 检查和 Matplotlib 版本守卫；本机 strict 全部通过，缺证据根时
返回 1。11/7 状态页“13份收据”更正为 12 项比较。没有模型调用、GPU 操作或 push。

## 2026-11-08：9B 服务成本实验设计稿

按 review 的建议设计“升级前级还是加后级”的同任务、同负载对照。核对了三项依据：V3 与规模对照的引擎日志
（9B 加载 16.8 GiB；0.80 下 KV 15.8–17.1 GiB；4B 在 0.40 下加载 7.99 GiB、KV 7.6–8.3 GiB），V1 `managed_services`
对双模型每个不超过 0.40 的限制，以及 V3 冻结政策的离线重放（class_conditional/qwen9b 766 题，confidence 743 题）。
据此写出设计稿与 JSON 草案，三种条件、四档速率、3 轮、每单元重启，主指标 500 ms correct goodput，次要 1,000 ms。
设计待 Codex 审核；未实现代码、未调用模型或 GPU。

## 2026-11-08：修订复审通过，9B 服务成本设计有条件通过

独立复核 `d2657cc` 与 `9cb3274`。R1–R5、coverage 区间、勘误与 strict 复现改动验收通过；
follow-up 原报告/清单字节未变，28 项 SHA 匹配。独立 strict 复跑 exit 0，主测试 365 passed、
11 skipped、64 subtests，基线 65 passed；Lab、五个审计命令、12 项收据、raw PASS、三图逐字节比较
和论文临时编译通过。尚未验证全新依赖安装。errata 的“没有排队”建议改成“没有客户端拒绝”。

9B 服务研究保留原三条件、四速率、三轮与每单元重启，不扩矩阵。新增
`docs/UPGRADE_STUDY_REVIEW_2026-11-08_ZH.md`，要求修订协议后实现：把 0.50 显存估算标为待验证，
补三部署的并发开发预检；新双服务控制器明确自有进程、独立目录/端口和失败清理；两阶段共享 worker-start
后的 30 秒预算，SLO 仍从 scheduled arrival 计时；实时质量按匹配 ID/出现次数比较，不强求复现整池比例；
class_conditional 明确为看过 test 后选定的固定回顾性部署。

独立重放 V3 原 3,200 条保存预测确认 class_conditional 766 次、confidence 743 次。766 仅是历史重放
回归数字，不是实时服务验收目标。0.40+0.50 符合 vLLM 每实例预算语义，但已有 9B 日志来自单模型 0.80，
不能证明新同驻留可行性。未改变草案、论文、科学源码或冻结证据；未运行模型、GPU、commit 或 push。

## 2026-11-08：9B 服务研究实现与本机验收

按 U1–U5 修订协议 Markdown/JSON（revision 2），实现 `decision_study/upgrade_study/`：`identity` 负责协议、
输入、冻结政策与历史重放；`runtime` 负责三个角色的 runtime-control、L40S 检查、GPU 所有权、外层清理栈与
停止验证；`client` 负责直接调用和共享截止时间的两阶段级联；`cell` 是单元子进程；`study` 负责 prepare、
preflight、run、freeze、evaluate、export；`cli` 处理命令与 SIGTERM/SIGHUP。入口为
`scripts/run_upgrade_study.sh`，节点步骤见 `docs/RUN_UPGRADE_STUDY_ZH.md`。

本机测试 27 项，包括一次 CLI 全流程演练；全仓 392 passed、11 skipped、64 subtests。13 处注入错误都被发现。
真实证据 dry run 得到 36 个单元、162,801 个计划 offer、72 分钟 offered 窗口，重放 766/743 次转交。
级联预检改为按实际路由原因确认强制转交。仓库为 private，节点 HTTPS fetch 需要凭据；运行文档里的 bundle
更新方式已在本机模拟 checkout 上跑通。未运行模型或 GPU。

## 2026-11-08：仓库改回 public

用户把 GitHub 仓库改回 public，匿名 API 查询返回 `private: false`；GitHub 上只有 `main`（`c938300`），
review 分支仍在本机。运行文档删去 bundle 一节，节点照 NODE_GIT_ZH.md 用 HTTPS fetch 更新到交付的 SHA。

## 2026-11-08：补齐验收清单第 5、6 项的测试

对照审核清单补三项测试：中断的单元尝试原样保留，`run` 与 `_attempt` 都要求显式 `--resume-interrupted`；
同一 repeat/rate 的配对单元跨 allocation 时评价报告 `false`（计算抽成 `pairing_same_allocation`，输出不变）；
allocation 续期在同一 job 内不能延长 deadline，换 job 必须带 `--new-allocation`，旧 session 进入
session-history。`tests/test_upgrade_study.py` 30 项，全仓 395 passed、11 skipped、64 subtests；
注入错误增至 19 处，全部被发现。

## 2026-11-08：预检并发检查改用自有子进程

复用的 `run_cell` 注明 socket timeout 不能给出绝对执行上限，需在独立子进程中运行；正式单元一直如此，
预检的 30 秒并发检查原先在控制器进程内调用。现改为与正式单元相同的 cell 子进程和父进程 watchdog
（审核 U3），用 `verify_cell` 核对 completion receipt，并要求 trace 与预检调度逐条一致。CLI 演练确认
级联预检的 `plan.json` 为 `forced_fallback`。注入错误增至 20 处。

## 2026-11-08：9B 实现独立交叉验收

核查 `9cb3274..65c57a9` 的四个提交。独立全仓测试 395 passed、11 skipped、64 subtests，exit 0；
30 个 upgrade 测试均包含在内。科学口径、历史重放、共同调度、实际标签 goodput、正式 trace/计划/资源校验、
全单元冻结后读取 gold、allocation 续期及主体生命周期实现符合修订方向。

额外探针发现四个 P2，详见 `docs/UPGRADE_IMPLEMENTATION_REVIEW_2026-11-08_ZH.md`：
I1 使用实际冻结 launcher 清理代码模拟后级 readiness 失败，组长退出后仍留下忽略 TERM 的自有 worker，
新控制器因只在成功进入上下文后登记组而漏清理；探针 finalizer 已清理自身残留。I2 删除假服务 CLI 演练副本的
级联预检 trace，不改任何收据，缓存预检与完成验证仍通过。I3 0.6 秒 watchdog 接受了 sleep 0.61 的真实子进程，
观察到约 0.763 秒返回成功。I4 实际 localhost HTTP 每 45 ms 分块输出，80 ms 预算在约 290–310 ms 后仍记成功；
串行预检/预热也不在有界子进程中。

本次尚不通过节点预检前的软件验收，建议 Claude 修 I1–I4 后复审，不扩实验矩阵。独立日志和探针保存在
`.validation/upgrade-review-2026-11-08/`。未独立重跑 Claude 的 20 项 mutation，未运行 GPU 或模型、未修改
实验源码/协议/冻结证据，未 merge/push。更新当前状态并新增审阅文档。

## 2026-11-08：修复实现验收 I1–I4

按 `docs/UPGRADE_IMPLEMENTATION_REVIEW_2026-11-08_ZH.md` 修复四项，旧冻结源码未改，协议升为 revision 3。
I1 启动失败时按本次新写的 `last_services.json` 登记进程组，停止核验按 `ps` 清理本次部署启动的残留成员
（SIGTERM 后 SIGKILL），`finished.json` 的停止确认成为单元完成条件。I2 统一的 `verify_preflight` 在复用、
开始、完成和冻结时核对全部预检原始文件，token 收据绑定原始 tokenizer 报告。I3 子进程退出后再查时钟与
allocation。I4 晚到的成功响应记 `deadline`，串行预检与预热进入有界子进程。

新增回归测试覆盖：忽略 SIGTERM 的残留 worker、遗留记录不发信号、早于部署的组成员不发信号、停止未确认的
单元重测、截止前写 receipt 而截止后退出、退出后时钟越界、慢分块 HTTP 的直接/保留前级/后级三条路径、
路由跨越截止时间、串行阶段超时清理、前级无效 schema，以及演练产物上的预检证据篡改。`tests/test_upgrade_study.py`
46 项，全仓 411 passed、11 skipped、64 subtests。注入错误在隔离副本上运行，36 处各自在对应断言处失败，逐项日志见
`.validation/upgrade-fixes-2026-11-08/`。未运行模型或 GPU，未合并推送。

## 2026-11-08：I1–I4 修复独立验收通过

复审 `3fc84e2f06a443a5b58c83f4a86545d91f23910d`，四项关闭，未发现新的阻断问题。允许 localhost/ps 的
独立全仓测试 exit 0，411 passed、11 skipped、64 subtests，109.76 秒；包含全部 46 项 upgrade 测试。
初次执行的失败来自沙箱本机端口和进程列表限制。核对 36 份完整 mutation 日志及隔离副本恢复后的文件身份，
未独立重跑全部 mutation。

重新执行上一轮运行时探针，部分启动失败的后级组已登记，忽略 TERM 的 worker 最终被清理；80 ms 预算的慢分块
HTTP 响应现在记 deadline 并保留 stage。独立复制 revision 3 演练证据，删除并发 trace 或原始 token 报告时，
缓存预检、单元完成和 freeze 均拒绝；删除停止确认时保留 measured 事实，但 completed/freeze 不通过，恢复后通过。

Linux procps 上游默认日期格式与本地 parser fixture 相容，但已有 Linux 镜像不含 ps，未安装新工具，实际节点行为
仍待预检。`unverified group members` 也可能来自更早启动的成员，不能仅凭该消息断定解析问题。30 秒约束保证
超期响应不记成功，整个阶段由父 watchdog 兜底，不承诺 socket 在第 30 秒立即取消。

新增 `docs/UPGRADE_FIX_ACCEPTANCE_2026-11-08_ZH.md`，更新状态。建议 Claude 将验收笔记一并提交、普通合并推送，
交付最终 SHA 后先做节点 `--preflight-only` 并回传，验收后再做正式单元。本次未修改运行源码/协议、未调用
真实模型或 GPU、未 commit/merge/push。独立证据保存在 `.validation/upgrade-fixes-review-2026-11-08/` 和
`.validation/upgrade-fix-review-2026-11-08/runtime-independent.json`。

## 2026-11-08：节点首次开发预检（revision 3）与 revision 4

远端：节点以 `fe7f9d2` 运行 `bash scripts/run_upgrade_study.sh … --preflight-only`。token 检查与 StartLux-4B
预检通过（串行 90/90；并发 310/310，全部在 500 ms 内；KV 7.6 GiB；启动 269 秒）。StartLux-9B 在 0.50 下
KV 2.45 GiB，vLLM 0.31 报 `max_num_seqs (256) exceeds available Mamba cache blocks (151)`，引擎退出，CLI
以 BLOCKED 结束并导出诊断包。回传验收（本机）：失败收据记录了启动失败的进程组，停止核验无残留，GPU 清空。

修订：份额不变，三个引擎加 `--max-num-seqs 64`（客户端每引擎最多 16 个并发，等于 `max_inflight`，低于 151）。
新增 `launcher.py`，用 V1 `_service_specs` 生成命令后只追加该参数；`engines_limited` 在预检核验与单元完成
核验中检查记录里的引擎命令。演练改由真实命令构造生成每个服务的命令。输出改到
`results/aux-anli-upgrade-service-v1-r2`，首次输出保留为证据。本机 upgrade 测试 52 项，全仓 417 passed、
11 skipped、64 subtests。未运行模型或 GPU。

## 2026-11-08：revision 4 独立复审与首次诊断归档

复审 `fe7f9d2..16e734f`，软件验收通过，可交付新一轮 `--preflight-only`。逐步对照新 launcher 与 V1 单模型
启动/清理流程、三个角色命令、源码身份及协议/收据上限绑定；冻结源码保持。独立全仓测试 exit 0：417 passed、
11 skipped、64 subtests，106.68 秒，包含 52 项 upgrade 测试。核查 8 份 mutation 日志，未独立重跑；n3 是
回退 V1 GPU doctor 的间接捕获，另有正常路径接参断言，不构成阻断。

原始诊断 ZIP SHA `c9392e10dfad32fba5ba62ee513bdc4d0e93538cc44a2868b322f413595a6183`，7,436,661 字节，
74 个成员；112 项独立证据检查通过。35 份源码绑定 fe7f9d2，17 份准备文件与三个角色的 3,290 条 token 输入
及原始预检记录一致。4B 串行 90/90、并发 310/310 均及时；9B 256 > 151 Mamba cache blocks 引擎启动失败，
失败组已登记、停止核验 0.1206 秒且 GPU 空。诊断 UTC 时间为 2026-11-09T06:25:35.387456+00:00。
级联未运行，没有正式单元或评价。原 ZIP 复制到外部证据根
`records/bundles/upgrade-preflight-r3-20261109-c9392e10dfad.zip`，再核对完整 SHA；原件不变。
canonical BUNDLE_INDEX 从 21 项增至 22 项，原条目内容保持。

新增 `docs/UPGRADE_REVISION4_REVIEW_2026-11-08_ZH.md`，更新当前状态。下一步由 Claude 合并推送最终 SHA，
节点在新 `aux-anli-upgrade-service-v1-r2` 输出根重新预检三个条件并回传；64 的硬件启动、Qwen9B KV 和同驻留
仍待验证。独立审计与测试记录在 `.validation/revision4-review-2026-11-09/`。本次未修改运行源码/协议、
未运行真实模型/GPU、未 commit/merge/push。Claudeception 已评估，候选硬件修复未验证，不另建修复技能。

## 2026-11-09：revision 4 节点预检通过

远端：节点以 `7c81947` 运行 `--preflight-only`（输出 `results/aux-anli-upgrade-service-v1-r2`），三个条件 PASS。
StartLux-4B（0.40）KV 8.05 GiB，并发 310/310 在 500 ms 内；StartLux-9B（0.50）KV 2.93 GiB、78,039 token，并发
310/310 在 500 ms 内；级联强制转交时 4B 0.40 与 Qwen-9B 0.50 同驻留（17.7 + 20.7 GiB），310/310 成功、69 个在
500 ms 内，p50 624.6 ms，重叠 148 秒，Qwen KV 2.93 GiB。四次引擎启动的 vLLM 参数均含 `max_num_seqs` 64，停止核验均无残留。

回传验收：诊断包 `8adafb29…` 归档到外部证据根并登记索引；仓库核验函数在解包副本上确认 settings、token 绑定和
三份 PASS 收据。待 Codex 验收预检证据，之后在同一 checkout（`7c81947`）运行正式 36 单元；settings 固定该 commit。

## 2026-11-09：9B 服务成本正式测量完成

远端：节点以 `7c81947` 在 job 1224376 内运行 `bash scripts/run_upgrade_study.sh …`，15:29–18:07 UTC 完成 36 个
单元，随后冻结、读取 gold、评价并导出。每单元一个尝试，没有中断。

回传验收：诊断包 `9baeca3e…` 归档到外部证据根并登记索引。仓库核验函数通过 settings、预检、36 个单元、freeze
与评价身份；本机用密封 gold 从原始 trace 独立重算全部单元、汇总与整池历史准确率，与评价文件逐项一致。

结果：500 ms 下中位正确 goodput（10/20/40/80 请求/秒）4B 7.26/14.85/29.96/38.26，9B 7.47/15.33/30.75/0.29，
级联 6.95/12.76/0.37/0.15；三轮至少 99% 及时成功的最高速率 500 ms 为 4B 40、9B 20、级联无，1,000 ms 为 40/40/10。
级联实时转交 23.3–24.7%，路由与历史一致 99.0–99.5%。非成功状态仅有过载时的客户端拒绝。`r0-startlux4b-rate20`
有一次约 3.7 秒、主要在模型调用之外的停顿（及时率 96.9%），原样保留。待 Codex 审计。

## 2026-11-10：9B 服务预检与正式结果独立验收

审阅仓库 HEAD `23033c12bd2a5fd7fcb00354c8fb78257734d797`，节点源码仍固定 `7c81947`、协议 revision 4。
两条独立 stdlib 审计分别直接核验原 ZIP 的身份/生命周期和全部原始指标，不调用生产 evaluate/verifier，
不运行模型或重新拟合。预检包 `8adafb29…`、正式包 `9baeca3e…` 均通过；2,097 项完整性检查、36 份源码、
17 份准备文件、78 份 completion、52 次 cap64 引擎启动与所有停止确认一致。36 单元同一 L40S UUID 和 job
1224376，每单元唯一 attempt-0001；3,985 个正式 GPU 采样身份一致，freeze/access/evaluation 链完整。

162,801 次 offers = 131,935 成功 + 30,866 客户端拒绝，无其他状态。140,576 个实际 HTTP stage 预测核验，
35,860 次实时级联从概率独立重算路由，9,600 条历史预测对照原 V3/规模对照 384 个 shard 全部一致。
500/1,000 ms 全部指标、matched-subset 历史比较、三轮汇总和标签变化与 evaluation 一致。没有重跑实验的需要。

解释修正：99% 规则的通过点在 500 ms 下为 4B{10,40}、9B{10,20}、级联{}；1,000 ms 为{10,40}、
{10,20,40}、{10}，不能写成连续稳定容量。4B20/s 的 78 个晚请求 arrival 跨度为3.6226秒，另有1次拒绝；
记录的调用之外存在明显等待，根因未隔离。强制级联预检的148秒是两两HTTP区间交集累计，真实在途重叠并集
14.7769秒，不是GPU kernel并行证明。正式KV为4B8.78GiB、9B/Qwen9B4.22GiB，不能复用早期预检2.93GiB；
graph与KV前后变化的因果未隔离。原实验文件保持，当前状态已修正上述文字。

新增 `records/upgrade-service-acceptance-2026-11-10.json`、`docs/UPGRADE_RESULTS_REVIEW_2026-11-10_ZH.md`；
更新最近两个 BUNDLE_INDEX 条目的独立验收状态，原先条目保持。审计代码与CSV/收据位于
`.validation/upgrade-return-review-2026-11-10/{integrity,metrics,delivery}/`，尚未整合为clone便携报告。
Lab现有V1–V3证据检查PASS。论文/README仍有“9B服务未测”旧句，新结果未进入统一复现和Lab；建议Claude完成
报告图表、论文/复现同步和独立follow-up页，再做贡献确认与演示排练。研究范围足够，不再把新增GPU实验当作交付前提。
本轮未修改运行源码/协议或原始结果，未提交或推送。Claudeception 已评估，不另建重复技能。

## 2026-11-10：9B 服务结果整合进报告、论文、复现与 Lab

本机整合（Claude Opus 5.5，max effort），基线为 HEAD `23033c1` 加 Codex 的五个未提交复审文件；这些文件保持原样，
只在当前状态与本日志中补记。未运行模型或 GPU，未修改冻结实验源码、协议、原始 ZIP、gold、验收收据或旧报告与
旧便携包，未提交或推送。

报告：新建 `reports/upgrade-9b-serving-2026-11-09/`。`evidence/` 复制验收时的 6 份审计收据（SHA 与验收收据记录
一致）；`audit/` 中两个脚本改为便携版本，只改输入输出位置并加 `-O` 保护，本机重跑 6 份输出逐字节一致，从其他
工作目录或经 `EVIDENCE_ROOT` 运行结果相同。`build_tables.py` 从收据生成 24 行汇总、质量表与资源表，写出前重算中位数、
通过集与总数；`render_figures.py` 画 500/1,000 ms 两栏图（三轮中位数与两端值，三轮都达 99% 时实心，已测速率之间
不连线）。报告的 24 个 goodput 区间、24 个最差轮及时比例与 6 个准确率由脚本对照 `summary.csv` 和收据核对；起初
手写的 11 个三位小数有误，已改为由表格生成。

论文：新增 Figure 5（`paper/figures/fig_upgrade_serving.pdf`）与 RQ3 段落，更新摘要、引言计数（约 720,000 次计时
服务请求）、方法中的份额说明、讨论、限制、结论与附录；`\dbltopfraction` 设为 0.85，使两张服务图同页。12 页，
无未定义引用与 overfull；旧三张图逐字节不变。

Lab：新增 `decision_study/lab/followups.py`、`followup_view.py` 与便携包 `reports/decision-lab-followups-2026-11-10/`，
`app.py` 增加可选的 follow-up 包与 Follow-ups 页，`cli.py` 增加 `build-followups` 并在 `check` 中核验新包。
新增测试 `tests/test_lab_followups.py`（48 项）与 `tests/test_reproduce_cpu.py`（2 项）；10 处注入错误全部被对应测试
发现。

复现：`scripts/reproduce_cpu.sh` 扩为 7 步并支持 `--only`。`--strict` 全程 exit 0（165 秒；主环境 467 passed、
11 skipped、64 subtests，基线环境 65 passed）。只缺新正式 ZIP 时 strict 返回 1、非 strict 显示 SKIP；仓库副本中改动
一份收据时第 4 步返回 1。

浏览器与干净副本：真实 Gradio 服务（7871）与干净源码副本自己的服务（7872）都通过 headless Chromium 检查，
检查后停止；7861 端口已有的服务未触碰。干净副本用现有依赖：测试 461 passed、17 skipped，报告与论文图逐字节一致；
没有做全新依赖安装。

新增 `docs/START_HERE_ZH.md` 与 `docs/PRESENTATION_ZH.md`，并更新 README、README_ZH、报告与文档索引、Lab 说明、
artifact 指南、9B 运行说明和论文 README。待人工：成员贡献与致谢措辞、演示录屏与队友排练、Codex 独立验收。

## 2026-11-10：按 Ju 的分工指示与 Codex 复审修订整合

Ju 11/10 回复“你合理分配一下分工就可以了；不用致谢算力”。论文 Contributions 改为 Team responsibilities：Ju Zhang
负责实验设计、运行时与流水线实现、整合和最终技术审阅，Weiyu Chen 负责统计核验、结果解读和稿件审阅，Yizhi Hu
负责复现检查、演示讲解和展示材料；只写分工，不记录尚未发生的核对。删去算力致谢的待办和橙色拟议文字。

按 Codex 复审：报告图与论文 Figure 5 改为四个已测速率分组（组内偏移只为区分标记，每组画 offered rate），去掉按
数值偏移的对数横轴和斜线，旧三张论文图逐字节不变；报告、论文图注与 Lab 说明写明正确 goodput 按 120 秒到达窗口
选请求、按各自 SLO 判断及时。论文改正 80 请求/秒（最高正确 goodput，三轮都未达 99% 规则）、结论中的级联 99%
规则、CLINC 服务研究两个 allocation 的范围与“每项研究一张 L40S”的硬件表述，Figure 5 图注改为分组横轴和
“78 个晚到请求的计划到达跨度 3.6 秒”。Lab 案例说明区分 3,200 条保存预测与该单元成功返回的实时请求，第三类标题改为
“两个直接模型都答对、级联答错”，并给出所属更大类别（245、164、157 条）；follow-up 便携包重建两次逐字节一致。
审计 README 写明 `--bundle` 等选项只属于 `audit_metrics.py`；Lab 说明改正源码保持原字节的范围。

验证：Lab 与复现相关测试 75 passed（复现测试拆成非 strict 与 strict 两项）；13 处注入错误重跑并保留完整日志，全部由
对应测试以预期失败信息发现，M8b 与直接 strict 探针确认 strict 失败行为（注入后 exit 0，原版 exit 1），M2 记录为由
错误信息区分（篡改仍被后面的汇总核对拒绝）。论文重新编译 12 页，无未定义引用与 overfull，Figure 5 所在页已目视检查。
全量 strict 与浏览器验收留给 Codex 在本轮之后执行。未运行模型或 GPU，未提交或推送。
## 2026-11-10 Codex 独立整合验收

Claude Code 使用同一 Opus 5.5 session、显式 max effort 完成整合及两轮修订。Codex 在最终工作区分两次运行
`reproduce_cpu.sh --strict` 的全部七个阶段：主测试 470 passed、11 skipped、64 subtests（112.05 秒），基线 65
passed（7.59 秒）；两批 12 + 6 项收据比较、raw PASS、2,097 项完整性检查、原始指标重放、Lab 包重建、报告表图、
四张论文图和临时论文编译全部通过。报告清单 18 项及 6 份验收收据副本验证一致。

最终真实浏览器检查使用 `http://127.0.0.1:7860/`，阻断所有外部请求，遍历 8 个 ANLI 组合、3 档 CLINC 负载、
4 个案例和 V1/V2/V3 标签页，零页面错误。Codex 停止了自己此前的 7861 测试实例，最终 7860 实例保留供本机查看。
13 项 mutation 完整日志另经复核，M2 的后续汇总拒绝与 M8/M8b 的直接 strict 返回码检查均明确记录。

与整合前工作区快照相比，17 个允许修改的跟踪文件更新，其余 332 个基线文件、原复审文档、BUNDLE_INDEX 与
9B 验收收据保持原字节。新报告/论文/便携包、分工与无算力致谢已同步；无新模型/GPU 实验，未 commit/push。
新增 `docs/INTEGRATION_ACCEPTANCE_2026-11-10_ZH.md` 与 `records/integration-acceptance-2026-11-10.json`，
当前状态更新为独立验收通过。干净源码副本复用了现有依赖；未验证全新联网安装，也未声称队友已完成排练。

## 2026-11-10：直接准确率矩阵补全的协议文档与论文表格（本机）

Ju 当天授权补齐论文 Table 2 中五个 LLM × 四个评价集的 8 个未测单元。两个 Claude Code 会话（Opus 5.5，max effort）
并行：一个实现 `decision_study/direct_matrix/`、`configs/direct_matrix_protocol.json`、`scripts/run_direct_matrix.sh`
和测试，交接在 `.validation/matrix-completion-2026-11-10/claude-handoff.md`；本会话只改论文与文档。两边都没有运行
模型或 GPU，没有提交、推送或重置。节点 GPU 进度为 0/8，没有新的模型准确率。

实现方面已观察到的结果（来自其交接，本会话未重跑全仓 pytest）：`tests/test_direct_matrix.py` 43 项通过（222.85 秒）；
补强后前 5 项针对性测试含完整的中途 SIGKILL CLI 演练，通过（110 秒）；17 处单一守卫注入错误都按预期原因失败。
全仓 pytest 第一次为 505 passed、11 skipped、8 failed，失败都来自演练输出位于主 checkout 内而触发的目录重叠保护；
夹具改到临时目录后，实现会话在最终代码上单独重跑全仓：514 passed、11 skipped、64 subtests，exit 0，346.76 秒，
含全部 44 项 direct-matrix 测试（`final-pytest.log`）。Codex 的审阅提出 M1–M6、S1、S2，M1–M5 与
S1、S2 在复查版本中关闭，M6 的修复通过针对性探针。

论文：Table 2 改为两级表头（CLINC main；SQuAD 2.0 的 S1 test 与 S2 holdout；ANLI test），8 个未测 LLM 格标 NM，
CPU 行在 CLINC-only 协议之外的格写明未尝试，StartLux-9B 的 ANLI 格与 CPU 行加回顾性标记，图注写明各格的研究来源与
状态，并说明没有一格是主终点；去掉了在有缺格的列上暗示“最佳”的粗体。引言、RQ1、摘要和讨论中的比较限定在实际测过
的模型。Table 4 图注写明它是 Study 2 的 10% 主预算与 Study 3 的 Qwen3.5-9B 主后级，标出主比较两臂，“--”为不适用，
0 为实测值；Figure 2、3 的图注同样写明所选设置。附录新增 Table 7，列出 V2 的 3 个预算 × 3 个政策、V3 的 2 个后级 × 4 个
政策，并单列 V1 gate 在 V2 holdout 上的重放（5% selective-risk 目标，没有调用预算，validation 拒绝 53/1,000，test
92.325%、219/4,000 转交、救回 61、误伤 39）。validation 转交率逐行取自已验收的 `policy_freeze.json`，如 5% 预算下
confidence 4.3%、class_conditional 4.5%，20% 预算下 19.8% 与 19.9%。四份源文件的 SHA-256（`d1493c56…`、`03f549a6…`、
`5bd4cf2a…`、`28efefe7…`）与 Codex 的提取记录一致，V1 gate 的 947/1,000 另与 `reports/formal-2026-11-02/analysis/metrics.json`
核对。RQ2 中“实际转交率与 validation 相差不超过 1.7 个百分点”限定为 10% 预算，三档预算合计为 2.3 个百分点以内。
Limitations 加入 8 个未测格及其影响。原 Table 9（只含 5%/20% 的预算敏感性）并入新表。

编译：pdflatex、bibtex、两遍 pdflatex，13 页（此前 12 页，多出的一页来自附录全宽表），无未定义引用、overfull、`??` 或
Type 3 字体；与基线源码的同条件编译相比，新增 2 个 underfull hbox（摘要换行）和 3 个 underfull vbox（附录浮动体）。
逐页渲染检查了 Table 2、Table 4、附录与改动段落；第一次编译时 Table 2 因跨列表头被缩得过小、附录 B 节只剩标题，
已改正。`bash scripts/reproduce_cpu.sh --only paper` 退出 0。Codex 用只读解析器在 `main.tex` SHA `e4a71a05…` 上核对了
附录 18 行、主/参照标记、validation 选中标记与 Table 2 的 12 个历史值和 8 个 NM，未发现数值或状态问题。之后的改动
只涉及版式（Table 2 表头与 † 位置、附录表顺序与引导句）；最终源码 `f103726e…`（PDF `753c2e3b…`）由本会话的只读脚本
`verify_tables.py` 对照已验收 JSON 与协议固定的正确数重新核对通过，改错附录数值、把 NM 改成数值、挪动主比较标记
三种注入都使它以预期信息失败。完整编译日志、渲染图与核对脚本在 `.validation/matrix-completion-2026-11-10/paper/`。

文档：新增 `docs/DIRECT_MATRIX_PROTOCOL_ZH.md`、`docs/RUN_DIRECT_MATRIX_ZH.md`，更新 `docs/CURRENT_STATE.md`、
README、README_ZH、`docs/README.md`、`docs/ARTIFACTS_ZH.md`、`scripts/README.md`、`paper/README.md` 和
`docs/PRESENTATION_ZH.md` 第 3 张幻灯片的同一句。历史报告、验收收据与已有日志条目不变。最终验收由 Codex 另行追加。

## 2026-11-10：Codex 验收直接准确率矩阵补全，交付节点前准备

实际 Claude Code Opus 5.5、max effort 完成实现与论文，Codex 独立审阅通过。最终全仓 514 passed、11 skipped、
64 subtests，exit 0（346.76 秒），其中新增模块 44 项；Codex 独立快速检查 24 passed。17 项隔离 mutation 原始日志
复核通过。首次全仓的 8 项目录重叠失败保留，修正测试夹具位置后重跑通过，生产保护不变。

第 53 行后 SIGKILL、50 条分片加 3 条 journal 的恢复、换 allocation 后只采剩余 3,947 条，以及两种末行故障均已
通过假服务 CLI 演练；12 个历史计数一致，离线 audit 通过。未运行真实模型或 GPU，新增真实预测为 0。
论文 13 页，Table 2 / Table 4 / 完整附录 Table 7 分别在第 5 / 7 / 12 页，数值、主次标记与排版验收通过。

新增本机验收文档与机器可读记录；370/381 个本轮基线文件保持原字节，11 个变更为论文和入口文档，旧科学源码、
协议、报告、Lab 包与历史收据保持。本轮交付采用独立分支 matrix-completion-2026-11-10，节点按最终交付 SHA
固定源码并运行自动预检与采集。运行说明用 git fetch origin 取得独立分支，并说明前置失败可能没有诊断包。
真实 8 单元、20,400 条测试预测待节点运行及回传验收；不将假服务标签当作实验结果。

## 2026-11-11：节点运行时恢复结果与共享 blob 验证修复（本机）

节点用恢复工具 `d1633be` 运行了 plan 与 execute，终端输出存于本机
`.validation/runtime-cache-layout-2026-11-11/node-recovery-output.txt`。plan 为 READY_TO_EXECUTE，可用磁盘 8,467.2 GiB，
四个研究锁空闲。execute 的七个阶段全部完成：两个环境在 legacy 上下文中为 Python 3.12.8，freeze 哈希与存档一致；
CUDA 工具链 tree `56cbb21d…`、安装器 `0a5f3eac…` 与需求文件 `5fd9773b…` 与存档收据一致；矩阵 settings（`28aa5eca…`）
校验通过，冻结 `plans()` 已算出，尚无 binding。两个源码检出的状态为 extras，多出的是安装后生成的 `build/lib` 与
egg-info。最终 verify 报 16 条快照问题（kev4b-adapter 3 条、kev4b-base 3 条、qwen9b 5 条、startlux9b 5 条），都是
`does not resolve to a blob of its repository`，状态为 MATRIX_RUNTIME_NOT_READY。命令链使用 `set -e`，随后的矩阵
preflight 没有运行。

节点确认了两条真实链，每个环节都存在，只有最后一个文件是普通文件：Kev `head.pt -> ../../blobs/dd633435… ->
../../blobs/fd/fdfd452d…`，StartLux-9B `model-00001-of-00004.safetensors -> ../../blobs/6de4c4ab… -> ../../blobs/76/76981d8e…`。
Root 与独立复核者对照本机 huggingface_hub 2.4.0（与存档 kev 环境相同）的 `file_download.py`、`utils/_shared_blobs.py`
和官方缓存文档，确认这是原生共享 blob 布局，独立复核还在临时快照中复现了同一报错。本会话经 WebFetch 读取
1.33.0（存档 vLLM 环境）的 `_shared_blobs.py`，标记名、版本号与目标形式相同。仓库 blob 名是 ETag，共享存储里的
文件名是 Xet 哈希，后者不是文件的 SHA-256。

Claude Code（Opus 5.5，max effort）只修改 `scripts/recover_node_runtime.py`：新增 `hub_blob_payload` 与两个常量，
`verify_snapshot` 改为经它定位负载（+54/−6 行，其中一行注释由 Root 在会话中改写）。普通文件形式的仓库 blob 沿用原规则；
链接形式只接受 `../../blobs/<xet[:2]>/<64 位小写十六进制>` 字面目标，要求仓库目录名形如 `models--…`、缓存根
`blobs/.huggingface-shared-blobs` 为内容 `1\n` 的普通文件、负载为普通文件，且缓存根以下的目录都不是链接。负载按仓库
blob 名流式计算 SHA-256 或 Git blob SHA-1。仓库自己的 `blobs` 目录为链接时现在也拒绝，原版本会跟随。拒绝原因写在报错
末尾的括号里。pin、TOOL 与 MATRIX_COMMIT 常量、旧验收记录、冻结源码与配置保持原样；脚本 SHA-256 由 `c1d6118f…`
变为 `b140e16c…`。新增 `tests/test_recover_node_runtime_shared_blobs.py`（31 项），并更新恢复手册（原因与第 8 节只读
verify 步骤）。

本机验证：针对性探针中，原脚本对 huggingface_hub 2.4.0 自己写出的共享布局报出与节点相同的错误，修复后通过，篡改负载后
报内容不符。新测试用真实临时缓存覆盖旧布局、共享布局、混合布局与跨仓库共用负载，18 种链接与目录逃逸，ETag 与 Xet 名
互换的检测，旧布局与原脚本逐项对照，以及一次 CLI 演练（共享缓存 READY，篡改一个负载后 NOT READY），每次 verify 后都核对
缓存未被写入。同一组新测试对原脚本运行时 24 项失败、7 项旧布局对照通过，CLI 演练得到与节点同类的报错。最终 74 项
（原 43 + 新 31）一次运行通过，52.49 秒，运行期间文件不变。5 个隔离破坏版本（以 Xet 名作为期望摘要、任意跟随链接、
不查标记、不查前缀、跟随存储目录链接）都被识别，失败原因已逐项查看。节点上 16 个大文件的内容哈希、GPU 与矩阵 preflight
待节点复核；本会话没有运行 GPU、下载、安装或远端命令，没有提交或推送。交接见本机
`.validation/runtime-cache-layout-2026-11-11/HANDOFF.md`。

## 2026-11-11：Codex 独立验收共享缓存修复

最终脚本 SHA-256 为 `b140e16c8fee33346011095c7f56b548420eb2f2524cbf2a0577559139f7ebb3`，pin 保持
`7f3e27c095ead75991eef9a822695d211bb7fe649506687bead57806810c4e56`。Codex 独立重跑恢复测试：74 passed，
exit 0，51.43 秒，无跳过，运行期间脚本与测试文件哈希不变；18 项独立临时缓存探针全部通过，缓存树保持原样。
5 项隔离破坏测试的实际失败原因已复核；其中 M5 的目录逃逸被错误接受是有效证据，其余仅因诊断措辞不同的失败不计为有效证据。

新增 `records/runtime-cache-layout-2026-11-11-acceptance.json`。与恢复发布 `d1633be` 的 409 个基线文件比较，
只修改恢复脚本、恢复手册、当前状态与追加实验日志，其余 405 个文件保持原字节；新增 31 项测试文件与独立验收记录。
旧验收、安装 pin、科学源码、协议、报告和矩阵提交身份均保持。

本机验收允许用户更新现有恢复检出并运行只读 verify；七个恢复阶段无需因本次链接布局错误重做。节点修复后验证与 GPU
preflight 仍待执行，新增矩阵实验仍为 0/8。本次没有在节点执行命令，也没有把本机假服务测试记为真实实验结果。
