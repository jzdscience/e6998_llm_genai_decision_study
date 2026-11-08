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
