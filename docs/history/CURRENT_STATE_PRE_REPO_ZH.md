# 当前状态与恢复入口

## 2026-11-04：CPU Lab 已覆盖V1–V3

后续源码目录 `decision_study_repo` 已完成三轮CPU演示；新入口为 `bash scripts/run_lab.sh`，
完整验收与下一步以新仓库 `docs/CURRENT_STATE.md` 为准。原始实验与以下历史资料保持原口径。

## 源码仓库已接管（2026-11-03）

当前源码与最新状态迁至 `../../../decision_study_repo`（相对本旧项目根需到 COMS6998）；
准确绝对位置为 `/Users/zhanj289/Desktop/COMS6998/decision_study_repo`。
用户导入提交 `fa113e5` 已含 V3；本机补齐报告、索引与管理说明，原科学源码保持不变。
V2/V3 原发布校验在临时数据树通过，CPU 233 项通过、9 项跳过、7 个子测试通过。
新仓库 `.git/objects` 仍有云端占位文件，diff 读取超时，须完整下载后再检查/提交；
本次未 commit/push。此旧树继续保存原始包、审计、环境和以下历史记录；后续开发先读新仓库状态。

最后核对：2026-11-03（用户时区 America/Los_Angeles）。证据文件中的时间保留 UTC。
这是当前状态摘要；历史事件见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)，
原始包索引见 [BUNDLE_INDEX.json](../records/BUNDLE_INDEX.json)。
`VALIDATION.md` 和早期恢复说明含历史状态；若文字冲突，按最新回传证据核实，
不要将早期“尚未通过”的描述套到已经完成的 Kev 阶段。

## 接手时先知道这五件事

1. **15,950 条正式预测已全部完成**：Kev 6,250、StartLux 6,250、Qwen 3,450；
   三者均零失败、零重试，逐样本/分片/原始响应及配置绑定核对通过。
2. **正式统计分析已完成并复算一致。** 两模型的温度/gate仅用calibration与
   validation数据重建，完整artifact一致；Kev原有253个受保护文件保持不变。
3. **StartLux在两项主测试领先**：CLINC93.50%、SQuAD91.25%；Qwen为88.00%、
   86.00%，Kev为77.50%、81.35%。这只支持本次冻结任务/模型的比较。
4. **级联收益有限且依赖模型。** StartLux→Qwen在CLINC无净增益，SQuAD仅净增
   5/2,000（+0.25pp，95%配对区间−0.293～+0.825pp）。5%验证集门槛不是测试风险保证。
5. **live benchmark也已完成**：60个测量单元、60,200次请求，零失败/重试；
   五种部署均支持并发8，双模型驻留通过。预定v1 GPU实验无需重跑；用户随后要求
   评估新增实验和前端，并已授权按收敛方案执行。**v2也已完整验收**：10,800条预测、
   政策冻结、36/36性能单元和4,000题最终评价全部通过；主比较未显示gain优于confidence。
   **v3也已完整验收**：17,700条ANLI预测、六组completion、freeze与3,200题评价齐全；
   positive_gain零转交，主比较+0.594pp、95%CI[−0.528,+1.674]pp。CPU Lab仍展示v1，
   尚未接入v2/v3结果。

## 最新回传：V3完整完成，零转交主政策与StartLux相同（2026-11-03 15:25:29 PST）

原包SHA `a8d0dc672d83c3edcb85d005e66fa4b3d000132729f98f0beaab6865dfc57a9e`，
819个安全唯一成员、818份清单文件大小及SHA通过。已归档、更新索引并保存
[完整报告与图表](../reports/v3-complete-2026-11-03/RESULTS_ZH.md)。**预定V3无需GPU补跑。**

- 六job共17,700正式预测、708分片；另1,080 smoke。18,780请求ID全局唯一，零失败/
  重试，原响应/输入/模型revision/源码/配置/token无截断/完成收据全部核验。四份准备
  数据与发布逐字节一致。原V3 25新文件+22依赖、此前91项报告/审计/准备证据保持不变。
- 三组fit→freeze→三组holdout→首次标签访问→评价/导出顺序成立。holdout共3,200题、
  2,854 premise组，含1个跨round组；独立统计脚本未使用项目评分/路由/存储函数，复现
  全部模型/政策/round/互补性与2,000次seed6998配对cluster bootstrap。
- 主比较positive_gain9B−confidence9B为+0.59375pp，95%CI[−0.527672,+1.674124]pp。
  两个positive_gain在validation均冻结mode=none，holdout零转交，与直接StartLux逐题
  一致；不是成功识别互补性的证据，区间跨零不能宣称主比较改善。
- 直接StartLux2374/3200=74.1875%，Qwen4B1773=55.40625%，Qwen9B2119=66.21875%。
  9B confidence2355=73.59375%，救回147/误伤166，实际转交743=23.21875%，高于
  validation10%。9B class_conditional2381=74.40625%、转交23.9375%，仅次要描述。
  9B可救回前级错误289题、反向误伤544题；oracle83.21875%不是可部署政策。
- 验证/测试准确率和置信度存在明显差异：StartLux90.5556%→74.1875%，confidence
  中位0.9520→0.8131；state字符中位397→402。只描述观察，不据此断言具体漂移成因。
- L40S、同记录host/job、BF16、vLLM0.31.0；9B指定revision两次成功加载，模型加载
  日志16.8GiB并非总运行显存。初始化至导出71分55秒；六预算closed、21日志无OOM/
  shutdown前ERROR，关闭后EngineDeadError保留。未测9B共驻留或steady-state goodput。
- StartLux raw health称prefix_cache=true，但两次启动参数与engine/plugin日志明确off；
  元数据矛盾已报告，上游health实现未在包中，根因未知。没有修改原健康响应或运行配置。
- 本机复拟合freeze有65处浮点差、最大8.88e−16，validation指标和全部fit/holdout
  路由一致。保留原freeze/hash，以原参数独立复算，不在本机覆盖冻结收据。
- 审计在 `.validation/v3-complete-2026-11-03-152529/`。下一步可整理源码仓库、报告与
  CPU Lab；当前ANLI holdout已查看，后续方法选优须另立协议和评价边界。包不能证明
  当前allocation是否还活跃，源码仓库整理尚未实施。

## V3之后的扩展建议（2026-11-03，尚未执行）

用户询问是否还值得扩展。最新建议见[POST_V3_SCOPE_ZH.md](POST_V3_SCOPE_ZH.md)：
先收敛源码并升级Lab接入V2/V3，再只增加一轮“固定路由+在线预算约束”的质量/服务
取舍实验；语义特征router为可选开发候选。旧MONTH_SCOPE中的Qwen9B与ANLI已完成，
StartLux9B、轻量分类基线、加强系统测量及稳健性/live尚未完成。用户追问与原计划关系后，
已在该文档补充逐项对照；收敛建议没有自动取消原待办，两个新增路由方向不冒充原必做项。
这些是候选建议，尚未冻结V4协议、启动新GPU实验或获得远端源码快照；正式结果不变。

## 仓库管理建议（2026-11-03）

用户询问本机与服务器目录不同如何管理。已核实本机活动目录尚无可用Git历史，
`v2/` 与 `v2 2/` 为不同实现；后者九个Python文件匹配原V2发布。建议旁建经过发布身份
核验的源码仓库，以commit统一代码、独立管理环境和实验产物，详见
[REPO_MANAGEMENT_ZH.md](REPO_MANAGEMENT_ZH.md)。当前V3存在运行状态绝对路径绑定，
发布校验也依赖默认prepared数据位置；此次仅记录建议，未迁移目录、初始化Git或发布远程仓库。
远端当前整树身份尚未获取；下方V3运行入口保持不变。

## 此前V3 ANLI准备与交付（2026-11-03，现已回传完成）

用户已明确“进行下一轮实验吧”，本机据此完成独立V3协议、实现、准备数据与交付。
[V3协议](V3_PROTOCOL_ZH.md)、[运行步骤](RUN_V3_ZH.md)、
[交付ZIP](../../decision-study-v3-ready.zip)。交付当时尚未运行真实V3模型预测或评价；
随后用户在公司allocation完成，最新验收见上。本机仍没有公司GPU执行入口。

- 新问题为ANLI三分类，StartLux4B前级、Qwen4B/9B后级分别比较；主比较预定为9B后级
  positive_gain−confidence，validation预算上限10%，允许不转交，不按validation换主方法。
  9B revision固定 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，是否更强/能稳定运行待实测。
- SQuAD本地477篇均已被V1/V2覆盖，剩余题目不再称新文章holdout。ANLI官方test全保留
  3,200题/2,854个规范化premise组；从官方train按组隔离后选cal1,800、val900、dev90。
  三个模型各采fit2,700+holdout3,200，共17,700正式预测；首次六启动另有1,080开发smoke。
- 准备manifest为 `775d236a3ba55ee0b07d1c51bdedd732b7e6960105595236c76553c92fe7b3c4`。
  官方train/dev与test确有同premise，已排除拟合候选中的test材料、冲突及重复训练input。
  拟合/采集/status不读sealed gold，所有fit完成→freeze→holdout采集→身份门禁→首次访问。
- 使用独立 `decision_study/v3/` 和 `results/solo-v3/`。不依赖当前本机缺件的v2包布局，
  不覆盖已发布v2或其“v2 2”副本。新runtime按角色隔离、单模型顺序运行，4B/env/toolkit
  只读复用，9B新增缓存写v3/runtime-work；本轮不宣称已验证9B同驻留级联性能。
- 启动入口 `bash scripts/run_v3.sh start` 从当前Slurm RUNNING作业读取实际余量并减5分钟。
  同allocation重复启动不延长deadline；真正换新allocation后才加 `--new-allocation`。
  如果Slurm不可读，需实际剩余小时数，不能沿用旧20小时或脚本变量污染值。
- **78项测试、5个子测试通过**；真实localhost HTTP仿真验证采集/重试/中断恢复/完整流程。
  24个隔离关键防护mutation被捕获，原版/时间戳版/当前三个真实接收树均通过实际ZIP验收，
  每个接收树的3种故意文件改动均被拒绝。无allocation时pipeline在GPU前阻断并导出诊断。
- 交付ZIP2,065,217字节，26个安全唯一member（25新文件+manifest），22旧依赖只核验。
  SHA256 `8fdd6f8dbbff4f9daac2cd342ebc0a2a67ac77f1f5d8a7b805862fdfb2237fb9`。
  原V1/V2源码/归档与67项报告/审计SHA保持不变。验证位于
  `.validation/v3-preparation-2026-11-03/`，发布记录 `records/releases/v3-anli-2026-11-03.json`。
- 交付时要求回传 `results/solo-v3/exports/v3_bundle.zip` 并核齐六份completion、freeze、
  首次access和evaluation收据；最新回传已满足，同名中间ZIP本身仍不能证明完成。

## 最新回传：V2完整完成，主比较未显示改善（2026-11-03 00:28:52 PST）

最终包 SHA 为 `f340f4b159cb21a0df35a6386db1800e263e85fc012ff85cd2449e7d21f2cdec`。
646个manifest文件/647个安全唯一member全量通过；已归档、更新索引并保存
[完整结果报告与图表](../reports/v2-complete-2026-11-03/RESULTS_ZH.md)。

- 原449个预测/数据/freeze文件及原12个openloop文件保持不变；此前10,800条预测验收
  继续有效。新增151文件、仅7个runtime文件更新；22个冻结v1依赖、原始发布ZIP和
  此前50项报告/审计SHA不变。复算从原发布ZIP隔离加载，不使用本机其他v2开发版本。
- **主比较 gain−confidence = −0.225pp，95%文章配对bootstrap区间[−0.908,+0.478]pp**。
  4,000题/57篇，2,000次重采样完全复现。gain 91.600%、confidence 91.825%，直接
  StartLux 91.775%、Qwen 85.950%。任务是SQuAD2可答性二分类，不是答案EM/F1。
- 主候选实际转交383/4,000=9.575%，救回0、误伤7；对照转交440=11%，救回93、误伤91。
  测试实际预算不相等，但均是validation冻结在线政策。区间跨0不证明等价或确定劣势。
  预定5%目标class-conditional为92.900%，实际转交6.5%，仅属次要描述，不能替换主比较。
- 36/36性能单元全部通过原响应、路由、到达表、占用事件与指标复算。26,984 offered，
  25,998接纳且成功，986队列满拒绝，零模型失败/重试；含预热27,457底层调用通过。
  40/秒时三轮平均500ms SLO goodput：StartLux与resident-idle均40.433/秒，gain级联
  4.889/秒，Qwen单独1.056/秒。范围是重复间描述性范围，不是置信区间。
- 跨allocation保留3项旧卡测量与33项新卡测量；两张均为同配置L40S。单独报告同新卡
  r1/r2敏感性，40/秒下分别40.133、40.133、6.483、1.317/秒，方向一致。r0-local_only
  10/秒有26项dispatch停顿后的超SLO观测，完整保留；日志不支持归因JIT/冷启动/GPU。
- 新benchmark预算正常关闭；最后服务停止UTC08:28:39.441，首次记录的holdout访问
  UTC08:28:46.618，评价收据08:28:50.192。评价前先验证冻结/输入/源码/收据，再读取
  已评价的gold独立复算。新增日志未见OOM或关闭前ERROR；历史last_error与未关闭旧
  prediction预算不表示最终未完成。不能据此解释此前中断根因或推断当前远端状态。
- 审计、CSV与复算脚本位于 `.validation/v2-complete-2026-11-03-002852/`。
  **预定V2无需继续GPU补跑。** 下一步可将结果纳入课程报告和CPU Lab；更强后级或允许
  不转交的政策属于后续独立协议，不能再用本次holdout选优后宣称验证性收益。

## 此前回传：V2性能3/36项，第二组启动未完成（2026-11-02 23:28 PST）

第二份v2_bundle.zip SHA为 `d1638668d6f8988067d3e9a0c9480541f393006aa3e7082176dabd84c5993199`，
495个manifest文件/496个安全唯一member通过。已归档、更新索引并保存
[新增测量与恢复状态报告](../reports/v2-openloop-partial-2026-11-02/STATUS_ZH.md)。

- 上一包的449个预测/数据/freeze文件逐字节不变，10,800条预测与政策冻结继续有效。
- 新增r0-selected_cascade在10/25/40每秒的三项有效测量；3/36单元通过summary/trace/
  occupancy/settings/source复算。40/秒有125拒绝、1055成功但超500ms，属于保留的过载
  观测，不是需删除重跑的坏单元。含warmup共2419底层请求零失败/重试。
- 最后证据在r0-resident_idle启动段：18:30:54 PST开始，18:31:59 StartLux健康，Qwen
  已创建但日志为空，缺health/ready/正常清理与新Python异常。不能从包判断OOM或退出原因。
  原发布没有只跑首组就正常成功退出的分支；尚无evaluation/holdout_access。
- 用户确认终端已返回prompt、未见报错。23:32PST当前节点sc1ns051is14的GPU为空，
  pgrep无相关进程；此前节点是sc1ns052is14。随后SLURM_JOB_ID与squeue确认当前仅有
  job65577162在sc1ns051 RUNNING，TIME_LEFT为20:41:20，查询中无旧节点活动作业。
  已具备本次新allocation恢复所需的用户报告证据；旧作业具体退出原因仍未确定。
- 后续保留原代码/配置，按刚才观测保守初始化20小时（命令延后执行须重新读取剩余时间），
  再分步benchmark→成功后evaluate→export；原恢复逻辑在服务启动前跳过完整首组，不重跑
  已验收预测或这三项测量。跨节点时保留allocation/设备边界。
- 已交付顺序命令采用失败即停止的subshell并打印退出码；只有所有阶段成功才打印
  V2_COMPLETE。本机未代替公司节点执行；完成仍以新回传的36项及评价证据核验。
- 审计位于 `.validation/v2-return-2026-11-02-232800/`。原42个报告/审计SHA未变；
  本轮未运行GPU、未读取sealed标签、未改freeze或发布科学源码。

## 上一回传：V2采集与政策冻结已验收（2026-11-02 18:26:57 PST快照）

用户报告“成功啦”，上传v2_bundle.zip。已验证473个manifest文件/474个安全唯一成员，
归档SHA为 `846dc48d2aba19709a3a0587692816dabada62c9e819ebc65f5df912c93779e2`。
完整报告见 [V2采集验收](../reports/v2-collection-2026-11-02/ACCEPTANCE_ZH.md)，
四项可复算审计位于 `.validation/v2-return-2026-11-02-182657/`。

- Qwen fitting 2,800条、StartLux holdout 4,000条、Qwen holdout 4,000条均完成。
  432分片、全部10,800条记录逐条核对通过，首次成功、零失败/重试，原响应解析一致。
  384条smoke对原v1基线复核通过；配置/请求/runtime/源码身份和无截断检查一致。
- holdout共57篇新文章，与v1的ID、文章及input/context hash无交集；准备数据四文件与
  原发布逐字节相同。只做sealed文件字节校验，没有打开标签或本地评分。
- policy freeze已验证并独立重拟合，九组validation决策零差异，最大浮点差1.55e-15。
  主预算10%选择gain；validation为935/1000，confidence926，直接StartLux936。
  这不是holdout结果，也不证明优于直接本地；当前冻结政策不修改。
- **这个包未包含任何openloop文件、holdout_access或evaluation，完整V2尚不能标记完成。**
  导出时间紧接最终采集completion后1.125秒，符合性能测量前的中间export；budget尚未
  close。用户后续确认终端已返回prompt、未见报错；更新证据见上方第二回传。
- 旧last_error为UTC01:56未初始化runtime，后续UTC02:00已初始化并完成采集。旧模型
  shutdown后的EngineDeadError与阶段清理一致，不能解释成新的请求失败或OOM。
- 当时请求重新export；用户已回传第二份快照，当前恢复入口以上方最新状态为准。

**此前v1完整回传（2026-11-02 13:07:35 PST）**：benchmark包921个manifest文件全部
校验通过，`experiment_ready=true`、`benchmark_ready=true`；正式预测/校准/分析共669个
文件与上一包逐字节一致。last_error仍是历史curand.h失败，当前benchmark阶段complete。
详见[正式结果报告](../reports/formal-2026-11-02/RESULTS_ZH.md)和
[性能报告](../reports/benchmark-2026-11-02/BENCHMARK_ZH.md)，审计位于
`.validation/{formal-experiment-2026-11-02,benchmark-2026-11-02}/`。
保存的内部deadline仍为 **2026-11-02 14:50:48 PST**；后续CPU文档工作不需要续期。

## 此前阻断与修复：v2发布兼容（2026-11-02）

用户在公司机执行 `verify_v2_release.py` 时报告 `artifacts.py` 与发布清单不同，
尚无本轮 v2 GPU 启动证据。这是制作包只验本机最新导出器、未验历史接收环境的遗漏；
公司机器的具体实际 SHA 尚未回传，不能把本地重建视为远端身份已确认。

- 已直接比较历史 ZIP；最终 v1 bundle + CUDA v3 补丁覆盖22个依赖，其中21个与当前
  相同，只有导出器不同。两个已交付旧版本的其他函数 AST 与当前一致，变化只在导出；
  v2 经 preflight 导入旧模块，但使用独立 exporter。没有放宽模型/数据/分析源代码身份。
- [小修复包](../../decision-study-v2-release-fix.zip)只更新发布清单和校验器，保留服务器
  `artifacts.py`。只允许已审计的 b0586c…1831、d7c873…9dd2 两个旧 SHA，未知仍阻断。
  安装器先在临时接收树验证完整候选，再备份并原子替换两个文件，支持幂等和中断续装。
- 已从真实交付 ZIP 重建三种接收环境；两个旧版均复现原错误，补丁后三者均通过30个
  交付文件与22个依赖检查，prepared status通过。无allocation的pipeline仍在GPU前拒绝。
  38项测试与8个子测试通过；verifier额外9项内存mutation全部检出。
- 修复ZIP为11,245字节，SHA256
  `b1e6eaea18341c642b8a2b2151e31efb110e515322db2824bd548f246852c53f`。
  原v2 ZIP、两份原始回传ZIP、33个报告SHA、13个v2源码/协议文件和v1冻结身份均保持不变。
  证据位于 `.validation/v2-release-repair-2026-11-02/`。
- **下一步按 [REPAIR_V2_RELEASE_ZH.md](REPAIR_V2_RELEASE_ZH.md) 安装修复并运行v2。**
  session仅在实际新allocation后用平台剩余时间初始化；同一有效allocation不重复初始化。
  本机没有运行GPU；等待公司机器的 `results/solo-v2/exports/v2_bundle.zip` 确认实测进展。

## 此前执行进展：v2本地实现（2026-11-02，首次GPU回传之前）

用户已明确“准备好了就可以照做”。执行协议见[V2_PROTOCOL_ZH.md](V2_PROTOCOL_ZH.md)，
服务器步骤见[CONTINUE_V2_ZH.md](CONTINUE_V2_ZH.md)，前端见[LAB.md](LAB.md)。

- 已准备`results/solo-v2-prepared`，固定Qwen fitting 2,800条及fresh SQuAD holdout
  4,000条（可答/不可答各2,000，57篇与全部v1输入不重叠的文章）。数据manifest
  `256ce56041ca2ef6717a9e7b8c5983772b2ee1ab800ecf356d6ddb2eefc44137`。
- 独立`decision_study/v2/`已实现class-conditional、gain及confidence对照；calibration
  拟合、validation选择/冻结在线阈值，主预算10%，holdout按文章配对bootstrap。
  采集、拟合和status路径不解析封存标签；首次评价另留freeze绑定和不可覆盖收据。
- 新GPU流程复用v1固定服务与v3 CUDA工具，独立runtime-control、job journal和返回包；
  初始化必须使用真实新allocation剩余时间。性能设计为4条件×3到达率×3重复共36单元，
  记录排队、拒绝、失败、SLO goodput和独立worker占用事件。
- CPU Decision Lab已从真实归档构建`reports/decision-lab-2026-11-02`，提供3,450个
  唯一test/stress输入、四例路线说明、阈值探索和60个已测性能单元。网页依赖装在独立
  `.venv-lab`。它显示v1证据，不冒充v2结果或实时模型推理。
- 数据/方法/runtime/openloop/Lab共65项新CPU测试通过；既有114项测试及2个socket
  子测试通过。socket测试需本机端口权限；GPU服务尚未在本轮运行。详细收据与网页验收
  位于`.validation/v2-implementation-2026-11-02/`。
- [v2交付ZIP](../../decision-study-v2-ready.zip)已生成并在独立临时目录做实际解压核验，
  包含新代码、准备数据和便携Lab证据，不包含模型权重/环境，也不覆盖原v1源文件。
  浏览器已验收四例切换、JSON下载、阈值控制、并发性能切换及390px手机布局。
- **尚未采集v2预测、拟合真实v2政策、评分新holdout或测量新GPU性能**。真实数据下fit
  因缺Qwen完成记录正确停止；没有用人工测试夹具替代实验。9B、live UI、ANLI暂不加入。

以下两段保留此前规划顺序，执行范围以上述协议为准。

**后续资源假设评估**：用户提出单L40S可持续使用至12月的假设，新增
[MONTH_SCOPE_ZH.md](MONTH_SCOPE_ZH.md)。建议完成v2后，再扩较大后级/规模对照、一个新任务、
便宜基线和重点系统测量。该假设不代表已取得allocation或授权续期；现有冻结v2与交付包不变。

**此前工作方向（同日，完成benchmark之后）**：用户希望利用后续时间扩展研究，并加入
前端演示。详见[扩展路线图](EXTENSION_ROADMAP_ZH.md)。建议聚焦“一个新ANLI任务上的
净收益路由 + Decision Lab”，先做CPU回放/阈值探索，再接常驻live服务；较大后级、风险
认证、稳健性与开放到达负载为备选，未全部列入必做。此为分析建议，不是已执行的v2。
原任务缺Qwen cal/val共2,800条；若直接做ANLI主线可以暂不补，此缺口不覆盖新任务成本。
新方法必须有独立协议/holdout/源码身份，不能修改v1冻结结果或继续在旧test上选优后称无偏。

**再后续：Claude方案交叉评审已完成**，见[CLAUDE_V2_REVIEW_ZH.md](CLAUDE_V2_REVIEW_ZH.md)。
按附件提出的较紧范围与提前收尾目标，建议转向旧任务补Qwen fitting数据、fresh SQuAD、
小型常驻/开放到达系统对照和CPU Lab；ANLI保留备选，尚未冻结执行协议。补齐两任务
为2,800条，若只做SQuAD则2,000条。新增CPU审计确认57篇未用文章有14,157题，原始
数量足够4,000平衡抽样，但不保证0.6pp功效。原benchmark转交率5%，不是正式集6.25%；
c8前级阶段在级联中137.58→163.67ms，Little定律不能单独归因23.6%吞吐损失。
详见`.validation/claude-review-2026-11-02/`。本轮没有启动v2、选定holdout、迁移工作区或Git初始化。

## 文件在哪里

| 用途 | 位置 |
| --- | --- |
| 本机源码与记录的主目录 | `/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study/` |
| Codex 项目入口指令 | [project/AGENTS.md](../../../AGENTS.md) |
| 用户原始 project / 交付目录 | `/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/` |
| 公司服务器项目 | `/home/zhanj289/e6998/decision-study/` |
| 公司运行输出 | `/home/zhanj289/e6998/decision-study/results/solo-v1/` |
| 本地原始 ZIP 归档 | `records/bundles/`（带时间和 SHA256 前缀，详见索引） |
| Kev 正式结果解压副本 | `results/returned-kev-2026-11-01/` |
| 最新完整正式实验副本 | `results/returned-experiment-2026-11-02-101054/` |
| 最新完整benchmark副本 | `results/returned-benchmark-2026-11-02-130735/` |
| 性能报告、表格及独立图 | `reports/benchmark-2026-11-02/` |
| 最新完整v2副本与报告 | `results/returned-v2-2026-11-03-002852/`、[V2完整结果](../reports/v2-complete-2026-11-03/RESULTS_ZH.md) |
| v2最终独立复算、逐单元CSV | `.validation/v2-complete-2026-11-03-002852/` |
| 新一轮V3协议与启动 | [V3_PROTOCOL_ZH.md](V3_PROTOCOL_ZH.md)、[RUN_V3_ZH.md](RUN_V3_ZH.md) |
| V3准备数据与交付 | `results/solo-v3-prepared/`、`results/solo-v3/`、[V3交付ZIP](../../decision-study-v3-ready.zip) |
| 当前v2执行协议与服务器续跑 | [V2_PROTOCOL_ZH.md](V2_PROTOCOL_ZH.md)、[CONTINUE_V2_ZH.md](CONTINUE_V2_ZH.md) |
| CPU网页演示及便携证据包 | [LAB.md](LAB.md)、`reports/decision-lab-2026-11-02/` |
| 早期扩展备选（当前范围以v2协议为准） | [EXTENSION_ROADMAP_ZH.md](EXTENSION_ROADMAP_ZH.md) |
| Claude方案复核与当前收敛建议 | [CLAUDE_V2_REVIEW_ZH.md](CLAUDE_V2_REVIEW_ZH.md) |
| 三模型正式结果报告与复算记录 | `reports/formal-2026-11-02/`、`.validation/formal-experiment-2026-11-02/` |
| 已验证的四例CPU历史结果回放 | [DEMO_REPLAY_ZH.md](../reports/formal-2026-11-02/DEMO_REPLAY_ZH.md) |
| 最新 pilot 副本（三个模型验收通过） | `results/returned-pilot-2026-11-02-071104/` |
| 上一包（工具链已匹配，缺cuRAND） | `results/returned-pilot-2026-11-01-235641/` |
| 上一包（CUDA compiler/header 版本冲突） | `results/returned-pilot-2026-11-01-232833/` |
| 上一包（CUDA 编译器路径错误） | `results/returned-pilot-2026-11-01-225354/` |
| 上一包（旧时限阻止启动） | `results/returned-pilot-2026-11-01-224029/` |
| StartLux 通过、Qwen Ninja 阻塞的上一包 | `results/returned-pilot-2026-11-01-170718/` |
| 历史 StartLux 缺依赖包 | `results/returned-startlux-pilot-2026-11-01/` |
| Kev 复算脚本、指标、图和错例 | `reports/kev-2026-11-01/` |

上述相对路径相对于本地源码主目录。Desktop 同名回传 ZIP 可能被下一次下载覆盖，
因此证据以 `records/bundles/` 的固定名称和索引哈希为准。
源码与证据已从另一台电脑复制到当前 Desktop 树；旧的
`/Users/zhanj289/Documents/ChatGPT/COMS6998/decision-study/` 路径不再作为恢复入口。
`results/returned-startlux-repair-2026-11-01-172548/` 是空目录，不代表修复已运行。
早期 `.validation/` 还包含软件测试夹具与历史验收；不要将人工预测当作实验结果。
本地文件不是远端实时状态；没有新回传，就不能推断远端已完成后续工作。

### 本机迁移核验（2026-11-01）

- 已恢复 [项目入口指令](../../../AGENTS.md)，修正导航和过期的 README/Quickstart 状态。
- 三个归档 ZIP 及其 39 / 295 / 310 个清单文件哈希均通过复核；Kev 报告的六个
  SHA256 条目匹配。现有 StartLux 补丁与本机四个对应文件逐字节一致，无需重打包。
- 当前 Python 3.13.3 下，99 个现有测试全部验证通过，包括 9 个 StartLux 修复测试。
  首轮两项 socket 测试被沙箱限制，获批准后只重跑这两项，连同 2 个子测试通过。
  `bash -n scripts/recover_startlux.sh` 通过；记录见 `.validation/migration-2026-11-01/`。
- 冻结核心源码与原始结果保持不变。服务器仍由用户操作；在迁移接手当时，用户确认
  修复尚未执行，随后已报告运行该脚本遭遇 Slurm OOM（见最新反馈）。

### 本机结果占位文件修复（2026-11-01）

- 此前解压副本核验停顿的文件具有 `SF_DATALESS` 标记，实际数据未在本机落地。
  例如 378 字节的 `runtime/download-startlux.json` 读取超时，metadata 显示
  `compressed,dataless`、`st_blocks=0`；这不是 OOM 证据。
- 三个已归档 ZIP 均可本地读取并通过外层哈希。仅将映射明确的 457 个占位文件
  从这些 ZIP 以 1 MiB 块流式恢复，逐文件 SHA256 校验后原子替换，共 35,547,027 字节。
  未改 SF_DATALESS 标志、未升级依赖、未修改冻结源码或原始 ZIP。
- 恢复后全部 647 个解压文件（644 个清单条目加 3 个 manifest）校验通过，弥补
  上次只完成部分解压副本核验的限制。此前阻塞文件可立即读取；核心组合哈希保持不变。
  记录与可复核脚本见 `.validation/migration-2026-11-01/materialization.json`
  和同目录 `restore_result_placeholders.py`。该脚本是本机恢复记录，不是公司 GPU 补丁。
- 新崩溃已确认发生在公司机器，需要其最新日志及 Slurm RAM 分配/内存记录。
  现有 smoke 先结束 reference 子进程再启动单模型服务，StartLux/Qwen 顺序执行；
  `gpu_memory_utilization` 仅作用于 vLLM，不能处理 reference 路径的内存问题。

## v1冻结实验范围（保持不变）

单人可恢复实验流水线，三个冻结模型：Kev-4B、StartLux-Decision-v1.1-4B-Instant、
Qwen3.5-4B；CLINC 16 类（15 个 banking intents + oos）与 SQuAD 给定段落可答性。
只拟合任务标量温度，不微调模型权重。全部正式预测15,950条已核实完成。

| 模型 | 预期正式预测 | 已核实完成 |
| --- | ---: | ---: |
| Kev | 6,250 | 6,250 |
| StartLux | 6,250 | 6,250 |
| Qwen | 3,450 | 3,450 |

固定 revision、原生温度与启动参数见 `configs/runtime_defaults.json` 和回传包
`runtime/lock.json`；统计规则见 `docs/metrics.md`。不得据已看到的测试结果更改
提示、标签描述、数据抽样、温度或 gate。验证集经验错误率 ≤5% 不是未来风险保证。

## 已核实的完整结果（2026-11-02）

完整包SHA256：`be748917f2f2184e3a7d3468b4fa222e7d1b77c0abbba863fce64f395c1840e7`。
原包位于`records/bundles/formal-experiment-20261102-181054-be748917f2f2.zip`；
768个清单文件、769个安全唯一member和所有解压副本通过哈希验证。

| 测试集 | Kev | StartLux | Qwen |
| --- | ---: | ---: | ---: |
| CLINC主测试（600） | 77.50% | 93.50% | 88.00% |
| CLINC OOS压力工作负载（1,300） | 89.62% | 96.62% | 94.08% |
| SQuAD可答性（2,000） | 81.35% | 91.25% | 86.00% |

- StartLux相对Qwen的主测试优势为+5.50pp（CLINC，95%配对区间3.667～7.333pp）
  和+5.25pp（SQuAD，3.591～6.822pp）。SQuAD按35篇文章聚类；均为固定配置条件区间。
- StartLux refitted gate接受错误率CLINC38/599=6.34%、SQuAD118/1875=6.29%，
  均超过5%经验目标；其本地覆盖率99.83%、93.75%。覆盖率不是端到端加速。
- Kev→Qwen refitted级联CLINC85.67%、SQuAD85.85%；后者本地覆盖35.40%，
  纠正184例同时损害94例，净增4.50pp，但未显示优于Qwen直接推理（86.00%）。
- StartLux→Qwen在SQuAD纠正29例、损害24例，净增0.25pp且区间跨零；CLINC准确率不变。
- 温度重拟合没有改善StartLux路由决策；SQuAD两模型接受集合均不变。
  Kev的SQuAD ECE下降，但不能据此声称风险控制或级联准确率改善。
- 压力工作负载复用450条主测试banking样本，加850条独立OOS，因此不是新增1,300条
  预测；主测试+压力工作负载合计1,450个CLINC唯一ID。OOS占比与OOS池均改变，
  总准确率上升不说明banking能力改善，也不是纯prevalence的因果实验。
- 51项统计复算全部一致，包括2,000次bootstrap、100次匹配转交配额随机对照、
  oracle上界及60条确定性错例。未调参、未改冻结模型或原始结果。

正式结果解释与证据链接见[RESULTS_ZH.md](../reports/formal-2026-11-02/RESULTS_ZH.md)。
原始9组PDF/PNG保存在正式解压副本`analysis/figures/`；后续性能结果见下一节。

四例[CPU结果回放](../reports/formal-2026-11-02/DEMO_REPLAY_ZH.md)已用现有CLI实际运行，
覆盖本地正确、Qwen救回、转交损害和高置信度误接受。回放使用独立输入副本，原始769个
文件执行前后哈希一致。样例按已知结果类别挑选，只作说明；不运行GPU，不作性能测量。
现有notebook的live路径尚未自动接入v3工具环境，不能仅切换RUN_LIVE就认为实时demo已验收。

## 已核实的真实性能（2026-11-02）

原包`records/bundles/benchmark-complete-20261102-210735-72138989815c.zip`，21,207,174字节，
SHA256 `72138989815c042223cf990517bb32306aec692fe3278055e40d88571fd1f4e8`。
921清单文件、922安全唯一member及全部解压哈希通过；正式结果、校准、分析和session保持不变。

五种部署×两个任务×并发1/8×三次重复共60个单元，每个任务固定200个唯一输入。
逐单元30.14～56.12秒，共60,200次输入执行、63,720次底层HTTP调用，零失败、零重试。
实际测量区间约12:24～13:06 PST；计时单元合计35.41分钟，不含服务启动、预热等全部开销。

以下为三次重复吞吐的算术均值，单位成功请求/秒，**不是全测试集的准确率结果**。

| 部署 | CLINC c1 | CLINC c8 | SQuAD c1 | SQuAD c8 |
| --- | ---: | ---: | ---: | ---: |
| Kev | 29.19 | 56.75 | 27.74 | 43.63 |
| StartLux | 26.02 | 59.10 | 25.84 | 57.88 |
| Qwen | 7.60 | 32.71 | 7.13 | 30.76 |
| Kev→Qwen | 12.62 | 25.16 | 7.61 | 17.27 |
| StartLux→Qwen | 25.04 | 56.69 | 21.80 | 44.21 |

- StartLux/Qwen吞吐比：c1为3.42×/3.63×，c8为1.81×/1.88×（CLINC/SQuAD）。
  两模型的运行实现不同；这是单L40S上的冻结系统比较，不是隔离架构效应。
- StartLux加入Qwen后级后，SQuAD c8吞吐下降23.6%，三轮p95均值150.60→249.67ms。
  正式质量结果仅+0.25pp且区间跨零，当前证据不支持为此默认增加级联复杂度。
- 记录的显存采样峰值最高38,458MiB，最小可用比例16.52%；双模型驻留成功。
  这是约0.5秒轮询的全设备显存摘要，不是连续峰值保证，也不是100GB主机RAM计量。
- 全60个单元的原始trace可复算p50/p95、正确率、fallback、周期/ID和吞吐关系；
  每周期恰好同一200个输入。图表使用三轮均值/范围，不能把重复执行当60,200个独立样本。
- 请求延迟从worker开始执行到HTTP级联返回，包含实际fallback，不包含控制器队列等待。
  吞吐使用完整测量轮次墙钟时间；GPU秒/1000是单设备该时间的分摊，不是能耗或整场费用。
- Kev/SQuAD c8有一个输入出现标签变化（direct共19/4,000次调用）；cascade最终标签未变，
  但另有2次路由变化。原因未确认，不宣称完全确定性；StartLux/Qwen本次标签保持一致。
- 新engine日志确认prefix caching关闭；StartLux health中的prefix_cache=true与实际
  EngineCore/worker日志不一致，保留该元数据差异，不把health字段当有效运行配置。

独立整理的图表与报告位于`reports/benchmark-2026-11-02/`；原始benchmark图未改写。

### Kev单模型历史复核（保留）

Kev 正式包：`dc82af44824ee5ed57ef7dfafaaa46a1617b1be78890f8101bcf34451584aac6`，
295 个文件校验通过；配置 `6ab2a00e7447d3612f61c6f675ee1e6a2dc96a5cead2c003deefeaa2ab24b9ac`。
数据、逐样本 ID、分片、配置与校准绑定都通过核对；8 并发 smoke 也已实际通过。

| 测试集 | Direct accuracy | Refitted gate 覆盖率 | 接受错误率 |
| --- | ---: | ---: | ---: |
| CLINC 主测试（600） | 465/600 = 77.50% | 385/600 = 64.17% | 29/385 = 7.53% |
| CLINC OOS 压力测试（1,300） | 1165/1300 = 89.62% | 1069/1300 = 82.23% | 30/1069 = 2.81% |
| SQuAD（2,000） | 1627/2000 = 81.35% | 708/2000 = 35.40% | 31/708 = 4.38% |

- CLINC 原始概率 gate 接受错误率 6.67%，refitted 为 7.53%，均未达到主测试 5% 目标。
- 450 条 banking 请求中 117 条误判 OOS；压力测试含更多容易识别的 OOS，不能把
  总准确率上升解释为 banking 能力改善。
- SQuAD 温度校准后 ECE 4.17%→2.14%，接受集合完全不变；接受错误率的 95%
  按文章聚类 bootstrap 区间约 2.57%–6.67%，不能保证错误率低于 5%。
- 正式预测阶段记录约 270.12 秒，只是该次运行记录，**不是预定 live benchmark**。
- 在该历史Kev包中尚无Qwen正式结果；现已由上方完整包补齐，真实cascade可计算。
  该历史包当时不含性能；当前完整benchmark见上方。

可复算来源：`reports/kev-2026-11-01/analyze.py`、`metrics.json`、`summary.csv`；
置信区间使用固定模型/温度/gate 下的 2,000 次 bootstrap。图与错例同目录。

## 历史验收与正式续跑依据（已完成）

最新回传 `pilot_bundle (7).zip` SHA256 为
`2b70a8090b751d25dd14ffd326b0606cbe7f5a072d09027168f93041316f1ce8`，
创建时间 `2026-11-02T15:11:04Z`。343个manifest文件、344个安全member通过；
较(6)有10变更、6新增。Kev253个受保护文件、StartLux smoke、session未变。
审计：`.validation/migration-2026-11-01/new-pilot-20261102-151104.json`。

v3实际setup与采样均complete。独立CUDA13.0五组件的wheel/manifest/installer
身份匹配交付；真实JIT日志记录sampling构建，2x128输入得到[3,7]且CUDA同步成功。
两次Qwen服务成功启动、提供请求并受控停止；存在EngineCore强制结束与semaphore
清理警告，不称为无警告优雅退出。原last_error逐字节未变，是历史错误。

Qwen串行128条、8客户端并发128条全为ok、一次成功、合法sole-key JSON标签，
无重试。CLINC64/64、SQuAD63/64一致（唯一差异sample
squad:5acd839007355d001abf450d，串行answerable、并发unanswerable），满足原≥98%
门槛。正式推理继续串行concurrency1。Qwen无独立reference，reference_check是
串行自比较；不能从smoke推出任务准确率或完全确定性。

三个模型各10项绑定核对通过，另独立35项验收重算通过，包含全部6,378条请求
prompt指纹、样本/标签/token关联。Qwen配置hash为
`23aefc3248ddcef3d39d0b7d946c6ab1baf303594252a46f7d25cac2820a701c`，
runtime hash为`5a087f7ce9c3b5492a3f5baa61ef68dcc0c831e7bae38c9649ef787da31f98d0`。
记录：`.validation/qwen-smoke-pass-2026-11-02/independent-acceptance.json`。

随后已按[CONTINUE_FORMAL_ZH.md](CONTINUE_FORMAL_ZH.md)完成startlux_full、qwen_test、
evaluate和experiment export；之后benchmark也已完成并验收，正式预测不重跑。

当时pilot的bundle_complete/model_results_complete为pilot范围字段；当时的
experiment_ready=false随后被完整experiment包更新为true；最新benchmark也已完成，当前待课程交付整理。

以下为上一份完成 StartLux 验收并触发工具路径修复的证据：

上一回传 ZIP SHA256：`7e7b7009d25d01b2984f7d8e8414ba2ffc530ac8dc4a8644a9610c80988d51dd`，
创建时间 `2026-11-02T01:07:18Z`；331 个清单文件、332 个安全唯一 member 校验通过。
与 Kev 原正式包比较，253 个预测/配置/索引/校准文件哈希保持不变。

- StartLux 补依赖 marker 为 complete，实际 reference/serial/concurrent 各 128 条
  成功、零重试；两个任务 reference-vs-service 标签一致率均为 100%。平均/p95
  最大坐标概率差 CLINC 0.001282/0.005554，SQuAD 0.001359/0.003667。
- 8 客户端并发检查实际通过。两次服务各只有一个记录到的 GPU worker，均明确
  记录 bidirectional ON、prefix caching OFF、chunked prefill OFF。不要把 8
  客户端并发称为 8 个 GPU worker 或性能 benchmark。
- 源码、prompt、数据、runtime、tokenizer、reference 等 10 项绑定通过；审计见
  `.validation/migration-2026-11-01/new-pilot-20261102-010718.json`。
- Qwen 已下载、通过 token 检查并加载约 7.99 GiB 权重，在 sampling warmup 的
  FlashInfer JIT 抛 `FileNotFoundError: 'ninja'`。具体日志为
  `runtime/logs/qwen-engine-1793581376926487327.log`。
- vLLM freeze 已含 Ninja 1.13.2；controller 只用绝对 Python 路径启动服务，
  未将该 venv 的 bin 加入 PATH。新 launcher 从已记录的环境定位 bin、验证 Ninja，
  前置 PATH，并固定 controller 绝对解释器。冻结 core、依赖、配置均不变。
- 修复包：[decision-study-qwen-tools-fix.zip](../../decision-study-qwen-tools-fix.zip)，
  4,238 字节，SHA256 `97df76838b7a49524b90d259499f9226e693e0ed4e55dac0949202ef9556565b`。
  本地真实 venv 子进程复现 + 启动器 9 项检查通过；远端 Ninja 可执行文件和 GPU
  JIT 完整验收待运行。用户随后提供终端输出确认远端 Ninja 命令已成功，controller
  仍为原 `.venv-controller/bin/python`。记录在 `.validation/qwen-tools-2026-11-01/`。
- Qwen 日志中的 chunked-prefill 警告已核对 vLLM v0.31.0 源码，为通用条件触发，
  相同警告也出现在成功的 StartLux 日志。当前长度预算满足关闭模式校验；不据此
  误改冻结计算配置，也不把 Qwen smoke 当成独立 reference 正确性证明。

历史 Ninja 补丁安装步骤（已完成，当前按上方正式续跑入口）：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-qwen-tools-fix.zip
cd decision-study
bash scripts/run_with_vllm_tools.sh smoke
```

返回新的 `results/solo-v1/exports/pilot_bundle.zip`。此前的 Accelerate 修复不再
作为当前待办。后续涉及 Qwen 的正式/benchmark 命令使用 CUDA 新 launcher 保留工具链环境。

## 资源与恢复注意事项

- 公司 L40S 单卡，约 46,068 MiB；可新开 24 小时 code-server session，持久目录保留。
- 用户最新报告主机 RAM 已调至 100 GB，尚无实际 Slurm allocation/峰值计量；
  GPU 显存、home 磁盘配额和作业 RAM 是不同资源。
  新 session 实际创建后再更新运行窗口，RAM 调整本身不更改已完成 Kev 的配置身份。
- home 配额：100 GB soft、150 GB hard；`du` 不是平台 quota。用户最新报告约 76G，
  上次为 52G。经用户明确同意的三个旧 HF 模型缓存已清理；不继续删除其他目录。
- 最新 session 文件截止 epoch `1793659848.3456855`，即 `2026-11-02T22:50:48Z`
  （2026-11-02 14:50:48 PST）。它是记录的运行窗口，不是实时平台保证。
  只在用户实际新开 session 后 renew；不要根据日期或 compaction 自动延长。
- 公司算力用 `--cost-mode provided`，不虚构美元成本；原额外云预算上限 $200。
- 本地无 NVIDIA GPU；CPU 测试通过不等于远端模型验收通过。
- `last_error.json` 可能留有历史错误；结合 stage 状态、证据时间和对应日志判断。

## 后续顺序与不可丢失的约束

1. 正式预测、统计分析和live benchmark均已完成。保留所有冻结配置、prompt、数据、精度、
   温度与gate；不因观察到结果重跑挑选更有利数字。
2. 按[当前v2执行说明](CONTINUE_V2_ZH.md)在真实新allocation上继续新增GPU工作，保留
   v1为完整交付基础；同步准备[课程报告与演示](FINISH_PROJECT_ZH.md)。CPU网页已实现，
   常驻live UI不在当前范围。v2只有本地软件验收，远端成功必须以新回传包核验。
3. 保持两套样本口径：正式测试用于质量结论；固定200输入的benchmark用于本实现性能。
   明示计时边界、三轮描述性范围及运行身份，不能将worker计时称为包含全部排队的用户延迟。
4. `adapters.py`、`runner.py`、`runtime.py`、`storage.py`、`common.py` 被 Kev 配置哈希绑定。
   当前组合 hash：`9f25b6350da44fb22cb5f8a644e1cb68b707a82272a024c931701f84c701551f`。
   不能修改后伪造旧哈希；若必须修改，先评估受影响的正式记录与迁移方案。
5. 原始回传文件保持不变，分析写到独立 reports 目录。每次进展更新本文件并追加日志。
