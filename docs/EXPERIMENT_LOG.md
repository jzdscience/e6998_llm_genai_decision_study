# 项目日志

此日志从源码仓库接管开始。此前完整恢复、发布和回传验收记录保留于外部旧项目树的
`docs/EXPERIMENT_LOG.md`，访问位置见 [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)。历史文件不因
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
