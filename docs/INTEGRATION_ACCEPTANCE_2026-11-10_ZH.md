# 9B 结果整合：Codex 独立验收

2026-11-10，America/Los_Angeles。结论：本次报告、论文、复现与演示整合通过验收。

实现由 Claude Code 的 `claude-opus-5-5` 完成，三次调用均显式配置 `--effort max` 和
`CLAUDE_CODE_EFFORT_LEVEL=max`，使用同一个 session。Codex 提出修订、独立检查证据与源码、执行最终测试和
浏览器检查。工作位于 `preflight-r2-results-2026-11-09` 的未提交工作区，基准 HEAD 为 `23033c1`；没有 commit
或 push，没有启动新模型或 GPU 实验。节点实验源码仍为 `7c81947`。

## 已完成的交付

- [9B 中文报告](../reports/upgrade-9b-serving-2026-11-09/RESULTS_ZH.md) 包含 36 单元与汇总表、质量和资源分析、
  500/1,000 ms 图、便携审计脚本及 6 份与原验收版本逐字节一致的收据。
- [论文 PDF](../paper/main.pdf) 为 12 页；新增研究进入摘要、方法、RQ3、Figure 5、讨论、限制与来源附录。
  原来的主要比较与回顾性分析边界保留，资源份额和不同 allocation 的范围已说明。
- Decision Lab 的独立 Follow-ups 页覆盖 9B 服务、CLINC CPU/GPU 服务、轻量基线和四个保存案例。
  旧 V1–V3 便携包保持原字节，V3 本身没有同驻留性能测量的说明保留。
- [入门指南](START_HERE_ZH.md)、[Lab 说明](LAB_CURRENT_ZH.md)、[演示讲稿](PRESENTATION_ZH.md) 和各入口已同步。
  三人的职责按 Ju 授权分配，论文删去算力致谢待办；职责分配不宣称队友已完成尚未发生的复核或排练。

## Codex 最终执行结果

最终 strict 验收分两次执行同一脚本的全部七个阶段。前六阶段与 Claude 的最后一处论文短语修正并行；
第七阶段在修正完成后执行，两次均 exit 0：

```bash
bash scripts/reproduce_cpu.sh --strict --only tests,lab,audits-1106,audits-9b,report-9b,figures
bash scripts/reproduce_cpu.sh --strict --only paper
```

| 检查 | 结果 |
|---|---|
| 主环境 CPU 测试 | 470 passed、11 skipped、64 subtests passed；112.05 秒 |
| 独立 CPU 基线环境 | 65 passed；7.59 秒 |
| 原 11/6 审计 | 原 12 项收据比较通过，raw recheck 为 PASS |
| 9B 独立审计 | 2,097 项完整性检查、原始指标重放 PASS，6 份输出逐字节一致 |
| 便携包、表格、图 | 新 Lab 双文件、3 张 CSV、报告 PNG/SVG、4 张论文图逐字节一致 |
| 论文 | 临时目录编译通过；最终跟踪 PDF 12 页，修改后的图页与职责文本已检查 |
| 真实浏览器 | 全部 8 种 ANLI 速率/SLO 组合、3 档 CLINC 负载、4 个案例与旧标签页通过；零页面错误 |
| 离线依赖边界 | 浏览器阻断所有非 localhost 请求，全部交互仍通过，实际未发起外部请求 |
| 冻结身份 | 349 个基线跟踪文件中，17 个属于本次允许修改范围，其他 332 个逐字节不变 |
| 先前审阅材料 | 原复审文档、BUNDLE_INDEX 和 9B 验收收据与整合前副本逐字节一致 |

`--strict` 表示七个复现阶段都执行并通过，不表示 pytest 的可选测试没有 skip。上述比较的图形环境为锁定的
Matplotlib 3.11.2；PDF 内含构建时间，因此验证编译与内容，不要求整个论文 PDF 每次构建的字节相同。

Codex 另核对了 Claude 保留的 13 个 mutation 完整日志，没有导入、语法或 GPU 环境造成的误判。M2 删除单元
算术检查后，后续汇总仍拒绝错误数据，测试通过错误信息差异发现回归；M8/M8b 的直接 strict 探针验证原版 exit 1、
变体 exit 0。缺新正式 ZIP 和篡改收据的隔离负向检查也符合预期。原证据没有被改动。

## 已关闭的复审问题

1. 新图横轴改为四个已测速率分组，组内错开只用于辨识模型，图注说明这一点；旧三张论文图保持原字节。
2. 正确 goodput 明确按 120 秒到达窗口的请求计数，每条请求使用自己的 SLO，分母为 120 秒。
3. 论文区分高负载的正确 goodput 与每轮至少 99% 及时规则，保留 4B 20/s 异常；3.6 秒是计划到达跨度。
4. CLINC 的两次 allocation 与新 9B 的另一次 allocation 分开表述；引言也不再暗示每项研究只有一次 allocation。
5. 三个历史案例的分母为 3,200 题，第四个为 5,746 次成功请求。三个子集/总体计数分别为 95/245、89/164、
   108/157，迟到且正确请求为 4,282/5,746；Codex 直接对照原 ZIP 与 gold 复算一致。
6. 成员职责、算力致谢、运行指南与 mutation 日志的缺项已补齐。

## 验证边界与后续交付

Claude 执行的干净源码副本包含跟踪文件和本次新增文件，导入路径确认为副本；便携检查和真实浏览器操作通过。
该副本复用本机已安装依赖，461 passed、17 skipped，多出的六项跳过需要外部原始回传。它发生在最后的文字、
图形和案例说明修订前；修订后 Codex 在最终工作区重跑了上面的七个 strict 阶段和全部浏览器组合。
本轮没有验证全新环境的联网依赖安装，也没有把这些检查称为队友复现。

科学解释仍限于已测部署和负载，ANLI 质量为回顾性、资源份额不同、重复请求不是新的独立样本；通过的速率点
不能推成连续容量边界。本次验收无需新增 GPU 实验。正式展示前仍需你们按讲稿排练、完成需要的录屏并最终提交。

结构化记录见 [integration-acceptance-2026-11-10.json](../records/integration-acceptance-2026-11-10.json)。
详细本机日志位于忽略目录 `.validation/integration-2026-11-10/`；别人 clone 后可按上述脚本复核，原始证据布局见
[ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)。
