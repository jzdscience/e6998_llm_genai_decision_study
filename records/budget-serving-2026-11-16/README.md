# 预算服务研究软件验收（2026-11-16）

协议 `anli-budget-serving-v1` revision 1 的完整软件已在本机验收，节点运行尚未发生。机器可读的计数、
跳过原因、变异证据和已测文件的 SHA-256 清单见 [software-acceptance.json](software-acceptance.json)。

## 发布

节点分支 `anli-budget-serving-2026-11-16` 以 `41dfb7547698691634fa0e5a4b8f87824436fc8f` 为基线，只加入预算服务所需的
17 个 `decision_study/budget_serving/` 模块、`configs/budget_serving_protocol.json`、`scripts/run_budget_serving.sh`、
`scripts/archive_learnability_embeddings.py`、7 个测试文件与采集回答 fixture、运行手册和协议说明，以及链接所需的
Study 4 报告与验收记录。提交 SHA 不能写进提交本身，由交接记录给出。节点命令：

```bash
bash scripts/run_budget_serving.sh /home/zhanj289/e6998/decision-study
```

## 测试

| 范围 | 结果 |
|---|---|
| 发布 checkout 全量 pytest（根 `.venv-lab`，发布目录在 `PYTHONPATH` 首位） | 828 passed，46 skipped，64 subtests，exit 0 |
| 主工作树 7 个预算/补存测试文件 | 107 passed，2 skipped |

发布全量的 46 个跳过都依赖环境或归档证据：Linux 原生进程归属与 pidfd（预算 runtime 2 个、learnability 6 个），
未安装的历史回传（direct matrix 20、upgrade study 6、V2 lab 8），可选 CPU 环境或数据缓存（3 个），以及
1 个只在存在 Study 4 解压包时运行的固定输入测试（在主工作树中运行并通过）。回环 HTTP 与整条命令模拟都已执行。
此前的 929 passed / 22 skipped 来自本轮修复前的主工作树全量运行；发布 checkout 的计数以上表为准。

整条命令模拟使用 10 个单元（5 条件 × 2 负载 × 1 轮）、40 条合成输入、约 1 秒的窗口，Slurm、进程归属自检、GPU
和 StartLux 服务均为伪造边界；控制器、`runtime.supervise`、worker 命令、接受、freeze、gold 门、审计与导出均为真实代码。
正式协议仍是 60 个单元、1,737 条输入、120 秒窗口。

## 变异证据

| 来源 | 结果 |
|---|---|
| Claude 自查（早期草稿脚本，只存摘要） | 26 个，25 个检出；剩下 1 个删去“9B 开始晚于许可”的显式检查，被已核对的尝试日志次序蕴含 |
| Claude 隔离副本（本轮新增，逐个保存完整 pytest 输出） | 7 个，全部在目标断言处检出：freeze 绑定、freeze 清单、attempt 历史、`null` 正文、JSON 声明重放、`null` 字节数、严格 JSON trace 写入 |
| Codex 独立隔离副本 | runtime 6 个、补存助手 5 个、采集回答 7 个 |

Root 报告的 Codex 独立验收：采集回答测试 14/14，协议审计回归 10/10，runtime 28 passed / 2 Linux skipped（其运行范围），
采集回答变异 7 个。审阅者的 freeze 复现脚本在修复后退出 0：原样审计通过（386 个冻结文件），清空清单并改写摘要后在
freeze 绑定处被拒绝。

## 本轮修复

- freeze 校验从磁盘重新计算全部绑定：证据清单与摘要逐项一致（只排除 session、自检指针、status 和 Finder 元数据），
  token、三个预检、全部单元收据和源码指纹与实际 SHA 相符。每个接受收据要求本阶段的 attempt 目录恰为计划记录的
  先前 attempt 加上自身，失败或中断的 attempt 不能被删除。
- 无法使用的 HTTP 200 正文（JSON `null`、非 JSON、非 UTF-8、超限、答案不合格）作为带解码 `phase` 的失败调用计入数据，
  审计按 phase 重放，有效回答不能被改记为失败；含 `NaN` 或孤立代理字符的回答不再使 trace 写入失败；HTTP 错误先于
  大小上限判定。
- 运行手册删去“超过 32 MiB 的日志不打包”的错误说法；freeze 覆盖的文件全部打包。

实验前固定的执行约定（pair 映射、每份 token 报告 1,200 秒、精确种子推导、`router_input`、驱动身份、watchdog 余量、
离线审计源码）写在[运行手册](../../docs/RUN_BUDGET_SERVING_ZH.md)和[协议说明](../../docs/BUDGET_SERVING_PROTOCOL_ZH.md)。

## 分工与边界

Claude Code Opus 5.5（max effort）实现 `budget_serving` 核心（runtime 除外）、包装脚本、运行手册、两个主测试文件和全部修复。
Codex 负责 runtime 与协议，五路独立审阅写下审阅意见和三个回归测试文件；Root 协调决策、验收和推送。
本记录收录公开的结论与计数；会话流、提示和推理过程留在 Git 之外。

节点同驻留、KV、延迟与正式 60 单元均未运行；macOS 上跳过的 Linux 原生检查在节点启动 GPU 前由原生自检执行。
