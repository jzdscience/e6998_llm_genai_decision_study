# 9B 服务研究 revision 4 独立复审

核对日期：2026-11-08，America/Los_Angeles。保留节点证据中的原始 UTC 时间。

**通过本轮软件复审，可以交付新一轮节点 `--preflight-only`。未发现新的阻断问题。**
审核范围是 `fe7f9d2..16e734f`，完整被审提交为
`16e734fc03f2755eb2c5592031042370e34c4743`。真实 vLLM 在上限 64 下的启动、Qwen-9B 的 KV 容量和
0.40 + 0.50 同驻留仍未验证；本结论不放行正式测量。

## 首次节点诊断包

原始 ZIP 的 SHA256 为 `c9392e10dfad32fba5ba62ee513bdc4d0e93538cc44a2868b322f413595a6183`，
7,436,661 字节，74 个成员。源码绑定 `fe7f9d26da959f51165213baf3dfa0cf860f7f7f`、协议 revision 3。
独立审计 112 项检查通过，包括 CRC、安全唯一成员、35 份源码 SHA、17 份准备文件、原始 tokenizer 报告、
嵌套收据及预检原始记录。此处 PASS 表示证据一致，节点预检仍是失败。

| 项目 | 原始证据确认 |
|---|---|
| Token 检查 | 三个角色各覆盖同一批 3,200 条正式输入和 90 条开发输入，ID、文本、顺序与保留输出预算一致 |
| StartLux-4B | 串行 90/90；并发 310/310 成功且均在 500 ms 内；并发 p95 82.97 ms，最大 104.45 ms |
| 4B 资源与停止 | KV 7.6 GiB / 203,075 tokens；启动 269.09 秒；停止核验 0.1273 秒，无残留 |
| StartLux-9B | 权重 16.8 GiB，graph profile 0.39 GiB，KV 2.45 GiB / 65,104 tokens；引擎报 `max_num_seqs (256) exceeds available Mamba cache blocks (151)` |
| 9B 失败清理 | 本次失败进程组 102824 已登记；停止核验 0.1206 秒；检查时 GPU compute processes 为空 |
| 研究进度 | 级联未运行；没有正式单元、freeze 或评价结果 |

诊断时间为 `2026-11-09T06:25:35.387456+00:00`。原 ZIP 已按 SHA 复制到外部证据根的
`records/bundles/upgrade-preflight-r3-20261109-c9392e10dfad.zip`，复制后重新核对字节数与 SHA；
原文件保持不变。[BUNDLE_INDEX](../records/BUNDLE_INDEX.json) 从 21 项增至 22 项，原 21 项内容保持。
审计脚本与 112 项收据在本机 `.validation/revision4-review-2026-11-09/{bundle_audit.py,bundle-audit.json}`。

沿用已有披露：StartLux health 的 `prefix_cache=true` 与实际启动参数、effective config 的 `false` 冲突，
不能据 health 宣称启用了缓存。清理记录证明本次检查无残留，不表示真实节点已覆盖残留进程的 SIGKILL 分支。

## 启动器和参数绑定

逐步对照 `upgrade_study/launcher.py` 与 V1 `runtime.managed_services` 的单模型路径，doctor、安装锁、
环境核验、端口检查、独立进程组、逐进程 `last_services.json`、readiness、StartLux attention/thinking
检查和 finally 停止顺序保持。新入口只支持单个 StartLux 或 Qwen，V1 的双模型检查由本研究既有控制器承担。
命令仍来自 V1 `_service_specs`；前端命令不变，只有唯一的 vLLM engine 命令追加 `--max-num-seqs 64`。
启动失败的外层所有权登记和残留成员清理继续生效。

调用 V1 私有辅助函数在当前固定源码身份下可接受；未来更改这些函数时仍须做同样的命令与生命周期对照。
本轮没有修改 V1、V3、规模对照或既有服务研究源码。新 launcher 已纳入 `source_identity()`，SHA256 为
`3c9f971785490c05e014ec4f0eb586962fd64304e590e06f2d1dacc35827638e`。

协议固定三个角色的引擎上限，settings 绑定协议与源码，`engines_limited()` 在预检复用及正式测量核验时检查
每个引擎的真实命令。V1 的 runtime_config/token effective hash 本身没有此字段，但新增绑定与命令检查覆盖
这一差异，不构成静默复用旧配置。输出改为 `results/aux-anli-upgrade-service-v1-r2`，三个条件全部重做预检。

上限 64 针对日志中的具体约束。vLLM 0.31.0 的该检查涉及 full CUDA graphs 下每个 decode sequence 的
Mamba 状态块需求；`max_num_seqs` 也影响每次迭代的序列数及相关缓冲区，不能把它等同于服务队列上限。
参见 [vLLM 0.31.0 config 源码文档](https://docs.vllm.ai/en/v0.31.0/api/vllm/config/) 与
[serve 参数文档](https://docs.vllm.ai/en/v0.31.0/cli/serve/)。

客户端 worker 数为 16，`max_inflight` 为 64；这些是不同层的限制。64 小于本次观察到的 151，足以支持把它
作为新的候选配置，但不能据此保证实际启动成功、KV 充足或吞吐不受影响。HTTP 超时后后端可能仍有工作，
不能仅凭 16 个 worker 推断引擎内部永远只有 16 条序列。三个条件统一上限有助于比较，正式性能属于该配置。

## 独立验证

- 独立全仓命令 `.venv-lab/bin/python -m pytest -q -p no:cacheprovider`，exit 0；
  **417 passed、11 skipped、64 subtests passed，106.68 秒**，包含全部 52 项 upgrade 测试。
  执行允许本机 localhost/进程列表访问；未调用 GPU 或真实模型。
- 真实证据初始化的三个角色 controls，其命令与 V1 逐项一致，仅引擎追加上限；份额分别为 0.4/0.5/0.5。
  正常退出与 frontend readiness 失败路径都核对了引擎登记和清理。
- 协议固定值、默认 launcher 接参、每个引擎的上限、完成单元记录及演练证据篡改均受测试覆盖。
- 核查 8 份 mutation 完整日志、隔离副本的 7 项测试基线及恢复后的六份文件身份；本次未独立重跑全部 mutants。
  n1/n2/n4–n8 是直接守卫或断言捕获；n3 回退到 V1 后由其 GPU doctor 报错，是间接捕获。
  正常测试另有新 wrapper 接收 64 的直接断言，因此不构成阻断。可选地给旧 launcher 加“禁止调用”哨兵，
  让 n3 的失败归因更直接；不要求为此推迟节点预检。

测试日志和独立收据在 `.validation/revision4-review-2026-11-09/`。11 项 skip 仍是 skip，本轮结果不等于
全新依赖安装或真实硬件验收。

## 下一步

1. Claude 可将验收笔记一并提交，按既有流程合并、普通 push，交付最终完整 SHA。
2. 节点 checkout 更新到该 SHA 后，按 [运行文档](RUN_UPGRADE_STUDY_ZH.md) 执行 `--preflight-only`。
   保留首次输出；不要在 revision 3 输出目录用重试来替代本次配置变更。
3. 回传新 `results/aux-anli-upgrade-service-v1-r2/exports/upgrade_study_bundle.zip`。重点核对三个条件的
   cap=64 命令、两个 9B 引擎的实际 KV、强制级联的重叠与响应、停止收据。通过后再开始 36 个正式单元。

本次仅新增审阅记录、更新状态/索引、归档诊断证据；未修改运行源码或协议，未 commit/merge/push。
Claudeception 已评估：硬件修复尚未实测，诊断与复审方法已有项目技能覆盖，本轮不将候选配置另存为已验证修复技能。
