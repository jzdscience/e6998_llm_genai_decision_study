# 9B 服务成本实现交叉验收（11/8）

审阅范围：`9cb3274..65c57a9`，含 `192da5c`、`b4f0282`、`f254bb6`、`65c57a9`。
工作分支 `review-fixes-2026-11-08`。下列源码行号指 `65c57a9`。本次只审查本地实现和软件行为；没有模型/GPU 调用，
没有合并、推送或修改仓库可见性。GitHub 当前状态不属于本次独立验证的范围。

## 结论

**设计与主体实现可以保留；本次尚不通过节点预检前的软件验收。** 四个故障边界已独立复现，建议 Claude 在同一分支
修复并补对应测试，再交叉确认。它们不要求扩大实验矩阵、重写论文或重跑已验收的历史实验。

主测试独立复跑 exit 0：**395 passed、11 skipped、64 subtests passed**，其中包含全部 30 项 upgrade-study 测试。
通过现有测试与下列缺口同时成立；这些是已有测试未覆盖的执行路径。

## I1 · P2：后级启动到一半失败时，自有 worker 可能不在清理集合中

位置：`decision_study/upgrade_study/runtime.py:303–306`、`:323–329`。

`owned` 只在 `stack.enter_context(launch(...))` 成功之后加入该角色的进程组。如果 Qwen 已启动进程，但 readiness
失败，第二个角色不会进入 `owned`。底层冻结 launcher 会向组发 SIGTERM，然后等待组长；组长退出不等于所有后代退出。

独立探针使用真实 `frozen.managed_services` 的清理代码，仅替换硬件检查和服务命令。后级组长启动一个忽略 SIGTERM 的
worker 后触发 readiness 错误。组长退出，worker 仍存活；外层只知道前级组，虽然看到剩余进程，却无法清理未登记的后级组。
探针返回后由自身 finalizer 清理了剩余组，没有留下测试进程。这是本机进程故障模拟，不是已观察到节点 GPU 泄漏。

修复要求：

- 在启动过程中登记本次实际创建的进程组，包括 `__enter__` 失败路径；外层验证并终止所有自有组的残留后代。
- 从 `last_services.json` 补获身份时，必须区分本次新记录与上次遗留记录，不能仅凭旧 PID 向进程发信号。
- 保持原 V1 源码身份，在新控制器内补齐此约定。
- 新测试让失败的 launcher **先创建真实子进程再失败**，并保留一个忽略 SIGTERM 的后代，检查它最终被清理、外部进程不受影响。
  当前 `test_fallback_start_failure_stops_the_front` 在创建后级进程前就抛异常，覆盖不到此路径。

## I2 · P2：缓存 PASS 没有核验它所引用的预检原始证据

位置：`decision_study/upgrade_study/study.py:154–159`、`:239–249`。

`preflight()` 复用已有 PASS，`preflight_hashes()` 也只检查顶层 `status/settings_hash` 并计算收据自己的 SHA。
二者均没有核验收据已有的 `evidence` 文件哈希、`evidence_dir` 和 `token_receipt_sha256`。

独立探针复制已保存的 CLI 假服务演练产物，仅删除级联预检的 `traces.jsonl`，没有改收据或重新封存任何哈希。
程序仍报告三个条件 PASS，`preflight_hashes()` 与删除前相同，三个正式单元的 `completed()` 也仍通过。
这不证明已有指标算错，但说明预检证据缺失后仍可以继续正式测量和完成验收。

修复要求：

- 提供一个统一的预检验证函数，在缓存复用、正式开始、完成/冻结验证时调用。
- 验证完整收据 schema、条件/settings 身份、预检原始文件哈希、实际调度/汇总，以及 token 收据和原始 token 报告。
- 原始 token 明细实际保存在 `runtime-controls/.../tokens-interactive-*.json`，并未丢失；需要将其路径、SHA、
  完整 3,290 条输入覆盖和有效配置纳入绑定，不能只检查摘要中的 `status=ok`。
- 新测试删除或改变底层文件，保持顶层 PASS 原字节，分别验证缓存预检、正式启动和完成验收都拒绝。

## I3 · P2：子进程在 watchdog 截止后退出仍可能被接受

位置：`decision_study/upgrade_study/study.py:328–337`。

时限只在 `while process.poll() is None` 循环内检查。进程在截止后、下一次轮询前成功退出时，循环直接结束，代码接受 exit 0。

独立真实子进程探针使用 0.2 秒 duration + 0.2 秒 drain + 0.2 秒 grace，启动 `sleep 0.61`，没有模拟时钟。
`_run_process` 在约 **0.763 秒后返回成功**，超过 0.6 秒总预算。该探针证明晚退出被接受，没有伪造完整科学单元。
如果此路径已有 completion receipt，当前代码也不会把它移为 unaccepted。

修复要求：循环结束后再次检查单调时钟和 allocation 时限，超过边界走已有中断处理，保留但不接受 completion。
旧 `service_study/study.py:165–166` 已有同类 post-loop 检查，可沿用其语义。
补“截止后成功退出并写出 receipt”的测试；当前挂起子进程测试只覆盖始终未退出的分支。

## I4 · P2：请求预算不是完整调用期限，串行预检/预热仍可越过它

位置：`decision_study/upgrade_study/client.py:57–60`、`:84–86`、`:103–105`；
`decision_study/upgrade_study/study.py:172–174`、`:375–379`。

共同 deadline 仅在决定调用后级前检查。直接调用、不转交的前级以及后级返回后，都没有确认响应是否已超过预算。
传给旧 `ModelClient` 的 socket timeout 也不是完整响应下载的绝对期限。

独立 localhost HTTP 探针每 45 ms 返回一块合法 JSON。给予 **80 ms** 预算，真实 `ModelClient` 和 `call` 在约
**310 ms（直接调用）/290 ms（不转交级联）** 后仍返回 `status="ok"`；未模拟客户端或时钟。
正式单元的父 watchdog 可限制整个单元，但不能据此声称每条请求都遵守其 30 秒预算。

`65c57a9` 正确地把并发预检放进了子进程；90 条串行开发检查及每单元串行预热仍直接运行在 controller 内。
因此持续缓慢输出的响应可以在这些阶段不断延后 socket 超时，缺少同样的有界阶段保护。

修复要求：

- 每个成功返回路径都检查共同 deadline；超期结果保存 stage 证据并记 `deadline`，不能记成功或再触发后级。
- 为串行预检和 warmup 增加明确有界的进程生命周期，或实现真正的 HTTP 总期限；仍遵守实际 allocation。
  仅对 Future 等待设置 timeout 不等于停止其运行线程。
- 在新模块实现，保持已冻结 adapter 字节不变；补真实慢分块 HTTP 测试，覆盖直接、保留前级和后级三个成功路径，
  以及串行阶段超时后的进程清理。

## 已通过的部分与非阻断说明

- U1–U5 的协议修订基本落实，0.50 仍标为候选配置；原始政策、输入身份和 766/743 历史重放正确。
- 三种部署、36 单元、共同到达表、配对条件相邻运行、两个 SLO 和回顾性解释保持。
- 真实返回标签用于 correct goodput，历史比较按成功 ID 和次数加权；全 offers 分母及跨 allocation 标记正确。
- 单元 trace/计划/资源哈希与汇总复算、完整单元冻结后读 gold、中断评分不静默重试均有实现。
- 三角色独立控制目录/端口、实际 allocation 校验、同 job 不延长期限、成功进入后的 ExitStack 清理均已覆盖。
- 原 V1 adapter/runtime、V3、规模对照和既有服务模块没有改动。无需修改旧 freeze 或放宽旧发布验证。

两个非阻断澄清：

1. HTTP 返回无效 schema 时，真实 `ModelClient` 将其记为 `invalid_response`，当前实现按前级调用失败转交；
   假客户端的 `router_input` 测试则注入 `status=ok` 的坏概率。这两个分支应在协议中明确并补一个真实 HTTP 测试。
   原 V3 也转交终态 error，原始错误仍保留，因此本次不单独判为路由缺陷。
2. `services/.../finished.json` 的清理确认尚未纳入 `completed()` 绑定。后续部署仍检查 GPU 空闲，不能据此宣称
   指标无效；修 I1 时宜一并明确“测量完成”和“清理确认”的状态，防止最后单元清理失败后恢复时直接声称完整成功。

## 独立验证记录与后续

命令 `.venv-lab/bin/python -m pytest -q -p no:cacheprovider` 使用本机已有环境、允许 localhost 测试端口；
98.68 秒完成。日志及探针保存在忽略目录 `.validation/upgrade-review-2026-11-08/`：

- `pytest-independent.log`、`watchdog-late-exit.txt`
- `upgrade_partial_start_repro.py`、`upgrade_runtime_review_repro.py`、`runtime-findings.json`
- `probe_preflight_integrity.py`、`preflight-integrity-result.json`

Claude 的 mutation 脚本及日志包含 20 个 CAUGHT。此次未独立重跑：它直接改 tracked 源码，并以测试非零退出判为捕获；
保留日志没有逐项 traceback，不能据此独立确认每一项都是预期断言失败。修复后的新增回归测试应保存可审查失败证据，
mutation 如需复跑使用隔离副本，避免干扰工作区。

建议先修 I1–I4 并交叉验收，再形成可推送的交付 SHA。之后按运行文档先做 `--preflight-only` 并回传，
核实真实 0.50 单模型及 0.40 + 0.50 同驻留后再开始正式测量。此次没有验证全新依赖安装，也没有任何节点性能结论。
