# 9B 服务研究 I1–I4 修复验收

2026-11-08。审阅提交 `3fc84e2f06a443a5b58c83f4a86545d91f23910d`，对比 `65c57a9`。
对应上一轮 [实现交叉验收](UPGRADE_IMPLEMENTATION_REVIEW_2026-11-08_ZH.md)。

**软件复审通过，I1–I4 关闭，可以进入节点开发预检。** 本轮未发现新的阻断问题；这不代表 GPU 可行性或正式测量已经通过。

## 独立验证

命令 `.venv-lab/bin/python -m pytest -q -p no:cacheprovider` 返回 0：
**411 passed、11 skipped、64 subtests passed，109.76 秒**，包含全部 46 项 upgrade-study 测试。
初次执行被沙箱限制 localhost bind 与 `ps`；开放相应本机测试权限后，同一命令通过，不是代码回归。
使用已有依赖环境，没有运行真实模型或 GPU。

| 项目 | 复验结果 |
|---|---|
| I1 部分启动清理 | 用上一轮独立探针及真实冻结 launcher 清理代码重放：后级 readiness 失败后其进程组已登记，忽略 TERM 的 worker 最终被 KILL 并确认消失，前后级记录保留。旧记录和更早成员不被新控制器误当成本次进程。 |
| I2 预检证据链 | 复制 revision 3 假服务演练产物，仅删除级联并发 trace 或 Qwen 原始 token 报告；缓存预检、预检哈希、单元完成及 freeze 四个入口均拒绝。恢复后通过。 |
| I3 watchdog 晚退出 | 使用剩余时间限制 wait，退出后再次核对时钟和 allocation；真实晚退出 receipt 测试与 post-loop 回归测试通过。 |
| I4 响应期限及串行阶段 | 原 80 ms 预算慢分块 HTTP 探针中，约 300 ms 的直接响应和约 295 ms 的保留前级响应均改记 deadline，stage 保留。三种成功路径及路由跨期限测试通过，串行预检/warmup 已在有界子进程内执行。 |

停止确认已成为完成条件。独立删除 `finished.json` 后，`measured()` 仍保留测量事实，`completed()` 返回未完成，
freeze 拒绝。revision 3 预先声明显式恢复时将未确认停止的测量保留为 unaccepted 后重测；此规则没有用质量结果挑选尝试。
后续报告应保留所有尝试及其排除原因。

原 V1 adapter/runtime、V3、规模对照和既有服务模块的源码保持不变。回顾性政策选择、实际返回标签计分、共同到达表和
完整单元冻结后读 gold 的边界保持。此次没有重新运行历史模型实验。

## Mutation 与平台边界

核对了隔离副本基线及全部 36 份 mutation 完整日志；对应目标测试或 fixture 的失败，没有发现语法、导入、权限等无关失败
造成的假捕获。隔离副本恢复后的六份关键文件与当前提交一致。本次没有独立重跑全部 mutation，不把 36 个捕获解释为穷尽所有边界。

Linux procps 上游默认 `lstart` 格式是 `%a %b %e %H:%M:%S %Y`，与代码压缩空格后用 `%d` 解析兼容；
独立格式 fixture 验证了单/双位日期、排除 zombie 和无法解析时标记未核验。来源为
[procps 上游实现](https://raw.githubusercontent.com/procps-ng/procps/master/src/ps/output.c)。
本机唯一已有 Linux 镜像不含 `ps`，没有为此安装或下载，因此实际公司节点的 procps 行为仍由预检验证。

两点表述需保留准确范围：

- `unverified group members` 不一定表示日期解析失败，也可能是成员启动时间早于本次部署。出现时应检查原始 `ps` 输出和
  部署记录，不据此直接改 parser 或放宽所有权检查。
- 30 秒预算保证超期响应不被记为成功，不表示底层 socket 在第 30 秒立即取消；整个阶段另有父 watchdog 的绝对期限。
  串行阶段 2,705 秒是故障上限，实际 allocation 检查仍可能更早中止它。

## 交付与下一步

可以让 Claude 将本次验收笔记一并提交，再按正常 Git 流程合并进 main 并普通 push。交付时给出最终完整 SHA；
如果仅增加验收文档，新的交付 SHA 与这里记录的被测实现 SHA不同不代表实现变更。若再修改运行源码、协议或测试，应单独核对变更。

节点按 [更新说明](NODE_GIT_ZH.md) 固定到交付 SHA 后，执行：

```bash
cd /home/zhanj289/e6998/decision-study-src
bash scripts/run_upgrade_study.sh /home/zhanj289/e6998/decision-study --preflight-only
```

回传 `results/aux-anli-upgrade-service-v1/exports/upgrade_study_bundle.zip`，核对三个条件的真实启动、token/运行配置、
KV、重叠调用与停止证据。预检验收后再开始 36 个正式单元。0.50 单模型、0.40 + 0.50 同驻留、真实延迟及全新依赖安装仍未验证。

独立证据位于忽略目录 `.validation/upgrade-fixes-review-2026-11-08/`（全仓日志、mutation 日志核对、证据删除探针和
ps 格式 fixture），运行时独立重放收据位于 `.validation/upgrade-fix-review-2026-11-08/runtime-independent.json`。
本次只新增验收文档、更新状态/日志，未合并或推送、未改运行源码和协议。
