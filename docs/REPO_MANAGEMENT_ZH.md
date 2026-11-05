# 本机与 GPU 服务器的仓库管理

2026-11-04。用户已建立本源码仓库，原始导入提交为
`fa113e5f0594fa35164ce4321332d143a9242295`。本机以此目录开发；旧项目树保留历史证据。
用户已将 Lab、报告和仓库整理提交为 `4d573513f7990b8149ef22bd48be59043f980d65`，
本次通过远端查询确认 GitHub main 含该提交。后续源码交付使用同一个仓库。

node 首次 clone、固定版本、连接旧 controller 与后续更新的可执行步骤见
[NODE_GIT_ZH.md](NODE_GIT_ZH.md)。采用旁建源码目录，保留旧运行目录绝对路径。

## 各类文件放哪里

| 内容 | 管理方式 |
|---|---|
| `decision_study/`、`scripts/`、`tests/`、`configs/`、依赖声明 | Git；执行时固定版本 |
| 协议、当前状态、计划、精选报告及小型汇总表图 | Git；报告注明对应实验与证据 |
| 原始数据、prepared/sealed 数据、预测、日志、回传 ZIP、审计产物 | Git 外；保留 SHA 和位置索引 |
| 虚拟环境、模型权重、CUDA/编译缓存、机器配置 | 各机器本地；记录版本与配置 |

[ARTIFACTS_ZH.md](ARTIFACTS_ZH.md) 说明当前证据根和取得数据的步骤。`records/BUNDLE_INDEX.json`
记录历史结果包身份；查看路径前先确认其外部根语义。忽略规则不会自动取消已有文件的 Git
跟踪；新增文件应逐项审阅，不直接把运行目录整树加入 Git。

## 日常流程

本机修改并验证 → 审阅 diff 后提交 → 服务器取得固定 commit → 核验源码、数据、环境和
allocation → 执行 → 按 SHA 保存回传包 → 本机独立验收 → 更新报告和索引。

每次运行除 commit 外，还记录工作区是否有改动、发布清单 SHA、协议和数据 manifest、
模型 revision、运行环境及 allocation。Git commit 不能替代这些已有身份检查。
开发可以用短期分支；运行期间不更新所用 checkout。服务器不能访问同一 Git remote 时，
可另用 Git bundle 传递已提交代码历史；它不包含实验结果或环境。

## 旧实验的迁移边界

- 服务器现有 `results/`、环境和缓存已被 runtime 元数据的绝对路径绑定，保持原位。
  不搬目录后改 JSON 冒充同一次运行，也不复制 Mac 虚拟环境到 Linux。
- 现有 V1/V2/V3 发布文件保持身份。本次源码整理已经发现用户导入含 V3，不再从旧树
  整体覆盖；旧树中另一版 `v2/` 也不作为当前包的替代品。
- 原 V2/V3 verifier 检查源码根下默认位置的 prepared 数据。干净 clone 必须先按指南
  恢复精确文件；仅传外部 `--output-dir` 不能替代该要求。不要为此放宽冻结校验器。
- V2/V3 launcher 已支持从新源码根运行并通过 `PYTHON_BIN` 指向旧 controller；运行时
  显式传递原绝对数据/runtime 路径，下一轮使用新的 run 目录。源码目录分离不代表任意
  参数重定位都受支持，也不能跳过 release verifier 的固定 prepared 路径要求。

提交及推送前检查实际 diff 和文件清单；本次整合证据见
[repo-adoption-2026-11-03.json](../records/repo-adoption-2026-11-03.json)。

