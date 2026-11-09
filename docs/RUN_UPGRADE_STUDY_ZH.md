# 在公司节点运行 StartLux9B 服务成本研究

2026-11-08。配置见[协议](UPGRADE_STUDY_PROTOCOL_ZH.md)，审核意见见
[11/8 审核](UPGRADE_STUDY_REVIEW_2026-11-08_ZH.md)。先做开发预检并回传审核，审核通过后再跑
36 个正式单元。两步都在已获分配的公司节点上执行，使用同一个完整 commit。

## 运行前

1. Codex 已验收实现提交，提交已合并进 GitHub 的 `main`，交付消息给出完整的 40 位 SHA。
2. 独立源码 checkout `/home/zhanj289/e6998/decision-study-src` 已按 [NODE_GIT_ZH.md](NODE_GIT_ZH.md)
   的“后续更新”取到该 SHA（把 `STUDY_COMMIT` 的赋值换成交付的 SHA），工作区干净。节点 checkout 若还停在
   11-07 改写历史之前的 commit，fetch 输出里出现 `origin/main (forced update)` 属于预期。
3. 旧运行根 `/home/zhanj289/e6998/decision-study` 保持原位，其中已有 `results/solo-v1`、`results/solo-v3`、
   `results/aux-size-control-v1-r2` 和 `.venv-controller`。StartLux9B 与 Qwen3.5-9B 沿用规模对照和 V3
   已下载的快照，本研究不下载模型；快照缺失时程序停止，不替换版本。
4. GPU 上没有其他 compute 进程，也没有其他实验命令在跑。控制器只读查询 GPU PID；遇到无关进程时控制器
   自己退出，那个进程保持原样。

## 第一步：开发预检

```bash
cd /home/zhanj289/e6998/decision-study-src
bash scripts/run_upgrade_study.sh /home/zhanj289/e6998/decision-study --preflight-only
```

脚本依次执行 `prepare`、`init-runtime`、`preflight`，输出目录为
`results/aux-anli-upgrade-service-v1`。剩余时间从当前 Slurm job 读取，每个条件开始前要求至少还剩
30 分钟。三种部署各冷启动一次，使用正式启动命令、显存份额和 token 上限：

1. 每个角色对 3,200 条正式输入和 90 条开发输入做 tokenizer 长度检查，不截断。
2. 90 条开发输入串行调用。级联对每条输入调用两个模型，确认两个接口都能回答。
3. 16 个 worker、每秒 10 个开发请求、持续 30 秒的并发检查。级联在这一步强制转交每个请求
   （`preflight_forced_deferral`），要求两个引擎的工作时间有重叠，且每条成功请求都调用了后级。
4. 记录启动参数、KV cache 容量、最大并发、GPU 进程与显存、health 和 preemption 信息。

出现调用错误、超时或预算用尽，检查后 health 失败，引擎日志报告 OOM 或引擎退出，并发检查超出父进程
watchdog 时限，或级联未满足第 3 条，该条件判为 FAIL，证据保存在 `preflight/<条件>-<时间戳>/`。预检不读 test gold，也不比较 goodput。

看到 `PREFLIGHT_COMPLETE` 后回传：

```text
/home/zhanj289/e6998/decision-study/results/aux-anli-upgrade-service-v1/exports/upgrade_study_bundle.zip
```

审核看三份 `preflight/<条件>.json` 中的 KV 容量、最大并发和级联重叠时间。收据里的
`deployment.roles[].startup_seconds` 可以用来估算正式运行的总时长。

### 预检失败

回传同一路径的诊断包和终端末尾输出。确认是临时原因（例如偶发的 I/O 或端口问题）后，在同一命令后加
`--retry-preflight` 重试，旧失败证据保留。OOM、KV cache 不足或引擎退出需要按审核意见新开协议修订版本
再预检；当前输出目录里的份额和冻结文件保持原样。

## 第二步：正式测量

预检审核通过后，在同一 checkout 执行：

```bash
cd /home/zhanj289/e6998/decision-study-src
bash scripts/run_upgrade_study.sh /home/zhanj289/e6998/decision-study
```

`start` 先核验准备文件和三份 PASS 收据，再按固定顺序跑 36 个单元（3 次重复 × 4 个速率 × 3 种部署）。
每个单元重新启动该部署的服务，串行预热 90 条开发输入，测 120 秒开环 Poisson 到达，最多 drain 60 秒，
然后停止服务并确认 GPU 上没有自有进程。offered 窗口合计 72 分钟，另加 36 次冷启动、预热和 drain。
每个单元开始前要求 allocation 至少还剩 18 分钟，不够时停止并导出，换新 allocation 后恢复。

36 个单元全部完成并冻结后才读取 gold 计算指标。结束标志是 `UPGRADE_STUDY_COMPLETE`，回传同一个
`exports/upgrade_study_bundle.zip`。模型权重和环境不进 ZIP。

## 中断与恢复

1. 同一 allocation 内中断后，用同一命令加 `--resume-interrupted`。已完成单元从原始 trace、计划和资源文件
   重新核验，通过的不重跑；中断的尝试原样保留，该单元另开一次新尝试。
2. 换了新的 Slurm job 时再加 `--new-allocation`。新 job ID 必须与旧的不同，这个开关不能延长旧 job。
   评价会报告每组配对单元是否来自同一 allocation。
3. 已有 `evaluation-access.json` 而缺少 `evaluation.json` 时程序停止，保留现场等待检查。
4. 已完成的研究再次 `start` 时只核验原有结果并重新导出，`evaluation.json` 保持原字节。

## 单独查看状态或导出

脚本只封装 `--preflight-only` 和完整 `start`。正式运行没有进行时，可在源码 checkout 中执行：

```bash
cd /home/zhanj289/e6998/decision-study-src
PY=/home/zhanj289/e6998/decision-study/.venv-controller/bin/python
OUT=/home/zhanj289/e6998/decision-study/results/aux-anli-upgrade-service-v1
"$PY" -m decision_study.upgrade_study status --output-dir "$OUT"
"$PY" -m decision_study.upgrade_study export --output-dir "$OUT"
```

| 命令 | 行为 |
|---|---|
| `status` | 核验已有状态，不读 gold，不访问 GPU |
| `export` | 导出当前状态，可能只含部分进度 |
| `evaluate` | 36 个单元全部完成后冻结并评价，不重新测量 |

## 本机验证范围

11/8 在没有 GPU 的 Mac 上：

1. `tests/test_upgrade_study.py` 的单元测试覆盖政策重放、共同预算、服务清理、计划、指标、恢复核验、
   中断尝试的保留、跨 allocation 配对的报告、allocation 续期和 gold 读取时机。
2. 一次 CLI 全流程演练在真实 V1/V3/规模对照证据上依次执行 `prepare`、`init-runtime`、`preflight`、
   `start` 和第二次 `start`。替换的部分只有 GPU launcher、nvidia-smi、硬件 doctor、CUDA 工具链检查、
   tokenizer、模型服务和旧环境作用域；日程缩短为每种部署一个 2 秒单元。
3. 20 处注入错误都被对应测试发现。

0.50 单模型和 0.40 + 0.50 同驻留能否启动、KV 容量是否够用、真实延迟多少，要由节点预检和正式测量回答。
