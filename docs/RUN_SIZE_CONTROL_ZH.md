# 在公司 code-server 运行 StartLux9B 对照

本轮固定ANLI3,200条，只新增一组StartLux9B预测。先按Git交付说明更新独立源码checkout，
保留旧运行根和原结果。配置见 [协议](SIZE_CONTROL_PROTOCOL_ZH.md)。

## 启动

在公司终端执行（当前Slurm allocation须仍在运行）：

```bash
cd /home/zhanj289/e6998/decision-study-src
export PYTHON_BIN=/home/zhanj289/e6998/decision-study/.venv-controller/bin/python
bash scripts/run_size_control.sh start \
  --v1-dir /home/zhanj289/e6998/decision-study/results/solo-v1 \
  --v3-dir /home/zhanj289/e6998/decision-study/results/solo-v3 \
  --output-dir /home/zhanj289/e6998/decision-study/results/aux-size-control-v1
```

`--v3-dir`必须是包含六组prediction job、policy_freeze和最终evaluation的**完整旧V3输出**，
不是新clone中的prepared数据。默认从当前 `SLURM_JOB_ID` / `squeue` 读取剩余时间并留5分钟；
若集群不支持查询，可加 `--session-hours-left 真实剩余小时数`，不要复制历史数值。
同一allocation重复启动不会延长原deadline。若整轮已完成，`start`/`pipeline`先核验
完整freeze/预测/access/evaluation链，然后直接复用导出，不查询Slurm、不续期或重新读gold；
这也允许整个后续实验套件带`--new-allocation`时跳过已经完成的规模对照。

首次运行下载约19.3GB tensor数据并加载单独9B模型，启动/编译时间另计。旧环境和checkpoint
不改写；任何报错保留日志和诊断包，不删除锁、不修改已冻结配置。运行前应已结束其他GPU实验，
因为本实验独占同一L40S和原HTTP端口。启动CUDA预检前会只读查询已有GPU compute PID；
若存在则停止，不会杀死或接管其他进程。

## 恢复与诊断

同一allocation中断后重跑上述 `start`，已持久化终止预测按ID复用。若真正换了Slurm job，
在同一命令额外加 `--new-allocation`。新job必须与旧job ID不同；不能用此开关延长旧job。

将 `start` 换成以下操作可分阶段运行，其他三个路径参数保持不变：

| 操作 | 行为 |
|---|---|
| `prepare` | 只核验并复制公开输入与历史预测，不读test gold，不下载模型 |
| `init-runtime` | 只绑定隔离运行/session元数据 |
| `download` | 下载固定9B；需要初始化后的有效session，不启动GPU服务 |
| `smoke` | 实际GPU/token检查和两遍development smoke；冻结配置，不采正式test |
| `collect` | 完成3,200条正式采集，不评价 |
| `pipeline` | 使用已初始化session继续准备、采集、评价、导出 |
| `status` | 核验已有状态，不读gold，不访问GPU |
| `evaluate` | 仅在完成预测与冻结一致时评价，不重新采集/拟合 |
| `export` | 导出当前状态，可能只是部分进度 |

若 `evaluation-access.json` 已存在但 `evaluation.json` 缺失，程序会停止并要求检查保留的
诊断，不能删收据来重试。软件状态与数据身份失败不通过改hash绕过。

结束时必须看到 `SIZE_CONTROL_COMPLETE`，回传：

```text
/home/zhanj289/e6998/decision-study/results/aux-size-control-v1/exports/size_control_bundle.zip
```

若报错，回传同一路径诊断包及末尾日志。模型weights和环境不会进入ZIP。
本机验证只验证软件路径；实际显存可行性、正式预测及准确率须待node执行/回传验收。
