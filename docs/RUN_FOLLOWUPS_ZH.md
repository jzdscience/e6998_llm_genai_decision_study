# 运行服务性能与 StartLux9B 两项收尾实验

2026-11-05。两套源码、固定协议与本机软件验证已交付；实际GPU运行仍在公司code-server
终端执行。本机无法进入该终端，所以这里不把准备完成写成实验完成。

## 一次启动两项

在已获分配的公司node上，先按交付消息中的完整commit更新独立源码checkout。保持旧
`/home/zhanj289/e6998/decision-study`运行根在原位。然后执行：

```bash
cd /home/zhanj289/e6998/decision-study-src
bash scripts/run_followup_suite.sh /home/zhanj289/e6998/decision-study
```

脚本先跑四个直接服务的60个正式cell，再跑StartLux9B的3,200条ANLI；两项串行使用单卡。
它使用现有controller和`.venv-cheap-baselines`，不安装或升级旧GPU环境。若CPU环境缺失，
按[CPU基线恢复说明](RUN_CHEAP_BASELINES_ZH.md)恢复固定依赖即可，无需重新训练基线。
固定已验收CPU产物须仍在旧根`results/aux-clinc-cheap-v1`，MiniLM文件在
`results/aux-clinc-cheap-v1-inputs/model`；脚本逐项验证它们，不能用其他训练结果替代。

默认从当前`SLURM_JOB_ID`和`squeue`读取真实剩余时间并留5分钟，不需要`HOURS_LEFT`变量，
不使用历史session过期时间。服务阶段3小时仅指offered窗口；另计60次服务启动/预热、
drain及导出。StartLux9B另需约19.3GB tensor下载、加载、开发smoke和正式采集。
当前实际剩余时间不能从旧对话推断，脚本会检查；完整运行时长由node实测决定。

只运行一个实验进程。若GPU已有compute进程，脚本停止并保留诊断，不终止其他实验。
也不要同时开启旧V3 pipeline或另一个本轮命令。

## 结束标志与回传

只有看到`FOLLOWUPS_COMPLETE`才代表两个pipeline都走到最终评价。分别回传：

```text
/home/zhanj289/e6998/decision-study/results/aux-clinc-service-v1/exports/service_study_bundle.zip
/home/zhanj289/e6998/decision-study/results/aux-size-control-v1/exports/size_control_bundle.zip
```

`MEASUREMENT BUNDLE`或`DIAGNOSTIC BUNDLE`只是阶段/故障快照，存在ZIP不代表最终完成。
结束后仍需独立验收回传，确认真实配置、请求记录、评价及硬件边界。

## 继续与诊断

若在同一个allocation恢复、没有中断的正式cell，重跑同一条suite命令即可。已完整完成
且通过验证的阶段直接复用。真正换到新的Slurm job后，额外加`--new-allocation`。
该开关要求job ID实际改变，不能延长同一job。

若服务测量已有中断attempt，先回传诊断包与最后报错供检查。明确恢复该未完成cell时：

```bash
bash scripts/run_followup_suite.sh /home/zhanj289/e6998/decision-study --resume-interrupted
```

若同时换了allocation，再加`--new-allocation`。旧attempt全部保留；已完成的慢轮/错误轮
不重跑。若首次evaluation-access存在但最终评价缺失，程序会阻断而不静默重评分，保留
收据供检查，不删除access、freeze或锁。

单独查看/导出服务阶段：

```bash
export PYTHON_BIN=/home/zhanj289/e6998/decision-study/.venv-controller/bin/python
bash scripts/run_service_study.sh status --output-dir /home/zhanj289/e6998/decision-study/results/aux-clinc-service-v1
bash scripts/run_service_study.sh export --output-dir /home/zhanj289/e6998/decision-study/results/aux-clinc-service-v1
```

服务CLI也支持`prepare`、`init-runtime`、`pipeline`、`evaluate`。`prepare/start`需显式给
`--v1-dir --bundle-dir --model-dir --cpu-python --output-dir`，suite已提供全部准确路径。
无法查询squeue时，单独start/init-runtime可给`--session-hours-left`实际数值，不猜测剩余时长。

9B单独启动与分阶段命令见[运行说明](RUN_SIZE_CONTROL_ZH.md)。实验边界见
[服务协议](SERVICE_STUDY_PROTOCOL_ZH.md)与[9B协议](SIZE_CONTROL_PROTOCOL_ZH.md)。
