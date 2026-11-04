# 运行 V3 ANLI 实验

2026-11-03。本轮已准备独立源码和数据；GPU 模型加载、9B 可行性及实验结果必须等待公司机
实际运行和回传验收。方案见 [V3_PROTOCOL_ZH.md](V3_PROTOCOL_ZH.md)。本机不持有远端执行入口。

## 安装新增文件

将 `decision-study-v3-ready.zip` 上传到公司机 `/home/zhanj289/e6998/`，与现有
`decision-study/` 文件夹同级。在 `e6998` 下解压；`-n` 保留现有同名文件，避免覆盖
已经开始的实验。如 verifier 报不同，保留文件和报错，不删除冻结身份或旧结果。

```bash
cd /home/zhanj289/e6998
unzip -n decision-study-v3-ready.zip
cd decision-study
.venv-controller/bin/python scripts/verify_v3_release.py
```

交付包只增加 `decision_study/v3/`、V3 脚本/说明/协议/测试和 `results/solo-v3/` 的准备数据，
不会修复或覆盖公司机已有 v2。当前本机 v2 目录存在不同版本布局，本轮使用独立 v3 入口，
不依赖 v2 的 Python import。校验器接受此前明确审计过的旧 exporter SHA，其他未知差异阻断。

## 一条命令启动

在当前仍有效的 Slurm allocation 的终端运行：

```bash
bash scripts/run_v3.sh start
```

启动命令读取当前 `SLURM_JOB_ID`，只接受 `squeue` 中该 ID 的 `RUNNING` 记录，按实际
`TIME_LEFT` 减去五分钟余量建立新的 V3 控制面时限。同一 allocation 再次 `start` 不延长
旧 deadline，直接恢复完成到的阶段。这里没有固定填写 16/20/24 小时，也不使用旧 shell
变量 `HOURS_LEFT`，避免把粘贴的命令误当时长。

若当前 shell 没有 Slurm 查询能力，但你已从平台确认真实剩余时间，可用数字显式初始化。
下面的 `实际剩余小时数` 是说明性占位，必须换成平台所示数字后执行，不可直接粘贴：
手动数字须满足 `0 < 小时数 ≤ 24`，不会再自动扣五分钟，请自行预留余量。

```text
bash scripts/run_v3.sh start --session-hours-left 实际剩余小时数
```

使用默认目录时无需额外参数。若服务器的 V1 结果位置不同，所有命令一致加上
`--v1-dir /实际/v1/目录 --output-dir /独立/v3/目录`。v3 必须与 v1/v2 分离，不放进旧输出树。

## 执行顺序与资源

1. 绑定现有 4B 权重、vLLM 环境和已修好的 CUDA 工具链；检查原文件身份及有效 session。
2. 顺序采集 StartLux4B、Qwen4B、Qwen9B 的 2,700 个 calibration/validation 输入。
   每个模型启动前做精确 token 预检、启动后做 90 个开发输入的两轮 serial smoke。
3. 在 CPU 拟合并冻结两种后级各四个政策。主方法与后级已预定，不按 validation 择优。
4. 顺序采集三个模型的 3,200 个 holdout 输入。
5. 完整校验后记录首次标签访问，计算主比较和描述性结果，导出最终包。

正式预测总数 **17,700**，初次无中断时六次启动的开发 smoke 另有 **1,080** 次调用。
恢复时只补未持久化正式预测，服务重启会再次做开发 smoke。已有 completion 的完整 job
通过证据核验后跳过，不重新启动服务。

只新增 Qwen9B 权重下载；官方参数约 19.3GB，另需缓存/编译/输出余量。现有两个 4B 权重、
Python 环境和 CUDA toolkit 只读复用，新增缓存写 `results/solo-v3/runtime-work/`。
三个模型不会同时驻留；9B BF16 单独预留 80% 显存。若仍有其他任务占 GPU，请在其完成后
运行，不能通过降低精度或替换模型绕过当前协议。实际运行时长无法由预测条数精确保证。

## 中断与续跑

同一个 allocation 仍有效：

```bash
bash scripts/run_v3.sh pipeline
```

确实换到新 allocation 后，确认 shell 的 `SLURM_JOB_ID` 对应新 job，再运行：

```bash
bash scripts/run_v3.sh start --new-allocation
```

该选项仅在实际 allocation 改变时更新运行时限，不是延长旧作业。旧数据、预测分片、freeze
与首次评价不变。失败会打印 `BLOCKED` 和诊断包路径；不要并发开启第二条 pipeline。

只查看进度或重新导出，不启动模型：

```bash
bash scripts/run_v3.sh status
bash scripts/run_v3.sh export
```

完成提示必须同时出现 `FINAL BUNDLE` 和 `V3_COMPLETE`。每个 job 后也会导出同名中间包，
有 ZIP 不代表全流程完成。最终回传：

```text
/home/zhanj289/e6998/decision-study/results/solo-v3/exports/v3_bundle.zip
```

若失败，仍回传上述诊断包及终端最后报错。包排除 `runtime-work` 中的权重、环境与缓存，
保留配置、日志、原始响应、分片、政策、标签访问和评价收据。正常结束会清理本轮服务；
是否关闭平台 allocation 由平台操作决定，不能从脚本结束推断。

## 本轮完成后判断什么

验收将分别确认 9B 是否稳定运行、是否提高直接准确率、有哪些救回/误伤、positive_gain
是否比 confidence 更好，以及实际转交量。无正收益、少转交或主区间跨零都允许正常完成。
本轮不提供 9B 同卡级联的 steady-state goodput；后续性能实验须单独配置与测量。
