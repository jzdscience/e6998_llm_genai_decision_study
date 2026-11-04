# Qwen 启动时找不到 Ninja（2026-11-01）

用户已将公司 session 的主机 RAM 调到 100 GB。新 pilot 的 331 个清单文件
通过校验；StartLux 完成参考对照、串行和 8 并发验收，Kev 正式结果保持不变。
目前 Qwen 在 FlashInfer sampling JIT 阶段退出，具体报错为：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
```

这是本次退出的直接错误。回传环境已记录 `ninja==1.13.2`，而 controller 用
绝对路径启动 vLLM Python，没有将该环境的 bin 放进 PATH。指定解释器不会自动
使该环境的裸工具命令可被子进程找到。包清单不单独证明可执行文件完好，所以新
启动器会先检查指定 bin/ninja 并实际运行 `ninja --version`，失败则停止。

## 运行补丁

上传 `decision-study-qwen-tools-fix.zip` 到公司 `/home/zhanj289/e6998/`，执行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-qwen-tools-fix.zip
cd decision-study
bash scripts/run_with_vllm_tools.sh smoke
```

此命令只验收 Qwen。它从现有 installations/lock 记录定位 vLLM 的 bin，
先固定 controller 的解释器，再前置工具路径；不会把 controller 切换到 vLLM
Python，也不运行 pip install 或更改模型环境。已有核心配置、StartLux smoke、
Kev 预测和 gate 均保留。正式模型验收仍以新回传为准。

结束后回传 `results/solo-v1/exports/pilot_bundle.zip`，失败也回传。
如果开头即报告 bin/ninja 不存在、不可执行或版本命令失败，请保留该输出；
不要通过升级共享 vLLM 环境绕过错误。

## 后续正式命令（先等 Qwen smoke 验收）

工具路径是进程环境的一部分，在新 shell/session 可能丢失；后续涉及 Qwen 的
运行继续用同一个启动器，避免只修当前终端而下次又找不到 Ninja。

```bash
bash scripts/run_with_vllm_tools.sh startlux_full
bash scripts/run_with_vllm_tools.sh qwen_test
.venv-controller/bin/python -m decision_study evaluate --output-dir results/solo-v1
.venv-controller/bin/python -m decision_study export --kind experiment --output-dir results/solo-v1
```

真实性能测量仍在正式结果核验后单独安排。启动器支持 benchmark，但此次不自动运行。

## 本次检查的其他事项

- vLLM 日志还提示关闭 chunked prefill 不受默认配置推荐。v0.31.0 源码显示该警告
  是通用条件触发，并非检测到 Qwen-specific 无效状态；当前 8192 的 batch token
  budget 等于 context limit，满足关闭模式的显式校验。同样警告也出现在已通过
  reference 对照的 StartLux 日志。保留冻结配置；不将该警告误当 Ninja 异常的根因。
- Qwen smoke 检查输出标签及并发一致性，没有独立 reference 模型对照；记录应保留
  此限制，不把 smoke 当成模型任务准确率或完整正确性证明。
- 新包内 session 记录仍截止 `2026-11-01 22:17:43 PST`，并未自动跟随 RAM 调整续期。
  只有实际新开 allocation 后，按真实剩余时间运行 session --renew-session；
  不能凭本机日期或“100 GB”推定还有 22 小时。
- CPU 真实子进程复现已证明同一解释器仅改变 PATH 可使裸 Ninja 从失败变成功。
  启动器的 CPU 检查不替代远端 Ninja 文件验证、CUDA 编译或 Qwen smoke。

## 来源

- 新包：`records/bundles/startlux-pass-qwen-blocked-pilot-20261102-010718-7e7b7009d25d.zip`
- 实际错误：`runtime/logs/qwen-engine-1793581376926487327.log`
- 已安装依赖：`runtime/resolved/vllm-0.31.0.txt`
- [Python venv 文档](https://docs.python.org/3/library/venv.html)
- [vLLM v0.31.0 配置警告条件](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/engine/arg_utils.py)
- [vLLM v0.31.0 scheduler 配置校验](https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/vllm/config/scheduler.py)
