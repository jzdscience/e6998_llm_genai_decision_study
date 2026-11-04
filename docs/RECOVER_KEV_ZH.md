# 修复 Kev 的 Python 环境与构建副本检查（2026-10-30）

最新 13:21 PDT 回传已确认此修复完成、Kev 串行验收通过。当前机器直接按
[QUICKSTART 的当前进度](QUICKSTART_ZH.md#当前进度kev-串行验收已通过开始正式预测)
更新代码并运行 kev_full 即可；下方步骤保留用于最初故障恢复。

公司机器的实际错误是 `fatal error: Python.h: No such file or directory`。
旧流程创建的 Kev 环境基于系统 Python 3.12.8，Triton 编译时查找
`/usr/include/python3.12`，但该位置没有开发头文件。
修复使用 uv 管理的 CPython 3.12.8，在安装模型依赖前检查头文件，随后执行
BF16 CUDA 和 Triton 辅助模块验收。无需修改系统 Python。

已确认并获准清理的旧缓存仅有：

- `mistralai/Mistral-7B-Instruct-v0.2`，用户目录报告约 14 GB。
- `google/gemma-7b-it`，约 16 GB。
- `meta-llama/Llama-2-13b-chat-hf`，约 25 GB。

它们位于 `~/.cache/huggingface/hub`，本项目不使用。目录统计不等于精确可回收
配额，清理后以平台报告为准。再次使用这些模型需要重新下载及相应访问权限。

## 执行

先停止当前实验进程。将**最新** `decision-study-build-fix.zip` 上传到
`/home/zhanj289/e6998/`，然后运行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-build-fix.zip
cd decision-study
bash scripts/recover_kev.sh
```

ZIP 只包含代码、notebook、测试和文档，解压不会覆盖 `results`、模型缓存或
`.venv-controller`。脚本使用现有 `.venv-controller`（找不到时使用 python3）。
无需重新执行控制器的 pip install。

三个指定旧缓存此前已清理成功，无需再加清理参数。
脚本顺序：继续修复 Kev Python 环境 → **只验收 Kev**。
修复保留本项目的模型权重、数据和已解析依赖版本；会替换 Kev 的虚拟环境。
已有正式预测时拒绝该操作，避免混合配置。该次运行尚无正式预测，可以修复后
重新 smoke。中断后可以重跑同一命令。

仍使用原来的 session 截止时间。如果实际已续开新的 24 小时 session，先执行：

```bash
.venv-controller/bin/python -m decision_study session --renew-session \
  --session-hours-left 22 --output-dir ./results/solo-v1
```

只有真实续期开启新 session 才执行上面命令；如果剩余不足 22 小时，填写更小值。

结束后返回 `results/solo-v1/exports/pilot_bundle.zip`。仅 Kev 通过时 pilot 仍然
标为部分完成，因为 StartLux/Qwen 尚未验收；这是预期状态。

如果前一版修复在 `Source checkout changed` 停止，而固定 HEAD 正确、变更列表
只有 `build/lib/kev/*.py`，请使用最新代码包重跑 `bash scripts/recover_kev.sh`。
这些文件可由 setuptools 安装生成；修复会核对每一个副本与固定 HEAD 中的源码
是否逐字节相同，再清理匹配副本。真正的源码改动、其他新增文件或不一致的副本
仍会阻止运行。无需手动删除源码目录或再次清理旧模型缓存。

## 分步执行与日志

查看清理范围（默认不删除）：

```bash
python scripts/cleanup_legacy_cache.py
```

单独执行已授权的清理：

```bash
python scripts/cleanup_legacy_cache.py --apply
```

单独修复 Python，不开始参考模型推理：

```bash
.venv-controller/bin/python -m decision_study repair-python \
  --models kev --managed-python-version 3.12.8 --cost-mode provided \
  --output-dir ./results/solo-v1
```

GPU 检查和安装日志位于 `results/solo-v1/runtime/logs/`。失败时主错误会指出
具体日志路径；返回新的 pilot 包即可。当前验证包含 CPU 测试和本地 managed
Python 的 C 扩展编译，尚未在公司 L40S 上执行修复后的 GPU 验收。
