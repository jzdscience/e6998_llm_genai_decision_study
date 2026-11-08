# 补齐 StartLux reference 依赖（2026-11-01）

本次回传的 310 个文件均通过 SHA256 校验。实际失败发生在 reference 模型加载：
`device_map` 需要 Accelerate，但环境未安装该包。StartLux reference 和 vLLM 的
BF16/Triton 小检查已通过，StartLux 权重和 tokenizer 检查已完成；Qwen 尚未开始。
用户当前 home 目录 `du` 占用约 76G，配额以平台报告为准。

此补丁只新增恢复脚本、测试和说明，不覆盖已用于 Kev 正式预测的核心代码或配置。
脚本只向 `startlux-reference` 增加固定版本的 `accelerate==1.15.0` 和
`psutil==7.2.2`，使用 `--no-deps` 禁止解析/升级原有依赖，再核对旧依赖完全保留。
两个 wheel 下载合计约 0.55 MB；这是包下载量，不是整个运行后新增空间的承诺。
后续 Qwen 验收仍需下载它的权重。

## 运行

把 `decision-study-startlux-fix.zip` 上传到 `/home/zhanj289/e6998/`，执行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-startlux-fix.zip
cd decision-study
bash scripts/recover_startlux.sh
```

顺序为：备份小型元数据 → 补齐依赖并验证 → 更新 reference 环境记录 →
重新运行 StartLux、Qwen smoke。恢复期间保留 Kev 的 6,250 条正式预测、校准和 gate，
保留现有权重和虚拟环境。已有 StartLux 正式记录时拒绝修改其依赖。
任何旧 StartLux smoke 会先归档，避免参考环境变更后复用旧验收。
脚本可恢复中断的依赖补装/元数据提交，不通过重新建环境解决本次问题。

结束后返回 `results/solo-v1/exports/pilot_bundle.zip`；失败时也会导出日志。
新增修复只在 CPU 上验证了恢复逻辑，目标机器的真实导入、GPU 小检查和完整 smoke
由以上命令执行。原 session 绝对截止时间继续生效；此脚本不自动续期。
本次不修改预测、模型、提示或校准算法，后续是否能通过模型验收以回传结果为准。

验证记录：99 个 CPU 测试、2 个子测试通过，其中新增 9 项覆盖依赖修复与中断恢复。
五个冻结 controller 源文件的组合 hash 仍与 Kev 正式配置一致；回传包内 253 个
Kev 预测/校准文件与上一份正式结果逐字节一致。恢复 shell 的顺序和失败即停行为
也已用模拟命令验证。

## 依据

- [固定 StartLux reference backend](https://raw.githubusercontent.com/StartLuxLabs/StartLux-Decision/a99977d9332decc9fb27a2488b4a83b533b6cd5f/startlux_decide/backends/transformers.py)
- [固定版本依赖声明](https://raw.githubusercontent.com/StartLuxLabs/StartLux-Decision/a99977d9332decc9fb27a2488b4a83b533b6cd5f/pyproject.toml)
- [Accelerate 1.15.0 依赖元数据](https://pypi.org/pypi/accelerate/1.15.0/json)
- [psutil 7.2.2 依赖元数据](https://pypi.org/pypi/psutil/7.2.2/json)
