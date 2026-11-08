# Qwen CUDA 编译器路径恢复（2026-11-01）

`pilot_bundle (4).zip` 已确认内部 session 更新成功、Ninja 已运行。新错误在
FlashInfer sampling JIT：`/usr/local/cuda-12.1/bin/nvcc: No such file or directory`。
这次有新的 Qwen 启动日志，但没有完成 Qwen smoke。Kev 253 个受保护文件与
StartLux 已通过的 smoke 保持不变。

## 运行

上传 `decision-study-qwen-cuda-fix.zip` 到公司 `/home/zhanj289/e6998/`，然后执行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-qwen-cuda-fix.zip
cd decision-study
bash scripts/run_with_cuda_tools.sh smoke
```

成功后会自动继续 Qwen smoke。结束后回传
`results/solo-v1/exports/pilot_bundle.zip`；如果失败，同样回传这个新导出的包。
预检开始会打印具体日志位置，可以在另一个终端 `tail -f` 查看。

已收到的内部时限截止 **2026-11-02 14:50:48 PST**，无需重复 renew。
它来自用户声明的剩余 allocation，不是平台实时保证。若实际平台 session 更换，
仍按新 allocation 的实际剩余时间更新；脚本不会自动续期。

## 修复范围

- 复用已安装的 `nvidia-cuda-nvcc==13.4.92` 等 CUDA 13 wheel，读取实际文件记录，
  不执行 pip install，不升级 vLLM/Torch，不修改冻结 core 或实验参数。
- 现有 wheel 使用 `nvidia/cu13` 统一布局。设置实际 CUDA_HOME/CUDA_PATH、
  nvcc 与 Ninja 路径；在项目 runtime/work 中创建链接目录，补足 FlashInfer
  使用 `-lcudart -lcuda` 所需的链接名称，不修改系统 CUDA 或 model venv。
- 使用真实驱动库，不把 driver stub 加入运行时库路径。编译并发限制为两个 Ninja
  job、每个 nvcc 一个线程；这是构建资源限制，不是模型 batch 或推理参数。
- 在读取既有 session、检查 primary profile 和依赖 freeze 后，独立子进程执行
  小型 FlashInfer sampling 编译/运行检查；它不加载模型权重。检查成功、子进程
  退出后，才释放预检锁并进入原 Qwen launcher。预检最多 20 分钟，且受 session
  剩余时间约束；失败会保留报告/日志并导出 diagnostic bundle。
- CUDA 13.4 编译器与 Torch CUDA 13.0 的版本差异需要实际验证。NVIDIA 文档允许
  CUDA 13.x 在 580 系列驱动上进行有条件的 minor compatibility；目标架构、
  PTX 和新驱动功能仍有限制。预检成功也不替代完整 Qwen smoke。

每次运行会生成独立 `runtime/cuda-toolchain-*.json` 与日志，记录实际工具链和
检查结果。不会删除旧失败日志、模型缓存、smoke 报告或预测。

## 后续

先核验这次 Qwen smoke，再通过同一新 launcher 依次运行 `startlux_full`、
`qwen_test`。之后再 evaluate/export；本次不自动启动正式实验或 benchmark。

本地 CPU 检查只能验证启动流程、环境选择与失败保护；本机没有 NVIDIA GPU，
远端编译、GPU probe 和 Qwen 验收以新回传为准。

## 来源

- 原始包 SHA256：`84cbf79519bc587bd9b3e2dd23ac5fa366ba8bbd5a91113d63026e01d697c903`
- 新日志：`runtime/logs/qwen-engine-1793602308827803883.log`
- 已安装依赖：`runtime/resolved/vllm-0.31.0.txt`
- [FlashInfer 0.7.0.post1 CUDA 路径、编译和链接实现](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.7.0.post1/flashinfer/jit/cpp_ext.py)
- [NVIDIA CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [NVIDIA CUDA Linux 安装说明](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
