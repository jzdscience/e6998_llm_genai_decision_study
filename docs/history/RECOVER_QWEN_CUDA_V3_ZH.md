# 补齐 FlashInfer 采样需要的 cuRAND

`pilot_bundle (6).zip` 证实独立 CUDA 13.0 工具链安装成功，nvcc、CUDA_VERSION、
CUDART_VERSION 与 Torch/FlashInfer CUDA 版本都匹配。新的编译失败是
`fatal error: curand.h: No such file or directory`。此前仅按 nvcc extra 准备四个
组件，漏掉了 FlashInfer sampling 直接使用的 cuRAND。Qwen 尚未进入模型验收。

## 运行

上传 **decision-study-qwen-cuda-v3-fix.zip** 到 `/home/zhanj289/e6998/`，执行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-qwen-cuda-v3-fix.zip
cd decision-study
bash scripts/run_with_cuda_tools.sh smoke
```

首次下载约 **161 MB**，将五个固定 SHA256 的 wheel 一起安装到新目录
`results/solo-v1/runtime/work/toolchains/cuda-13.0-flashinfer-curand10.4.0.35`。
这样保留 v2 已完成目录及其校验收据，避免覆盖或部分混入组件。新 linker 目录
也独立命名。此后按完整哈希收据复用；不改变模型 venv 或系统 CUDA。

v3 沿用 nvcc/crt/nvvm13.0.88 和 runtime13.0.96，新增 NVIDIA 官方 CUDA13.0.3
对应的 `nvidia-curand==10.4.0.35`。预检明确验证 `curand.h`、`curand_kernel.h`、
`curand_philox4x32_x.h`，随后实际编译并运行 sampling，成功后自动进入 Qwen smoke。

已检查该 sampling 模块的三个编译入口、15个项目源码/头文件和全部17个 cuRAND
头文件的引用。其余 CUDA/cooperative-groups/FP 头文件在现有 runtime wheel 中，
CCCL 的 cub/cuda/nv 头文件由 FlashInfer 自带。该模块使用 cuRAND device API，
没有 host generator 调用或额外 `-lcurand` 要求，保留原有链接参数。
这项源码检查不等于真实 nvcc 编译或完整模型运行成功。

预检仍受原 session 和20分钟上限约束，失败保留日志并导出诊断包。已有内部
deadline 是 **2026-11-02 14:50:48 PST**，本次不再次 renew。若实际 allocation
更换，按平台实际剩余时间更新。

结束后回传新导出的 `results/solo-v1/exports/pilot_bundle.zip`，失败也回传。
Kev 结果及 StartLux smoke 保留；正式实验仍等待 Qwen smoke 核验。

## 来源

- (6) 包 SHA256：`32cf9f95cbe80cf1376fa70271650de10b45b087a9f269ea8626f5d989e01b32`
- 新日志：`runtime/logs/cuda-toolchain-1793606174948841258.log`
- [FlashInfer 固定版本采样头文件](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/v0.7.0.post1/include/flashinfer/sampling.cuh)
- [cuRAND10.4.0.35 官方元数据](https://pypi.org/pypi/nvidia-curand/10.4.0.35/json)
- [CUDA13.0.3 组件元数据](https://pypi.org/pypi/cuda-toolkit/13.0.3.0/json)
- 本地依赖审查与验证：`.validation/cuda-tools-v3-2026-11-02/`
