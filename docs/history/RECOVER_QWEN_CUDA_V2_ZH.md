# Qwen CUDA 编译器与头文件版本对齐

`pilot_bundle (5).zip` 的预检已成功找到 nvcc 13.4、Ninja、真实 NVIDIA 驱动，
但 FlashInfer 编译失败，错误为：

```text
CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths
```

现有 model venv 混合安装了 nvcc/crt/nvvm 13.4.92 和 runtime 13.0.96。
FlashInfer 所带 CCCL 检查的是编译器与 `CUDART_VERSION` 的 major/minor 一致性。
驱动对 CUDA 13.x 的兼容能力不能替代这项编译期检查。上一版路径补丁未解决该问题。
这次停止发生在独立预检中，Qwen 没有再次加载模型；原有实验记录均保留。

## 操作

上传 **decision-study-qwen-cuda-v2-fix.zip** 到 `/home/zhanj289/e6998/`，执行：

```bash
cd /home/zhanj289/e6998
unzip -o decision-study-qwen-cuda-v2-fix.zip
cd decision-study
bash scripts/run_with_cuda_tools.sh smoke
```

首次会从官方 PyPI 文件地址下载约 **101 MB** 的四个 CUDA wheel，按固定 SHA256
校验后安装到独立目录 `results/solo-v1/runtime/work/toolchains/cuda-13.0-nvcc13.0.88`。
这是模型环境之外的构建工具目录，不加入 PYTHONPATH，不升级或降级原 Torch/vLLM。
后续运行验证完成收据和文件哈希后直接复用。

工具链版本采用 NVIDIA 官方 `cuda-toolkit==13.0.3` 的 nvcc extra：

| 组件 | 版本 |
| --- | --- |
| nvidia-cuda-nvcc | 13.0.88 |
| nvidia-cuda-crt | 13.0.88 |
| nvidia-nvvm | 13.0.88 |
| nvidia-cuda-runtime | 13.0.96 |

脚本先检查 nvcc 13.0、`CUDA_VERSION=13000`、`CUDART_VERSION=13000`，再运行
实际小型 FlashInfer sampling 检查。通过后自动继续 Qwen smoke。当前 GCC 8.5
与 C++17 在 CUDA 13.0 和该版 CCCL 的支持范围内，不额外更换宿主编译器。

预检与下载使用既有 session 和排他锁，合计最多 20 分钟且不超过内部 deadline。
准备失败会保留暂存目录和日志；不发布为可复用工具链，也不运行模型。已有目标
内容与收据不符时停止，不覆盖现有目录。模型环境 freeze 在运行前后均核对。

运行日志位置会打印到终端。结束后回传新导出的
`results/solo-v1/exports/pilot_bundle.zip`，失败也回传。setup 和 probe 报告均进入包。
本次不用再次 renew；已记录的内部时限截止 **2026-11-02 14:50:48 PST**。
若实际平台 allocation 已更换，按实际剩余时间另行更新，不照抄旧 16 小时。

## 保持不变

- 不修改 Kev 绑定的五个核心模块、模型权重、数据、prompt、gate 或推理配置。
- 不在冻结 model venv 中执行安装，不重新运行 Kev，不删除旧结果、缓存或日志。
- 不关闭 CCCL 版本检查，不将编译器/头文件不匹配当作可以忽略的警告。
- 仍使用两个 Ninja 构建 job、每个 nvcc 一个线程。预检进程结束后才启动模型。

本地验证不包含真正的 Linux nvcc/GPU 编译。Qwen smoke 仍须以远端新回传为准；
通过并核验之后再继续 StartLux/Qwen 正式预测。

## 来源

- 回传包 SHA256：`3841808c03ae6145353b8dd6f983e9c47ba231a7158d9acdf00c058b76dee9f1`
- 实际错误：`runtime/logs/cuda-toolchain-1793604486293663067.log`
- [NVIDIA cuda-toolkit 13.0.3 的组件元数据](https://pypi.org/pypi/cuda-toolkit/13.0.3.0/json)
- [固定 CCCL 的编译器/头文件检查](https://raw.githubusercontent.com/NVIDIA/cccl/16bd510c9b712e82b0ab6cbb630d8e29ba1f7116/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h)
- [CUDA 13.0 宿主编译器支持](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-installation-guide-linux/index.html#host-compiler-support-policy)
- [固定 CCCL 支持范围](https://github.com/NVIDIA/cccl/blob/16bd510c9b712e82b0ab6cbb630d8e29ba1f7116/README.md)
