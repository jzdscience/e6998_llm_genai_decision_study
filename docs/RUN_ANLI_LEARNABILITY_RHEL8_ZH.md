# RHEL 8 节点兼容修复与 ANLI 完整实验续行

2026-11-15 实际节点核查：控制器 Python 3.11.5 和模型环境 Python 3.12.8 均未提供 pidfd API；
内核 `4.18.0-553.8.1.el8_10.x86_64` 对 `pidfd_open` 返回 ENOSYS，但通过 `/proc/self` 目录
描述符调用 `pidfd_send_signal(..., 0)` 成功。固定 StartLux-4B 已恢复且 13 个文件内容校验通过，
旧 lock/installations 字节未变。HF 未认证提示没有阻止此次下载。

## 本次改动

新模块优先调用可用的 Python pidfd API；缺少封装时，在明确支持的 Linux x86_64 LP64 ABI 上
调用同一个内核接口。仅 `pidfd_open` 的 ENOSYS 允许打开 `/proc/PID` 目录并以信号 0 验证句柄。
权限、无效参数、文件描述符资源错误均照常失败；不会降级为普通数值 PID kill。
前后两次当前祖先关系核对、关闭描述符、subreaper、无关 decoy、孤儿后代、停止收据和 GPU 前自检保留。
Linux 对这种描述符及 PID 复用保护的说明见 [pidfd_send_signal(2)](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html)。

科学协议、三个模型 revision、输入分配、拟合/选择方法和评价规则均不变。
源码身份已改变，因此本发布默认写入 `results/aux-anli-learnability-v1-r2`。
旧 `aux-anli-learnability-v1` 保留首次失败的输入、元数据、诊断；不改写 source hash 或导入旧验收收据。
只可复用其 ANLI ZIP，且重新核对固定 SHA 后复制。模型缓存及原环境直接复用。

## 节点操作

使用独立修复分支 `anli-learnability-rhel8-2026-11-15`，新源码目录建议
`/home/zhanj289/e6998/decision-study-learnability-rhel8`。按本次交付的完整提交 SHA 固定 checkout，
不在旧 c3c8b65 checkout 中手工替换文件。随后执行：

```bash
cd /home/zhanj289/e6998/decision-study-learnability-rhel8
bash scripts/run_learnability.sh /home/zhanj289/e6998/decision-study
```

入口会先做原生 Linux CPU 自检，通过后自动继续三个模型的开发预检和完整研究。
无需重新下载模型，也不需要给这次 API 存在性失败加 `--retry-failed`。
若最终为 `STUDY_CHECKPOINT`，在实际新 allocation 中运行同一命令即可；仍核对 Slurm 真实余量。
`STUDY_COMPLETE` 才表示整套研究完成，回传最后打印的结果包。
后续真实失败仍须保留诊断并调查；兼容修复不保证 GPU 时延、内存或模型启动结果。

本机 Linux 容器和单元验证与公司 RHEL8 实际 GPU 执行分别记录，不能互相替代。
