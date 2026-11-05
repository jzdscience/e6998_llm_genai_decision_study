# V1–V3 Decision Lab

这个版本把三轮已经完成的实验放进一个 CPU 演示。V1 的原始回放与阈值探索保留，新增
V2/V3 的主比较、冻结政策、实际转交预算、案例和性能证据。不会运行模型或修改实验参数。

## 在本机启动

本机已为这个源码仓库准备独立 `.venv-lab`。在仓库根执行：

```bash
bash scripts/run_lab.sh
```

浏览器打开 `http://127.0.0.1:7860`。停止用 Ctrl-C。如果这个端口已被其他应用占用：

```bash
bash scripts/run_lab.sh --port 7861
```

在另一台电脑首次安装，使用 Python 3.11–3.13：

```bash
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[lab]'
bash scripts/run_lab.sh
```

只看演示不需要公司 GPU、原始结果目录或模型权重。两个便携证据包都在 `reports/` 中。
启动器也支持已存在的 `.venv`，或显式设置 `DECISION_LAB_PYTHON`；不会自动安装依赖、
寻找远端服务或续期 allocation。

## 建议的演示顺序

1. **Overview**：先看两项预定主比较及配对区间。V2/V3 区间均跨零，不宣称主方法改善。
2. **V2 · SQuAD**：比较冻结路由的准确率、救回与误伤，再查看不同到达率下实际测得的
   SLO goodput。主政策表采用 validation 10% 目标；实际 holdout 转交率单列。
3. **V3 · ANLI**：观察 positive_gain 零转交，以及 confidence 在 validation 与 holdout
   的预算差异。Qwen9B 确实采集了预测，但政策是否选用这些预测是另一件事。
4. **V1 replay**：使用原有案例、阈值探索和性能页。滑块是在已见数据上的探索，不改变
   冻结 gate，也没有任意滑块工作点的实测时延承诺。

案例按已知结果类型挑选以帮助解释，不代表总体发生比例。所有总体指标来自完整评价池。
V3 没有共驻留性能测量；界面会明确显示缺项，不把顺序采集时延相加作为级联性能。

## 数据与复核

- V1 原便携包：`reports/decision-lab-2026-11-02/`。
- V2/V3 新便携包：`reports/decision-lab-2026-11-04/`。
- 新程序：`decision_study/lab/`。旧 V1/V2/V3 实验及 Lab 源码保持原字节。
- 原始结果、冻结参数和独立审计的来源见 [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md)。

校验便携包：

```bash
.venv-lab/bin/python -m decision_study.lab check
```

只有需要重新制作便携包时，才需要原始外部证据。选择一个尚不存在的输出目录：

```bash
.venv-lab/bin/python -m decision_study.lab build \
  --evidence-root /path/to/verified-evidence-root \
  --bundle /path/to/new-lab-bundle
```

构建器核验来源和冻结结果，只读取已验收证据。新的演示包不替代原结果 ZIP，不用本机
重拟合值覆盖原 freeze。查看时仅加载便携 JSON，公开分享关闭，默认只监听 localhost。
