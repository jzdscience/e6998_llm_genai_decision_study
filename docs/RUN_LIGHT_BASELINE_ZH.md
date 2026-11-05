# 运行第一批 CPU 分类基线

本轮固定 400 个训练标签，只运行 TF-IDF＋LogisticRegression，详见
[实验方案](LIGHT_BASELINE_PROTOCOL_ZH.md)。旧 GPU 环境、模型和 V1/V2/V3 结果保持原位。

## 源码与独立依赖

先按本轮交付的完整 commit 更新 `/home/zhanj289/e6998/decision-study-src`，不要在旧
`decision-study` 目录执行 Git checkout。Git 更新步骤见 [NODE_GIT_ZH.md](NODE_GIT_ZH.md)。
只有源码 checkout 干净且没有任务使用它时才更新。

在新源码目录创建独立 CPU 环境；不要把这些依赖装入 `.venv-controller` 或 vLLM 环境。
要求 Python 3.11–3.13，可复用旧 controller 的 Python 来创建新 venv：

```bash
(
  set -eu
  cd /home/zhanj289/e6998/decision-study-src
  /home/zhanj289/e6998/decision-study/.venv-controller/bin/python -m venv .venv-baselines
  .venv-baselines/bin/python -m pip install -r configs/baseline_cpu_requirements.txt
  .venv-baselines/bin/python -m pip check
)
```

正常完成后，从新源码目录运行：

```bash
(
  set -eu
  cd /home/zhanj289/e6998/decision-study-src
  bash scripts/run_light_baseline.sh run \
    --v1-dir /home/zhanj289/e6998/decision-study/results/solo-v1 \
    --v2-dir /home/zhanj289/e6998/decision-study-src/results/solo-v2 \
    --output-dir /home/zhanj289/e6998/decision-study/results/aux-clinc-tfidf400-v1
)
```

`--v1-dir` 使用旧运行树的完整 V1 输入和 gold；`--v2-dir` 使用已在 Git 配置中校验并复制
的 prepared 数据。全程只读取这两个输入目录。新输出目录与 V1/V2/V3 原结果分离。
脚本不初始化 session、不加载 LLM，也不执行旧 pipeline。

## 分阶段运行与回传

`run` 顺序完成拟合/冻结、评价和导出。也可以用相同三个路径参数分别执行 `fit` 与
`evaluate` 检查阶段边界；分阶段完成后再执行一次 `run`，核验并复用已有结果后导出 ZIP。
输入根绝对路径也写入冻结身份，运行后不要搬动输入目录。
发现哈希不符、未收敛、已有不完整输出或身份变化时保留目录和
完整错误输出，不删除冻结文件后悄悄重跑。已有完整产物的重复调用会核验后复用。

回传新目录中的 `exports/light_baseline_bundle.zip`。它包含模型与预测身份、评价和运行
元数据；不会把原 89 GB 树复制进来。验收时仍需独立核对样本覆盖、训练/评价边界和指标。

这个 CPU 实验无需为它新申请 GPU；也不要依据旧记录把已有 allocation 当成仍有效。
