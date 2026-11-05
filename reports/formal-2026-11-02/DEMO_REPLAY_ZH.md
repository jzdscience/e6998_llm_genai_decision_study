# 正式结果的 CPU 录制回放示例

2026-11-02 已实际运行现有 `decision_study demo` CLI，核验以下四个回放。它们读取已保存的 StartLux/Qwen 预测，并用冻结的 refitted temperature 和 gate 重建路由。**这是预先记录的结果回放，不运行模型，也不是推理测速。**

这四例按已知结果类别挑选，用于说明行为与失败方式，不是随机样本或无偏评估。不能用四例的表现估计总体准确率、纠错收益或错误频率；总体结论应使用完整正式测试分析。Gold 为原数据集标签，此处未重新人工裁定。

| 示例 | Gold | StartLux | 单独记录的 Qwen | 校准后 confidence / gate | 回放路由 | 最终结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 保留本地正确结果 | `account_blocked` | `account_blocked` | `freeze_account` | 0.980706 / 0.369877 | 接受本地 | `account_blocked`（正确） |
| Qwen 救回本地错误 | `unanswerable` | `answerable` | `unanswerable` | 0.529333 / 0.656121 | 转交 Qwen | `unanswerable`（正确） |
| 转交反而造成错误 | `unanswerable` | `unanswerable` | `answerable` | 0.642751 / 0.656121 | 转交 Qwen | `answerable`（错误） |
| 高置信度错误被接受 | `answerable` | `unanswerable` | `answerable` | 0.949628 / 0.656121 | 接受本地 | `unanswerable`（错误） |

Qwen 一列取自正式 baseline 记录。本地接受的两例在回放响应中均为 `fallback: null`，并没有额外调用 Qwen；这里把 baseline 并列展示，只为解释模型间的差异。Confidence 是温度调整后的最大类别概率，不是该样本正确率的保证。

## 在本机重现

使用本项目当前 Python 环境（Python 3.11–3.13，已安装 NumPy），进入源码目录：

```bash
cd "/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study"
```

CLI 会在 `OUTPUT_DIR` 创建互斥锁，因此本示例使用独立的 `reports/formal-2026-11-02/demo-input/` 副本。副本只包含回放需要的数据、StartLux/Qwen 预测和 StartLux 校准，不是完整实验包。原始回传目录的 769 个文件在四次执行前后逐文件哈希一致。

### 1. 保留本地正确结果

样本 `clinc:test:2921:2740289483263792`（CLINC）。

> how can i unfreeze my bank account

StartLux 被 gate 接受，输出与 Gold 一致。单独记录的 Qwen 将“解除冻结”归为 `freeze_account`；本例说明一律转交不一定改善结果。

```bash
python3 -m decision_study demo \
  --output-dir reports/formal-2026-11-02/demo-input \
  --models startlux \
  --sample-id "clinc:test:2921:2740289483263792"
```

已核验的完整响应：[JSON](demo-replays/01-local_helps.json)。其中保留完整输入段落、Gold、冻结温度/gate、逐级预测以及路由状态。

### 2. Qwen 救回本地错误

样本 `squad:5a836496e60761001a2eb65a`（SQUAD）。

> What direction do cilia usually swim?

StartLux 的 confidence 低于冻结阈值，因此回放取用既有 Qwen 结果。最终标签由 `answerable` 改为与 Gold 一致的 `unanswerable`。

```bash
python3 -m decision_study demo \
  --output-dir reports/formal-2026-11-02/demo-input \
  --models startlux \
  --sample-id "squad:5a836496e60761001a2eb65a"
```

已核验的完整响应：[JSON](demo-replays/02-qwen_rescue.json)。其中保留完整输入段落、Gold、冻结温度/gate、逐级预测以及路由状态。

### 3. 转交反而造成错误

样本 `squad:5acff4b177cf76001a686616`（SQUAD）。

>  Nobody has generalized the meaning of the word imperialism down to general-purpose what?

StartLux 原本与 Gold 一致，但 confidence 略低于冻结阈值而被转交。Qwen 的 `answerable` 成为最终错误结果，说明 fallback 也会损害本地正确答案。

```bash
python3 -m decision_study demo \
  --output-dir reports/formal-2026-11-02/demo-input \
  --models startlux \
  --sample-id "squad:5acff4b177cf76001a686616"
```

已核验的完整响应：[JSON](demo-replays/03-fallback_harms.json)。其中保留完整输入段落、Gold、冻结温度/gate、逐级预测以及路由状态。

### 4. 高置信度错误被接受

样本 `squad:56de11154396321400ee25aa`（SQUAD）。

> What were the origins of the Raouliii family?

StartLux 的校准后 confidence 约 0.95，超过 gate，错误标签仍被接受。单独记录的 Qwen 正确，但这一冻结规则没有转交该例；不据此修改 gate。

```bash
python3 -m decision_study demo \
  --output-dir reports/formal-2026-11-02/demo-input \
  --models startlux \
  --sample-id "squad:56de11154396321400ee25aa"
```

已核验的完整响应：[JSON](demo-replays/04-confident_miss.json)。其中保留完整输入段落、Gold、冻结温度/gate、逐级预测以及路由状态。

## 核验与选例规则

- 四次 CLI 均正常退出，响应模式均为 `RECORDED REPLAY — no model inference executed`；接受/转交、逐级标签和最终结果与保存的正式记录一致。
- 回放移除了最终 `latency_ms`。嵌套预测内可能保留采集时的时间信息，不能将其当作此次回放延迟或真实 cascade 延迟。
- 只从主测试集选例。候选在各行为类别内按 `SHA256("6998-formal-demo-v1:" + sample_id)` 排序后取第一例；第一类限制为 CLINC 以展示银行意图输入，第四类另要求 confidence ≥ 0.9。其余两类按固定排序选取。类别本身利用了已知 Gold 和预测，因此这些例子仅供说明。
- 原始证据目录为 `results/returned-experiment-2026-11-02-101054/`；未修改冻结源码、温度、阈值或正式预测。

详细选例、实际命令与原始目录不变核验见 [selection-and-verification.json](demo-replays/selection-and-verification.json)。
