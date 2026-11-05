# 独立重放本次验收

脚本只审计已有完整回传；不拟合分类器、不调参、不调用生产fit/evaluate。最后一项会在
CPU上重新执行固定ONNX编码器，以比较原特征的数值稳定性，不写回原特征或评价。
不要使用 `python -O`，这些分析脚本使用断言核验身份。

需要外部证据根中的V1完整包及解压目录、V2 prepared、TF-IDF400原包、本轮原包和
训练专用prepared；布局见 `docs/ARTIFACTS_ZH.md`。模型目录需包含协议固定的八份文件。
输入路径可按自己的存储位置修改，SHA不得改变。下面从源码仓库根运行，输出新目录：

```bash
EVIDENCE_ROOT=/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study
BASELINE_ARCHIVE="$EVIDENCE_ROOT/records/bundles/cheap-baselines-20261105-073350-7f00178c7bd1.zip"
PREPARED_DIR="$EVIDENCE_ROOT/results/cheap-baselines-prepared"
MODEL_DIR="$PWD/.validation/cheap-baselines-2026-11-04/model"
AUDIT_CODE=reports/cheap-baselines-2026-11-04/audit
AUDIT_OUT=$(mktemp -d)

python3 "$AUDIT_CODE/audit_identity.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" --artifact-root "$EVIDENCE_ROOT" \
  --prepared-dir "$PREPARED_DIR" --model-dir "$MODEL_DIR" --output "$AUDIT_OUT/identity.json"
python3 "$AUDIT_CODE/audit_metrics.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" --artifact-root "$EVIDENCE_ROOT" \
  --prepared-dir "$PREPARED_DIR" --output "$AUDIT_OUT/metrics.json"
.venv-cheap-baselines/bin/python "$AUDIT_CODE/replay.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" --artifact-root "$EVIDENCE_ROOT" \
  --prepared-dir "$PREPARED_DIR" --tokenizer "$MODEL_DIR/tokenizer.json" --output "$AUDIT_OUT/replay.json"
.venv-cheap-baselines/bin/python "$AUDIT_CODE/audit_encoder.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" --prepared-dir "$PREPARED_DIR" \
  --examples "$EVIDENCE_ROOT/results/returned-experiment-2026-11-02-101054/data/examples.jsonl" \
  --model-dir "$MODEL_DIR" --output "$AUDIT_OUT/encoder.json"
```

前两项只依赖Python标准库；后两项使用独立CPU环境的NumPy/tokenizers/ORT。
核验固定release提交`1b70899…`；当前科学源码若已改动，使用原版本独立checkout。
跨机器输入路径、环境字段或浮点微差可能改变收据字节，核验以记录的数值容差和完整标签
一致性为准。ONNX重放在Mac上通过不代替原Linux耗时，也不覆盖原始矩阵hash。
