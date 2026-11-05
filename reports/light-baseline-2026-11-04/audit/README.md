# 重放本次验收

先按 `docs/ARTIFACTS_ZH.md` 恢复外部 evidence root；需要本轮原 ZIP、V1 完整原 ZIP及
其解压目录、V2 prepared 文件。Git clone 自带报告和脚本，不包含这些输入。
在源码仓库根目录运行，`EVIDENCE_ROOT` 改为自己的存储根。不要使用 `python -O`，
这些独立分析脚本使用断言检查身份。

```bash
EVIDENCE_ROOT=/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study
BASELINE_ARCHIVE="$EVIDENCE_ROOT/records/bundles/light-baseline-20261105-064105-7fcd28785a06.zip"
AUDIT_OUT=$(mktemp -d)
AUDIT_CODE=reports/light-baseline-2026-11-04/audit

python3 "$AUDIT_CODE/audit_identity.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" \
  --artifact-root "$EVIDENCE_ROOT" --output "$AUDIT_OUT/identity-audit.json"
python3 "$AUDIT_CODE/audit_metrics.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" \
  --artifact-root "$EVIDENCE_ROOT" --output "$AUDIT_OUT/metrics-audit.json"
.venv-baselines/bin/python "$AUDIT_CODE/replay.py" \
  --bundle "$BASELINE_ARCHIVE" --repo "$PWD" \
  --examples "$EVIDENCE_ROOT/results/returned-experiment-2026-11-02-101054/data/examples.jsonl" \
  --output "$AUDIT_OUT/replay-audit.json"
```

前两项只依赖 Python 标准库；最后一项使用 NumPy，已在独立 baseline 环境运行。
脚本不训练分类器、不启动模型服务、不调用项目的 fit/evaluate。输出到新临时目录，
不覆盖原回传或仓库收据。源版本固定为 `797aff1…`；后续若修改 baseline 科学源码或
协议，身份核验会拒绝当前文件，应使用包含原版本的独立 checkout。

跨平台重放以标签全部相同、概率最大绝对误差不超过 `1e-12` 为核验条件；机器环境字段、
本地输入路径和最小浮点误差可随重放环境变化，不要求跨机器 JSON 字节相同。
