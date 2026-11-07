# 重跑 11/6 两项回传的审计

脚本只读已有回传，不拟合、不调参、不启动模型或服务，也不写回原证据。前四个脚本与
`.validation/followups-return-2026-11-06/` 中 11/6 使用的版本逐字节相同；`recheck_from_raw.py`
是 11/7 另写的标准库复算。四个原脚本用断言核验身份，不要用 `python -O` 运行。

需要外部证据根中的两份 SHA 命名归档及其解压目录、V1 正式包、轻量基线包、V3 回传包，以及
V1 的 CLINC gold，布局见 [ARTIFACTS_ZH.md](../../../docs/ARTIFACTS_ZH.md)。`audit_service.py`
需要 numpy，可用 `.venv-cheap-baselines`。从源码仓库根运行，输出写入新的临时目录：

```bash
EVIDENCE_ROOT=/Users/zhanj289/Desktop/COMS6998/project/solo-experiments-2026-10-30/decision-study
SERVICE_ZIP="$EVIDENCE_ROOT/records/bundles/service-complete-20261106-d212c130c707.zip"
SIZE_ZIP="$EVIDENCE_ROOT/records/bundles/size-control-complete-20261106-f427f7762bac.zip"
V3_ZIP="$EVIDENCE_ROOT/records/bundles/v3-return-20261103-232529-a8d0dc672d83.zip"
A=reports/followups-2026-11-06/audit
PY=.venv-cheap-baselines/bin/python
OUT=$(mktemp -d)/rerun

$PY $A/audit_service.py --archive "$SERVICE_ZIP" --evidence-root "$EVIDENCE_ROOT" --out "$OUT/service"
$PY $A/audit_supplement.py --archive "$SERVICE_ZIP" --evidence-root "$EVIDENCE_ROOT" --out "$OUT/service-supplement"
$PY $A/audit_service_runtime.py --repo-root "$PWD" --evidence-root "$EVIDENCE_ROOT" \
  --result-root "$EVIDENCE_ROOT/results/returned-service-complete-2026-11-06" --out "$OUT/runtime"
$PY $A/audit_size_control.py --bundle "$SIZE_ZIP" --v3-bundle "$V3_ZIP" --repo "$PWD" --out "$OUT/size"
python3 -I $A/recheck_from_raw.py \
  --service-root "$EVIDENCE_ROOT/results/returned-service-complete-2026-11-06" \
  --clinc-gold "$EVIDENCE_ROOT/results/returned-kev-2026-11-01/data/gold.json" \
  --size-root "$EVIDENCE_ROOT/results/returned-size-control-complete-2026-11-06" \
  --out "$OUT/raw-recheck.json"
```

11/7 在本机按上面的命令运行，五个脚本合计约 30 秒。四个原脚本的 14 份输出中，13 份与
`../evidence/` 和 `.validation/` 中的原收据逐字节一致；`service/metrics-audit.json` 只有
`archive` 路径字段不同，因为这里传入的是 SHA 命名的归档，11/6 用的是桌面上的同 SHA 原文件。
`recheck_from_raw.py` 的输出与 `../evidence/raw-recheck-2026-11-07.json` 相同，状态为 PASS。

`recheck_from_raw.py` 的 bootstrap 区间使用 Python 标准随机数生成器，与项目用 numpy 得到的
区间会有小幅差别；点估计和计数应完全相同。
