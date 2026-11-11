# 重跑 9B 服务研究的独立审计

两个脚本只用 Python 标准库，直接读取原始 ZIP 与密封 gold，不导入项目的评价、核验或路由代码，不运行模型，
不拟合，也不写回任何输入。`audit_metrics.py` 的检查是 `assert` 语句，两个脚本在 `python -O` 下都会拒绝运行。

| 脚本 | 输入 | 输出 |
|---|---|---|
| `audit_integrity.py` | 正式包、预检包、V3 回传包（按 `records/BUNDLE_INDEX.json` 的相对路径），以及包含 commit `7c81947` 的 git checkout | `integrity-audit.json`（2,097 项检查） |
| `audit_metrics.py` | 正式包、V3 密封 gold、V3 回传包、规模对照包 | `audit-receipt.json`、`recomputed.json`、`cells.csv`、`stall-analysis.json`、`stall_late_rows.csv` |

## 运行

外部证据根的布局见 [ARTIFACTS_ZH.md](../../../docs/ARTIFACTS_ZH.md)。在源码仓库根执行，输出写入一个新的临时
目录：

```bash
export EVIDENCE_ROOT=/path/to/decision-study-evidence-root   # 含 records/bundles/ 与 results/solo-v3-prepared/
A=reports/upgrade-9b-serving-2026-11-09/audit
OUT=$(mktemp -d)
python3 -I $A/audit_integrity.py --repo-root "$PWD" --evidence-root "$EVIDENCE_ROOT" --output "$OUT/integrity"
python3 -I $A/audit_metrics.py --evidence-root "$EVIDENCE_ROOT" --output "$OUT/metrics"
```

`audit_metrics.py` 也可以不设环境变量，逐个传入 `--bundle`、`--gold`、`--v3-bundle`、`--size-bundle`。
`audit_integrity.py` 没有这些选项，它按 `records/BUNDLE_INDEX.json` 的相对路径从 `--evidence-root`（或
`EVIDENCE_ROOT`）读取三个 ZIP，所以需要完整的证据根布局。两者在任何检查失败时以非零状态退出。`audit_integrity.py` 用 `git show` 核对 36 份源码在 `7c81947` 的哈希，因此需要完整的 git
历史；不带 `.git` 的源码快照无法运行这一项。

`scripts/reproduce_cpu.sh` 的第 4 步运行同样的两条命令，再把 6 份输出与 `../evidence/` 中的收据逐字节比较；
`--strict` 下任一输入缺失即失败。11/10 在本机运行时，完整性审计约 3 秒，指标重放约 8 秒。

## 与验收版本的关系

验收收据 [upgrade-service-acceptance-2026-11-10.json](../../../records/upgrade-service-acceptance-2026-11-10.json)
记录了验收时脚本与输出的 SHA-256（当时位于本机忽略的 `.validation/upgrade-return-review-2026-11-10/`）。

| 文件 | 验收版本 SHA-256 |
|---|---|
| `integrity/audit_integrity.py` | `27a59de292b99b3aff11ba3e6c376a28caf199db91c4cf96f78a051863c0691c` |
| `metrics/audit_metrics.py` | `d0431f3090c33aa92c32070221ac3ccd739e117bd989affc15f7f40594a052ad` |

验收版本从脚本所在位置推断仓库根，并把证据根写死为仓库旁的固定相对路径，输出写在脚本旁边。这里的版本
改为用参数或 `EVIDENCE_ROOT` 指定输入、用必填的 `--output` 指定输出，并加了 `-O` 保护；检查与计算的代码
没有改动。完整差异见 [CHANGES_FROM_ACCEPTED.diff](CHANGES_FROM_ACCEPTED.diff)。

2026-11-10 在本机用这两个便携脚本重跑，6 份输出与验收收据逐字节一致：

| 输出 | SHA-256 |
|---|---|
| `integrity-audit.json` | `99b1d5c0451eced1a09d97e15ecd4322a43238bf38d2b2f0efd19f56c53af4be` |
| `audit-receipt.json` | `e4d0dcc4ed75b5d485c82ee07708f64c3247a30495cd646dcd49e52698452896` |
| `recomputed.json` | `812493682b545e6f98ef439b4ec541f8d43a3e2901db5a9baf16a7ef4eb1f8d7` |
| `cells.csv` | `aa60854cf91dd4319abc9c2c1479b41bf155a80553799df061974da26d2eac3a` |
| `stall-analysis.json` | `336e8533b4c3019b07c85dcc2ec63b30a8efc8ffbaa72f84501eb088a76fcaf5` |
| `stall_late_rows.csv` | `6419eb0d160e4d75d66c0478af5e3b844d8690df74574b8b716293ac2df3b6b6` |

`integrity-audit.json` 中的包身份只取 BUNDLE_INDEX 条目的不可变字段（id、路径、SHA、字节数、源码 commit、
协议版本、成员数与时间），复审备注的修改不会改变输出。
