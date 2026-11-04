# Frozen data protocol: solo-v1

The seed string is `6998-solo-v1`. Source bytes are SHA256-verified before JSON
parsing. `decision_study/data.py` contains the four official URLs and their
verified hashes. Source changes fail closed; do not bypass these checks to run a
different dataset under the same experiment identity.

## Model inputs and gold

`data/examples.jsonl` contains ordered label IDs, shared label descriptions,
task instructions, raw visible text, source IDs and grouping metadata. Gold
labels are stored only in `data/gold.json`; they are not sent to model adapters.
`data/manifest.json` is written last and serves as a completion marker. It
contains source hashes, file hashes, deterministic dataset fingerprint, counts,
article assignments and the exposure limitation. `data/exclusions.jsonl` gives
every removed duplicate/conflicting example and its reason.

Formal records total **6,250 distinct inputs**. Kev and StartLux each use every
formal record. Qwen uses only `test` and `stress`, totaling **3,450**. There are
also **128 development fixtures**, 64 per task, excluded from formal results.

## CLINC banking + OOS

The official `data_full.json` and `domains.json` define fifteen banking labels.
The labels are sorted lexicographically, followed by `oos`. The sequence in each
record's `labels` list is authoritative; never infer option order by sorting a
probability dictionary.

| Role | Banking | OOS | Total |
|---|---:|---:|---:|
| Calibration | 20 per intent = 300 | 100 official oos_train | 400 |
| Validation | 20 per intent = 300 | 100 official oos_val | 400 |
| Main test | 30 per intent = 450 | 150 official oos_test | 600 |
| Additional stress inputs | 0 | Remaining 850 oos_test | 850 |

IDs include original source split, source row index, and a text hash. Within each
banking class, calibration is selected by sorting
`sha256(seed + '|cal|' + sample_id)`. Main OOS test uses role `test`; all remaining
OOS records become `stress`. Development uses four unused training inputs per
banking class plus four clearly labeled, authored OOS interface fixtures.
No formal OOS records are consumed for prompt development.

The stress **workload** is the 450 main-test banking inputs plus 850 additional
OOS inputs (1,300 total). The stored `stress` partition contains only the 850 new
inputs to avoid duplicate inference. Main/stress share banking observations and
have different OOS samples; their difference is not a pure causal prevalence
effect. The fixed-pool prevalence curve reweights all 450 banking and 1,000 OOS
test inputs, without another model call.

## SQuAD answerability

Gold is `unanswerable` iff `is_impossible` is true. Inputs preserve the complete
raw passage and question. Answer spans, plausible answers, and `is_impossible`
are never included in model-visible input. This is answerability classification,
not extractive QA; do not report SQuAD answer EM/F1.

Deduplication keys apply NFKC, case folding and whitespace collapse separately
to `(context, question)`. Only keys are normalized, not model inputs. Same-label
groups keep the smallest question ID. Conflicting-label groups are removed in
their entirety; no heuristic chooses the supposedly correct label.

Frozen-source audit:

| Source | Raw questions | Retained after input deduplication/conflict removal |
|---|---:|---:|
| train-v2.0 | 130,319 | 130,183 |
| dev-v2.0 | 11,873 | 11,852 |

The 442 train titles are ordered by `sha256(seed + '|article|' + title)`. The first
88 titles form development, the next 176 calibration, and the remaining 178
validation. Calibration and validation each select 500 questions per class by
`sha256(seed + '|' + role + '|' + qid)`. Official dev supplies 1,000 questions per
class to project test using role `test`. This is not the hidden SQuAD test set.

Development selects 32 per class: the two longest raw context+question pairs and
30 additional hash-selected pairs. It stays article-disjoint from the other
roles. Source validation rejects contexts shared across article titles and
train/dev overlap. The resulting test set covers the 35 official dev articles;
bootstrap units are articles rather than individual questions.

## Input length and benchmark selection

Data preparation does not tokenize, truncate, or substitute shorter examples.
The runtime performs a separate whole-manifest check using each actual renderer
and tokenizer, including instructions, options and output reservation. An
overlength input blocks the configuration and records IDs and lengths.

Performance workloads are selected from main test using role `benchmark`:
CLINC takes ten per banking label plus fifty OOS; SQuAD takes 100 per class.
The fixed execution order uses role `benchmark-order`. Selection never reads
model predictions.

## Interpretation

These controls prevent project-level development/test leakage. They do not prove
that released models were never exposed to CLINC or SQuAD upstream. Keep the
proposal's model-specific exposure caveats in the final report.

Sources: [CLINC](https://github.com/clinc/oos-eval),
[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
