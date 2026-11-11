# Final report (ACL two-column)

First draft of the COMS E6998-007 final report, 2026-11-07, updated 2026-11-10 with the ANLI 9B serving
follow-up (Figure 5 and the last RQ3 paragraph). A second 2026-11-10 revision marks the eight unmeasured cells
of Table 2 (NM) and the status of every cell, states that Table 4 shows selected primary settings, and adds
Appendix Table 7 with every routing policy evaluated on the two sealed holdouts. `main.pdf` is the current
build (13 pages; the main text still ends on page 10, and the extra page holds the full-width appendix table).

## Build

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The source compiles on a full TeX Live or Overleaf (upload `main.tex`, `references.bib`,
`acl.sty`, `acl_natbib.bst` and `figures/*.pdf`). It also compiles on macOS BasicTeX. There,
`inconsolata` is missing and the typewriter font falls back to Latin Modern. `\usepackage{acl}`
is used without the `review` option, because review mode needs the Helvetica fonts that
BasicTeX lacks; the course report is not anonymised anyway.

`acl.sty` and `acl_natbib.bst` come from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files), commit `d5adc823ff0f`.

## Checks

`bash scripts/reproduce_cpu.sh` (from the repository root) regenerates the figures and builds this
paper in a temporary directory. With the pinned Matplotlib installed it fails if a figure differs from
the tracked PDF; `--strict` also fails when any step had to be skipped.

## Figures

```bash
.venv-lab/bin/python paper/figures/make_figures.py   # run from the repository root
```

`requirements-figures.txt` pins the Matplotlib and NumPy versions that produced the tracked PDFs;
with those versions the script reproduces them byte for byte.

The script reads only tracked, audited summaries: `reports/decision-lab-2026-11-04/study.json`,
`reports/v2-complete-2026-11-03/chart_data.json`,
`reports/followups-2026-11-06/evidence/service/aggregates.csv` and
`reports/upgrade-9b-serving-2026-11-09/summary.csv` (built from the accepted 9B audit receipts by that
report's `build_tables.py`). The Study 1 rescue and harm
counts (29 and 24 of 2,000) are hardcoded from `reports/formal-2026-11-02/RESULTS_ZH.md`.
The palette is the validated dataviz reference palette; every series is also direct-labelled
and has its own marker, so the figures survive grayscale printing. Figure 5 adds slot 6 (green) for
StartLux-9B; blue, green and yellow pass the all-pairs check on white. Its x-axis holds four categorical
groups, one per tested rate, with the deployments offset inside each group only for legibility and a
grey bar at the group's offered rate; with three rounds per rate the marker and the whisker ends are
the three measured values.

## Where the numbers come from

| Paper element | Source |
|---|---|
| Table 2, CLINC rows | `reports/cheap-baselines-2026-11-04/summary.csv`, `reports/formal-2026-11-02/RESULTS_ZH.md` |
| Table 2, SQuAD and ANLI | `reports/formal-2026-11-02`, `reports/v2-complete-2026-11-03`, `reports/v3-complete-2026-11-03`, `reports/followups-2026-11-06/size-evaluation.json` |
| Table 2 notation | NM marks the eight LLM cells with no prediction (StartLux-9B and Qwen3.5-9B on CLINC, S1 and S2; Kev-4B on S2 and ANLI). `--` marks CPU cells outside the CLINC-only baseline protocol. † marks retrospective cells (StartLux-9B on ANLI, CPU rows). The 12 measured LLM values equal the `accepted_correct` counts pinned in `configs/direct_matrix_protocol.json` |
| Appendix Table 7 (every routing policy) | Test columns: `reports/v2-complete-2026-11-03/RESULTS_ZH.md` and `reports/v3-complete-2026-11-03/RESULTS_ZH.md`. Validation deferral rates: the accepted `policy_freeze.json` of each study, outside Git (V2 `03f549a6…`, V3 `28efefe7…`, under `results/returned-v2-2026-11-03-002852/` and `results/returned-v3-2026-11-03-152529/` of the evidence root). Study 1 gate: 947 of 1,000 validation questions accepted, `reports/formal-2026-11-02/analysis/metrics.json` |
| Table 3 (primary endpoint) | `reports/formal-2026-11-02/analysis/metrics.json`, a byte-identical copy of `analysis/metrics.json` in the Study 1 bundle (SHA-256 `a6922bee…`) |
| Table 4, Figures 2–3 | `reports/decision-lab-2026-11-04/study.json`, V2/V3 reports |
| Table 5, Figure 4 | `reports/followups-2026-11-06/evidence/service/aggregates.csv`, `reports/v2-complete-2026-11-03/chart_data.json`, `reports/benchmark-2026-11-02` |
| Figure 5 and the 9B serving paragraph of RQ3, Discussion, Limitations | `reports/upgrade-9b-serving-2026-11-09/` (`summary.csv`, `quality.csv`, `resources.csv`, `evidence/`), accepted in `records/upgrade-service-acceptance-2026-11-10.json` |
| Methods, memory shares by study | `configs/runtime_defaults.json`, `configs/service_study_protocol.json`, `configs/v3_protocol.json`, `configs/size_control_protocol.json`, `configs/upgrade_study_protocol.json` |

## Open items before submission

1. Team responsibilities. Allocated on 2026-11-10 at Ju's instruction: Ju Zhang (experimental
   design, runtime and pipeline implementation, integration, final technical review), Weiyu Chen
   (statistical verification, interpretation, manuscript review), Yizhi Hu (reproducibility checks,
   demo walkthrough, presentation materials). The section states responsibilities; record each check
   with its date and result once it has actually been run.
2. Acknowledgments. Only the TA is thanked; at Ju's decision the paper has no compute acknowledgment.
3. Literature. The 2026 entries (JevCascade, StartLux report, Typed Decision Models, Kev
   card) were checked against their arXiv or Hugging Face pages on 2026-11-07; recheck the
   versions cited before submission.
4. Figures and tables use the evidence accepted through 2026-11-10, including the 9B serving
   follow-up. If new experiments run, regenerate the figures and update Tables 2–5 and Figure 5.
5. Table 2 keeps NM in its eight unmeasured cells until the direct-matrix bundle
   (`docs/RUN_DIRECT_MATRIX_ZH.md`) has run on the node, been returned and passed independent acceptance. Then
   write the eight accepted values with the † retrospective mark, keep the 12 historical values unchanged, update
   the RQ1 sentence and the Limitations paragraph that list the missing cells, and leave every Study 1–3 primary
   designation as it is.
