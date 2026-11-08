# Final report (ACL two-column)

First draft of the COMS E6998-007 final report, 2026-11-07. `main.pdf` is the current build.

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

## Figures

```bash
.venv-lab/bin/python paper/figures/make_figures.py   # run from the repository root
```

The script reads only tracked, audited summaries: `reports/decision-lab-2026-11-04/study.json`,
`reports/v2-complete-2026-11-03/chart_data.json` and
`reports/followups-2026-11-06/evidence/service/aggregates.csv`. The Study 1 rescue and harm
counts (29 and 24 of 2,000) are hardcoded from `reports/formal-2026-11-02/RESULTS_ZH.md`.
The palette is the validated dataviz reference palette; every series is also direct-labelled
and has its own marker, so the figures survive grayscale printing.

## Where the numbers come from

| Paper element | Source |
|---|---|
| Table 2, CLINC rows | `reports/cheap-baselines-2026-11-04/summary.csv`, `reports/formal-2026-11-02/RESULTS_ZH.md` |
| Table 2, SQuAD and ANLI | `reports/formal-2026-11-02`, `reports/v2-complete-2026-11-03`, `reports/v3-complete-2026-11-03`, `reports/followups-2026-11-06/size-evaluation.json` |
| Table 3 (primary endpoint) | Study 1 `analysis/metrics.json` in the external evidence root (`results/returned-experiment-2026-11-02-101054`) |
| Table 4, Figures 2–3 | `reports/decision-lab-2026-11-04/study.json`, V2/V3 reports |
| Table 5, Figure 4 | `reports/followups-2026-11-06/evidence/service/aggregates.csv`, `reports/v2-complete-2026-11-03/chart_data.json`, `reports/benchmark-2026-11-02` |

## Open items before submission

1. Contributions. Lines in orange (`\proposed{...}`) are proposed roles for Weiyu and Yizhi
   on the remaining deliverables. Keep each line only after that person has done the work,
   edit it to match, then remove the colour.
2. Acknowledgments. Decide whether and how to name the compute provider (`% TODO(Ju)`).
3. Literature. The 2026 entries (JevCascade, StartLux report, Typed Decision Models, Kev
   card) were checked against their arXiv or Hugging Face pages on 2026-11-07; recheck the
   versions cited before submission.
4. Figures and tables use the evidence as of the 11/6 follow-ups. If new experiments run,
   regenerate the figures and update Tables 2–5.
