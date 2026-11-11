# When Can Open Decision Models Replace Generative LLM Calls?

COMS E6998-007, Fall 2026, final project. Ju Zhang, Weiyu Chen, Yizhi Hu (Columbia University).
中文说明见 [README_ZH.md](README_ZH.md).

Decision models score a fixed set of candidate labels in one forward pass and return a probability
distribution. This repository evaluates them as replacements for generative LLM calls on
classification-shaped tasks (CLINC150 banking intents with out-of-scope queries, SQuAD 2.0
answerability, ANLI), and evaluates cascades that defer uncertain inputs to a generative LLM. It
holds the evaluation harness, frozen protocols, audited results, a CPU-only demo and the report.

The report is [paper/main.pdf](paper/main.pdf) (ACL format, first draft). A Chinese guide to the reading order,
the demo and the presentation is [docs/START_HERE_ZH.md](docs/START_HERE_ZH.md).

## Results at a glance

| Question | Finding |
|---|---|
| Can a decision model make the call directly? | StartLux-4B beats Qwen3.5-4B on CLINC (93.50% vs. 88.00%), SQuAD answerability (91.25% vs. 86.00%) and ANLI (74.19% vs. 55.41%), at 1.8–1.9× the throughput on one L40S. |
| Does deferring to a generative model help? | A gain-aware router trailed confidence routing on a fresh 4,000-question holdout (−0.225 pp, 95% CI −0.91 to 0.48). On ANLI the router fitted on validation deferred nothing. |
| Do validation operating points hold on test? | The 5% selective-risk target was missed for three of four front-model/task pairs, and a 10% deferral budget became 23.2% on ANLI. |
| What changes under load? | At 100 requests/s with a 500 ms SLO, a CPU MiniLM classifier (93.17% on CLINC) returned 93.4 on-time correct answers per second. Under our client and vLLM settings the GPU services completed about 65 (StartLux) and 47 (Qwen) requests per second, almost all late. |
| A bigger front model or a fallback? | StartLux-9B adds 2.50 points on ANLI (95% CI 1.34–3.81), more than any of the frozen cascades we evaluated with a 9B generative fallback. In a follow-up on one L40S at 10–80 requests/s, it returned more correct answers within 500 ms than the most accurate of those cascades at every tested load, with a 0.50 GPU memory share against the cascade's 0.90; at 80 requests/s StartLux-4B alone returned the most (38.3 per second). See [reports/upgrade-9b-serving-2026-11-09](reports/upgrade-9b-serving-2026-11-09/RESULTS_ZH.md). |

Eight cells of the five-LLM direct-accuracy matrix (paper Table 2, marked NM) have never been measured:
StartLux-9B and Qwen3.5-9B on CLINC and both SQuAD splits, and Kev-4B on the Study 2 SQuAD holdout and ANLI. A
retrospective protocol that fills them with fixed models and prompts has passed
[local software and paper acceptance](docs/DIRECT_MATRIX_ACCEPTANCE_2026-11-10_ZH.md); it has not run on the GPU,
so no new accuracy exists yet
([docs/DIRECT_MATRIX_PROTOCOL_ZH.md](docs/DIRECT_MATRIX_PROTOCOL_ZH.md)).

## Repository layout

```text
decision_study/   Python package: V1 pipeline (adapters, runner, runtime, storage; hash-frozen),
                  v2/, v3/, baselines/, cheap_baselines/, service_study/, size_control/,
                  upgrade_study/ (9B serving follow-up), direct_matrix/ (direct-accuracy matrix completion;
                  its GPU run is pending), lab/ (CPU Decision Lab)
configs/          frozen protocol and runtime configurations
scripts/          entry points (see scripts/README.md)
tests/            CPU test suite
reports/          accepted result reports, figures and audit receipts, one folder per return
records/          bundle index, acceptance and preparation receipts, patch receipts
docs/             protocols, runbooks and project state (see docs/README.md)
paper/            report source, figure script and build notes
notebooks/        V1 analysis and run notebooks
*_MANIFEST.json, SOURCE_SHA256.json, VALIDATION.md
                  release identities read by the verifiers and the packager; keep them at the root
```

## Quick start (CPU only)

Python 3.11–3.13. No GPU, model weights or raw results are needed.

```bash
git clone https://github.com/jzdscience/e6998_llm_genai_decision_study.git
cd e6998_llm_genai_decision_study
python3 -m venv .venv-lab
.venv-lab/bin/python -m pip install -e '.[lab,test]'
.venv-lab/bin/python -m pytest -q        # CPU tests
bash scripts/run_lab.sh                   # Decision Lab at http://127.0.0.1:7860
```

The Decision Lab replays recorded cases from Studies 1–3 and shows the measured follow-ups (CPU
baselines, CPU/GPU serving, 9B serving) using the portable evidence in `reports/decision-lab-*`; it never
calls a model. A good tour is Overview, then V2 · SQuAD, then V3 · ANLI, then Follow-ups, then V1 replay.
Use `bash scripts/run_lab.sh --port 7861` if port 7860 is taken, and Ctrl-C to stop it.

## Reproducing the reported numbers

```bash
bash scripts/reproduce_cpu.sh
```

The script runs seven checks and writes everything to a temporary directory, leaving tracked files
untouched. It takes a few minutes on a laptop. Steps whose inputs are missing are reported as SKIP;
`bash scripts/reproduce_cpu.sh --strict` treats any skip as a failure, for a full acceptance run, and
`--only <step>` runs single steps (`tests`, `lab`, `audits-1106`, `audits-9b`, `report-9b`, `figures`, `paper`).

1. The CPU test suite. The CLINC baseline tests need scikit-learn and ONNX Runtime in a separate
   pinned environment; the script uses `.venv-cheap-baselines` when it exists:
   ```bash
   python3 -m venv .venv-cheap-baselines
   .venv-cheap-baselines/bin/python -m pip install -r configs/cheap_baseline_requirements.txt pytest
   ```
2. The Decision Lab evidence check (hashes and semantic guards of all portable V1–V3 and follow-up evidence).
3. A rerun of the audits for the service study and the StartLux-9B size control, compared byte for
   byte with the receipts in `reports/followups-2026-11-06/evidence/`. This step reads the raw result
   bundles, which live outside Git; point `EVIDENCE_ROOT` at them, or the step is skipped.
4. A rerun of the 9B serving audits (2,097 integrity checks and the raw metric replay of all 36 cells)
   and a rebuild of the Lab follow-up evidence, compared byte for byte with
   `reports/upgrade-9b-serving-2026-11-09/evidence/` and `reports/decision-lab-followups-2026-11-10/`.
   It also needs the raw bundles and a git clone (the integrity audit checks source files at commit `7c81947`).
5. The 9B serving report tables, rebuilt from the tracked receipts and compared byte for byte, and its figure.
6. The paper figures, regenerated from tracked data and compared byte for byte with
   `paper/figures/*.pdf`. Byte equality needs the versions in `paper/requirements-figures.txt`;
   with other versions the figures are regenerated and the comparison is skipped.
7. The paper PDF, built in a temporary copy (needs `pdflatex` and `bibtex`).

Raw bundles, prepared data, model weights and environments stay out of Git.
[records/BUNDLE_INDEX.json](records/BUNDLE_INDEX.json) lists every accepted bundle with its SHA-256,
and [docs/ARTIFACTS_ZH.md](docs/ARTIFACTS_ZH.md) explains the layout and how to restore it. Each
report folder under `reports/` names its source bundle and the scripts that re-derive its numbers.

## Running the GPU experiments

The GPU studies ran on one NVIDIA L40S (48 GB) with 100 GB of host RAM in a Slurm allocation,
driven from a code-server terminal on the node. The node runs a clean checkout pinned to one commit
and reuses an existing runtime directory that holds the environments and weights:

```bash
git clone https://github.com/jzdscience/e6998_llm_genai_decision_study.git decision-study-src
cd decision-study-src && git checkout --detach <commit>
<runtime-root>/.venv-controller/bin/python scripts/prepare_node_checkout.py \
  --runtime-root <runtime-root> --expected-commit <commit>
```

| Study | Entry point | Runbook |
|---|---|---|
| Study 1: three 4B models on CLINC and SQuAD, closed-loop serving | `bash scripts/run_with_cuda_tools.sh {smoke,startlux_full,qwen_test,benchmark}`, then `python -m decision_study {evaluate,export}` | [docs/QUICKSTART_ZH.md](docs/QUICKSTART_ZH.md) |
| Study 2: routing on a fresh SQuAD holdout, open-loop serving | `bash scripts/run_v2.sh pipeline --v1-dir ... --output-dir ...` | [docs/CONTINUE_V2_ZH.md](docs/CONTINUE_V2_ZH.md) |
| Study 3: ANLI with 4B and 9B fallbacks | `bash scripts/run_v3.sh start` | [docs/RUN_V3_ZH.md](docs/RUN_V3_ZH.md) |
| CLINC CPU baselines | `bash scripts/run_cheap_baseline_suite.sh <runtime-root>` | [docs/RUN_CHEAP_BASELINES_ZH.md](docs/RUN_CHEAP_BASELINES_ZH.md) |
| CPU/GPU service study and StartLux-9B size control | `bash scripts/run_followup_suite.sh <runtime-root> [--new-allocation]` | [docs/RUN_FOLLOWUPS_ZH.md](docs/RUN_FOLLOWUPS_ZH.md) |
| 9B serving follow-up: StartLux-4B, StartLux-9B and a 4B → Qwen3.5-9B cascade on ANLI | `bash scripts/run_upgrade_study.sh <runtime-root> --preflight-only`, then without the flag | [docs/RUN_UPGRADE_STUDY_ZH.md](docs/RUN_UPGRADE_STUDY_ZH.md) |
| Direct-accuracy matrix completion: the 8 unmeasured cells of Table 2, 20,400 test predictions (prepared locally; GPU run pending) | `bash scripts/run_direct_matrix.sh <runtime-root>` (preflight, collection, evaluation and export; `--preflight-only`, `--resume`, `--new-allocation`) | [docs/RUN_DIRECT_MATRIX_ZH.md](docs/RUN_DIRECT_MATRIX_ZH.md) |

Every run exports a result bundle. A result is reported only after the bundle has been archived by
SHA-256, re-scored by an independent audit script and added to the bundle index. Policies and
thresholds are frozen before holdout predictions, and holdout gold stays sealed until evaluation.

## Troubleshooting

[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) covers the failures met during the project: Slurm
out-of-memory kills, session deadlines, CUDA toolchain mismatches, port checks, iCloud placeholder
files, git lock files, oversized result bundles, zsh glob errors and LaTeX on BasicTeX.

## Documentation

[docs/README.md](docs/README.md) indexes the protocols, runbooks and history. Most protocol and runbook
documents are written in Chinese. [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md) is the current
project status and [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) the full experiment log.
