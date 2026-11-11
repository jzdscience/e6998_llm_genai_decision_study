# Documentation index

Most documents are in Chinese (`_ZH` suffix). Files marked **bound** are hashed by
`SOURCE_SHA256.json` or a V2/V3 release manifest and are checked by the release verifiers, so they
keep their path and bytes. Study 1, 2 and 3 in the report correspond to V1, V2 and V3 here.

## Start here

| Document | Content |
|---|---|
| [START_HERE_ZH.md](START_HERE_ZH.md) | Reading order, demo commands and the presentation entry (Chinese) |
| [CURRENT_STATE.md](CURRENT_STATE.md) | Current status, newest section first |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Chronological log of every run, failure, repair and acceptance |
| [ARTIFACTS_ZH.md](ARTIFACTS_ZH.md) | What lives outside Git (bundles, prepared data, weights) and how to restore it |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Failures met during the project and their fixes (English) |
| [ORIGINAL_PROPOSAL.md](ORIGINAL_PROPOSAL.md) | The submitted proposal, 26 Oct 2026 (English) |

## Protocols, frozen before each run

| Document | Study |
|---|---|
| [data_protocol.md](data_protocol.md) **bound** | V1 data splits, hashing and deduplication |
| [metrics.md](metrics.md) **bound** | V1 metric definitions |
| [runtime.md](runtime.md) **bound** | V1 runtime profiles and environments |
| [V2_PROTOCOL_ZH.md](V2_PROTOCOL_ZH.md) **bound** | V2: routers on a fresh SQuAD holdout, open-loop serving |
| [V3_PROTOCOL_ZH.md](V3_PROTOCOL_ZH.md) **bound** | V3: ANLI with Qwen3.5 4B and 9B fallbacks |
| [LIGHT_BASELINE_PROTOCOL_ZH.md](LIGHT_BASELINE_PROTOCOL_ZH.md) | TF-IDF with 400 CLINC labels |
| [CHEAP_BASELINES_PROTOCOL_ZH.md](CHEAP_BASELINES_PROTOCOL_ZH.md) | TF-IDF-1600, MiniLM-400 and MiniLM-1600 |
| [SERVICE_STUDY_PROTOCOL_ZH.md](SERVICE_STUDY_PROTOCOL_ZH.md) | CPU/GPU service study, 500 ms SLO |
| [SIZE_CONTROL_PROTOCOL_ZH.md](SIZE_CONTROL_PROTOCOL_ZH.md) | StartLux-9B size control on ANLI |
| [UPGRADE_STUDY_PROTOCOL_ZH.md](UPGRADE_STUDY_PROTOCOL_ZH.md) | Serving cost of StartLux-9B versus a 4B to Qwen3.5-9B cascade on ANLI; revision 4, preflight and 36 formal cells accepted on 2026-11-10 ([report](../reports/upgrade-9b-serving-2026-11-09/RESULTS_ZH.md)) |
| [DIRECT_MATRIX_PROTOCOL_ZH.md](DIRECT_MATRIX_PROTOCOL_ZH.md) | Retrospective completion of the five-LLM direct-accuracy matrix: 8 new cells, 20,400 test predictions, 12 accepted cells reused (`configs/direct_matrix_protocol.json`, revision 1). Local acceptance passed; the GPU run is pending |

## Runbooks

| Document | Use |
|---|---|
| [QUICKSTART_ZH.md](QUICKSTART_ZH.md) **bound** | V1 setup and execution |
| [RECOVER_KEV_ZH.md](RECOVER_KEV_ZH.md) **bound** | V1 Kev recovery |
| [CONTINUE_V2_ZH.md](CONTINUE_V2_ZH.md) **bound** | Running V2 on the node |
| [RUN_V3_ZH.md](RUN_V3_ZH.md) **bound** | Running V3 on the node |
| [RUN_LIGHT_BASELINE_ZH.md](RUN_LIGHT_BASELINE_ZH.md) | Running the TF-IDF-400 baseline |
| [RUN_CHEAP_BASELINES_ZH.md](RUN_CHEAP_BASELINES_ZH.md) | Running the TF-IDF-1600 and MiniLM baselines |
| [RUN_FOLLOWUPS_ZH.md](RUN_FOLLOWUPS_ZH.md) | Running the service study and the size control together |
| [RUN_SIZE_CONTROL_ZH.md](RUN_SIZE_CONTROL_ZH.md) | Running the size control alone |
| [RUN_UPGRADE_STUDY_ZH.md](RUN_UPGRADE_STUDY_ZH.md) | Running the StartLux-9B serving-cost study: development preflight, then 36 formal cells |
| [RUN_DIRECT_MATRIX_ZH.md](RUN_DIRECT_MATRIX_ZH.md) | Running the direct-accuracy matrix completion on the node (one command with automatic preflight; `--preflight-only`, `--resume`, `--new-allocation`) and the offline audit of the returned bundle |
| [NODE_GIT_ZH.md](NODE_GIT_ZH.md) | Preparing the node from a pinned Git checkout |
| [REPO_MANAGEMENT_ZH.md](REPO_MANAGEMENT_ZH.md) | How this repository, the node and the evidence root relate |
| [LAB.md](LAB.md) **bound** | The V1 Decision Lab |
| [LAB_CURRENT_ZH.md](LAB_CURRENT_ZH.md) | The current Decision Lab (V1–V3 and the follow-up page) and demo order |
| [PRESENTATION_ZH.md](PRESENTATION_ZH.md) | Presentation outline, speaker notes, demo clicks and selected examples |

## Reviews

| Document | Content |
|---|---|
| [DIRECT_MATRIX_ACCEPTANCE_2026-11-10_ZH.md](DIRECT_MATRIX_ACCEPTANCE_2026-11-10_ZH.md) | Codex acceptance of the matrix-completion software, recovery tests and paper tables; 8 new GPU cells still unrun |
| [REVIEW_2026-11-08_ZH.md](REVIEW_2026-11-08_ZH.md) | Codex review of the analysis, paper draft and repository reorganization (R1–R5) |
| [UPGRADE_STUDY_REVIEW_2026-11-08_ZH.md](UPGRADE_STUDY_REVIEW_2026-11-08_ZH.md) | Codex review of the serving-cost study design (U1–U5) and its acceptance checklist |
| [UPGRADE_IMPLEMENTATION_REVIEW_2026-11-08_ZH.md](UPGRADE_IMPLEMENTATION_REVIEW_2026-11-08_ZH.md) | Codex cross-check of the serving-cost implementation (I1–I4) |
| [UPGRADE_FIX_ACCEPTANCE_2026-11-08_ZH.md](UPGRADE_FIX_ACCEPTANCE_2026-11-08_ZH.md) | Codex acceptance of the I1–I4 fixes; cleared for the node preflight |
| [UPGRADE_REVISION4_REVIEW_2026-11-08_ZH.md](UPGRADE_REVISION4_REVIEW_2026-11-08_ZH.md) | First node preflight audit and Codex review of revision 4 (engine sequence limit); cleared for a new preflight |
| [UPGRADE_RESULTS_REVIEW_2026-11-10_ZH.md](UPGRADE_RESULTS_REVIEW_2026-11-10_ZH.md) | Independent acceptance of the revision 4 preflight and the 36 formal 9B serving cells, with the interpretation corrections |

## History (`history/`)

Superseded runbooks and planning notes, moved here on 2026-11-07. Their content is unchanged
apart from relative links.

| Document | Content |
|---|---|
| [RECOVER_STARTLUX_ZH.md](history/RECOVER_STARTLUX_ZH.md), [RECOVER_QWEN_TOOLS_ZH.md](history/RECOVER_QWEN_TOOLS_ZH.md), [RECOVER_QWEN_CUDA_ZH.md](history/RECOVER_QWEN_CUDA_ZH.md), [RECOVER_QWEN_CUDA_V2_ZH.md](history/RECOVER_QWEN_CUDA_V2_ZH.md), [RECOVER_QWEN_CUDA_V3_ZH.md](history/RECOVER_QWEN_CUDA_V3_ZH.md) | Node repairs of 1–2 Nov; receipts in `records/patches/` |
| [LEGACY_V1_README.md](history/LEGACY_V1_README.md) | The original V1 README |
| [FINISH_PROJECT_ZH.md](history/FINISH_PROJECT_ZH.md) | Course delivery checklist after V1 (2 Nov) |
| [EXTENSION_ROADMAP_ZH.md](history/EXTENSION_ROADMAP_ZH.md) | Extension and front-end roadmap (2 Nov) |
| [CLAUDE_V2_REVIEW_ZH.md](history/CLAUDE_V2_REVIEW_ZH.md) | Review of an alternative V2 plan (2 Nov) |
| [MONTH_SCOPE_ZH.md](history/MONTH_SCOPE_ZH.md) | Scope for a month of single-GPU time (3 Nov) |
| [POST_V3_SCOPE_ZH.md](history/POST_V3_SCOPE_ZH.md) | Post-V3 options compared with the original plan (4 Nov) |
| [NEXT_STEPS_2026-11-06_ZH.md](history/NEXT_STEPS_2026-11-06_ZH.md) | Remaining-work options before the follow-up analysis (6 Nov) |
| [EXPERIMENT_LOG_PRE_REPO_ZH.md](history/EXPERIMENT_LOG_PRE_REPO_ZH.md) | Full log from 30 Oct to 4 Nov, before this repository took over |
| [CURRENT_STATE_PRE_REPO_ZH.md](history/CURRENT_STATE_PRE_REPO_ZH.md) | Last status page of the pre-repository project tree (4 Nov) |

The two `PRE_REPO` files were copied on 2026-11-07 byte for byte from the evidence root's `docs/`
(SHA-256 `39adf7bc…` and `38cac56a…`). Their relative links are written for that tree, as in the
historical reports, so links to `records/`, `results/` and `.validation/` resolve there.
