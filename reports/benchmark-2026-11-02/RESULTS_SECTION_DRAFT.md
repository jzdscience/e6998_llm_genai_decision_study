# Results section draft

This is an evidence-backed writing draft, not a submitted course report. Formal predictive
quality and serving measurements use different sample scopes, as specified below.

## Selective risk and direct predictive quality

The frozen validation-selected gates did not consistently retain their 5% empirical local-error
target on the main test sets. With refitted probabilities, StartLux accepted 599 of 600 CLINC
inputs (99.83% coverage), with 38 errors among accepted inputs (6.34% selective risk). On SQuAD
answerability, it accepted 1,875 of 2,000 inputs (93.75% coverage), with 118 accepted errors
(6.29%). Kev's corresponding results were 64.17% coverage and 7.53% risk on CLINC, and 35.40%
coverage and 4.38% risk on SQuAD. The latter risk estimate had a 95% article-cluster bootstrap
interval of 2.57–6.67%, so even its sub-5% point estimate does not establish a risk guarantee.

StartLux achieved the highest direct accuracy on both main test sets: 93.50% on CLINC and
91.25% on SQuAD, compared with 88.00% and 86.00% for Qwen, and 77.50% and 81.35% for Kev.
The paired StartLux-minus-Qwen differences were 5.50 percentage points on CLINC (95% interval
3.67–7.33) and 5.25 points on SQuAD (3.59–6.82). These comparisons use 600 CLINC examples and
2,000 SQuAD examples; SQuAD is binary given-passage answerability, not extractive question-answering
EM or F1. Intervals condition on the frozen models, prompts, calibration and gates.

## Routing and calibration

Fallback improved Kev relative to its own direct predictions, but did not outperform the best
direct system. Kev's refitted cascade attained 85.67% CLINC accuracy and 85.85% SQuAD accuracy.
For SQuAD, it rescued 184 local errors but introduced 94 errors by deferring locally correct
cases, producing a net improvement of 90/2,000. The resulting accuracy was close to direct
Qwen's 86.00%; their paired difference interval included zero.

The stronger StartLux model benefited little from Qwen fallback. Its CLINC cascade deferred
only one input and did not change accuracy. On SQuAD, 125 deferrals rescued 29 errors and
introduced 24, yielding a net improvement of five cases, or 0.25 percentage points. The
paired 95% difference interval was −0.293 to +0.825 percentage points. Thus, the observed
gain does not establish that the cascade is more accurate than direct StartLux. Confidence
routing nevertheless exceeded the expected accuracy of random routing at matched deferral
quotas; this is a different comparison from outperforming the strongest direct model.

Temperature refitting produced mixed probability-quality changes. It improved Kev's SQuAD
NLL, Brier score and ECE, but did not change either model's accepted set on that binary task.
CLINC ECE decreased slightly while NLL and Brier score increased for both decision models.
The OOS stress workload also raised aggregate accuracy and reduced accepted risk, but reused
the same 450 banking examples while changing both OOS prevalence and the OOS pool. It is
therefore neither independent replication nor a pure causal prevalence intervention.

## Live serving performance

Serving measurements used one NVIDIA L40S, fixed 200-input workloads per task, five
deployments, concurrency levels 1 and 8, and three timed rounds per configuration. All
60 cells completed at least 30 seconds, for 60,200 input executions and 63,720 underlying
HTTP calls without failures or retries. Repeated executions are performance observations,
not additional independent test examples. Values below are arithmetic means of three rounds.

At concurrency 8, direct StartLux processed 59.10 successful inputs/s on CLINC and 57.88/s
on SQuAD, versus 32.71/s and 30.76/s for Qwen, corresponding to throughput ratios of 1.81×
and 1.88×. At concurrency 1, the ratios were 3.42× and 3.63×. These results compare the
specific frozen serving systems; they do not isolate an architectural cause.

Adding Qwen reduced StartLux's SQuAD throughput from 57.88/s to 44.21/s at concurrency 8,
a 23.6% decrease. The mean of round-specific p95 request latencies increased from 150.60 ms
to 249.67 ms. Both model pairs successfully resided on the same GPU. Across all timed cells,
the largest sampled whole-device memory use was 38,458 MiB and the minimum recorded free
fraction was 16.52%. This is sampled VRAM evidence, not a continuous peak guarantee or a
measurement of host RAM.

The request timer begins when a controller worker starts executing a call and ends after
the local or fallback response returns. It excludes pre-worker queue waiting, model startup
and warm-up. Throughput uses the full measured-round wall time. GPU-seconds per 1,000 inputs
is derived from that wall time for one reserved GPU, rather than kernel-active time, energy,
or the entire session's cost. Fixed deployment order, one GPU, and three consecutive rounds
limit broader performance generalization. Kev additionally showed small observed
concurrency-dependent SQuAD label variability; this was retained and reported, not filtered.

## Combined interpretation

In these frozen tasks, direct StartLux offers the strongest demonstrated combination of
predictive accuracy and serving efficiency relative to direct Qwen. However, high local
coverage did not imply compliance with the desired test-time selective-risk target. The
results support evaluating the complete local-plus-fallback system, including harmful
deferrals and extra serving costs, instead of assuming that a generative fallback is a
stronger predictor or that fewer fallback calls necessarily improve overall efficiency.

## Evidence

- [Formal results and statistical limits](../formal-2026-11-02/RESULTS_ZH.md)
- [Benchmark analysis and measurement definitions](BENCHMARK_ZH.md)
- [Recorded illustrative replay](../formal-2026-11-02/DEMO_REPLAY_ZH.md)
- [Raw bundle index](../../records/BUNDLE_INDEX.json)
