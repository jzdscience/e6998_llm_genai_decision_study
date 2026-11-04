# Metrics and analysis protocol

`decision_study.analysis` consumes saved predictions only. It never invokes a model, tunes on test labels, invents missing predictions, or assumes that a fallback is correct. Numerical analysis uses NumPy float64. The software tests use explicitly marked artificial fixtures; plots of fixtures carry a visible **SOFTWARE TEST FIXTURE — NOT MODEL RESULTS** watermark.

## Inputs and provenance

- Each model has one row per required sample ID, including terminal failures, and one immutable `config_hash` for the analysis run.
- Decision models require calibration, validation, main test and extra OOS predictions. Qwen requires main test and extra OOS predictions only.
- Calibration artifacts contain the SHA256 of the loaded `analysis.py` source, the model config hash, a hash of calibration/validation inputs and labels, and a hash of their semantic predictions. Adding test predictions does not change the calibration hash. Changing a fitting prediction, prompt/input, gold label, config, or analysis implementation invalidates it. An analysis-only change requires refitting from cached calibration/validation predictions, not rerunning models. Plot-only changes do not invalidate it. The complete artifact is self-hashed, including its temperatures and thresholds. `validate_calibration(...)` checks these identities before any gate is used in evaluation, replay, live inference, or timing; it does not refit. Unknown prediction IDs outside the supplied input manifest are rejected. The checksum detects accidental changes, not an adversary who can replace both content and checksum.
- Missing predictions, duplicate IDs, mixed config hashes, mixed fixture/real rows, and stale fitting artifacts cause an exception. Failures are not silently dropped.
- A valid decision row has `status="ok"`, a permitted label, every class probability exactly once, finite probabilities in [0,1], sum within 1e-5 of one, and a chosen label tied for the maximum within 1e-8. Tiny probability sum roundoff is normalized. Invalid rows count as incorrect and force fallback.

Kev's pinned HTTP serializer rounds each probability to four decimals. The adapter
first verifies that precision, finite [0,1] values, a positive total, and a total
mass error no larger than `number_of_labels * 0.00005` (plus floating-point roundoff).
It then restores unit mass for the saved `probabilities`; `raw_response` preserves
the original wire values. The server-selected label is retained when rounding
creates a tie. Probability metrics, confidence, temperature fitting and demo all
use this same reconstructed distribution. This cannot recover precision lost on
the wire and must be described in reports; it does not relax reference/service
smoke thresholds or change StartLux validation.

## Temperature and validation gate

The released probabilities already include model-native calibration. We fit only an additional scalar:

```
p_T = softmax(log(clip(p_released, 1e-12, 1)) / T)
T in [0.05, 20]
```

Temperature is optimized on valid calibration predictions by bounded golden-section search in log(T), minimizing mean negative log likelihood. T=1 is always a candidate; improvements <=1e-10 use T=1. The artifact records the objective, valid calibration denominator, failures and whether the optimum is at a search boundary. No valid calibration distribution is a blocking error. The native temperature/runtime configuration remains in the frozen runtime manifest; it is not applied a second time here.

The released and refitted arms independently select gates on validation data. Sort finite valid maximum probabilities, move across entire tied-score groups, and take the largest accepted set with empirical error <=0.05 and at least 50 accepted cases. The search continues through an infeasible prefix: feasibility need not be monotonic. If none qualifies, freeze `accept_none=true`. At test time the threshold stays fixed even when the target is missed.

Positive scalar temperature preserves argmax. For two classes it also preserves maximum-probability ranking, so a reselected gate can accept the same cases before and after temperature scaling. This is an expected possible result, not a broken experiment.

## Reported denominators and metrics

Direct accuracy includes all intended inputs; failed/invalid rows are wrong. The confusion matrix has an explicit `__failure__` prediction column. Macro-F1 averages across the fixed task classes. Per-class precision/recall includes flags for undefined zero denominators; zero division displays as zero in the macro score.

Probability metrics use only complete valid distributions, alongside `n_total`, `n_valid`, and `invalid_count`:

- NLL: mean negative log probability assigned to the gold class (probabilities clipped at 1e-12 for numerical evaluation).
- Brier: mean over samples of the **sum** of squared probability errors across classes. It is not divided by class count.
- ECE: 15 equal-width confidence bins on [0,1], with 1 included in the last bin; sum of bin fraction times absolute accuracy-minus-confidence. Accuracy in a tied-probability bin uses the actual model prediction.

A cascade accepts the local decision when valid and above its frozen threshold; otherwise it uses the actual saved Qwen prediction. Its reported accuracy uses all inputs, including fallback failures. `selective_risk` is errors among accepted inputs only. Zero accepted cases gives null/N/A risk and target-miss status, never risk=0. OOS is a normal task label, not a routing action.

## Matched routing controls and curves

At a frozen gate's exact deferral count k, invalid local rows are mandatory deferrals for every policy. Random routing selects the remaining quota uniformly without replacement, with 100 reproducible draws from a seeded generator. We report mean, SD, 2.5/97.5 percentiles of randomization variability, and the exact expected accuracy. That range is not a sampling confidence interval, and random routing is not a mathematical lower bound.

The hindsight oracle sorts optional rows by

```
gain = 1[Qwen correct] - 1[local correct]
```

and fills exactly the remaining quota, even if the quota eventually forces negative-gain deferrals. It is a gold-dependent upper bound for this pair of predictions and this mandatory-deferral constraint. It cannot be deployed. Oracle uncertainty is not presented as out-of-sample inference.

Risk–coverage and accuracy–deferral curves traverse whole tied-score groups. They are descriptive test-set curves, not a mechanism for changing the selected operating point. The same real Qwen predictions are used throughout. Random/oracle reference curves include every legal exact quota, even when tied confidence scores offer fewer attainable gate operating points. Complementarity tables report both correct, only local correct, only Qwen correct, and both wrong.

## Paired uncertainty

Default: 2,000 bootstrap repetitions with a fixed RNG seed. All systems and frozen gates use the **same sampled observations in each replicate**:

- CLINC: resample cases with replacement within each gold class, preserving class counts.
- SQuAD: resample article groups with replacement, carrying all selected questions of each sampled article together. The ratio estimands use the resulting number of questions. The official planned test sample has 35 article clusters, not 2,000 independent cases.

Report percentile 95% intervals for direct/cascade accuracy, coverage, selective risk, and paired accuracy differences from Qwen and the local model. These condition on the sampled workload, frozen upstream models/configurations, fitted temperature and selected gate; they do not include model-training or calibration-selection variation. They are not future-risk guarantees.

For zero accepted cases, risk intervals are undefined. If all accepted cases are correct or all are wrong, empirical bootstrap cannot estimate the unseen tail: risk interval endpoints are null and an explicit warning explains the limitation. Accepted counts under 50 and replicates with no acceptance are flagged. Other degenerate intervals also carry warnings.

## CLINC workload stress and prevalence reweighting

The main workload has 450 banking + 150 OOS inputs; the stress workload reuses the same banking inputs and uses the other 850 OOS inputs. Paired stress-change intervals resample the banking pool once per replicate, use it in both workloads, and resample each distinct OOS pool separately. The comparison changes both prevalence and OOS composition; it is not a pure prevalence intervention.

A separate descriptive curve reweights a fixed conditional pool of 450 banking + all 1,000 OOS inputs, with frozen gates. Each banking observation receives `(1-alpha)/450`; each OOS observation `alpha/1000`. Alpha runs 0, .1, .2, .25, .3, ..., .7, .75, .8, .9, 1. These dependent points add no new observations. Accuracy and coverage are weighted means; selective risk is weighted accepted errors divided by weighted acceptance mass.

## Outputs and reading order

`analysis/metrics.json` is the full machine-readable record. `summary.csv` and `probability_metrics.csv` are lookup tables. `curves.json` contains the exact chart data. `error_cases.jsonl` contains up to 30 deterministically selected records per task, balanced in round-robin order across model/error category/OOS buckets; categories with no examples are not fabricated. Cases come from the main test workload and include both local-helpful and fallback-harmful outcomes.

`plot_results(analysis_dir, fixture=False)` creates PNG and PDF versions of direct/complementarity, calibration/risk, routing, probability-quality and CLINC prevalence charts. It also automatically recognizes fixture evidence in `metrics.json`. The plots do not use cached prediction latency as evidence of real serving performance; live benchmark artifacts are a separate stage.

## Primary references

- [Temperature scaling, Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html)
- [CLINC official data](https://github.com/clinc/oos-eval)
- [SQuAD 2.0 official data](https://rajpurkar.github.io/SQuAD-explorer/)
- [StartLux probability API](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/api.md)
