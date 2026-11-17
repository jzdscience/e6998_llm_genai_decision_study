# ANLI learnability results

The primary population is the locally audited eligible dev subset. External usage is unconfirmed.
All other fits, the train-source holdout and deployment thresholds are secondary/descriptive.

## eligible_dev (1737 inputs)

| Configuration | Accuracy (%) | Deferred |
|---|---:|---:|
| direct_qwen9b | 61.716 | — |
| direct_startlux4b | 72.712 | — |
| direct_startlux9b | 74.554 | — |
| qwen9b/confidence10 | 72.251 | 173 |
| qwen9b/oracle10_hindsight | 80.541 | 173 |
| qwen9b/random10 | 71.963 | 173 |
| startlux9b/confidence10 | 73.633 | 173 |
| startlux9b/oracle10_hindsight | 80.138 | 173 |
| startlux9b/random10 | 72.942 | 173 |
| qwen9b/n1800/front | 71.042 | 173 |
| qwen9b/n1800/text | 71.963 | 173 |
| qwen9b/n18000/front | 70.984 | 173 |
| qwen9b/n18000/text | 71.906 | 173 |
| qwen9b/n6000/front | 70.409 | 173 |
| qwen9b/n6000/text | 71.618 | 173 |
| startlux9b/n1800/front | 74.036 | 173 |
| startlux9b/n1800/text | 72.827 | 173 |
| startlux9b/n18000/front | 74.266 | 173 |
| startlux9b/n18000/text | 73.230 | 173 |
| startlux9b/n6000/front | 74.093 | 173 |
| startlux9b/n6000/text | 72.884 | 173 |

Text − front gain: 0.921 pp. Interval: {'ci95': [-0.0011396499238964992, 0.01881819621363276], 'premise_groups': 1540, 'replicates': 2000, 'scope': 'conditional on fitted models; paired normalized-premise percentile bootstrap; rerank exact quota per draw', 'seed': 20261113, 'status': 'PASS'}.

## train_holdout (1800 inputs)

| Configuration | Accuracy (%) | Deferred |
|---|---:|---:|
| direct_qwen9b | 80.889 | — |
| direct_startlux4b | 89.389 | — |
| direct_startlux9b | 91.278 | — |
| qwen9b/confidence10 | 87.889 | 180 |
| qwen9b/oracle10_hindsight | 92.111 | 180 |
| qwen9b/random10 | 88.500 | 180 |
| startlux9b/confidence10 | 90.722 | 180 |
| startlux9b/oracle10_hindsight | 93.056 | 180 |
| startlux9b/random10 | 89.722 | 180 |
| qwen9b/n1800/front | 88.778 | 180 |
| qwen9b/n1800/text | 88.722 | 180 |
| qwen9b/n18000/front | 89.000 | 180 |
| qwen9b/n18000/text | 88.389 | 180 |
| qwen9b/n6000/front | 89.444 | 180 |
| qwen9b/n6000/text | 88.389 | 180 |
| startlux9b/n1800/front | 90.056 | 180 |
| startlux9b/n1800/text | 89.778 | 180 |
| startlux9b/n18000/front | 90.167 | 180 |
| startlux9b/n18000/text | 89.556 | 180 |
| startlux9b/n6000/front | 90.111 | 180 |
| startlux9b/n6000/text | 89.556 | 180 |

Text − front gain: -0.611 pp. Interval: {'ci95': [-0.012115256426280049, 0.002200524255965418], 'premise_groups': 199, 'replicates': 2000, 'scope': 'conditional on fitted models; paired normalized-premise percentile bootstrap; rerank exact quota per draw', 'seed': 20261113, 'status': 'PASS'}.

The 1 pp practical reference is not an equivalence margin. The offline exact quota is not an online budget guarantee.
