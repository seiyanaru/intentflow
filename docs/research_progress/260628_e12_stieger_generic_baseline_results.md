# 260628 E12 Stieger: generic feature-selection baseline results

## One-line verdict

Stieger E12-fastは完了。失敗0。

結論:

```text
Stiegerのlongitudinal compact-core gainは、
単純なwithin-subject self-source Fisher/variance selectionでは説明できない。
```

これは重要。

Lee2019では `self_source_variance q0.25` がかなり強かったが、
Stiegerでは generic variance はむしろ大きく悪化した。

したがって、Stieger側の主張はかなり守れる:

```text
Stieger LR/UDで効いているのは、単なるcompactnessやvariance pruningではなく、
source-side population / longitudinal statistics に基づくsubspace rankingである。
```

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/stieger_generic_feature_selection_e12_fast.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_e12_stieger_generic_feature_selection_fast/
```

Main files:

```text
summary.json
summary.csv
paired_comparisons.csv
selection_records.csv
run.log
subjects/
```

Run status:

```text
n_records: 16736
n_reference_records: 6276
n_generic_records: 10460
n_failures: 0
subjects_done: 62
```

## Design

Reused E9 records:

```text
full_q1p00
longitudinal_q0p05
longitudinal_q0p10
longitudinal_q0p20
source_only_q0p25
source_only_q0p50
```

Newly computed generic baselines:

```text
self_source_fisher q0.05/q0.10/q0.20/q0.25/q0.50
self_source_variance q0.05/q0.10/q0.20/q0.25/q0.50
```

Protocol:

```text
source session labels only
zero target labels
prefix EA
broad_all60 + fb_sensorimotor21_mu_beta
equal posterior fusion
pure_lr / pure_ud / pooled summaries
```

## Main result: pure LR

| method | acc | gain | 95% CI | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| longitudinal q0.05 | 70.928 | +4.054 | [+2.856, +5.414] | 10.406 | 10.8% |
| longitudinal q0.10 | 70.431 | +3.558 | [+2.341, +4.901] | 9.688 | 13.4% |
| longitudinal q0.20 | 69.560 | +2.687 | [+1.647, +3.854] | 10.620 | 13.6% |
| source_only q0.25 | 68.936 | +2.063 | [+1.106, +3.065] | 9.645 | 14.2% |
| source_only q0.50 | 68.132 | +1.259 | [+0.626, +1.907] | 9.075 | 12.6% |
| self_source_fisher q0.05 | 67.170 | +0.296 | [-0.344, +0.968] | 11.387 | 20.4% |
| full q1.00 | 66.873 | +0.000 | [0.000, 0.000] | 0.000 | 0.0% |
| self_source_variance q0.25 | 65.500 | -1.374 | [-2.336, -0.366] | 18.194 | 26.7% |
| self_source_variance q0.05 | 63.285 | -3.589 | [-5.152, -2.059] | 24.735 | 42.2% |

Interpretation:

```text
LRでは generic selection は完全に負け。
特に variance pruning は強く壊す。
```

## Main result: pure UD

| method | acc | gain | 95% CI | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| longitudinal q0.20 | 64.944 | +1.287 | [+0.354, +2.274] | 12.376 | 17.6% |
| longitudinal q0.10 | 64.740 | +1.084 | [+0.107, +2.095] | 14.027 | 20.6% |
| source_only q0.50 | 64.498 | +0.841 | [+0.288, +1.445] | 8.481 | 11.0% |
| source_only q0.25 | 64.168 | +0.512 | [-0.212, +1.268] | 12.249 | 19.2% |
| full q1.00 | 63.657 | +0.000 | [0.000, 0.000] | 0.000 | 0.0% |
| self_source_fisher q0.50 | 63.420 | -0.236 | [-0.613, +0.141] | 6.437 | 7.4% |
| self_source_variance q0.25 | 62.380 | -1.276 | [-2.141, -0.443] | 15.565 | 27.4% |
| self_source_variance q0.05 | 60.821 | -2.836 | [-4.350, -1.327] | 25.304 | 35.6% |

Interpretation:

```text
UDでも generic variance/fisher は改善を説明しない。
UDは source_only q0.50 が安全寄りに強いが、gain最大は longitudinal q0.20。
```

## Main result: pooled LR+UD

| method | acc | gain | 95% CI | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| longitudinal q0.10 | 67.681 | +2.357 | [+1.455, +3.321] | 12.152 | 17.1% |
| longitudinal q0.05 | 67.601 | +2.277 | [+1.305, +3.295] | 13.867 | 18.5% |
| longitudinal q0.20 | 67.338 | +2.013 | [+1.175, +2.894] | 11.517 | 15.5% |
| source_only q0.25 | 66.638 | +1.313 | [+0.657, +2.003] | 10.991 | 16.8% |
| source_only q0.50 | 66.394 | +1.070 | [+0.598, +1.572] | 8.754 | 11.7% |
| full q1.00 | 65.325 | +0.000 | [0.000, 0.000] | 0.000 | 0.0% |
| self_source_fisher q0.50 | 65.076 | -0.248 | [-0.460, -0.046] | 6.435 | 8.5% |
| self_source_fisher q0.25 | 64.924 | -0.401 | [-0.803, -0.015] | 9.441 | 16.1% |
| self_source_variance q0.25 | 64.003 | -1.322 | [-2.054, -0.618] | 16.969 | 27.0% |
| self_source_variance q0.05 | 62.081 | -3.243 | [-4.471, -2.123] | 25.034 | 39.0% |

Interpretation:

```text
pooledでも generic baselines は全滅。
self-source variance はLee2019では強かったが、Stiegerでは明確に悪い。
```

## Paired differences

Key paired comparisons:

| scope | comparison | mean diff | 95% CI | P(diff<0) |
|---|---|---:|---:|---:|
| LR | long q0.05 - variance q0.05 | +7.643 | [+5.880, +9.436] | 0.000 |
| LR | long q0.10 - variance q0.10 | +6.656 | [+5.155, +8.188] | 0.000 |
| LR | source_only q0.25 - variance q0.25 | +3.436 | [+2.193, +4.722] | 0.000 |
| UD | long q0.20 - variance q0.20 | +2.994 | [+1.906, +4.128] | 0.000 |
| UD | source_only q0.25 - fisher q0.25 | +1.003 | [+0.280, +1.896] | 0.004 |
| pooled | long q0.10 - variance q0.10 | +5.294 | [+4.315, +6.348] | 0.000 |
| pooled | source_only q0.25 - variance q0.25 | +2.635 | [+1.757, +3.551] | 0.000 |
| pooled | source_only q0.25 - fisher q0.25 | +1.714 | [+1.130, +2.363] | 0.000 |

These are not close.

## Scientific interpretation

### What this result protects

This protects the Stieger-side novelty:

```text
Stieger gains are not just generic source-session feature pruning.
```

In particular, the strongest Stieger point:

```text
LR longitudinal q0.05: +4.054 pp
```

cannot be explained by:

```text
self-source Fisher q0.05: +0.296 pp
self-source variance q0.05: -3.589 pp
```

### What this result changes

The cross-dataset story becomes sharper:

```text
Stieger:
  source-side population / longitudinal statistics are necessary.
  self-source variance pruning is harmful.

Lee2019:
  generic variance pruning is already useful,
  but source-side population discriminative selection is still best.
```

So the mechanism is not simply:

```text
smaller tangent subspace is always better.
```

The mechanism is:

```text
The right compact subspace must be ranked by source-side statistics.
The useful ranking signal differs across datasets/tasks.
```

## Paper implication

Keep the main title/framing:

```text
Source-side tangent subspace selection for zero-target-label cross-session EEG-MI
```

Do not title it:

```text
Longitudinal subspace selection
```

because Lee2019 is source_only-best.

But Stieger analysis can claim:

```text
In Stieger, longitudinal compact-core selection provides gains that generic
within-subject Fisher/variance selection cannot reproduce.
```

## Next action

The next most useful step is not another method.

Next:

```text
E14: final paper table + mechanism figure
```

Required final table:

```text
Dataset / task
Full tangent
Best generic self-source baseline
Best source_only
Best longitudinal
E10 source-side selected policy
```

This table will make the story clean and defensible.

