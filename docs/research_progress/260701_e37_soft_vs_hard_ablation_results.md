# E37-A: soft reliability weighting vs hard top-k ablation

Date: 2026-07-01

## One-line verdict

E37-Aの結果は、E36の方法の核をかなり支持している。

```text
Positive regimes:
  soft reliability-weighted Ridge > hard top-k / full Ridge

Negative-control regime:
  BNCI full-sourceではsoftもhardも不要で、source-anchor guardがLDA fullに戻す
```

つまり、勝っている理由は単なる次元削減ではない。

現時点の解釈は:

```text
soft feature reliability weighting changes the regularized linear classifier's
inductive bias in useful regimes;
the source anchor prevents using that bias in regimes where LDA is already strong.
```

## Artifacts

Script:

`intentflow/offline/scripts/analysis/e37_soft_vs_hard_ablation.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_e37_soft_vs_hard_ablation/`

Main files:

- `e37_method_summary.csv`
- `e37_family_best_summary.csv`
- `e37_key_contrasts.csv`
- `e37_interpretation.csv`
- `e37_ablation_records.csv`
- `summary.json`

## Protocol

E37-A does not fit new models.

It aggregates:

- E32 fixed-method records;
- E36 frozen guarded records.

Families:

```text
source_anchor:
  lda_full

full_ridge:
  ridge_full_a100

hard_topk_lda:
  lda_source_q25
  lda_long_q10
  lda_long_q70

hard_topk_ridge:
  ridge_hard_source_q25_a100
  ridge_hard_long_q10_a100
  ridge_hard_long_q70_a100

soft_weighted_ridge:
  ridge_source_rank_g1_a100
  ridge_source_rank_g2_a100
  ridge_longitudinal_rank_g2_a100

frozen_guarded:
  e36_frozen_guarded_repeat
  e36_frozen_guarded_stable
```

Important caution:

```text
family-best comparisons are descriptive mechanism checks,
not honest deployable selectors.
```

The honest deployable method remains E36 frozen guarded.

## Result 1: soft weighting beats hard top-k in positive regimes

### Lee2019 sensorimotor20

Best fixed family methods:

| Family | Best method | Acc | Gain vs LDA full |
|---|---|---:|---:|
| soft weighted | `ridge_source_rank_g2_a100` | **71.991** | **+2.384** |
| hard top-k LDA | `lda_long_q70` | 70.579 | +0.972 |
| hard top-k Ridge | `ridge_hard_long_q10_a100` | 69.560 | -0.046 |
| full Ridge | `ridge_full_a100` | 67.616 | -1.991 |
| LDA full | `lda_full` | 69.606 | 0.000 |

Key contrast:

```text
best soft - best hard:
  +1.412pp
  95% CI [0.139, 2.639]
  q05 = -6.313
  P(diff<-5pp) = 7.4%
```

Interpretation:

```text
In Lee sensorimotor20, soft weighting is clearly stronger on mean accuracy
than hard top-k. But subject-level lower-tail risk remains.
```

This supports the method's mean-accuracy claim, not a no-harm claim.

### BNCI2014_001 m8/class

Best fixed family methods:

| Family | Best method | Acc | Gain vs LDA full |
|---|---|---:|---:|
| soft weighted | `ridge_longitudinal_rank_g2_a100` | **70.010** | **+2.327** |
| hard top-k Ridge | `ridge_hard_long_q10_a100` | 69.302 | +1.619 |
| hard top-k LDA | `lda_long_q10` | 68.730 | +1.047 |
| full Ridge | `ridge_full_a100` | 67.812 | +0.129 |
| LDA full | `lda_full` | 67.683 | 0.000 |

Key contrast:

```text
best soft - best hard:
  +0.708pp
  95% CI [-0.302, 1.568]
  q05 = -1.512
  P(diff<-5pp) = 0.0%
```

Interpretation:

```text
Soft weighting is better on average than hard top-k in BNCI m8,
but the CI still crosses zero.
```

So the claim should be moderate:

```text
soft weighting improves the mean and preserves low lower-tail risk;
the soft-vs-hard margin is not statistically decisive on BNCI m8 alone.
```

## Result 2: full Ridge is not the answer

Full Ridge results:

| Regime | `ridge_full_a100` vs LDA full |
|---|---:|
| Lee sensorimotor20 | -1.991pp |
| BNCI full-source | -3.047pp |
| BNCI m8/class | +0.129pp |

Interpretation:

```text
The gain is not because RidgeClassifier alone is better.
Unweighted full Ridge is weak or neutral.
```

This is important. It means the useful ingredient is not simply:

```text
replace LDA with Ridge
```

but:

```text
apply reliability-aware soft weighting before regularized linear classification.
```

## Result 3: BNCI full-source is a true negative control

BNCI full-source family best:

| Family | Best method | Acc | Gain vs LDA full |
|---|---|---:|---:|
| hard top-k LDA | `lda_long_q70` | **78.047** | +0.448 |
| LDA full | `lda_full` | 77.599 | 0.000 |
| soft weighted | `ridge_source_rank_g1_a100` | 77.061 | -0.538 |
| hard top-k Ridge | `ridge_hard_source_q25_a100` | 76.882 | -0.717 |
| full Ridge | `ridge_full_a100` | 74.552 | -3.047 |

Key contrast:

```text
best soft - LDA full:
  -0.538pp

ridge_full - LDA full:
  -3.047pp
```

E36 frozen guarded selection:

```text
lda_full 9/9
gain vs LDA full = 0.000pp
```

Interpretation:

```text
BNCI full-source is not a reliability-weighted Ridge regime.
The source-anchor guard is necessary and works.
```

This is one of the strongest pieces of evidence for the source-anchor design.

## Result 4: guard cost is acceptable but real

The E36 guard sacrifices some fixed-soft performance in positive regimes:

| Regime | Guarded vs best soft |
|---|---:|
| Lee sensorimotor20 | -0.347pp |
| BNCI m8 repeat | -0.566pp |
| BNCI m8 stable | -0.185pp |
| BNCI full-source | +0.538pp vs best soft |

Interpretation:

```text
The guard trades a small amount of positive-regime accuracy for protection
against wrong-regime weighted Ridge.
```

This is exactly the desired behavior.

The cost is not zero:

- Lee loses 0.35pp vs fixed `ridge_source_rank_g2`;
- BNCI m8 repeat loses 0.57pp vs fixed `ridge_longitudinal_rank_g2`;

but the guard:

- keeps +2.04pp on Lee;
- keeps +1.76pp on BNCI m8 repeat;
- avoids -0.54pp on BNCI full-source.

## What E37-A proves

### Supported

```text
soft reliability weighting is a meaningful mechanism in the positive regimes.
```

Evidence:

- Lee: soft beats best hard by +1.41pp with positive CI;
- BNCI m8: soft beats best hard by +0.71pp on average;
- full Ridge is weak/neutral, so the effect is not Ridge alone.

### Supported

```text
source-anchor guard is essential.
```

Evidence:

- BNCI full-source: weighted Ridge loses, guard selects LDA full 9/9;
- positive regimes: guard preserves most of the soft-weighted gain.

### Not fully proven

```text
soft weighting strictly dominates hard top-k for every subject/regime.
```

This is not true.
Lee has lower-tail risk and BNCI m8 soft-vs-hard CI crosses zero.

So the paper should not claim universal dominance.

## Updated thesis claim

After E37-A, the cleanest claim is:

```text
Reliability-weighted Ridge is not merely a hard feature-selection effect.
In positive cross-session regimes, continuous feature weighting gives better
mean accuracy than hard top-k or unweighted full Ridge.
Because this bias is harmful or unnecessary in full-source regimes, a
source-anchor guard is required.
```

This is a stronger and more mechanistic story than:

```text
we built a selector
```

## Next experiment: E37-B

The next step should diagnose whether source validation truly predicts the
right score family.

E37-B should compute, for each outer subject/repeat:

```text
inner source-score gain vs LDA full
inner longitudinal-score gain vs LDA full
selected family
outer source-score gain
outer longitudinal-score gain
outer selected-method gain
```

Questions:

1. Does inner gain correlate with outer gain?
2. How often does source validation select the outer-best weighted family?
3. Are failures concentrated in Lee lower-tail subjects?
4. Does the guard reject weighted Ridge when both inner weighted gains are weak?

If E37-B is positive, the method story becomes:

```text
source validation identifies when and how to apply reliability weighting.
```

If E37-B is weak, the method is still useful empirically, but the selection
mechanism should be described more cautiously.

