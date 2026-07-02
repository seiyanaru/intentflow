# E39/E40: frozen comparison and source-anchor interpolation

Date: 2026-07-01

## One-line verdict

E39はE38 practical ruleを正式に通した。

しかし、E40の結果により、主役候補は更新すべき。

```text
binary fallback:
  use LDA or use weighted Ridge

よりも、

source-anchor score interpolation:
  score = (1 - beta) * LDA_margin + beta * weighted_Ridge_margin

の方が、平均精度と下側リスクのバランスが良い。
```

現時点の最有力候補は:

```text
Regime-anchored reliability-weighted Ridge
+ source-margin-calibrated interpolation
```

固定 beta なら、accuracy-first は `beta=0.7`、risk-first は `beta=0.6` が有力。

ただし、beta はE40ではtarget評価を見て解釈しているため、まだmethodとして凍結されていない。
次はE41でsource-validationだけから beta を選べるかを検証する必要がある。

## Artifacts

Scripts:

- `intentflow/offline/scripts/analysis/e39_frozen_method_comparison.py`
- `intentflow/offline/scripts/analysis/e40_source_anchor_interpolation.py`

Outputs:

- `intentflow/offline/results/research_outputs/260701_e39_frozen_method_comparison/`
- `intentflow/offline/results/research_outputs/260701_e40_source_anchor_interpolation/`

Main files:

- `e39_method_summary.csv`
- `e39_key_contrasts.csv`
- `e39_pass_fail.csv`
- `e40_interpolation_records.csv`
- `e40_interpolation_summary.csv`
- `e40_interpolation_contrasts.csv`
- `e40_best_risk_utility_candidates.csv`

## E39: frozen practical rule check

E39 compared:

- LDA full
- fixed source/longitudinal weighted Ridge
- hard top-k baselines
- E36 frozen guarded selector
- E38 global family
- E38 local guard + global family

Pass/fail:

| Criterion | Value | Pass |
|---|---:|---|
| Lee subject: E38 practical beats E36 repeat | +0.347pp | yes |
| BNCI full: E38 practical keeps LDA anchor | +0.000pp | yes |
| BNCI m8 unit: E38 practical improves E36 by >= +0.1pp | +0.148pp | yes |
| BNCI m8 unit: harm not worse than E36 by > +1pp | -0.0035 | yes |

Interpretation:

```text
E38 practical rule is defensible.
```

But the improvement over E36 is small:

```text
BNCI m8 unit:
  E38 practical - E36 repeat = +0.148pp
```

So E39 alone is not a strong new method result.
It is a cleanup/freeze result.

## E40 protocol

E40 tested continuous interpolation between LDA and the regime-level weighted Ridge family.

For binary decision margins:

```text
z_lda   = lda_margin / std(lda_source_margin)
z_ridge = ridge_margin / std(ridge_source_margin)

score_beta = (1 - beta) * z_lda + beta * z_ridge
```

This source-margin calibration is important.
Raw LDA and Ridge scores are not on the same scale.

The endpoints are preserved:

```text
beta = 0 -> LDA full prediction
beta = 1 -> weighted Ridge prediction
```

Regime families:

| Regime | Family |
|---|---|
| Lee2019 sensorimotor20 | source-weighted Ridge |
| BNCI2014_001 full-source | LDA |
| BNCI2014_001 m8/class | longitudinal-weighted Ridge |

Important correction:

An initial E40 run accidentally used Lee all62 cache with the `lee2019_sensorimotor20` label.
That run was discarded.  The final E40 output uses:

```text
intentflow/offline/results/research_outputs/260629_lee2019_channel_dimension_ablation_e20/sensorimotor20/subject_cache
```

and matches E32 anchors:

```text
Lee LDA full:                 69.606
Lee ridge_source_rank_g2:     71.991
```

## E40 main result: beta controls the risk-utility frontier

### Lee2019 sensorimotor20

Unit/subject are identical because Lee has one held target unit per subject.

| Method | Gain vs LDA | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---:|---:|---:|---:|---:|
| beta=1.0 weighted Ridge | +2.384 | -6.250 | -7.500 | 9.3% | 33.3% |
| beta=0.7 interpolation | **+2.500** | -6.250 | -6.875 | **7.4%** | **24.1%** |
| beta=0.6 interpolation | +2.222 | **-4.625** | -6.250 | **5.6%** | 25.9% |
| LDA full | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |

Interpretation:

```text
beta=0.7 weakly dominates beta=1 on mean and risk in Lee.
beta=0.6 is safer but gives up some mean.
```

### BNCI2014_001 m8/class: unit-level

This is the hardest and most important risk view.

| Method | Gain vs LDA | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---:|---:|---:|---:|---:|
| beta=1.0 weighted Ridge | +2.327 | -7.258 | -7.898 | 8.3% | 28.8% |
| beta=0.7 interpolation | **+2.470** | -5.645 | -6.201 | 6.3% | 26.0% |
| beta=0.6 interpolation | +2.361 | -4.839 | -5.506 | 3.8% | 26.4% |
| beta=0.5 interpolation | +2.285 | **-4.032** | **-4.755** | **3.5%** | 22.9% |
| E38 practical beta=1 with local fallback | +1.910 | -6.169 | -7.036 | 6.3% | 24.3% |
| LDA full | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |

Interpretation:

```text
E40 interpolation is better than E38 binary fallback on BNCI m8.
```

Specifically:

- beta=0.7 has much higher mean than E38 practical with similar P(gain<-5);
- beta=0.5/0.6 sharply reduce lower-tail risk while keeping nearly all mean gain;
- beta=1 is no longer the best practical choice.

### BNCI2014_001 full-source

The regime-level family is LDA, so all beta variants reduce to LDA:

```text
Gain vs LDA = 0.000
P(gain<-5) = 0.0%
```

This preserves the negative-control behavior.

## What changed

### Keep

```text
Regime-level family selection is useful.
```

E40 still depends on the E38 family decision:

- Lee -> source-weighted
- BNCI full -> LDA
- BNCI m8 -> longitudinal-weighted

### Revise

E38 practical was:

```text
regime family + local LDA fallback
```

E40 suggests a stronger mechanism:

```text
regime family + continuous source-anchor interpolation
```

The local fallback is no longer the main risk-control mechanism.
Interpolation controls risk more smoothly.

### Reject / weaken

Do not make binary guarded selection the main novelty.
It is now a useful baseline, not the final method.

## Current best method candidate

Accuracy-first frozen candidate:

```text
if regime selects LDA:
  use LDA
else:
  use source-margin-calibrated interpolation with beta = 0.7
```

Risk-first frozen candidate:

```text
if regime selects LDA:
  use LDA
else:
  use source-margin-calibrated interpolation with beta = 0.6
```

Why beta=0.7 is attractive:

| Regime | Gain | P(gain<-5) |
|---|---:|---:|
| Lee sensorimotor20 | +2.500 | 7.4% |
| BNCI full-source | 0.000 | 0.0% |
| BNCI m8 unit | +2.470 | 6.3% |

Why beta=0.6 is attractive:

| Regime | Gain | P(gain<-5) |
|---|---:|---:|
| Lee sensorimotor20 | +2.222 | 5.6% |
| BNCI full-source | 0.000 | 0.0% |
| BNCI m8 unit | +2.361 | 3.8% |

## Critical caveat

E40 is not yet a fully honest final method.

Reason:

```text
beta=0.6/0.7 are chosen after seeing held target results.
```

So the right conclusion is not:

```text
beta=0.7 is validated.
```

The right conclusion is:

```text
source-anchor interpolation has a strong empirical signal and should become
the next frozen/nested experiment.
```

## Next experiment: E41

E41 should test whether beta can be chosen without target labels.

Recommended design:

```text
Candidate beta:
  {0.0, 0.1, ..., 1.0}

Selection unit:
  regime-level, not subject-level

Selection data:
  source-subject inner validation only

Rule:
  choose the largest-mean beta among candidates satisfying
  P(inner gain < -5pp) <= 0.08

Fallback:
  if no beta passes, beta = 0
```

Why regime-level:

E37-B showed that source validation does not predict individual outer gain.
So beta should not be personalized per subject/repeat yet.

Pass criteria:

```text
Lee:
  selected beta should give >= +2.0pp vs LDA
  and P(gain<-5) lower than beta=1.

BNCI full:
  selected family remains LDA.

BNCI m8 unit:
  selected beta should give >= +2.0pp vs LDA
  and P(gain<-5) <= 6.3%.
```

If E41 passes, the research story becomes much stronger:

```text
Source-side evidence determines the reliability family;
a source-anchored interpolation coefficient controls the risk-utility frontier;
the final method needs zero target labels.
```

That is stronger than the previous E38 story.
