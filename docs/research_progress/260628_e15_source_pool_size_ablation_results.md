# 260628 E15: source-pool size ablation results

## One-line verdict

E15は完了。

結論は二分された。

```text
Stieger:
  source-side validation is surprisingly stable.
  m=10 already all-source E10 gainの約87-89%を保持。
  m=20で約93%。

Lee2019:
  source-side validation is not stable with small/moderate source pools.
  m=40でもall-source gainの約59%しか保持しない。
```

これは重要。

この研究の主張は:

```text
source-side validation is generally stable with small cohorts
```

ではない。

より正確には:

```text
Stieger-like longitudinal multi-session cohortでは少数sourceでもかなり安定。
Lee2019ではsource_only q0.25の選択に大きなsource poolが必要。
```

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/source_pool_size_ablation_e15.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_e15_source_pool_size_ablation_full/
```

Main files:

```text
summary.json
summary.csv
selected_records.csv
source_validation_diagnostics.csv
run.log
```

Run status:

```text
n_selected_records: 2,148,146
n_diagnostics: 2,162,160
```

## Design

Candidate set is fixed to E10 primary:

```text
full_q1p00

source_only:
  q0.15, q0.25, q0.50

longitudinal:
  q0.05, q0.10, q0.20, q0.25, q0.50
```

Pool sizes:

```text
m = 3, 5, 10, 20, 40, all
```

Repeats:

```text
200
```

Policies:

```text
Stieger:
  global_risk
  condition_risk

Lee2019:
  global_risk
```

Selection rule:

```text
maximize source-validation mean gain
subject to P(gain < -5pp) <= 0.20
```

## Stieger global_risk

All-source E10:

```text
selected: longitudinal q0.10
gain: +2.357 pp
```

Pool-size result:

| m | gain | repeat 95% interval | retention vs all | regret | R10 | P(gain<-5) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | +1.813 | [+1.414, +2.223] | 76.9% | +0.544 | 11.956 | 15.8% |
| 5 | +1.907 | [+1.592, +2.305] | 80.9% | +0.450 | 12.187 | 16.5% |
| 10 | +2.040 | [+1.792, +2.289] | 86.6% | +0.317 | 12.472 | 17.0% |
| 20 | +2.206 | [+1.999, +2.387] | 93.6% | +0.151 | 12.614 | 17.4% |
| 40 | +2.258 | [+2.118, +2.393] | 95.8% | +0.098 | 12.598 | 17.6% |
| all | +2.357 | deterministic | 100.0% | 0.000 | 12.105 | 17.1% |

Selection stability:

```text
m=3:
  longitudinal family 73.2%
  source_only family 21.4%
  full 5.5%

m=10:
  longitudinal family 88.1%
  source_only family 11.5%
  full 0.4%

m=20:
  longitudinal family 97.3%

m=40:
  longitudinal family 99.99%
```

Interpretation:

```text
Stieger global rule is already useful at m=3,
and practically stable by m=10-20.
```

## Stieger condition_risk

All-source E10:

```text
pure_lr: longitudinal q0.05
pure_ud: longitudinal q0.20
gain: +2.719 pp
```

Pool-size result:

| m | gain | repeat 95% interval | retention vs all | regret | R10 | P(gain<-5) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | +2.112 | [+1.793, +2.390] | 77.7% | +0.607 | 11.164 | 14.0% |
| 5 | +2.243 | [+1.915, +2.522] | 82.5% | +0.476 | 11.160 | 14.1% |
| 10 | +2.427 | [+2.209, +2.637] | 89.2% | +0.292 | 11.181 | 14.0% |
| 20 | +2.528 | [+2.348, +2.702] | 93.0% | +0.191 | 11.307 | 14.1% |
| 40 | +2.607 | [+2.470, +2.732] | 95.9% | +0.112 | 11.498 | 14.3% |
| all | +2.719 | deterministic | 100.0% | 0.000 | 11.442 | 14.1% |

Selection stability:

```text
m=3:
  longitudinal family 70.1%
  source_only family 24.1%
  full 5.8%

m=10:
  longitudinal family 83.2%
  source_only family 15.4%
  full 1.4%

m=20:
  longitudinal family 89.8%

m=40:
  longitudinal family 97.2%
```

Interpretation:

```text
Condition-specific rule is stronger than global.
It reaches nearly 90% gain retention by m=10.
```

This is a good practical result because LR/UD condition is known before evaluation.

## Lee2019 global_risk

All-source E10:

```text
selected: source_only q0.25
gain: +2.176 pp
```

Pool-size result:

| m | gain | repeat 95% interval | retention vs all | regret | R10 | P(gain<-5) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | +1.333 | [+0.253, +2.639] | 61.3% | +0.843 | 10.866 | 11.8% |
| 5 | +1.470 | [+0.598, +2.593] | 67.5% | +0.706 | 11.287 | 12.6% |
| 10 | +1.334 | [+0.369, +2.384] | 61.3% | +0.842 | 11.492 | 13.1% |
| 20 | +1.274 | [+0.254, +2.153] | 58.5% | +0.902 | 11.689 | 13.7% |
| 40 | +1.280 | [+0.436, +2.083] | 58.8% | +0.896 | 11.688 | 13.7% |
| all | +2.176 | deterministic | 100.0% | 0.000 | 7.500 | 7.4% |

Selection stability:

```text
m=3:
  source_only family 37.6%
  longitudinal family 54.2%
  full 8.2%

m=10:
  source_only family 48.7%
  longitudinal family 49.4%

m=20:
  source_only family 58.2%
  longitudinal family 41.6%

m=40:
  source_only family 75.7%
  longitudinal family 24.3%
```

Even at m=40, the rule often selects longitudinal candidates,
which are not Lee2019-optimal.

Interpretation:

```text
Lee2019 source-side validation needs nearly the full cohort to reliably select source_only q0.25.
Small/moderate source pools produce unstable family selection.
```

## Scientific interpretation

### What E15 supports

E15 supports:

```text
Stieger-like multi-session longitudinal cohorts can support source-side
candidate selection from a moderate source pool.
```

For Stieger:

```text
m=10:
  global retention 86.6%
  condition-specific retention 89.2%

m=20:
  global retention 93.6%
  condition-specific retention 93.0%
```

This is strong enough to say the method is not purely a large-cohort artifact
on Stieger.

### What E15 weakens

E15 weakens a broad claim:

```text
source-side validation is generally stable with small source pools.
```

Lee2019 contradicts that.

Lee2019 appears to require a large source pool for reliable source_only-vs-longitudinal
family discrimination.

## Revised research story

The story should be:

```text
Source-side tangent subspace selection improves zero-target-label cross-session EEG-MI.

However, the reliability of source-side candidate selection depends on the
source-pool size and dataset/task structure.

Stieger-like longitudinal multi-session data:
  stable with moderate pools, especially condition-specific selection.

Lee2019 two-session LR:
  all-source source_only q0.25 is best, but the family decision is unstable
  under small/moderate source pools.
```

This is more nuanced, but stronger scientifically.

## Practical implication

For deployment:

```text
If a moderate Stieger-like cohort exists:
  source-side validation is usable.

If only a small source pool exists:
  avoid adaptive family/q selection.
  use a pre-fixed robust policy.

For Lee-like settings:
  fixed source_only q0.25/q0.50 may be safer than small-pool validation.
```

## Next action

The next experiment should be E16:

```text
fixed-policy robustness vs source-pool validation
```

Why:

E15 shows Lee2019 small-pool validation is unstable.
So the practical question becomes:

```text
When source pool is small, is a fixed robust candidate better than validation?
```

Compare:

```text
Fixed policies:
  source_only q0.25
  source_only q0.50
  longitudinal q0.10
  longitudinal q0.20

Adaptive policies:
  E15 source-pool validation m=3/5/10/20
```

Decision:

```text
If fixed policy dominates small-pool validation:
  recommend fixed policy in low-source regimes.

If validation dominates:
  keep validation as practical method.
```

