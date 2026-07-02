# 260627 Decision memo after V3: what to test next

## One-line conclusion

次にやるべき最優先実験は、新手法を増やすことではない。

```text
E6: source-side nested model selection
```

である。

理由は単純:

```text
Stiegerでは longitudinal q0.10 が強い。
Lee2019では source-only q0.25 が強い。
したがって、今のままだと「datasetを見て都合よくbestを選んだ」と査読で刺される。
```

この批判を潰すには、held-out target subject を完全に除外したまま、
source subjectsだけで

```text
score family:
  source-only
  longitudinal
  sep-no-drift

fraction:
  q0.25
  q0.10
```

を選ぶ必要がある。

## Current evidence recap

### Stieger primary pooled

Unit: session-condition, LR+UD, n=1046.

| method | acc | gain vs full | q05 | R10 loss | P(gain<-5) | P(gain>0) |
|---|---:|---:|---:|---:|---:|---:|
| longitudinal q0.10 | 67.84 | +2.38 | -11.28 | 12.38 | 17% | 56% |
| longitudinal q0.25 | 67.28 | +1.82 | -10.20 | 11.65 | 17% | 55% |
| source-only q0.25 | 66.82 | +1.36 | -10.11 | 11.12 | 17% | 52% |
| source-only q0.10 | 66.83 | +1.36 | -11.62 | 13.90 | 19% | 52% |

Condition split:

```text
LR:
  longitudinal q0.10 gain +3.68
  source-only q0.25 gain +2.17

UD:
  longitudinal q0.25 gain +1.20
  source-only q0.25 gain +0.56
```

Interpretation:

```text
Stiegerではlongitudinal termが平均的に上乗せを持つ。
ただしP(gain>0)は56%程度で、全sessionに効く万能薬ではない。
```

### Lee2019

Unit: subject, n=54.

| method | acc | gain vs full | q05 | R10 loss | P(gain<-5) | P(gain>0) |
|---|---:|---:|---:|---:|---:|---:|
| source-only q0.25 | 71.85 | +2.18 | -6.25 | 7.50 | 7% | 59% |
| longitudinal q0.10 | 71.53 | +1.85 | -12.50 | 13.12 | 20% | 63% |
| longitudinal q0.25 | 71.04 | +1.37 | -7.50 | 8.33 | 11% | 50% |
| source-only q0.10 | 70.14 | +0.46 | -11.25 | 13.54 | 22% | 50% |

Direct comparison:

```text
longitudinal q0.10 - source-only q0.25:
  -0.32 pp
  q05 -10.44
  P(longitudinal < source-only) = 46%
```

Interpretation:

```text
Lee2019ではsource-only q0.25が最も強く、最も安全寄り。
longitudinal q0.10は平均では改善するが、tail riskが悪い。
```

## Important pattern: method helps weak/full-overfit cases

### Stieger full-baseline quartiles

```text
Q1 low-full, full mean 49.24:
  longitudinal q0.10 gain +5.25
  source-only q0.25 gain +3.47

Q4 high-full, full mean 85.29:
  longitudinal q0.10 gain +0.25
  source-only q0.25 gain -0.06
```

### Lee2019 full-baseline quartiles

```text
Q1 low-full, full mean 52.05:
  longitudinal q0.10 gain +3.12
  source-only q0.25 gain +3.21

Q4 high-full, full mean 90.96:
  longitudinal q0.10 gain +0.29
  source-only q0.25 gain +1.44
```

Interpretation:

```text
この方法の本体は「適応」ではなく、
高次元tangent LDAの過学習/不安定featureをsource側で落とす正則化である。

full featureが弱いケースでは効く。
full featureがすでに強いケースでは利得が小さい、または負になる。
```

## Hypotheses after V3

### H1: robust core hypothesis

```text
source-side supervised tangent-subspace selection improves zero-target-label
cross-session EEG-MI by removing unstable/noisy tangent dimensions.
```

Evidence:

```text
Stieger:
  random q0.25 = -3.81 pp
  random q0.10 = -6.44 pp

Lee2019:
  random q0.25 = -6.21 pp
  random q0.10 = -9.47 pp
```

Thus, fewer dimensions alone is not enough.
Which dimensions are selected matters.

### H2: conditional longitudinal hypothesis

```text
longitudinal same-class drift helps only when drift can be estimated reliably.
```

Why this is plausible:

```text
Stieger:
  up to 11 sessions / subject
  many source-subject longitudinal transitions
  longitudinal > source-only

Lee2019:
  only 2 sessions / subject
  one transition per subject
  source-only >= longitudinal
```

Prediction:

```text
If Stieger metric learning is artificially limited to only 2 sessions,
the longitudinal advantage should shrink or disappear.
```

This is the cleanest mechanism test.

## Next experiments, in priority order

## E6: Source-side nested model selection

Purpose:

```text
Remove post-hoc/dataset-specific cherry-picking.
```

For each held-out target subject:

```text
1. Exclude target subject completely.
2. On source subjects only, run inner leave-one-source-subject validation.
3. Candidate set:
     source-only q0.25
     source-only q0.10
     longitudinal q0.25
     longitudinal q0.10
     sep-no-drift q0.25
     sep-no-drift q0.10
     full q1.00
4. Choose one candidate using source-validation risk-utility.
5. Evaluate once on held-out target subject/session.
```

Primary selection rule:

```text
Choose the candidate with the highest source-validation mean gain
subject to:
  P(gain < -5pp) <= 0.20
```

If no candidate satisfies the risk condition:

```text
fall back to full q1.00
```

Also report a simpler mean-only selector as secondary.

Pass criteria:

```text
On both Stieger and Lee2019:
  nested-selected method beats full by > +1.0 pp
  CI lower bound is not strongly negative
  R10 loss is not worse than the selected fixed candidate by > 2 pp
```

Scientific interpretation:

```text
If E6 passes:
  paper can claim source-validated subspace selection.

If E6 fails:
  do not claim adaptive method selection.
  use fixed source-only q0.25 as the robust baseline and treat longitudinal as analysis.
```

## E7: Stieger session-depth ablation

Purpose:

```text
Test whether longitudinal advantage depends on having enough longitudinal sessions.
```

Design:

```text
Metric-learning source sessions:
  K=2
  K=3
  K=5
  K=all

Evaluation:
  same held-out Stieger target sessions as E5b/V1/V2.

Compare:
  longitudinal q0.10/q0.25
  source-only q0.25
```

Prediction:

```text
K=2:
  longitudinal ≈ source-only or worse

K=5/all:
  longitudinal > source-only
```

Pass criteria:

```text
longitudinal - source-only increases with K
and reaches > +0.75 pp with CI excluding 0 for K=all.
```

Scientific interpretation:

```text
If E7 passes:
  longitudinal is not dead.
  It becomes a conditional mechanism:
    use longitudinal stability only when enough source longitudinal data exists.

If E7 fails:
  remove longitudinal from the method title.
  keep it only as failed/fragile variant.
```

## E8: Mechanistic selected-feature analysis

Purpose:

```text
Explain what source-only q0.25 and longitudinal q0.10 select.
```

Required comparisons:

```text
Stieger LR:
  longitudinal q0.10 vs source-only q0.25

Stieger UD:
  longitudinal q0.25 vs source-only q0.25

Lee2019 LR:
  source-only q0.25 vs longitudinal q0.10
```

Report:

```text
top channels
diagonal vs off-diagonal covariance fraction
mu / low-beta / high-beta if filterbank available
overlap between source-only and longitudinal selected dimensions
```

Pass criteria:

```text
Selected features concentrate around C3/C4/CP3/CP4/Cz or sensorimotor covariance terms.
Overlap/difference explains why longitudinal helps Stieger but not Lee.
```

## What not to do next

Do not do these yet:

```text
new label-free gate
new active-label policy
Tent/AdaBN/deep OTTA
another posterior fusion/selector
```

Reason:

```text
The current evidence says the gain comes from feature-space selection/regularization.
Gate/adaptation experiments are not the bottleneck now.
```

## Recommended next command-level direction

Implement E6 first.

Expected new script:

```text
intentflow/offline/scripts/analysis/source_side_nested_subspace_selection.py
```

or two dataset-specific wrappers:

```text
stieger_source_side_nested_selection.py
lee2019_source_side_nested_selection.py
```

Minimum output:

```text
nested_selection_records.csv
nested_summary.json
chosen_candidate_counts.csv
risk_utility_nested_vs_fixed.csv
```

Then run E7 only after E6 tells us whether nested source-side selection is viable.

