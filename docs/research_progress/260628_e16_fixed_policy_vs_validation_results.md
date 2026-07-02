# 260628 E16: fixed-policy robustness vs source-pool validation

## One-line verdict

E16は完了。

結論:

```text
finite source-pool validation is useful, but it does not beat a good fixed policy.
```

特に Lee2019 では明確:

```text
small/moderate-pool validation is worse and riskier than fixed source_only q0.25.
```

Stiegerでも同じ方向:

```text
validation approaches fixed policy as m grows,
but finite mでは fixed longitudinal policy が上。
```

したがって実用指針は:

```text
small source pool:
  use fixed robust policy.

large enough source pool:
  validation becomes reasonable, especially Stieger-like settings.
```

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/fixed_policy_vs_validation_e16.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_e16_fixed_policy_vs_validation/
```

Main files:

```text
summary.json
fixed_policy_summary.csv
validation_vs_fixed.csv
```

## Fixed policies

### Lee2019

| fixed policy | gain | 95% CI | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|
| source_only q0.25 | +2.176 | [+0.486, +3.774] | 7.500 | 7.4% |
| source_only q0.50 | +1.759 | [+0.417, +2.986] | 7.708 | 5.6% |
| longitudinal q0.10 | +1.852 | [-0.324, +3.935] | 13.125 | 20.4% |
| longitudinal q0.20 | +1.528 | [-0.139, +3.125] | 8.125 | 9.3% |

Best:

```text
source_only q0.25
```

### Stieger

| fixed policy | gain | 95% CI | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|
| condition long LR q0.05 / UD q0.20 | +2.719 | [+1.838, +3.679] | 11.442 | 14.1% |
| longitudinal q0.10 | +2.357 | [+1.455, +3.321] | 12.105 | 17.1% |
| longitudinal q0.05 | +2.277 | [+1.305, +3.295] | 13.847 | 18.5% |
| longitudinal q0.20 | +2.013 | [+1.175, +2.894] | 11.487 | 15.5% |
| source_only q0.25 | +1.313 | [+0.657, +2.003] | 10.953 | 16.8% |
| source_only q0.50 | +1.070 | [+0.598, +1.572] | 8.740 | 11.7% |

Best:

```text
condition-known fixed policy:
  LR: longitudinal q0.05
  UD: longitudinal q0.20
```

If a single global fixed policy is required:

```text
longitudinal q0.10
```

## Validation vs fixed: Lee2019

Best fixed:

```text
source_only q0.25
gain +2.176
P(gain<-5) 7.4%
```

| source pool m | validation gain | best fixed gain | validation - fixed | validation P<-5 | fixed P<-5 |
|---:|---:|---:|---:|---:|---:|
| 3 | +1.333 | +2.176 | -0.843 | 11.8% | 7.4% |
| 5 | +1.470 | +2.176 | -0.706 | 12.6% | 7.4% |
| 10 | +1.334 | +2.176 | -0.842 | 13.1% | 7.4% |
| 20 | +1.274 | +2.176 | -0.902 | 13.7% | 7.4% |
| 40 | +1.280 | +2.176 | -0.896 | 13.7% | 7.4% |
| all | +2.176 | +2.176 | 0.000 | 7.4% | 7.4% |

Interpretation:

```text
Lee2019では small/moderate source-pool validation は使うべきではない。
fixed source_only q0.25 の方が高利得かつ低リスク。
```

E15で見たように、small/mid pool validation は longitudinal を選びすぎる。
それが Lee2019 の性能を落としている。

## Validation vs fixed: Stieger global

Best global fixed:

```text
longitudinal q0.10
gain +2.357
P(gain<-5) 17.1%
```

| source pool m | validation gain | best fixed gain | validation - fixed | validation P<-5 | fixed P<-5 |
|---:|---:|---:|---:|---:|---:|
| 3 | +1.813 | +2.357 | -0.544 | 15.8% | 17.1% |
| 5 | +1.907 | +2.357 | -0.450 | 16.5% | 17.1% |
| 10 | +2.040 | +2.357 | -0.317 | 17.0% | 17.1% |
| 20 | +2.206 | +2.357 | -0.151 | 17.4% | 17.1% |
| 40 | +2.258 | +2.357 | -0.098 | 17.6% | 17.1% |
| all | +2.357 | +2.357 | 0.000 | 17.1% | 17.1% |

Interpretation:

```text
Stieger validation is not bad.
But if longitudinal q0.10 is already known as a fixed policy,
finite-m validation is unnecessary and slightly worse.
```

Validation may have slightly lower P<-5 at very small m, but the gain loss is
larger than the risk benefit.

## Validation vs fixed: Stieger condition-specific

Best fixed:

```text
LR: longitudinal q0.05
UD: longitudinal q0.20
gain +2.719
P(gain<-5) 14.1%
```

| source pool m | validation gain | best fixed gain | validation - fixed | validation P<-5 | fixed P<-5 |
|---:|---:|---:|---:|---:|---:|
| 3 | +2.112 | +2.719 | -0.607 | 14.0% | 14.1% |
| 5 | +2.243 | +2.719 | -0.476 | 14.1% | 14.1% |
| 10 | +2.427 | +2.719 | -0.292 | 14.0% | 14.1% |
| 20 | +2.528 | +2.719 | -0.191 | 14.1% | 14.1% |
| 40 | +2.607 | +2.719 | -0.112 | 14.3% | 14.1% |
| all | +2.719 | +2.719 | 0.000 | 14.1% | 14.1% |

Interpretation:

```text
Condition-specific fixed policy is the cleanest Stieger recommendation.
```

If LR/UD task condition is known, there is little reason to run small-pool validation.

## Important caveat

The fixed policies here are not magic.

They are fixed because previous E8-E15 evidence identified them:

```text
Stieger:
  longitudinal q0.10 globally
  LR q0.05 / UD q0.20 condition-specifically

Lee2019:
  source_only q0.25
```

So E16 does not prove these fixed policies will transfer to every unseen dataset.

It proves a practical point inside the current research setting:

```text
Once a robust fixed policy is known for a dataset/task family,
small-pool source validation is worse than simply using that fixed policy.
```

## Scientific implication

E16 changes the practical method recommendation:

Previous:

```text
Use source-side validation to choose family/q.
```

Revised:

```text
Use fixed robust policies when source pool is small or the task family is known.
Use source-side validation only when a sufficiently large and relevant source pool is available,
or when no reliable fixed policy exists.
```

## Recommended final method framing

The paper/research direction should distinguish two regimes:

```text
1. Discovery/large-cohort regime:
   source-side validation discovers the best compactness/family.

2. Deployment/small-cohort regime:
   use a fixed robust policy learned from prior source-side analyses.
```

Concrete recommendations:

```text
Stieger-like LR/UD:
  if condition known:
    LR -> longitudinal q0.05
    UD -> longitudinal q0.20
  if one global policy required:
    longitudinal q0.10

Lee-like LR:
  source_only q0.25
  source_only q0.50 if lower-tail risk is prioritized over mean gain
```

## Next action

Do not add another selector.

Next useful experiment:

```text
E17: cross-dataset transfer of fixed policies / task-family rules
```

Question:

```text
Can we decide fixed policy from dataset/task descriptors
instead of target performance?
```

Minimal version:

```text
Train rule on one dataset/task family:
  Stieger LR/UD -> longitudinal compact core
  Lee LR -> source_only q0.25

Then audit whether simple descriptors explain the rule:
  number of source sessions
  task type LR vs UD
  source-vs-target separability statistics
  generic variance baseline strength
```

But if time is short, stop here and consolidate:

```text
E8-E16 already support a coherent research story.
```

