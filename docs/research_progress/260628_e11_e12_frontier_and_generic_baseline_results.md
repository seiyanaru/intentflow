# 260628 E11/E12: compactness frontier and generic feature-selection baselines

## One-line verdict

E11は完了。論文図として使える。

E12-fastはLee2019で完了。

結果は:

```text
source_only_q0.25 is not explained away by simple within-subject Fisher selection.
```

ただし:

```text
self-source variance q0.25 is a nontrivial baseline and should be reported.
```

現時点での正直な判断:

```text
source-side population discriminative subspace selection remains stronger on Lee2019,
but generic dimensionality reduction contributes part of the gain.
```

## E11 artifacts

Script:

```text
intentflow/offline/scripts/analysis/compactness_frontier_e11.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_e11_compactness_frontier/
```

Main files:

```text
compactness_frontier_table.csv
compactness_best_by_panel.csv
e10_nested_policy_points.csv
compactness_gain_frontier.png
compactness_gain_frontier.pdf
compactness_risk_utility_frontier.png
compactness_risk_utility_frontier.pdf
summary.json
```

## E11 result: best compactness by panel

| panel | best method | acc | gain | 95% CI | R10 | P(gain<-5) |
|---|---|---:|---:|---:|---:|---:|
| Stieger LR | longitudinal q0.05 | 70.928 | +4.054 | [+2.856, +5.414] | 10.406 | 10.8% |
| Stieger UD | longitudinal q0.20 | 64.944 | +1.287 | [+0.354, +2.274] | 12.376 | 17.6% |
| Stieger pooled | longitudinal q0.10 | 67.681 | +2.357 | [+1.455, +3.321] | 12.152 | 17.1% |
| Lee2019 LR | source_only q0.25 | 71.852 | +2.176 | [+0.486, +3.820] | 7.500 | 7.4% |

Interpretation:

```text
The compactness-frontier story is now visually supportable:

Stieger LR:
  very compact stable core.

Stieger UD:
  moderate compactness.

Lee2019:
  broader source-discriminative source_only subspace.
```

This is stronger and safer than:

```text
longitudinal is universally best.
```

## E12 artifacts

Full generic script:

```text
intentflow/offline/scripts/analysis/lee2019_generic_feature_selection_e12.py
```

This full script recomputes many high-dimensional LDA fits and was too slow for
interactive use.

Fast script:

```text
intentflow/offline/scripts/analysis/lee2019_generic_feature_selection_e12_fast.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_e12_lee2019_generic_feature_selection_fast/
```

Main files:

```text
selection_records.csv
summary.csv
summary.json
failures.csv
```

## E12-fast design

E12-fast reuses E9 records for:

```text
full_q1p00
source_only_q0p25
source_only_q0p50
longitudinal_q0p10
longitudinal_q0p20
```

and newly computes generic held-subject source-session baselines:

```text
self_source_fisher q0.10/q0.25
self_source_variance q0.10/q0.25
```

These generic baselines use only the held subject's labeled session0 calibration data.
They use zero target labels.

## E12-fast result: Lee2019

| method | acc | gain | 95% CI | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| source_only q0.25 | 71.852 | +2.176 | [+0.532, +3.866] | 7.500 | 7.4% |
| longitudinal q0.10 | 71.528 | +1.852 | [-0.278, +3.959] | 13.125 | 20.4% |
| source_only q0.50 | 71.435 | +1.759 | [+0.440, +3.032] | 7.708 | 5.6% |
| longitudinal q0.20 | 71.204 | +1.528 | [-0.046, +3.125] | 8.125 | 9.3% |
| self_source_variance q0.25 | 71.181 | +1.505 | [-0.208, +3.241] | 9.167 | 14.8% |
| self_source_fisher q0.25 | 69.838 | +0.162 | [-0.880, +1.250] | 6.458 | 9.3% |
| full q1.00 | 69.676 | +0.000 | [0.000, 0.000] | 0.000 | 0.0% |
| self_source_variance q0.10 | 69.421 | -0.255 | [-2.292, +1.759] | 14.792 | 25.9% |
| self_source_fisher q0.10 | 68.935 | -0.741 | [-2.107, +0.626] | 9.583 | 16.7% |

Paired differences:

| comparison | mean diff | 95% CI | P(diff<0) |
|---|---:|---:|---:|
| source_only q0.25 - self_source_variance q0.25 | +0.671 | [-0.973, +2.199] | 20.7% |
| source_only q0.25 - self_source_fisher q0.25 | +2.014 | [+0.532, +3.472] | 0.3% |
| source_only q0.25 - source_only q0.50 | +0.417 | [-0.810, +1.620] | 24.9% |
| self_source_variance q0.25 - full | +1.505 | [-0.208, +3.334] | 4.5% |

## Interpretation

### What E12 supports

E12 supports:

```text
The Lee2019 gain is not merely within-subject source-label Fisher selection.
```

`self_source_fisher q0.25` is weak:

```text
gain +0.162 pp
```

while `source_only q0.25` is:

```text
gain +2.176 pp
```

### What E12 weakens

E12 weakens any claim that:

```text
all the gain is from sophisticated source-side class geometry.
```

Because `self_source_variance q0.25` is nontrivial:

```text
gain +1.505 pp
```

This means generic dimensionality reduction / variance structure explains part of
the effect.

However, it does not currently beat `source_only q0.25`.

## Current paper implication

The paper should include `self_source_variance` as a serious baseline.

Suggested wording:

```text
Simple within-subject source-session variance selection improves over full tangent
features on Lee2019, indicating that compactness itself is useful.
However, population source-side discriminative selection gives the best Lee2019
point, suggesting that cross-subject source statistics provide information beyond
generic variance pruning.
```

## Remaining gap

E12-fast is Lee2019 only.

Before a journal submission, run an analogous Stieger generic baseline:

```text
self-source Fisher / variance within each held subject's source session
for LR and UD separately, q0.05/q0.10/q0.20/q0.25.
```

But for the immediate research direction, E12-fast is enough to decide:

```text
Do not pivot away.
The source-side subspace selection story survived the strongest quick generic baseline.
```

