# E30: Lee2019 weighted Ridge soft-regularization pilot

Date: 2026-07-01

## 結論

E30は予想より面白い。

ただし、勝ち筋は target-prefix penalty ではなかった。

> Hard top-k source-side feature selection を、source score の soft weighting に置き換えた Ridge が Lee2019 で最も良かった。

Best pilot result:

```text
ridge_weight_source_rank_g2p0, alpha=100:
  acc = 72.616%
```

Reference:

```text
LDA source_only_q0p25:
  acc = 71.852%
```

Paired difference:

```text
weighted Ridge - LDA source_only_q0p25:
  +0.764pp
  95% CI [-0.139, +1.690]
```

So this is **promising but not yet conclusive**.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/lee2019_weighted_ridge_e30.py`

Outputs:

- `intentflow/offline/results/research_outputs/260701_lee2019_weighted_ridge_e30_alpha0p1/`
- `intentflow/offline/results/research_outputs/260701_lee2019_weighted_ridge_e30_alpha1/`
- `intentflow/offline/results/research_outputs/260701_lee2019_weighted_ridge_e30_alpha10/`
- `intentflow/offline/results/research_outputs/260701_lee2019_weighted_ridge_e30_alpha100/`

Main files:

- `weighted_ridge_records.csv`
- `weighted_ridge_summary.csv`
- `summary.json`

## Protocol

Dataset:

- Lee2019 LR all62
- 54 subjects
- source session0 train labels available
- target session1 prefix unlabeled
- target eval labels used only for evaluation

Classifier:

- RidgeClassifier
- source-standardized features
- feature weights applied after standardization

Weights:

```text
source_rank_gγ:
  weight_j = percentile_rank(source_score_j)^γ

source_minus_prefix:
  adjusted_j = percentile_rank(source_score_j) - λ * percentile_rank(prefix_energy_j)
  weight_j = percentile_rank(adjusted_j)
```

Important:

LDA diagonal scaling was previously effectively no-op.  
Ridge is used because feature scaling changes the effective penalty.

## Main alpha sweep result

Top methods by accuracy:

| alpha | method | acc | gain vs Ridge full | q05 gain vs Ridge full | P(gain<-5 vs Ridge full) |
|---:|---|---:|---:|---:|---:|
| 100 | ridge_weight_source_rank_g2p0 | **72.616** | +3.449 | -3.750 | 1.9% |
| 100 | ridge_weight_source_rank_g1p0 | 72.292 | +3.125 | -3.750 | 3.7% |
| 10 | ridge_weight_source_rank_g2p0 | 72.199 | +3.194 | -4.188 | 1.9% |
| 1 | ridge_weight_source_rank_g2p0 | 72.060 | +3.079 | -3.750 | 1.9% |
| 0.1 | ridge_weight_source_rank_g2p0 | 72.037 | +3.056 | -3.750 | 1.9% |
| reference | LDA source_only_q0p25 | 71.852 | n/a | n/a | n/a |

Key point:

```text
source_rank_g2p0 is consistently best across alpha values.
alpha=100 is best, but the source-weighting signal is not alpha-specific.
```

## Prefix penalty did not win

At alpha=100:

| method | acc |
|---|---:|
| ridge_weight_source_rank_g2p0 | **72.616** |
| ridge_weight_source_rank_g1p0 | 72.292 |
| ridge_weight_source_minus_prefix_l0p25 | 71.852 |
| ridge_weight_source_minus_prefix_l0p5 | lower |
| ridge_weight_source_minus_prefix_l1p0 | lower |
| ridge_weight_source_minus_prefix_l2p0 | bad |

This matches E29:

```text
target-prefix metrics have weak harm diagnostic signal,
but do not solve hard candidate selection.
```

For now, do not frame the method as target-prefix adaptation.

## Paired comparison to LDA source_only_q0p25

Best method:

```text
ridge_weight_source_rank_g2p0, alpha=100
```

Comparison:

| Contrast | Mean | 95% CI | q05 | P(diff<-5pp) |
|---|---:|---:|---:|---:|
| weighted Ridge - LDA source_only_q0p25 | +0.764 | [-0.139, +1.690] | -5.000 | 1.9% |
| weighted Ridge - LDA full | +2.940 | [+1.365, +4.560] | -5.438 | 9.3% |
| weighted Ridge - Ridge full | +3.449 | [+1.944, +4.931] | -3.750 | 1.9% |

Interpretation:

- It clearly improves over Ridge full.
- It clearly improves over LDA full.
- It is better than LDA q0.25 on mean, but CI still crosses zero.

## Nested-long failure subjects

For the 15 subjects where E25 nested wrongly chose `longitudinal_q0p10`:

| Contrast | Mean | 95% CI |
|---|---:|---:|
| weighted Ridge - LDA source_only_q0p25 | +0.083 | [-1.583, +1.750] |
| weighted Ridge - LDA longitudinal_q0p10 | **+3.000** | **[+0.167, +5.833]** |

Interpretation:

Weighted Ridge does not beat the best fixed Lee policy on those subjects, but it largely avoids the bad longitudinal choice.

## Why this is scientifically interesting

The previous best Lee method was:

```text
hard source_only_q0p25:
  keep top 25% dimensions, delete the rest.
```

E30 suggests a better inductive bias:

```text
soft source-score regularization:
  keep all dimensions, but penalize low source-score dimensions more strongly.
```

This matches the mechanism from E20/E21:

- Lee2019 needs a broad motor + posterior/mixed covariance subspace.
- Hard selection can remove useful context.
- Full tangent keeps too much noisy covariance.
- Soft weighting is a natural middle ground.

## Judgment

### 守る

Source-score soft weighting is now a serious method candidate.

### 弱める

Target-prefix penalty is not currently the main signal.

### まだ言えない

`ridge_weight_source_rank_g2p0, alpha=100` is a validated method.

Why:

- alpha/gamma were chosen after seeing Lee target results.
- Need nested source-side hyperparameter validation.

## Next experiment: E31

Run exact nested validation for weighted Ridge hyperparameters.

Candidate hyperparameters:

```text
alpha ∈ {0.1, 1, 10, 100}
gamma ∈ {0.5, 1, 2}
prefix penalty λ ∈ {0, 0.25}
```

Decision:

```text
If nested E31 selects source_rank_g2p0/alpha~100 and keeps >72.2%:
  weighted source-score Ridge becomes the new best Lee method.

If nested E31 collapses to weaker hyperparameters:
  E30 is a post-hoc tuning artifact; use it only as a mechanistic hint.
```

