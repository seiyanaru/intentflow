# E43/E43B/E43C: weight-shape × alpha × beta method exploration

Date: 2026-07-02

## One-line verdict

E43は、手法探索としてかなり良い結果を出した。

ただし、勝ったのは

```text
local source-selected hyperparameter selector
```

ではない。

勝ったのは:

```text
sharper rank reliability weighting
+ lower source-anchor beta
+ source-validation modal stability rule
```

である。

結論:

```text
E42の g2 / alpha=100 / beta=0.7 はまだ最終形ではない。
rank-gammaを強くし、betaを0.5-0.6へ下げると、
Lee2019とBNCI m8の両方で mean と lower-tail risk が同時に改善する。
```

ただし、paired CI はまだ完全には決定的でない。主張は:

```text
E43 improves the risk-utility frontier and gives a better frozen candidate.
```

であり、

```text
E43 proves a statistically decisive universal optimum.
```

ではない。

## Artifacts

Main script:

`intentflow/offline/scripts/analysis/e43_weight_shape_alpha_beta_search.py`

Rank-grid output:

`intentflow/offline/results/research_outputs/260702_e43_weight_shape_alpha_beta_search/`

Sigmoid-grid output:

`intentflow/offline/results/research_outputs/260702_e43b_sigmoid_weight_shape_search/`

Combined rank+sigmoid post-hoc analysis:

`intentflow/offline/results/research_outputs/260702_e43c_combined_rank_sigmoid_analysis/`

Important files:

- `e43_grid_records.csv`
- `e43_summary.csv`
- `e43_top_candidates.csv`
- `e43_benchmark_vs_e42.csv`
- `e43_modal_stability_rule_summary.csv`
- `e43c_combined_modal_rule_summary.csv`
- `e43c_combined_choice_summary.csv`

## Protocol

E43 searched the internals of the current weighted method:

```text
family    ∈ {source, longitudinal}
transform ∈ {rank}
gamma     ∈ {0.5, 1, 1.5, 2, 3, 4}
floor     ∈ {0.02, 0.05, 0.10, 0.20}
alpha     ∈ {10, 30, 100, 300, 1000}
beta      ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
```

Total:

```text
590,031 grid records
351 source-selected records
```

E43B additionally searched sigmoid weights:

```text
transform ∈ {sigmoid}
gamma     ∈ {0.5, 1, 2, 4}
floor     ∈ {0.02, 0.05, 0.10}
alpha     ∈ {10, 30, 100, 300, 1000}
beta      ∈ {0.4, ..., 1.0}
```

Total:

```text
295,191 grid records
```

E43C combined rank+sigmoid records without refitting and recomputed source-validation choices.

Important distinction:

- target-grid best is exploratory;
- source-selected is deployable-style but local;
- modal stability rule is the new candidate control structure.

## Main result: rank grid vs E42

### E42 current candidates

| Regime | Current E42 candidate | Gain | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---|---:|---:|---:|---:|---:|
| Lee2019 sensorimotor20 | `source rank g2 alpha100 beta0.7` | +2.500 | -6.250 | -6.875 | 7.4% | 24.1% |
| BNCI full-source | LDA | +0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| BNCI m8/class | `longitudinal rank g2 alpha100 beta0.7` | +2.470 | -5.645 | -6.201 | 6.3% | 26.0% |

### Best rank candidates under comparable harm

| Regime | Candidate | Gain | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---|---:|---:|---:|---:|---:|
| Lee2019 sensorimotor20 | `source rank g4 alpha300 beta0.5` | **+2.685** | -5.875 | -5.625 | **5.6%** | **22.2%** |
| BNCI full-source | `source rank g0.5 alpha300 beta0.6` | +0.717 | -1.290 | -1.613 | 0.0% | 22.2% |
| BNCI m8/class | `longitudinal rank g4 alpha30 beta0.6` | **+2.699** | -5.363 | -6.174 | **5.2%** | **24.0%** |

Interpretation:

```text
The useful direction is sharper reliability weighting, not softer weighting.
But sharper weighting must be paired with lower beta.
```

The pattern is consistent:

```text
old: gamma=2, beta=0.7
new: gamma=4, beta=0.5/0.6
```

This says the weighted model should be more selective internally, but the final decision should remain closer to the LDA source anchor.

## Modal stability rule

The local source-selected rule again failed:

| Regime | Local source-selected gain | P(gain<-5) | Interpretation |
|---|---:|---:|---|
| Lee2019 sensorimotor20 | +2.361 | 11.1% | worse than fixed/modal |
| BNCI full-source | -0.986 | 11.1% | bad; local selection overfits |
| BNCI m8/class | +2.195 | 8.3% | worse than fixed/modal |

So do not use local candidate switching.

Instead, E43C tested:

```text
For each regime:
  compute nested source-validation choices excluding held subject.
  take the modal candidate.
  apply it only if modal rate >= 0.70.
  otherwise use LDA.
```

Combined rank+sigmoid modal choices:

| Regime | Mode candidate | Mode rate | Applied |
|---|---|---:|---|
| Lee2019 sensorimotor20 | `source rank g4 alpha300 beta0.5` | 47/54 = 87.0% | mode |
| BNCI full-source | `longitudinal sigmoid g2 alpha300 beta0.4` | 6/9 = 66.7% | LDA |
| BNCI m8/class | `longitudinal rank g4 alpha30 beta0.6` | 224/288 = 77.8% | mode |

Modal rule performance:

| Regime | Applied candidate | Gain | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---|---:|---:|---:|---:|---:|
| Lee2019 sensorimotor20 | `source rank g4 alpha300 beta0.5` | **+2.685** | -5.875 | -5.625 | **5.6%** | **22.2%** |
| BNCI full-source | LDA | +0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| BNCI m8/class | `longitudinal rank g4 alpha30 beta0.6` | **+2.699** | -5.363 | -6.174 | **5.2%** | **24.0%** |

Compared with E42:

| Regime | Modal - E42 mean diff | 95% bootstrap CI | Notes |
|---|---:|---:|---|
| Lee2019 sensorimotor20 | +0.185pp | [-0.394, +0.741] | mean/risk improves, CI crosses 0 |
| BNCI full-source | 0.000pp | [0.000, 0.000] | keeps negative-control anchor |
| BNCI m8/class | +0.230pp | [-0.008, +0.479] | nearly positive, risk improves |

This is not a huge gain, but it is the first result that improves both positive regimes while preserving the full-source anchor.

## Sigmoid check

Sigmoid weights were useful as a diagnostic but should not replace rank weights as the main method.

Best sigmoid candidates:

| Regime | Best sigmoid under comparable harm | Gain | P(gain<-5) | Interpretation |
|---|---|---:|---:|---|
| Lee2019 sensorimotor20 | `source sigmoid g0.5 alpha300 beta1.0` | +2.431 | 5.6% | safer but weaker than rank |
| BNCI full-source | `longitudinal sigmoid g2 alpha300 beta0.4` | +0.896 | 0.0% | intriguing but n=9 only |
| BNCI m8/class | `longitudinal sigmoid g1 alpha300 beta1.0` | +2.470 | 5.2% | no mean improvement over E42 |

Interpretation:

```text
rank weighting remains the main positive-regime method.
sigmoid may reveal a full-source BNCI opportunity, but this is too fragile to claim yet.
```

BNCI full-source is especially dangerous:

- target-grid best sigmoid gives +0.896pp with no severe harm;
- local source-selected still loses;
- combined modal rate is 6/9 = 66.7%, just below 0.70.

Do not claim this yet. Treat it as a validation target.

## Mechanistic interpretation

E43 sharpens the hypothesis:

Old hypothesis:

```text
soft reliability weighting helps because it avoids hard feature deletion.
```

Updated hypothesis:

```text
Cross-session tangent features contain a small set of highly reliable dimensions.
The classifier should emphasize them strongly, but the final decision should be
anchored toward LDA to avoid overcommitting to the weighted model.
```

In short:

```text
stronger feature-side specialization
+ stronger output-side source anchor
```

This is a cleaner mechanism than the previous `g2 beta0.7` story.

## Keep / reject / revise

### Keep

```text
source-margin-calibrated interpolation
```

It remains essential.

### Keep and strengthen

```text
rank reliability weighting
```

But move from `gamma=2` to `gamma=4` candidates.

### Reject

```text
local source-selected candidate switching
```

It failed again, now with an even larger candidate set.

### Revise

The control structure should become:

```text
source-validation modal stability rule
```

not:

```text
per-subject selector
```

## Current best candidate

Accuracy-first / risk-aware frozen candidate:

```text
For a regime:
  compute nested source-validation choices over the E43 candidate grid.
  if top modal rate >= 0.70:
      apply the modal candidate to all units in that regime.
  else:
      use LDA.
```

With the current combined rank+sigmoid grid:

| Regime | Candidate |
|---|---|
| Lee2019 sensorimotor20 | `source rank gamma=4, floor=0.02/0.05, alpha=300, beta=0.5` |
| BNCI full-source | LDA |
| BNCI m8/class | `longitudinal rank gamma=4, floor=0.02/0.05, alpha=30, beta=0.6` |

The exact floor is not important in these runs; several floors tie.

## Next action

Do not run another broad selector.

Next should be E44:

```text
Freeze the modal-stability control rule and test its sensitivity.
```

Required checks:

1. threshold sensitivity:
   - modal threshold ∈ {0.60, 0.67, 0.70, 0.75, 0.80}
   - BNCI full-source is the critical edge case.

2. candidate-grid shrinkage:
   - broad E43 grid may overfit source-validation;
   - test a smaller pre-defined grid:

```text
family ∈ {source, longitudinal}
transform = rank
gamma ∈ {2, 3, 4}
alpha ∈ {30, 100, 300}
beta ∈ {0.4, 0.5, 0.6, 0.7}
floor = 0.05
+ LDA
```

3. compare:
   - E42 final;
   - E43 target best;
   - E43 local source-selected;
   - E44 modal-stability rule.

Pass criterion:

```text
Lee gain >= +2.60 and P(gain<-5) <= 6%
BNCI full-source stays LDA or P(gain<-5)=0 with clear modal support
BNCI m8 gain >= +2.60 and P(gain<-5) <= 5.5%
```

If E44 passes, the method story becomes substantially stronger than E42:

```text
source-validation should not personalize decisions,
but it can identify a stable regime-level reliability-weight/interpolation configuration.
```

