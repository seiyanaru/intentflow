# E31: exact nested validation of Lee2019 weighted Ridge

Date: 2026-07-01

## 結論

E30のbestだった

```text
ridge_source_rank_g2_l0_a100
```

は、post-hocに選ばれただけではなかった。

E31の exact nested source-validation で、全54 held-out subjectsに対して、

```text
nested_mean:
  ridge_source_rank_g2_l0_a100 selected 54/54

nested_risk20:
  ridge_source_rank_g2_l0_a100 selected 54/54
```

だった。

つまり、source側だけで見てもこの hyperparameter が一貫して最良。

ただし、LDA `source_only_q0p25` に対する paired gain は平均では上だが、CIはまだ0を跨ぐ。

```text
weighted Ridge nested - LDA source_only_q0p25:
  +0.764pp
  95% CI [-0.139, +1.690]
```

したがって判断は:

```text
守る:
  source-score soft-weighted Ridge is a serious Lee2019 method candidate.

まだ言えない:
  statistically decisive improvement over LDA source_only_q0p25 on Lee2019 alone.
```

## Artifacts

Script:

`intentflow/offline/scripts/analysis/lee2019_weighted_ridge_nested_e31.py`

Merge script:

`intentflow/offline/scripts/analysis/merge_lee2019_weighted_ridge_nested_e31.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_lee2019_weighted_ridge_nested_e31/`

Merged:

`intentflow/offline/results/research_outputs/260701_lee2019_weighted_ridge_nested_e31/merged/`

Main files:

- `nested_selection_records.csv`
- `fixed_candidate_records.csv`
- `source_validation_records.csv`
- `nested_weighted_ridge_summary.csv`
- `paired_comparisons.csv`
- `summary.json`

## Protocol

For each held subject H:

1. Load all 54 subjects.
2. For each inner source-validation subject S != H:
   - compute source-side score from all subjects excluding H and S;
   - compute optional target-prefix penalty from S prefix only, without labels;
   - evaluate weighted Ridge on S session0 -> session1.
3. Select candidate by:
   - `nested_mean`: max source-validation mean accuracy;
   - `nested_risk20`: max source-validation mean accuracy under `P(gain vs same-alpha full < -5pp) <= 0.20`.
4. Recompute scores excluding H.
5. Evaluate once on held H.

Candidate grid:

```text
alpha ∈ {0.1, 1, 10, 100}
gamma ∈ {0.5, 1, 2}
prefix_lambda ∈ {0, 0.25}
plus ridge_full for each alpha
```

Total:

```text
28 candidates
```

## Main result

| Method | Acc | 95% CI | q05 |
|---|---:|---:|---:|
| nested_mean | **72.616** | [68.634, 76.737] | 49.563 |
| nested_risk20 | **72.616** | [68.634, 76.737] | 49.563 |
| fixed `ridge_source_rank_g2_l0_a100` | **72.616** | [68.634, 76.737] | 49.563 |
| fixed `ridge_source_rank_g1_l0_a100` | 72.292 | [68.472, 76.319] | 52.500 |
| fixed `ridge_source_rank_g2_l0_a10` | 72.199 | [68.379, 76.181] | 50.813 |
| LDA `source_only_q0p25` reference | 71.852 | previous E25 | n/a |
| LDA full reference | 69.676 | previous E25 | n/a |

Nested and fixed are identical because source-validation selected the same method for every held subject.

## Paired comparisons

| Comparison | Mean diff | 95% CI | q05 diff | P(diff<-5) |
|---|---:|---:|---:|---:|
| nested_mean - LDA source_only_q0p25 | +0.764 | [-0.139, +1.690] | -5.000 | 1.9% |
| nested_risk20 - LDA source_only_q0p25 | +0.764 | [-0.139, +1.690] | -5.000 | 1.9% |
| weighted Ridge - LDA full | +2.940 | [+1.319, +4.537] | -5.438 | 9.3% |
| weighted Ridge - LDA longitudinal_q0p10 | +1.088 | [-0.347, +2.546] | -6.250 | 11.1% |

## Interpretation

### What E31 confirms

E30's best method was not merely target leakage or post-hoc hyperparameter cherry-picking within Lee2019.

The source-validation objective consistently selects:

```text
source-score soft weighting
gamma = 2
prefix_lambda = 0
alpha = 100
```

This also confirms that target-prefix penalty is not currently useful in this form.

### Why this is a better method than hard top-k

Hard `source_only_q0p25` says:

```text
keep top 25%, delete the rest
```

E31 best says:

```text
keep all features, but make Ridge regularization respect source-side feature reliability.
```

This matches the Lee mechanism:

- useful information is broader than pure motor channels;
- full tangent includes noisy dimensions;
- hard deletion can remove useful context;
- soft weighting is a better middle ground.

## Current claim strength

### Strong claim

```text
Source-score soft-weighted Ridge is validated by exact source-side nested selection on Lee2019.
```

### Moderate claim

```text
It improves mean accuracy over the previous LDA source_only_q0p25 baseline by +0.76pp.
```

### Weak / not yet claimable

```text
It is statistically decisively better than LDA source_only_q0p25 on Lee2019 alone.
```

Because the paired CI still crosses zero.

## Next action

E32 should test external/regime transfer:

1. Lee2019 sensorimotor20:
   - prediction: soft weighting should give little or no benefit if full tangent is already compact.
2. BNCI2014_001 full-source:
   - prediction: little or no benefit in low p/n regime.
3. BNCI2014_001 m8/class:
   - prediction: source-score soft weighting may help, possibly more stably than hard q0.10.

If this pattern holds, the method story becomes:

```text
Source-score soft-weighted Ridge improves high-dimensional/source-scarce tangent regimes
by replacing brittle hard subspace deletion with continuous source-informed regularization.
```

