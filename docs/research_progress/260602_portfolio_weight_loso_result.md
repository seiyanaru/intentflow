# 2026-06-02 Portfolio weight LOSO result

## Question

The static portfolio

```text
p = 0.3 p_source + 0.4 p_fullEA + 0.3 p_shrinkEA
```

worked well on BCIC2a seed0. Is this just a dataset-specific fit?

## Implemented

`intentflow/offline/scripts/analysis/eval_portfolio_weight_grid.py`

This evaluates convex weights over selected experts:

```text
w_source + w_fullEA + w_shrinkEA = 1
w_k >= 0
```

It reports:

- global grid search
- leave-one-subject-out weight selection
- held-out accuracy
- harm count vs source and full EA

## Main results

Experts:

```text
source, full_ea, shrink_0.1
```

### Static baselines

| Method | Mean acc | Harm vs source | Harm vs full EA |
|---|---:|---:|---:|
| source | 82.72 | 0 | 6 |
| full EA | 83.06 | 3 | 0 |
| uniform 1/3 each | 85.73 | 1 | 1 |
| static 0.3/0.4/0.3 | 85.76 | 1 | 1 |

### Global grid, step 0.01

Best global no-harm portfolio:

```text
p = 0.46 p_source + 0.43 p_fullEA + 0.11 p_shrinkEA
```

Result:

```text
mean acc: 85.96
harm vs source: 0
harm vs full EA: 0
```

This is stronger than 0.3/0.4/0.3 on BCIC2a, but it is still selected using all
subjects and should be treated as an oracle-like analysis baseline.

### LOSO selection

If weights are selected on the other 8 subjects and tested on the held-out
subject:

| Objective | Grid | LOSO mean | Readout |
|---|---:|---:|---|
| mean accuracy | 0.05 | 84.26 | unstable; badly misses S2/S6 |
| worst accuracy | 0.05 | 84.45 | too full-EA heavy; hurts source-favorable subjects |
| mean minus harm | 0.05 | 85.80 | best LOSO setting, close to static |
| mean minus harm | 0.01 | 85.61 | still stable-ish, but not better than static |

## Interpretation

The important result is not that a single ratio is universally optimal.

The robust result is:

1. Source / full EA / shrink EA are complementary.
2. Convex portfolio fusion is stronger and safer than picking one expert.
3. Subject-wise weight learning from only 8 subjects is unstable.
4. A conservative fixed portfolio is a strong baseline.
5. Any adaptive router must beat ~85.8 and avoid increasing harm count.

This argues against "learn a dataset-specific weight vector" as the final
method. The stronger design is:

```text
base portfolio:
  p = w_source p_source + w_full p_fullEA + w_shrink p_shrinkEA

default:
  conservative fixed weights around source/full dominant, shrink small

adaptive correction:
  only change weights when reliability/collapse diagnostics indicate risk
```

## Next

Evaluate this same portfolio principle on BCIC2b/HGD when logits exist. If
weights shift across datasets, that supports a reliability-conditioned router.
If the same source/full-dominant portfolio remains strong, then the main method
can be a simple "safe adaptation portfolio" with optional TTA gating.

