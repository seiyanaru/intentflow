# E21: Lee2019 selected channel-pair category ablation results

Date: 2026-06-30

## One-line verdict

E21 clarifies the mechanism behind the Lee2019 all-channel `source_only_q0p25`
gain:

> The gain is not produced by posterior-only or sensorimotor-only features.
> It requires a broad selected subspace whose core includes sensorimotor
> covariance pairs and whose useful context includes posterior-related pairs.

In other words:

```text
not:  hand-defined motor channels are enough
not:  posterior channels alone are useful
yes:  source-side selection keeps a broader motor + posterior/mixed covariance
      subspace and removes irrelevant all-channel covariance noise
```

This is consistent with E20:

- sensorimotor20 full tangent is already compact and competitive;
- all62 selection gains disappear if the representation is pre-restricted to
  sensorimotor channels;
- the selected all62 subspace needs information outside pure sensorimotor
  pairs, but not arbitrary non-motor/non-posterior pairs.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/lee2019_channel_pair_category_e21.py`

Output:

`intentflow/offline/results/research_outputs/260630_lee2019_channel_pair_category_e21/`

Main files:

- `summary.csv`
- `summary.json`
- `selection_records.csv`
- `category_counts.csv`
- `selected_feature_categories.csv`

## Category definitions

Sensorimotor:

```text
FC5, FC3, FC1, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6,
CP5, CP3, CP1, CPz, CP2, CP4, CP6
```

Posterior:

```text
P7, P3, Pz, P4, P8, P1, P2,
PO9, PO10, POz, PO3, PO4,
O1, Oz, O2
```

Pair categories:

- `both_sensorimotor`
- `sensorimotor_posterior`
- `both_posterior`
- `sensorimotor_other`
- `posterior_other`
- `other_other`

## Composition of selected dimensions

### `source_only_q0p25`

Average selected dimensions: 489.

| Category | Mean count | Fraction |
|---|---:|---:|
| both_sensorimotor | 118.6 | 24.3% |
| sensorimotor_other | 111.6 | 22.8% |
| other_other | 87.4 | 17.9% |
| both_posterior | 72.6 | 14.9% |
| sensorimotor_posterior | 70.2 | 14.4% |
| posterior_other | 28.5 | 5.8% |

The selection is broad.  Only about one quarter is pure sensorimotor-sensorimotor.
About 20% directly involves posterior channels.

### `longitudinal_q0p10`

Average selected dimensions: 196.

| Category | Mean count | Fraction |
|---|---:|---:|
| both_sensorimotor | 75.7 | 38.6% |
| sensorimotor_other | 40.9 | 20.9% |
| both_posterior | 33.7 | 17.2% |
| sensorimotor_posterior | 28.8 | 14.7% |
| other_other | 14.6 | 7.5% |
| posterior_other | 2.3 | 1.2% |

Longitudinal is more sensorimotor-concentrated and more compact.

## Performance ablation: `source_only_q0p25`

Baseline:

| Method | n selected | Acc | Gain vs full | P(gain<-5) |
|---|---:|---:|---:|---:|
| full_q1p00 | 1953 | 69.676 | 0.000 | 0.000 |
| source_only_q0p25 all selected | 489 | 71.852 | +2.176 | 0.074 |

Category-only subsets:

| Subset | n selected | Gain vs full | Interpretation |
|---|---:|---:|---|
| only both_sensorimotor | 118.6 | +0.440 | weak alone |
| only any_sensorimotor | 300.4 | +0.093 | almost no gain |
| only any_posterior | 171.4 | -7.361 | bad alone |
| only both_posterior | 72.6 | -10.532 | bad alone |
| only sensorimotor_posterior | 70.2 | -10.787 | bad alone |
| only no_sensorimotor | 188.6 | -10.579 | bad alone |
| only motor_or_posterior | 401.6 | +2.500 | slightly above all selected, exploratory |

Leave-category-out subsets:

| Subset | n selected | Gain vs full | Delta vs all selected | Interpretation |
|---|---:|---:|---:|---|
| drop both_sensorimotor | 370.4 | -4.769 | -6.944, CI [-8.866, -5.069] | sensorimotor core is essential |
| drop any_posterior | 317.6 | -0.949 | -3.125, CI [-5.185, -1.134] | posterior-related dimensions are necessary in combination |
| drop sensorimotor_posterior | 418.8 | +1.574 | -0.602, CI [-1.413, +0.162] | mixed motor-posterior helps but is not the whole story |
| drop no_sensorimotor | 300.4 | +0.093 | -2.083, CI [-3.912, -0.301] | non-sensorimotor dimensions matter |

Key message:

```text
both_sensorimotor pairs are necessary, but not sufficient.
posterior-related pairs are also necessary in combination, but not useful alone.
other_other pairs are likely noisy: removing them gives +2.50 pp, slightly above
the full selected subspace, but this is exploratory and must not be claimed as
a validated method yet.
```

## Relation to E20

E20 showed:

| Feature | Method | p | Gain |
|---|---|---:|---:|
| all62 | source_only_q0p25 | 1953 | +2.176 |
| sensorimotor20 | source_only_q0p25 | 210 | +0.093 |

E21 explains why:

- the all62 selected subspace is not equivalent to sensorimotor restriction;
- pure sensorimotor selected dimensions alone are weak;
- posterior-related dimensions are harmful alone but necessary in combination;
- arbitrary `other_other` dimensions are probably noise.

So the mechanism is not simply:

```text
select motor channels
```

It is closer to:

```text
select a broad but denoised covariance subspace anchored by motor pairs and
augmented by posterior/mixed context.
```

## Scientific implication

This gives a better novelty story:

> Source-side tangent subspace selection does not just shrink the feature space
> or hand-select motor channels.  It discovers a cross-session useful covariance
> subspace that is broader than canonical motor regions but cleaner than the
> full all-channel tangent representation.

This is much stronger than the earlier "source_only q0.25 works on Lee" result.

## Caution

The `only_motor_or_posterior` result is post-hoc:

- `only_motor_or_posterior`: +2.500 pp
- original all selected: +2.176 pp

This suggests `other_other` dimensions may be removable.  But it is not yet a
validated method because the category rule was chosen after seeing E20/E21.

If we want to claim it as a method, it needs a nested evaluation or external
validation.

## Next action

Do not immediately build another selector.

The clean next validation is:

1. Define a fixed, pre-registered category-constrained policy:
   - start from source-side `source_only_q0p25`;
   - keep only pairs involving sensorimotor or posterior channels;
   - drop `other_other`.
2. Evaluate it without tuning on:
   - Lee2019 all62;
   - BNCI2014_001 if channel taxonomy can be mapped;
   - optionally Stieger broad_all60.

Decision criterion:

- If category-constrained q0.25 improves or matches Lee with lower harm, it can
  become a meaningful method variant.
- If it fails externally, keep E21 as mechanism analysis only.
