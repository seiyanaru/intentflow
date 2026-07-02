# E22: Lee2019 category-constrained q0.25 control results

Date: 2026-06-30

## One-line verdict

E22 partially supports the E21 category story, but not strongly enough to claim
a validated new method yet.

```text
yes:  category-constrained q0.25 beats same-size random pruning
no:   it does not clearly beat source-score top-k or the original q0.25 policy
so:   this is a promising mechanistic clue, not yet a publishable method claim
```

The result is useful because it rules out the weakest alternative explanation:

> The E21 gain is not just "remove any 87 random dimensions from q0.25".

But it does not rule out the stronger alternative:

> The gain may simply be another compact source-score subspace close to q0.20,
> with a neuroscience-looking mask on top.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/lee2019_category_control_e22.py`

Output:

`intentflow/offline/results/research_outputs/260630_lee2019_category_control_e22_r20_keep/`

Main files:

- `summary.csv`
- `summary.json`
- `paired_contrasts.csv`
- `random_family_summary.csv`
- `selection_records.csv`
- `selection_size_by_subject.csv`
- `subject_level_deltas.csv`

## Experimental design

For each held-out Lee2019 subject:

1. Compute source-side scores from the other 53 subjects.
2. Select `source_only_q0p25`.
3. Define the E21 neuro-category subset:
   - keep selected covariance dimensions involving sensorimotor or posterior
     channels;
   - drop selected `other_other` dimensions.
4. Compare against:
   - full tangent baseline;
   - original `source_only_q0p25`;
   - same-size `score_top_same_k`;
   - same-size random keep-k control, 20 repeats.

Important detail:

`random_keep_k` and `random_drop_same_count` have the same marginal
distribution when the retained size is fixed.  I therefore used the keep-k
control only to avoid redundant LDA fits.

## Selection size

Full tangent dimension: 1953.

Original `source_only_q0p25`: 489 dimensions.

Category-constrained subset:

| Quantity | Value |
|---|---:|
| mean kept dimensions | 401.6 |
| min kept dimensions | 396 |
| max kept dimensions | 415 |
| mean dropped dimensions | 87.4 |
| mean kept fraction of q0.25 | 82.1% |
| equivalent full fraction | about q0.206 |

The category rule is not an extreme compression.  It removes a fairly stable
~18% of the already-selected q0.25 dimensions.

## Main performance

| Method | n selected | Acc | Gain vs full | 95% CI | q05 gain | P(gain < -5) |
|---|---:|---:|---:|---:|---:|---:|
| full_q1p00 | 1953 | 69.676 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 |
| random_keep_k_mean | 401.6 | 70.786 | +1.110 | [-0.341, 2.642] | -7.547 | 0.093 |
| score_top_same_k | 401.6 | 71.458 | +1.782 | [0.139, 3.495] | -6.250 | 0.130 |
| source_only_q0p25 | 489.0 | 71.852 | +2.176 | [0.509, 3.889] | -6.250 | 0.074 |
| category_keep_motor_or_posterior | 401.6 | 72.176 | +2.500 | [0.718, 4.214] | -7.938 | 0.111 |

Nominal ranking by mean gain:

```text
category_keep_motor_or_posterior  +2.500
source_only_q0p25                 +2.176
score_top_same_k                  +1.782
random_keep_k_mean                +1.110
full                              +0.000
```

## Paired contrasts

| Contrast | Mean delta | 95% CI | Pos / Neg / Zero | Verdict |
|---|---:|---:|---:|---|
| category - source_only_q0p25 | +0.324 | [-0.486, 1.134] | 24 / 21 / 9 | not clear |
| category - score_top_same_k | +0.718 | [-0.509, 2.014] | 26 / 20 / 8 | not clear |
| category - random_keep_k_mean | +1.390 | [0.540, 2.168] | 38 / 16 / 0 | clear |

This is the key result.

The category rule clearly beats random pruning of the same size.  However, it
does not clearly beat the stronger compactness control `score_top_same_k`, and
it does not clearly beat simply keeping all q0.25 selected dimensions.

## Subject-level behavior

Category vs random is relatively stable:

- category beats random in 38 / 54 subjects;
- median category-minus-random is +1.41 pp;
- q05 is -2.15 pp.

Category vs q0.25 is not stable:

- category beats q0.25 in 24 / 54 subjects;
- loses in 21 / 54;
- ties in 9 / 54;
- median is 0 pp.

Category vs score top-k is also not stable:

- category beats score top-k in 26 / 54;
- loses in 20 / 54;
- ties in 8 / 54;
- median is 0 pp.

Worst category cases:

| Subject | Full | Category | Category gain | q0.25 gain | score-top-k gain |
|---:|---:|---:|---:|---:|---:|
| 7 | 76.25 | 65.00 | -11.25 | -10.00 | -2.50 |
| 15 | 47.50 | 37.50 | -10.00 | -5.00 | -3.75 |
| 48 | 65.00 | 56.25 | -8.75 | -12.50 | -6.25 |
| 51 | 65.00 | 57.50 | -7.50 | -6.25 | -6.25 |
| 49 | 77.50 | 71.25 | -6.25 | +2.50 | -2.50 |

Best category cases:

| Subject | Full | Category | Category gain | q0.25 gain | score-top-k gain |
|---:|---:|---:|---:|---:|---:|
| 9 | 56.25 | 72.50 | +16.25 | +12.50 | +13.75 |
| 10 | 67.50 | 83.75 | +16.25 | +13.75 | +10.00 |
| 28 | 75.00 | 90.00 | +15.00 | +15.00 | +18.75 |
| 5 | 76.25 | 91.25 | +15.00 | +12.50 | +11.25 |
| 22 | 58.75 | 72.50 | +13.75 | +10.00 | +15.00 |

The winners and losers are large.  This is not a tiny numerical perturbation.
The method changes individual-subject outcomes substantially.

## Risk interpretation

This is not yet a safety method.

Compared with original `source_only_q0p25`:

- mean gain improves from +2.176 to +2.500;
- but q05 worsens from -6.250 to -7.938;
- P(gain < -5) worsens from 0.074 to 0.111.

So the honest framing is:

```text
category constraint may improve compact mean accuracy,
but it does not improve lower-tail safety on Lee2019.
```

If the paper story is "safe adaptation", this result is not enough.  If the
paper story is "neuro-constrained source-side subspace selection for compact
zero-label cross-session EEG-MI", this is promising but still needs external
validation.

## What this result says about the hypothesis

Original E21 hypothesis:

> The useful all-channel Lee2019 subspace is broad but neurostructured:
> sensorimotor covariance pairs form the core, posterior-related pairs provide
> useful context, and selected neither-motor-nor-posterior pairs are mostly
> noise.

E22 updates it to:

> The selected neither-motor-nor-posterior dimensions are worse than a fixed
> motor/posterior category subset when compared against random same-size
> pruning.  However, the category subset has not yet proven that neuroscience
> categories add information beyond source-score compactness.

This is a narrower and safer hypothesis.

## What not to claim

Do not claim:

- "we found the optimal neurophysiological subspace";
- "posterior channels are generally helpful";
- "category-constrained selection is safer";
- "the method is validated".

The data do not support those.

Safe claim:

> In Lee2019, the gain from source-side subspace selection is not explained by
> arbitrary dimensionality reduction.  The selected dimensions show a
> motor/posterior structure, and removing selected dimensions outside that
> structure is better than random same-size pruning, but not yet clearly better
> than score-based compactness.

## Next decision

The next experiment should not be another selector tweak on Lee2019.

The bottleneck is now external validity:

```text
Does the fixed motor/posterior category rule transfer beyond Lee2019?
```

Recommended E23:

1. Freeze the rule:
   - source-side `source_only_q0p25`;
   - keep only selected dimensions involving sensorimotor or posterior channels;
   - no tuning of fraction, posterior list, or category after seeing results.
2. Run the same category-vs-controls evaluation on another dataset where channel
   taxonomy is mappable:
   - Stieger broad all-channel setting, if the channel list can be mapped;
   - BNCI2014_001 as a weaker but useful stress test.
3. Decision criterion:
   - if category beats same-size random and matches/exceeds score-top-k on a
     second dataset, it becomes a real method candidate;
   - if it fails externally, keep it as a Lee2019 mechanism audit only and do
     not build the thesis around the category rule.

