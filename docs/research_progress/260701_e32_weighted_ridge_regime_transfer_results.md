# E32: weighted Ridge regime-transfer results

Date: 2026-07-01

## One-line verdict

E31の `ridge_source_rank_g2_a100` は、Lee2019 all62だけの偶然ではない。

ただし、これは万能な「常に勝つ適応器」ではない。現時点で守るべき主張は:

```text
soft source/longitudinal reliability weighting + Ridge is useful in
source-scarce or cross-session fragile tangent regimes,
but low-p/full-source regimes still prefer simple LDA baselines.
```

つまり、前の

```text
source-side longitudinal subspace selection
```

よりも、今の主役は

```text
source-side reliability-weighted linear regularization
```

に寄せるべき。

## Artifacts

Script:

`intentflow/offline/scripts/analysis/weighted_ridge_regime_transfer_e32.py`

Outputs:

- `intentflow/offline/results/research_outputs/260701_e32_lee2019_sensorimotor20_weighted_ridge/`
- `intentflow/offline/results/research_outputs/260701_e32_bnci2014_001_full_source_weighted_ridge/`
- `intentflow/offline/results/research_outputs/260701_e32_bnci2014_001_m8_weighted_ridge/`

Main files:

- `weighted_ridge_transfer_summary.csv`
- `weighted_ridge_transfer_records.csv`
- `summary.json`

## Protocol

Fixed E31-derived method:

```text
ridge_source_rank_g2_a100
```

was evaluated without retuning on:

1. Lee2019 sensorimotor20
   - 54 subjects
   - p = 210
   - all source trials
2. BNCI2014_001 full-source
   - 9 subjects
   - p = 253
   - all source trials
3. BNCI2014_001 source-scarce simulation
   - 9 subjects
   - p = 253
   - 8 trials/class
   - 32 repeats

Comparators:

- Ridge full
- Ridge source-rank weighted
- Ridge longitudinal-rank weighted
- Ridge hard source/longitudinal top-k
- LDA full
- LDA source/longitudinal top-k

## Main results

### Lee2019 sensorimotor20

| Method | Acc | Gain vs Ridge full | Gain vs LDA full | P(gain vs LDA full < -5pp) |
|---|---:|---:|---:|---:|
| `ridge_source_rank_g2_a100` | **71.991** | **+4.375** CI [2.778, 6.042] | **+2.384** CI [0.787, 3.959] | 9.3% |
| `ridge_longitudinal_rank_g2_a100` | 71.829 | +4.213 | +2.222 | n/a |
| `ridge_source_rank_g1_a100` | 71.273 | +3.657 | +1.667 | n/a |
| `lda_long_q70` | 70.579 | n/a | +0.972 | n/a |
| `lda_source_q25` | 69.699 | n/a | +0.093 | n/a |
| `lda_full` | 69.606 | n/a | 0.000 | 0.0% |
| `ridge_full_a100` | 67.616 | 0.000 | -1.991 | n/a |

This is the most surprising E32 result.

Earlier E20/E21 suggested that sensorimotor20 was already compact, so source-side hard selection should not help much. That remains true for LDA hard selection:

```text
LDA source_q25: 69.699
LDA full:       69.606
```

But soft weighted Ridge is different:

```text
ridge_source_rank_g2_a100: 71.991
```

So the effect is not merely:

```text
remove high-dimensional all-channel noise
```

It is closer to:

```text
Ridge benefits from continuous reliability weighting in cross-session tangent features,
even when the channel set is already sensorimotor-compact.
```

### BNCI2014_001 full-source

| Method | Acc | Gain vs LDA full |
|---|---:|---:|
| `lda_long_q70` | **78.047** | +0.448 CI [-0.717, 1.703] |
| `lda_full` | 77.599 | 0.000 |
| `ridge_source_rank_g1_a100` | 77.061 | -0.538 |
| `ridge_source_rank_g2_a100` | 77.061 | -0.538 |
| `ridge_longitudinal_rank_g2_a100` | 76.971 | -0.627 |
| `ridge_full_a100` | 74.552 | -3.047 |

This is a negative transfer check.

The E31 fixed weighted Ridge does not beat LDA in BNCI full-source.
This weakens any universal-method story, but strengthens the regime story:

```text
when p is modest and source data are sufficient, simple LDA is already strong.
weighted Ridge is unnecessary or slightly worse.
```

### BNCI2014_001 m8/class source-scarce

| Method | Acc | Gain vs Ridge full | Gain vs LDA full | P(gain vs LDA full < -5pp) |
|---|---:|---:|---:|---:|
| `ridge_longitudinal_rank_g2_a100` | **70.010** | **+2.198** CI [0.230, 4.441] | **+2.327** CI [0.356, 4.525] | 0.0% |
| `ridge_source_rank_g2_a100` | 69.738 | +1.927 CI [0.224, 3.968] | +2.055 CI [0.375, 4.080] | 0.0% |
| `ridge_source_rank_g1_a100` | 69.554 | +1.742 | +1.871 CI [0.731, 3.332] | n/a |
| `ridge_hard_long_q10_a100` | 69.302 | +1.490 | +1.619 | n/a |
| `ridge_hard_source_q25_a100` | 69.184 | +1.372 | +1.501 | n/a |
| `lda_long_q10` | 68.730 | n/a | +1.047 CI [-0.753, 3.075] | n/a |
| `lda_source_q25` | 68.702 | n/a | +1.019 CI [0.042, 2.338] | n/a |
| `ridge_full_a100` | 67.812 | 0.000 | +0.129 | n/a |
| `lda_full` | 67.683 | n/a | 0.000 | 0.0% |

This is the cleanest support for the new method direction.

In source-scarce BNCI, soft weighted Ridge beats:

- LDA full
- Ridge full
- hard source top-k
- hard longitudinal top-k

Also, the best score family changes:

```text
Lee2019: source_rank_g2 is best
BNCI m8: longitudinal_rank_g2 is best
```

So the strong claim is not "source score is always best".
The strong claim is:

```text
soft reliability weighting is better than hard deletion/full features;
the reliability score should be selected by source-side validation or regime rule.
```

## What to keep / revise / retract

### Keep

```text
Soft reliability-weighted Ridge is a serious method candidate.
```

It now has evidence from:

- Lee2019 all62 exact nested validation: E31
- Lee2019 sensorimotor20 external regime: E32
- BNCI2014_001 source-scarce regime: E32

### Revise

The mechanism is not only high-dimensional all-channel denoising.

Old story:

```text
high p tangent features contain noisy dimensions, so select reliable subspace.
```

Revised story:

```text
cross-session tangent features have uneven feature reliability.
Hard deletion is brittle; full Ridge/LDA overuses unreliable dimensions.
Continuous source/longitudinal reliability weighting gives the linear classifier
a better inductive bias, especially when source data are scarce or the session
shift is fragile.
```

### Retract / weaken

Do not claim:

```text
ridge_source_rank_g2_a100 is universally best.
```

BNCI full-source prefers LDA.
BNCI m8 prefers longitudinal weighting over source-only weighting.

## Research implication

The best next method should not be another hard selector.

The more promising direction is:

```text
Reliability-Weighted Regularized Adaptation for Zero-Target-Label Cross-Session EEG-MI
```

Core ingredients:

1. compute source-side and/or longitudinal feature reliability from source sessions only;
2. convert reliability into continuous feature weights;
3. train a regularized linear classifier on weighted tangent features;
4. choose the score family/strength by nested source-session validation;
5. evaluate on held target subjects/sessions with no target labels.

## Next experiment: E33

E33 should test whether the score family can be selected honestly.

### Why E33 is necessary

E32 shows:

```text
Lee2019 best: source_rank_g2
BNCI m8 best: longitudinal_rank_g2
BNCI full best: LDA baseline
```

If we hard-code the best per dataset after seeing target results, the method is not publishable.

So E33 must answer:

```text
Can source-side validation decide when to use source-weighted Ridge,
longitudinal-weighted Ridge, or LDA?
```

### E33 design

For each dataset/regime:

1. outer held subject/session;
2. inner source-validation over remaining subjects;
3. candidate family:
   - `lda_full`
   - `lda_source_q25`
   - `lda_long_q10`
   - `lda_long_q70`
   - `ridge_source_rank_g1_a100`
   - `ridge_source_rank_g2_a100`
   - `ridge_longitudinal_rank_g2_a100`
4. choose by:
   - mean inner accuracy;
   - risk-constrained mean, e.g. maximize mean under `P(diff vs lda_full < -5pp) <= 0.20`;
5. report:
   - accuracy;
   - paired gain vs `lda_full`;
   - paired gain vs best fixed baseline known before E33;
   - q05 gain;
   - `P(gain<-5pp)`.

### Decision criterion

E33 succeeds if nested selection:

- matches or nearly matches the best fixed method within 0.5pp;
- beats `lda_full` on Lee2019 and BNCI m8;
- does not choose weighted Ridge on BNCI full-source if LDA is better;
- does not increase `P(gain<-5pp)` beyond the best fixed method by more than 5 percentage points.

If E33 fails, the honest conclusion is:

```text
weighted Ridge is promising but regime-dependent;
we need either a better regime diagnostic or present it as an analysis result,
not as an automatic method.
```

