# E33: source-side nested family selection for weighted Ridge

Date: 2026-07-01

## One-line verdict

E33の結論は:

```text
weighted Ridge の有効性は守れる。
ただし、per-subject/per-repeatで細かく候補を切り替えるselectorは弱い。
source-side validationで安定にregime-level familyを選ぶ方が筋が良い。
```

これはかなり重要な方向修正。

勝ち筋は「賢い細粒度selector」ではなく、

```text
Stable source-validated reliability-weighted Ridge
```

としてまとめるべき。

## Artifacts

Script:

`intentflow/offline/scripts/analysis/weighted_ridge_nested_selector_e33.py`

Outputs:

- `intentflow/offline/results/research_outputs/260701_e33_lee2019_sensorimotor20_nested_selector/`
- `intentflow/offline/results/research_outputs/260701_e33_bnci2014_001_full_source_nested_selector/`
- `intentflow/offline/results/research_outputs/260701_e33_bnci2014_001_m8_nested_selector/`
- `intentflow/offline/results/research_outputs/260701_e33_nested_selector_combined_summary.csv`

Main files:

- `nested_selector_summary.csv`
- `nested_outer_records.csv`
- `nested_inner_validation_records.csv`
- `nested_selection_records.csv`
- `stable_nested_summary.csv`
- `stable_nested_outer_records.csv`
- `stable_nested_selection_records.csv`

## Protocol

For each outer held subject:

1. hold out the subject completely;
2. use remaining subjects as source-side validation subjects;
3. compute source/longitudinal reliability scores excluding both outer held subject and inner validation subject;
4. evaluate candidates on inner validation subject source session -> target session;
5. select candidate by:
   - `nested_mean`: maximum inner mean accuracy;
   - `nested_risk20`: maximum inner mean accuracy under `P(gain vs LDA full < -5pp) <= 0.20`;
6. evaluate selected candidate on outer held subject.

Candidates:

```text
lda_full
lda_source_q25
lda_long_q10
lda_long_q70
ridge_source_rank_g1_a100
ridge_source_rank_g2_a100
ridge_longitudinal_rank_g2_a100
```

Fixed comparators also include Ridge full and hard Ridge top-k variants.

Two nested variants were saved:

1. `nested_*`
   - select separately for each outer subject and repeat;
2. `stable_nested_*`
   - average all inner repeats for each outer subject, then select once per outer subject.

The stable variant is important for source-scarce simulation, because per-repeat selection overreacts to random subset noise.

## Results

### Lee2019 sensorimotor20

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `ridge_source_rank_g2_a100` | **71.991** | **+2.384** | -6.250 | 9.3% |
| `ridge_longitudinal_rank_g2_a100` | 71.829 | +2.222 | -6.250 | 13.0% |
| `nested_mean` / `nested_risk20` | 71.644 | +2.037 | -6.250 | 11.1% |
| `lda_long_q70` | 70.579 | +0.972 | -5.438 | 7.4% |
| `lda_full` | 69.606 | 0.000 | 0.000 | 0.0% |

Selection:

```text
ridge_source_rank_g2_a100        49/54
ridge_longitudinal_rank_g2_a100   5/54
```

Interpretation:

```text
source-side validation mostly identifies the right family.
Nested loses only 0.35pp vs fixed best, while keeping +2.04pp over LDA full.
```

This is a success.

### BNCI2014_001 full-source

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `lda_long_q70` | **78.047** | +0.448 | -1.774 | 0.0% |
| `lda_full` | 77.599 | 0.000 | 0.000 | 0.0% |
| `ridge_source_rank_g2_a100` | 77.061 | -0.538 | -4.839 | 0.0% |
| `ridge_longitudinal_rank_g2_a100` | 76.971 | -0.627 | -4.194 | 0.0% |
| `nested_mean` / `nested_risk20` | 76.882 | -0.717 | -3.548 | 0.0% |

Selection:

```text
lda_full                         7/9
ridge_source_rank_g1_a100        1/9
ridge_source_rank_g2_a100        1/9
```

Inner validation aggregate ranked `lda_full` best, not `lda_long_q70`.
So the fact that held-out outer result has `lda_long_q70` at +0.45pp is probably not a stable enough signal with only 9 subjects.

Interpretation:

```text
BNCI full-source is not a weighted Ridge regime.
Source validation correctly avoids selecting weighted Ridge most of the time.
```

This is not a method win, but it supports the regime story.

### BNCI2014_001 m8/class

Per-repeat nested:

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `ridge_longitudinal_rank_g2_a100` | **70.010** | **+2.327** | -1.361 | 0.0% |
| `ridge_source_rank_g2_a100` | 69.738 | +2.055 | -1.013 | 0.0% |
| `nested_risk20` | 69.430 | +1.747 | -1.371 | 0.0% |
| `nested_mean` | 69.346 | +1.663 | -1.447 | 0.0% |
| `lda_full` | 67.683 | 0.000 | 0.000 | 0.0% |

Per-repeat selection was too noisy:

```text
nested_risk20 selections over 288 outer subject/repeats:

ridge_longitudinal_rank_g2_a100  113
ridge_source_rank_g2_a100         58
ridge_source_rank_g1_a100         44
lda_long_q10                      31
lda_full                          19
lda_long_q70                      12
lda_source_q25                    11
```

Stable nested:

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `ridge_longitudinal_rank_g2_a100` | **70.010** | **+2.327** | -1.361 | 0.0% |
| `stable_nested_mean` / `stable_nested_risk20` | 69.825 | +2.142 | -1.361 | 0.0% |
| `ridge_source_rank_g2_a100` | 69.738 | +2.055 | -1.013 | 0.0% |
| `lda_full` | 67.683 | 0.000 | 0.000 | 0.0% |

Stable selection:

```text
ridge_longitudinal_rank_g2_a100  8/9 subjects
ridge_source_rank_g2_a100        1/9 subjects
```

Interpretation:

```text
The method family is identifiable,
but not if we let the selector react to every random source subset.
```

This is the key lesson of E33.

## What E33 changes

### Keep

```text
soft reliability-weighted Ridge is a real method candidate.
```

It improves over LDA full in:

- Lee2019 sensorimotor20: +2.38pp fixed, +2.04pp nested;
- BNCI m8/class: +2.33pp fixed, +2.14pp stable nested.

### Revise

Do not frame the method as:

```text
choose the best adapter/classifier per subject/session.
```

That repeats the old selector failure mode.

Frame it as:

```text
use source-side validation to choose a stable regime-level reliability weighting family.
```

The selected family may differ by regime:

```text
Lee2019 sensorimotor20: source reliability weighting
BNCI m8/class:          longitudinal reliability weighting
BNCI full-source:       LDA baseline
```

### Weaken

Do not claim:

```text
source-side validation perfectly recovers the fixed oracle best.
```

It does not.

It gets close on Lee and BNCI m8, but not perfectly. On BNCI full-source, it mostly chooses LDA full, while the outer fixed best is LDA long_q70 by only +0.45pp. With 9 subjects, that small advantage is not strong enough to trust.

## Current best research claim

The cleanest claim is:

```text
Zero-target-label cross-session EEG-MI benefits from stable source-validated
feature-reliability weighting in regularized linear classifiers.

The gain is not from post-hoc target tuning or hard subspace deletion.
It appears when the regime is source-scarce or cross-session fragile,
and source-side validation can select the appropriate weighted family
without target labels.
```

This is stronger than the old channel/subspace-selection story.

## Next action: E34

E34 should stop chasing a clever selector and consolidate the method.

### E34 goal

Define a final method:

```text
Stable Source-Validated Reliability-Weighted Ridge
```

Candidate set should be small:

```text
lda_full
ridge_source_rank_g2_a100
ridge_longitudinal_rank_g2_a100
```

Maybe include:

```text
lda_long_q70
```

only as a conservative LDA baseline, not as the method centerpiece.

### Why reduce the candidate set?

E33 showed that adding too many weak candidates lets the selector pick noise.
The useful candidates are already known:

- `ridge_source_rank_g2_a100`
- `ridge_longitudinal_rank_g2_a100`
- `lda_full`

The hard top-k variants and many LDA q variants mostly add variance.

### E34 decision criterion

Across Lee sensorimotor20 and BNCI m8/class:

- stable nested method should be within 0.3pp of the best fixed weighted Ridge;
- should beat LDA full by at least +1.5pp;
- should keep `P(gain vs LDA full < -5pp)` at 0-12%;
- should avoid weighted Ridge on BNCI full-source.

If E34 passes, this becomes the dissertation method core.

