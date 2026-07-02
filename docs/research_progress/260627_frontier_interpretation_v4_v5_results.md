# 260627 V4/V5: risk-utility frontier and selected-feature interpretation

## One-line verdict

`source-side longitudinal subspace selection` は、平均精度改善としてはかなり強い。
ただし「安全性まで解いた」とはまだ言えない。
現時点で守れる主張は、

```text
zero-target-label cross-session EEG-MI において、
source側の縦断ラベル情報からセッション安定なRiemann tangent subspaceを選ぶと、
full feature / random subspace / source-only discriminability より一貫して高い平均精度を出す。
一方で、下側リスクは完全には消えないため、risk-utility frontier として報告する。
```

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/stieger_longitudinal_frontier_and_interpretation.py
```

Output:

```text
intentflow/offline/results/research_outputs/260627_stieger_longitudinal_frontier_interpretation/
```

Main files:

```text
risk_utility_frontier.csv
risk_utility_frontier_primary.png
selected_feature_rows.csv
selected_channel_participation.csv
top_channels_pure_lr_broad_all60.png
top_channels_pure_lr_fb_sensorimotor21_mu_beta.png
top_channels_pure_ud_broad_all60.png
top_channels_pure_ud_fb_sensorimotor21_mu_beta.png
summary.json
```

## V4: risk-utility frontier

Comparator is fixed to E4a's LOSO outer-best single branch.
This avoids changing the baseline depending on the method being evaluated.

Primary pooled LR+UD:

| method | acc | gain vs E4a outer-best single | R10 loss | P(gain < -5pp) | q05 gain |
|---|---:|---:|---:|---:|---:|
| E5b longitudinal outer best | 67.734 | +3.919 | 10.589 | 12.55% | -8.824 |
| E5b longitudinal q0.10 | 67.681 | +3.866 | 11.580 | 13.50% | -9.524 |
| E5b longitudinal q0.25 | 67.082 | +3.267 | 10.668 | 14.06% | -9.091 |
| source-only outer best | 67.002 | +3.187 | 11.507 | 14.31% | -10.390 |
| V2 sep-no-drift outer best | 66.805 | +2.990 | 12.471 | 17.29% | -11.628 |
| V2 source-only outer best | 66.777 | +2.962 | 12.277 | 16.10% | -11.111 |
| V2 target-only outer best | 65.988 | +2.172 | 12.954 | 15.37% | -11.842 |
| E4a equal posterior + prefix-EA | 65.325 | +1.509 | 11.241 | 16.15% | -9.877 |

Pareto judgement:

```text
Primary pooled:
  longitudinal outer best is the only meaningful non-dominated point
  when utility = mean gain and risk = R10 loss.

LR:
  longitudinal q0.10 is cleanly best:
    acc 70.431
    gain +4.079
    R10 loss 8.562
    P(gain < -5pp) 11.2%

UD:
  longitudinal q0.25 gives best gain:
    acc 64.848
    gain +3.855
    R10 loss 12.359
  but E4a equal posterior and source-only fraction keep lower R10 at lower gain.
  UD is therefore still a genuine risk-utility trade-off, not solved.
```

## V5: selected-feature interpretation

Interpretation was done on source-side aggregate statistics only.
This is explanatory, not a deployment component.

### LR, broad 8-30 Hz all 60 channels

longitudinal q0.10 selected 183 tangent dimensions.

Top channel participation:

```text
C4 31
CP4 31
CP2 25
CP3 25
C2 23
C1 22
CP1 20
C3 16
FC2 10
FCz 9
```

Interpretation:

```text
LR selection concentrates on bilateral central and centro-parietal electrodes.
This is consistent with motor imagery physiology.
Most selected dimensions are off-diagonal covariance terms, so the method is
not merely choosing single-channel power. It is selecting stable spatial
covariance interactions around sensorimotor cortex.
```

### LR, mu/beta sensorimotor filterbank

longitudinal q0.10 selected 70 tangent dimensions.

Band counts:

```text
mu_8_13: 14
low_beta_13_20: 17
high_beta_20_30: 39
```

Top channels:

```text
CP4 20
CP3 18
C4 18
C2 17
C3 13
FCz 10
C1 9
FC2 9
```

Interpretation:

```text
The LR result is encouraging:
the selected subspace lives mainly in C/CP motor regions and is beta-heavy.
This gives a plausible neurophysiological story:
session-stable discriminative structure is not arbitrary covariance noise,
but motor-network covariance in mu/beta rhythms.
```

### UD caveat

UD broad features include CP/C channels but also strong posterior/frontal participation:

```text
CP4, CP3, POz, C4, C3, Fp1, Fpz, Oz ...
```

This is weaker neurophysiological evidence.
For UD, the current method improves accuracy, but the selected subspace is less clean.
Do not overclaim UD motor physiology from this result.

## What this changes in the paper story

### Stronger than before

We no longer have only a negative safety story.
We now have a positive method:

```text
Source-side longitudinal subspace selection for zero-target-label cross-session EEG-MI.
```

The nontrivial part is not target-label gating.
It is using source subjects' longitudinal label structure to choose a stable tangent subspace
before adapting/evaluating the unseen target session.

### Still not safe to claim

Do not claim:

```text
the method prevents harm
the method solves safe adaptation
the selected subspace is universally neurophysiological
```

The lower tail remains non-negligible:

```text
primary P(gain < -5pp) = 12.55%
primary q05 gain = -8.824 pp
```

This is better than many alternatives, but not safe.

## Next required experiment

The next decisive experiment is V3:

```text
Run the same source-side longitudinal subspace selection on a second multi-session MI dataset.
```

Minimum acceptable V3:

```text
dataset: Lee2019_MI if cached data is usable
split: subject-level holdout
source information: training subjects' session-to-session labeled structure only
target information: zero target labels
comparators:
  full tangent features
  random top-k
  source-only discriminability top-k
  longitudinal stability top-k
primary endpoint:
  longitudinal top-k > source-only top-k and full features
```

If V3 reproduces the direction, the project becomes a real paper candidate.
If V3 fails, the honest thesis story becomes:

```text
Stieger-specific discovery plus a rigorous negative/fragility analysis.
```

