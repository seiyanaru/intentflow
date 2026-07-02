# E20: Lee2019 channel/dimension ablation results

Date: 2026-06-29

## One-line verdict

E20 strongly supports the mechanism that the Lee2019 `source_only_q0p25` gain is
not just "motor-channel selection".

When Lee2019 is reduced from all 62 channels to the 20 available sensorimotor
channels:

- the `full_q1p00` baseline stays almost unchanged;
- the `source_only_q0p25` gain almost disappears;
- `longitudinal_q0p10` also disappears or becomes slightly negative;
- aggressive selection becomes unstable and lower-tail risk increases.

So the best current interpretation is:

> Lee2019 benefits from selecting a broader discriminative tangent subspace from
> all channels.  If we pre-restrict to sensorimotor channels, the full tangent
> representation is already compact enough, and additional subspace selection
> mostly becomes unnecessary or harmful.

This supports the high-dimensional/noisy-feature-regime story.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/lee2019_channel_dimension_ablation_e20.py`

Output:

`intentflow/offline/results/research_outputs/260629_lee2019_channel_dimension_ablation_e20/sensorimotor20/`

Main files:

- `summary.csv`
- `summary.json`
- `selection_records.csv`
- `subject_cache/`

## Protocol

- Dataset: Lee2019_MI
- Paradigm: LeftRightImagery
- Subjects: 1-54
- Source: session 0
- Target: session 1
- Target prefix: 20 unlabeled trials
- Eval trials: 80
- Classifier: shrinkage LDA
- Feature:
  - all62 reference: existing E9, p = 1953
  - sensorimotor20: Lee-available sensorimotor channels, p = 210

Lee2019 does not have `FCz`, so the sensorimotor set has 20 channels rather
than the Stieger-style 21 channels.

Sensorimotor20 channels:

```text
FC5, FC3, FC1, FC2, FC4, FC6,
C5, C3, C1, Cz, C2, C4, C6,
CP5, CP3, CP1, CPz, CP2, CP4, CP6
```

## Main sensorimotor20 result

| Method | Acc | Gain vs full | 95% CI | P(gain<-5) |
|---|---:|---:|---:|---:|
| `full_q1p00` | 69.606 | 0.000 | [0.000, 0.000] | 0.000 |
| `source_only_q0p10` | 70.995 | +1.389 | [-0.417, +3.218] | 0.204 |
| `source_only_q0p50` | 70.440 | +0.833 | [-0.833, +2.384] | 0.130 |
| `longitudinal_q0p25` | 70.347 | +0.741 | [-1.157, +2.546] | 0.167 |
| `longitudinal_q0p50` | 70.185 | +0.579 | [-0.995, +2.037] | 0.130 |
| `source_only_q0p25` | 69.699 | +0.093 | [-1.806, +1.991] | 0.259 |
| `source_only_q0p80` | 69.583 | -0.023 | [-0.903, +0.926] | 0.056 |
| `longitudinal_q0p80` | 69.491 | -0.116 | [-1.296, +1.019] | 0.111 |
| `longitudinal_q0p10` | 69.421 | -0.185 | [-2.222, +1.875] | 0.241 |

No method gives a robust positive gain in sensorimotor20.

## all62 vs sensorimotor20 comparison

| Feature | Method | p | Acc | Gain | P(gain<-5) |
|---|---|---:|---:|---:|---:|
| all62 | `full_q1p00` | 1953 | 69.676 | 0.000 | 0.000 |
| sensorimotor20 | `full_q1p00` | 210 | 69.606 | 0.000 | 0.000 |
| all62 | `source_only_q0p25` | 1953 | 71.852 | +2.176 | 0.074 |
| sensorimotor20 | `source_only_q0p25` | 210 | 69.699 | +0.093 | 0.259 |
| all62 | `source_only_q0p50` | 1953 | 71.435 | +1.759 | 0.056 |
| sensorimotor20 | `source_only_q0p50` | 210 | 70.440 | +0.833 | 0.130 |
| all62 | `longitudinal_q0p10` | 1953 | 71.528 | +1.852 | 0.204 |
| sensorimotor20 | `longitudinal_q0p10` | 210 | 69.421 | -0.185 | 0.241 |

Paired comparison:

- `full_q1p00`: sensorimotor20 - all62 = -0.069 pp
  - CI: [-2.292, +2.060]
- `source_only_q0p25` accuracy: sensorimotor20 - all62 = -2.153 pp
  - CI: [-4.259, -0.139]
- `longitudinal_q0p10` accuracy: sensorimotor20 - all62 = -2.106 pp
  - CI: [-4.236, -0.023]

This is the key result:

> Reducing to sensorimotor channels does not hurt the full baseline, but it
> removes the all-channel source-side selection advantage.

## Mechanistic interpretation

E8 already showed:

- Lee `source_only_q0p25` is less sensorimotor-concentrated than
  `longitudinal_q0p10`.
- Lee `source_only_q0p25` includes posterior/POz/P features.
- Yet it beats the more motor-concentrated `longitudinal_q0p10`.

E20 now confirms that simply restricting to sensorimotor channels does not
reproduce the Lee gain.

Therefore:

```text
Lee gain is not explained by "more motor-cortical features".
Lee gain is better explained by selecting an appropriately broad but denoised
all-channel tangent subspace.
```

This also explains why BNCI2014_001 did not show the same fixed-policy transfer:

- BNCI LR full feature space is already low-dimensional: p = 253, source n = 144
- Lee all62 is high-dimensional: p = 1953, source n = 100
- Lee sensorimotor20 becomes low-dimensional: p = 210, source n = 100
- In the low-dimensional regimes, full tangent is already competitive.

## What to claim

Good claim:

> Source-side tangent subspace selection is useful when the all-channel tangent
> feature space is high-dimensional and contains noisy/irrelevant covariance
> dimensions.  It is not a generic benefit of using fewer channels, and it does
> not reduce to motor-channel selection.

Bad claim:

> Sensorimotor restriction is the mechanism.

Bad claim:

> Fixed q0.25 policy transfers across all Lee-like LR datasets.

BNCI already refutes that.

## Next action

The next experiment should be small and decisive:

1. Map all62 `source_only_q0p25` selected dimensions to channel-pair categories.
2. Compare selected all62 dimensions with the sensorimotor20 feature space.
3. Quantify how much of the selected source-only advantage comes from:
   - sensorimotor-sensorimotor pairs
   - sensorimotor-posterior pairs
   - posterior/posterior or non-motor pairs

If posterior/mixed pairs are important, the novelty becomes:

> source-side selection discovers a useful cross-session covariance subspace
> that is broader than hand-defined motor channels but safer than all-channel
> full tangent features.

That is a much stronger mechanism story than "channel reduction".
