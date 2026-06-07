# 260601 EA-aware TCFormer 9-subject seed0 result

## Purpose

Accuracy-first plan Phase 1 として、BCIC IV 2a 全9被験者で
train/test-consistent session-wise Euclidean Alignment (EA) を入れた
TCFormer再学習を seed0 で評価した。

## Setup

- Dataset: BCIC IV 2a
- Subjects: S1-S9
- Seed: 0
- Model: TCFormer
- Epochs: 1000
- InterAug: enabled
- EA: enabled
- EA placement: z-scale 後
- EA reference:
  - train/val: train split reference
  - test: test session reference

Source baseline は
`intentflow/offline/results/c_aug_true_9subj_20260506_004923/summary.md`
の `source_only` 行を使用。

## Results

| subject | source | EA-aware | delta |
|---:|---:|---:|---:|
| S1 | 85.76 | 87.85 | +2.09 |
| S2 | 72.57 | 75.69 | +3.12 |
| S3 | 93.06 | 92.36 | -0.70 |
| S4 | 81.94 | 83.33 | +1.39 |
| S5 | 77.43 | 72.57 | -4.86 |
| S6 | 70.49 | 73.61 | +3.12 |
| S7 | 90.97 | 83.68 | -7.29 |
| S8 | 84.38 | 90.28 | +5.90 |
| S9 | 87.85 | 88.19 | +0.34 |

## Summary

| metric | value |
|---|---:|
| source mean | 82.72 |
| EA-aware mean | 83.06 |
| mean delta | +0.35 pp |
| worst delta | -7.29 pp |
| harmed subjects, delta < 0 | 3/9 |
| harmed subjects, delta < -1pp | 2/9 |
| harmed subjects, delta < -2pp | 2/9 |

## Interpretation

EA-aware retraining is not a clean Phase 1 win in this configuration.
It improves S1/S2/S4/S6/S8/S9, but harms S5 and S7 substantially. The mean
gain is only +0.35pp, and worst-subject drop is too large for this to be the
base alignment recipe.

This result supports continuing the alignment direction, but not freezing this
EA implementation as the baseline. The next step should be an EA placement and
reference ablation before moving to RAA or selective TTA.

## Next Ablations

1. EA before z-scale vs EA after z-scale.
2. Train/val/test each own session reference vs val aligned with train reference.
3. z-scale off + EA.
4. shrinkage/eps sweep for EA covariance.
5. Per-subject failure analysis for S5/S7, especially covariance spectrum and
   confusion-pair changes.
