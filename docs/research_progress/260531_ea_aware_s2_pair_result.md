# 260531 EA-aware TCFormer S2 pair result

## Purpose

Accuracy-first plan の Phase 1 として、BCIC IV 2a subject 2 で
train/test-consistent session-wise Euclidean Alignment (EA) を TCFormer 再学習に入れた。

比較は同一コード・同一 seed・同一 subject の paired run:

- no-EA TCFormer
- EA-aware TCFormer

## Setup

- Dataset: BCIC IV 2a
- Subject: S2
- Seed: 0
- Model: TCFormer
- Epochs: 1000
- InterAug: enabled (default)
- GPU: 3
- EA: train/val は train session reference、test は test session reference

## Results

| condition | test acc | kappa | test loss | train time |
|---|---:|---:|---:|---:|
| no-EA TCFormer | 72.92 | 0.639 | 0.747 | 28.97 min |
| EA-aware TCFormer | 75.69 | 0.676 | 0.651 | 28.99 min |

EA-aware gain:

- Accuracy: **+2.78 pp**
- Kappa: **+0.037**
- Test loss: **-0.095**

## Artifact paths

- no-EA: `intentflow/offline/results/source_tcformer_s2_seed0_pair_20260531/`
- EA-aware: `intentflow/offline/results/ea_aware_tcformer_s2_seed0_20260531/`

## Interpretation

This is a first paired smoke/subject result, not a conclusion. Still, it is a
good sign: after test-only EA failed catastrophically, train/test-consistent
EA-aware retraining gives a positive S2 gain under the current pipeline.

Next step: run all 9 BCIC2a subjects for seed 0, then repeat multi-seed only if
the all-subject mean and worst-subject drop look acceptable.
