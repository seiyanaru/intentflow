# 2026-06-02 RAA stress first results

## Purpose

Check whether per-channel reliability weighted EA helps when test channels are
explicitly corrupted before EA.

The stress is applied after z-scale and before EA, only to the test session.
Evaluation here is checkpoint-only inference using the clean full-EA S5 model:

```text
checkpoint: intentflow/offline/results/ea_aware_tcformer_s5_seed0_20260601
subject: S5
corrupted channels: [8, 13]
```

## New tooling

- `intentflow/offline/train_pipeline.py`
  - added test artifact stress CLI flags
  - added CPU `num_workers=0` fallback for test-only stability
- `intentflow/offline/datamodules/base.py`
  - added reproducible test-only artifact injection
  - added optional reliability-weighted EA covariance
  - added optional post-EA channel gate
  - added optional pre-EA channel repair from train-channel correlation
- `intentflow/offline/scripts/analysis/eval_checkpoint_under_stress.py`
  - direct checkpoint evaluator, bypassing Lightning Trainer/DataLoader

## Results

Clean full-EA S5 reference was 72.57%.

| Stress | Method | Accuracy | Kappa | Interpretation |
|---|---:|---:|---:|---|
| highvar level 4 | standard EA | 68.40 | 0.579 | stress hurts but EA survives somewhat |
| highvar level 4 | RAA artifact, strength 1.0 | 65.97 | 0.546 | worse than standard |
| highvar level 4 | RAA artifact, strength 0.25 | 67.01 | 0.560 | closer, still worse |
| highvar level 4 | RAA + post gate | 59.03 | 0.454 | bad; suppressing these channels loses useful MI information |
| highvar level 4 | RAA + repair | 63.89 | 0.519 | bad; simple correlation repair is not enough |
| line level 4 | standard EA | 64.24 | 0.523 | line stress hurts more |
| line level 4 | RAA artifact, strength 0.25 | 62.85 | 0.505 | still worse |

## Readout

The reliability detector itself works: under high-variance stress, ch8/ch13 are
the only low channels in `raa_var` / `raa_artifact`.

The current alignment action is the weak point. Downweighting covariance,
gating, or simple correlation repair all reduce S5 accuracy when the corrupted
channels are still useful for MI decoding.

## Decision

Do not present the current RAA covariance weighting as a positive method yet.
It is a negative result and a useful diagnostic harness.

Next useful branch:

1. Run the stress harness across non-motor vs motor channel groups to separate
   "bad channel is redundant" from "bad channel carries class evidence".
2. If RAA only helps on redundant channels, it is not strong enough as a main
   paper idea.
3. To rescue the idea, move from covariance weighting to reliability-conditioned
   method selection:
   - clean/no channel outlier: use selector/fusion v2
   - global session shift: standard EA or shrink EA
   - localized bad channel: try imputation-aware TTA or channel-drop training
4. If pursuing RAA further, train with channel dropout / artifact augmentation so
   the decoder learns to survive reliability gating. Test-only gating is too
   distribution-shifting for the current TCFormer.

