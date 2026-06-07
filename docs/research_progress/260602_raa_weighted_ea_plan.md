# 2026-06-02 RAA weighted EA first implementation

## Why

Adaptive EA selector v2 reached 85.19% on BCIC2a seed0, but the novelty is still mostly selection/fusion. The next step is per-channel reliability weighted EA, where reliability protects the alignment covariance itself.

## Key implementation choice

A naive weighted covariance `R_w = D R D` is dangerous. If a channel has low weight, its diagonal variance becomes small, and `R_w^{-1/2}` can amplify that channel.

The implemented first version therefore preserves diagonal variance and attenuates only cross-channel covariance:

```text
D = diag(sqrt(w))
R_raw = mean(x x^T / T)
R_weighted = D R_raw D
diag(R_weighted) = diag(R_raw)
x_aligned = R_weighted^{-1/2} x
```

This treats unreliable channels as less trusted for cross-channel alignment structure without making whitening blow them up.

## Reliability features

Computed label-free from train/test sessions:

- relative high variance
- relative flatline
- relative kurtosis
- relative 50/60 Hz line-noise ratio
- relative abnormal channel correlation
- relative covariance leverage

The important correction was channel-relative scoring. A global session shift should not downweight every channel.

## Diagnostic outcome

With relative scoring:

- S2: mean full weight 0.877, no low channels. Good: shrink collapse case should not be aggressively weighted down.
- S5: mean full weight 0.741, low channels [8, 13, 15, 17, 18]. Good: this is the clearest bad-channel-like subject.
- S7: mean full weight 0.910, no low channels. Good but may be insufficient because S7 failure may not be simple channel artifact.
- S8: mean full weight 0.910, no low channels. Good: clean EA winner should be preserved.

## First GPU test

Run `RAA-full` on S2/S5/S7/S8:

```text
--ea --ea_weight_mode full
```

Success target:

- S2 stays near full EA 75.69
- S5 reaches at least source/shrink range 77%
- S7 improves over full EA 83.68, ideally >90
- S8 stays near 90

If S7 is not rescued, try selector/fusion with RAA as one candidate rather than expecting reliability weights alone to solve it.
