# 260627 E7: Stieger source-session depth ablation

## One-line verdict

E7は、longitudinalを「source sessionsが多いときだけ効く」と救う仮説を支持しない。

```text
K=2でもlongitudinalはStiegerで既に効く。
K=2 -> 3 -> 5 -> all でlongitudinal-source差が単調に増えるわけではない。
```

したがって、Lee2019でlongitudinalがsource-onlyに負けた理由を

```text
Lee2019は2 sessionsしかないからdrift推定が弱い
```

だけで説明するのは無理。

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/stieger_session_depth_ablation.py
```

Output:

```text
intentflow/offline/results/research_outputs/260627_stieger_session_depth_ablation/
```

Main files:

```text
summary.json
summary.csv
comparisons.csv
selection_records.json
selection_records.csv
depth_stats/
subjects/
```

## Design

Held-out evaluation protocol is the same as E5b/V1/V2:

```text
held-out target subject:
  train decoder on first session labels
  evaluate later sessions
  target prefix used only for EA reference
  target later labels used only for final evaluation
```

The intervention is only in source-side metric learning:

```text
K=2:
  source session + first later session

K=3:
  source session + first 2 later sessions

K=5:
  source session + first 4 later sessions

K=all:
  all available later sessions
```

Compared methods:

```text
longitudinal q0.10 / q0.25
source-only q0.10 / q0.25
full q1.00 baseline
```

## Primary pooled LR+UD result

| K | method | acc | gain vs full | 95% CI | R10 loss | P(gain<-5) |
|---|---|---:|---:|---:|---:|---:|
| 2 | longitudinal q0.10 | 67.534 | +2.209 | [+1.170, +3.268] | 13.008 | 18.9% |
| 2 | longitudinal q0.25 | 66.971 | +1.646 | [+0.775, +2.611] | 12.656 | 17.4% |
| 2 | source-only q0.25 | 66.710 | +1.385 | [+0.676, +2.125] | 10.990 | 16.6% |
| 3 | longitudinal q0.10 | 67.303 | +1.978 | [+0.973, +3.048] | 13.969 | 17.9% |
| 3 | longitudinal q0.25 | 67.089 | +1.765 | [+0.896, +2.684] | 11.714 | 17.1% |
| 3 | source-only q0.25 | 66.724 | +1.399 | [+0.683, +2.143] | 10.751 | 15.7% |
| 5 | longitudinal q0.10 | 67.630 | +2.305 | [+1.388, +3.263] | 13.146 | 16.7% |
| 5 | longitudinal q0.25 | 67.136 | +1.812 | [+0.944, +2.741] | 11.586 | 16.1% |
| 5 | source-only q0.25 | 66.636 | +1.311 | [+0.579, +2.079] | 10.862 | 16.8% |
| all | longitudinal q0.10 | 67.681 | +2.357 | [+1.449, +3.326] | 12.152 | 17.1% |
| all | longitudinal q0.25 | 67.082 | +1.758 | [+0.946, +2.635] | 11.529 | 16.7% |
| all | source-only q0.25 | 66.638 | +1.313 | [+0.633, +2.039] | 10.991 | 16.8% |

## Direct longitudinal - source-only differences

### Primary pooled, same q

| K | long q0.10 - source q0.10 | 95% CI | P(diff<0) | long q0.25 - source q0.25 | 95% CI | P(diff<0) |
|---|---:|---:|---:|---:|---:|---:|
| 2 | +0.968 | [+0.364, +1.599] | 37.1% | +0.261 | [-0.365, +0.864] | 40.3% |
| 3 | +0.600 | [-0.034, +1.223] | 45.2% | +0.366 | [-0.148, +0.905] | 46.8% |
| 5 | +0.989 | [+0.448, +1.555] | 33.9% | +0.501 | [-0.010, +1.015] | 38.7% |
| all | +0.918 | [+0.394, +1.481] | 37.1% | +0.444 | [-0.032, +0.928] | 35.5% |

### Primary pooled, strongest practical comparison

`longitudinal q0.10 - source-only q0.25`:

| K | mean | 95% CI | q05 | P(diff<0) |
|---|---:|---:|---:|---:|
| 2 | +0.824 | [+0.120, +1.568] | -2.848 | 38.7% |
| 3 | +0.579 | [-0.118, +1.283] | -3.345 | 37.1% |
| 5 | +0.994 | [+0.380, +1.588] | -2.306 | 33.9% |
| all | +1.044 | [+0.475, +1.615] | -2.201 | 30.6% |

Interpretation:

```text
There is a small longitudinal advantage over source-only,
but it does not emerge only at large K.

K=2 already shows a positive effect.
K=3 is weaker.
K=5/all are slightly stronger.
This is not a clean monotonic session-depth mechanism.
```

## LR/UD split

### LR

LR has a stable longitudinal q0.10 advantage.

`longitudinal q0.10 - source-only q0.25`:

```text
K=2:   +1.249  CI [+0.554, +1.999]
K=3:   +1.258  CI [+0.592, +1.964]
K=5:   +1.511  CI [+0.802, +2.216]
K=all: +1.495  CI [+0.770, +2.206]
```

This is the cleanest positive result.

Interpretation:

```text
For LR, longitudinal helps beyond source-only,
and the effect is present even with K=2.
More sessions slightly strengthen it, but are not necessary.
```

### UD

UD is weaker and depends on q.

`longitudinal q0.25 - source-only q0.25`:

```text
K=2:   +0.225  CI [-0.795, +1.175]
K=3:   +0.483  CI [-0.487, +1.444]
K=5:   +0.744  CI [-0.178, +1.614]
K=all: +0.680  CI [+0.027, +1.339]
```

Interpretation:

```text
For UD, more sessions help somewhat,
but the evidence is modest and not a clean K-dose-response.
```

## What E7 proves and does not prove

### Supported

```text
Stieger longitudinal feature scoring is genuinely better than source-only,
especially for LR.
```

### Not supported

```text
The reason Lee2019 does not benefit from longitudinal is simply that Lee has only 2 sessions.
```

Why not:

```text
Stieger K=2 already gives a positive longitudinal advantage.
```

### Revised mechanism

The likely dataset difference is not just number of sessions.
More plausible factors:

```text
1. label/task structure:
   Stieger has multiple MI task contexts; Lee2019 is only LR.

2. session/data quality:
   Lee2019 source-only discriminability may already capture the stable MI subspace.

3. covariance dimensionality vs trial count:
   Lee2019 has 62 channels but only 100 trials/session.
   drift estimates may be noisier even with same K=2.

4. selected-feature geometry:
   Stieger longitudinal may suppress unstable off-diagonal covariance terms
   differently from Lee2019.
```

## Decision after E7

Do not use this title:

```text
Source-side longitudinal subspace selection
```

It overclaims.

Use:

```text
Source-Side Tangent Subspace Selection
for Zero-Target-Label Cross-Session EEG Motor Imagery
```

Then present longitudinal as:

```text
a Stieger-effective, LR-strong stability variant,
not a universal cross-dataset default.
```

## Next experiment

The next useful experiment is E8, not more nested selection.

E8:

```text
Compare selected feature geometry between:
  Stieger LR longitudinal q0.10
  Stieger LR source-only q0.25
  Lee2019 source-only q0.25
  Lee2019 longitudinal q0.10
```

Required outputs:

```text
selected feature overlap
diagonal vs off-diagonal ratio
top channels
sensorimotor concentration
if possible, band counts for Stieger filterbank branch
```

Decision rule:

```text
If Stieger longitudinal selects a clearly more sensorimotor/stable covariance subspace
than source-only, while Lee source-only already selects it:
  mechanism story becomes credible.

If feature geometry is not interpretable:
  keep the paper as an empirical source-side subspace selection paper,
  and stop making a mechanistic longitudinal claim.
```

