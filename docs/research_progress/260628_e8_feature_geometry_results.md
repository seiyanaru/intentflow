# 260628 E8: selected-feature geometry analysis

## One-line verdict

E8は、単純な機構仮説を半分壊した。

```text
longitudinal q0.10 は、source-only q0.25 とまったく別の特徴を選んでいるわけではない。
むしろ多くの場合、source-only q0.25 の中の高濃度subsetである。
```

したがって、現時点で一番正確な解釈は:

```text
source-only q0.25:
  broader discriminative tangent subspace

longitudinal q0.10:
  more compact, more sensorimotor-concentrated core subspace
```

Stieger LRではcompact coreが効く。
Lee2019ではbroader source-only subspaceの方がよい。

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/source_side_feature_geometry_e8.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_source_side_feature_geometry_e8/
```

Main files:

```text
feature_geometry_summary.csv
feature_overlap.csv
selected_feature_geometry_rows.csv
top_channel_rows.csv
lee2019_top_channels_source_only_q0.25.png
lee2019_top_channels_longitudinal_q0.10.png
summary.json
```

## Key result 1: Stieger LR longitudinal is more sensorimotor-concentrated

### Stieger LR broad 8-30 Hz

| method | n | diagonal | any sensorimotor pair | both sensorimotor pair | sensorimotor participation |
|---|---:|---:|---:|---:|---:|
| longitudinal q0.10 | 183 | 4.4% | 95.1% | 41.5% | 68.3% |
| source-only q0.25 | 458 | 4.1% | 73.4% | 26.6% | 50.0% |

Top channels:

```text
longitudinal q0.10:
  C4, CP4, CP2, CP3, C2, C1, CP1, C3

source-only q0.25:
  C1, C4, C2, CP4, CP2, CP3, C3, CP1
```

Interpretation:

```text
Both are motor-area centered.
But longitudinal q0.10 is much more concentrated around sensorimotor covariance pairs.
This supports the idea that longitudinal scoring acts as a compact motor-covariance core selector.
```

### Stieger LR sensorimotor mu/beta filterbank

Both methods are restricted to sensorimotor channels by construction.

| method | n | diagonal | high-beta count |
|---|---:|---:|---:|
| longitudinal q0.10 | 70 | 21.4% | 39 |
| source-only q0.25 | 174 | 16.1% | 80 |

Top channels:

```text
longitudinal q0.10:
  CP4, CP3, C4, C2, C3, FCz, C1, FC2

source-only q0.25:
  CP4, C4, CP3, C2, C1, C3, CP1, FC2
```

Interpretation:

```text
Again, the difference is not location category but compactness/ranking.
longitudinal q0.10 keeps a smaller core within the same sensorimotor family.
```

## Key result 2: Lee2019 source-only wins despite being less sensorimotor-concentrated

### Lee2019 broad 8-30 Hz

| method | n | diagonal | any sensorimotor pair | both sensorimotor pair | sensorimotor participation |
|---|---:|---:|---:|---:|---:|
| longitudinal q0.10 | 196 | 13.8% | 74.0% | 38.8% | 56.4% |
| source-only q0.25 | 489 | 10.4% | 61.3% | 24.1% | 42.7% |

Top channels:

```text
longitudinal q0.10:
  C3, C1, C4, CP4, CP2, CP1, C2, FC1

source-only q0.25:
  C3, C4, C1, POz, FC1, CP3, CP4, CP2
```

Interpretation:

```text
Lee longitudinal q0.10 is actually more sensorimotor-concentrated than source-only q0.25.
Yet Lee performance favors source-only q0.25.

Therefore, "more sensorimotor concentration" alone does not explain performance.
Lee seems to need a broader discriminative subspace, including posterior/POz/P features.
```

This is important:

```text
The neuro story cannot be:
  longitudinal wins because it is more motor-cortical.

Because in Lee, longitudinal is also more motor-cortical but does not win.
```

## Key result 3: longitudinal q0.10 is mostly contained inside source-only q0.25

### Overlap: longitudinal q0.10 vs source-only q0.25

| dataset/condition/feature | n longitudinal | n source-only | intersection | longitudinal covered by source-only | Jaccard |
|---|---:|---:|---:|---:|---:|
| Lee2019 LR broad | 196 | 489 | 190 | 96.9% | 38.4% |
| Stieger LR broad | 183 | 458 | 168 | 91.8% | 35.5% |
| Stieger LR mu/beta | 70 | 174 | 69 | 98.6% | 39.4% |
| Stieger UD broad | 183 | 458 | 153 | 83.6% | 31.4% |
| Stieger UD mu/beta | 70 | 174 | 67 | 95.7% | 37.9% |

Interpretation:

```text
longitudinal q0.10 is not discovering a separate feature family.
It is usually selecting a compact subset of the broader source-only q0.25 feature set.
```

That reframes the mechanism:

```text
source-only:
  finds broad discriminative dimensions.

longitudinal:
  re-ranks those dimensions toward a smaller stable core.
```

## What this means for the paper

### Keep

```text
Source-side tangent subspace selection is the main contribution.
```

### Modify

```text
Longitudinal scoring is not a separate neurophysiological discovery.
It is a compactness/stability bias over an already discriminative source-side feature set.
```

### Do not claim

```text
Longitudinal wins because it is more brain-like.
```

That is too simplistic and contradicted by Lee2019.

## Updated mechanism hypothesis

The best current hypothesis is:

```text
There are two useful regimes:

1. compact-core regime:
   A small, high-confidence motor covariance core is best.
   This appears in Stieger LR.
   longitudinal q0.10 works well.

2. broader-subspace regime:
   A wider discriminative source-side subspace is needed.
   This appears in Lee2019.
   source-only q0.25 works better.
```

This is more defensible than the previous session-depth story.

## Next experiment decision

The next experiment should not be another nested selector.

The next useful test is:

```text
E9: fraction sweep / compactness frontier
```

Reason:

```text
E8 suggests the key axis may be compactness:
  q0.10 compact core vs q0.25 broader source subspace.
```

Design:

```text
For Stieger and Lee2019:
  evaluate source-only and longitudinal across q:
    0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50

Primary question:
  Does Stieger prefer smaller q while Lee prefers larger q?
```

Pass criterion:

```text
If optimal q differs systematically:
  Stieger LR optimum near q0.10
  Lee2019 optimum near q0.25 or larger

then the compact-core vs broader-subspace story becomes credible.
```

If not:

```text
Treat longitudinal/source-only difference as dataset-specific empirical behavior,
not a mechanistic result.
```

