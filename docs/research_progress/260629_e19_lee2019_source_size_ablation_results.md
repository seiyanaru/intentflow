# E19: Lee2019 source-size ablation results

Date: 2026-06-29

## One-line verdict

E19 supports the revised regime hypothesis, but with an important split:

- `longitudinal_q0p10` behaves like a source-scarce regularizer.
- `source_only_q0p25` is broadly strong across source sizes and is not explained
  only by source scarcity.

So the updated mechanism should be:

> Source-side tangent subspace selection has two effects:
>
> 1. compact longitudinal cores help most when the held-subject source decoder is
>    highly source-scarce;
> 2. source-only ranking removes noisy/irrelevant tangent dimensions in Lee2019
>    even when all 100 source trials are available.

This is stronger and more honest than the previous single-factor p/n story.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/lee2019_source_size_ablation_e19.py`

Chunk runner:

`intentflow/offline/scripts/analysis/run_lee2019_e19_chunks.sh`

Output:

`intentflow/offline/results/research_outputs/260629_lee2019_source_size_ablation_e19_primary_chunks/merged/`

Main files:

- `source_size_records.csv`
- `summary.csv`
- `summary.json`

## Protocol

- Dataset/cache: Lee2019_MI LR cache from E9
- Subjects: 1-54
- Source session: session 0
- Target session: session 1
- Target prefix: 20 unlabeled trials, already encoded in cached features
- Eval trials: 80
- Tangent dimension: p = 1953
- Source-side ranking: all non-held subjects
- Held-subject source training trials were balanced subsamples
- Source per class: 8, 16, 24, 36, 50
- Source total: 16, 32, 48, 72, 100
- Repeats: 4 for subsampled settings; 1 for full source = 100
- Methods:
  - `full_q1p00`
  - `source_only_q0p25`
  - `longitudinal_q0p10`

## Main result

| Source total | p/n | Full acc | source_only q0.25 gain | longitudinal q0.10 gain |
|---:|---:|---:|---:|---:|
| 16 | 122.06 | 60.203 | +2.593 | +3.791 |
| 32 | 61.03 | 64.277 | +2.922 | +2.801 |
| 48 | 40.69 | 65.729 | +2.546 | +3.420 |
| 72 | 27.13 | 67.870 | +2.917 | +2.234 |
| 100 | 19.53 | 69.676 | +2.176 | +1.852 |

All compact policies beat `full_q1p00` on mean accuracy at every source size.

At full source, the E19 result reproduces E9:

- E9 `source_only_q0p25`: +2.176 pp
- E19 source total 100 `source_only_q0p25`: +2.176 pp
- E9 `longitudinal_q0p10`: +1.852 pp
- E19 source total 100 `longitudinal_q0p10`: +1.852 pp

This validates the E19 implementation.

## Trend test

### `longitudinal_q0p10`

`longitudinal_q0p10` shows the clearest source-scarce trend.

- Spearman between p/n and gain: 0.90, p = 0.037
- Source total 16 vs 100 paired delta: +1.939 pp
  - bootstrap 95% CI: [+0.122, +3.796]
- Source total 48 vs 100 paired delta: +1.568 pp
  - bootstrap 95% CI: [+0.203, +3.015]

Interpretation:

`longitudinal_q0p10` is not simply a universally better Lee policy.  It becomes
more useful when the source decoder is starved of labels.

### `source_only_q0p25`

`source_only_q0p25` is strong but not clearly source-size dependent.

- Spearman between p/n and gain: 0.50, p = 0.391
- Source total 16 vs 100 paired delta: +0.417 pp
  - bootstrap 95% CI: [-1.024, +1.863]
- Source total 32 vs 100 paired delta: +0.747 pp
  - bootstrap 95% CI: [-0.561, +2.141]

Interpretation:

The Lee `source_only_q0p25` effect is not just "small n regularization".
It probably reflects a stable source-side ranking that removes noisy tangent
dimensions in Lee2019.

## Risk / lower-tail behavior

| Source total | source_only q0.25 P(gain<-5) | longitudinal q0.10 P(gain<-5) |
|---:|---:|---:|
| 16 | 0.000 | 0.093 |
| 32 | 0.037 | 0.056 |
| 48 | 0.019 | 0.111 |
| 72 | 0.056 | 0.148 |
| 100 | 0.074 | 0.204 |

`longitudinal_q0p10` has higher upside in the most source-scarce setting, but
also substantially worse lower-tail risk.  `source_only_q0p25` is the more
deployable Lee policy.

## Updated interpretation after BNCI + Lee

The previous one-factor story was:

> compact selection helps when p/n is high.

E19 suggests a sharper two-factor story:

> In high-dimensional tangent regimes, source-side subspace selection is useful.
> Compact longitudinal cores act as source-scarce regularizers, while source-only
> ranking can be broadly useful when the dataset contains many noisy tangent
> dimensions.

This explains:

- Lee2019:
  - p/n is high even at full source: 1953 / 100 = 19.53
  - `source_only_q0p25` remains strong at all source sizes
  - `longitudinal_q0p10` strengthens when source is scarce
- BNCI2014_001 LR:
  - p/n is low at full source: 253 / 144 = 1.76
  - aggressive compact policies do not transfer
  - source-scarce simulation makes compact selection mildly useful again

## Scientific consequence

Do not frame the method as:

> fixed task-family policies transfer across datasets.

That is false after BNCI.

Frame it as:

> source-side tangent subspace selection is useful in high-dimensional /
> source-scarce cross-session EEG-MI, but the safe compactness/family depends on
> the regime.

## Next action

The next high-value experiment is a dimension/channel ablation, not another
selector:

1. Lee2019 all channels:
   - p = 1953
2. Lee2019 sensorimotor subset:
   - around 21 channels, p = 231
3. Optional smaller motor subset:
   - C3/Cz/C4 neighborhood

Prediction:

- If the p/n story is correct, reducing Lee to sensorimotor channels should
  shrink the benefit of compact selection.
- If `source_only_q0p25` remains strong even at low p, then Lee has a deeper
  dataset-specific source-ranking effect.

This is the cleanest next mechanism test.
