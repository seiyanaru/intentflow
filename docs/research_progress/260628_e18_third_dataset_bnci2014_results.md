# E18/E18b: Third-dataset check on BNCI2014_001

Date: 2026-06-28

## Verdict

BNCI2014_001 does **not** support the simple claim that the Lee-like two-session
LR setting should prefer `source_only_q0p25`.

The stronger interpretation is now:

> Source-side tangent subspace selection is useful mainly in high-dimensional /
> source-scarce regimes.  When the full tangent representation is already
> statistically manageable, compact selection gives little or negative benefit.

This revises the previous "task-family" story.  Task family still matters, but
it is not sufficient.  The dimension-to-source-trial regime is probably a core
descriptor.

## Dataset/protocol

- Dataset: MOABB `BNCI2014_001` / BCI Competition IV 2a
- Paradigm: `LeftRightImagery`
- Subjects: 1-9
- Source: `0train`
- Target: `1test`
- Target prefix: first 20 trials, unlabeled EA/tangent reference only
- Evaluation: target trials after the prefix
- Features: 8-30 Hz covariance -> Riemann tangent
- Classifier: shrinkage LDA
- Held-out target subject labels are not used for subspace selection or fitting.

Per subject:

- source trials: 144
- target eval trials: 124
- channels: 22
- tangent dimension: 253

For comparison, Lee2019 cache used in E9 has:

- source trials: 100
- target eval trials: 80
- tangent dimension: 1953

This difference is central.

## E18: fraction sweep on full source session

Output:

`intentflow/offline/results/research_outputs/260628_bnci2014_001_fraction_sweep_e18_dense/`

Main result:

| Method | Mean acc | Gain vs full | 95% subject bootstrap CI | Harm P(gain < -5pp) |
|---|---:|---:|---:|---:|
| `full_q1p00` | 77.60 | 0.00 | [0.00, 0.00] | 0.000 |
| `longitudinal_q0p70` | 78.05 | +0.45 | [-0.63, +1.79] | 0.000 |
| `source_only_q0p80` | 77.96 | +0.36 | [-1.16, +2.15] | 0.000 |
| `source_only_q0p50` | 76.97 | -0.63 | [-2.42, +1.16] | 0.000 |
| `source_only_q0p25` | 75.81 | -1.79 | [-3.85, +0.36] | 0.111 |
| `longitudinal_q0p10` | 75.27 | -2.33 | [-4.30, -0.45] | 0.222 |

Interpretation:

- The Lee fixed policy `source_only_q0p25` does not transfer.
- The Stieger compact policy `longitudinal_q0p10` is actively bad here.
- Very weak pruning (`q0.70` to `q0.80`) can be slightly positive, but the
  effect is tiny and not statistically stable.
- Therefore, the clean third-dataset claim is not "our selected subspace wins";
  it is "the correct amount of compactness is regime-dependent."

## E18b: source-size ablation inside BNCI2014_001

Motivation:

BNCI has only 253 tangent dimensions and 144 source trials.  Full tangent is not
as underdetermined as Lee2019.  If selection is a regularizer, its benefit
should grow when labeled source trials are artificially reduced.

Output:

- light sweep:
  `intentflow/offline/results/research_outputs/260628_bnci2014_001_source_size_ablation_e18b_light/`
- m=8 per class confirmation:
  `intentflow/offline/results/research_outputs/260628_bnci2014_001_source_size_ablation_e18b_m8_r32/`

Light sweep, gain vs full:

| Source per class | Source total | Best method | Gain vs full | 95% subject bootstrap CI |
|---:|---:|---|---:|---:|
| 8 | 16 | `longitudinal_q0p10` | +2.11 | [+0.07, +4.10] |
| 16 | 32 | `longitudinal_q0p70` | +0.24 | [-0.34, +0.82] |
| 36 | 72 | `source_only_q0p80` | +0.09 | [-0.63, +1.04] |
| 72 | 144 | `longitudinal_q0p70` | +0.45 | [-0.63, +1.79] |

m=8 per class, 32-repeat confirmation:

| Method | Mean acc | Gain vs full | 95% subject bootstrap CI |
|---|---:|---:|---:|
| `full_q1p00` | 68.21 | 0.00 | [0.00, 0.00] |
| `longitudinal_q0p10` | 69.49 | +1.28 | [-0.31, +2.94] |
| `longitudinal_q0p35` | 68.98 | +0.77 | [-0.14, +1.72] |
| `longitudinal_q0p70` | 68.68 | +0.47 | [+0.05, +0.82] |
| `source_only_q0p25` | 68.64 | +0.43 | [-0.16, +0.96] |
| `source_only_q0p80` | 68.46 | +0.25 | [-0.04, +0.59] |

Interpretation:

- The mechanism hypothesis is partly supported:
  selection helps most when source is extremely scarce.
- The strongest compact `q0.10` result weakens after 32 repeats, but remains
  positive on average.
- The more conservative `longitudinal_q0p70` is small but stable in the
  m=8 confirmation.
- This is not a big accuracy result.  It is a regime explanation.

## What changed in the research story

Previous candidate story:

> Stieger-like multi-session tasks prefer longitudinal compact subspaces;
> Lee-like two-session LR tasks prefer source-only compact subspaces.

Revised story:

> The useful compactness level depends on the statistical regime of the tangent
> classifier.  In high-dimensional/source-scarce settings, source-side subspace
> selection can regularize cross-session EEG-MI.  In low-dimensional/source-rich
> settings, the full tangent baseline is already strong and aggressive compact
> selection removes useful information.

This is a better story because it explains both positive and negative evidence.

## Regime meta table

Output:

`intentflow/offline/results/research_outputs/260628_e18_regime_meta_table.csv`

| Dataset / condition | Feature regime | Source n | p/n descriptor | Best method | Gain |
|---|---|---:|---:|---|---:|
| Stieger pooled | posterior ensemble: broad p=1830 + fb sensorimotor p=693 | median 135 | not single-p comparable | `longitudinal_q0p10` | +2.357 |
| Stieger LR | posterior ensemble: broad p=1830 + fb sensorimotor p=693 | median 138 | not single-p comparable | `longitudinal_q0p05` | +4.054 |
| Stieger UD | posterior ensemble: broad p=1830 + fb sensorimotor p=693 | median 132 | not single-p comparable | `longitudinal_q0p20` | +1.287 |
| Lee2019 LR | single tangent, 62ch | 100 | 19.53 | `source_only_q0p25` | +2.176 |
| BNCI2014_001 LR | single tangent, 22ch | 144 | 1.76 | `longitudinal_q0p70` | +0.448 |
| BNCI2014_001 LR m8/class | single tangent, 22ch, source-scarce simulation | 16 | 15.81 | `longitudinal_q0p10` | +1.277 |

Interpretation:

- Lee and BNCI are directly comparable single-tangent settings.
- In Lee, p/n is high and compact source-side selection helps.
- In BNCI full-source, p/n is low and aggressive compact selection hurts.
- In BNCI source-scarce simulation, p/n becomes high and compact selection
  becomes helpful again.
- Stieger is not a single-tangent p/n setting because it uses a two-branch
  posterior ensemble, but it is also a source-scarce/high-dimensional regime.

This makes the revised regime story much more credible than a pure
task-family story.

## Immediate next action

Do not add another selector.

The next experiment should test the regime story more directly:

1. On Lee2019, run the same source-size ablation.
   - Prediction: as source n increases, the advantage of `source_only_q0p25`
     or compact longitudinal should shrink.
2. On BNCI2014_001, run a slightly stronger confirmation only for the most
   informative source sizes.
   - Suggested: per-class `8, 16, 72`, repeats `32`.
   - Avoid spending compute on many intermediate points.
3. Reframe the method as:
   "source-side tangent subspace selection with regime-aware compactness."

If this holds, the novelty becomes sharper:

- not just another feature selector;
- not just a dataset-specific trick;
- a source-only, zero-target-label rule explaining when compact tangent
  subspaces help cross-session EEG-MI.

## Caution

BNCI2014_001 has only 9 subjects.  It is useful as an external sanity check,
not as a statistically decisive third pillar by itself.  Do not overclaim this
dataset.
