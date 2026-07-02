# 260627 Unified ablation: Stieger vs Lee2019

## One-line verdict

The robust cross-dataset result is `source-side subspace selection`.
The dataset-dependent result is the `longitudinal drift penalty`.

```text
Stieger:
  longitudinal score is best.

Lee2019:
  source-only score is best.

Both:
  random top-k is bad.
  therefore the selected subspace is not a trivial dimensionality-reduction artifact.
```

## Unified table

All gains are against the full tangent feature baseline in the same dataset/protocol.

| Dataset | Method | Acc | Gain vs full | 95% CI | R10 loss | P(gain < -5pp) |
|---|---|---:|---:|---:|---:|---:|
| Stieger primary | longitudinal q0.25 | 67.082 | +1.758 | [+0.954, +2.612] | 11.529 | 16.7% |
| Stieger primary | longitudinal q0.10 | 67.681 | +2.357 | [+1.455, +3.321] | 12.152 | 17.1% |
| Stieger primary | source-only q0.25 | 66.638 | +1.313 | [+0.657, +2.003] | 10.991 | 16.8% |
| Stieger primary | source-only q0.10 | 66.763 | +1.439 | [+0.574, +2.310] | 13.658 | 19.0% |
| Stieger primary | sep-no-drift q0.25 | 66.397 | +1.072 | [+0.422, +1.719] | 12.053 | 16.7% |
| Stieger primary | target-only q0.25 | 65.994 | +0.669 | [+0.061, +1.285] | 11.957 | 17.9% |
| Stieger primary | random q0.25 | 61.518 | -3.807 | n/a | n/a | n/a |
| Stieger primary | random q0.10 | 58.881 | -6.444 | n/a | n/a | n/a |
| Lee2019 | source-only q0.25 | 71.852 | +2.176 | [+0.555, +3.866] | 7.500 | 7.4% |
| Lee2019 | longitudinal q0.10 | 71.528 | +1.852 | [-0.278, +3.935] | 13.125 | 20.4% |
| Lee2019 | longitudinal q0.25 | 71.042 | +1.366 | [-0.301, +3.079] | 8.333 | 11.1% |
| Lee2019 | sep-no-drift q0.25 | 70.903 | +1.227 | [-0.370, +2.801] | 8.125 | 18.5% |
| Lee2019 | target-only q0.25 | 70.556 | +0.880 | [-0.602, +2.315] | 8.958 | 13.0% |
| Lee2019 | random q0.25 | 63.466 | -6.209 | n/a | 18.724 | 53.2% |
| Lee2019 | random q0.10 | 60.205 | -9.470 | n/a | 24.271 | 65.5% |

## What this means

### Claim to keep

```text
Before zero-target-label cross-session evaluation,
selecting a compact Riemann tangent subspace from source-side labeled data
improves over using all tangent dimensions.
```

This is supported on both Stieger and Lee2019.

### Claim to weaken

```text
Longitudinal same-class drift is the essential universal scoring term.
```

This is supported on Stieger but not Lee2019.
On Lee2019, source-only discriminability is stronger and safer.

### Why the result is still useful

The negative random controls are strong:

```text
Stieger random q0.25: -3.807 pp
Lee random q0.25: -6.209 pp
```

So the effect is not "LDA likes fewer dimensions".
It is "the chosen source-side tangent dimensions matter".

## Revised title candidates

Avoid:

```text
Source-side longitudinal subspace selection for zero-target-label cross-session EEG-MI
```

Better:

```text
Source-Side Tangent Subspace Selection for Zero-Target-Label Cross-Session EEG Motor Imagery
```

or, if we want the longitudinal angle but not overclaim it:

```text
When Does Longitudinal Stability Help? Source-Side Subspace Selection for Zero-Label Cross-Session EEG-MI
```

## Next experiment

The next useful analysis is not another gate.
It is an interaction analysis:

```text
When does longitudinal beat source-only?
```

Concrete dependent variable:

```text
per held-out unit:
  Δ = acc(longitudinal best fixed q) - acc(source_only best fixed q)
```

Candidate predictors:

```text
source_between concentration
same_class_drift magnitude
drift / between ratio
number of source sessions used to estimate drift
baseline full-feature accuracy
```

Decision rule:

```text
If the interaction is interpretable:
  paper story = source-side subspace selection + when longitudinal stability matters.

If not:
  paper story = simple source-side discriminative subspace baseline beats unsafe adaptation attempts;
                longitudinal term is exploratory/dataset-specific.
```

