# 260628 E9: fraction / compactness sweep results

## One-line verdict

E9は完了。

結果は、E8で立てた compact-core vs broader-subspace 仮説をかなり支持する。
ただし、主張は少し修正する必要がある。

```text
Stieger LR:
  very compact q0.05/q0.10 が最も強い。

Stieger UD:
  中間 q0.15-0.25 が最も強い。

Lee2019:
  source-only は q0.25 が最も強い。
  longitudinal は q0.10 が最も強いが、source-only q0.25には負ける。
```

つまり、

```text
compactness optimum is dataset/task dependent.
```

という主張は守れる。

一方で、

```text
Lee2019は単に大きいqを好む
```

という単純化はやや危険。
Lee2019で本当に強いのは `source-only q0.25` であり、
`longitudinal` family 内では q0.10 が最良。

## Artifacts

Stieger:

```text
intentflow/offline/results/research_outputs/260628_stieger_fraction_sweep_e9_full/
```

Lee2019:

```text
intentflow/offline/results/research_outputs/260628_lee2019_fraction_sweep_e9_full/
```

Main files:

```text
summary.json
summary.csv
selection_records.csv
selection_records.json  # Stieger only
```

## Stieger primary LR+UD

Best method:

```text
longitudinal q0.10
  acc 67.681
  gain +2.357
  CI [+1.455, +3.321]
  R10 loss 12.152
  P(gain<-5) 17.1%
```

Top methods:

| method | acc | gain | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|
| longitudinal q0.10 | 67.681 | +2.357 | 12.152 | 17.1% |
| longitudinal q0.05 | 67.601 | +2.277 | 13.867 | 18.5% |
| longitudinal q0.15 | 67.508 | +2.184 | 11.439 | 17.4% |
| longitudinal q0.20 | 67.338 | +2.013 | 11.517 | 15.5% |
| longitudinal q0.25 | 67.082 | +1.758 | 11.529 | 16.7% |
| source-only q0.15 | 66.938 | +1.614 | 11.964 | 16.1% |
| source-only q0.20 | 66.793 | +1.469 | 11.830 | 17.6% |
| source-only q0.25 | 66.638 | +1.313 | 10.991 | 16.8% |

Interpretation:

```text
For Stieger pooled LR+UD, the best point is compact longitudinal q0.10.
```

## Stieger LR

Best method:

```text
longitudinal q0.05
  acc 70.928
  gain +4.054
  CI [+2.856, +5.414]
  R10 loss 10.406
  P(gain<-5) 10.8%
```

Notable:

```text
longitudinal q0.10:
  gain +3.558

source-only q0.05:
  gain +3.548

source-only q0.25:
  gain +2.063
```

Interpretation:

```text
Stieger LR clearly prefers a very compact subspace.
The family effect is smaller at q0.05:
  source-only q0.05 nearly matches longitudinal q0.10.

This suggests that part of the longitudinal advantage is compactness,
not only the drift term.
```

## Stieger UD

Best method:

```text
longitudinal q0.20
  acc 64.944
  gain +1.287
  CI [+0.354, +2.274]
  R10 loss 12.376
  P(gain<-5) 17.6%
```

Near-best:

```text
longitudinal q0.15:
  gain +1.277

longitudinal q0.25:
  gain +1.192

longitudinal q0.10:
  gain +1.084
```

Interpretation:

```text
UD does not prefer the smallest q.
It prefers a moderate compactness range, q0.15-0.25.
```

## Lee2019

Best method:

```text
source-only q0.25
  acc 71.852
  gain +2.176
  CI [+0.486, +3.820]
  R10 loss 7.500
  P(gain<-5) 7.4%
```

Top methods:

| method | acc | gain | R10 | P(gain<-5) |
|---|---:|---:|---:|---:|
| source-only q0.25 | 71.852 | +2.176 | 7.500 | 7.4% |
| longitudinal q0.10 | 71.528 | +1.852 | 13.125 | 20.4% |
| source-only q0.50 | 71.435 | +1.759 | 7.708 | 5.6% |
| longitudinal q0.20 | 71.204 | +1.528 | 8.125 | 9.3% |
| source-only q0.15 | 71.134 | +1.458 | 8.542 | 14.8% |
| source-only q0.20 | 71.111 | +1.435 | 8.750 | 16.7% |
| longitudinal q0.25 | 71.042 | +1.366 | 8.333 | 11.1% |

Interpretation:

```text
Lee2019 source-only prefers q0.25.
Longitudinal family itself still prefers q0.10.
But longitudinal q0.10 has much worse lower-tail risk than source-only q0.25.
```

## What E9 changes

### Stronger

The compactness story is now supported:

```text
Stieger LR:
  compact q0.05/q0.10 best

Stieger UD:
  moderate q0.15/q0.25 best

Lee2019:
  source-only q0.25 best
```

### Weaker

The story is not:

```text
longitudinal is universally better.
```

Nor:

```text
more sensorimotor concentration always wins.
```

Because Lee2019 longitudinal q0.10 is compact and sensorimotor-concentrated,
but source-only q0.25 is better and safer.

## Revised paper framing

Best current framing:

```text
Source-side tangent subspace selection improves zero-target-label
cross-session EEG-MI.

The key design axis is subspace compactness.
The optimal compactness is task/dataset dependent:
  Stieger LR needs a compact core.
  Stieger UD needs a moderate core.
  Lee2019 benefits from a broader source-discriminative subspace.
```

Longitudinal should be framed as:

```text
a stability-biased compactness mechanism,
not the universal default.
```

## Next action

No more broad method search for now.

Next should be paper consolidation:

```text
1. Create final result table:
   Stieger LR / Stieger UD / Stieger primary / Lee2019
   best fixed source-side subspace candidate

2. Create one compactness frontier figure:
   x-axis: q
   y-axis: gain vs full
   lines: source-only vs longitudinal
   panels: Stieger LR, Stieger UD, Lee2019

3. Reframe contribution:
   from safe selective adaptation
   to source-side subspace selection and compactness frontier.
```

