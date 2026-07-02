# 260628 E17: task-family descriptor audit results

## One-line verdict

E17は完了。

結論:

```text
StiegerとLee2019の違いは、かなり説明可能になった。
```

ただし、万能な自動ルールが見つかったわけではない。

最も安全な解釈は:

```text
Stieger:
  source-side / longitudinal population ranking is necessary.
  validation is acceptable when m>=20.

Lee2019:
  generic variance pruning contributes, but source_only q0.25 is still best.
  small/mid-pool validation is unstable, so fixed source_only policy is preferred.
```

つまり今後の方法論は:

```text
large/discovery regime:
  source-side validation can discover task-family policies.

small/deployment regime:
  use fixed robust policies learned from prior source-side analyses.
```

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/task_family_descriptor_audit_e17.py
```

Output:

```text
intentflow/offline/results/research_outputs/260628_e17_task_family_descriptor_audit/
```

Main files:

```text
summary.json
task_panel_descriptors.csv
validation_stability_descriptors.csv
regime_recommendations.csv
```

## Task-panel descriptor result

| dataset/task | best source-side | gain | best generic | gain | source-side - generic | interpretation |
|---|---|---:|---|---:|---:|---|
| Stieger LR | longitudinal q0.05 | +4.054 | self-source Fisher q0.05 | +0.296 | +3.758 | source-side ranking necessary |
| Stieger UD | longitudinal q0.20 | +1.287 | self-source Fisher q0.50 | -0.236 | +1.523 | source-side ranking necessary |
| Stieger pooled | longitudinal q0.10 | +2.357 | self-source Fisher q0.50 | -0.248 | +2.605 | source-side ranking necessary |
| Lee2019 LR | source_only q0.25 | +2.176 | self-source variance q0.25 | +1.505 | +0.671 | generic variance contributes, but source-side still best |

Interpretation:

```text
Stieger:
  generic self-source feature selection does not explain the effect.

Lee2019:
  generic variance pruning is nontrivial,
  but it still does not beat source_only q0.25.
```

This supports the current mechanism:

```text
The useful compact subspace is not just smaller.
It must be ranked by the right source-side statistic.
```

## Validation stability descriptors

### Stieger global

| m | retention vs all | family entropy | longitudinal rate | source_only rate | validation - fixed |
|---:|---:|---:|---:|---:|---:|
| 10 | 86.6% | 0.554 | 88.1% | 11.5% | -0.317 |
| 20 | 93.6% | 0.178 | 97.3% | 2.7% | -0.151 |
| 40 | 95.8% | 0.001 | 99.99% | 0.01% | -0.098 |

Recommendation:

```text
validation acceptable with m>=20,
but fixed longitudinal q0.10 remains slightly better.
```

### Stieger condition-specific

| m | retention vs all | family entropy | longitudinal rate | source_only rate | validation - fixed |
|---:|---:|---:|---:|---:|---:|
| 10 | 89.2% | 0.722 | 83.2% | 15.4% | -0.292 |
| 20 | 93.0% | 0.485 | 89.8% | 10.0% | -0.191 |
| 40 | 95.9% | 0.183 | 97.2% | 2.8% | -0.112 |

Recommendation:

```text
validation acceptable with m>=20,
but condition-fixed LR q0.05 / UD q0.20 is still the cleanest deployment rule.
```

### Lee2019

| m | retention vs all | family entropy | longitudinal rate | source_only rate | validation - fixed |
|---:|---:|---:|---:|---:|---:|
| 10 | 61.3% | 1.119 | 49.4% | 48.7% | -0.842 |
| 20 | 58.5% | 0.998 | 41.6% | 58.2% | -0.902 |
| 40 | 58.8% | 0.801 | 24.3% | 75.7% | -0.896 |

Recommendation:

```text
fixed policy preferred; validation unstable.
```

Even m=40 still selects longitudinal too often.

## Scientific meaning

E17 resolves the E15/E16 tension:

```text
Stieger validation becomes stable with moderate pools.
Lee validation does not.
```

Why?

Most likely:

```text
Stieger has a strong and repeated longitudinal compact-core signal.
Lee2019 has a weaker family margin: source_only is best, but longitudinal is close enough
on source-validation samples to confuse small/mid-pool selection.
```

The evidence:

```text
Lee m=20:
  source_only family 58.2%
  longitudinal family 41.6%
  validation - fixed = -0.902 pp
```

So Lee is not a failure of source-side subspace selection.
It is a failure of small-pool adaptive family selection.

## Current best research framing

The method should be framed as:

```text
Source-side tangent subspace selection for zero-target-label cross-session EEG-MI.
```

With two regimes:

```text
1. Policy discovery:
   Use source-side validation on a sufficiently large historical cohort
   to identify task-family policies.

2. Deployment:
   Use fixed robust policies for the task family,
   rather than re-validating from a small source pool.
```

Concrete fixed policies from current evidence:

```text
Stieger-like LR:
  longitudinal q0.05

Stieger-like UD:
  longitudinal q0.20

Stieger global if task condition unavailable:
  longitudinal q0.10

Lee-like LR:
  source_only q0.25 for mean gain
  source_only q0.50 for lower-tail risk
```

## What is still unresolved

The remaining serious limitation is externality:

```text
Only two datasets.
Only one Lee-like two-session LR dataset.
Fixed task-family policies may still be dataset-specific.
```

Therefore the next truly high-value experiment is not another selector.

It is either:

```text
A. third dataset / external task-family check
```

or, if data is not available:

```text
B. exact Stieger double-LOSO audit
```

But method-development-wise, E8-E17 already converged.

