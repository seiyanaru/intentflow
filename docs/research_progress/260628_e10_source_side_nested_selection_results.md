# 260628 E10: source-side nested/proxy selection on the E9 compactness grid

## One-line verdict

E10は通った。

ただし、正確な主張は:

```text
source-side validation can choose the family/compactness at the dataset/task level
without looking at the held-out target subject.
```

まだ言いすぎな主張は:

```text
exact double-LOSO Stieger selection is proven.
per-target-subject adaptive candidate search is robust.
```

Stiegerは既存LOSO recordからのfast proxyなので、journalで中央主張にするなら
exact double-LOSOの注記または追加検証が必要。

## Artifacts

Script:

```text
intentflow/offline/scripts/analysis/source_side_nested_selection_e10.py
```

Primary candidate output:

```text
intentflow/offline/results/research_outputs/260628_e10_source_side_nested_selection_primary/
```

With sep-no-drift output:

```text
intentflow/offline/results/research_outputs/260628_e10_source_side_nested_selection_with_sep/
```

All source-side q-grid output:

```text
intentflow/offline/results/research_outputs/260628_e10_source_side_nested_selection_all_source_side/
```

Main files:

```text
nested_summary.json
nested_summary.csv
nested_selection_records.csv
source_validation_diagnostics.csv
```

## Primary candidate set

Predefined small candidate set:

```text
full_q1p00

source_only:
  q0.15, q0.25, q0.50

longitudinal:
  q0.05, q0.10, q0.20, q0.25, q0.50
```

Selection rule:

```text
maximize source-validation mean gain
subject to P(gain < -5pp) <= 0.20
```

## Main result

### Stieger global source-side rule

Source-side validation selects:

```text
longitudinal_q0.10 for all units
```

Result:

| policy | acc | gain vs full | 95% CI | R10 loss | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| global_risk | 67.681 | +2.357 | [+1.449, +3.326] | 12.383 | 17.5% |

Interpretation:

```text
The E9 Stieger fixed-q result is not merely target-picked.
The same q/family is selected by source-subject validation.
```

### Stieger condition-specific source-side rule

Source-side validation selects:

```text
pure_lr: longitudinal_q0.05
pure_ud: longitudinal_q0.20
```

Result:

| policy | acc | gain vs full | 95% CI | R10 loss | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| condition_risk | 68.044 | +2.719 | [+1.840, +3.673] | 11.753 | 14.1% |

Interpretation:

```text
This is the strongest current Stieger result.
It supports the E9 compactness-frontier story:
  LR prefers very compact q0.05.
  UD prefers moderate q0.20.
```

This is also practically interpretable because LR/UD task condition is known
before evaluation.  It is not target-label selection.

### Lee2019 global source-side rule

Source-side validation selects:

```text
source_only_q0.25 for all subjects
```

Result:

| policy | acc | gain vs full | 95% CI | R10 loss | P(gain<-5) |
|---|---:|---:|---:|---:|---:|
| global_risk | 71.852 | +2.176 | [+0.509, +3.843] | 7.500 | 7.4% |

Interpretation:

```text
Lee2019 also passes the anti-post-hoc test.
The rule selects the same source_only_q0.25 that E9 identified as best.
```

## Candidate-set sensitivity

### with-sep candidate set

Adding:

```text
sep_no_drift_q0.20
sep_no_drift_q0.50
```

does not change the selected primary policies:

```text
Stieger global: longitudinal_q0.10
Stieger condition-specific: LR q0.05, UD q0.20
Lee2019: source_only_q0.25
```

So the main result is not fragile to this small candidate expansion.

### all-source-side q-grid

Using all 22 source-side candidates:

| dataset/policy | gain | P(gain<-5) | selected |
|---|---:|---:|---|
| Stieger global_risk | +2.357 | 17.5% | longitudinal_q0.10 |
| Stieger condition_risk | +2.558 | 15.2% | LR q0.05, UD q0.20/q0.15 |
| Lee2019 global_risk | +2.176 | 7.4% | source_only_q0.25 |

Interpretation:

```text
Global policies are stable.
But Stieger condition-specific becomes slightly worse when too many q values are allowed.
```

This is important:

```text
Do not frame the method as "search many candidates".
Frame it as a small, pre-registered compactness candidate set.
```

## Scientific interpretation

E10 strengthens the current paper story:

```text
Source-side tangent subspace selection improves zero-target-label cross-session EEG-MI.

The key design axis is subspace compactness.

The compactness/family can be selected by source-subject validation,
without target labels.
```

Dataset/task-specific result:

```text
Stieger LR:
  very compact longitudinal q0.05

Stieger UD:
  moderate longitudinal q0.20

Lee2019:
  broader source-discriminative source_only q0.25
```

This is stronger than the previous E8/E9 story because it addresses the most obvious
post-hoc criticism.

## What not to overclaim

Do not claim:

```text
longitudinal is universally better.
more sensorimotor concentration always wins.
per-subject candidate adaptation is solved.
```

Also do not hide the Stieger limitation:

```text
Stieger E10 is a nested/proxy audit from existing LOSO records,
not an exact recomputation excluding both outer and inner held-out subjects.
```

## Next decision

The next experiment should not be another method search.

The next useful steps are:

1. Make the compactness-frontier figure.
2. Add generic feature-selection baselines.
3. Decide whether exact Stieger double-LOSO is needed for the target venue.

Priority:

```text
E11: compactness frontier figure and paper table
E12: generic feature-selection baseline
E13: exact Stieger double-LOSO only if aiming journal-level claim
```

