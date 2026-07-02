# 260627 E6 nested source-side candidate selection results

## One-line verdict

E6は半分通ったが、主張は慎重にするべき。

```text
通った:
  source subjectsだけで候補を選ぶと、
  Stiegerではlongitudinal、Lee2019ではsource-onlyが選ばれ、
  どちらもfull featureより改善する。

注意:
  Stiegerは既存LOSO recordからのfast proxyであり、厳密double-LOSOではない。
  Lee2019 exact nested 12-subject subsetでは、subject-wise候補選択がやや不安定。
```

したがって現時点での安全な主張は:

```text
source-side validation can identify a good tangent-subspace candidate at the dataset/task level.
```

まだ危ない主張は:

```text
per-target-subject nested candidate selection is robust.
```

## Artifacts

Fast source-side nested from existing records:

```text
intentflow/offline/scripts/analysis/source_side_nested_selection_from_records.py
intentflow/offline/results/research_outputs/260627_source_side_nested_selection_from_records/
```

Lee2019 exact nested subset/full script:

```text
intentflow/offline/scripts/analysis/lee2019_exact_nested_subspace_selection.py
intentflow/offline/results/research_outputs/260627_lee2019_exact_nested_subspace_selection_12/
```

The full 54-subject exact Lee run was stopped because it was too slow with the current implementation.
It performs many high-dimensional shrinkage-LDA fits and needs caching/optimization before full use.

## E6-fast result: Stieger

Important limitation:

```text
This uses existing Stieger LOSO result records.
For inner source-validation subjects, the original metric excluded the inner subject,
but not necessarily the outer held-out subject.
So this is a fast proxy, not exact double-LOSO.
```

### Global risk selection

Source-validation rule:

```text
maximize source-validation mean gain
subject to P(gain < -5pp) <= 0.20
```

Result:

```text
chosen candidate:
  longitudinal_q0.10 for all units

accuracy:
  67.681

gain vs full:
  +2.357 pp
  CI [+1.449, +3.326]

risk:
  q05 -11.285
  R10 loss 12.383
  P(gain < -5pp) 17.5%
```

### Condition-specific risk selection

Result:

```text
chosen candidates:
  pure_lr: longitudinal_q0.10
  pure_ud: longitudinal_q0.25

accuracy:
  67.734

gain vs full:
  +2.409 pp
  CI [+1.540, +3.324]

risk:
  q05 -10.145
  R10 loss 11.451
  P(gain < -5pp) 16.4%
```

Interpretation:

```text
On Stieger, source-side validation consistently selects longitudinal candidates.
This supports the earlier Stieger result, but exact double-LOSO is still needed
if this becomes a central journal claim.
```

## E6-fast result: Lee2019 full n=54

Result:

```text
chosen candidate:
  source_only_q0.25 for all subjects

accuracy:
  71.852

gain vs full:
  +2.176 pp
  CI [+0.509, +3.843]

risk:
  q05 -6.250
  R10 loss 7.500
  P(gain < -5pp) 7.4%
```

Interpretation:

```text
Lee2019 does not select longitudinal.
It selects source-only q0.25.
This reinforces the revised paper axis:
  source-side tangent subspace selection
not:
  universal longitudinal stability selection
```

## Lee2019 exact nested subset n=12

This is the stricter implementation:

```text
For held-out H:
  inner validation subject S is evaluated with stats excluding both H and S.
```

The 54-subject run was too slow in the current implementation, so n=12 was used as a smoke/pilot.

Result:

```text
nested_risk:
  accuracy 72.500
  gain vs full +4.167
  CI [+0.938, +7.188]
  q05 -5.438
  R10 loss 5.625
  P(gain < -5pp) 8.3%

chosen candidates:
  source_only_q0.25: 7/12
  longitudinal_q0.10: 3/12
  source_only_q0.10: 2/12
```

Fixed candidates on the same n=12 subset:

```text
fixed source_only_q0.25:
  accuracy 73.646
  gain +5.313
  CI [+3.021, +7.708]
  P(gain < -5pp) 0.0%

fixed longitudinal_q0.10:
  accuracy 72.292
  gain +3.958
  CI [+1.042, +6.563]
  P(gain < -5pp) 8.3%
```

Interpretation:

```text
Exact nested selection improves over full, but on this small subset it is worse
than simply using fixed source_only_q0.25.

This suggests subject-wise candidate selection can overfit source-validation noise.
For the paper, fixed source-side subspace selection may be a cleaner and stronger
primary method than per-target candidate selection.
```

## Scientific decision after E6

### Keep

```text
source-side tangent subspace selection is robust across Stieger and Lee2019.
```

### Weaken

```text
nested per-subject candidate selection is the main method.
```

### Reframe

The most defensible method is:

```text
Use source-side validation to choose a small candidate family at the dataset/task level,
or use a fixed robust candidate such as source_only_q0.25 as the cross-dataset baseline.
```

Then report longitudinal as:

```text
beneficial in Stieger-like multi-session settings,
but not universally superior.
```

## Next action

Do not spend the next major effort on more candidate-selection logic.

The next best experiment is E7:

```text
Stieger session-depth ablation:
  K=2,3,5,all source sessions
  compare longitudinal vs source-only
```

Why:

```text
If longitudinal only helps when enough source longitudinal sessions exist,
then Lee2019's failure is explainable rather than fatal.
```

Decision rule:

```text
If longitudinal - source_only increases with K:
  keep longitudinal as conditional mechanism.

If not:
  drop longitudinal from method title.
  use source-side discriminative tangent subspace selection as the main method.
```

