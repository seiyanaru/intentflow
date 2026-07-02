# E24: Stability-protected category selection results

Date: 2026-06-30

## One-line verdict

The corrected hypothesis was tested, and it is still not enough.

```text
Hard anatomy mask:
  fails externally on Stieger.

Stability-protected anatomy mask:
  protects Stieger by becoming almost identical to all-selected,
  but does not improve Lee2019.

Conclusion:
  source-subject selection stability is not a useful enough signal for pruning.
```

This is a clean negative result.  It improves our understanding, but it should
not become the main method.

## Hypothesis tested

After E23, the revised idea was:

> Do not delete non-motor/non-posterior dimensions by anatomy alone.  Keep them
> if they are consistently reselected by source-subject leave-one-out.

The rule:

```text
base selected set = top-q source-side subspace

keep(feature) =
  feature is motor/posterior
  OR source-subject leave-one-out selection frequency >= phi
```

Tested thresholds:

```text
phi = 0.95, 0.98, 1.00
```

The earlier informal value `phi=0.75` was rejected before full evaluation
because it kept almost everything on both datasets.

## Artifacts

Lee2019 script:

`intentflow/offline/scripts/analysis/lee2019_stability_protected_category_e24.py`

Lee2019 output:

`intentflow/offline/results/research_outputs/260630_lee2019_stability_protected_category_e24/`

Stieger script:

`intentflow/offline/scripts/analysis/stieger_stability_protected_category_e24.py`

Stieger output:

`intentflow/offline/results/research_outputs/260630_stieger_stability_protected_category_e24/`

Main files in both outputs:

- `summary.csv`
- `summary.json`
- `paired_contrasts.csv`
- `selection_records.csv`
- `selection_stability.csv`
- `selection_size_by_subject.csv`

## E24a: selection stability distributions

### Lee2019, `source_only_q0p25`

Mean selected dimensions:

| Rule | Mean kept | Mean dropped |
|---|---:|---:|
| all selected | 489.0 | 0.0 |
| hard category | 401.6 | 87.4 |
| protected phi=0.95 | 477.2 | 11.8 |
| protected phi=0.98 | 470.9 | 18.1 |
| protected phi=1.00 | 436.5 | 52.5 |

Selection-frequency distribution by category:

| Category | Mean freq | q05 | Median | q95 |
|---|---:|---:|---:|---:|
| both_sensorimotor | 0.983 | 0.887 | 1.000 | 1.000 |
| both_posterior | 0.983 | 0.887 | 1.000 | 1.000 |
| sensorimotor_other | 0.975 | 0.868 | 1.000 | 1.000 |
| sensorimotor_posterior | 0.972 | 0.849 | 1.000 | 1.000 |
| other_other | 0.969 | 0.830 | 0.981 | 1.000 |
| posterior_other | 0.947 | 0.717 | 0.981 | 1.000 |

Important:

> Lee selected `other_other` dimensions are not generally unstable.  Most are
> repeatedly selected by source-subject leave-one-out.

So "unstable other_other noise" is not the right explanation.

### Stieger pure LR, `longitudinal_q0p10`

Mean selected dimensions:

| Rule | Mean kept | Mean dropped |
|---|---:|---:|
| all selected | 183.0 | 0.0 |
| hard category | 176.0 | 7.0 |
| protected phi=0.95 | 183.0 | 0.0 |
| protected phi=0.98 | 183.0 | 0.0 |
| protected phi=1.00 | 182.9 | 0.1 |

Selection-frequency distribution by category:

| Category | Mean freq | q05 | Median | q95 |
|---|---:|---:|---:|---:|
| other_other | 0.999 | 1.000 | 1.000 | 1.000 |
| both_sensorimotor | 0.983 | 0.902 | 1.000 | 1.000 |
| sensorimotor_other | 0.981 | 0.885 | 1.000 | 1.000 |
| posterior_other | 0.972 | 0.885 | 0.975 | 1.000 |
| sensorimotor_posterior | 0.965 | 0.787 | 1.000 | 1.000 |

Important:

> The Stieger frontal/frontopolar block removed by hard category pruning is
> almost perfectly stable.  Stability protection keeps it.

## E24b: Lee2019 performance

Baseline:

| Method | n selected | Gain vs full | 95% CI | q05 | P(gain < -5) |
|---|---:|---:|---:|---:|---:|
| source_only_q0p25 all | 489.0 | +2.176 | [0.509, 3.820] | -6.250 | 0.074 |
| hard category | 401.6 | +2.500 | [0.718, 4.306] | -7.938 | 0.111 |

Stability-protected rules:

| Method | n selected | Gain vs full | 95% CI | q05 | P(gain < -5) |
|---|---:|---:|---:|---:|---:|
| protected phi=0.95 | 477.2 | +1.690 | [0.046, 3.356] | -8.750 | 0.093 |
| protected phi=0.98 | 470.9 | +1.759 | [0.162, 3.380] | -7.938 | 0.093 |
| protected phi=1.00 | 436.5 | +1.759 | [0.185, 3.333] | -6.688 | 0.093 |

Score-top-k controls:

| Method | n selected | Gain vs full | 95% CI |
|---|---:|---:|---:|
| score-top-k hard-size | 401.6 | +1.782 | [0.139, 3.495] |
| score-top-k phi=0.95-size | 477.2 | +2.315 | [0.810, 3.843] |
| score-top-k phi=0.98-size | 470.9 | +2.060 | [0.486, 3.657] |
| score-top-k phi=1.00-size | 436.5 | +1.667 | [0.093, 3.310] |

Paired contrasts:

| Contrast | Mean delta | 95% CI | Verdict |
|---|---:|---:|---|
| phi=0.95 protected - all | -0.486 | [-1.042, +0.023] | worse / not useful |
| phi=0.98 protected - all | -0.417 | [-0.950, +0.093] | worse / not useful |
| phi=1.00 protected - all | -0.417 | [-1.042, +0.208] | worse / not useful |
| phi=0.95 protected - hard | -0.810 | [-1.644, +0.046] | worse |
| phi=0.98 protected - hard | -0.741 | [-1.574, +0.093] | worse |
| phi=1.00 protected - hard | -0.741 | [-1.551, +0.023] | worse |

Interpretation:

```text
Stability-protected pruning does not reproduce the Lee hard-category gain.
```

The hard category rule still has the highest Lee mean gain, but E23 already
showed it fails externally.  The stability-protected version is more plausible,
but empirically weaker.

## E24c: Stieger performance

Baseline:

| Method | n selected | Gain vs full | 95% CI | q05 | P(gain < -5) |
|---|---:|---:|---:|---:|---:|
| longitudinal_q0p10 all | 183.0 | +4.169 | [3.103, 5.343] | -2.370 | 0.016 |
| hard category | 176.0 | +2.444 | [1.097, 3.945] | -4.766 | 0.048 |

Stability-protected rules:

| Method | n selected | Gain vs full | 95% CI | Verdict |
|---|---:|---:|---:|---|
| protected phi=0.95 | 183.0 | +4.169 | [3.127, 5.313] | identical to all |
| protected phi=0.98 | 183.0 | +4.160 | [3.138, 5.363] | almost identical |
| protected phi=1.00 | 182.9 | +4.122 | [3.067, 5.266] | almost identical |

Paired contrasts:

| Contrast | Mean delta | 95% CI | Verdict |
|---|---:|---:|---|
| phi=0.95 protected - all | 0.000 | [0.000, 0.000] | exact no-op |
| phi=0.98 protected - all | -0.009 | [-0.028, 0.000] | negligible |
| phi=1.00 protected - all | -0.047 | [-0.101, -0.009] | tiny loss |
| phi=0.95 protected - hard | +1.725 | [0.973, 2.554] | fixes hard failure |
| phi=0.98 protected - hard | +1.716 | [0.965, 2.549] | fixes hard failure |
| phi=1.00 protected - hard | +1.678 | [0.966, 2.498] | fixes hard failure |

Interpretation:

```text
Stability protection repairs the Stieger hard-mask failure by keeping almost
everything that hard category would have dropped.
```

But this is not a useful new method.  It is basically a no-op relative to the
already-good `longitudinal_q0p10` all-selected policy.

## What this means for the method idea

The revised method was:

```text
anatomy prior + source selection stability
```

The result says:

```text
source selection stability is too saturated.
```

Nearly all selected features are stable under source-subject leave-one-out,
including:

- Lee `other_other`;
- Stieger frontal/frontopolar `other_other`;
- sensorimotor and posterior dimensions.

So stability does not separate useful noncanonical context from removable noise.

## Updated scientific conclusion

Keep:

```text
source-side compact subspace selection is useful.
selected subspaces are often neurostructured.
hand masks are useful for interpretation.
```

Retract:

```text
anatomy-constrained hard pruning is a robust method.
selection-frequency-protected anatomy pruning is a strong improvement.
```

Revise:

```text
Anatomy should be used only as an analysis lens or weak explanatory prior,
not as a pruning rule, unless there is an additional evidence signal beyond
source-selection stability.
```

## Next methodological implication

The path forward is not another anatomical pruning rule.

The only robust method signal so far is:

```text
score-based compactness / longitudinal stability
```

For Lee:

- hard category has the best mean, but fails external validity.
- score-top-k at phi=0.95-equivalent size gives +2.315 pp, slightly above
  original q0.25, without using anatomy.

For Stieger:

- longitudinal q0.10 all-selected remains the strongest and cleanest.
- protected anatomy collapses to the same method.

Therefore the next useful step is not "fix the anatomy mask" again.  It is:

```text
nested source-side compactness selection:
  choose family/fraction by source-subject validation,
  report anatomy only as interpretation.
```

This aligns with the broader evidence:

- dataset/task-specific compactness optimum exists;
- hand anatomical masks are brittle;
- source-score compactness controls are very competitive.

## Practical next action

E25 should be:

```text
Exact nested source-side family/fraction selection across Lee + Stieger + BNCI.
```

Primary question:

> Can source-only validation choose near-optimal compactness without target
> labels and without anatomy masks?

If yes, that becomes the method.

If no, the honest thesis contribution becomes:

> compact source-side subspace selection is promising but not reliably selectable
> without stronger task/dataset descriptors.

