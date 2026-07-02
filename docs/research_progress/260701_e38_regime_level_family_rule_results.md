# E38: regime-level reliability-family rule

Date: 2026-07-01

## One-line verdict

E38は、E37-Bの結論を支持した。

```text
subject-levelに source vs longitudinal を細かく切り替えるより、
regime-levelで使うweighted familyを固定した方が平均精度は上がる。
```

特に:

```text
Lee sensorimotor20: source-weightedに固定
BNCI full-source:   LDA fullに固定
BNCI m8/class:      longitudinal-weightedに固定
```

がsource-validation aggregateから選ばれた。

ただし、BNCI m8/classではglobal longitudinal固定にするとrepeat単位のharm率が少し増える。
したがって最も実用寄りなのは:

```text
regime-level weighted familyを固定しつつ、
repeat-levelではLDA fullへ戻るlocal guardを残す
```

という折衷案。

## Artifacts

Script:

`intentflow/offline/scripts/analysis/e38_regime_level_family_rule.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_e38_regime_level_family_rule/`

Main files:

- `e38_combined_summary.csv`
- `e38_global_choices.csv`
- `e38_combined_selection_records.csv`
- per-regime `e38_summary.csv`
- per-regime `e38_records.csv`
- per-regime `e38_selection_records.csv`
- per-regime `e38_global_choice.csv`

## Protocol

E38 uses the same frozen candidate set as E36:

```text
lda_full
ridge_source_rank_g2_a100
ridge_longitudinal_rank_g2_a100
```

Same weighted-vs-LDA guard:

```text
best weighted gain over LDA full >= +0.5pp
and P(gain < -5pp) <= 0.20
```

Variants:

### 1. `e38_global_regime_family`

Aggregate all source-validation evidence in a regime, choose one family, and apply it to all units.

This is a diagnostic upper/simplification estimate.
It is not a fully independent external validation, because the global family is selected from the same regime's source-validation records.

### 2. `e38_local_guard_global_family_repeat`

Choose the weighted family at regime level, but keep a repeat-level local guard:

```text
if local weighted-vs-LDA guard passes:
  use the regime-level weighted family
else:
  use LDA full
```

This avoids noisy local source-vs-longitudinal switching while preserving an LDA fallback.

### 3. `e38_local_guard_global_family_stable`

Same, but guard is stable/subject-level rather than repeat-level.

## Global choices

Source-validation aggregate selected:

| Regime | Global selected | Best weighted | Source gain | Long gain | Best p_harm |
|---|---|---|---:|---:|---:|
| Lee sensorimotor20 | `ridge_source_rank_g2_a100` | source | +2.353 | +2.141 | 10.4% |
| BNCI full-source | `lda_full` | longitudinal | -0.874 | -0.493 | 2.8% |
| BNCI m8/class | `ridge_longitudinal_rank_g2_a100` | longitudinal | +1.949 | +2.297 | 8.6% |

This matches the qualitative regime story:

```text
Lee: source reliability regime
BNCI full-source: no weighted-Ridge regime
BNCI m8/class: longitudinal reliability regime
```

## Main results: subject-level summary

### Lee2019 sensorimotor20

| Method | Acc | Gain vs LDA full | q05 | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| E36 frozen guarded | 71.644 | +2.037 | -6.250 | 11.1% |
| E38 global family | **71.991** | **+2.384** | -6.250 | 9.3% |
| fixed source-weighted | **71.991** | **+2.384** | -6.250 | 9.3% |
| fixed longitudinal-weighted | 71.829 | +2.222 | -6.250 | 13.0% |
| LDA full | 69.606 | 0.000 | 0.000 | 0.0% |

E38 improves over E36 by removing five local longitudinal choices:

```text
E36:  source 49, longitudinal 5
E38:  source 54
```

Interpretation:

```text
Lee should be treated as a source-weighted regime, not a subject-level
source-vs-longitudinal switching problem.
```

### BNCI2014_001 full-source

All guarded/global variants select:

```text
lda_full 9/9
```

Result:

```text
Acc = 77.599
Gain vs LDA full = 0.000
P(gain<-5pp) = 0.0%
```

Interpretation:

```text
BNCI full-source remains the negative-control regime.
E38 does not accidentally force weighted Ridge here.
```

### BNCI2014_001 m8/class

Subject-level:

| Method | Acc | Gain vs LDA full | q05 | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| E36 repeat | 69.444 | +1.761 | -1.235 | 0.0% |
| E36 stable | 69.825 | +2.142 | -1.361 | 0.0% |
| E38 local guard + global family repeat | 69.593 | +1.910 | -1.326 | 0.0% |
| E38 global family / stable | **70.010** | **+2.327** | -1.361 | 0.0% |
| fixed source-weighted | 69.738 | +2.055 | -1.013 | 0.0% |
| fixed longitudinal-weighted | **70.010** | **+2.327** | -1.361 | 0.0% |

Selection:

```text
E36 repeat:
  LDA full                         52
  source-weighted                  81
  longitudinal-weighted           155

E38 local guard + global family repeat:
  LDA full                         52
  longitudinal-weighted           236

E38 global/stable:
  longitudinal-weighted           288
```

Interpretation:

```text
BNCI m8 is a longitudinal-weighted regime.
Local source-vs-longitudinal switching mostly adds noise.
```

## Unit-level risk caveat

For BNCI m8/class, subject-level averages hide repeat-level harm.

Unit-level summary:

| Method | Mean gain | q05 | P(gain<-5pp) | P(gain<0) |
|---|---:|---:|---:|---:|
| E38 global longitudinal | +2.327 | -7.258 | 8.3% | 28.8% |
| E38 local guard + global family repeat | +1.910 | -6.169 | 6.3% | 24.3% |
| LDA full | 0.000 | 0.000 | 0.0% | 0.0% |

This matters.

E38 global longitudinal gives the highest mean, but repeat-level harm increases.
The local guard variant gives lower mean but slightly better repeat-level risk.

Therefore the practical recommendation is not simply:

```text
always use the global weighted family
```

but:

```text
use regime-level family selection,
and decide whether to retain a local LDA fallback depending on the desired
mean-risk tradeoff.
```

## What E38 changes

### Keep

```text
source validation is useful as a regime/cohort-level signal.
```

It selects the expected family in all three regimes.

### Revise

E36 should not be described as:

```text
subject-level family selector
```

E38 shows a cleaner description:

```text
source-validation identifies the regime's reliability-weighted family;
the LDA fallback controls whether to apply that family locally.
```

### Reject

Do not keep local source-vs-longitudinal switching as the main novelty.
It created:

- 5 unnecessary longitudinal choices in Lee;
- noisy source/long switching in BNCI m8 repeat.

## Current best method candidates

There are now two defensible variants:

### Variant A: accuracy-first regime-level family

```text
choose one family per regime/cohort:
  Lee -> source-weighted
  BNCI full -> LDA full
  BNCI m8 -> longitudinal-weighted
```

Pros:

- best mean accuracy;
- simplest story;
- matches E37-B evidence that subject-level prediction is weak.

Cons:

- same-regime source-validation selection is not independent;
- BNCI m8 repeat-level harm increases.

### Variant B: guarded practical family

```text
choose weighted family at regime level,
but keep local weighted-vs-LDA guard.
```

Pros:

- better mean than E36 repeat;
- lower repeat-level harm than global weighted family;
- avoids local source-vs-long switching noise.

Cons:

- slightly lower mean than global family.

Current recommendation:

```text
Use Variant B as the practical method.
Use Variant A as the mechanism/upper-simplification result.
```

## Next experiment: E39

E39 should freeze Variant B:

```text
Regime-level weighted family + local LDA fallback
```

and compare it directly against:

- E36 frozen guarded;
- fixed best soft family;
- LDA full;
- hard top-k best.

Key outputs:

1. subject-level mean/risk;
2. unit-level mean/risk for BNCI m8;
3. selection counts;
4. method complexity comparison.

Pass criteria:

```text
Lee:
  match fixed source-weighted and beat E36.

BNCI full:
  equal LDA full.

BNCI m8 repeat:
  beat E36 repeat by >= +0.1pp,
  keep unit P(gain<-5pp) no worse than E36 repeat by more than +1pp.
```

If E39 passes, the method story becomes:

```text
First choose the cohort/regime reliability family;
then apply a conservative LDA fallback locally.
```

This aligns with all evidence so far.

