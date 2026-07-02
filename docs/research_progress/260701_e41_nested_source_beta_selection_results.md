# E41: nested source-validation beta selection

Date: 2026-07-01

## One-line verdict

E41は、E40の一番大きな弱点だった

```text
beta=0.6/0.7 は target 結果を見た post-hoc 選択ではないか？
```

に対して、かなり良い答えを出した。

結論:

```text
source-validation は beta=0.7 を強く支持する。
ただし、held-foldごとに family/beta を細かく切り替えると性能は落ちる。
```

したがって、最終候補は:

```text
regime-level family + source-validation modal beta=0.7
```

であり、

```text
outer-heldごとの local family/beta selector
```

ではない。

これはE37-B/E38の教訓と一致している。

## Artifacts

Script:

`intentflow/offline/scripts/analysis/e41_nested_source_beta_selection.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_e41_nested_source_beta_selection/`

Main files:

- `e41_combined_summary.csv`
- `e41_combined_selection_records.csv`
- `e41_combined_outer_records.csv`
- `e41_combined_family_summaries.csv`
- `e41_combined_beta_summaries.csv`
- per-regime `e41_inner_validation_records.csv`
- per-regime `e41_selection_records.csv`
- per-regime `e41_outer_records.csv`
- per-regime `e41_summary.csv`

## Protocol

For each outer held subject:

1. exclude the held subject from selection;
2. use remaining source subjects only;
3. use E33 source-validation records to choose family:

```text
LDA vs source-weighted Ridge vs longitudinal-weighted Ridge
```

4. if weighted family passes:

```text
family mean gain >= +0.5pp
and P(inner gain < -5pp) <= 0.20
```

then choose beta from:

```text
beta ∈ {0.0, 0.1, ..., 1.0}
```

using source-validation only:

```text
choose highest mean gain among beta with
P(inner gain < -5pp) <= 0.08
```

5. fit/evaluate once on the held subject.

Important:

```text
held target labels are never used to select family or beta.
```

## Selection behavior

### Lee2019 sensorimotor20

Selection counts:

```text
source family:        49/54
longitudinal family:   5/54
```

Beta counts:

```text
beta=0.7: 39/54
beta=0.8:  7/54
beta=0.6:  3/54
beta=1.0:  2/54
beta=0.4:  1/54
beta=0.5:  1/54
beta=0.9:  1/54
```

Mean selected beta:

```text
0.713
```

Mode:

```text
0.7
```

Interpretation:

```text
Source-validation overwhelmingly says:
  Lee = source-weighted family, beta around 0.7.
```

The five longitudinal choices are probably local selection noise, same as E38.

### BNCI2014_001 full-source

Selection:

```text
LDA: 9/9
```

Interpretation:

```text
Negative-control behavior holds.
```

### BNCI2014_001 m8/class

Selection counts:

```text
longitudinal family: 8/9
source family:       1/9
```

Beta counts:

```text
beta=0.7: 5/9
beta=0.8: 2/9
beta=0.6: 2/9
```

Mean selected beta:

```text
0.700
```

Mode:

```text
0.7
```

Interpretation:

```text
Source-validation again points to beta=0.7.
```

The one source-family selection is likely local noise.

## Main performance

### E41 local source-selected method

| Regime | Summary level | Gain vs LDA | q05 | CVaR10 | P(gain<-5) | P(gain<0) |
|---|---|---:|---:|---:|---:|---:|
| Lee sensorimotor20 | subject/unit | +2.153 | -5.438 | -6.667 | 7.4% | 27.8% |
| BNCI full-source | subject/unit | +0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| BNCI m8/class | subject | +2.184 | -0.842 | -1.588 | 0.0% | 11.1% |
| BNCI m8/class | unit | +2.184 | -6.452 | -6.646 | 6.3% | 27.4% |

This passes the minimal E41 criteria:

```text
Lee:
  gain >= +2.0pp
  P(gain<-5) lower than beta=1 weighted Ridge

BNCI full:
  returns to LDA

BNCI m8 unit:
  gain >= +2.0pp
  P(gain<-5) <= 6.3%
```

So E41 does not kill the method.

But it also shows local selection is not the best control structure.

## Local selection loses to fixed regime beta

### Lee2019 sensorimotor20

| Method | Gain vs LDA | P(gain<-5) | P(gain<0) |
|---|---:|---:|---:|
| E41 local selected | +2.153 | 7.4% | 27.8% |
| fixed source beta=0.7 | **+2.500** | 7.4% | **24.1%** |
| fixed source beta=0.6 | +2.222 | **5.6%** | 25.9% |
| fixed source beta=1.0 | +2.384 | 9.3% | 33.3% |

Paired:

```text
E41 local selected - fixed source beta=0.7:
  -0.347pp
```

### BNCI2014_001 m8/class, unit-level

| Method | Gain vs LDA | q05 | P(gain<-5) | P(gain<0) |
|---|---:|---:|---:|---:|
| E41 local selected | +2.184 | -6.452 | 6.3% | 27.4% |
| fixed longitudinal beta=0.7 | **+2.470** | -5.645 | 6.3% | **26.0%** |
| fixed longitudinal beta=0.6 | +2.361 | **-4.839** | **3.8%** | 26.4% |
| fixed longitudinal beta=1.0 | +2.327 | -7.258 | 8.3% | 28.8% |

Paired:

```text
E41 local selected - fixed longitudinal beta=0.7:
  -0.286pp

E41 local selected - fixed longitudinal beta=0.6:
  -0.176pp
```

Interpretation:

```text
Local source-validation selection is conservative enough to pass,
but it injects avoidable family/beta switching noise.
```

This exactly mirrors E38:

```text
subject/repeat-level switching is weaker than regime-level family fixation.
```

## What E41 changes

### Keep

```text
source-validation supports beta around 0.7.
```

This removes the most damaging objection against E40:

```text
beta=0.7 was only chosen from held target labels.
```

Now beta=0.7 is also the modal source-validation choice in both positive regimes.

### Revise

Do not use:

```text
outer-held local family/beta selector
```

as the final method.

It passes, but it is not best.

### Strengthen

The actual final control structure should be:

```text
1. choose family at regime level;
2. choose beta at regime level;
3. apply the same family/beta to all held units in that regime.
```

In the current evidence:

| Regime | Source-validation decision |
|---|---|
| Lee sensorimotor20 | source family, beta=0.7 |
| BNCI full-source | LDA |
| BNCI m8/class | longitudinal family, beta=0.7 |

## Current best final candidate

```text
Regime-anchored reliability-weighted interpolation
```

Rule:

```text
if source-validation selects LDA regime:
  use LDA
else:
  choose weighted family at regime level
  choose beta as the source-validation modal/risk-constrained beta
  use source-margin-calibrated interpolation
```

For the current regimes:

```text
Lee:
  source-weighted, beta=0.7

BNCI full:
  LDA

BNCI m8:
  longitudinal-weighted, beta=0.7
```

Result:

| Regime | Gain vs LDA | P(gain<-5) |
|---|---:|---:|
| Lee sensorimotor20 | +2.500 | 7.4% |
| BNCI full-source | +0.000 | 0.0% |
| BNCI m8 unit | +2.470 | 6.3% |

This is stronger than:

- E36 frozen guarded;
- E38 practical binary fallback;
- E41 local beta selector;
- beta=1 pure weighted Ridge.

## Critical caveat

The current evidence still has one weakness:

```text
regime-level beta=0.7 is inferred from the same regimes being evaluated.
```

E41 makes this much less post-hoc because the modal beta is selected by
source-validation excluding held subjects, but it is not the same as a fully
external fourth dataset.

Therefore the honest claim is:

```text
source-validation consistently selects beta around 0.7 in the positive regimes,
and applying this regime-level beta improves the risk-utility frontier.
```

not:

```text
beta=0.7 is universally validated.
```

## Next action

Do not build another local selector.

The next experiment should be E42:

```text
freeze the final rule:
  regime-level family
  + regime-level source-validation modal beta
```

and produce a clean final comparison table against:

- LDA full;
- beta=1 weighted Ridge;
- E36 frozen guarded;
- E38 binary fallback;
- E41 local selected;
- fixed beta=0.6 risk-first alternative.

If E42 is clean, this becomes the main method result.
