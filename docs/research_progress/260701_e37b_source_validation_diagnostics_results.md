# E37-B: source-validation diagnostic results

Date: 2026-07-01

## One-line verdict

E37-Bは、E36の解釈をかなり重要に修正した。

```text
source validationは、個々のsubject/repeatが得するかを予測できていない。
しかし、regime-levelでweighted Ridgeを使うべきかどうかを決める信号としては効いている。
```

つまり、E36を

```text
personalized selector
```

として売ってはいけない。

正しい売り方は:

```text
source-validated regime-level / cohort-level guard
```

である。

## Artifacts

Script:

`intentflow/offline/scripts/analysis/e37b_source_validation_diagnostics.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_e37b_source_validation_diagnostics/`

Main files:

- `e37b_diagnostic_records.csv`
- `e37b_correlation_summary.csv`
- `e37b_selection_summary.csv`
- `e37b_failure_cases.csv`
- `summary.json`

## Protocol

For each outer subject/repeat:

1. compute inner source-validation gain:
   - `ridge_source_rank_g2_a100 - lda_full`
   - `ridge_longitudinal_rank_g2_a100 - lda_full`
2. compute outer held-subject gain for the same two weighted methods;
3. recompute E36 frozen guard:

```text
choose best weighted candidate only if:
  inner best weighted gain >= +0.5pp
  and inner P(gain < -5pp) <= 0.20
otherwise choose lda_full
```

4. evaluate:
   - inner/outer gain correlation;
   - inner best weighted family vs outer best weighted family;
   - selected method vs outer best overall among `{lda_full, source-weighted, longitudinal-weighted}`;
   - harmful selected cases;
   - missed weighted opportunities.

## Main result 1: source validation does not predict subject-level outer gains

Correlations between inner validation gain and outer held-subject gain:

| Regime | Mode | Signal | Spearman |
|---|---|---|---:|
| Lee sensorimotor20 | stable/repeat | source gain | -0.544 |
| Lee sensorimotor20 | stable/repeat | longitudinal gain | -0.612 |
| BNCI full-source | stable/repeat | best weighted gain | -0.750 |
| BNCI m8/class | repeat | best weighted gain | -0.012 |
| BNCI m8/class | stable | best weighted gain | -0.900 |

This looks shocking at first, but it makes sense.

For leave-one-subject source validation, the inner validation mean for held subject `H` is computed on:

```text
all subjects except H
```

So if `H` is a high-gain subject, removing it can lower the inner mean.
That creates a complement effect and can even induce negative correlation.

Interpretation:

```text
leave-one-source-subject validation is not an individual-benefit predictor.
```

Therefore, do not claim:

```text
we can identify which subject will benefit.
```

## Main result 2: E36 works as a regime-level guard, not a per-subject oracle

### Lee2019 sensorimotor20

E36 selected:

```text
ridge_source_rank_g2_a100        49/54
ridge_longitudinal_rank_g2_a100   5/54
lda_full                          0/54
```

Performance:

```text
mean selected gain = +2.037pp
95% CI [0.440, 3.634]
q05 = -6.250
P(gain<-5pp) = 11.1%
P(gain<0) = 35.2%
```

Outer best overall among `{lda_full, source, longitudinal}`:

```text
lda_full                         16/54
ridge_source_rank_g2_a100        24/54
ridge_longitudinal_rank_g2_a100  14/54
```

Selected method matched outer best overall only:

```text
37.0%
```

Interpretation:

```text
The guard improves mean accuracy on Lee,
but it does not identify the harmed subjects.
```

This is exactly why we must not call it no-harm or personalized safe adaptation.

### BNCI2014_001 full-source

E36 selected:

```text
lda_full 9/9
```

Performance:

```text
mean selected gain = 0.000pp
P(gain<-5pp) = 0.0%
```

Outer best overall:

```text
lda_full                          6/9
ridge_source_rank_g2_a100         2/9
ridge_longitudinal_rank_g2_a100   1/9
```

Missed large weighted opportunity:

```text
3/9
```

Interpretation:

```text
The guard is conservative. It gives up some possible subject-level wins
to avoid using weighted Ridge in a regime where weighted Ridge is not
reliably beneficial.
```

This is acceptable for the current method story, because BNCI full-source is a negative-control regime.

### BNCI2014_001 m8/class

#### Repeat-level, conservative practical view

E36 selected over 288 subject/repeats:

```text
ridge_longitudinal_rank_g2_a100  155
ridge_source_rank_g2_a100         81
lda_full                          52
```

Performance:

```text
mean selected gain = +1.761pp
95% CI [1.162, 2.391]
q05 = -6.169
P(gain<-5pp) = 6.6%
P(gain<0) = 25.7%
```

This is stricter than the E36 pass/fail table.
E36 reported subject-averaged repeat risk, where `P(subject-mean gain<-5pp)=0`.
E37-B shows that individual source-scarce repeats can still be harmed.

Interpretation:

```text
BNCI m8 repeat still supports the method on mean accuracy,
but the per-repeat harm rate is not zero.
```

#### Stable regime-level view

E36 stable selected:

```text
ridge_longitudinal_rank_g2_a100  8/9
ridge_source_rank_g2_a100        1/9
```

Performance:

```text
mean selected gain = +2.142pp
P(gain<-5pp) = 0.0%
P(gain<0) = 11.1%
```

Interpretation:

```text
At the regime/family level, BNCI m8 is clearly a longitudinal-weighting regime.
```

## Main result 3: family selection is only moderately aligned with outer best

Match rates:

| Regime | Mode | Inner best weighted = outer best weighted | Selected = outer best overall |
|---|---|---:|---:|
| Lee sensorimotor20 | stable/repeat | 51.9% | 37.0% |
| BNCI full-source | stable/repeat | 11.1% | 66.7% |
| BNCI m8/class | repeat | 47.9% | 34.7% |
| BNCI m8/class | stable | 44.4% | 44.4% |

These are not strong enough for a personalized selector claim.

They are strong enough for a more modest claim:

```text
source validation can choose a useful cohort-level bias,
not the best action for every subject/session.
```

## Failure cases

Worst selected harms:

| Regime | Subject/repeat | Selected | Outer selected gain |
|---|---|---|---:|
| BNCI m8 | S3/r25 | longitudinal | -15.323pp |
| Lee | S5 | source | -13.750pp |
| BNCI m8 | S5/r31 | longitudinal | -12.903pp |
| BNCI m8 | S3/r14 | longitudinal | -11.290pp |
| BNCI m8 | S6/r15 | longitudinal | -10.484pp |

These are severe enough that we must keep harm metrics visible.

Again:

```text
guarded mean-accuracy method: yes
no-harm adaptation: no
```

## What E37-B changes

### Keep

```text
E36 frozen guarded method improves mean accuracy in positive regimes
and refuses weighted Ridge in the full-source negative-control regime.
```

### Revise

Change the explanation from:

```text
source validation predicts which subject benefits
```

to:

```text
source validation estimates whether a regime/cohort supports a reliability-weighted bias.
```

### Reject

Do not frame the method as:

```text
personalized safe selector
subject-level benefit predictor
no-harm gate
```

The data do not support that.

## Updated research claim

After E37-B, the strongest honest claim is:

```text
A source-anchored guard can decide whether reliability-weighted Ridge is
worth using at the regime/cohort level. This improves mean cross-session
EEG-MI accuracy without target labels, but does not reliably predict
individual harmed sessions.
```

This is still a good result.
It is just narrower and more honest than the tempting personalized-safety story.

## Next experiment: E38

E38 should simplify the method to match what E37-B actually supports.

Because subject-level family selection is weak, try a more stable regime-level
family rule:

```text
For each outer fold/regime:
  compute source-validation aggregate over all available source subjects/repeats.
  choose one of:
    lda_full
    ridge_source_rank_g2_a100
    ridge_longitudinal_rank_g2_a100
  apply that same family to all held subjects/repeats in the regime.
```

Two variants:

### E38-A: global inner-aggregate family

Use:

```text
best aggregate weighted gain >= +0.5pp
and p_harm <= 0.20
```

Expected choices:

```text
Lee sensorimotor20: source-weighted
BNCI full-source:   lda_full
BNCI m8/class:      longitudinal-weighted
```

This may improve over E36 by avoiding noisy subject-level source-vs-long switches.

### E38-B: family-stability guard

Switch to a weighted family only if:

```text
one weighted family wins by at least delta over the other
and weighted-vs-LDA passes the E36 margin/risk rule.
```

If source vs longitudinal are too close, use the cohort-global winner or LDA full.

Goal:

```text
reduce Lee's 5 bad longitudinal choices
and reduce BNCI m8 repeat's noisy family switching.
```

E38 is the logical next step because it aligns the method with the evidence:

```text
regime-level validation is supported;
subject-level personalized prediction is not.
```

