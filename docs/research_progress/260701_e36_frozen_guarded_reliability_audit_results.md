# E36: frozen guarded reliability-weighted Ridge audit

Date: 2026-07-01

## One-line verdict

E36の固定プロトコルは、事前に決めた3つの主要pass基準をすべて満たした。

```text
Lee2019 sensorimotor20: pass
BNCI2014_001 full-source: pass
BNCI2014_001 m8/class repeat: pass
```

ただし、これは「安全適応が完成した」という意味ではない。
Leeではまだ `P(gain vs LDA full < -5pp) = 11.1%` が残る。

正しい主張は:

```text
source-anchor guard付きのreliability-weighted Ridgeは、
target labelsなしで平均精度を上げ、
不要なregimeではLDA fullに留まれる。
```

であって、

```text
no-harm adaptation
```

ではない。

## Frozen rule

E36では、E35で決めたルールを一切動かしていない。

```text
baseline:
  lda_full

weighted candidates:
  ridge_source_rank_g2_a100
  ridge_longitudinal_rank_g2_a100

switch condition:
  inner mean(best_weighted - lda_full) >= +0.5pp
  and inner P(best_weighted - lda_full < -5pp) <= 0.20

otherwise:
  lda_full
```

Two audit modes:

```text
repeat:
  choose candidate separately for each source-scarce repeat.

stable:
  average inner source-validation over repeats, then choose one family per outer subject.
```

For BNCI m8/class, the repeat mode is the more conservative / practical lower-bound.
The stable mode estimates the cleaner regime-level family choice.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/frozen_guarded_reliability_audit_e36.py`

Outputs:

`intentflow/offline/results/research_outputs/260701_e36_frozen_guarded_reliability_audit/`

Main files:

- `e36_frozen_combined_summary.csv`
- `e36_frozen_combined_outer_records.csv`
- `e36_frozen_combined_selection_records.csv`
- `e36_pass_fail.csv`
- per-regime `e36_frozen_summary.csv`
- per-regime `e36_frozen_selection_records.csv`
- per-regime `e36_frozen_outer_records.csv`

## Pass/fail summary

| Regime | Mode | Gain vs LDA full | P(gain<-5pp) | Criterion | Pass |
|---|---|---:|---:|---|---|
| Lee2019 sensorimotor20 | stable | +2.037 | 11.1% | gain >= +1.5pp and P<12% | yes |
| BNCI full-source | stable | 0.000 | 0.0% | gain >= -0.2pp and P<5% | yes |
| BNCI m8/class | repeat | +1.761 | 0.0% | gain >= +1.5pp and P<5% | yes |

The important conservative result is BNCI m8/class repeat:

```text
e36_frozen_guarded_repeat:
  acc = 69.444
  gain vs LDA full = +1.761pp
  95% CI [0.230, 3.478]
  q05 gain = -1.235
  P(gain<-5pp) = 0.0%
```

So the method is not only winning under the stable/averaged setting.
It still wins in the more realistic noisy repeat setting.

## Main results

### Lee2019 sensorimotor20

| Method | Acc | Gain vs LDA full | 95% CI gain | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|---:|
| `ridge_source_rank_g2_a100` | **71.991** | +2.384 | [0.764, 3.958] | -6.250 | 9.3% |
| `ridge_longitudinal_rank_g2_a100` | 71.829 | +2.222 | [0.648, 3.773] | -6.250 | 13.0% |
| `e36_frozen_guarded_stable` | 71.644 | +2.037 | [0.463, 3.611] | -6.250 | 11.1% |
| `lda_long_q70` | 70.579 | +0.972 | [-0.347, 2.199] | -5.438 | 7.4% |
| `lda_full` | 69.606 | 0.000 | [0.000, 0.000] | 0.000 | 0.0% |

Selection:

```text
ridge_source_rank_g2_a100        49/54
ridge_longitudinal_rank_g2_a100   5/54
```

Interpretation:

```text
The frozen guard preserves a strong mean gain over LDA full.
However, it does not eliminate lower-tail harm.
```

This is a method win for mean accuracy, not a no-harm guarantee.

### BNCI2014_001 full-source

| Method | Acc | Gain vs LDA full | 95% CI gain | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|---:|
| `lda_long_q70` | **78.047** | +0.448 | [-0.717, 1.792] | -1.774 | 0.0% |
| `e36_frozen_guarded_stable` | 77.599 | 0.000 | [0.000, 0.000] | 0.000 | 0.0% |
| `lda_full` | 77.599 | 0.000 | [0.000, 0.000] | 0.000 | 0.0% |
| `ridge_source_rank_g2_a100` | 77.061 | -0.538 | [-2.599, 1.613] | -4.839 | 0.0% |
| `ridge_longitudinal_rank_g2_a100` | 76.971 | -0.627 | [-2.599, 1.613] | -4.194 | 0.0% |

Selection:

```text
lda_full  9/9
```

Interpretation:

```text
The source-anchor guard does exactly what we wanted:
it refuses weighted Ridge in the full-source regime.
```

This is important because E34 without the margin guard had one false switch.
E36 removes that false switch.

### BNCI2014_001 m8/class

#### Repeat mode: conservative primary result

| Method | Acc | Gain vs LDA full | 95% CI gain | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|---:|
| `ridge_longitudinal_rank_g2_a100` | **70.010** | +2.327 | [0.378, 4.492] | -1.361 | 0.0% |
| `ridge_source_rank_g2_a100` | 69.738 | +2.055 | [0.395, 4.127] | -1.013 | 0.0% |
| `e36_frozen_guarded_repeat` | 69.444 | +1.761 | [0.230, 3.478] | -1.235 | 0.0% |
| `lda_long_q70` | 68.271 | +0.588 | [-0.092, 1.204] | -1.058 | 0.0% |
| `lda_full` | 67.683 | 0.000 | [0.000, 0.000] | 0.000 | 0.0% |

Selection over 288 subject/repeats:

```text
ridge_longitudinal_rank_g2_a100  155
ridge_source_rank_g2_a100         81
lda_full                          52
```

Interpretation:

```text
Even when candidate selection is repeated under noisy source-scarce samples,
the frozen guard still beats LDA full by +1.76pp.
```

This is the most important BNCI result to emphasize.

#### Stable mode: regime-level upper estimate

| Method | Acc | Gain vs LDA full | 95% CI gain | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|---:|
| `ridge_longitudinal_rank_g2_a100` | **70.010** | +2.327 | [0.378, 4.492] | -1.361 | 0.0% |
| `e36_frozen_guarded_stable` | 69.825 | +2.142 | [0.291, 4.172] | -1.361 | 0.0% |
| `ridge_source_rank_g2_a100` | 69.738 | +2.055 | [0.395, 4.127] | -1.013 | 0.0% |
| `lda_full` | 67.683 | 0.000 | [0.000, 0.000] | 0.000 | 0.0% |

Selection:

```text
ridge_longitudinal_rank_g2_a100  8/9 subjects
ridge_source_rank_g2_a100        1/9 subjects
```

Interpretation:

```text
The source-validation signal correctly identifies that BNCI m8/class is mostly
a longitudinal-weighting regime.
```

But this mode averages over repeats, so it should be presented as a cleaner
regime-level estimate, not the primary practical result.

## What E36 strengthens

### 1. The method is no longer just post-hoc picking

The final rule is fixed:

```text
margin = +0.5pp
risk limit = 0.20
candidate set = 3 methods
```

E36 applies that rule directly.

### 2. Source anchor is essential

BNCI full-source is the key negative-control regime.
Weighted Ridge alone loses there:

```text
ridge_source_rank_g2:       -0.538pp vs LDA full
ridge_longitudinal_rank_g2: -0.627pp vs LDA full
```

The frozen guard selects:

```text
lda_full 9/9
```

This prevents the false adaptation.

### 3. The method still works under conservative source-scarce evaluation

BNCI m8 repeat mode still gives:

```text
+1.761pp vs LDA full
```

This matters because stable mode alone could be criticized as too optimistic.

## What E36 does not prove

### 1. It does not prove no-harm safety

Lee lower-tail risk remains:

```text
P(gain<-5pp) = 11.1%
q05 gain = -6.25pp
```

So do not use language like:

```text
safe adaptation
```

Use:

```text
guarded
source-anchored
risk-aware evaluation
```

### 2. It is not an independent external confirmation of the +0.5pp margin

The margin was chosen after E34/E35 analysis on these regimes.
E36 freezes the rule and audits it, but it is still on the same datasets/regimes.

The right phrasing is:

```text
frozen audit after method selection
```

not:

```text
fully independent validation
```

### 3. Lee all62 is not included in E36

Lee all62 has E31 evidence for source-weighted Ridge, but E31 records do not
contain the exact E36 candidate set and `lda_full` inner validation needed for
the guard.

Therefore:

```text
Lee all62 should be cited as supportive E31 evidence,
not as an E36 frozen-guard result.
```

If we need all62 under E36, we must run a separate cache-based E36-all62 job.
That is computationally heavier and should be treated as E37.

## Current thesis-level claim

The strongest honest claim after E36 is:

```text
For zero-target-label cross-session EEG-MI, a source-anchored guard can decide
when reliability-weighted Ridge is worth using.

The frozen guarded method improves mean accuracy on Lee2019 sensorimotor20 and
BNCI2014_001 source-scarce evaluation, while reverting to LDA full in the
BNCI full-source negative-control regime.
```

Do not claim:

```text
universal adaptation
no-harm adaptation
deep TTA replacement
```

## Next experiment: E37

E37 should not add a new method.
It should explain why E36 works.

### E37-A: mechanism / ablation

Run or aggregate:

```text
full LDA
full Ridge
hard top-k LDA
hard top-k Ridge
soft reliability-weighted Ridge
frozen guarded method
```

Goal:

```text
show that the gain is from soft reliability weighting + source anchor,
not merely from dimensionality reduction or arbitrary model selection.
```

### E37-B: score-family diagnosis

For each regime, report inner validation signals:

```text
source_rank_g2 gain over LDA full
longitudinal_rank_g2 gain over LDA full
which score family was selected
outer gain
```

Goal:

```text
explain why Lee selects source weighting, BNCI m8 selects longitudinal weighting,
and BNCI full-source selects LDA full.
```

### E37-C: optional Lee all62 frozen guard

Only if compute budget allows:

```text
run E36 candidate set on Lee all62 cache:
  lda_full
  ridge_source_rank_g2_a100
  ridge_longitudinal_rank_g2_a100
  frozen guard
```

This would connect E31 all-channel result to the final E36 method.
But it is optional, because the current core claim already has:

- Lee sensorimotor20;
- BNCI full-source negative control;
- BNCI source-scarce positive control.

