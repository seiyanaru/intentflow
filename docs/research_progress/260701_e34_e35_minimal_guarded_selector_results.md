# E34/E35: minimal stable and source-anchor guarded selector results

Date: 2026-07-01

## One-line verdict

E34/E35で、現時点の一番筋の良い方法はこれになった。

```text
Guarded stable source-validated reliability-weighted Ridge
```

具体的には:

```text
default: LDA full
candidate weighted methods:
  - ridge_source_rank_g2_a100
  - ridge_longitudinal_rank_g2_a100

switch from LDA full to weighted Ridge only if:
  inner source-validation gain over LDA full >= +0.5pp
  and inner P(gain < -5pp) <= 0.20
```

この `+0.5pp` margin guard が重要。

E34のminimal selectorだけだと、BNCI full-sourceで1/9 subjectsだけweighted Ridgeを誤選択して、LDA fullより落ちた。
E35のguardを入れると、その誤選択が消え、Lee/BNCI m8の利得は維持された。

## Artifacts

Scripts:

- `intentflow/offline/scripts/analysis/minimal_stable_selector_e34.py`
- `intentflow/offline/scripts/analysis/guarded_minimal_selector_e35.py`

Outputs:

- `intentflow/offline/results/research_outputs/260701_e34_minimal_stable_selector/`
- `intentflow/offline/results/research_outputs/260701_e35_guarded_minimal_selector/`

Main files:

- `e34_minimal_selector_combined_summary.csv`
- `e35_guarded_selector_combined_summary.csv`
- per-regime `guarded_summary.csv`
- per-regime `guarded_selection_records.csv`
- per-regime `guarded_outer_records.csv`

## Why E34 was needed

E33 showed that the full candidate set was too selector-like.

Weak candidates such as:

```text
lda_source_q25
lda_long_q10
lda_long_q70
ridge_source_rank_g1_a100
```

increased selection variance, especially in BNCI m8/class source-scarce repeats.

Therefore E34 reduced the candidate set to:

```text
lda_full
ridge_source_rank_g2_a100
ridge_longitudinal_rank_g2_a100
```

This is not just engineering cleanliness. It is a research claim:

```text
the useful axis is not arbitrary model selection;
it is whether to use stable source-validated reliability weighting at all,
and if so whether the reliability score should be source-only or longitudinal.
```

## E34 result: minimal stable selector

### Lee2019 sensorimotor20

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `ridge_source_rank_g2_a100` | **71.991** | **+2.384** | -6.250 | 9.3% |
| `ridge_longitudinal_rank_g2_a100` | 71.829 | +2.222 | -6.250 | 13.0% |
| `minimal_stable` | 71.644 | +2.037 | -6.250 | 11.1% |
| `lda_full` | 69.606 | 0.000 | 0.000 | 0.0% |

Selection:

```text
ridge_source_rank_g2_a100        49/54
ridge_longitudinal_rank_g2_a100   5/54
```

Interpretation:

```text
minimal selector is close to the best fixed weighted Ridge,
but loses 0.35pp by occasionally choosing longitudinal weighting.
```

### BNCI2014_001 full-source

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `lda_long_q70` | **78.047** | +0.448 | -1.774 | 0.0% |
| `lda_full` | 77.599 | 0.000 | 0.000 | 0.0% |
| `minimal_stable` | 77.061 | -0.538 | -2.903 | 0.0% |
| `ridge_source_rank_g2_a100` | 77.061 | -0.538 | -4.839 | 0.0% |
| `ridge_longitudinal_rank_g2_a100` | 76.971 | -0.627 | -4.194 | 0.0% |

Selection:

```text
lda_full                    8/9
ridge_source_rank_g2_a100   1/9
```

Interpretation:

```text
E34 mostly avoids weighted Ridge in full-source BNCI,
but one false switch is enough to lose 0.54pp.
```

This is exactly the kind of failure that argues for a source-anchor guard.

### BNCI2014_001 m8/class

| Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) |
|---|---:|---:|---:|---:|
| `ridge_longitudinal_rank_g2_a100` | **70.010** | **+2.327** | -1.361 | 0.0% |
| `minimal_stable` | 69.825 | +2.142 | -1.361 | 0.0% |
| `ridge_source_rank_g2_a100` | 69.738 | +2.055 | -1.013 | 0.0% |
| `lda_full` | 67.683 | 0.000 | 0.000 | 0.0% |

Selection:

```text
ridge_longitudinal_rank_g2_a100  8/9 subjects
ridge_source_rank_g2_a100        1/9 subjects
```

Interpretation:

```text
E34 works well in source-scarce BNCI.
It is only 0.19pp below fixed best and keeps +2.14pp over LDA full.
```

## E35 result: source-anchor margin guard

E35 tested margins:

```text
0.0, 0.25, 0.5, 1.0, 1.5 pp
```

Rule:

```text
choose weighted Ridge only if best weighted method beats LDA full
by at least margin on source-side validation.
otherwise choose LDA full.
```

The key result:

```text
margin >= 0.5pp removes the BNCI full-source false switch,
without changing Lee2019 or BNCI m8 selections.
```

### E35 stable guard with margin = 0.5pp

| Regime | Method | Acc | Gain vs LDA full | q05 gain | P(gain<-5pp) | Selection |
|---|---|---:|---:|---:|---:|---|
| Lee2019 sensorimotor20 | guarded stable m0.5 | 71.644 | +2.037 | -6.250 | 11.1% | source 49, longitudinal 5 |
| BNCI full-source | guarded stable m0.5 | 77.599 | 0.000 | 0.000 | 0.0% | LDA full 9 |
| BNCI m8/class | guarded stable m0.5 | 69.825 | +2.142 | -1.361 | 0.0% | longitudinal 8, source 1 |

This is the cleanest current method behavior:

- it gains on Lee sensorimotor20;
- it gains on BNCI source-scarce;
- it refuses weighted Ridge on BNCI full-source;
- it uses no target labels from the held subject.

## What to keep / revise / reject

### Keep

```text
Reliability-weighted Ridge is a real cross-session EEG-MI method candidate.
```

It is not just a Lee2019 artifact. It works in BNCI m8/class too.

### Revise

Do not sell this as:

```text
automatic fine-grained selector
```

Sell it as:

```text
source-anchored stable model-family selection
```

The source anchor is not a detail. It is what prevents the full-source BNCI failure.

### Reject

Do not keep expanding the candidate pool.

E33 already showed that more candidates made the selector noisier, not smarter.

## Current method definition to freeze

Name candidate:

```text
Guarded Stable Source-Validated Reliability-Weighted Ridge
```

Processing:

1. For a held target subject, do not use any target labels.
2. On labeled source subjects, run leave-one-source-subject validation.
3. Evaluate:
   - `lda_full`
   - `ridge_source_rank_g2_a100`
   - `ridge_longitudinal_rank_g2_a100`
4. Average inner validation over available source repeats.
5. Let `best_weighted` be the better of source-weighted and longitudinal-weighted Ridge.
6. If:

```text
inner mean(best_weighted - lda_full) >= +0.5pp
and inner P(best_weighted - lda_full < -5pp) <= 0.20
```

then choose `best_weighted`.

Otherwise choose `lda_full`.

## Scientific interpretation

The story is now sharper:

```text
Cross-session EEG-MI does not need another target-time adaptation trick here.
It needs a conservative source-side training rule that knows when feature
reliability weighting is worth the risk.
```

This differs from the earlier EA/selector work:

- no target labels from the held subject;
- no target-session fitting;
- no broad post-hoc model zoo;
- source anchor is explicit;
- weighted Ridge changes the training-side inductive bias, not merely the decision rule.

## Hard limitation

This is still not final-proof.

E34/E35 reuse E33 computed records, and the `+0.5pp` margin was chosen after seeing the E34 failure.
So the next step must freeze this rule and rerun it as a single official protocol.

## Next action: E36

E36 should be a frozen-method audit.

### E36 frozen rule

```text
candidate set:
  lda_full
  ridge_source_rank_g2_a100
  ridge_longitudinal_rank_g2_a100

guard:
  switch to weighted Ridge only if inner gain >= +0.5pp
  and inner P(gain < -5pp) <= 0.20

selection:
  stable source-validation, one family per outer subject/regime
```

### E36 datasets/regimes

Minimum:

1. Lee2019 sensorimotor20
2. BNCI2014_001 full-source
3. BNCI2014_001 m8/class

If compute allows:

4. Lee2019 all62

### E36 pass criteria

The frozen method should:

- beat LDA full by >= +1.5pp on Lee sensorimotor20;
- beat LDA full by >= +1.5pp on BNCI m8/class;
- match LDA full within -0.2pp on BNCI full-source;
- keep `P(gain vs LDA full < -5pp)` below 12% on Lee and 0-5% on BNCI.

If E36 passes, this becomes the thesis method core.

