# E23: Stieger broad-all60 category-constrained control results

Date: 2026-06-30

## One-line verdict

E23 rejects the fixed motor/posterior category rule as a general method.

```text
Lee2019: category constraint looked mildly promising.
Stieger: category constraint clearly hurts.

Therefore:
do not claim a validated neuro-category selection method.
```

This is not a small ambiguous failure.  On Stieger pure LR, removing the
selected `other_other` dimensions hurts both fixed policies:

- Lee-like `source_only_q0p25`;
- Stieger-like `longitudinal_q0p10`.

The stronger conclusion is:

> Hand-defined motor/posterior masks are too crude.  Source-side selection can
> choose noncanonical but session-stable dimensions that are useful, and
> deleting them by anatomy can destroy performance.

## Artifacts

Script:

`intentflow/offline/scripts/analysis/stieger_category_control_e23.py`

Output:

`intentflow/offline/results/research_outputs/260630_stieger_category_control_e23/`

Main files:

- `summary.csv`
- `summary.json`
- `paired_contrasts.csv`
- `selection_records.csv`
- `summary_records_with_random_mean.csv`
- `selection_size_by_subject.csv`
- `category_counts.csv`
- `subject_level_deltas.csv`
- `random_repeat_summary.csv`

## Experimental design

Dataset/protocol:

- Dataset: Stieger2021
- Condition: `pure_lr`
- Feature: `broad_all60`
- Subjects: 62
- Target sessions evaluated: 523 total subject-sessions
- Baseline: full broad all60 tangent, session-1 source decoder, target prefix-EA
- Prefix/eval: prefix 32, evaluate from trial 65 onward

Fixed policies:

1. `source_only_q0p25`
   - Lee2019 best compact source-side policy.
2. `longitudinal_q0p10`
   - Stieger compact longitudinal policy.

For each policy:

- `all_selected`
- `category_keep_motor_or_posterior`
- `score_top_same_k`
- `random_keep_k_mean`

Category definition:

- Sensorimotor: Stieger `SENSORIMOTOR21`, including `FCz`
- Posterior: P/PO/O channels available in Stieger all60
- Category rule: keep selected tangent dimensions if at least one endpoint is
  sensorimotor or posterior; drop selected `other_other`.

## Main performance

### Source-only q0.25 family

| Method | n selected | Acc | Gain vs full | 95% CI | q05 gain | P(gain < -5) |
|---|---:|---:|---:|---:|---:|---:|
| full_q1p00 | 1830 | 63.863 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 |
| source_only_q0p25 all | 458.0 | 66.008 | +2.144 | [1.311, 2.980] | -3.080 | 0.032 |
| source_only_q0p25 score-top-k | 433.7 | 65.952 | +2.088 | [1.222, 2.969] | -2.581 | 0.032 |
| source_only_q0p25 random-k | 433.7 | 65.777 | +1.914 | [1.082, 2.730] | -3.987 | 0.032 |
| source_only_q0p25 category | 433.7 | 64.691 | +0.828 | [-0.209, 1.870] | -7.804 | 0.097 |

Paired contrasts:

| Contrast | Mean delta | 95% CI | Pos / Neg | Verdict |
|---|---:|---:|---:|---|
| category - all | -1.316 | [-1.954, -0.731] | 17 / 39 | clear loss |
| category - score-top-k | -1.260 | [-1.973, -0.625] | 17 / 42 | clear loss |
| category - random-k | -1.086 | [-1.729, -0.496] | 22 / 38 | clear loss |

### Longitudinal q0.10 family

| Method | n selected | Acc | Gain vs full | 95% CI | q05 gain | P(gain < -5) |
|---|---:|---:|---:|---:|---:|---:|
| full_q1p00 | 1830 | 63.863 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 |
| longitudinal_q0p10 all | 183.0 | 68.033 | +4.169 | [3.073, 5.314] | -2.370 | 0.016 |
| longitudinal_q0p10 score-top-k | 176.0 | 67.937 | +4.074 | [3.004, 5.252] | -2.545 | 0.016 |
| longitudinal_q0p10 random-k | 176.0 | 67.843 | +3.980 | [2.896, 5.177] | -1.950 | 0.016 |
| longitudinal_q0p10 category | 176.0 | 66.308 | +2.444 | [1.087, 3.895] | -4.766 | 0.048 |

Paired contrasts:

| Contrast | Mean delta | 95% CI | Pos / Neg | Verdict |
|---|---:|---:|---:|---|
| category - all | -1.725 | [-2.544, -0.984] | 18 / 41 | clear loss |
| category - score-top-k | -1.629 | [-2.479, -0.871] | 18 / 40 | clear loss |
| category - random-k | -1.535 | [-2.309, -0.827] | 21 / 41 | clear loss |

## Selection size and composition

The category rule removes only a small number of selected dimensions on
Stieger:

| Policy | all selected | category kept | dropped `other_other` | kept fraction |
|---|---:|---:|---:|---:|
| source_only_q0p25 | 458.0 | 433.7 | 24.1 | 94.7% |
| longitudinal_q0p10 | 183.0 | 176.0 | 7.0 | 96.2% |

This makes the failure more informative:

> The rule drops only a few dimensions, but those dimensions are useful.

Mean selected category composition:

| Policy | both sensorimotor | sensorimotor-posterior | sensorimotor-other | posterior-other | other-other |
|---|---:|---:|---:|---:|---:|
| source_only_q0p25 | 121.6 | 110.2 | 102.4 | 99.7 | 24.1 |
| longitudinal_q0p10 | 76.3 | 33.2 | 64.6 | 1.9 | 7.0 |

The selected subspace is indeed mostly sensorimotor-centered.  But the small
non-motor/posterior residue should not be deleted.

## What was dropped?

The dropped `other_other` dimensions were not random garbage.  They were stable
frontal/frontopolar covariance pairs.

### Source-only q0.25 dropped pairs

Top dropped pairs appeared in almost every held-subject split:

```text
AF3-F7: 62
F5-Fp1: 62
F5-Fpz: 62
F6-Fp2: 62
F6-Fpz: 62
F7-F8: 62
F7-FT8: 62
F7-Fp1: 62
F7-Fp2: 62
F7-Fpz: 62
F8-Fp1: 62
F8-Fp2: 62
F8-Fpz: 62
FT8-Fp1: 62
FT8-Fpz: 62
```

Top channels:

```text
F7, Fp1, F8, Fpz, FT8, Fp2, F5, AF3, F6
```

### Longitudinal q0.10 dropped pairs

```text
F7-F8: 62
F7-Fp1: 62
F7-Fp2: 62
F7-Fpz: 62
F8-Fp1: 62
F8-Fp2: 62
F8-Fpz: 62
```

Top channels:

```text
F8, F7, Fp1, Fp2, Fpz
```

So the rule specifically removes a stable frontal/frontopolar block.  In this
Stieger LR protocol, that block improves prediction.

## Interpretation

E21/E22 suggested:

> Lee2019 selected `other_other` dimensions may be noise.

E23 says:

> That is not general.  In Stieger, selected `other_other` frontal/frontopolar
> pairs are useful.  Hand-deleting them hurts more than random same-size
> pruning.

This changes the mechanistic story.

The robust story is no longer:

```text
source-side selection works because it removes non-motor/non-posterior pairs
```

The robust story is closer to:

```text
source-side selection discovers a compact, task-specific, session-stable
covariance subspace.  It is often sensorimotor-centered, but useful dimensions
can be noncanonical and dataset-specific.
```

That is less pretty, but more true.

## Relation to previous evidence

E20:

- Lee all62 q0.25 works;
- Lee sensorimotor20 q0.25 loses the gain;
- therefore all-channel context matters.

E21:

- Lee q0.25 selected subspace has motor/posterior structure;
- dropping `both_sensorimotor` or `any_posterior` hurts;
- dropping Lee `other_other` looked mildly beneficial.

E22:

- Lee category rule beats same-size random;
- but does not clearly beat q0.25 all-selected or score-top-k.

E23:

- Stieger category rule loses to all-selected, score-top-k, and random;
- therefore the category rule is not externally valid.

The correct update is:

```text
Keep: selected subspaces are neurostructured and compact.
Retract: motor/posterior category filtering is a validated method.
Revise: noncanonical selected dimensions may be useful and should be audited,
        not automatically removed.
```

## What not to claim

Do not claim:

- "motor/posterior category constraint is a new method";
- "selected other_other dimensions are generally noise";
- "posterior/motor hand mask improves safety";
- "anatomical pruning is robust across datasets".

The Stieger result directly contradicts those claims.

## What can still be claimed

Safe claim:

> Across datasets, source-side compact subspace selection improves
> cross-session EEG-MI, but the optimal compact subspace is task/dataset
> dependent.  It is usually sensorimotor-centered, yet hand-defined anatomical
> masks can remove useful noncanonical predictors.

This is a more defensible research contribution than a fragile anatomy mask.

## Next decision

Stop developing the fixed category-constrained method.

The strongest next direction is:

```text
source-side compactness / longitudinal stability as the method,
channel-pair anatomy as interpretation and failure analysis,
not as a hard selection rule.
```

Recommended next experiment E24:

Run a mechanism audit of the noncanonical Stieger frontal block:

1. Compare four policies on Stieger LR:
   - `longitudinal_q0p10` all selected;
   - remove only the stable frontal/frontopolar block;
   - keep only the selected sensorimotor-centered block;
   - random same-size removal.
2. Ask whether the frontal block is:
   - a robust auxiliary predictor;
   - an artifact/task-context cue;
   - or a statistical quirk of the Stieger protocol.
3. If it is consistently useful, frame it honestly:
   - "noncanonical but stable covariance context";
   - not "pure motor physiology".

