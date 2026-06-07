# 2026-06-02 Expert portfolio first results

## Purpose

Move from a heuristic EA selector to an adaptation portfolio foundation.

The first step is to collect saved logits from multiple adaptation experts into
a single table, then evaluate simple label-free portfolio baselines.

## Implemented

- `intentflow/offline/scripts/analysis/build_expert_prediction_table.py`
  - builds trial-level expert table from saved logits
  - writes per-expert summary, per-subject oracle, and dense NPZ arrays
- `intentflow/offline/scripts/analysis/eval_online_expert_portfolio.py`
  - evaluates label-free online multiplicative-weight expert portfolios
  - also reports static equal ensemble and oracle gap

Outputs:

- `docs/research_progress/260602_expert_portfolio_table/expert_summary.csv`
- `docs/research_progress/260602_expert_portfolio_table/expert_subject_oracle.csv`
- `docs/research_progress/260602_expert_portfolio_table/expert_trial_table.csv`
- `docs/research_progress/260602_expert_portfolio_table/expert_portfolio_arrays.npz`
- `docs/research_progress/260602_online_portfolio_eval_static_304030/`
- `docs/research_progress/260602_online_portfolio_eval/`

## Expert means on BCIC2a seed0

| Expert | Mean acc | Subjects |
|---|---:|---:|
| source | 82.72 | 9 |
| full EA | 83.06 | 9 |
| shrink 0.1 | 81.44 | 9 |
| full+shrink mean | 84.07 | 9 |
| source+full mean | 85.69 | 9 |
| source/full/shrink 0.3/0.4/0.3 | 85.76 | 9 |
| partial 0.5 + shrink 0.1 | 80.90 | 4 |
| shrink 0.03 | 77.78 | 4 |

## Key result

The strongest simple baseline is not the current online weight update. It is a
static convex portfolio:

```text
p = 0.3 p_source + 0.4 p_fullEA + 0.3 p_shrinkEA
```

Accuracy:

```text
source mean: 82.72
full EA mean: 83.06
adaptive selector v2: 85.19
source+full mean: 85.69
source/full/shrink 0.3/0.4/0.3: 85.76
oracle over current experts: 86.23
```

This is an important pivot. A simple adaptation portfolio already beats the
previous selector without per-subject rules.

## Online multiplicative update result

Default label-free online update:

```text
score = entropy + 0.5 disagreement + 0.1 class-collapse
alpha_k <- alpha_k exp(-eta score_k)
```

Mean accuracy was 84.07, worse than static equal/convex fusion. The current
score over-trusts confident but wrong experts and hurts S2/S6/S9.

## Interpretation

The algorithmic opportunity is now clearer:

1. Keep the static source/full/shrink portfolio as the new hard baseline.
2. Treat online routing as an optional refinement, not the base method.
3. A good router must beat 85.76 and approach the 86.23 oracle.
4. Reliability diagnostics should be used to decide when to deviate from the
   static portfolio, not to continuously reweight every trial.

## Next experiment

Build a conservative reliability-gated portfolio:

```text
base: p = 0.3 source + 0.4 fullEA + 0.3 shrinkEA

if fullEA looks unsafe:
    lower fullEA weight
if shrinkEA looks unsafe:
    lower shrinkEA weight
if experts strongly disagree:
    freeze weights and do not TTA-update
```

The immediate target is mean > 85.76 and harmed-subject count no worse than the
static portfolio.

