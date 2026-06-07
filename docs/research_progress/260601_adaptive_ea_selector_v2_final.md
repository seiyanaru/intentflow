# 2026-06-01 Adaptive EA selector v2 final check

## Setup

Candidates:

- source/no EA
- full EA
- shrink EA: `power=1.0`, `shrinkage=0.1`
- partial EA only for S2/S5/S7/S8: `power=0.5`, `shrinkage=0.1`
- v2 fusion: when full EA and shrink EA both pass reliability checks, average their probabilities.

Reliability checks use labels only after selection for evaluation. Selection uses:

- prediction prior KL
- predicted class dominance
- entropy/confidence shift from source
- covariance diagnostics: test condition number, channel variance CV, train-test covariance distance

Files:

- selector: `intentflow/offline/scripts/analysis/eval_adaptive_ea_selector.py`
- covariance diagnostics: `docs/research_progress/260601_ea_cov_diagnostics.json`
- v1 output: `docs/research_progress/260601_adaptive_ea_selector_v1_cov.json`
- v2 output: `docs/research_progress/260601_adaptive_ea_selector_v2_fusion.json`

## Candidate results

| subj | source | full EA | shrink .1 | selected v1 | selected v2 |
|---:|---:|---:|---:|---:|---:|
| S1 | 85.76 | 87.85 | 85.42 | 87.85 full | 86.81 fusion |
| S2 | 72.57 | 75.69 | 47.22 | 75.69 full | 75.69 full |
| S3 | 93.06 | 92.36 | 91.32 | 92.36 full | 92.01 fusion |
| S4 | 81.94 | 83.33 | 85.76 | 83.33 full | 85.76 fusion |
| S5 | 77.43 | 72.57 | 77.78 | 77.78 shrink | 77.78 shrink |
| S6 | 70.49 | 73.61 | 73.26 | 73.61 full | 76.39 fusion |
| S7 | 90.97 | 83.68 | 93.06 | 93.06 shrink | 93.06 shrink |
| S8 | 84.38 | 90.28 | 90.28 | 90.28 full | 90.28 fusion |
| S9 | 87.85 | 88.19 | 88.89 | 88.89 shrink | 88.89 shrink |

## Means

| method | mean acc | delta vs source | delta vs full EA |
|---|---:|---:|---:|
| source | 82.72 | - | -0.35 |
| full EA | 83.06 | +0.35 | - |
| selector v1 | 84.76 | +2.04 | +1.70 |
| selector v2 fusion | 85.19 | +2.47 | +2.12 |

## Interpretation

v2 is the current best seed0 result.

Important wins:

- S2 shrink collapse is rejected by prediction prior collapse and high test covariance condition number.
- S5 full EA collapse is rejected by prior KL/dominance; shrink recovers source-level accuracy.
- S7 full EA collapse is rejected by entropy-up/confidence-down; shrink reaches 93.06.
- S4/S6 benefit from full+shrink probability fusion.
- S9 is correctly sent to shrink rather than source once shrink candidate is available.

Important warnings:

- v2 fusion slightly hurts S1 and S3 compared with the best available candidate.
- This is still seed0 and thresholded after exploratory analysis; it needs multi-seed and leave-one-subject threshold validation.
- Current v2 requires multiple trained candidate models. It is accuracy-first, not yet the lightest online implementation.

## Next decision

Current best path:

1. Treat v2 as the leading Phase-1 accuracy result.
2. Validate the rule without label tuning:
   - freeze thresholds,
   - run seed 1/2 on all subjects,
   - optionally do leave-one-subject threshold fitting.
3. Then convert the selector into a named method:
   - Reliability-Aware EA Selector, or
   - TRUST-EA v2 with reliability-gated fusion.
