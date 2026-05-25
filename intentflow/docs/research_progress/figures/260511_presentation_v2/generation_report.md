# generation_report (260511 v2)

started: 2026-05-08T16:06:43.421406
finished: 2026-05-08T16:06:43.535056

## 1. Probed result directories
- c_aug_true: /mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/c_aug_true_9subj_20260506_004923 (MISSING)
- b_5seed:    /mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/b_5seed_4subj_20260506_005153 (MISSING)
- trace npz:  MISSING

## 2. seed-0 fallback paths (S7)
- source_only: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/source_only/results.txt)
- policy_safe_no_shallow: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/policy_safe_no_shallow/results.txt)
- replay_safe_uniform: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/replay_safe_9subject_20260505_212942/eval/s7/replay_safe_default/results.txt)
- replay_h6_weighted: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/replay_h6_9subject_20260506_004038/eval/s7/replay_h6_weighted/results.txt)

## 3. Generated artefacts
### PNG

### CSV
- all_results_summary.csv

## 4. Skipped figures
- fig_main_source_vs_replay: c_aug_true summary.json missing
- fig_per_subject_delta: c_aug_true summary.json missing
- fig_replay_candidate_trace: no trace npz found
- fig_safety_tradeoff: c_aug_true summary.json missing
- fig_s7_seed_stability: b_5seed summary.json missing

## 5. Missing-data notes
- trace npz missing — skipped trace figure

## 6. Manual-input audit
- All numeric values in figures and CSVs come from result files in this repo.
- The single value entered as a documented previous-seminar reference is hybrid@0.01 = 81.98% in fig_safety_tradeoff.csv with is_reference=1 and mean_delta_vs_source='missing'. This represents a previously published (260420 seminar) measurement, not a fresh re-evaluation.

## 7. Reproduce
```bash
cd /mnt/data/seiya.narukawa/intentflow
python intentflow/offline/scripts/analysis/make_260511_presentation_figures.py
```

## 8. Live log
```
# make_260511_presentation_figures.py (started 2026-05-08T16:06:43.421406)
REPO: /mnt/data/seiya.narukawa/intentflow/intentflow
OUT_FIG: /mnt/data/seiya.narukawa/intentflow/intentflow/docs/research_progress/figures/260511_presentation_v2
OUT_TBL: /mnt/data/seiya.narukawa/intentflow/intentflow/docs/research_progress/tables/260511_presentation_v2

## 1. Data discovery
  c_aug_true_9subj: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/c_aug_true_9subj_20260506_004923)
  b_5seed_4subj: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/b_5seed_4subj_20260506_005153)
  seed-0 source_only: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/source_only/results.txt)
  seed-0 policy_safe_no_shallow: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/policy_safe_no_shallow/results.txt)
  seed-0 replay_safe_uniform: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/replay_safe_9subject_20260505_212942/eval/s7/replay_safe_default/results.txt)
  seed-0 replay_h6_weighted: MISSING (/mnt/data/seiya.narukawa/intentflow/intentflow/intentflow/offline/results/replay_h6_9subject_20260506_004038/eval/s7/replay_h6_weighted/results.txt)
  trace npz: MISSING (None)

## 2. fig_main_source_vs_replay
  [SKIP] c_aug_true summary.json missing

## 3. fig_per_subject_delta
  [SKIP] c_aug_true summary.json missing

## 4. fig_replay_candidate_trace
  [SKIP] no trace npz found

## 5. fig_safety_tradeoff

## 6. fig_s7_seed_stability

## 7. all_results_summary.csv
  wrote CSV: all_results_summary.csv

## 8. caption_draft.md
  wrote: caption_draft.md
```
