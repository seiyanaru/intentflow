# generation_report (260511 v2)

started: 2026-05-08T16:30:18.033017
finished: 2026-05-08T16:30:21.873510

## 1. Probed result directories
- c_aug_true: /mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/c_aug_true_9subj_20260506_004923 (FOUND)
- b_5seed:    /mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/b_5seed_4subj_20260506_005153 (FOUND)
- trace npz:  /mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/_smoke_replay_v3_s2_212138/replay_safe_otta_stats_s2_tcformer_replay_safe_otta.npz

## 2. seed-0 fallback paths (S7)
- source_only: FOUND (intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/source_only/results.txt)
- policy_safe_no_shallow: FOUND (intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/policy_safe_no_shallow/results.txt)
- replay_safe_uniform: FOUND (intentflow/offline/results/replay_safe_9subject_20260505_212942/eval/s7/replay_safe_default/results.txt)
- replay_h6_weighted: FOUND (intentflow/offline/results/replay_h6_9subject_20260506_004038/eval/s7/replay_h6_weighted/results.txt)

## 3. Generated artefacts
### PNG
- fig_main_source_vs_replay.png
- fig_per_subject_delta.png
- fig_replay_candidate_trace.png
- fig_safety_tradeoff.png
- fig_s7_seed_stability.png

### CSV
- fig_main_source_vs_replay.csv
- fig_per_subject_delta.csv
- fig_replay_candidate_trace.csv
- fig_safety_tradeoff.csv
- fig_s7_seed_stability.csv
- all_results_summary.csv

## 4. Skipped figures
- (none)

## 5. Missing-data notes
- hybrid@0.01_reference: included only for visual reference. It was NOT re-run in the same regime as c_aug_true, so its mean Δ vs the current source_only is unfair and is left as 'missing'.

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
# make_260511_presentation_figures.py (started 2026-05-08T16:30:18.033017)
REPO: /mnt/data/seiya.narukawa/intentflow
OUT_FIG: /mnt/data/seiya.narukawa/intentflow/docs/research_progress/figures/260511_presentation_v2
OUT_TBL: /mnt/data/seiya.narukawa/intentflow/docs/research_progress/tables/260511_presentation_v2

## 1. Data discovery
  c_aug_true_9subj: FOUND (/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/c_aug_true_9subj_20260506_004923)
  b_5seed_4subj: FOUND (/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/b_5seed_4subj_20260506_005153)
  seed-0 source_only: FOUND (intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/source_only/results.txt)
  seed-0 policy_safe_no_shallow: FOUND (intentflow/offline/results/a1_9subject_no_shallow_20260505_182651/eval/s7/policy_safe_no_shallow/results.txt)
  seed-0 replay_safe_uniform: FOUND (intentflow/offline/results/replay_safe_9subject_20260505_212942/eval/s7/replay_safe_default/results.txt)
  seed-0 replay_h6_weighted: FOUND (intentflow/offline/results/replay_h6_9subject_20260506_004038/eval/s7/replay_h6_weighted/results.txt)
  trace npz: FOUND (/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/_smoke_replay_v3_s2_212138/replay_safe_otta_stats_s2_tcformer_replay_safe_otta.npz)

## 2. fig_main_source_vs_replay
  wrote CSV: fig_main_source_vs_replay.csv
  wrote PNG: fig_main_source_vs_replay.png

## 3. fig_per_subject_delta
  wrote CSV: fig_per_subject_delta.csv
  wrote PNG: fig_per_subject_delta.png

## 4. fig_replay_candidate_trace
  wrote CSV: fig_replay_candidate_trace.csv
  wrote PNG: fig_replay_candidate_trace.png

## 5. fig_safety_tradeoff
  wrote CSV: fig_safety_tradeoff.csv
  wrote PNG: fig_safety_tradeoff.png

## 6. fig_s7_seed_stability
  wrote CSV: fig_s7_seed_stability.csv
  wrote PNG: fig_s7_seed_stability.png

## 7. all_results_summary.csv
  wrote CSV: all_results_summary.csv

## 8. caption_draft.md
  wrote: caption_draft.md
```
