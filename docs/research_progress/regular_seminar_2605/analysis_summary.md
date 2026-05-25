# 2605 regular seminar result analysis

## Headline

- Plan C: the highest-mean DC variant improves over source, but introduces one harmful subject.
- Plan B: seed stability favors replay_safe_uniform under HSC<=1; DC variants trade small mean gains for larger HSC.
- The strongest DC gains are mainly from L1/L2 prediction correction and external memory, not from L3 model-state commits.

## Key numbers

- Plan C best mean: `dc_grid_a04_m55_n8_tol5e5` mean=83.95, delta=+1.23pp, worst=-1.38pp, HSC=1/9.
- Plan C best HSC=0: `replay_safe_uniform` mean=83.41, delta=+0.69pp, worst=-0.35pp.
- Plan B best mean: `dc_grid_a08_m45_n8_tol0` mean=76.13, delta=+0.49pp, worst=-1.04pp, HSC=4/12.
- Plan B best HSC<=1: `replay_safe_uniform` mean=76.01, delta=+0.38pp, worst=-0.69pp, HSC=1/12.

## L3 diagnostics

| plan | variant | delta | HSC | model_commit | memory_admitted | candidate_trials | sim_finite | sim_pos |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C | dc_correction_memory_no_commit | +0.85 | 1 | 0 | 769 | 61 | 0 | 0 |
| C | dc_replay_gated | +0.85 | 1 | 111 | 769 | 1085 | 195 | 111 |
| C | dc_commit_no_replay_gate | +0.85 | 1 | 150 | 769 | 842 | 0 | 0 |
| C | dc_random_sparse_p10 | +0.85 | 1 | 33 | 769 | 233 | 0 | 0 |
| C | dc_grid_a04_m55_n8_tol5e5 | +1.23 | 1 | 0 | 750 | 1853 | 376 | 244 |
| B | dc_correction_memory_no_commit | +0.26 | 3 | 0 | 743 | 972 | 0 | 0 |
| B | dc_replay_gated | +0.26 | 3 | 101 | 743 | 2193 | 207 | 101 |
| B | dc_commit_no_replay_gate | +0.26 | 3 | 153 | 743 | 1919 | 0 | 0 |
| B | dc_random_sparse_p10 | +0.26 | 3 | 23 | 743 | 1118 | 0 | 0 |
| B | dc_grid_a04_m55_n8_tol5e5 | +0.44 | 4 | 0 | 707 | 2958 | 373 | 215 |

## Interpretation

1. Replay-SafeCommit remains the most defensible current main result because it improves mean accuracy while preserving a low HSC in the seed-stability setting.
2. DC-style correction has a real accuracy signal: Plan C reaches +1.23pp and Plan B reaches +0.49pp.
3. The price is safety: the best DC variants increase HSC, especially for S4/S6 seeds.
4. The fact that no-commit or zero-commit variants can match the best DC performance means the current novelty should be framed as commitless memory-corrected OTTA, with L3 deferred commit left as an open extension.
