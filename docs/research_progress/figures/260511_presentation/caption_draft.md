# 図キャプション草案 (260511 ゼミ)

各図 1 文。スライド下部にそのまま貼れる長さ。

## fig_main_source_vs_replay.png

> aug-True 9 被験者・1 seed・同一 source checkpoint・同一 split。replay_safe_uniform は素の TCFormer (source_only=82.72%) に対し +0.69pp の平均精度改善 (83.41%) を達成し、HSC=0/9 を保った。

## fig_per_subject_delta.png

> 被験者別 Δ accuracy (vs source_only)。replay_safe_uniform は 9 中 7 で 0pp 以上、唯一の負側 S4 でも policy_safe_no_shallow と同水準 (−1.73pp)。

## fig_safety_tradeoff.png

> Safety–gain tradeoff: 横軸 mean Δ、縦軸 worst-subject Δ。理想点は右上 (gain & no harm)。replay_safe_uniform は WSD 同等 (−0.35pp) で mean Δ 最大 (+0.69pp)。

## fig_s7_seed_stability.png

> S7 (gain subject) の 4 seed 結果 (seed 0 from a1, seeds 1-3 from b_5seed, no_aug regime)。source_only: 89.58 ± 1.96、replay_safe_uniform: 91.41 ± 1.48、replay_h6_weighted: 91.58 ± 1.54。replay は variance reduction にも効く (std 削減率 uniform: 25%, h6: 22%)。

## fig_replay_candidate_trace.png

> S2 の代表 trial (trial 24)。Policy は 5 候補 operator を順に提案、Tier 2 (replay) gate で前段の shallow_var / hybrid_BN / deep_BN / prototype を sim_score ≤ 0 で reject、最後に logit_bias_update が sim > 0 で commit された。SafeCommit が「動いている」直接証拠。

## fig_method_flow_replay_safecommit.png

> Replay-Validated SafeCommit の処理フロー。trial ごとに candidate operator を順に試し、Tier 1 (即時ガード) と Tier 2 (replay buffer 上の simulated reward) の両方を通った operator のみ commit する。
