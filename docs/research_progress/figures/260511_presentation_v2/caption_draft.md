# 図キャプション草案 (260511 ゼミ, v2)

## fig_main_source_vs_replay.png
- Slide: 主結果
- Message: Replay-SafeCommit は source_only に対して平均精度を改善し、HSC=0/9 を維持した。
- Notes: data: c_aug_true_9subj_20260506_004923/summary.json. variants: source_only / policy_safe_no_shallow / replay_safe_uniform / replay_h6_weighted. 1 seed, aug-True, same source checkpoint and split.

## fig_per_subject_delta.png
- Slide: 結果の読み取り
- Message: 被験者別 Δ を見ることで、平均改善が一部被験者の大きな悪化を隠していないかを確認する。
- Notes: data: c_aug_true_9subj summary.json. 9 subjects. policy_safe_no_shallow / replay_safe_uniform / replay_h6_weighted を vs source_only で比較。−0.5pp ライン併記。

## fig_replay_candidate_trace.png
- Slide: replay は何をしているか
- Message: Replay-SafeCommit は候補更新を順に仮適用し、replay 上で悪化する候補を reject してから、改善が見込める候補だけを commit する。
- Notes: data: _smoke_replay_v3_s2_*/replay_safe_otta_stats_s2_tcformer_replay_safe_otta.npz. trial は実ログから「複数 reject + 1 commit」パターンを自動選定。

## fig_safety_tradeoff.png
- Slide: 現時点で言えること
- Message: 安全制約を満たした上で mean Δ が高い手法を採択する、という本研究の評価方針を示す。
- Notes: data: c_aug_true_9subj summary.json. hybrid@0.01_reference は同一 regime での再評価が無いため reference 表示 (mean Δ は missing)。

## fig_s7_seed_stability.png
- Slide: 結果の読み取り (補足)
- Message: S7 では replay 系の手法が複数 seed で一貫して改善し、variance reduction の可能性を示している。
- Notes: data: b_5seed_4subj summary.json (seeds 1-3) + seed 0 from a1 / replay sweep dirs. no_aug regime. 4 seeds combined.
