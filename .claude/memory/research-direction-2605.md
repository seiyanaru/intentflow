---
name: research-direction-2605
description: 研究方向の確定(精度主軸、alignment-first selective minimal feature adaptation)
metadata:
  type: project
---

2026-05下旬、内部分析(oracle天井 / L3 no-op / Hybrid失敗)と外部Deep Research 3本で確定した研究方向。

- **主軸は「精度を大きく上げる」。** safety と軽量は制約に降格(誰も大きく壊さない / 平均コスト軽量)。ユーザー判断「精度が上がらなきゃ意味がない」。
- **commitless posterior/prior correction(DC-Replay/CMC)は捨てた。** 精度天井が低く([[dc-replay-empirical-ceilings]])新規性も弱い(LAME/AdaNPC/BFT 2026)。
- **本命: alignment-first selective minimal feature adaptation。** 常時は EA/RA + AdaBN + Base凍結で即時予測、難例だけ tiny adapter/head/BN を更新。軽量性は「平均コスト低」で守る。
- **新規性の核**: EEG-MIのtop2 slack(2a +11.84pp)を、EA条件付き・難例限定のminimal feature adaptationで回収。BFTの核=transformation aggregation とは別。
- **必須要素(Hybrid失敗の教訓 [[tcformer-hybrid-failure]])**: EA前段 + Base TCFormer凍結。triggered更新"だけ"では勝てない。強baseline(HGD)では適応最小。
- 外部相場: T-TIME +2.9〜6.1pp(全パラ更新+batch8)、dual-stage +3.6pp。大gainはfeature update側。strict freeze+no-BPはBFTと被り不利。

**Why:** データ(oracle)と文献が「出力補正では上がらない、特徴適応が要る」で一致。ただし雑な特徴適応は Hybrid のように壊れる。
**How to apply:** 新実装は EA+Base凍結を土台に。最初の実験は EA+AdaBN only から(再学習不要、既存checkpoint `sources/sN/checkpoints/` 使用)。計画 `docs/research_progress/260526_research_plan_v1.md` 参照。
