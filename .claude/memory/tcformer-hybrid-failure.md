---
name: tcformer-hybrid-failure
description: TCFormer_Hybrid(TTT adapter)がBaseに負けた事実と、特徴適応の落とし穴
metadata:
  type: project
---

TCFormer_Hybrid(学習時からTTT Adapterを並列配置、Entropy gating + 2-Pass で難例だけ適応)は Base TCFormer に負けている: 2a **-1.15**、2b **-1.91**、HGD **-13.66pp**(README.md §4、README_Hybrid.md)。

**なぜ負けたか(落とし穴)**:
- 強baseline(HGD source 92.95%、適応余地小)で特徴適応 → 決定境界を壊す。
- alignment(EA)が無いまま適応 → T3Aが cross-session で -5〜-9pp 壊れたのと同型。
- Base TCFormer(84.67%)を凍結せず**別アーキを訓練** → 強い特徴を捨てている。
- entropy gating(= uncertainty-triggered)を持つのに負けた → **triggered更新"だけ"では不十分**。

**Why:** 「難例だけ適応(triggered)」は新規アイデアでなく repo で実装済み・失敗済み。Deep Research が推奨した triggered 方向は、そのままでは二の舞。
**How to apply:** 特徴適応を入れるなら必ず (1) EA前段で整列、(2) Base凍結で強い特徴を活かす、(3) 強baseline(HGD)では適応最小/オフ。この3つを欠くと Hybrid を再現する。関連: [[research-direction-2605]]、[[dc-replay-empirical-ceilings]]。
