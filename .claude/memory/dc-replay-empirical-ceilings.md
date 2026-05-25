---
name: dc-replay-empirical-ceilings
description: DC-Replay検証で確定した、TCFormer凍結+予測補正型OTTAの精度天井とL3 commitのno-op
metadata:
  type: project
---

TCFormer凍結 + posterior/prior/memory補正型OTTA(DC-Replay / CMC)の精度上限を、source-only logitsのoracle分析と実測で確定(2026-05-26)。

- **L3 model-state commit は精度に no-op**: BCIC2aで commit回数 0/33/111/150 でも test精度が小数点5桁まで同一(83.56556%)。最高精度variant(+1.23pp)は commit=0。`l3_diagnostics.csv`。
- **Oracle天井(後知恵=上界、3データセット)**:
  - 静的グローバルprior補正 = Δ0.00(全データセット、テストがクラスバランスのため無価値)
  - per-class bias = **L1の上限**: 2a +3.36 / 2b +1.73 / HGD +3.18 pp
  - top2(特徴を変えない絶対上限) = source精度に逆相関: 2a +11.84 / HGD +5.00(2bは2-classで自明)
- 実測 best DC は 2a single-seed で +1.23pp = L1天井の37%、top2天井の10%。

**Why:** 「commitlessで大きく(+5pp)上げる」は3データセットで物理的に不可能と確定。L1補正の天井が低いのは2a特有でなく汎用的。これが [[research-direction-2605]] の方向転換の根拠。
**How to apply:** posterior/prior/memory補正の改善案を評価する前に、この天井(L1 ≲ +3.4pp)を上限として見積もる。+5pp級を狙うなら特徴適応が必須(=安全性トレードオフ)。再現: `intentflow/offline/scripts/analysis/oracle_ceiling.py`、詳細 `docs/research_progress/260526_oracle_ceiling/`。
