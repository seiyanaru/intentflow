# 260604 進捗: Head-Frozen Cross-Family Decoder（2法則駆動）と精度天井の確定

本ノートは、DA-DC を起点に「新規性・安全性・論理性・精度・汎用性」を満たす手法を探索した
一連の検証（2026-06-03〜04）の確定事実と、到達した提案手法をまとめる。
すべて **TCFormer**、**BCIC-IV-2a / 2b**、**session_E（cross-session）評価**、seed0（一部seed1）。

---

## 結論先行
- **精度の上積みは全方向で閉じた**（label-free / CFCS / 深層多様体 / 最小ラベル / ErrP）。
  「DA-DC の上に +5」は不可能と**実測で確定**。"+5" は **無適応(source)比**であり DA-DC 自体が達成済み。
- **本物の成果は精度ではなく 2 つの測定された法則と、それが導く安全デコーダ**。
  - **法則1（族脱相関）**: 深層 ⊥ リーマン幾何 ⊥ スペクトルの誤りは脱相関 → confident-wrong を検出。
  - **法則2（drift→head）**: session drift は分類 **head** を壊し、**深層 feature は生存**（2a+2b で成立、新規）。

---

## Exact config / scripts（再現）
- 特徴・logits（TCFormer, EA-aware, session_E penultimate 64-d）:
  `intentflow/offline/results/ea_aware_tcformer_s{sid}_seed0_*/{features,logits}_s{sid}_TCFormer.npz/.npy`（2a）、
  `ea_aware_tcformer_bcic2b_s*_seed0_*/...`（2b）。
- 異族モデル（2a）: `260603_diverse_riemann_preds.npz`(R), `260603_d2_csp_preds.npz`(C), `260603_d3_spectral_preds.npz`(S)。
- portfolio probs: `260602_expert_portfolio_table/expert_portfolio_arrays.npz`（source/full_ea/shrink_0.1）。
- 解析:
  `scripts/analysis/tcformer_manifold_headroom.py`（法則2, 2a）、上記 2b 版、
  `scripts/analysis/head_recalib_curve.py`（最小ラベル/ErrP 開錠カーブ）、
  `scripts/analysis/tcformer_labelfree_battery.py`（label-free 総当たり）、
  `scripts/analysis/drift_vs_illiteracy.py`（drift vs illiteracy 診断）、
  `scripts/analysis/make_law_decoder_figures.py`（本ノートの図）。

## Dataset and split
- BCIC-IV-2a（4class, 9被験者, 288 trials/session）、2b（2class, 9被験者）。
- train=session_T、eval=session_E（被験者内・別日＝session drift）。ラベルは採点専用。

---

## Metrics & Delta from baseline

### 1) DA-DC は label-free 天井（DA-DC比で超える手は無い）
| 手法 (2a, seed0) | acc | vs DA-DC |
|---|---|---|
| source（無適応） | 82.7 | -4.4 |
| DA-DC = blend+0.3R | **87.2** | 0 |
| DADC+stack（agreement-seed） | 88.0 | **+0.7**（label-free 最良, seed1も+0.6） |
| stack/cotrain/multi-D/transductive | ≤ DA-DC | ≤0 |
- cross-family per-trial **oracle T+R+S = 94.75（+12pp）だが label-locked**（ルーティングに真ラベルが必要）。

### 2) DA-DC の分布特性（source比, 2a）— ゼミ図1
- 平均 +4.5pp、**下位半分 +5.2pp / 上位半分 +3.8pp**、**最悪でも +0.35pp（非回帰）**。
- = 弱被験者を上げ、強被験者を崩さない（目標プロファイルを label-free で達成）。

### 3) 法則1（安全 / confident-wrong 検出）— ゼミ図3
- T 高確信領域の confident-wrong を、abstain 予算15%で:
  **cross-family 不一致 47%捕捉 vs single-UQ(margin) 0%**（単一最良異族 Riemann 単独なら ~60%）。
- single-UQ は高確信領域で **構造的に 0%**（自分の壊れた確信度を測るため）。

### 4) 法則2（drift→head）— ゼミ図2
| | softmax(head) | LDA-probe(特徴, 真ラベル) | head-headroom |
|---|---|---|---|
| 2a 平均 | 83.1 | 85.6 | **+2.5pp** |
| 2b 平均 | 84.1 | 88.3 | **+4.2pp** |
| 2b S5（劇的例） | 64.1 | 97.8 | **+33.8pp** |
- 深層 feature は session_E で 85–95% 線形分離可能なのに、学習済み head は 73–84%。
  **drift は決定境界(head)のズレで、特徴崩壊ではない**。→ **CFCS（特徴再学習）失敗を機構的に説明**。

### 5) 精度を上げる試みは全方向で閉じた（確定）
| 試み | 結果 |
|---|---|
| label-free（DA-DC/stacking/routing/multi-D/transductive） | 天井 ~87（+5.2/source）。DA-DC比 最大 +0.7 |
| CFCS（test時に深層 head/feature 再学習, deep-gambler/humble） | -10pp級 崩壊（法則2: 標的が違う） |
| 深層多様体 transductive denoising | label-free では害（-0.9, 5/9 regression） |
| **最小ラベル/ErrP head-recalib** | **2a: K=64クリーンでも 84.3 < DA-DC（単一head天井<アンサンブル）。2b: ErrP品質ノイズ ε≥0.2 で softmax以下に崩壊** |

---

## Interpretation（観測 / 解釈 / 不確実性を分離）
- **観測**: 上表の数値。法則2は 2a(+2.5)・2b(+4.2) の両方で正。
- **解釈**: label-free 精度は構造天井（confident-wrong は label-locked）。DA-DC のアンサンブルが単一head/単一表現の天井を上回るため、ラベルで head を直しても DA-DC は超えない。illiteracy 被験者は古典族では救えない（表現限界, `drift_vs_illiteracy.py`）。
- **新規性の所在**: 法則2（drift localizes to head, CFCS失敗を説明）が**最も新規**（先行に未見）。法則1の機構（cross-model disagreement→confident-error）は LLM/VLM で先行（`prior-art-novelty-bounds` 参照）だが、**EEG-drift × 同族vs異族 × 安全ゲート**の instantiation は防衛可能。
- **実用上の意味**: 精度ブレークスルーではない。**label-free・軽量・実機・非回帰・confident-wrong安全**という"安全で誠実なデコーダ"＋"2法則"が貢献。現実的 venue: TNSRE / JNE。

## 提案手法: Head-Frozen Cross-Family Decoder
1. 深層 feature を凍結（法則2: feature は生きている → 動かすな）。
2. cross-family（R: EA-リーマン接空間, S: スペクトル）と融合 → source比 +5、非回帰。
3. source-fit 異族で **confident-wrong 安全ゲート**（fail-closed abstain, 法則1）。
4. 全 label-free・軽量。CFCS（head/feature 再学習）と ErrP head-recalib は**やらない**（実測で不発）。

## Figures（`260604_law_based_decoder/`）
- `fig1_subjectwise_2a.png`: 被験者別 source→DA-DC（弱者救済＋非回帰）。
- `fig2_head_headroom_law2.png`: head-headroom 2a/2b（法則2, S5劇的例）。
- `fig3_confidentwrong_law1.png`: confident-wrong 捕捉 cross-family vs single-UQ（法則1）。

## Remaining uncertainty
- 法則2の **model 跨ぎ汎用性**（EEGNet/ATCNet で再現するか）未検証＝論文の生命線。
- 法則1 の confident-wrong 捕捉率は異族の選び方依存（R単独 ~60% / R+S 47%）。最適witness未確定。
- 2b の DA-DC baseline（2b 異族 preds）未整備。
- 疲労(SADT/SEED-VIG)での跨パラダイム検証 未着手。

## Recommended next experiment
**法則2を第2backbone（EEGNet/ATCNet）で再現**: session_E 特徴を抽出し softmax vs LDA-probe(真ラベル)。
2a/2b × 複数backbone で head-headroom>0 が一貫すれば「**drift localizes to the head**」を model 跨ぎ一般法則に昇格し、
Head-Frozen Cross-Family Decoder の土台が確定する。
