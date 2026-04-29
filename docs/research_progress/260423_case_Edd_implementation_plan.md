# 260423 案 E'' (Hierarchical Prototype OTTA refined) 実装プラン

- 目的: 先行研究精読後に refinement された `案 E''` を、最小差分で既存 repo に実装するための step-by-step plan。
- 前提ドキュメント:
  - [260422_offline_model_design_discussion.md](260422_offline_model_design_discussion.md) — 案 E'' 確定経緯
  - [260422_precedent_differentiation.md](260422_precedent_differentiation.md) — MI-IASW/BTTA-DG/T3A/SAR 等との差別化
  - [260422_next_implementation_plan.md](260422_next_implementation_plan.md) — Phase C (2b 横展開 + train-time 改良) の全体計画
- 本ファイルの scope: **案 E'' の源コード実装 + 初回比較 (A/B/E''-deep-only) まで**。2b 横展開と 5-seed sweep は別。

---

## 0. 段階分割

| 段階 | 内容 | ブロッカー |
|---|---|---|
| S1 | EA 前処理を datamodule に追加（既存 z-score と共存） | なし |
| S2 | Source 側 prototype 収集機構（最終 epoch で shallow/deep class prototype を保存） | S1 |
| S3 | 案 E'' OTTA wrapper: BN 凍結 + Tri-Lock per-depth + prototype EMA | S2 |
| S4 | Smoke (2a, 1 subject) で forward-only 動作確認 | S3 |
| S5 | 初回比較: A (hybrid@0.01) / B (vanilla-both) / E''-deep-only、2a 全 9 subject, 1 seed | S4 |
| S6 | 判定: 採択条件を満たせば 5-seed sweep へ、満たさなければ refinement | S5 |

**並行監視**: 2b source model の TCFormer 公式準拠学習（pid 2331305, [260422_2b_preprocess_diff.md](260422_2b_preprocess_diff.md) 追記参照）は独立に完走を確認する。案 E'' の 2b 展開は S6 判定後。

---

## S1. EA (Euclidean Alignment) 前処理の追加

### 目的
T-TIME / BFT / BTTA-DG で前処理デフォルトの EA を、source 学習 + test-time 両方に入れる。skip すると baseline 側が弱くなり、案 E'' の gain が過大評価される。

### 実装箇所
- [intentflow/offline/datamodules/base.py](../../intentflow/offline/datamodules/base.py) — 既存 `z_scale` の前段に `euclidean_align` オプションを追加（subject-level 単位）。
- EA 数式: 各 subject の trial 集合 $\{X_i\}$ に対し、reference matrix $R = \frac{1}{N}\sum_i X_i X_i^\top$ を計算し、$\tilde{X}_i = R^{-1/2} X_i$ で正規化する。
- 実装参考: T-TIME の `EA` 実装（<https://github.com/sylyoung/DeepTransferEEG>）または BFT 実装。15 行程度の pure PyTorch。

### 設定
- 新設定 key: `dataset.euclidean_align: bool = false`（既存実験を壊さないため default False）
- 案 E'' config では `true` に設定する。
- 既存 baseline (`hybrid@0.01`, `vanilla-both`) も EA 有効で再測定する（A/B/E'' が同じ前処理条件で比較される）。

### 検証
- Unit test: shape 不変 `(B, C, T) → (B, C, T)`。
- 2a subject 1 の 1 trial で NaN/Inf が出ないことを確認。
- EA on/off で baseline (source_only) の精度差を smoke 実行で把握（T-TIME 報告: EA で 2-3 pp 向上）。

---

## S2. Source 側 prototype 収集

### 目的
最終 epoch で class-wise prototype を shallow / deep の 2 層で作り、source artifact として保存する。T3A 失敗（高次元単一平均の脆弱性）を回避するため **augmentation forward × N 回集約** で robust 化する。

### 実装箇所
- [intentflow/offline/models/tcformer/tcformer.py](../../intentflow/offline/models/tcformer/tcformer.py)
  - `forward_features(x, return_layers=['shallow', 'deep'])` を追加。
  - `shallow` = MK-CNN 出力直後（(B, F1, T')）のチャネル平均 → (B, F1)
  - `deep` = Transformer 出力（(B, N_tokens, D)）の token 平均 → (B, D)
- [intentflow/offline/train_pipeline.py](../../intentflow/offline/train_pipeline.py)
  - 最終 epoch 完了後、`collect_source_prototypes(model, train_loader, n_aug=16)` を呼び出す。
  - 各 trial に対し `n_aug` 回 augmentation forward → shallow/deep feature を集め、class-wise 平均 → `source_protos = {shallow: (C, F1), deep: (C, D)}` を .pt で保存。
  - 保存先: `{results_dir}/source_protos.pt`

### 設定
- `model.collect_prototypes: bool = false`（default）
- 案 E'' config では `true`、かつ `n_aug_prototype: 16`。
- augmentation は既存 `interaug` のうち**単純な noise/jitter 系のみ**を使う（mixup は label が混ざるので prototype 構築時は無効）。

### 検証
- prototype の shape が (C, F1) と (C, D) で正しい。
- 同一 class 内の prototype の L2 distance (intra) < 異クラス間 distance (inter) の 1/2 以下（silhouette 相当の最低要件）。
- 既存 source training の loss/acc curve が変わらない（最終 epoch 後の追加計算のみで学習は不変）。

---

## S3. 案 E'' OTTA wrapper

### 目的
BN 完全凍結 + Tri-Lock per-depth + deep prototype EMA + forward-only の OTTA 推論経路を作る。

### 実装箇所
- 新規ファイル: [intentflow/offline/models/tcformer_proto_otta.py](../../intentflow/offline/models/tcformer_proto_otta.py)
  - `class TCFormerProtoOTTA(nn.Module)`:
    - 既存 TCFormer を保持し、**BN の `track_running_stats=False` + `eval()` 強制 + 全 param `requires_grad=False`**。
    - 初期化時に `source_protos.pt` を読み込む。
    - `forward(x, state)` が `(B, C_cls)` の logits を返す。state は per-subject の `deep_proto_ema` を保持。
  - `adapt_step(x, state)`:
    1. EA + z-score 前処理（datamodule 側で既に済んでいる前提なので pass-through でも可）
    2. Forward → `{shallow_feat, deep_feat, logits}`
    3. **Tri-Lock gate**:
       - pmax = max(softmax(logits))
       - SAL = cosine(shallow_feat, source_proto_shallow[ŷ])（既存 SAL と同じ）
       - energy = -T × logsumexp(logits / T)
       - pass = (pmax > τ_pmax) AND (SAL > τ_SAL) AND (energy_quantile > τ_energy)
       - **shallow と deep で別々に gate 評価**（hierarchical Tri-Lock）
    4. gate pass なら:
       - `state.deep_proto_ema[ŷ] ← (1-m) × state.deep_proto_ema[ŷ] + m × deep_feat`
       - predict = argmax(softmax(logits) + α × cosine_to_deep_proto_ema)
    5. gate fail なら:
       - predict = argmax(logits) または abstain（既存 abstain API に合わせる）

### 設定
- 新 config: [intentflow/offline/configs/tcformer_proto_otta/tcformer_proto_otta_Edd.yaml](../../intentflow/offline/configs/tcformer_proto_otta/tcformer_proto_otta_Edd.yaml)
  - `otta.bn_update: false` (構造)
  - `otta.prototype_update: "deep_ema"` (shallow は ablation 用、初回は off)
  - `otta.ema_momentum: 0.05`
  - `otta.gate: hierarchical_tri_lock`
  - `otta.tau_pmax: 0.7`（既存 hybrid と同じ）
  - `otta.tau_sal: 0.5`
  - `otta.energy_quantile: 0.95`
  - `otta.logit_fusion_alpha: 0.3`（logits + prototype distance fusion の重み、smoke で確定）
- [intentflow/offline/utils/get_model_cls.py](../../intentflow/offline/utils/get_model_cls.py) に `tcformer_proto_otta` を登録。

### 検証
- 全 param `requires_grad=False`（autograd graph が作られないことを `torch.no_grad()` 包含で確認）
- Batch=1, single trial forward で NaN/Inf 出ず、gate pass/fail 両経路が動く
- Gate が一度も pass しない subject で source_only と同一 prediction になる（backprop なし、prototype 更新なし → 等価）

---

## S4. Smoke 実行

### 内容
- 2a subject 1 のみ、1 seed、source model は既存 hybrid@0.01 学習済み checkpoint を流用（EA 有り版を別途再学習する必要あり → S2 と同時に走らせる）
- `source_only` / `vanilla-both` / `E''-deep-only` の 3 条件を 1 subject で完走させ、logs と final acc を確認

### 完了条件
- 3 条件が全て NaN なしで完走
- Gate 統計（pass 率、shallow/deep 別 pass 率）が log に出る
- `E''-deep-only` の accuracy が `source_only` ± 10pp 以内（完全崩壊していない）
- 所要時間: 1 subject × 3 条件 ≤ 30 分

---

## S5. 初回比較（2a 全 9 subject, 1 seed）

### 条件
1. `A`: 既存 `hybrid@0.01 (shallow_mean_deep_both)` + EA 前処理追加
2. `B`: `vanilla-both` + EA 前処理追加
3. `E''-deep-only`: BN frozen + deep prototype EMA + hierarchical Tri-Lock + EA + mixup source

### 報告指標
- per-subject accuracy
- 9-subject mean ± std
- worst-subject Δ (vs source_only)
- **hybrid gain** = E''-deep-only - source_only（adaptability 指標）
- Gate pass 率（shallow / deep 別）
- Prototype EMA 更新回数（誤 pseudo-label 監視用）
- WSD, NTR-S@0.5pp

### 採択条件（[260422_next_implementation_plan.md の 5 本柱](260422_next_implementation_plan.md)）
- `2b consistency`: 段階 6 で判定
- `worst-subject rescue`: 2a worst-subject Δ > 0（1 seed なので seed 条件は skip、参考値）
- `all-9 mean`: source_only ± 0.5 pp 以内
- `safety`: WSD / NTR-S@0.5pp が A / B より悪化しない
- `adaptability`: `E''-deep-only - source_only` > `hybrid@0.01 - source_only - 0.5pp`（暫定しきい値、2a baseline gain 基準）

---

## S6. 判定と次アクション

### 3 つの出口
1. **採択**: 5 本柱を全満たす → `5-seed sweep` に進む（別計画）。2b 展開と並列で新規性主張論文化の初稿を書く。
2. **条件付き採択**: adaptability と safety は通るが worst-subject rescue が unstable → **案 E''-dual (shallow prototype 追加)** の ablation を 1 run 追加
3. **却下**: mean or safety が崩れる → prototype EMA momentum / Tri-Lock 閾値を 1 軸だけ動かして再 smoke、それでもダメなら **案 E'' 白紙 → forward-only を捨てて MI-IASW 追従系 (MABN + prototype) に路線変更**

### 白紙ケースの備え
案 E'' の基本前提（BN 完全凍結で EEG MI が動く）が S5 で崩れた場合、**MI-IASW 再現実装** が最短復帰路。MI-IASW は 2a で +6.92 pp vs source を達成しており、既存 baseline として強い。案 E'' が負けた場合は MI-IASW を baseline に追加して案 E''-refined を検討する。

---

## 実装リスクと対策

| リスク | 検出方法 | 対策 |
|---|---|---|
| EA が既存 z-score と競合し精度低下 | S1 smoke で baseline acc が -2pp 超 | EA の適用順序を `z-score 前` → `z-score 後` に切り替えて比較 |
| Prototype 収集で augmentation が強すぎて class が混ざる | S2 検証で silhouette < 0 | augmentation 強度を半減、または `n_aug` を 4 に削減 |
| Tri-Lock が全 sample で fail し、prototype EMA が一度も起きない | S3/S4 で pass 率 ≤ 5% | τ_pmax を 0.6、τ_SAL を 0.3 に緩める（smoke で再測定） |
| Deep prototype EMA が collapse（全 class が同一 prototype に集約） | S5 で prototype L2 distance が epoch 進行で減少 | EMA momentum を 0.01 に下げる、または gate 失敗時の reset を実装 |
| 2a baseline (hybrid@0.01) が EA 導入で再現しない | S1 で既存結果と ±1pp 以上ズレる | EA を**案 E'' 側のみ**に適用し、baseline は既存のまま（公平性の議論を別途追記） |

---

## 実装時間見積もり

| 段階 | 見積もり |
|---|---|
| S1 (EA) | 2 時間（実装 1 + smoke 1） |
| S2 (prototype 収集) | 3 時間（実装 2 + source 再学習 smoke 1） |
| S3 (OTTA wrapper) | 5 時間（実装 3 + unit test 1 + smoke 1） |
| S4 (smoke) | 1 時間 |
| S5 (2a 全 9 subject) | GPU 3-4 時間 + 集計 1 時間 |
| S6 (判定と分岐) | 1 時間 |
| **合計** | **~15 時間 (2 日)** |

---

## 次の 1 アクション

1. 本 plan を Codex に共有し、`案 E''` 実装順序と risk マトリクスの合意を取る
2. S1 (EA 前処理) の実装に着手
3. 並行で 2b source model (pid 2331305) の完走を monitor、完走次第 2b first pass の 3 条件比較スクリプトを起票（案 E'' 2b 展開の前提）
