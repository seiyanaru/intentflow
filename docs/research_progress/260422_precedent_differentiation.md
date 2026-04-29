# 260422 先行研究との差別化マトリクス — 案 E'' 防御

## 目的
- 案 E''（Hierarchical Prototype OTTA refined）が既存 EEG MI TTA / prototype TTA と**どの軸で独立か**を明示する。
- 精読済み 5 本 (MI-IASW / BTTA-DG / T-TIME / BFT / T3A / SAR / TopA-architecture) に対する差分を 1 枚で把握可能にする。
- ゼミで「なぜ既存手法の単純拡張でないか」を 30 秒で説明できる粒度まで整理する。

## 案 E'' 最小仕様（再掲）
1. **Source 側**:
   - TCFormer + inter-subject mixup
   - **EA (Euclidean Alignment) 前処理 必須**（T-TIME/BFT/BTTA-DG で前処理デフォルト）
   - 最終 epoch で **shallow と deep の class prototype を augmentation forward × N 回集約で構築**（robust 化）
2. **Test 側 (forward-only)**:
   - EA + z-score 前処理
   - Forward → `{shallow_feat, deep_feat, logits}`
   - **BN 完全凍結**（stats も affine も更新しない）
   - **Hierarchical Tri-Lock gate** (pmax × SAL × Energy を depth 別判定)
   - Gate pass: `deep_proto_c` を EMA 更新 (momentum ≤ 0.05)、shallow は ablation で有効化
   - Gate fail: abstain（何も更新しない）
3. **初回比較**: A = hybrid@0.01 / B = vanilla-both / **E''-deep-only** / (optional) E''-dual

---

## 差別化マトリクス

| 軸 | MI-IASW (IJCNN 2025) | BTTA-DG (ICLR 2026) | T-TIME (2024) | BFT (2026) | T3A (NeurIPS 2021) | SAR (ICLR 2023) | **案 E''** |
|---|---|---|---|---|---|---|---|
| **Backbone 想定** | EEGNet (BN) | SincAdaptNet (LN, BN-free) | EEGNet (BN) | EEGNet (BN) | ResNet/ViT (BN frozen) | ResNet GN/LN | **TCFormer (BN)** |
| **BN 扱い** | MABN (mixed + 勾配更新) | LN なので無関係 | TBN (target stats) + 全 param 更新 | 勾配フリー (forward-only) | 凍結 | 凍結 (GN/LN affine のみ更新) | **完全凍結** |
| **適応媒体** | BN 統計 + 全 param | Dirichlet → GMM Bayesian fusion | 全 param + ensemble + CEM | logit aggregation + MDR | 分類器 weight (prototype) cumulative set | 全 LN affine + SAM | **shallow + deep prototype EMA** |
| **Depth / Hierarchy** | 単層 (classifier 入力) | 単層 (post-softmax temporal) | 単層 (classifier 入力) | 単層 (logit) | 単層 (encoder 出力) | 単層 (loss surface) | **shallow + deep 2 層 (TCFormer 内部)** |
| **Prototype 扱い** | source-fixed（classifier weights 代用） | なし（Dirichlet α 使用） | なし | なし | **cumulative set 平均** | なし | **source+target EMA / per-depth** |
| **Gating / Filter** | CAW (source-guided soft weighting) | confidence τ_conf + entropy τ_ent (memory 受入) | MDR (mean-distance rejection) | learning-to-rank | entropy top-M | entropy < E_0 | **pmax × SAL × Energy per-depth (hard abstain)** |
| **Backprop 有無** | あり (全 param) | **なし (gradient-free)** | あり (全 param, CEM) | **なし (forward-only)** | **なし (template 更新のみ)** | あり (SAM) | **なし (prototype EMA のみ)** |
| **EA 前処理** | 明示なし | **あり** | **あり** | **あり** | N/A (CV) | N/A (CV) | **あり（追加決定）** |
| **Dataset / 成績** | 2a 68.94% (+6.92 vs src), 2b 84.25% (+6.68) | 2a 78.70% (+5 vs EEGNet src) | 2a 79.30% (EEGNet) | 2a 77.39% | CV DG のみ、T-TIME で EEG MI は 68.23% 失敗 | CV のみ | **未評価** |

---

## 4 軸の非被覆性（案 E'' が先行と重ならない主張）

### (1) **BN 完全凍結** — 先行なし
- MI-IASW は MABN で **BN 統計に勾配を流す**（完全凍結でない）
- T-TIME は全 param 更新、BN はむしろ積極的に適応対象
- BFT は forward-only だが BN stats は TBN (target batch stats) で更新
- BTTA-DG は LN 前提のため BN 凍結の議論自体が発生しない
- **案 E''** は TCFormer (BN 入り) に対して BN 完全凍結 + 別媒体 (prototype) で適応 → **BN を持つ EEG MI backbone で BN を使わない TTA 設計**

### (2) **Shallow + Deep dual prototype** — 先行なし
- 既存 prototype 手法 (MI-IASW, T3A) はすべて **classifier 直前の単層特徴**
- BTTA-DG も post-softmax temporal embedding の単層
- **案 E''** は MK-CNN 出力 (shallow) と TCFormer Transformer 出力 (deep) の **2 層に prototype を配置し、独立 gate で更新**
- 物理的意味: shallow は spectral-spatial 特徴（被験者差に弱い）、deep は sequence-level 意味（被験者差が吸収される）→ 適応速度が違う depth で独立媒体を持つ

### (3) **Per-depth Tri-Lock (pmax × SAL × Energy)** — 先行なし
- MI-IASW の CAW は **SAL-only** (source alignment のみ)
- BTTA-DG は conf + entropy の 2 条件 (pmax なし, energy なし)
- T3A は entropy top-M のみ
- SAR は entropy < E_0 のみ
- **案 E''** は 3 条件 AND (pmax × SAL × Energy) を **depth 別に独立評価** → shallow で reject でも deep で accept という非対称判定が可能

### (4) **Forward-only + Target prototype EMA update** — 先行なし
- T3A は forward-only だが cumulative set（EMA でない、過去全部積む）
- BTTA-DG は forward-only だが memory bank（Dirichlet params 保存、EMA でない）
- MI-IASW は EMA を **teacher network weights** に適用（prototype EMA でない）
- **案 E''** は **prototype 自体を EMA 更新**。forward-only + prototype のみ更新 + gated → 既存 combo にない

---

## 引用できる先行の負のエビデンス (案 E'' を強化する)

### BN 起因の不安定性
- **SAR (CV)**: "The batch norm (BN) layer is a crucial factor hindering TTA stability"
- **BTTA-DG (EEG)**: "noisy trials induce misleading gradients that update BN weights and overwrite pre-trained structure—i.e., catastrophic forgetting"
- **我々の 260406 実験**: S2 worst-subject で BN-update が worse than source → 同じ結論が EEG MI + TCFormer で観察される
- → **案 E'' の BN 凍結は CV + EEG の独立 3 ソースで動機が補強される**

### Prototype TTA の EEG MI 固有困難性
- **T-TIME**: T3A = 68.23% < EEGNet source 73.52% on BNCI2014001 （**prototype TTA は EEG MI で壊れる**）
- **T-TIME 著者の説明**: "building class prototypes with high dimensionality is difficult"
- **MI-IASW ablation**: CAW+SAL 単体 = 67.69% (2a) → MABN+WA 追加で 68.94% に到達（**prototype + weighting 単体は弱く、BN 適応が支配的**）

### 案 E'' の対抗軸
- prototype 次元を下げる（shallow = 192-d, deep = 128-d projection head 追加、T3A の 2048-d よりずっと低い）
- prototype 更新を cumulative でなく gated EMA にする（誤 pseudo-label の積算回避）
- 単層 prototype でなく depth hierarchy にする（MI-IASW が失敗した "prototype 単体" の限界を超える）

---

## 新規性ステートメント（論文書き出し用ドラフト）

> 既存の EEG MI TTA は大きく 2 系統に分かれる — (i) **BN 統計/重みを更新**して target 側に合わせる系 (TBN, Tent, MI-IASW の MABN, T-TIME の全 param 更新) と、(ii) **BN 自体を避ける** ために **LN ベース構造**に置き換える系 (BTTA-DG の SincAdaptNet)。(i) は catastrophic forgetting と worst-subject 悪化の問題を、(ii) は既存 BN ベース強力 backbone (TCFormer, EEGNet) と構造的に切り離される問題を抱える。
>
> 我々は第三の路線として、**BN を凍結したまま、モデル内部の複数深度に配置した prototype を gated EMA で更新する**ことで、BN 更新を要さず、かつ既存 BN ベース backbone にそのまま装着できる OTTA 機構を提案する。Per-depth Tri-Lock (pmax × SAL × Energy) により誤 pseudo-label を hard abstain で排除し、T3A 型 cumulative prototype の EEG MI 失敗（T-TIME 報告）を回避する。

---

## PMANet について
- Deep research で引用された "PMANet" は複数の web 検索で特定できなかった（Zhao 2025 + EEG MI + prototype + TTA のヒットなし）
- 近接候補として **EDPNet / SST-DPN (Song+ 2024 arXiv:2407.03177)** が「prototype + EEG MI」で存在するが、**source 学習時の metric learning loss** であり **test-time adaptation ではない**
- TopA (Shen&Namiki, KBS 2025) も同様に **source architecture** のみ
- → **案 E'' の test-time prototype adaptation という主張は、現時点で EEG MI 領域で被っていない**と判断できる

## 残リスク
- MI-IASW の後続論文（本グループは活発）で "BN-frozen prototype EMA" が出る可能性。**実装完了時点で再度 arxiv + IJCNN/ICASSP 2026 を check** する。
- 非 EEG の Decoupled Prototype Learning 系 (arXiv:2401.08703 等) が類似構造を持っている可能性。**案 E'' 精査時に 1 本 spot check** する。
