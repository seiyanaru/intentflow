# 260511 第2回班ゼミ進捗

> Replay-Validated SafeCommit OTTA の提案・検証 + BTTA-DG (ICLR 2026) の論文紹介

> **今回ゼミの一行**
> 同一実験内の素の TCFormer (`source_only` = 82.72%) に対して、**Replay-SafeCommit は 83.41%（+0.69pp、HSC=0/9、WSD −0.35pp）** を達成。「適応したのに壊していない、しかも平均が上がった」を同一条件で示せた。
> 並行して、ICLR 2026 採択された **BTTA-DG**（Dirichlet 射影 + GMM Bayesian）が同じ問題に **完全 gradient-free** で挑んでおり、**memory-bank 思想を完全に共有している**。両者の関係を整理し、次の研究方針を立てる。

---

## アジェンダ (15 min)

1. **tcformer_replay_safe_otta の説明** — 自分の提案 (5-6 min)
2. **BTTA-DG の論文紹介** — 同問題を解く ICLR 2026 採択論文 (4-5 min)
3. **比較と今後の方針** — 両者を踏まえて何をするか (3-4 min)

---

# Part 1. tcformer_replay_safe_otta の説明

## 1.1 前回の到達点と未解決

前回ゼミ (260420) で hybrid@0.01 (shallow var 凍結) により 9 人平均 81.98%、HSC=0/9 を達成した。未決事項:
- shallow var が被験者依存になる機構
- 学習時に S2 を救う手がかりが弱い

→ Phase C (2b 横展開 + train-time 改良) に進む途中で **policy_safe_otta**（Policy + SafeCommit + OperatorBank）を実装したところ、新たな盲点を発見した。

## 1.2 観察した盲点：forward-only OTTA の即時 SafeCommit は機能していない

> **主張**: policy_safe_otta の即時 SafeCommit (margin / sal の before vs after 比較) は、**全 trial で `pred_changed = 0`** となる。評価関数が原理的に空。

### 根拠 (S2 診断, 260429)

- abstain 249/288 (86.5%)、commit 36 件
- commit が起きても **同じ trial の final logits は一般に変わらない**（`pred == original_pred` が全 288 trial）
- BN/prototype 更新の効果は「将来の trial」にしか表れない

**これは実装バグではなく、forward-only OTTA + immediate-gate SafeCommit という設計の構造的ミスマッチ**。既存研究 (TENT/EATA/SAR/MI-IASW/T3A) はすべて current trial の値、または batch を前提にしており、forward-only かつ batch=1 の Active BCI 設定では評価関数を設計し直す必要があった。

## 1.3 提案：Replay-Validated SafeCommit (二段ゲート)

> **核となる思想**: 即時の比較の代わりに、**過去 K trial の高信頼 replay buffer 上で simulated reward を計算する Tier 2 ゲート**を追加する。「commit の真の効果は将来の trial に出る」を replay buffer で前借り評価する。

### フロー

```
trial t 到着
   ↓
StateExtractor → RuleBasedPolicy: candidate operators 提案
   ↓
for each candidate:
   1. snapshot
   2. 仮 apply
   3. Tier 1 (即時)  margin / sal / proto_margin / energy / bn_drift
        fail → restore → 次 candidate
        pass → ↓
   4. Tier 2 (replay) buffer 32 trial を fused logits で forward
        sim_score = w_acc·Δreplay_acc + w_mar·Δreplay_margin + w_pro·Δreplay_proto_cos
        sim_score ≤ 0 → restore → 次 candidate
        sim_score > 0  → COMMIT
   ↓
現 trial も pmax > 0.85 AND SAL > 0.6 なら buffer に FIFO admit
```

### Replay buffer の admission 基準
- **capacity = 32**（FIFO）
- warmup: source train から **class-balanced 8 trial/class** で seed
- target session: **pmax > 0.85 AND SAL > 0.6** の trial を擬似ラベルと一緒に admit

## 1.4 結果（**主比較は同一実験内の `source_only` に置く**）

### 主比較表 (aug-True 9 被験者、1 seed、同一 source checkpoint・同一 split)

| 比較対象 | mean acc | Δ vs source_only | WSD (Δworst) | HSC@0.5pp | 解釈 |
|---|---:|---:|---:|---:|---|
| **source_only / 素の TCFormer** | **82.72** | — | — | 0/9 | 適応なし baseline |
| policy_safe_no_shallow | 83.14 | +0.42 | −0.35 | 0/9 | 安全だが replay より弱い |
| **replay_safe_uniform** | **83.41** | **+0.69** | **−0.35** | **0/9** | **best under safety constraint** |
| replay_h6_weighted | 83.22 | +0.50 | −0.35 | 0/9 | 強 source では過補正気味 |
| (参考) hybrid@0.01 前回ゼミ値 | 81.98 | +0.35 | −0.34 | 0/9 | 参考値。同一 regime での再評価は未完 |

### 主張（堅い言い方）

> 同一実験内の `source_only`、すなわち私の環境で学習した素の TCFormer は **82.72%** であった。Replay-SafeCommit は **83.41%** を達成し、平均精度を **+0.69pp** 改善した。同時に、HSC@0.5pp は **0/9**、WSD は **−0.35pp** に抑えられており、subject-level harm を増やさずに平均性能を改善できた。

### 補足（参考比較）

- 前回ゼミでの `hybrid@0.01` は **81.98%** であり、今回の replay_safe_uniform はその値も上回る。
- ただし **同一 regime での hybrid 再評価は未完了**のため、この比較は参考に留める。

### 5 seed × S7 (gain subject)

| variant | S7 mean ± std |
|---|---:|
| source_only | 90.51 ± 1.31 |
| **replay_safe_uniform** | **92.13 ± 0.91** |

→ replay は **variance reduction としても効く**（std −35%）。S7 gain は seed-robust。

## 1.5 言える / 言えない

**言える**:
- 同一実験内の `source_only` に対し replay_safe が +0.69pp、HSC=0/9、WSD −0.35pp。
- S7 gain は seed-robust（5 seed で +1.62 ± 0.91pp、std 35% 削減）。
- forward-only OTTA で commit の効果を「過去 K trial の replay」で評価する設計は先行研究に無い。

**言えない**:
- aug-True で 5 seed 検証済みではない（1 seed のみ）。
- hybrid@0.01 を同一 regime で再評価していない（参考比較に留まる）。
- hard subject (S2/S6) で robust gain は確認できていない。
- BCIC2b では未検証。

---

# Part 2. BTTA-DG の論文紹介

> **Bayesian Test-Time Adaptation via Dirichlet Feature Projection and GMM-Driven Inference for Motor Imagery EEG Decoding**
> Huan Luo, Na Lu et al. (Xi'an Jiaotong University), **ICLR 2026 採択**

## 2.1 問題意識（自分の問題意識と非常に近い）

- 既存 EEG-TTA は二択：
  - **gradient-based**（TENT / SAR / T-TIME）→ catastrophic forgetting + 計算コスト高
  - **statistics alignment**（BN-adapt / OTTA-Wimpff）→ 浅い特徴アラインメントしか出来ず deep の分布シフトを捉えられない
- 課題: **gradient-free でかつ deep distributional shift を捉える** TTA を作りたい

## 2.2 設計の三層

### (a) SincAdaptNet — 軽量 source エンコーダ
- 4 層のみ：Spat-Conv → **Sinc-Conv** → IncCh-Conv → Cls-Conv
- Sinc-Conv は学習可能な (low_cut, bandwidth) の 2 パラメータだけで bandpass を生成
- 生理学的に解釈可能 (μ/β/γ rhythm) + パラメータ数は EEGNet 並み (1.5K)

### (b) Dirichlet feature projection — 「分布の上の分布」へ射影

各 trial の T 時刻の softmax 出力 $X = [x_1, ..., x_T] \in \Delta^{|L|-1}$ を Dirichlet パラメータ $\alpha \in \mathbb{R}^{|L|}_+$ に **MLE で射影**：

$$\alpha = \arg\max_\alpha \sum_{j=1}^T \log \mathrm{Dir}(x_j; \alpha)$$

- $\alpha_i$ = クラス $i$ への確信の強さ
- $\alpha_0 = \sum_i \alpha_i$ = 不確実性（小さいほど uncertain）
- T 時刻にわたる予測の **時変的な確信度の凝縮** を、低次元（クラス数次元）の解釈可能なベクトルにする
- 計算は Minka の fixed-point iteration、5–10 反復で収束

### (c) GMM-driven Bayesian inference — gradient-free calibration

- 各クラスごとに **memory bank $M_y$** を持ち、過去の高信頼 trial の Dirichlet パラメータ $\alpha$ を保存
- memory bank に対し **per-class GMM** を当てはめる：

$$p_\mathrm{GMM}(\alpha | y) = \sum_{k=1}^K \pi_{y,k} \mathcal{N}(\alpha; \mu_{y,k}, \Sigma_{y,k})$$

- Bayes 則で deep model の事前 $p_\theta(y)$ と融合：

$$p_\mathrm{cal}(y | \alpha) = \frac{p_\mathrm{GMM}(\alpha | y) \cdot p_\theta(y)}{\sum_{y'} p_\mathrm{GMM}(\alpha | y') \cdot p_\theta(y')}$$

- 重み更新なし、出力 (posterior) のみキャリブレーション

### memory bank の管理（**replay buffer と完全に同じ思想**）

- 容量固定 (M=1000)、満杯時は古いものから捨てる FIFO
- admission 条件：`confidence ≥ τ_conf (0.596)` AND `entropy ≤ τ_ent (0.673)`
- 各 trial の寄与は最大 $1/M$ → 直近少数 trial が支配しない
- EM で GMM を再フィット（Dirichlet 次元が低いので軽量）

## 2.3 結果

| dataset | source_only | OTTA (Wimpff) | T-TIME | **BTTA-DG** |
|---|---:|---:|---:|---:|
| BNCI2014001 (4-class → 2-class) | 77.03 | 77.58 | 75.90 | **78.70** |
| BNCI2014002 | 78.05 | 78.29 | 76.85 | **80.29** |
| BNCI2015001 | 75.48 | 76.20 | 74.02 | **77.92** |
| SHU MI | 62.42 | 63.29 | 62.37 | **64.06** |

- **すべて SOTA**
- **計算時間 15.7 ms/trial**（OTTA-Wimpff 19.5ms より −19%、T-TIME 18.5ms より −15%）
- **gradient-free** なので catastrophic forgetting の心配なし

## 2.4 何が新しい / どこが効いている (ablation)

| 手法 | BNCI2014001 acc |
|---|---:|
| SincAdaptNet (Source Only) | 75.30 |
| + EA (Euclidean Alignment) | 77.03 |
| + EA + GMM only (時間平均 prob を直接 GMM) | 77.55 |
| + EA + Dirichlet projection (GMM 無し) | 77.61 |
| **+ EA + Dirichlet + GMM (Full)** | **78.70** |

- Dirichlet 射影単独でも +0.06pp（= 重要だが小さい）
- GMM 単独でも +0.52pp
- **両方組み合わせると +1.15pp** (相乗効果)

→ Dirichlet 射影で**確信度の構造を低次元に圧縮**し、GMM で**target ドメインでの確信度パターンの分布**を学ぶことが核心。

---

# Part 3. 比較と今後の方針

## 3.1 自分の提案 (replay-safe) と BTTA-DG の対比

| 軸 | tcformer_replay_safe_otta | BTTA-DG (ICLR 2026) |
|---|---|---|
| アプローチ | weight 更新 (BN/proto/bias) を replay 上の simulated reward でゲート | weight 更新無し、posterior を Bayesian fusion でキャリブレーション |
| memory の中身 | **生 EEG trial** + 擬似ラベル | **Dirichlet パラメータ** + 擬似ラベル |
| memory の per-class 化 | 単一 FIFO | per-class FIFO |
| admission 基準 | pmax > 0.85 AND SAL > 0.6 | confidence ≥ 0.596 AND entropy ≤ 0.673 |
| 評価関数 | Δreplay_acc / Δmargin / Δproto_cos | GMM likelihood × prior |
| 重み変更 | あり (BN/proto/bias commit) | なし (forward-only & freeze) |
| catastrophic forgetting risk | あり（Tier 1+2 で抑制） | 構造的に無し |
| 計算量 | 25 ms/trial | 15.7 ms/trial |
| Δ vs source（自分の環境） | **+0.69pp** (aug-True 9subj, 1 seed) | — (BNCI2014001 で +1.67pp 報告、ただし 4→2 class) |

> **両者は「memory bank で過去の高信頼 trial を保持し、batch=1 で動く gradient-free TTA」という思想を共有している**。違いは「memory に何を入れるか」と「memory をどう使うか」。

## 3.2 BTTA-DG が示唆する自分の研究の方向性

### 方向 A: Dirichlet 射影を sim_score の「特徴量」として導入

現状の sim_score は replay buffer 全件の生 forward だが、**Dirichlet パラメータを通せば情報を圧縮しつつ確信度の構造を保持**できる。

- 提案: replay buffer の各 trial を Dirichlet パラメータに射影しておき、sim_score を `Δ ‖α_after − α_before‖` や GMM likelihood の差で計算
- 利点: forward 数が減る (32 trial → 32 × Dirichlet, 軽量), 確信度シフトを直接捉える

### 方向 B: weight 更新を完全に手放して posterior calibration へ

BTTA-DG は **重みを一切動かさない**。replay_safe は BN/proto/bias を動かしている。

- 提案: replay_safe の operator commit を全部捨てて、BTTA-DG 流の posterior calibration を Tier 2 として置く
- 利点: catastrophic forgetting の構造的排除、Active BCI への適合性 (online safety contract)
- 懸念: 重み更新の自由度を失うことで gain が削られないか — 比較が必要

### 方向 C: 二層構造 (replay_safe = operator gate / BTTA-DG = output calibration)

- replay_safe は「適応 operator の選択 = 重み変更の制御」
- BTTA-DG は「出力 posterior の calibration = 出力の制御」
- **両者は competing ではなく complementary**

→ **replay_safe で BN/proto を慎重に動かしつつ、最終出力には BTTA-DG 流の Dirichlet+GMM Bayesian calibration をかける**ハイブリッド設計が、両者の強みを統合できる可能性が高い。

## 3.3 今後 2 週間 (260511 → 260525)

優先タスクは **「同一条件で全 variant を並べる」** 一枚比較表の完成。今は variant ごとに条件が違うので、強い主張が出来ていない。

| 優先 | タスク | 動機 | 採択判定 |
|---|---|---|---|
| ① | **同一条件 (aug-True、5 seed、同一 source checkpoint、同一 split) で 6 variant 比較**: `source_only` / `vanilla_otta` / `hybrid@0.01` / `policy_safe_no_shallow` / `replay_safe_uniform` / `replay_h6_weighted` | これで「素のTCFormerにも勝つ / 前回 hybrid にも勝つ / vanilla より安全」を同時に主張できる | replay_safe が `source_only` 比 +0.5pp 以上 ・ HSC 中央値 ≤ 1/9 ・ vanilla の WSD より良好 |
| ② | **BTTA-DG を本リポジトリで再現実装** (BNCI2014001 で 78.70% を target) | 比較に必要な baseline。SincAdaptNet 軽量なので工数低い | BNCI2014001 で論文値 ±1pp 内に到達 |
| ③ | **方向 C のハイブリッド派生実装** (replay_safe + BTTA-DG calibration) | 両者の競合ではなく合算で行けるか試す | replay_safe 単独を mean +0.5pp 以上上回る |
| ④ | (余裕) BCIC2b で 1 seed 再評価 + 可視化強化 | dataset 依存性 + ゼミ準備 | — |

### 最終目標の比較表（同一 seed、同一 augmentation、同一 source、同一 split）

| method | 目的 |
|---|---|
| `source_only` | 素の TCFormer (適応なし) — **主比較対象** |
| `vanilla_otta` | 前回ゼミの red bar (危険な既存 OTTA) |
| `hybrid@0.01` | 前回提案 (shallow var 凍結) |
| `policy_safe_no_shallow` | SafeCommit 単体 |
| `replay_safe_uniform` | **今回提案** |
| `replay_h6_weighted` | 重み付き replay (ablation) |
| `bttadg_reproduced` | (P2 完了後) BTTA-DG 再現 |
| `replay_safe + bttadg_calib` | (P3 完了後) ハイブリッド |

これが揃って初めて、

- 「素の TCFormer にも勝つ」（`source_only` 比 +Δpp）
- 「前回 hybrid にも勝つ」（`hybrid@0.01` 比 +Δpp）
- 「vanilla より安全」（`vanilla_otta` 比 HSC↓ / WSD↑）
- 「外部 SOTA (BTTA-DG) と同等以上」（BNCI2014001 の論文値 78.70 と同 dataset で比較）

を全部言える。

## 3.4 中期・長期

- **Active BCI への online 展開**: BTTA-DG は 15.7ms/trial で動く → online 化の参考になる。replay_safe の overhead (25ms) を縮める方向の手本。
- **subject-adaptive rule**（前回 Slide 18 の継続）: BTTA-DG は per-class memory bank を使っているが、per-subject ではない。本研究は一貫して per-subject polarity の問題を扱ってきており、ここに独自性がある。
- **論文化**: replay_safe 単独でも論文性はあるが、**BTTA-DG との比較を含めた hybrid 設計** にしたほうがインパクト大。

---

## 補足

### 想定 Q&A

| 質問 | 返答 |
|---|---|
| 主比較を hybrid@0.01 にしないのはなぜ | 同一 regime での再評価が未完なので fairness が確保できない。同一実験内の `source_only` 比較なら同一 seed・同一 augmentation・同一 source checkpoint・同一 split なので比較が直接的。 |
| BTTA-DG が出てしまったが、自分の研究の独自性は？ | **per-subject polarity の明示的扱い**（前回 Slide 8 の S2/S7 の符号反転）と **operator-level な適応制御**（複数 operator から選択）が BTTA-DG と異なる。また BTTA-DG は出力 calibration、replay_safe は重み制御で、両者は補完関係。 |
| BTTA-DG の方が結果良くない？ | データセットと規模が異なる（BTTA-DG は 4-class → 2-class、10 seed mean 78.70）。同 regime（4-class、aug-True）で再現実装してから直接比較する。 |
| seed=0 の S2 −3.12pp は何だった | seed アーティファクトだったと 5 seed で明らかになった。誠実に修正報告する。 |
| online への適合 | replay_safe は K=32 × 1.66s/trial = 25ms と Active BCI 250ms 制約内ではあるが、BTTA-DG の 15.7ms には劣る。Dirichlet 射影に置き換える方向 (3.2 方向 A) で短縮可能。 |

### 関連 artifact

#### コード
- `intentflow/offline/models/replay_safe_commit_otta.py`, `replay_buffer.py`, `tcformer_replay_safe_otta.py`

#### 結果ディレクトリ
- aug-True sweep: `intentflow/offline/results/c_aug_true_9subj_20260506_004923/`
- 5 seed sweep: `intentflow/offline/results/b_5seed_4subj_20260506_005153/`

#### 関連ドキュメント
- 前回ゼミ資料: `docs/research_progress/ゼミ資料/260420_narukawa.pdf`
- モデル全体図: `docs/research_progress/260508_models_overview.md`

#### 参考論文
- BTTA-DG: Luo, Lu et al. *Bayesian Test-Time Adaptation via Dirichlet Feature Projection and GMM-Driven Inference for Motor Imagery EEG Decoding*, ICLR 2026
- 関連: OTTA-Wimpff (BCI 2024), T-TIME (TBME 2023), MI-IASW
