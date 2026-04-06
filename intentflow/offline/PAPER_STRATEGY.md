# 論文戦略 — 合意版 v2 (2026-03-14)

## 主仮説

**All target samples should not be adapted equally.**

既存のOTTA手法（Tent, EATA, CoTTA等）はテスト時に全サンプルで適応を試みるが、
EEG-MI BCIでは非定常性・アーティファクト・個人差により、
適応が精度を悪化させる（negative transfer）ケースが頻発する。

提案手法は **risk-aware gated OTTA** として、
confidence（pmax）, source agreement（SAL）, OOD detection（energy）の
三重ゲーティングにより「いつ適応しないか（when not to adapt）」を判断する。

## 問題設定の固定

**Cross-session OTTA on MI-BCI** を主問題に固定する。

- Source: session_T（訓練セッション）で学習したモデル
- Target: session_E（評価セッション）に対してオンライン順序で逐次的に適応
- 被験者は共通（同一被験者の異なるセッション間のドリフトに対処）
- Cross-subject plug-and-play OTTA は本論文のスコープ外とし、future work に回す

理由: cross-session は BCI の実用シナリオとして最も自然であり、
session間のnon-stationarityはMI-BCIの中核課題として広く認知されている。
cross-subject を混ぜると問題設定が曖昧になり、査読で突かれるリスクが高い。

## 核となる区別

- **Update abstention**（更新の棄却）: ゲート不通過時にモデル更新を凍結し、元のモデルで予測
- **Prediction rejection**（予測の棄却）: 予測自体を出さない（EMBC 2017のreject option）
- 本手法は前者。両者は直交的に組み合わせ可能。論文で明示すること。

## 論文構成

### Title候補
- "Risk-Aware Test-Time Adaptation for Motor Imagery BCI: Learning When Not to Adapt"
- "When Not to Adapt: Safe Test-Time Adaptation for Motor Imagery Brain-Computer Interfaces"

### Problem
既存のonline/calibration-free BCI適応は平均精度を改善するが、
一部の被験者/試行でnegative transferを引き起こす。
（予備実験では無条件OTTAで特定被験者にbaseline比の悪化が観測されている。
ただし単一seed由来のため、本実験では5-seed / outer fold で再確認する。）

### Method
Risk-aware OTTA with tri-lock gating:
1. **Pmax gate**: softmax確信度による低確信サンプルの除外
2. **SAL gate**: ソースプロトタイプとの整合性チェック
3. **Energy gate**: OOD検出によるdistribution shift検出
   - 校正方法: source-train logitsからenergy score分布を計算し、
     指定分位点（e.g., 95th percentile）を閾値として設定。
     temperature-scaled energy E(x) = -T * logsumexp(logits/T) を使用。
     これにより閾値は source distribution に対して相対的に定義され、恣意性を低減。

Optional module:
- **Neuro-beta gate**: モーター領域チャネル注意に基づく生理学的ゲーティング
  - 残す条件: negative transfer率(NTR-S) or worst-subject dropの一貫した改善 + C3/C4 ERD/attention対応
  - 弱ければ切る（core paper = tri-lock, optional = neuro-beta）

### Evaluation axes（精度以外を主軸に）

**主指標（論文の主張を直接支える）:**
1. **Mean accuracy**: 全被験者平均の分類精度
2. **Negative transfer rate (subject-level, NTR-S)**: baseline比で精度が悪化した被験者の割合
   - 定義: NTR-S = |{s : acc_adapted(s) < acc_source(s)}| / |S|
   - 被験者単位を主指標とする（試行単位は粒度が細かすぎ、ノイズに弱い）
3. **Worst-subject delta (WSD)**: 最も悪化した被験者のbaseline比変化量
   - 定義: WSD = min_s (acc_adapted(s) - acc_source(s))
4. **Coverage-risk tradeoff curve**: 適応率（coverage）を横軸、NTR-SまたはWSDを縦軸にプロット
   - 閾値を変化させたときのcoverageと安全性のトレードオフを可視化

**補助指標（報告するが主張の中心にしない）:**
5. Coverage of adaptation: 全試行中で適応が実行された割合
6. Accuracy on accepted updates: 適応実行した試行での精度
   - 注意: 選別バイアスが強い。「簡単なサンプルだけ適応して精度が高い」と解釈される可能性あり。
     報告はするが、主軸に置かない。
7. Negative transfer rate (trial-level, NTR-T): 試行単位での悪化割合（補助参考）
8. Latency: 推論+適応の所要時間

### Claim
Safer adaptation with competitive mean accuracy.
「精度が少し上がる手法」ではなく「危険な適応を避ける手法」。

---

## 実験プロトコル

### 主プロトコル: Nested subject-wise CV on BCIC-IV 2a

3段階の被験者分割を明示する:

```
Outer fold k (k=1,2,3):
  ┌─────────────────────────────────────────────────────────────┐
  │ Source model training subjects (3 subjects)                  │
  │   → session_T で通常の教師あり学習                            │
  │   → ソースモデル M_k を得る                                   │
  ├─────────────────────────────────────────────────────────────┤
  │ Threshold selection subjects (3 subjects)                    │
  │   → M_k を session_E に適応しながら、閾値グリッドを探索        │
  │   → 最良の (pmax_th, sal_th, energy_quantile) を固定          │
  │   → この段階で閾値が決定。以降変更しない                       │
  ├─────────────────────────────────────────────────────────────┤
  │ Final test subjects (3 subjects)                             │
  │   → 固定閾値で session_E にOTTAを適用                         │
  │   → ここの結果のみを最終報告に使用                             │
  └─────────────────────────────────────────────────────────────┘
```

- 9被験者を3群に分割 × 3 fold（ローテーション）
- **後付けハイパラ最適化を完全排除**（TTAB準拠）
- 各foldのtest subjects結果を集約して最終指標を報告

### 統計検定

- **Paired Wilcoxon signed-rank test**: 被験者ペアでの比較（N=9、ノンパラメトリック）
- **Paired permutation test**: Wilcoxonの補強として
- **95% confidence interval**: bootstrap法（被験者リサンプリング, B=10000）
- 安全性主張は「差があるように見える」だけでは不十分。統計的有意差を必ず示す。
- 5-seedの結果も報告し、seed間の安定性を確認

### 比較手法（優先順）

| 優先度 | 手法 | 役割 | 参考文献 |
|--------|------|------|----------|
| 必須 | **Source-only** | ベースライン（適応なし） | — |
| 必須 | **Tent** | 無条件適応の危険性を示す | [Wang+ ICLR 2021](https://arxiv.org/abs/2006.10726) |
| 必須 | **EATA** | sample selection + anti-forgetting、最直接的な競合 | [Niu+ ICML 2022](https://proceedings.mlr.press/v162/niu22a.html) |
| 必須 | **提案法（Gated OTTA）** | tri-lock gating | — |
| 高 | **T-TIME** | BCI-TTA先行研究（実装コスト次第） | [Jiang+ IEEE TBME 2024](https://doi.org/10.1109/TBME.2023.3303289) |
| 中 | **CoTTA** | continual TTA（余力時） | [Wang+ CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Continual_Test-Time_Domain_Adaptation_CVPR_2022_paper.html) |

関連評価フレームワーク:
- [TTAB (Zhao+ ICML 2023)](https://proceedings.mlr.press/v202/zhao23d.html): TTA評価プロトコルの厳密性に関する指針

### データセット
1. **BCIC-IV 2a**（主評価、4クラスMI、22ch、9被験者）
2. **BCIC-IV 2b**（追加評価、2クラスMI、3ch、9被験者、再現性確認）
3. HGD or OpenBMI（さらに余力があれば）

### Stress test
1. **EMG artifact injection**（最優先）
   - band-limited high-frequency noise（20-100Hz帯域）or 実EMGテンプレート混合
   - 単純ガウシアンノイズは非現実的で弱い
   - SNR = {-5, 0, 5, 10} dB の複数条件
2. **Time-window misalignment**
   - 既存結果で時間窓ずれに敏感であることが判明済み（0.0-4.0s → 0.5-4.5sで-10.72%）
   - 「危険な条件で適応を止める」ストーリーと自然に接続
   - onset shift = {-0.5, -0.25, 0, +0.25, +0.5} s の複数条件

### 閾値転送実験（追加実験として）
- 生の閾値ではなく、source train分布に対する分位点 / z-score正規化後の相対リスク閾値を転送
- cross-dataset threshold transferとして報告
- 主プロトコルではない（2aと2bでクラス数・チャネル数が異なるため）

---

## Ablation Study設計

| 条件 | pmax | SAL | energy | neuro-beta | 目的 |
|------|------|-----|--------|------------|------|
| Source-only | - | - | - | - | ベースライン |
| Tent (全適応) | - | - | - | - | 無条件適応の危険性 |
| pmax-only | ✓ | - | - | - | 確信度ゲート単体の効果 |
| pmax + SAL | ✓ | ✓ | - | - | dual-lock |
| pmax + SAL + energy (tri-lock) | ✓ | ✓ | ✓ | - | **core提案** |
| tri-lock + neuro-beta | ✓ | ✓ | ✓ | ✓ | optional module |

**各条件で報告する指標**: mean acc, NTR-S, WSD, coverage, 95% CI

---

## 実装優先順位

### Phase 1: SAL sweep完了 → 基本効果確認（現在進行中）
- SAL = {0.35, 0.40, 0.45, 0.50} の結果で hard gate の negative transfer 抑制効果を確認
- 推定完了: 2026-03-15 朝

### Phase 2: Tent実装 + 比較基盤構築
- Tent（BN統計量のエントロピー最小化）を数十行で実装
- Source-only vs Tent vs 提案法 の3条件比較を走らせる
- 評価指標を harm-aware metrics (NTR-S, WSD, coverage) に拡張

### Phase 3: EATA実装
- sample-efficient TTA + anti-forgetting の再実装
- 最も直接的な競合手法との比較

### Phase 4: Nested CV プロトコル実装
- 3-fold subject split の自動化
- source training / threshold selection / final test の分離
- 統計検定（Wilcoxon, permutation test, bootstrap CI）の実装

### Phase 5: データセット追加
- BCIC-IV 2b を追加（既存パイプライン対応済み）

### Phase 6: Stress test
- EMG artifact injection
- Time-window misalignment

### Phase 7: 論文ドラフト
- Introduction → Related Work → Method → Experiments → Discussion

---

## 投稿先

| 投稿先 | 適合度 | 備考 |
|--------|--------|------|
| **IEEE TNSRE** | ★★★★★ | BCI + safe adaptation ストーリーに最適 |
| **Journal of Neural Engineering** | ★★★★★ | 同上 |
| NeurIPS/ICML Workshop | ★★★★☆ | BCI workshop があれば |
| AAAI/IJCAI | ★★★☆☆ | 追加データセット + benchmark設計必須 |
| NeurIPS/ICML main | ★★☆☆☆ | 現状では遠い |
| EMBC | ★★★★☆ | 安全な投稿先、ただしインパクト低め |

---

## 注意事項

- 「safe BCI」とまでは言わない。「safe adaptation」「risk-aware adaptation」に留める
- 2a/2bは同期型MI。「誤作動しない安全性」まで強く言うには弱い
- 論文ではまず safe adaptation に留め、future work で非同期BCIへの拡張を述べる
- オンライン順序を壊した後付け最適化はTTABで批判済み。厳密に避ける
- Subject 2の-7.99%悪化は単一seed由来。論文本文ではpreliminary observation扱いとし、
  5-seed / outer folds での再確認後に前面に出す
- negative transfer rate は被験者単位（NTR-S）を主指標。試行単位（NTR-T）は補助
- accuracy on accepted updates は選別バイアスに注意。報告するが主張の中心にしない
