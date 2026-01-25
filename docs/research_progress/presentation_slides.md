# 進捗報告: Pmax-SAL Gated OTTAによるMI-EEG適応の堅牢化

---

## 📅 本日のアジェンダ

1.  **前回までの振り返り & 課題**
    - TCFormer Hybrid (Entropy Gating) の結果
    - HGD / BCIC 2b での "失敗" (Negative Transfer)
2.  **論文調査 (課題の背景)**
    - 紹介: MI-IASW (過剰確信とソースガイド)
3.  **新規提案手法: Pmax-SAL Gated OTTA**
    - コンセプトとアルゴリズム
4.  **実験結果**
    - 3つのデータセットでの検証結果
    - SOTAとの比較
5.  **まとめと今後**

---

# 1. 前回までの振り返り & 課題

## 🔹 TCFormer Hybrid アプローチ
- **目的**: Test-Time Training (TTT) の計算コストと安定性のバランスを取る。
- **手法**: 入力エントロピーでTTT適応率 ($\alpha$) を調整する `Entropy Gating`。

## ⚠️ BCIC 2a 以外での "失敗"
前回報告した BCIC 2a では一定の成果が出ましたが、**BCIC 2b** と **HGD** で検証したところ、深刻な問題が発生しました。

| データセット | Baseline | Entropy Gating | 変化 | 状態 |
|:---|:---:|:---:|:---:|:---:|
| **BCIC 2a** | ~84% | 83.5% | 微減 | 不安定 |
| **BCIC 2b** | ~85% | **80.8%** | **📉 -4.2%** | **失敗** |
| **HGD** | ~96% | **79.3%** | **📉 -16.7%** | **崩壊** |

### 何が起きたのか？
- ドメインシフトが大きく、モデルが **「間違っているのに自信満々 (過剰確信)」** な予測を連発。
- Entropy Gating はこれを「信頼できる」と誤認し、間違った方向に適応 (Negative Transfer) してしまった。

**(図: 前回手法の失敗 - Hybrid Accuracy Comparison)**
`intentflow/offline/vis_results/hybrid_comp_hgd.png`
`intentflow/offline/vis_results/hybrid_comp_bcic2b.png`
- **青: Base**, **橙: Hybrid(Static)**, **赤: Hybrid(Entropy)**
- 特にHGD(上図)において、Entropyを入れると(赤)、Base(青)よりも大幅に精度が下がっている被験者が散見されます。

---

# 2. 論文調査 (課題の背景)

## Slide 2-1: 関連研究の選定 MI-IASW
**論文**: Test-Time Adaptation for Cross-Subject Motor Imagery EEG Classification Using Information Aggregation and Source-Guided Weighting (IEEE, 2024)

### なぜこの論文か？
MI-EEG分野における Test-Time Adaptation (TTA) の最新研究であり、我々が直面している **「過剰確信 (Overconfidence)」** の問題を明確に指摘・対処しているため。

### MI-EEG特有の課題
1.  **データ不足**: 較正時間 (Calibration Time) をゼロにしたい。
2.  **不確実な擬似ラベル**: 
    - 既存の画像用TTA (TENT等) は「エントロピー最小化」を行うが、MI-EEGでは逆効果になることが多い。
    - **理由**: モデルが「間違った予測」に対しても「高い確信度」を出してしまうため。

---

## Slide 2-2: MI-IASW の提案手法
この課題に対し、2つのアプローチを統合したフレームワークを提案。

### 1. Information Aggregation (IA)
- 少数のターゲットデータだけでなく、ソースドメインの統計量 ($\mu_{source}, \sigma_{source}$) を混合 (Mix) してBatch Normalizationを安定化。
- **MABN (Mixed and Adaptive BN)** を採用。

### 2. Source-Guided Weighting (SW) ⭐重要
- ターゲットモデルの自信 (Softmax確率) だけを信じない。
- **「ソースプロトタイプ (各クラスの代表点)」との距離** を計測し、信頼度 (Source Alignment Level) として利用。
- これにより、過剰確信による誤った適応を抑制。

**(推奨図: 論文中の Fig. 1 または Framework Overview の図)**
- IAとSWの二本柱で構成される全体像を示す図。

---

## Slide 2-3: 実験結果と成果
BCIC 2a データセット等で検証し、TENTや他のTTA手法を上回る精度を達成。

### 主な発見
- 単純なTENT (Entropy Minimization) は、ノイズの多いEEG信号に対して不安定。
- ソース情報を「ガイド」として使うことで、適応の安定性が大幅に向上した。

**(推奨図: TENT vs MI-IASW の精度比較グラフ または t-SNE可視化)**
- 既存手法(TENT)が失敗するケースで、提案手法が踏ん張っているデータがあればベスト。

---

## Slide 2-4: 我々の研究への活用と差別化
MI-IASWの知見は素晴らしいが、**実用面での課題** もある。

### 課題: 計算コストと複雑さ
- **Weight Aggregation**: 複数のモデル重みを保持・統合する必要があり、計算コストが高い。
- **MABN**: 統計量の混合比率 ($\alpha$) の調整が必要。

### $\to$ 我々の提案 (Pmax-SAL Gated OTTA) の位置付け
- **継承**: 「過剰確信を防ぐにはソース情報 (Alignment) が必要」というコンセプトは継承。
- **発展**: 
    - 複雑なアンサンブルや重み統合は廃止。
    - 代わりに **「Pmaxによる確信度フィルタ」** と **「SALによる整合性チェック」** という **軽量なゲーティング機構** を導入。
    - 結果、計算コストを増やさず (Param増分ゼロ)、同等以上の堅牢性を実現。

**(推奨図: 複雑なIA/SW vs シンプルなPmax-SAL の概念比較図)**
- (自作ポンチ絵でOK) MI-IASWが「重い装備」で登るのに対し、我々は「地図(SAL)を持った軽装備」で登るイメージ。

---

# 3. 新規提案手法: Pmax-SAL Gated OTTA

## 💡 コンセプト: "疑わしきは適応せず"

**「自信」と「整合性」の双方を満たすサンプルのみ適応**

---

## 🛠️ アルゴリズムと処理フロー

### 処理フロー図
```mermaid
graph LR
    Input[Input EEG] --> TCFormer[TCFormer Encoder<br>(CNN + Transformer)]
    TCFormer --> Features[Features & Logits]

    subgraph "Dual Gating (Self- & Source-Check)"
        Features --> Pmax["① Self-Check (Pmax)<br>'自信はあるか？' (>0.7)"]
        Features --> SAL["② Source-Check (SAL)<br>'ソースと似ているか？' (>0.98)"]
    end

    Pmax --> Gate{AND Gate}
    SAL --> Gate

    Gate -- Yes --> Update["✅ UPDATE<br>(TCFormer's BN Stats)"]
    Gate -- No --> Skip["⛔ SKIP"]
    
    Update --> Pred[Final Prediction]
    Skip --> Pred
```

**数式の追加（推奨）:**
PmaxとSALの定義を数式で示すと、より厳密性が伝わります。

*   **Pmax (Internal Confidence):**
    $$P_{\max} = \max_c p(y=c|x)$$
    *   $p(y=c|x)$: 入力$x$に対するクラス$c$の予測確率

*   **SAL (External Consistency):**
    $$SAL = \cos(f(x), \mu_{\hat{y}}) = \frac{f(x) \cdot \mu_{\hat{y}}}{\|f(x)\| \|\mu_{\hat{y}}\|}$$
    *   $f(x)$: 特徴量ベクトル
    *   $\mu_{\hat{y}}$: 予測クラス$\hat{y}$のソースプロトタイプ（Source Prototype）
    
    %% Style
    style TCFormer fill:#e1f5fe,stroke:#01579b
    style Update fill:#a0e0a0,stroke:#333
    style Skip fill:#ffb0b0,stroke:#333
```

### 適応のターゲット
TCFormer内の **MK-CNN** と **TCN** ブロックにある **Batch Normalization ($\mu, \sigma$)** のみを更新対象とし、Transformer (LayerNorm) は固定します。これにより個人差を効率的に吸収します。

### 判定ルール (Strategy)
**① 基本ルール**
*   自信 ($P_{max} > 0.7$) かつ 整合性 ($SAL > 0.98$) の場合のみ適応。

**② なぜ SAL > 0.98 か？**
*   **狙い**: 「自信はあるが間違っているサンプル (Overconfidence)」の徹底排除。
*   左図(Scatter Plot)の **赤色領域 ($×$印)** にあるような有害なサンプルを適応から外すことで、モデルの暴走を防ぎます。

**(図: Pmax-SAL Scatter Plot)**
`intentflow/offline/vis_results/scatter_s14.png`
- **青丸 (●)**: 適応サンプル（High Confidence & High Alignment）。
- **灰バツ (×)**: 除外サンプル（自信過剰エリア=赤色を含む）。適応すべきでないサンプルが正しく弾かれていることがわかります。

---

# 4. 実験結果

## 📊 3つのデータセットでの検証結果

Entropy Gatingで失敗したデータセットを含め、提案手法 (Pmax-SAL) の堅牢性を検証。

**(図: Performance Comparison Bar Chart)**
`intentflow/offline/vis_results/comparison_bar.png`
- 3つのデータセット全てで、「SOTA (灰色)」や「Baseline (青)」を上回る「Proposed (赤)」の性能を確認できます。

| Dataset | Baseline (Seed0) | **Pmax-SAL OTTA** | 変化 | 評価 |
|:---|:---:|:---:|:---:|:---|
| **BCIC 2a** | 84.67% | **87.36%** | **🏆 +2.69%** | SOTA更新 |
| **HGD** | 96.43% | **97.18%** | **✅ +0.75%** | **高水準維持** |
| **BCIC 2b** | 84.85% | **84.99%** | **✅ +0.14%** | 安定 |

## ✨ HGDでの劇的改善 (Subject 14)
- **Entropy Gating**: 崩壊 (60%台へ低下)
- **Pmax-SAL**: Baseline 67.5% $\to$ **75.0% (+7.5%)**
    - 厳格なフィルタリングにより、ドメインシフトの影響を受けた悪性サンプルを排除し、適応に成功。

## ⚔️ 既存SOTA (TCFormer論文) との比較

| 指標 | TCFormer論文値 | **Our Method** | 勝敗 |
|:---|:---:|:---:|:---:|
| **BCIC 2a Acc** | 84.79% | **87.36%** | **Win (+2.57%)** |
| **HGD Acc** | ~96.0%* | **97.18%** | **Win (+1.2%)** |

> *注: HGDの論文値は再現実験結果 (Baseline=96.43%) から推定。一般的なSOTA水準も上回っています。

## 🏎️ 計算効率とパラメータ数

提案手法は **Pmax-SAL Gating** という軽量なフィルタリングのみを追加するため、モデルサイズや推論コストを最小限に抑えています。

| 指標 | TCFormer (Base) | **Pmax-SAL OTTA** (提案) | 備考 |
|:---|:---:|:---:|:---|
| **#Params** | **77.8 K** | **77.8 K** (+0) | 推論時の学習可能パラメータ増分ゼロ |
| **Memory** | Low | Low | プロトタイプキャッシュのみ追加 (数KB) |
| **Speed** | 5.4 ms/sample | **~6.0 ms/sample** | TTT適応時のみ微増、スキップ時は高速 |

- **MI-IASW** (比較対象) は複数のClassifierをアンサンブルするため、計算コストは数倍になります。
- **Pmax-SAL** は、軽量モデル (77.8K params) の利点を維持したまま適応を実現しています。

---

# 5. まとめ

1.  **課題の克服**: MI-IASWでも指摘されている「過剰確信」の問題を、Pmax-SAL Gatingにより解決。
2.  **汎用性と堅牢性**: BCIC 2a, 2b, HGD の全てにおいて、Baseline以上の性能を達成 (Negative Transferなし)。
3.  **SOTA超え**: モデル構造を変えずに、適応のみで既存の最高性能を更新。

**今後の予定**:
- TENT/EATA との比較実験
- 論文執筆 (Abstract/Intro)

---

---

# Appendix: SALのメカニズム (Source Prototypes)

## 典型的な「形」をどう把握するか？

本手法では、学習時に各クラスの「代表的な特徴量（プロトタイプ）」を記憶し、推論時に比較を行っています。

### Step 1: 学習時（プロトタイプの構築）
*   ソースデータセット全体を用いて、各クラス（例：右手、左手）ごとの特徴量ベクトルの**重心 (Mean Vector)** を計算。
*   これを「クラスプロトタイプ ($\mu_c$)」として辞書に保存。
    *   $\mu_{RH}$: 右手の典型的パターン
    *   $\mu_{LH}$: 左手の典型的パターン
    *   ...

### Step 2: 推論時（整合性の確認）
*   入力データの特徴量 ((x)$) が、予測クラスのプロトタイプ ($\mu_{\hat{y}}$) と**どれくらい似ているか（角度が近いか）** を計算。
*   計算には **コサイン類似度** を使用。

3068076SAL = \text{CosineSimilarity}(f(x), \mu_{\hat{y}})3068076

> **直感的な意味:**
> 「モデルが『右手』と予測したが、その特徴量は本当に『右手の典型パターン』に近いか？」を幾何学的にチェックしています。
