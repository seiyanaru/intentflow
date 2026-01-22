# MI-IASW: Test-Time Adaptation for Cross-Subject Motor Imagery EEG Classification

## 基本情報
- **著者**: 不明（IEEE論文）
- **発表年**: 2024-2025
- **会議/ジャーナル**: IEEE (Waseda University Library経由でアクセス)
- **PDF**: [Test-Time_Adaptation_for_Cross-Subject_Motor_Imagery_...pdf](./Test-Time_Adaptation_for_Cross-Subject_Motor_Imagery_EEG_Classification_Using_Information-Aggregation_and_Source-Guided_Weighting.pdf)

## 概要
**MI-EEG専用のTTAフレームワーク**。Information-Aggregation (IA) と Source-Guided Weighting (SW) を統合し、MI-EEG特有の課題に対応。

## MI-EEGにおけるTTAの課題

### 1. 限られたデータ
- MI-EEG収集は集中力を要し、試行数が少ない
- リアルタイム推論では限られたデータでの即時適応が必要

### 2. 信頼できない擬似ラベル（⚠️重要）
> **MI-EEGモデルは過剰確信（overconfident）な予測を生成しやすい**
> - 低エントロピー・高信頼度の評価が多すぎる
> - ターゲットモデルの擬似ラベル評価能力が低下

## 提案手法（MI-IASW）

### Information Aggregation (IA)

#### 1. MABN (Mixed and Adaptive Batch Normalization)
```python
# ソース統計量とターゲット統計量を混合
μ_mixed = α * μ_source + (1-α) * μ_target
σ_mixed = α * σ_source + (1-α) * σ_target
```

#### 2. Weight Aggregation (WA)
- 限られたデータでの汎化性能を向上
- 複数のモデル重みを集約

### Source-Guided Weighting (SW)

#### 1. Source-Guided Pseudo-label Evaluation
```python
# ソースセンターを基準にターゲット予測を評価
SAL = alignment_score(target_prediction, source_center)
# SAL: Source Alignment Level
```

#### 2. Class-Aware Weighting (CAW)
```python
# SALに基づき各サンプルの重みを計算
sample_weight = f(SAL)
loss = weighted_cross_entropy(predictions, pseudo_labels, sample_weight)
```

## 実験設定
- **Leave-One-Subject-Out (LOSO)**: 我々と同じ設定
- ソース被験者で訓練 → ターゲット被験者でオンライン適応

## 我々の研究との関連

### 🔴 過剰確信問題（最重要）
> 「ターゲットモデルは低エントロピー・高信頼度の評価を多数生成するため、擬似ラベルの評価能力が低下する」

**我々のpmax filteringとの関連**:
- MI-EEGモデルは高いpmaxを出しやすい
- だからこそ、pmax < threshold でフィルタリングすることに意義がある
- エントロピー閾値だけでは不十分

### ソースガイド評価
- ターゲットモデルの自己評価だけでは不十分
- ソース情報を活用した評価が有効
- 我々のアプローチでは、学習済みパラメータ（ソース知識）を保持するアンカー正則化が類似

### 引用に使える知見
- 「MI-EEGシナリオでは、ターゲットモデルはCV対応物と比較してさらに過剰確信な予測を生成する傾向がある」
- 「ターゲットモデルが多くの擬似ラベルに対して低エントロピー・高信頼度の評価を容易に生成するため、擬似ラベルの評価能力が低下する」
