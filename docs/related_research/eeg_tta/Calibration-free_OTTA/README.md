# Calibration-free Online Test-time Adaptation for EEG Motor Imagery Decoding

## 基本情報
- **著者**: Martin Wimpff, Mario Döbler, Bin Yang
- **発表年**: 2024
- **所属**: University of Stuttgart
- **PDF**: [2311.18520v2.pdf](./2311.18520v2.pdf)

## 概要
**キャリブレーション不要**のオンラインTTAによるEEG運動イメージ分類。プライバシー保護（ソースデータアクセス不要）とキャリブレーション不要を両立。

## 設定

### Cross-session vs Cross-subject
| 設定 | 説明 | 難易度 |
|------|------|-------|
| Cross-session | 同一被験者の異なるセッション | 中 |
| Cross-subject | 異なる被験者 | 高 |

### Online Test-Time Adaptation (OTTA)
```
[ソースで訓練] → [ソースデータ破棄] → [ターゲットでオンライン適応 + 推論]
                    ↑
              プライバシー保護
```

## 提案手法

### 1. Alignment（アライメント）
```python
# 共分散行列を参照行列に正規化
X_aligned = reference_matrix^(-1/2) @ covariance_matrix^(1/2) @ X
```

- **Euclidean Alignment (EA)**: ユークリッド空間でのアライメント
- **Riemannian Alignment**: リーマン多様体上でのアライメント

### 2. Adaptive Batch Normalization
```python
# BatchNorm統計量をターゲットドメインに適応
μ_target, σ_target = compute_statistics(target_batch)
normalized = (x - μ_target) / σ_target
```

### 3. Entropy Minimization (TENT, EATA, SAR)
```python
# 予測エントロピーを最小化
loss = -sum(p * log(p))  # Entropy
update_bn_parameters(loss)  # BNパラメータのみ更新
```

## 実験結果
- BNCI2014001 (BCIC 2a と同等) と Cho2017 データセット
- 3つの設定: Cross-session, Cross-subject, Mixed
- 適応手法により最新の結果を達成

## 転移学習手法の比較

| アプローチ | ソースデータ | ターゲットデータ | 適応タイミング |
|-----------|-----------|--------------|-------------|
| Fine-tuning | オフライン利用 | ラベルあり | オフライン |
| UDA | 利用 | ラベルなし | 学習時 |
| DG | 利用 | なし | なし |
| **OTTA** | **なし** | **オンラインで取得** | **推論時** |

## 我々の研究との関連

### EEG-TTAの先行研究
- 我々のアプローチの最も直接的な関連研究
- 同じ問題設定（Cross-subject MI-EEG）
- 同じ目標（キャリブレーション不要）

### キャリブレーション不要の重要性
> 「BCIのキャリブレーションデータ収集は時間とコストがかかる」
> 「身体障害者にとって特に負担が大きい」

### プライバシー保護
- ソースデータへのアクセス不要
- 臨床応用において重要

### 引用に使える知見
- 「Inter-subject differences により大きな分布シフトが発生する」
- 「BCI分野ではDG（データ拡張、アライメント）が一般的だが、OTTAは新しい方向」
