# ETAGE: Enhanced Test Time Adaptation with Integrated Entropy and Gradient Norms

## 基本情報
- **著者**: Afshar Shamsi, Rejisa Becirovic, Ahmadreza Argha, et al.
- **発表年**: 2024
- **会議**: arXiv preprint
- **PDF**: [2409.09251v1.pdf](./2409.09251v1.pdf)
- **コード**: https://github.com/afsharshamsi/ETAGE

## 概要
エントロピー最小化に**勾配ノルム**と**PLPD（Pseudo Label Probability Difference）**を統合した改良版TTA手法。従来のエントロピーのみの信頼度評価の限界を克服。

## 主要貢献
1. **複合指標によるサンプル選択**: Entropy + Gradient Norm + PLPD
2. **バイアスシナリオへの対応**: エントロピーが信頼できない状況でも頑健
3. **ノイズへの過学習防止**: 高エントロピー＋高勾配ノルムのサンプルを除外

## 提案手法

### 3つの指標の統合

| 指標 | 意味 | 判断基準 |
|------|------|---------|
| Entropy | 予測の不確実性 | 高い = 不確実 |
| Gradient Norm | 更新の大きさ | 高い = 不安定な更新 |
| PLPD | 入力変形前後の予測差 | 大きい = 信頼できない |

### サンプル選択基準
```python
# 高エントロピー + 高勾配ノルム → 除外
if entropy > threshold_E and gradient_norm > threshold_G:
    skip(sample)  # ノイズへの過学習を防ぐ
else:
    update_model(sample)
```

## 従来手法（TENT, EATA, SAR）との比較
- TENT: エントロピーのみ → バイアスに弱い
- EATA: エントロピー + Fisher → 忘却防止は改善
- SAR: Sharpness-aware → 安定性向上
- **ETAGE**: 全てを統合 → 最も頑健

## 我々の研究との関連

### 複合ゲーティングの理論的根拠
- EATAGEの「Entropy + Gradient Norm」≈ 我々の「Entropy + pmax」
- 単一指標ではなく、複数の観点からサンプルの信頼性を評価

### pmax と Gradient Norm の違い
| 指標 | 計算タイミング | 意味 |
|------|--------------|------|
| pmax | 推論時（フォワードのみ） | モデルの確信度 |
| Gradient Norm | 適応時（バックプロップ必要） | 更新の大きさ |

→ pmaxはGradient Normより計算効率が良い

### 引用に使える知見
- 「エントロピーのみを信頼度指標として使用することには限界がある」
- 「複数の指標を組み合わせることで、より頑健なサンプル選択が可能」
