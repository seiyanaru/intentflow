# EATA: Efficient Test-Time Model Adaptation without Forgetting

## 基本情報
- **著者**: Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, et al.
- **発表年**: 2022
- **会議**: ICML 2022 (International Conference on Machine Learning)
- **PDF**: [niu22a.pdf](./niu22a.pdf)
- **コード**: https://github.com/mr-eggplant/EATA

## 概要
テスト時適応（TTA）において、**計算効率**と**忘却防止**の2つの課題を同時に解決する手法。全てのサンプルで適応するのではなく、信頼性のあるサンプルのみを選択して更新する。

## 主要貢献
1. **Active Sample Selection**: 高エントロピーサンプルはノイズ勾配を生むため除外し、信頼性のある非冗長なサンプルのみを選択
2. **Fisher Regularizer**: 重要なモデルパラメータの急激な変化を制約し、破滅的忘却を防止
3. **効率性**: 全サンプルでのバックプロパゲーションを回避

## 提案手法

### 1. Active Sample Selection（能動的サンプル選択）
```python
# 高エントロピーサンプルをフィルタリング
if entropy(prediction) < threshold_E:
    # 信頼できるサンプル → 適応に使用
    update_model(sample)
else:
    # 高エントロピー → スキップ（ノイズ勾配を防ぐ）
    skip(sample)
```

### 2. Fisher Regularizer
- Fisher情報行列を使用してパラメータの重要度を推定
- 重要なパラメータの変化を制約
- ソースドメインの知識を保持

## 実験結果
- CIFAR-10-C, ImageNet-C, ImageNet-R で検証
- TENTより高精度かつ低計算コスト
- 忘却問題を大幅に軽減

## 我々の研究との関連

### 直接的な関連
1. **pmax filteringの理論的根拠**
   - EATAの「高エントロピーサンプルを除外」という考え方は、我々のpmax filteringと一致
   - pmax < threshold のサンプルのみで適応 ≈ 信頼できるサンプルのみで適応

2. **崩壊防止の正則化**
   - EATAのFisher正則化 ≈ TTT内部のアンカー正則化（ttt_reg_lambda）
   - 両方とも元のパラメータからの逸脱を制約

### 引用に使える知見
- 「全てのサンプルが適応に等しく貢献するわけではない」
- 「高エントロピーサンプルはノイズ勾配を生み、モデルを乱す可能性がある」
