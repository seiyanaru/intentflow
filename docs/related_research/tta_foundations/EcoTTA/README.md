# EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled Regularization

## 基本情報
- **著者**: Junha Song, Jungsoo Lee, In So Kweon, Sungha Choi
- **発表年**: 2023
- **会議**: CVPR 2023 (Computer Vision and Pattern Recognition)
- **所属**: Qualcomm AI Research, KAIST
- **PDF**: [Song_EcoTTA_...pdf](./Song_EcoTTA_Memory-Efficient_Continual_Test-Time_Adaptation_via_Self-Distilled_Regularization_CVPR_2023_paper.pdf)

## 概要
**メモリ効率**と**継続的適応**を両立するTTA手法。エッジデバイス向けに設計され、CoTTAと比較して86%のメモリ削減を達成。

## 主要貢献
1. **Lightweight Meta Networks**: 元のネットワークを凍結し、軽量なメタネットワークのみを更新
2. **Self-distilled Regularization**: メタネットワークの出力が元のネットワークから逸脱しないよう制約
3. **メモリ86%削減**: ResNet-50でCoTTAと比較

## 提案手法

### 1. Lightweight Meta Networks
```
[凍結されたメインネット] → [軽量メタネット] → [出力]
                              ↑
                        ここだけ更新
```

**メモリ削減の理由**:
- バックプロパゲーション用の中間活性化（activation）がメモリのボトルネック
- メタネットのみを更新することで、活性化の保存量を大幅削減

### 2. Self-distilled Regularization
```python
# メタネットの出力が凍結ネットの出力から逸脱しないよう制約
loss = entropy_loss(meta_output) + λ * KL(meta_output, frozen_output)
```

**効果**:
- ソースドメインの学習済み知識を保持
- 誤差累積と破滅的忘却を防止
- 追加メモリなしで実現

## 実験結果
| 手法 | メモリ (ResNet-50) | CIFAR-10-C Error |
|------|-------------------|------------------|
| TENT | 300 MB | 中程度 |
| CoTTA | 1350 MB | 良い |
| **EcoTTA (K=4)** | **200 MB** | **最良** |

## 我々の研究との関連

### TTT層のアダプター方式
- EcoTTAの「メインネット凍結＋メタネット更新」
- 我々の「Attention凍結＋TTT層更新」（ttt_drop_prob=1.0時）
- 同様の設計思想

### 自己蒸留正則化
- EcoTTAの自己蒸留 ≈ TTT内部のアンカー正則化
- 両方とも「元の出力から逸脱しすぎない」ことを目的

### エッジデバイス向け設計
- BCIデバイスもメモリ制約あり
- EcoTTAのメモリ効率は参考になる

### 引用に使える知見
- 「パラメータではなく、活性化がメモリのボトルネック」
- 「凍結ネットワーク＋軽量アダプターで効率的な適応が可能」
