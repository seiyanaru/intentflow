# TCFormer: Temporal Convolutional Transformer for EEG Motor Imagery Classification

## 基本情報
- **著者**: Altaheri et al.
- **発表年**: 2025
- **ジャーナル**: Scientific Reports (Nature)
- **PDF**: [s41598-025-16219-7.pdf](./s41598-025-16219-7.pdf)
- **コード**: https://github.com/altaheri/TCFormer

## 概要
**Multi-kernel CNN + Transformer + TCN** を統合したEEG運動イメージ分類モデル。BCIC IV-2a で 84.79% の精度を達成（SOTA）。

## アーキテクチャ

```
[EEG Input] → [Multi-Kernel Conv Block] → [Transformer Encoder] → [TCN Head] → [Classification]
                     ↓                           ↓                     ↓
                局所特徴抽出              グローバルコンテキスト      長距離時間パターン
```

### 1. Multi-Kernel Convolution Block
- 複数のカーネルサイズ: Kc = {20, 32, 64}
- 異なる時間スケールの特徴を抽出
- Grouped Squeeze-and-Excitation Attention

### 2. Transformer Encoder
- **Grouped-Query Attention (GQA)**: MHAとMQAの中間
- **Rotary Positional Embedding (RoPE)**: 相対位置関係をエンコード
- Pre-norm configuration

### 3. TCN Head
- 時間的因果関係を保持
- 残差接続付き

## 主要なハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|---|------|
| F1 | 32 | Temporal filters |
| D | 2 | Depth multiplier |
| q_heads | 4 | Query heads |
| kv_heads | 2 | Key-value groups |
| trans_depth | 2 | Transformer layers |
| lr | 0.0009 | Learning rate |
| warmup_epochs | 20 | Warmup epochs |

## 実験結果

### BCIC IV-2a (Within-subject)
| モデル | 精度 | Kappa |
|--------|------|-------|
| EEGNet | 72.6% | - |
| EEGConformer | 75.4% | - |
| CTNet | 81.9% | - |
| ATCNet | 83.8% | - |
| **TCFormer** | **84.79%** | **0.80** |

### Ablation Study
| Variant | 精度 |
|---------|-----|
| CNN-only | 80.14% |
| CNN + TCN | 83.26% |
| **CNN + Transformer + TCN** | **84.79%** |

## 我々の研究との関連

### ベースモデル
- TCFormer Hybrid の基盤
- 論文設定の再現が重要（lr=0.0009, kv_heads=2, scheduler=True）

### GQAの効率性
- MHAより計算効率が良い
- TTT層追加時のオーバーヘッド軽減に貢献

### データ拡張（S&R）
- Segmentation & Reconstruction
- 8分割してランダムに再構成
- オーバーフィッティング防止

## 論文設定の再現に必要な設定

```yaml
lr: 0.0009
kv_heads: 2  # G = H/2
scheduler: True
warmup_epochs: 20
F1: 32
D: 2
temp_kernel_lengths: [20, 32, 64]
```
