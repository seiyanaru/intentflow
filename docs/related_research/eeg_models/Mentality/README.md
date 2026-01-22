# Mentality: A Mamba-based EEG Foundation Model

## 基本情報
- **タイトル**: Mentality: A Mamba-based Approach for EEG Foundation Models
- **発表年**: 2024
- **会議**: ICLR 2024 (International Conference on Learning Representations)
- **PDF**: [38_Mentality_A_Mamba_based_App.pdf](./38_Mentality_A_Mamba_based_App.pdf)

## 概要
EEGのための**Foundation Model**を目指した、Mambaベースのアーキテクチャ。自己教師あり再構成タスクでの事前学習後、発作検出タスクで評価。TUSZ v2.0.1データセットで AUROC 0.72 を達成。

## 主要貢献
1. **Mamba-based EEG Foundation Model**: EEG解析のための大規模事前学習モデル
2. **Self-supervised Pretraining**: 再構成タスクによる事前学習
3. **Spectral Loss**: 周波数ドメインでの損失関数

## アーキテクチャ

```
[EEG Input] 
    ↓
[1D CNN (kernel=100)] ← 周波数フィルタ学習（〜50Hz）
    ↓
[Channel Mixing Linear Layer]
    ↓
[Mamba Blocks] ← 時間的ダイナミクスの SSM 表現
    ↓
[U-Net Downsampling]
    ↓
[Mamba Blocks (Bottleneck)]
    ↓
[U-Net Upsampling + Skip Connections]
    ↓
[Reconstruction Output]
```

### インスピレーション
- **EEGNet**: 軽量CNN、周波数フィルタ初期層
- **SaShiMi**: U-Net + S4 (Mamba前身)
- **U-Net**: エンコーダ-デコーダ構造、スキップ接続

## データセット
- **Temple University Hospital EEG Seizure Corpus (TUSZ) v2.0.1**
- 訓練: 579患者, 2,138発作イベント
- テスト: 43患者, 469発作イベント
- 19チャンネル（10-20システム）、200Hz

## 実験結果

| タスク | 指標 | 値 |
|--------|------|---|
| 再構成 (Pretrain) | MSE | 0.0063 |
| 発作検出 | AUROC (事前学習後) | **0.72** |
| 発作検出 | AUROC (スクラッチ) | 0.64 |

→ **事前学習により +0.08 AUROC 改善**

## 我々の研究との関連

### Foundation Model アプローチ
- Mentality: 自己教師あり再構成 → 下流タスク
- 我々: 教師あり学習 → TTTによるオンライン適応
- **異なるアプローチだが、分布シフト対応という目的は共通**

### Mamba と Attention の比較
- Mentality: Mamba (SSM) による長距離依存性モデリング
- TCFormer: Grouped-Query Attention (Transformer)
- **どちらも時間的ダイナミクスをモデリングするが、メカニズムが異なる**

### 解釈可能性の課題
> 「Mambaブロックはその表現の解釈可能性を分析する方法がない」

- 我々のTTTアプローチでも同様の課題
- alpha / entropy ゲーティングが一種の解釈可能性を提供

### 引用に使える知見
- 「EEGは高次元、非線形、高い被験者間変動を持つ」
- 「従来の機械学習モデルはEEGの時空間ダイナミクスを効果的に捉えられない」
- 「Foundation Modelは様々なタスクに汎化するための大規模DLモデル」
