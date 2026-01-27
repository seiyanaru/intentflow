# Neuro-Gated OTTA: 実装計画書 (2026/01/27)

本ドキュメントは、教授からのフィードバック（新規性、ロバスト性、動的閾値）および計算効率の懸念に対応する、新しい適応手法「Neuro-Gated OTTA」の完全な実装計画である。

## 1. コアコンセプト: "Neuroscience-Informed Dynamic Adaptation"

既存のTTAは「確率分布」や「特徴空間の距離」といった数学的指標のみに依存しており、「脳のどこを見ているか（解剖学的妥当性）」を無視していた。これにより、瞬きなどのアーチファクト（ノイズ）に対して脆弱性があった。

本提案では、以下のロジックを導入することで、「脳科学的に正しい判断」をしている時のみ学習を進める。
> **「正しい場所（運動野）を見て自信がある時だけ、閾値を下げて積極的に適応する」**
> **「誤った場所（前頭葉・側頭葉）を見ている時は、自信があっても学習を拒否する」**

---

## 2. 軽量化技術: Parameter-Free Attention (ECA-Module)

従来のSE-Block（パラメータ数増大）の代わりに、**ECA (Efficient Channel Attention)** を採用し、計算コストを極限まで抑える。

### ECA-Blockの実装詳細
- **パラメータ数**: **3 (カーネルサイズのみ)**。計算負荷は無視できるレベル。
- **配置**: `MultiKernelConvBlock` の入力直後、またはConv層の並列ブランチとして配置。
- **機能**: 入力EEGの各タイムステップにおける「チャンネル重要度 $w \in \mathbb{R}^{C}$」を算出する。

```python
class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T]
        y = self.avg_pool(x)       # [B, C, 1]
        y = y.transpose(-1, -2)    # [B, 1, C]
        y = self.conv(y)           # [B, 1, C] (局所的なチャンネル相関を学習)
        y = y.transpose(-1, -2)    # [B, C, 1]
        return self.sigmoid(y)     # Importance weights [0, 1]
```

---

## 3. ロバスト化技術: Anatomically Guided Neural-Gating

算出されたチャンネル重み $w$ と、脳科学的知識に基づく「役割定義（Role）」を突き合わせ、適応の閾値を動的に制御する。

### 3.1 Neuro-Score ($S_{neuro}$) の定義
電極を以下の3つの役割に分類し、バイアス値を与える。

| グループ | 電極名 (例) | バイアス ($b$) | 意味 |
| :--- | :--- | :--- | :--- |
| **Motor (正解)** | C3, C4, Cz, FC3... | **+1.0** | 運動想起の信号源。ここを見るべき。 |
| **Noise (不正解)** | Fp1, Fp2, F7, T7... | **-1.0** | 瞬き・噛み締めのノイズ源。見てはいけない。 |
| **Neutral** | Pz, Oz... | **0.0** | どちらとも言えない。 |

スコア計算:
$$ S_{neuro} = \frac{\sum (w_c \cdot b_c)}{\sum w_c + \epsilon} $$

### 3.2 動的閾値決定ロジック
適応判定に使う `Pmax Threshold` （ベース値 $\tau_{base}=0.98$）を、Neuro-Scoreに応じて変動させる。

$$ \tau_{dynamic} = \tau_{base} - \alpha \cdot \tanh(S_{neuro}) $$

- **ケースA: 正しい推論 (Motor注目)**
    - $S_{neuro} > 0$ $\to$ $\tau_{dynamic} \approx 0.88$
    - **効果**: **学習加速**。「自信が多少低くても、見ている場所が正しいなら信頼する」。
- **ケースB: カンニング推論 (EOG注目)**
    - $S_{neuro} < 0$ $\to$ $\tau_{dynamic} > 1.0$
    - **効果**: **完全拒否**。「自信満々でも、瞬きを見ているなら絶対に学習させない（閾値1.0超え）」。

---

## 4. ユニバーサル対応: Initial-Time Automated Mapping

「あらゆるデータセット・あらゆるヘッドセット」に対応しつつ、リアルタイム性を損なわない仕組み。

### 4.1 自動マッピング (Auto-Mapping)
モデル内部に「標準10-20法辞書」を持ち、データセットのチャンネル名と照合して役割を自動決定する。

```python
STANDARD_ROLES = {
    "MOTOR": ["C3", "C4", "Cz", "CP3", "CP4", ...],
    "NOISE": ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "EOG", ...]
}
```

### 4.2 リアルタイム処理への最適化 (Init-Time Config)
推論時の遅延（文字列処理）を防ぐため、処理を2段階に分ける。

1.  **初期化時 (One-Time Calibration)**:
    - ヘッドセット接続時に1回だけ実行。
    - チャンネル名リストを受け取り、`motor_indices` (例: `[7, 11]`) と `noise_indices` の整数リストを生成・キャッシュする。
2.  **推論時 (Real-Time Inference)**:
    - キャッシュされた整数インデックスを使って、高速なテンソル操作のみを行う。
    - オーバーヘッドはほぼゼロ。

### 4.3 データセット別の挙動 (Fallback Strategy)
- **BCIC 2a / HGD / 一般的なヘッドセット**:
    - 上記のノイズ監視ロジックがフル機能で動作。
- **BCIC 2b (3ch: C3, Cz, C4のみ)**:
    - ノイズ電極が存在しないため、自動的に **「左右整合性チェック (Lateralization Check)」** モードに切り替わる。
    - 例: 「右手クラス」と予測したのに「右脳(C4)」が活性化していたらReject。

---

## 5. 新規性のまとめ (論文主張点)

1.  **世界初の "Anatomically-Gated TTA"**:
    データ分布だけでなく、脳の解剖学的構造を制約としてTTAに導入した点。
2.  **軽量かつ強力**:
    ECAモジュールの採用により、追加パラメータわずか「3」で、ディープラーニングのブラックボックス問題（Clever Hans）を解決した点。
3.  **ユニバーサルな実装**:
    特定のデータセットに依存せず、チャンネル構成を自動認識して最適化するプラグ・アンド・プレイ性を実現した点。
