# TCFormer_Hybrid モデル詳細

## 概要

`TCFormer_Hybrid` は、EEG信号の分類タスクにおいて、従来の **TCFormer** (Temporal Convolutional Transformer) の強力な特徴抽出能力と、**Test-Time Training (TTT)** による動的適応能力を融合させたモデルです。

最大の特徴は、**固定されたSelf-Attention（普遍的な特徴表現）** と **適応的なTTTアダプター（個人差への適応）** を**並列**に配置した「ハイブリッドエンコーダ」構造にあります。これにより、安定した性能を維持しつつ、テストデータ（被験者ごとの分布）に柔軟に適応することが可能です。

### 最新の改良点（v2.0）

1. **エントロピー駆動型動的ゲーティング**: モデルの予測不確実性に基づいてTTT適応のオン/オフを自動制御
2. **内部ループ勾配クリッピング**: EEGノイズによる勾配爆発を防止
3. **2-Pass Forward アーキテクチャ**: 適応前の状態を見て適応強度を決定

---

## アーキテクチャ詳細

モデル全体のデータフローは以下の通りです：
```
Input (EEG) -> [Multi-Kernel Conv Block] -> [Hybrid Encoder] -> [TCN Head] -> Output (Class Prob)
```

各コンポーネント、特に **Hybrid Encoder** の内部ロジックについて詳述します。

### 1. Multi-Kernel Conv Block (特徴抽出)
*   **入力**: 生のEEG信号 $X \in \mathbb{R}^{B \times C \times T}$ ($B$:バッチサイズ, $C$:チャンネル数, $T$:時間長)
*   **処理**: 複数の異なるカーネルサイズ ($K_1, K_2, ...$) を持つ1D畳み込み層を並列に適用し、出力を結合します。
*   **出力**: 特徴マップ $H_{conv} \in \mathbb{R}^{B \times T' \times D_{model}}$
    *   時間次元 $T$ はプーリングにより縮小され、チャンネル次元はモデル次元 $D_{model}$ に拡張されます。

### 2. Hybrid Encoder (特徴変換・適応)
このブロックが本モデルの核心です。入力特徴 $H$ に対して、以下の計算を行います。

#### A. 並列処理構造 (Parallel Hybrid Block)
通常のTransformerブロックが $H + \text{Attention}(H)$ を計算するのに対し、Hybridブロックは以下のように計算します：

$$ H' = H + \text{Attention}(\text{Norm}_1(H)) + \alpha \cdot \text{TTT\_Adapter}(\text{Norm}_1(H)) $$
$$ H_{out} = H' + \text{MLP}(\text{Norm}_2(H')) $$

*   **Main Path: Self-Attention (Fixed)**
    *   事前学習済みの重み $W_Q, W_K, W_V$ を使用し、テスト時には**固定**されます。
    *   入力シーケンス全体の広域的な依存関係（Global Context）を捉え、被験者に共通する普遍的な特徴を抽出します。
    
*   **Adapter Path: TTT Adapter (Adaptive)**
    *   テストデータを用いて、その場で重みを更新しながら変換を行うパスです。

*   **統合と $\alpha$ (Gating Factor)**
    *   Attention出力とAdapter出力は、同じ次元 $D_{model}$ を持ち、**要素ごとの加算 (Element-wise Sum)** によって統合されます。
    *   $\alpha$ の決定方法は2種類あります（後述）。

#### B. TTT Adapter の内部ロジック
計算コストを抑えつつ適応能力を持たせるため、ボトルネック構造を採用しています。

1.  **次元圧縮 (Down-projection)**:
    *   入力次元 $D_{model}$ を $D_{adapter} = r \times D_{model}$ に圧縮します（$r$ は `adapter_ratio`、例: 0.25）。
    *   $X_{low} = \sigma(X W_{down})$

2.  **Test-Time Training (TTT) Layer**:
    *   圧縮された特徴 $X_{low}$ を用いて、内部状態（重み $W_{hidden}$）を更新しながら出力 $Y_{low}$ を計算します。
    *   **更新則 (Dual Form)**:
        $$ W_{t} = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t) $$
        ここで $\ell$ は自己教師あり損失（再構成誤差）です。この更新により、モデルは「現在の入力データの分布」に適応します。
    *   **特徴変換**:
        $$ y_t = f(x_t; W_t) $$
        更新された重みを用いて特徴を変換するため、出力 $Y_{low}$ は入力データに適応した表現となります。

3.  **次元復元 (Up-projection)**:
    *   適応後の特徴 $Y_{low}$ を元の次元 $D_{model}$ に戻します。
    *   $Y_{out} = \text{Norm}(Y_{low}) W_{up}$
    *   ここで、Zero-Init戦略の一環として、$W_{up}$ はゼロで初期化されることが一般的で、これにより初期段階でのAdapter出力は完全にゼロになります。

### 3. TCN Head (分類)
*   **入力**: ハイブリッドエンコーダからの出力 $H_{out}$
*   **処理**: 時間方向の情報を集約するTemporal Convolutional Network (TCN)。
*   **出力**: クラス分類のロジット（確率）。

---

## エントロピー駆動型動的ゲーティング

### 動機：なぜ動的制御が必要か？

初期のHybridモデルでは、$\alpha$ は学習可能な**静的スカラー**でした。しかし、以下の問題が発生しました：

| 被験者 | Base | Hybrid (静的α) | 問題 |
|--------|------|----------------|------|
| S1 | 91.4% | 77.6% | **-13.8%** 低下 |
| S2 | 63.8% | 74.1% | +10.3% 改善 |

**原因分析**：
- **S1**: Baseモデルで既に高精度 → 決定境界は最適に近い
- TTT適応が決定境界を**不必要に動かし**、かえって精度低下

**理想的な動作**：
- 自信がある時（S1）→ TTT **OFF**
- 迷っている時（S2）→ TTT **ON**

### 予測不確実性の指標：シャノンエントロピー

予測確率分布 $p = \text{softmax}(\text{logits})$ に対するエントロピー：

$$ H(p) = -\sum_{i=1}^{K} p_i \log p_i $$

| エントロピー値 | 意味 | 予測分布例 |
|---------------|------|-----------|
| $H \approx 0$ | **自信あり** | [0.95, 0.05] |
| $H \approx \log K$ | **迷い** | [0.5, 0.5] |

### 2-Pass Forward アーキテクチャ

```python
# forward() メソッド内の処理フロー

if use_dynamic_gating and gating_mode == "entropy":
    # Pass 1: Attention のみで予測 → エントロピー計算
    x_main = hybrid_encoder(x, enable_ttt=False)  # TTT無効
    logits_main = tcn_head(x_main)
    p = softmax(logits_main)
    entropy = -(p * log(p)).sum(dim=-1)  # [Batch]
    
    # エントロピーからα、lr_scaleを計算
    alpha = entropy_alpha_gate(entropy)
    lr_scale = entropy_lr_gate(entropy)
    
    # Pass 2: 決定したα、lr_scaleでハイブリッド推論
    x = hybrid_encoder(x, gate_alpha=alpha, lr_scale=lr_scale, enable_ttt=True)
else:
    # 従来パス（静的α）
    x = hybrid_encoder(x)
```

**ポイント**：
- 2-Pass は**推論時のみ**適用（訓練時はデフォルトで無効）
- `entropy_gating_in_train=True` で訓練時にも有効化可能

### EntropyGating モジュール

```python
class EntropyGating(nn.Module):
    def forward(self, entropy: Tensor) -> Tensor:
        # Dead-zone: 閾値以下は完全にOFF
        h = entropy - threshold
        mask = (h > 0).float()
        h = clamp(h, min=0.0)
        
        # シグモイドゲート
        z = w * h + b
        return sigmoid(z) * mask * alpha_max
```

**数式**：
$$ \alpha = \sigma(w \cdot \max(H - \tau, 0) + b) \cdot \mathbb{1}_{H > \tau} \cdot \alpha_{\max} $$

**パラメータ**：
| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `entropy_threshold` | Dead-zone閾値 $\tau$ | 0.95 |
| `alpha_max` | α上限 | 0.5 |
| `lr_scale_max` | 学習率スケール上限 | 0.5 |
| `entropy_alpha_init_w` | ゲート傾き初期値 | 2.0 |
| `entropy_alpha_init_b` | ゲートバイアス初期値 | -3.0 |

**Dead-zone の効果**：
```
     α (適応強度)
     ↑
α_max|           ___________
     |         /
     |        /
     |       /
   0 |______/________________→ H (エントロピー)
           τ
      Dead-zone
      (安定した被験者を保護)
```

---

## 内部ループ勾配クリッピング

### 目的

EEGデータには筋電ノイズ、眼球運動などのアーティファクトが含まれており、これらが**異常に大きな勾配**を生成する可能性があります。

$$ \|\nabla_W \mathcal{L}\| \gg \text{正常範囲} \quad \Rightarrow \quad \text{重みの爆発} $$

### 解決策：L2ノルムクリッピング

勾配の**方向は維持**しつつ、**大きさのみを制限**：

$$ \hat{g} = \begin{cases} g & \text{if } \|g\|_2 \leq c \\ \displaystyle\frac{c}{\|g\|_2} \cdot g & \text{if } \|g\|_2 > c \end{cases} $$

### 実装

```python
# ttt_layer.py 内の compute_mini_batch() 関数

# 勾配計算
grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

# 損失スケーリング（オプション）
if loss_scale != 1.0:
    grad_l_wrt_Z1 = grad_l_wrt_Z1 * loss_scale

# 勾配クリッピング（各トークン×ヘッド単位）
if grad_clip > 0.0:
    # grad_l_wrt_Z1: [Batch, Heads, MiniBatch, HeadDim]
    norm = torch.linalg.vector_norm(grad_l_wrt_Z1, ord=2, dim=-1, keepdim=True)
    scale = (grad_clip / (norm + eps)).clamp(max=1.0)
    grad_l_wrt_Z1 = grad_l_wrt_Z1 * scale
```

**ポイント**：
- バッチ全体ではなく、**各ヘッド・各トークン単位**でノルムを計算
- `dim=-1`（HeadDim軸）でノルムを計算 → 局所的なノイズを個別に抑制

### パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `ttt_grad_clip` | クリッピング閾値 $c$ | 1.0 |
| `ttt_loss_scale` | 損失スケーリング係数 | 1.0 |

---

## アンカリング戦略（正則化）

### 目的

TTT適応が進むにつれて、重みが初期値から大きく離れ、**普遍的な特徴を忘れる**リスクがあります。

### 解決策：初期重みへの正則化

$$ W_t = (1 - \lambda_{\text{reg}}) W_{t-1} + \lambda_{\text{reg}} W_0 $$

- $W_0$: 学習済み初期重み（普遍的特徴）
- $\lambda_{\text{reg}}$: 正則化係数（`ttt_reg_lambda`）

### アンカリングスケールモード

```python
# ttt_layer.py 内

mode = config.ttt_anchor_scale_mode  # "none" or "same"

if mode == "same" and lr_scale is not None:
    # lr_scale でアンカリングもスケール
    alpha = effective_base_lr * reg_lambda
else:
    # アンカリングは固定（推奨）
    alpha = base_lr * reg_lambda
```

| モード | 動作 | 推奨度 |
|--------|------|--------|
| `"none"` | アンカリングは**固定** | ✅ 推奨 |
| `"same"` | lr_scale と連動 | ⚠️ ドリフト悪化の恐れ |

**推奨理由**：迷っている被験者（高エントロピー）で lr_scale が大きくなった時、アンカリングも弱めるとドリフトが悪化する。

---

## 設定パラメータ一覧

`configs/tcformer_hybrid/tcformer_hybrid.yaml` にて設定可能です。

```yaml
model_kwargs:
  # 基本設定
  adapter_ratio: 0.25          # アダプターの圧縮率
  trans_depth: 2               # Hybrid Encoderの層数
  
  # 動的ゲーティング
  use_dynamic_gating: true     # 動的ゲーティング有効化
  gating_mode: "entropy"       # "entropy" or "feature_stats"
  entropy_threshold: 0.95      # Dead-zone閾値
  alpha_max: 0.5               # α上限
  lr_scale_max: 0.5            # 学習率スケール上限
  entropy_gating_in_train: false  # 訓練時の2-Pass有効化
  
  # TTT設定
  ttt_config:
    base_lr: 0.1               # TTT内部学習率
    ttt_reg_lambda: 0.05       # アンカリング正則化係数
    ttt_grad_clip: 1.0         # 勾配クリッピング閾値
    ttt_loss_scale: 0.1        # 損失スケーリング
    ttt_anchor_scale_mode: "none"  # アンカリングスケールモード
```

---

## 評価指標と可視化のロジック

本プロジェクトで生成される図表（`fig/` フォルダ内）の生成ロジックと見方についての解説です。

### 1. エントロピー分布 (Entropy Distribution)
**「AIがどれくらい自信を持って答えているか？」** を数値化して分布図にしたものです。

*   **計算ロジック**:
    1.  **ロジット (Logits)**: モデルの最終出力（確率になる前の生スコア）。
    2.  **確率への変換 (Softmax)**: 
        $$ p_i = \frac{e^{z_i}}{\sum e^{z_k}} $$
    3.  **エントロピー計算 (Shannon Entropy)**:
        $$ H = -\sum p_i \log(p_i) $$
        
*   **値の意味**:
    *   **低い ($H \approx 0$)**: 「自信満々」。特定のクラスの確率が非常に高い状態。
    *   **高い**: 「迷っている」。どのクラスの確率も似たり寄ったりな状態。

*   **グラフの見方**:
    *   **左側（0付近）に山がある**: モデルは迷いなく明確な判断を下しています（理想的）。
    *   **右側に広がっている**: モデルは判断に迷いが生じています（不確実性が高い）。

### 2. t-SNE (特徴量空間の可視化)
高次元の特徴ベクトルを2次元に圧縮し、モデルがデータをどのように「見て」いるかを可視化します。

*   **ロジック**: 分類ヘッド直前の特徴ベクトル（通常100次元以上）を入力とし、似た特徴を持つデータは近くに、異なるデータは遠くになるように配置します。
*   **見方**:
    *   **きれいな塊（クラスタ）ができている**: クラス分離がうまくいっています。
    *   **色が混ざり合っている**: モデルがクラスの違いを識別できていません。

---

## なぜこの構造が有効なのか？ (Design Rationale)

1.  **安定性と可塑性のバランス (Stability-Plasticity Balance)**:
    *   **Attentionパス (安定性)**: 学習データ全体から得られた知識を保持し、未知のデータに対しても最低限の性能を保証します（破滅的忘却の防止）。
    *   **TTTパス (可塑性)**: テスト時のドメインシフト（被験者ごとの脳波パターンの違いなど）に対して、パラメータを動的に変化させて対応します。

2.  **Zero-Initによる学習の安定化**:
    *   $\alpha=0$ から開始することで、初期学習段階でのTTTによる不安定な挙動がメインパスの学習を阻害するのを防ぎます。

3.  **エントロピーゲーティングによる選択的適応**:
    *   全サンプルに一律に適応するのではなく、**必要な時だけ適応**することで、安定した被験者の精度低下を防止します。

4.  **効率的な適応**:
    *   巨大なSelf-Attention層全体を適応させるのではなく、軽量なAdapter部分のみを適応させることで、計算コストとメモリ使用量を大幅に削減しています。

---

## 想定される疑問点 (Q&A)

### Q1: なぜTTTモデル単体ではなく、Hybrid構造にする必要があるのですか？
**A:** EEGのようなノイズが多く個人差の大きいデータでは、TTTモデル単体だとテストデータへの**過剰適合（Overfitting）**や、事前学習で得た知識の**破滅的忘却（Catastrophic Forgetting）**が起きやすいためです。
Hybrid構造にすることで、Self-Attentionパスが「変わらない普遍的な特徴」を維持し、TTTパスが「個人差や変動」のみを補正するという役割分担ができ、安定性と適応性の両立が可能になります。

### Q2: エントロピーゲーティングと静的αの違いは何ですか？
**A:**

| 方式 | α の決定 | 適用 |
|------|---------|------|
| 静的α | 学習で固定値を決定 | 全サンプルに同じα |
| エントロピーゲーティング | 推論時にエントロピーから計算 | サンプルごとに異なるα |

エントロピーゲーティングにより、自信がある予測（低エントロピー）ではTTTをスキップし、迷っている予測（高エントロピー）でのみTTTを適用できます。

### Q3: なぜ2-Passが必要なのですか？1-Passではできないのですか？
**A:** エントロピーを計算するには**予測確率分布**が必要ですが、これはTTT適応後の出力から計算するとTTTの影響を受けてしまいます。
「適応**前**の状態を見て、適応の**強度**を決める」という因果関係を構築するために、2-Pass（Pass 1で状態確認、Pass 2で適応）が必要です。

### Q4: Dead-zone（不感帯）の役割は何ですか？
**A:** 閾値 $\tau$ 以下のエントロピーでは $\alpha = 0$ に**強制的に**設定することで、安定した被験者（S1など）を**物理的に保護**します。
シグモイド関数だけでは完全に0にならないため、ハードなマスク（`mask = (h > 0).float()`）を導入しています。

### Q5: 推論速度（レイテンシ）への影響はどの程度ですか？
**A:** 
- **静的α**: 1-Pass のため最速（約7-8ms）
- **エントロピーゲーティング**: 2-Pass のため約2倍（約13-15ms）

リアルタイムBCIでは許容範囲内ですが、レイテンシが重要な場合は静的αを使用することも選択肢です。

### Q6: 訓練時に2-Passを使わないのはなぜですか？
**A:** 訓練時に2-Passを使うと以下の問題があります：
1. **計算コストが2倍**になる
2. **Dropoutの影響**でPass 1とPass 2で異なる出力になり、学習が不安定化

デフォルトでは推論時のみ2-Passを適用し、`entropy_gating_in_train=True` で訓練時にも有効化できます（実験的機能）。

---

## 実験結果サマリー

### 全体比較

| モデル | 平均精度 | 標準偏差 |
|--------|----------|----------|
| Base (TCFormer) | 84.68% | ±9.25 |
| Hybrid (Static α) | 82.57% | ±6.21 |
| **Hybrid + Entropy Gating** | **83.91%** | **±5.86** |

### 主な改善点

1. **標準偏差の減少**: 9.25 → 5.86（被験者間のばらつきが減少）
2. **困難な被験者の改善**: S2: 63.8% → 75.9%（+12.1%）
3. **安定した被験者の保護**: S7: 94.8% → 94.8%（維持）

### 最良パラメータ設定

```yaml
ttt_config:
  base_lr: 0.1
  ttt_loss_scale: 0.1
  ttt_reg_lambda: 0.05
  ttt_grad_clip: 1.0

entropy_threshold: 0.95
alpha_max: 0.5
lr_scale_max: 0.5
```
