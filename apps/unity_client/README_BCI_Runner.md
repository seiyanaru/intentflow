# Unity BCI Runner ゲーム

TCFormer Hybrid + TTT モデルを使用したリアルタイムBCI制御のランニングゲームです。

## システム構成

```
┌─────────────────────────────┐     WebSocket      ┌─────────────────────────────┐
│   Python TTT Broadcaster    │ ←───────────────→  │      Unity Runner Game      │
│                             │                    │                             │
│  - OnlineTCFormerWrapper    │  IntentMessage:    │  - MiSource (WebSocket)     │
│  - GDF/LSL Input            │  {"type":"intent", │  - RunnerInputAdapter       │
│  - Prediction Loop          │   "intent":"left", │  - RunnerController         │
│                             │   "conf":0.85}     │  - BciFeedback UI           │
└─────────────────────────────┘                    └─────────────────────────────┘
```

## クイックスタート

### 1. Python サーバー起動

```bash
cd /workspace-cloud/seiya.narukawa/intentflow

# conda環境を有効化
source ~/anaconda3/etc/profile.d/conda.sh
conda activate intentflow

# TTT Broadcaster を起動
python intentflow/online/server/ttt_broadcaster.py \
    --checkpoint intentflow/offline/results/paper_experiments/bcic2a/20260104_105140/data/checkpoints/subject_1_model.ckpt \
    --subject 1 \
    --session T \
    --port 8765 \
    --trial_interval 2.0 \
    --two_class_only
```

### 2. Unity ゲーム起動

1. Unity Hub で `apps/unity_client` プロジェクトを開く
2. `Assets/Scenes/Runner3Lane_MI.unity` シーンを開く
3. Play ボタンを押す
4. 自動的に Python サーバーに接続

## コマンドラインオプション (Python)

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--checkpoint` | (必須) | 学習済みモデルのパス |
| `--subject` | 1 | 被験者ID (1-9) |
| `--session` | T | セッション (T: 訓練, E: 評価) |
| `--port` | 8765 | WebSocketポート |
| `--trial_interval` | 2.0 | 予測間隔 (秒) |
| `--confidence_threshold` | 0.3 | 最小信頼度閾値 |
| `--two_class_only` | True | 左右のみ送信 |
| `--loop` | False | トライアルを繰り返す |
| `--gpu_id` | 0 | GPU ID (-1でCPU) |

## Unity側の設定

### MiSource コンポーネント

| プロパティ | デフォルト | 説明 |
|-----------|-----------|------|
| Server Url | ws://localhost:8765 | サーバーURL |
| Auto Connect | true | 自動接続 |
| Reconnect Interval | 3.0 | 再接続間隔 (秒) |
| Log Messages | true | デバッグログ出力 |

### BciFeedback コンポーネント

予測結果をHUDに表示します：
- 予測クラス (← LEFT / RIGHT →)
- 信頼度バー (0-100%)
- 接続状態インジケーター

## ゲーム操作

| 入力 | アクション |
|-----|----------|
| 左手 MI | 左レーンへ移動 |
| 右手 MI | 右レーンへ移動 |
| (足/舌) | 無視 (2クラスモード) |

## トラブルシューティング

### 接続できない

1. Python サーバーが起動しているか確認
2. ポート番号が一致しているか確認
3. ファイアウォール設定を確認

### 予測が遅い

1. `--trial_interval` を小さくする
2. GPU が使用されているか確認 (`--gpu_id 0`)

### 精度が低い

1. 別の被験者のモデルを試す
2. `--confidence_threshold` を調整

## ファイル構成

```
apps/unity_client/
├── Assets/
│   ├── Scenes/
│   │   ├── Runner3Lane_MI.unity    # MI制御シーン
│   │   └── Runner3Lane_Keyboard.unity
│   └── Scripts/
│       ├── Inputs/MI/
│       │   └── MiSource.cs         # WebSocket MI入力
│       └── Tasks/
│           ├── Core/
│           │   └── RunnerController.cs
│           └── UI/
│               ├── Hud.cs
│               └── BciFeedback.cs  # BCI フィードバック

intentflow/online/server/
└── ttt_broadcaster.py              # TTT予測サーバー
```

## 開発者向け

### IntentMessage プロトコル

Python → Unity へ送信されるJSON形式:

```json
{
    "type": "intent",
    "intent": "left",    // "left" | "right" | "idle"
    "conf": 0.85,        // 0.0 - 1.0
    "ts": 1234567890.123,
    "protocol_version": 1
}
```

### クラスマッピング

| BCIC2a クラス | クラスID | Intent |
|--------------|---------|--------|
| left_hand | 0 | "left" |
| right_hand | 1 | "right" |
| feet | 2 | "idle" |
| tongue | 3 | "idle" |


