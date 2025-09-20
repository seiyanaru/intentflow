# intentflow

## 1. 概要
intentflow は Motor Imagery (MI) EEG 信号をリアルタイム推論に乗せ、Unity/Unreal などのゲームエンジンへ安全に制御信号を届けるためのフレームワークです。Offline Training・Online Inference・Application Bridge の 3 レイヤを明確に分離し、信号処理とモデル更新を高速に反復できるようスキャフォールドしています。

## 2. ディレクトリ構造
```
intentflow/
├─ README.md
├─ pyproject.toml
├─ environment.yml
├─ .gitignore
├─ .gitattributes
├─ .pre-commit-config.yaml
├─ configs/
│   ├─ data.yaml
│   ├─ preprocess.yaml
│   ├─ offline.yaml
│   ├─ online.yaml
│   ├─ adapt.yaml
│   └─ ports.yaml
├─ data/
│   ├─ raw/.gitkeep
│   ├─ interim/.gitkeep
│   ├─ processed/.gitkeep
│   └─ external/.gitkeep
├─ notebooks/
├─ intentflow/
│   ├─ __init__.py
│   ├─ core/
│   │   ├─ io_lsl.py
│   │   ├─ io_unicorn.py
│   │   ├─ filters.py
│   │   ├─ features.py
│   │   ├─ schema.py
│   │   └─ utils.py
│   ├─ offline/
│   │   ├─ models/
│   │   │   ├─ eegencoder.py
│   │   │   ├─ ciacnet.py
│   │   │   ├─ mamba_encoder.py
│   │   │   ├─ eegnet.py
│   │   │   └─ registry.py
│   │   ├─ train.py
│   │   ├─ eval.py
│   │   ├─ export_onnx.py
│   │   └─ distill.py
│   ├─ online/
│   │   ├─ inference/
│   │   │   ├─ tcnet_head.py
│   │   │   ├─ runner.py
│   │   │   └─ stabilizer.py
│   │   ├─ adapt/
│   │   │   ├─ otta.py
│   │   │   ├─ head_ft.py
│   │   │   ├─ errp_detector.py
│   │   │   └─ bandit.py
│   │   ├─ server/
│   │   │   ├─ app.py
│   │   │   └─ clients.py
│   │   ├─ recorder/
│   │   │   ├─ logger.py
│   │   │   └─ replay.py
│   │   └─ cli.py
│   └─ datasets/
│       ├─ bci_iv_2a.py
│       ├─ bci_iv_2b.py
│       └─ unicorn_live.py
├─ apps/
│   ├─ unity_client/
│   └─ unreal_client/
├─ dashboards/
│   └─ app.py
├─ scripts/
│   ├─ simulate.py
│   ├─ calibrate.py
│   ├─ bench_latency.py
│   └─ export_unity_input.py
├─ deployment/
│   ├─ docker/
│   │   ├─ Dockerfile.offline
│   │   └─ Dockerfile.online
│   ├─ systemd/
│   │   ├─ intentflow-bridge.service
│   │   └─ intentflow-agent.service
│   └─ ci/
│       └─ github-actions.yml
├─ tests/
│   ├─ test_filters.py
│   ├─ test_runner.py
│   ├─ test_adapt.py
│   └─ test_ws_protocol.py
└─ docs/
    ├─ DESIGN.md
    ├─ CALIBRATION.md
    ├─ PROTOCOL.md
    └─ ADR/.gitkeep
```

## 3. クイックスタート
1. Conda 環境構築:
   ```bash
   conda env create -f environment.yml
   conda activate intentflow
   ```
2. 開発インストール:
   ```bash
   pip install -e .
   pre-commit install
   ```
3. オフライン学習:
   ```bash
   python intentflow/offline/train.py --config configs/offline.yaml
   python intentflow/offline/eval.py --config configs/offline.yaml
   ```
4. ONNX 書き出し:
   ```bash
   python intentflow/offline/export_onnx.py --checkpoint outputs/checkpoints/best.ckpt --onnx-path models/intentflow_head.onnx
   ```
5. オンライン推論起動:
   ```bash
   intentflow --config configs/online.yaml --host 0.0.0.0 --port 8000
   ```
6. ダミー送信:
   ```bash
   python scripts/simulate.py
   ```

## 4. 設定ファイル解説
- `configs/data.yaml`:
  - `dataset`: 使用データセット (`bci_iv_2a`, `bci_iv_2b`, `unicorn_live`)
  - `sample_rate`: EEG サンプリングレート
  - `epochs.window_seconds` / `hop_seconds`: ウィンドウ長とストライド
- `configs/preprocess.yaml`:
  - `bandpass.low/high`: 4–40 Hz を標準とするパスバンド
  - `notch.enabled` / `freqs`: 50/60 Hz のハム除去
  - `car.enabled`: 共通平均参照の ON/OFF
- `configs/offline.yaml`:
  - `model.name`: `eegnet`, `mamba_encoder` など
  - `trainer.max_epochs`: 学習エポック
  - `export.int8`: 量子化の有無
- `configs/online.yaml`:
  - `onnx.path`: 推論で利用する ONNX ファイル
  - `stabilizer.n_consecutive`: N 連続一致による安定化閾値
  - `stabilizer.ema_ms`: 推定の指数移動平均窓
  - `transport.ws_port`: WebSocket 配信ポート
- `configs/adapt.yaml`:
  - `otta.momentum`: テスト時適応の EMA 係数
  - `bandit.arms`: 強化学習ベース適応の候補数
- `configs/ports.yaml`:
  - `offline_dashboard`, `online_server`, `unity_bridge`: 各コンポーネントの待受ポート

## 5. オンライン推論パイプライン
```
Acquisition → Preprocess → Inference → Stabilizer → Adapt → WS Broadcast
```
- Acquisition: LSL や Unicorn から EEG を取得。
- Preprocess: バンドパス・ノッチ・CAR を適用して特徴抽出の安定性を確保。
- Inference: ONNX Runtime で Tensor モデルを実行し、意図確率を算出。
- Stabilizer: 連続一致・EMA・閾値制御でノイズを低減。
- Adapt: オンラインドメイン適応や ERRP フィードバックでモデルを調整。
- WS Broadcast: WebSocket 経由でゲームクライアントへ送信。

## 6. WebSocket プロトコル
### JSON 例
```json
{
  "type": "intent",
  "intent": "left",
  "conf": 0.87,
  "ts": 1718000000.123,
  "meta": {
    "run_id": "2024-06-07T12:00:00Z",
    "subject": "A01"
  },
  "protocol_version": 1
}
```
- `type`: `intent`, `metrics`, `errp` の 3 種。
- `intent`: `left | right | none`（MI の推論結果）。
- `conf`: 0.0–1.0 の信頼度。
- `ts`: Unix 時刻秒。`meta` は柔軟なキーを想定。
- `protocol_version`: 互換性管理用。

## 7. 遅延KPIと安定化
- 目標レイテンシ: 300–500 ms（取得〜配信まで）。
- 推奨設定:
  - N 連続一致: `n_consecutive = 3`
  - EMA: 100–200 ms の平滑化窓
  - 最小インターバル: 200 ms
  - 信頼度閾値: `conf >= 0.7`（±0.05 調整）
- ベンチ: `python scripts/bench_latency.py` で推論・送信ループを測定し、KPI を記録。

## 8. データ管理
- `data/raw` は Git 管理対象外（`.gitignore` 参照）。DVC や Git LFS の利用を推奨。
- BCI Competition IV-2a/2b:
  - `data/raw/bci_iv_2a/A01/` に `.gdf` を配置。
  - `data/raw/bci_iv_2b/B01/` など ID ごとに整理。
- 前処理後の特徴量は `data/processed`、オンライン適応ログは `data/external` を利用。

## 9. Unity/Unreal 接続
- クライアントはサブモジュール管理が推奨:
  ```bash
  git submodule add https://github.com/your-org/intentflow-unity-client apps/unity_client
  git submodule add https://github.com/your-org/intentflow-unreal-client apps/unreal_client
  ```
- 受信エンドポイント例: `ws://localhost:8000/control`
- 推奨受信レート: 20–50 Hz（フレーム同期に合わせて補間）。
- Unity/Unreal 双方で JSON パーサと信頼度フィルタを実装し、非同期処理でゲームプレイを阻害しないようにすること。

## 10. 再現性
- `intentflow/online/recorder/logger.py`: 生波形・特徴量・モデル出力・メトリクスを共通 `run_id` で保存。
- `intentflow/online/recorder/replay.py`: 過去ログからオンライン経路を再生して UI/UX を検証。
- `scripts/export_unity_input.py`: 同一 run を Unity 用フォーマットに変換。
- 乱数シードは学習・推論両方で固定し、構成を `configs/*.yaml` に保存して追跡。

## 11. CI/整形
- `pre-commit`: black・ruff・nbstripout を有効化してスタイルとノートブック差分を統制。
- GitHub Actions (`deployment/ci/github-actions.yml`): lint → type check → pytest の最小ワークフローを提供。
- Dockerfile を用いて offline/online ワークロードそれぞれを再現可能にする。

## 12. ライセンス/貢献/謝辞
- ライセンスは後日決定（暫定的に "All Rights Reserved"）。OSS 化時は Apache-2.0 を想定。
- Issue/Pull Request での貢献歓迎。ガイドラインは今後 `CONTRIBUTING.md` へ集約予定。
- EEG データセット提供者、オープンソース BCI コミュニティ、MI 研究コミュニティに感謝します。
