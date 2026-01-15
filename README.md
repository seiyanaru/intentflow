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
├─ setup_scaffold.sh
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
│   │   │   ├─ eeg_transformer.py
│   │   │   ├─ eegnet_lawhern.py
│   │   │   └─ registry.py
│   │   ├─ train.py
│   │   ├─ eval.py
│   │   ├─ export_onnx.py
│   │   ├─ distill.py
│   │   └─ data_npz.py
│   ├─ online/
│   │   ├─ inference/
│   │   │   ├─ tcnet_head.py
│   │   │   ├─ runner.py
│   │   │   ├─ stabilizer.py
│   │   │   ├─ worker.py
│   │   │   └─ onnx_model.py
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
│   │   ├─ dsp/
│   │   │   └─ filters.py
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
│   ├─ export_unity_input.py
│   ├─ gen_intents_demo.py
│   ├─ make_dummy_mi_npz.py
│   ├─ replay_intents_to_control.py
│   └─ replay_npz_to_ingest.py
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
│   ├─ test_ws_protocol.py
│   ├─ test_eegencoder_shape.py
│   ├─ test_ingest_pipeline.py
│   └─ test_models_and_flow.py
└─ docs/
    ├─ DESIGN.md
    ├─ CALIBRATION.md
    ├─ PROTOCOL.md
    ├─ seminar_analysis.md   ← 実験結果の詳細分析
    └─ ADR/
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
3. 論文実験の再現 (Offline Experiments):
   ```bash
   # BCIC IV-2a データセットでの実験
   python intentflow/offline/train_pipeline.py --model tcformer --dataset bcic2a --gpu_id 0
   python intentflow/offline/train_pipeline.py --model tcformer_hybrid --dataset bcic2a --gpu_id 0

   # BCIC IV-2b データセットでの実験
   python intentflow/offline/train_pipeline.py --model tcformer --dataset bcic2b --gpu_id 0 --no_interaug
   python intentflow/offline/train_pipeline.py --model tcformer_hybrid --dataset bcic2b --gpu_id 0 --no_interaug

   # HGD データセットでの実験
   python intentflow/offline/train_pipeline.py --model tcformer --dataset hgd --gpu_id 0
   python intentflow/offline/train_pipeline.py --model tcformer_hybrid --dataset hgd --gpu_id 0
   ```
   ※ 結果は `intentflow/offline/results/paper_experiments/{dataset}/{timestamp}/` に保存されます。
   
   **一括実行スクリプト:**
   ```bash
   ./scripts/run_bcic2b_experiments.sh 0   # GPU 0 で BCIC 2b 実験
   ./scripts/run_hgd_experiments.sh 0      # GPU 0 で HGD 実験
   ```

4. オフライン学習 (通常):
   ```bash
   python intentflow/offline/train.py --config configs/offline.yaml
   python intentflow/offline/eval.py --config configs/offline.yaml
   ```
5. ONNX 書き出し:
   ```bash
   python intentflow/offline/export_onnx.py --checkpoint outputs/checkpoints/best.ckpt --onnx-path models/intentflow_head.onnx
   ```
6. オンライン推論起動:
   ```bash
   intentflow --config configs/online.yaml --host 0.0.0.0 --port 8000
   ```
7. ダミー送信:
   ```bash
   python scripts/simulate.py
   ```

## 4. 論文用オフライン実験モデル (TCFormer Hybrid)
本リポジトリでは、以下の3つのモデルアーキテクチャを比較実験しています。

1. **TCFormer (Base)**: ベースとなるTemporal Convolutional Transformer。
2. **TCFormer_TTT**: 全層を Test-Time Training (TTT) レイヤーに置き換えたモデル。
3. **TCFormer_Hybrid**: Self-AttentionとTTT Adapterを並列配置したハイブリッドモデル。
   - **Entropy-driven Dynamic Gating**: 予測の不確実性（エントロピー）に基づいてTTT適応強度を動的制御
   - **2-Pass Forward**: Pass1でエントロピー計算、Pass2でTTT適応

### Hybrid Model Architecture
![TCFormer Hybrid Architecture](hybrid_arch.png)

Hybridモデルの詳細なアーキテクチャ、数式、処理フローについては、以下のドキュメントを参照してください：
- [**README_Hybrid.md**](intentflow/offline/README_Hybrid.md)

### 実験結果サマリー

#### 精度比較（3データセット）
| Dataset | Base (TCFormer) | Hybrid (Dynamic) | 差分 |
|---------|-----------------|------------------|------|
| **BCIC IV 2a** | **84.67%** ± 9.25 | 83.52% ± 5.52 | -1.15% |
| **BCIC IV 2b** | **82.67%** ± 6.73 | 80.76% ± 7.39 | -1.91% |
| **HGD** | **92.95%** ± 7.01 | 79.29% ± 14.61 | -13.66% |

#### 訓練時間比較
| Dataset | Base | Hybrid | 短縮率 |
|---------|------|--------|--------|
| BCIC IV 2a | 43.3 min | 23.7 min | **1.8x** |
| BCIC IV 2b | 131.3 min | 12.1 min | **10.9x** |
| HGD | 212.2 min | 79.5 min | **2.7x** |

#### 主な知見
- **訓練効率**: Hybridモデルは訓練時間を大幅に短縮（最大10.9倍）
- **安定性**: 標準偏差が改善（2a: 9.25→5.52）
- **課題**: 高品質データ（HGD）では精度低下が顕著

詳細な分析は [docs/seminar_analysis.md](docs/seminar_analysis.md) を参照。

### 実験結果の構成
実験スクリプトを実行すると、`intentflow/offline/results/paper_experiments/{dataset}/{timestamp}/` 以下にデータが生成されます。
- `data/`: 学習履歴、テスト結果、特徴量データ (`.json`, `.npy`, `.npz`)
- `figures/`: 生成された図表（t-SNE, Entropy, Confusion Matrix, Learning Curves）
- `curves/`: 被験者別の学習曲線

## 5. 設定ファイル解説
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

## 6. オンライン推論パイプライン
```
Acquisition → Preprocess → Inference → Stabilizer → Adapt → WS Broadcast
```
- Acquisition: LSL や Unicorn から EEG を取得。
- Preprocess: バンドパス・ノッチ・CAR を適用して特徴抽出の安定性を確保。
- Inference: ONNX Runtime で Tensor モデルを実行し、意図確率を算出。
- Stabilizer: 連続一致・EMA・閾値制御でノイズを低減。
- Adapt: オンラインドメイン適応や ERRP フィードバックでモデルを調整。
- WS Broadcast: WebSocket 経由でゲームクライアントへ送信。

## 7. WebSocket プロトコル
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

## 8. 遅延KPIと安定化
- 目標レイテンシ: 300–500 ms（取得〜配信まで）。
- 推奨設定:
  - N 連続一致: `n_consecutive = 3`
  - EMA: 100–200 ms の平滑化窓
  - 最小インターバル: 200 ms
  - 信頼度閾値: `conf >= 0.7`（±0.05 調整）
- ベンチ: `python scripts/bench_latency.py` で推論・送信ループを測定し、KPI を記録。

## 9. データ管理
- `data/raw` は Git 管理対象外（`.gitignore` 参照）。DVC や Git LFS の利用を推奨。
- **BCI Competition IV-2a/2b**:
  - `data/raw/BCICIV_2a_gdf/` に `.gdf` ファイルを配置。
  - `data/raw/BCICIV_2b_gdf/` に `.gdf` ファイルを配置。
- **HGD (High Gamma Dataset)**:
  - MOABB経由で自動ダウンロード（初回実行時）
  - または手動で `data/raw/HGD/` に配置
- 前処理後の特徴量は `data/processed`、オンライン適応ログは `data/external` を利用。

## 10. Unity/Unreal 接続
- クライアントはサブモジュール管理が推奨:
  ```bash
  git submodule add https://github.com/your-org/intentflow-unity-client apps/unity_client
  git submodule add https://github.com/your-org/intentflow-unreal-client apps/unreal_client
  ```
- 受信エンドポイント例: `ws://localhost:8000/control`
- 推奨受信レート: 20–50 Hz（フレーム同期に合わせて補間）。
- Unity/Unreal 双方で JSON パーサと信頼度フィルタを実装し、非同期処理でゲームプレイを阻害しないようにすること。

## 11. 再現性
- `intentflow/online/recorder/logger.py`: 生波形・特徴量・モデル出力・メトリクスを共通 `run_id` で保存。
- `intentflow/online/recorder/replay.py`: 過去ログからオンライン経路を再生して UI/UX を検証。
- `scripts/export_unity_input.py`: 同一 run を Unity 用フォーマットに変換。
- 乱数シードは学習・推論両方で固定し、構成を `configs/*.yaml` に保存して追跡。

## 12. CI/整形
- `pre-commit`: black・ruff・nbstripout を有効化してスタイルとノートブック差分を統制。
- GitHub Actions (`deployment/ci/github-actions.yml`): lint → type check → pytest の最小ワークフローを提供。
- Dockerfile を用いて offline/online ワークロードそれぞれを再現可能にする。

## 13. 今後の課題と研究方向

### Hybridモデルの改善
現在のHybridモデルには以下の課題があり、改善を進めています：

1. **訓練/テスト挙動の統一**
   - `entropy_gating_in_train: True` で訓練時もEntropy Gatingを有効化
   
2. **パラメータチューニング**
   - エントロピー閾値の最適化（現在: 0.95 → 提案: 0.7）
   - TTT学習率の低減（0.01 → 0.001）

3. **データセット適応的なゲーティング**
   - データセット特性に応じた動的な適応強度制御

### 研究目標
- 精度維持（±1%以内）+ 訓練時間3-10倍短縮
- 国際学会投稿レベルの新規性確保

## 14. ライセンス/貢献/謝辞
- ライセンスは後日決定（暫定的に "All Rights Reserved"）。OSS 化時は Apache-2.0 を想定。
- Issue/Pull Request での貢献歓迎。ガイドラインは今後 `CONTRIBUTING.md` へ集約予定。
- EEG データセット提供者、オープンソース BCI コミュニティ、MI 研究コミュニティに感謝します。
