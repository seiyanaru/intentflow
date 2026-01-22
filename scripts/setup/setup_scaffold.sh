#!/usr/bin/env bash
set -euo pipefail

mkdir -p configs \
data/raw \
data/interim \
data/processed \
data/external \
notebooks \
intentflow \
intentflow/core \
intentflow/offline \
intentflow/offline/models \
intentflow/online \
intentflow/online/inference \
intentflow/online/adapt \
intentflow/online/server \
intentflow/online/recorder \
intentflow/datasets \
apps/unity_client \
apps/unreal_client \
dashboards \
scripts \
deployment/docker \
deployment/systemd \
deployment/ci \
tests \
docs/ADR

touch data/raw/.gitkeep
touch data/interim/.gitkeep
touch data/processed/.gitkeep
touch data/external/.gitkeep
touch docs/ADR/.gitkeep

cat <<'EOR' > README.md
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
EOR

cat <<'EOR' > pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "intentflow"
version = "0.1.0"
description = "Motor Imagery EEG → Online inference → Game control framework"
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Seiya Narukawa" }]
dependencies = [
  "torch",
  "numpy", "scipy", "mne", "einops",
  "fastapi", "uvicorn", "websockets", "pydantic>=2",
  "onnxruntime-gpu; platform_system!='Darwin'",
  "onnxruntime; platform_system=='Darwin'",
  "pylsl",
  "gradio; extra == 'dash'",
  "streamlit; extra == 'dash'",
]

[tool.setuptools.packages.find]
include = ["intentflow*"]

[project.scripts]
intentflow = "intentflow.online.cli:main"

[project.optional-dependencies]
dev = ["pytest", "black", "ruff", "mypy", "pre-commit"]
dash = []
EOR

cat <<'EOR' > environment.yml
name: intentflow
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - pytorch
  - numpy
  - scipy
  - mne
  - pip
  - pip:
      - einops
      - fastapi
      - uvicorn
      - websockets
      - pydantic>=2
      - onnxruntime
      - onnxruntime-gpu
      - pylsl
      - gradio
      - streamlit
      - pytest
      - black
      - ruff
      - mypy
EOR

cat <<'EOR' > .gitignore
# Python artifacts
__pycache__/
*.py[cod]
*$py.class
*.so

# Environments
.env
.venv/
.conda/
.mypy_cache/
.pytest_cache/
.ipynb_checkpoints/

# Build outputs
build/
dist/
*.egg-info/
htmlcov/
coverage.xml

# Editors
.vscode/
.idea/
*.swp
*.swo

# Data and models
data/raw/
data/interim/
data/processed/
data/external/
models/
checkpoints/
logs/
*.onnx
*.pt
*.ckpt
*.parquet

# Misc
.DS_Store
Thumbs.db

# Preserve placeholders
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep
EOR

cat <<'EOR' > .gitattributes
*.onnx filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
*.flac filter=lfs diff=lfs merge=lfs -text
EOR

cat <<'EOR' > .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.5
    hooks:
      - id: ruff
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
EOR

cat <<'EOR' > configs/data.yaml
dataset: bci_iv_2a
root: data/raw/bci_iv_2a
subject: A01
sessions:
  train:
    - session01
    - session02
  eval:
    - session03
  adapt:
    - session04
sample_rate: 250
channels:
  include: ["C3", "Cz", "C4"]
  montage: standard_1020
epochs:
  window_seconds: 4.0
  hop_seconds: 0.5
labels:
  left: 1
  right: 2
EOR

cat <<'EOR' > configs/preprocess.yaml
bandpass:
  enabled: true
  low: 4.0
  high: 40.0
  order: 4
notch:
  enabled: true
  freqs: [50.0, 60.0]
  q: 30
car:
  enabled: true
artifact_rejection:
  variance_threshold: 5.0
  emg_band: [60.0, 120.0]
feature_extraction:
  method: logvar
  window_overlap: 0.5
EOR

cat <<'EOR' > configs/offline.yaml
model:
  name: eegnet
  encoder: eegencoder
  hidden_size: 128
training:
  batch_size: 64
  max_epochs: 150
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  scheduler: cosine
  gradient_clip: 1.0
validation:
  metric: balanced_accuracy
  early_stopping_patience: 15
export:
  onnx: models/intentflow_head.onnx
  int8: false
  opset: 18
logging:
  run_name: mi-eegnet-baseline
  output_dir: outputs
EOR

cat <<'EOR' > configs/online.yaml
acquisition:
  backend: lsl
  inlet_name: IntentflowStream
  buffer_seconds: 2.0
preprocess:
  config: configs/preprocess.yaml
inference:
  onnx_path: models/intentflow_head.onnx
  providers:
    - CUDAExecutionProvider
    - CPUExecutionProvider
  input_window_seconds: 4.0
  hop_seconds: 0.25
stabilizer:
  n_consecutive: 3
  ema_ms: 150
  min_interval_ms: 200
  confidence_threshold: 0.7
adaptation:
  config: configs/adapt.yaml
transport:
  ws_host: 0.0.0.0
  ws_port: 8000
  protocol_version: 1
metrics:
  emit_interval_ms: 500
  include_latency: true
EOR

cat <<'EOR' > configs/adapt.yaml
otta:
  enabled: true
  momentum: 0.01
  update_interval_seconds: 5
head_finetune:
  enabled: true
  learning_rate: 5e-5
  batch_size: 32
  warmup_steps: 100
errp_detector:
  enabled: true
  threshold: 3.0
  refractory_seconds: 2.0
bandit:
  enabled: false
  arms: ["baseline", "aggressive", "conservative"]
  exploration: 0.1
EOR

cat <<'EOR' > configs/ports.yaml
offline_dashboard: 8501
online_server: 8000
unity_bridge: 8700
unreal_bridge: 8800
metrics_exporter: 9100
EOR

cat <<'EOR' > intentflow/__init__.py
"""Top-level package for intentflow workflows."""
# TODO: Expose cohesive APIs for offline training and online inference orchestration.
EOR

cat <<'EOR' > intentflow/core/io_lsl.py
"""IO integration utilities for Lab Streaming Layer EEG acquisition."""
# TODO: Implement robust LSL inlet management with latency compensation.
EOR

cat <<'EOR' > intentflow/core/io_unicorn.py
"""Bluetooth Unicorn headset streaming interface helpers."""
# TODO: Provide calibration and sampling utilities for Unicorn EEG headsets.
EOR

cat <<'EOR' > intentflow/core/filters.py
"""Signal processing filters for EEG preprocessing."""
# TODO: Implement bandpass, notch, and CAR filters with numpy/scipy primitives.
EOR

cat <<'EOR' > intentflow/core/features.py
"""Feature extraction transforms for MI EEG signals."""
# TODO: Add CSP, Riemannian, and deep feature computation pipelines.
EOR

cat <<'EOR' > intentflow/core/schema.py
"""Data schemas and validation utilities for intentflow payloads."""
# TODO: Define Pydantic models covering acquisition, inference, and transport records.
EOR

cat <<'EOR' > intentflow/core/utils.py
"""Shared helper functions for configuration, logging, and timing."""
# TODO: Implement configuration loading, structured logging, and latency tracking.
EOR

cat <<'EOR' > intentflow/offline/models/eegencoder.py
"""Compact convolutional encoder tailored for MI EEG spectra."""
# TODO: Implement EEG encoder blocks compatible with ONNX export.
EOR

cat <<'EOR' > intentflow/offline/models/ciacnet.py
"""Channel-Intertwined Attention ConvNet (CIACNet) for MI classification."""
# TODO: Translate research prototype into production-ready architecture.
EOR

cat <<'EOR' > intentflow/offline/models/mamba_encoder.py
"""State space sequence encoder using Mamba blocks for EEG."""
# TODO: Prototype SSM layers tuned for low-latency inference.
EOR

cat <<'EOR' > intentflow/offline/models/eegnet.py
"""EEGNet baseline implementation for MI tasks."""
# TODO: Port canonical EEGNet with configurable temporal and spatial kernels.
EOR

cat <<'EOR' > intentflow/offline/models/registry.py
"""Model registry for resolving offline training architectures."""
# TODO: Map string keys to model constructors and manage checkpoints.
EOR

cat <<'EOR' > intentflow/offline/train.py
"""Offline training entry point for intentflow models."""
# TODO: Wire up dataset loading, preprocessing, and trainer orchestration.
EOR

cat <<'EOR' > intentflow/offline/eval.py
"""Evaluation harness for trained intentflow checkpoints."""
# TODO: Implement metrics logging and subject-wise benchmarking.
EOR

cat <<'EOR' > intentflow/offline/export_onnx.py
"""Export trained intentflow models to ONNX for online inference."""
# TODO: Add torch->ONNX conversion with dynamic axes and quantization.
EOR

cat <<'EOR' > intentflow/offline/distill.py
"""Knowledge distillation tools for compact online deployable models."""
# TODO: Integrate teacher-student training with consistency regularization.
EOR

cat <<'EOR' > intentflow/online/inference/tcnet_head.py
"""TCNet-style classification head for streaming inference."""
# TODO: Implement temporal convolutional layers optimized for low latency.
EOR

cat <<'EOR' > intentflow/online/inference/runner.py
"""Online inference runner that bridges preprocessing and ONNX execution."""
# TODO: Manage sliding windows, runtime providers, and scheduling hooks.
EOR

cat <<'EOR' > intentflow/online/inference/stabilizer.py
"""Signal stabilizer combining EMA, voting, and confidence thresholds."""
# TODO: Implement adaptive smoothing with configurable hysteresis.
EOR

cat <<'EOR' > intentflow/online/adapt/otta.py
"""Online test-time adaptation (OTTA) strategies for MI EEG."""
# TODO: Prototype entropy minimization and pseudo-label refinement.
EOR

cat <<'EOR' > intentflow/online/adapt/head_ft.py
"""Lightweight head fine-tuning routines for online personalization."""
# TODO: Update classifier weights incrementally without drift.
EOR

cat <<'EOR' > intentflow/online/adapt/errp_detector.py
"""Error-related potential (ErrP) detection for feedback-driven updates."""
# TODO: Implement ErrP classifiers and adaptive thresholds.
EOR

cat <<'EOR' > intentflow/online/adapt/bandit.py
"""Bandit-based policy selection for adaptive online strategies."""
# TODO: Explore contextual bandits to choose stabilizer or adaptation modes.
EOR

cat <<'EOR' > intentflow/online/server/app.py
"""FastAPI application exposing health, metrics, and control streams."""
# TODO: Attach real inference pipeline and recorder integration.

from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

def create_app() -> FastAPI:
  app = FastAPI(title="Intentflow Online API", version="0.1.0")
  register_routes(app)
  return app

def register_routes(app: FastAPI) -> None:
  @app.get("/health")
  async def health() -> dict[str, str]:
    return {"status": "ok"}

  @app.get("/metrics")
  async def metrics() -> dict[str, float]:
    return {"latency_ms": 0.0, "throughput_hz": 0.0}

  @app.websocket("/control")
  async def control(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
      while True:
        message = await websocket.receive_text()
        # TODO: Route messages through stabilizer and adaptation layers.
        await websocket.send_text(message)
    except WebSocketDisconnect:
      # TODO: Track disconnections and cleanup resources.
      return

app = create_app()
EOR

cat <<'EOR' > intentflow/online/server/clients.py
"""Client management utilities for downstream WebSocket consumers."""
# TODO: Maintain connection registries and broadcast mechanisms.
EOR

cat <<'EOR' > intentflow/online/recorder/logger.py
"""Runtime recorder capturing raw signals, features, outputs, and metrics."""
# TODO: Stream structured logs to disk and observability backends.
EOR

cat <<'EOR' > intentflow/online/recorder/replay.py
"""Replay engine for regenerating online sessions from recorded logs."""
# TODO: Implement deterministic playback with speed controls and hooks.
EOR

cat <<'EOR' > intentflow/online/cli.py
"""Command-line entry point for the intentflow online stack."""
# TODO: Extend CLI options for adapters, logging sinks, and dashboards.

from __future__ import annotations

import argparse

import uvicorn

def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Intentflow online inference server")
  parser.add_argument("--config", type=str, default="configs/online.yaml", help="Path to online inference config file")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface for FastAPI server")
  parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
  parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
  return parser

def run_server(host: str, port: int, reload: bool) -> None:
  """Launch the uvicorn server with the FastAPI application."""
  # TODO: Load config and initialize pipeline before serving.
  from intentflow.online.server.app import create_app
  uvicorn.run(create_app(), host=host, port=port, reload=reload)

def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  run_server(args.host, args.port, args.reload)

if __name__ == "__main__":
  main()
EOR

cat <<'EOR' > intentflow/datasets/bci_iv_2a.py
"""Dataset utilities for BCI Competition IV 2a motor imagery data."""
# TODO: Implement subject/session loaders with preprocessing hooks.
EOR

cat <<'EOR' > intentflow/datasets/bci_iv_2b.py
"""Dataset utilities for BCI Competition IV 2b motor imagery data."""
# TODO: Provide binary classification loaders and augmentation options.
EOR

cat <<'EOR' > intentflow/datasets/unicorn_live.py
"""Live data interface for Unicorn headset streaming."""
# TODO: Expose asynchronous generators yielding inference-ready batches.
EOR

cat <<'EOR' > dashboards/app.py
"""Streamlit dashboard scaffolding for online monitoring."""
# TODO: Build latency, confidence, and adaptation visualization panels.
EOR

cat <<'EOR' > scripts/simulate.py
"""Simulate intent events via WebSocket for integration testing."""
# TODO: Make the payload and timing configurable via CLI options.

import asyncio
import json
import time
from typing import List, Dict, Any

import websockets

async def send_events(uri: str, payloads: List[Dict[str, Any]]) -> None:
  async with websockets.connect(uri) as websocket:
    for payload in payloads:
      payload["ts"] = time.time()
      await websocket.send(json.dumps(payload))
      await asyncio.sleep(0.2)

async def main() -> None:
  uri = "ws://localhost:8000/control"
  events = [
    {"type": "intent", "intent": "left", "conf": 0.9, "meta": {"source": "simulate.py"}, "protocol_version": 1},
    {"type": "intent", "intent": "left", "conf": 0.92, "meta": {"source": "simulate.py"}, "protocol_version": 1},
    {"type": "intent", "intent": "left", "conf": 0.94, "meta": {"source": "simulate.py"}, "protocol_version": 1},
  ]
  await send_events(uri, events)

if __name__ == "__main__":
  asyncio.run(main())
EOR

cat <<'EOR' > scripts/calibrate.py
"""Calibration script entry point for subject-specific MI sessions."""
# TODO: Implement guided calibration routines with visual feedback.
EOR

cat <<'EOR' > scripts/bench_latency.py
"""Benchmark tool for measuring end-to-end inference latency."""
# TODO: Measure acquisition, preprocessing, inference, and transport timing.
EOR

cat <<'EOR' > scripts/export_unity_input.py
"""Utility to convert recorded runs into Unity ingestion format."""
# TODO: Map recorded intents to animator parameters and export JSON/CSV.
EOR

cat <<'EOR' > deployment/docker/Dockerfile.offline
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /opt/intentflow
COPY . /opt/intentflow

RUN pip install --no-cache-dir -e .[dev]

CMD ["python", "intentflow/offline/train.py", "--config", "configs/offline.yaml"]
EOR

cat <<'EOR' > deployment/docker/Dockerfile.online
FROM python:3.10-slim

WORKDIR /opt/intentflow
COPY . /opt/intentflow

RUN pip install --no-cache-dir uvicorn[standard] onnxruntime websockets fastapi && \
    pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["intentflow", "--config", "configs/online.yaml", "--host", "0.0.0.0", "--port", "8000"]
EOR

cat <<'EOR' > deployment/systemd/intentflow-bridge.service
[Unit]
Description=Intentflow Game Bridge
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/intentflow
ExecStart=/usr/bin/python scripts/export_unity_input.py
Restart=on-failure
User=intentflow

[Install]
WantedBy=multi-user.target
EOR

cat <<'EOR' > deployment/systemd/intentflow-agent.service
[Unit]
Description=Intentflow Online Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/intentflow
ExecStart=/usr/local/bin/intentflow --config configs/online.yaml --host 0.0.0.0 --port 8000
Restart=always
User=intentflow

[Install]
WantedBy=multi-user.target
EOR

cat <<'EOR' > deployment/ci/github-actions.yml
name: intentflow-ci

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      - name: Lint
        run: |
          ruff check intentflow tests
          black --check intentflow tests
      - name: Type check
        run: mypy intentflow
      - name: Tests
        run: pytest
EOR

cat <<'EOR' > tests/test_filters.py
"""Tests for signal filtering utilities."""
# TODO: Replace smoke test with spectral assertions.

def test_filters_placeholder() -> None:
  assert True
EOR

cat <<'EOR' > tests/test_runner.py
"""Tests for the online inference runner."""
# TODO: Mock ONNX runtime and validate sliding window scheduling.

def test_runner_placeholder() -> None:
  assert True
EOR

cat <<'EOR' > tests/test_adapt.py
"""Tests covering online adaptation strategies."""
# TODO: Simulate adaptation cycles and verify parameter updates.

def test_adapt_placeholder() -> None:
  assert True
EOR

cat <<'EOR' > tests/test_ws_protocol.py
"""Tests for WebSocket protocol compatibility."""
# TODO: Validate schema evolution and backward compatibility.

def test_ws_protocol_placeholder() -> None:
  assert True
EOR

cat <<'EOR' > docs/DESIGN.md
# intentflow DESIGN

## Goals
- Ensure low-latency MI intent decoding with reproducible pipelines.
- Decouple offline experimentation from online serving and application bridges.

## Architecture
1. Offline: preprocessing, feature extraction, supervised training, ONNX export。
2. Online: signal acquisition、streaming inference、stabilizer、adaptation。
3. Bridge: WebSocket/HTTP interfaces to external apps (Unity/Unreal/Robotics)。

## Key Decisions
- ONNX Runtime を採用し、GPU/CPU 両対応で低レイテンシ推論を実現。
- 全コンポーネントを YAML 設定で統合し、CI で静的検証。
- Recorder/Replay を導入し、回帰テストと UX 検証を高速化。

## Open Questions
- Mamba 系エンコーダの軽量化と ONNX 互換性の担保。
- EEG ハードウェアごとのキャリブレーション手順標準化。
EOR

cat <<'EOR' > docs/CALIBRATION.md
# Calibration Guide

1. センサー装着とインピーダンス確認。
2. `python scripts/calibrate.py` を起動し、視覚フィードバックで MI タスクを案内。
3. `intentflow/online/recorder/logger.py` で run_id を発行し、calibration run を保存。
4. 保存したデータを `configs/preprocess.yaml` に合わせて前処理し、ベースラインモデルを学習。
5. 成功判定:
   - 安定した α/β リズムの左右差
   - バランスの取れたクラス分布
   - モデル精度 > 70%（BCI IV-2a 目安）
EOR

cat <<'EOR' > docs/PROTOCOL.md
# Online Protocol

## Session Flow
1. Pre-check: health/metrics エンドポイントでサービス状態を確認。
2. Warm-up: `scripts/simulate.py` でクライアント接続確認。
3. Live Run: プレイヤーにタスク指示を出し、意図をリアルタイムに送信。
4. Post-run: Recorder のログを保存し、replay でフィードバックを実施。

## Safety
- ERRP 検出時は `type="errp"` イベントを配信し、ゲーム側で安全モードへ。
- Confidence が閾値を下回った場合は `intent="none"` を送信し、不確実操作を抑制。
EOR
