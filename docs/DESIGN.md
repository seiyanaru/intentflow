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
