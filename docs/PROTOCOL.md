# Online Protocol

## Session Flow
1. Pre-check: health/metrics エンドポイントでサービス状態を確認。
2. Warm-up: `scripts/simulate.py` でクライアント接続確認。
3. Live Run: プレイヤーにタスク指示を出し、意図をリアルタイムに送信。
4. Post-run: Recorder のログを保存し、replay でフィードバックを実施。

## Safety
- ERRP 検出時は `type="errp"` イベントを配信し、ゲーム側で安全モードへ。
- Confidence が閾値を下回った場合は `intent="none"` を送信し、不確実操作を抑制。
