# Calibration Guide

1. センサー装着とインピーダンス確認。
2. `python scripts/calibrate.py` を起動し、視覚フィードバックで MI タスクを案内。
3. `intentflow/online/recorder/logger.py` で run_id を発行し、calibration run を保存。
4. 保存したデータを `configs/preprocess.yaml` に合わせて前処理し、ベースラインモデルを学習。
5. 成功判定:
   - 安定した α/β リズムの左右差
   - バランスの取れたクラス分布
   - モデル精度 > 70%（BCI IV-2a 目安）
