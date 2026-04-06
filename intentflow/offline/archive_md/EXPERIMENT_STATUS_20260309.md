# 実験ステータス - 5-Seed Baseline (2026-03-09)

## 🚀 実験開始！

### 実験情報
- **実験名**: 5-Seed Baseline (Clean)
- **開始時刻**: 2026-03-09 12:21
- **推定完了**: 2026-03-10 06:21 (約18時間後)
- **Process ID**: 4111176
- **GPU**: NVIDIA GeForce RTX 2080 Ti (GPU 0)

---

## 📊 実験詳細

### Seeds
```
Seed 0: 実行中 (Epoch 0 開始確認)
Seed 1: 待機中
Seed 2: 待機中
Seed 3: 待機中
Seed 4: 待機中
```

### 推定時間
- **各Seed**: ~3.5時間 (1000エポック)
- **合計**: ~17.5時間

### 期待結果
```
現状データ:
  Seed 0 (既存): 82.79% ± 8.05%
  Seed 1 (既存): 83.87% ± 8.20%

期待値:
  5-seed平均: 84.0 ± 2.0%
  論文値とのGap: -0.79% (十分許容範囲)
```

---

## 💻 モニタリングコマンド

### ログ監視（リアルタイム）
```bash
tail -f /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/launcher_5seeds_20260309_122106.log
```

### プロセス確認
```bash
# バックグラウンドプロセス
ps -p 4111176

# 訓練プロセス
ps aux | grep train_pipeline | grep -v grep
```

### GPU使用状況
```bash
nvidia-smi

# またはリアルタイム監視
watch -n 1 nvidia-smi
```

### 現在のエポック確認
```bash
# ログの最新30行
tail -30 /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/launcher_5seeds_20260309_122106.log

# 特定のseedのログ
tail -50 /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/baseline_5seed_s0_20260309_122106.log
```

---

## 📁 出力ファイル

### ディレクトリ構造
```
results/
├── launcher_5seeds_20260309_122106.log         # 全体ログ
├── run_5seeds_sequential_20260309_122106.sh    # 実行スクリプト
├── baseline_5seed_s0_20260309_122106/
│   ├── results.txt                              # 最終結果
│   ├── checkpoints/                             # モデルチェックポイント
│   └── config.yaml                              # 使用設定
├── baseline_5seed_s0_20260309_122106.log       # Seed 0ログ
├── baseline_5seed_s1_20260309_122106/
│   └── ...
└── ... (s2, s3, s4)
```

### 結果確認コマンド
```bash
# 完了したseedの結果一覧
find results -name "results.txt" -path "*baseline_5seed*" -path "*20260309_122106*" \
  -exec grep -H "Average Test Accuracy" {} \;

# 詳細確認
cat results/baseline_5seed_s0_20260309_122106/results.txt
```

---

## 🔔 通知・チェックポイント

### 各Seed完了時（約3.5時間ごと）
```bash
# 完了確認
grep "Seed .* Completed" results/launcher_5seeds_20260309_122106.log

# 精度確認
grep "Average Test Accuracy" results/baseline_5seed_s*_20260309_122106/results.txt
```

### 期待されるマイルストーン
```
12:21 - Seed 0 開始
15:51 - Seed 0 完了 (予想精度: 82-83%)
      - Seed 1 開始
19:21 - Seed 1 完了 (予想精度: 83-84%)
      - Seed 2 開始
22:51 - Seed 2 完了 (予想精度: 83-85%)
      - Seed 3 開始
02:21 - Seed 3 完了 (予想精度: 83-84%)
      - Seed 4 開始
06:21 - Seed 4 完了 (予想精度: 82-84%)
      - 全実験完了
```

---

## 🎯 完了後のアクション

### Step 1: 結果集計
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline

# 全結果表示
for SEED in 0 1 2 3 4; do
    RESULTS_FILE="results/baseline_5seed_s${SEED}_20260309_122106/results.txt"
    if [ -f "${RESULTS_FILE}" ]; then
        ACC=$(grep "Average Test Accuracy" "${RESULTS_FILE}" | awk '{print $4}')
        echo "Seed ${SEED}: ${ACC}%"
    fi
done
```

### Step 2: 平均計算
```bash
# Python簡易計算
python3 << EOF
import glob
import re

accuracies = []
for f in glob.glob("results/baseline_5seed_s*_20260309_122106/results.txt"):
    with open(f) as file:
        for line in file:
            if "Average Test Accuracy" in line:
                acc = float(line.split()[3])
                accuracies.append(acc)
                break

if accuracies:
    import numpy as np
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)
    print(f"5-Seed Baseline Results:")
    print(f"  Mean: {mean:.2f}%")
    print(f"  Std:  {std:.2f}%")
    print(f"  Individual: {accuracies}")
else:
    print("No results found yet.")
EOF
```

### Step 3: 次の実験判断

**Case A: 平均 ≥ 84%**
```bash
# SAL Sweep実験へ進む
./scripts/run_otta_sal_sweep.sh
```

**Case B: 平均 < 84%**
```bash
# 追加分析が必要
# - Seed間のばらつき確認
# - 被験者別パフォーマンス確認
# - Hyperparameter微調整検討
```

---

## 🐛 トラブルシューティング

### プロセスが停止した場合
```bash
# 確認
ps -p 4111176

# 停止していた場合
tail -100 results/launcher_5seeds_20260309_122106.log

# 特定のseedから再開
./scripts/run_baseline_single_seed.sh <SEED> 0
```

### GPUメモリエラー
```bash
# GPU確認
nvidia-smi

# プロセスkill（緊急時のみ）
pkill -f train_pipeline

# GPUメモリクリア
python3 -c "import torch; torch.cuda.empty_cache()"
```

### ディスク容量不足
```bash
# 確認
df -h /workspace-cloud/seiya.narukawa/intentflow

# 古い結果の削除（注意！）
# rm -rf results/old_experiment_name
```

---

## 📈 現在のGPU状態

```
GPU 0: NVIDIA GeForce RTX 2080 Ti
  - 使用率: 53%
  - メモリ使用: 10138 MiB / 11264 MiB (90%)
  - ステータス: ✅ 正常動作中
```

---

## 🎓 この実験の意義

### 科学的重要性
1. **再現性の確立**: 複数seedで安定性を検証
2. **統計的信頼性**: 論文投稿に必須の複数試行
3. **OTTA評価の土台**: 正確なBaselineがないとOTTA効果を測定できない

### 期待される成果
- ✅ Baseline平均84.0%達成
- ✅ 標準偏差2.0%以下（安定性確認）
- ✅ OTTA実験の準備完了

### 次のステップへの道筋
```
現在: 5-Seed Baseline実験中
  ↓ (~18時間)
完了: Baseline 84.0 ± 2.0%
  ↓
次: SAL Sweep (0.35, 0.40, 0.45, 0.50)
  ↓ (~8時間)
目標: OTTA 86.5%達成
  ↓
最終: 論文投稿準備完了 ✓
```

---

## 📝 メモ

- **開始時刻**: 2026-03-09 12:21:06
- **推定完了**: 2026-03-10 06:21:06
- **GPU占有**: 継続的（他の実験は避ける）
- **次回確認**: 明日朝（完了予定時刻前後）

---

**ステータス**: 🟢 **実験実行中**
**最終更新**: 2026-03-09 12:21
**次回アクション**: 2026-03-10 06:30 結果確認

---

## 🔗 関連ドキュメント

- [EXPERIMENT_GUIDE_OTTA.md](EXPERIMENT_GUIDE_OTTA.md) - 実験ガイド完全版
- [COMPREHENSIVE_ANALYSIS_SUMMARY.md](COMPREHENSIVE_ANALYSIS_SUMMARY.md) - 包括的分析
- [ANALYSIS_OTTA_GAPS.md](ANALYSIS_OTTA_GAPS.md) - OTTA課題分析
