# OTTA 86.5%到達のための実験ガイド

## 🎯 目標

```
現状: Baseline 83.87% (seed 1)
      OTTA    82.99% (標準設定)

目標: Baseline 84.0% (5-seed平均)
      OTTA    86.5% (+2.5%改善)
```

---

## 📋 実験計画

### Phase 1: Baseline確立（最優先） ⭐⭐⭐

**目的**: OTTA評価の土台となる安定Baselineを確立

**方法1: インタラクティブ実行（推奨・安全）**
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline
./scripts/run_clean_baseline_5seeds.sh
```

**方法2: バックグラウンド実行（長時間作業向け）**
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline
./scripts/launch_5seed_background.sh

# モニタリング
tail -f results/launcher_5seeds_*.log

# 進捗確認
ps aux | grep train_pipeline
nvidia-smi
```

**方法3: 単一Seed実行（テスト・追加実験向け）**
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline

# Seed 2を実行
./scripts/run_baseline_single_seed.sh 2 0

# バックグラウンドで実行
nohup ./scripts/run_baseline_single_seed.sh 3 0 > results/baseline_s3.log 2>&1 &
```

**所要時間**:
- 各seed: 3.5時間
- 5 seeds: 17.5時間（連続実行の場合）

**期待結果**: 84.0 ± 2.0%

---

### Phase 2: SAL閾値最適化 ⭐⭐

**目的**: 最適なSAL閾値を発見（0.35, 0.40, 0.45, 0.50）

**実行方法**:
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline

# インタラクティブ実行
./scripts/run_otta_sal_sweep.sh

# またはバックグラウンド
nohup ./scripts/run_otta_sal_sweep.sh > results/sal_sweep.log 2>&1 &

# 進捗モニタリング
tail -f results/sal_sweep.log
```

**所要時間**:
- 各SAL値: 2時間
- 合計: 8時間

**期待結果**:
- SAL=0.40で最高精度（推定84.5-85.0%）
- Subject 2の適応率改善（16.7% → 30-35%）

---

### Phase 3: Pmax/Energy微調整（オプション）

Phase 2の結果に応じて実施

---

## 🔍 現状分析サマリー

### 過去の実験結果

| 実験 | pmax | SAL | Energy | 精度 | 評価 |
|------|------|-----|--------|------|------|
| 標準設定 | 0.7 | 0.5 | 0.95 | 82.99% | 🟡 Baseline並 |
| 緩和版（exp3） | 0.6 | **0.3** | 0.95 | 82.60% | ❌ 過度緩和 |
| Energy除外 | 0.7 | 0.5 | ∞ | 80.75% | ❌ 悪化 |

### 重要な発見

1. **SAL=0.3は緩すぎ** → ノイズ適応で悪化
2. **SAL=0.5は厳しすぎ** → 適応不足（Subject 2で16.7%のみ）
3. **最適範囲: 0.35-0.45** （推定）
4. **Energy gateは必要** → 除外すると-2.24%悪化

---

## 📊 実験の優先順位

| Phase | 実験 | 所要時間 | 期待改善 | 優先度 |
|-------|------|----------|----------|--------|
| **1** | **5-seed Baseline** | 17.5h | **+1.2%** | 🔴 最優先 |
| **2** | **SAL探索** | 8h | **+1.5-2.0%** | 🟠 高 |
| 3 | Pmax調整 | 4h | +0.5-1.0% | 🟡 中 |
| 4 | Energy最適化 | 6h | +0.5-1.0% | 🟡 中 |

---

## 💻 クイックスタート

### 今すぐ実行（推奨順序）

**1. Baseline 5-seed（最優先）**
```bash
cd /workspace-cloud/seiya.narukawa/intentflow/intentflow/offline

# バックグラウンド実行
./scripts/launch_5seed_background.sh

# モニタリング
tail -f results/launcher_5seeds_*.log
```

**2. 完了後、SAL Sweep実行**
```bash
# Baseline完了を確認
ls -lht results/baseline_5seed_*/results.txt

# SAL探索開始
nohup ./scripts/run_otta_sal_sweep.sh > results/sal_sweep.log 2>&1 &

# モニタリング
tail -f results/sal_sweep.log
```

---

## 📈 進捗確認コマンド

### 実験の実行状態確認
```bash
# 実行中のPythonプロセス
ps aux | grep train_pipeline | grep -v grep

# GPU使用状況
nvidia-smi

# 最新ログの監視
tail -f results/*.log
```

### 結果の確認
```bash
# 最新の結果ファイル
ls -lht results/*/results.txt | head -10

# 精度一覧
find results -name "results.txt" -mtime -1 -exec grep -H "Average Test Accuracy" {} \;

# 詳細確認
cat results/baseline_5seed_s0_*/results.txt
```

---

## 🐛 トラブルシューティング

### プロセスが停止した場合
```bash
# プロセス確認
ps aux | grep train_pipeline

# ログ確認
tail -100 results/run_*.log

# GPU確認
nvidia-smi

# 再実行（特定のseed）
./scripts/run_baseline_single_seed.sh <SEED> 0
```

### ディスク容量確認
```bash
df -h /workspace-cloud/seiya.narukawa/intentflow
du -sh results/
```

### メモリ/GPU確認
```bash
# メモリ
free -h

# GPU
nvidia-smi

# GPUメモリクリア（必要時）
python3 -c "import torch; torch.cuda.empty_cache()"
```

---

## 📝 実験ログの解釈

### 成功の確認
```bash
# results.txtの最終行付近に以下が表示されればOK
grep -A 5 "Summary Statistics" results/*/results.txt
```

期待される出力:
```
--- Summary Statistics ---
Average Test Accuracy: 84.XX ± X.XX
Average Test Kappa:    0.7XX ± 0.XXX
Total Training Time: XXX.XX min
```

### エラーの確認
```bash
# エラーメッセージ検索
grep -i "error\|exception\|failed" results/*.log
```

---

## 🎯 期待される最終結果

### 保守的シナリオ（確率70%）
```
Phase 1: Baseline 84.0 ± 2.0%
Phase 2: OTTA 86.0 ± 2.5% (SAL=0.40)

→ 論文投稿可能 ✓
```

### 標準シナリオ（確率25%）
```
Phase 1-2: OTTA 86.5 ± 2.0%

→ 国際会議採択レベル ✓✓
```

### 楽観的シナリオ（確率5%）
```
Phase 1-3: OTTA 87.5 ± 2.0%

→ SOTA級 ✓✓✓
```

---

## 📅 推奨スケジュール

```
Day 1（今日）:
  - Phase 1開始（5-seed Baseline）
  - 所要時間: 17.5時間（夜間実行）

Day 2（明日）:
  - Phase 1結果確認
  - Phase 2開始（SAL Sweep）
  - 所要時間: 8時間

Day 3（2日後）:
  - Phase 2結果分析
  - 最適SAL決定
  - 追加実験判断

Day 4（3日後）:
  - 最終設定確定
  - 論文投稿準備
```

---

## 🔗 関連ドキュメント

- [ANALYSIS_OTTA_GAPS.md](ANALYSIS_OTTA_GAPS.md) - OTTA詳細分析
- [COMPREHENSIVE_ANALYSIS_SUMMARY.md](COMPREHENSIVE_ANALYSIS_SUMMARY.md) - 包括的分析
- [NEXT_STEPS.md](NEXT_STEPS.md) - 次のステップ

---

**最終更新**: 2026-03-09
**ステータス**: 準備完了
**次のアクション**: Phase 1実験開始
