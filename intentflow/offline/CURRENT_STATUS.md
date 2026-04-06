# IntentFlow OTTA研究 — 現状整理 (2026-03-15)

## この研究は何をしているか

TCFormer（EEG Motor Imagery分類モデル）に対して、テスト時適応（OTTA）を加えて精度と安全性を両立できるか検証している。論文の方向性は **"When Not to Adapt"** — 全サンプルを適応するのではなく、信頼できるサンプルのみで適応し、危険な適応を回避する。

## モデル構成

- **Source model**: TCFormer (77.8k params) = MK-CNN + Transformer(GQA) + TCN
  - BN層12個 (Conv部 + TCN部), LN層4個 (Transformer部)
- **OTTA**: tri-lock gating (pmax × SAL × energy) で選別 → BN running stats更新
- **データ**: BCIC-IV 2a, 4クラスMI, 9被験者, session1=train / session2=test

## 確認済みの事実

### 源モデル精度
| 設定 | 精度 | 備考 |
|------|------|------|
| 論文値 (augあり, 5seeds) | 84.79 ± 0.43% | ターゲット |
| 論文値 (augなし, 5seeds) | 83.06 ± 0.54% | |
| 我々 1000ep baseline (seed=0) | 82.79% | 論文に近い。session_T 80%で学習 |
| 我々 500ep (OTTA config, seed=0) | 80.32% | **underfit。S2=58.33%** |

### OTTA効果 (500ep源モデル上、参考値)
| 条件 | 精度 | vs source | NTR-S |
|------|------|-----------|-------|
| source_only (500ep) | 80.32% | — | — |
| bn_stat (現行OTTA) | 80.86% | +0.54% | 22.2% (2/9) |
| bn_stat_clean (Dropout off) | 81.10% | +0.77% | 22.2% (2/9) |
| tent_bn | 失敗 | — | — |
| tent_bn_ln | 失敗 | — | — |

**注意**: 上記は500ep underfitモデル上の結果。1000epモデルでの再測定が必要。

### 判明した問題点
1. **500ep underfit**: OTTA configのmax_epochsが500だった。1000に修正済み
2. **tent失敗**: PL 2.1.3がinference_modeを使用。enable_gradで上書き不可。修正未実施
3. **バッチ汚染**: 適応後にバッチ全体を再forward。skippedサンプルも更新後モデルで推論される
4. **train-once問題**: 5条件で毎回再学習している。checkpoint共有に変えるべき
5. **calibration**: prototype/energyがaugmented dataで計算されている（論文で明記すればOK）

## コード変更の現状

| ファイル | 変更内容 | 状態 |
|----------|----------|------|
| tcformer_otta.yaml | max_epochs: 500→1000 | **完了** |
| pmax_sal_otta.py | tent_bn/tent_bn_ln 5条件対応 | 完了（tentは動作しない） |
| pmax_sal_otta.py | inference_mode(False) 修正 | **未実施** |
| train_pipeline.py | EarlyStopping無効化のコメント更新 | 完了 |
| train_pipeline.py / 実行スクリプト | train-once/test-many | **未実施** |
| pmax_sal_otta.py | test batch_size=1 対応 | **未実施** |

## 次にやるべきこと（優先順）

### 1. tent修正 (inference_mode)
pmax_sal_otta.pyのtentブランチで`torch.inference_mode(False)`を追加。単一被験者で動作確認。

### 2. train-once/test-many化
源モデルを1回だけ1000epで学習 → checkpointを保存 → 各OTTA条件はロードしてテストのみ。
- 実験時間: 18時間 → 約4時間に短縮
- 比較の純度: 全条件が完全に同じ源モデルを使用

### 3. test batch_size=1
OTTA評価時のbatch_sizeを1にする。オンラインBCIの実運用シミュレーション。バッチ汚染問題を解消。

### 4. 5条件比較の再実行
source_only / bn_stat / bn_stat_clean / tent_bn / tent_bn_ln

### 5. 結果分析と論文方針の確定

## ファイル構成

```
intentflow/offline/
  CURRENT_STATUS.md     ← このファイル（現状の全体像）
  PAPER_STRATEGY.md     ← 論文戦略 ("When Not to Adapt")
  QUICK_START.md        ← 実験実行手順
  README_Hybrid.md      ← Hybridモデル仕様
  archive_md/           ← 過去の分析・ステータス（参照用）
```
