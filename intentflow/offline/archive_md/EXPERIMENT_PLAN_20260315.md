# 実験計画: Update Operator比較 (v2) — 2026-03-15

## 1. 背景と動機

### 前回実験の問題点

2026-03-14に実施した5条件update operator比較に以下の問題が発覚した:

1. **源モデルのunderfit**: OTTA configの`max_epochs=500`で学習したが、論文は`max_epochs=1000`。結果、全体精度が80.32%（論文値83.06%に対して-2.74%）
2. **S2の異常低下**: S2は500epで58.33%だが、1000epでは71.88%（+13.55%）。500ep時点でval_acc=79.3%に対しtest_acc=58.33%と、val-testギャップが21%に達しており、session間ドメインシフトが大きい被験者で学習不足が深刻だった
3. **tent_bn / tent_bn_lnの実行失敗**: PyTorch Lightning 2.1.3の`_EvaluationLoop`が`torch.inference_mode()`を使用しており、`torch.enable_grad()`でオーバーライドできない。Tentの勾配ベース適応が動作しなかった

### 前回実験の有効な知見（再利用可能）

- **bn_stat vs bn_stat_clean**: bn_stat_cleanが微差で優位（+0.24%）。Dropout/DropPathノイズは副次的要因
- **ゲーティング品質**: 適応されたサンプルの正答率97%、適応率20%。ゲートは保守的だが高精度
- **NTR-S**: 両条件とも22.2%（2/9被験者が悪化）。安全性に課題あり

## 2. 今回の実験設計

### 2.1 変更点（前回からの差分）

| 項目 | 前回 | 今回 | 理由 |
|------|------|------|------|
| max_epochs | 500 | **1000** | 論文準拠。1000ep baselineで82.79% ≈ 論文83.06%を確認済み |
| Early stopping | なし | **なし** | S2等のsession shift被験者はvalが頭打ちでもtestが改善し続ける。Early stoppingはこれを害する |
| tent実装 | `torch.enable_grad()` | **`torch.inference_mode(False)`** | PL 2.1.3対応。inference_modeをオーバーライドする正しい方法 |
| val split | session_T 80/20 | **session_T 80/20（変更なし）** | 監視用。学習には80%使用。baseline 82.79%で論文水準を確認済み |

### 2.2 5条件の定義

| 条件名 | 適応操作 | 更新対象 | 勾配 | Dropout/DropPath |
|--------|----------|----------|------|------------------|
| **source_only** | なし（enable_otta=false） | — | — | — |
| **bn_stat** | model.train() → forward | BN running stats | なし | **ON**（model.train()の副作用） |
| **bn_stat_clean** | BN層のみtrain() → forward | BN running stats | なし | **OFF** |
| **tent_bn** | BN affine params entropy min | BN weight/bias (24 params) | あり | OFF |
| **tent_bn_ln** | BN+LN affine params entropy min | BN+LN weight/bias (40 params) | あり | OFF |

### 2.3 各条件で検証する仮説

1. **source_only**: 適切に学習された源モデルのベースライン精度。論文値（~83%）と一致するか
2. **bn_stat vs source_only**: 現行OTTAの真の効果量。前回は+0.54%だったが、properly trainedモデルで変わるか
3. **bn_stat_clean vs bn_stat**: Dropout/DropPathノイズの影響。前回は+0.24%だったが、モデル品質が変わると差が変わるか
4. **tent_bn vs bn_stat_clean**: 勾配ベース適応 vs 統計量更新。更新操作の質が本当にボトルネックか
5. **tent_bn_ln vs tent_bn**: LayerNorm（Transformer部）まで適応範囲を拡大する効果。TCFormerのBN(Conv/TCN) vs LN(Transformer)のどちらが適応に有効か

### 2.4 評価指標

| 指標 | 定義 | 論文上の意味 |
|------|------|-------------|
| Mean Accuracy | 9被験者の平均テスト精度 | 適応の有効性 |
| NTR-S (Negative Transfer Rate, Subject-level) | 精度が悪化した被験者の割合 | 適応の安全性 |
| WSD (Worst-Subject Delta) | 最大の精度低下幅 | 最悪ケースのリスク |
| Adaptation Rate | 適応が実行された試行の割合 | ゲーティングの保守度 |
| Per-subject delta | 各被験者のsource_onlyからの変化量 | 被験者依存性の構造 |

## 3. 技術的修正事項

### 3.1 max_epochs修正（完了）

```yaml
# intentflow/offline/configs/tcformer_otta/tcformer_otta.yaml
max_epochs: 1000  # Changed from 500 to match paper protocol
```

### 3.2 tent_bn / tent_bn_ln の inference_mode 修正（未実施）

**問題**: PL 2.1.3の`_EvaluationLoop.run()`が`torch.inference_mode()`を使用。`torch.enable_grad()`では上書きできない。

**修正方針**: `_update_bn_stats`のtentブランチで`torch.inference_mode(False)`コンテキストを使用する。

```python
# pmax_sal_otta.py の tent_bn/tent_bn_ln ブランチ
# 変更前:
with torch.enable_grad():
    logits = self.model(x)

# 変更後:
with torch.inference_mode(False):
    with torch.enable_grad():
        logits = self.model(x)
```

**検証方法**: 修正後、単一被験者で`adapt_mode=tent_bn`を実行し、backward()が成功することを確認。

### 3.3 Early Stoppingについて（導入しない）

**理由**:
- 論文プロトコルは1000epoch固定、最終チェックポイント使用
- session_T内のval_lossが頭打ちでも、session_E（test）への汎化は改善し続ける被験者がいる（S2: val +3.5% → test +13.55%）
- 源モデルの学習プロトコルに変更を入れると、OTTA効果と学習プロトコル差の切り分けが困難になる
- val_datasetは監視用として残す（学習後の分析に使用可能）

**将来の検討**: 論文投稿後のablationとして、early stopping patience=100/200 の比較を検討。

## 4. 実行計画

### 4.1 前提条件

- GPU: 1x (GPU 0)
- 推定時間: 1条件 × 9被験者 × ~24min = ~3.6時間/条件
- 全5条件: ~18時間
- OTTA configのmax_epochs=1000（修正済み）
- tent修正（3.2節）を実施後に実行

### 4.2 実行順序

```
1. tent_bn修正を実施・検証（単一被験者で動作確認）
2. 5条件を順次実行:
   source_only → bn_stat → bn_stat_clean → tent_bn → tent_bn_ln
3. 全条件完了後に包括的分析
```

### 4.3 成功基準

1. source_onlyの精度が82-84%の範囲に入ること（論文再現の確認）
2. tent_bn / tent_bn_lnが実行エラーなく完了すること
3. 全9被験者×5条件の結果が揃うこと

## 5. 期待される結果と分岐シナリオ

### シナリオA: tent_bn > bn_stat_clean > bn_stat > source_only

→ 更新操作の質がボトルネックだった。勾配ベース適応が有効。
→ 論文ストーリー: 「ゲーティングは適応タイミングの制御に有効だが、BN stat更新では効果が限定的。Tent + tri-lock gatingの組み合わせが最適」

### シナリオB: tent_bn ≈ bn_stat_clean ≈ bn_stat（全て source_only と大差なし）

→ TCFormerのアーキテクチャがtest-time adaptationに向いていない（BN層の寄与が小さい）
→ 論文ストーリー: 「BNパラメータが全体の1.8%のモデルでは、BN-based OTTAの効果は本質的に限定的。アーキテクチャ選択が適応性能の上限を決める」

### シナリオC: tent_bnがnegative transferを悪化させる

→ 勾配ベース適応はノイジーなEEGデータに対して過剰反応する
→ 論文ストーリー: 「"When Not to Adapt"の重要性が更に強調。勾配ベースOTTAはゲーティングなしでは危険」

### シナリオD: tent_bn_ln >> tent_bn（LN適応が大きく効く）

→ TransformerのLayerNormが適応の主要経路
→ 論文ストーリー: 「CNN-Transformer hybridモデルでは、Conv/TCNのBNよりTransformerのLNの適応が重要」

## 6. 前回結果との比較ポイント

今回の結果は前回（500ep）と以下の点で比較する:

| 比較項目 | 前回（500ep） | 今回（1000ep） | 解釈 |
|----------|--------------|---------------|------|
| source_only精度 | 80.32% | ~83%（期待） | 源モデル品質の改善 |
| OTTA効果量 (bn_stat - source) | +0.54% | ? | 適切な源モデルでの真のOTTA効果 |
| NTR-S | 22.2% | ? | 源モデル品質がnegative transferに影響するか |
| S2の精度 | 58.33% | ~72%（期待） | underfit解消の確認 |
| bn_stat_clean - bn_stat差 | +0.24% | ? | Dropout/DropPathノイズ効果がモデル品質に依存するか |

## 7. 議論用の論点

Codexとの議論で検討すべきポイント:

1. **ゲーティング vs 更新操作のどちらが本質的か**: tent結果で判明する。もしtentでもNTR-Sが改善しないなら、ゲーティングが不十分
2. **BNパラメータ1.8%の限界**: TCFormerでBN-based OTTAが効かないなら、他のアーキテクチャ（BN比率が高いモデル）でも検証すべきか
3. **tent_bn_lnのリスク**: LN適応はTransformerの表現力を直接変える。勾配の大きさやstep sizeの制御が重要になる
4. **seed=0単一の限界**: 5-seed平均をどの段階で取るべきか。5条件×5seeds×9被験者 = 225実験は現実的か
5. **論文のframing**: シナリオB（全て効かない）の場合、「negative result」として論文が成立するか
