# G1a: channel-quality harm audit — 2026-06-26

## 結論

**Fail / stop.** target prefixから得た無ラベルspatial-patch品質は、
`broad_all60` prefix-EAがsourceより改善するか、あるいは害を出すかを
被験者LOSOで予測できなかった。

従って、現在のStieger cacheとこの品質定義に基づく
`dynamic electrode selection` / `channel reliability gate` / `quality-aware EA`
を次の本線にはしない。高価なpatch-ablation counterfactual (G1b) も実行しない。

これは電極品質が原理的に無関係という結論ではない。cacheにはimpedance、
line noise、高周波EMGなどのhardware quality signalが無い。しかし、今回使える
無ラベルcovariance / power / spatial-consistency signalには、EA害を制御する
実証的根拠が無い。

## 問いと設計

問い:

```text
target prefixのpatch品質が悪いsessionでは、broad prefix EAが害を出すのではないか？
```

```text
Stieger2021, subjects=62
condition = pure_lr / pure_ud
source = session 1
target prefix = first 32 trials, labels unused
outcome = existing broad_all60 prefix-EA delta vs source
evaluation suffix = trial 65 onward
eligible session = n_eval >= 40
outer evaluation = leave-one-subject-out
```

60 channelsを8つの連続spatial patchに分け、各patchについてprefixだけから
次を計算した。

```text
1. source比のbroad-band log variance shift
2. prefix内のtrial-to-trial log variance instability
3. source比のspatial correlation shift
4. μ / low-β / high-βのratio shift inconsistency
```

全8 patch × 4指標をRidgeでEA deltaに回帰し、予測deltaの負値で
`P(EA delta < -5pp)` を検出した。

実装:

```text
intentflow/offline/scripts/analysis/stieger_channel_quality_harm_audit.py
```

出力:

```text
intentflow/offline/results/research_outputs/260626_stieger_channel_quality_harm_audit/
```

## 主結果

| condition | sessions | delta Spearman ρ | harm5 AUROC | harm5 rate |
|---|---:|---:|---:|---:|
| pure_lr | 523 | +0.018 | 0.535 | 9.75% |
| pure_ud | 523 | +0.037 | 0.518 | 14.91% |
| pooled | 1,046 | +0.079 | 0.542 | 12.33% |

予測相関は実質ゼロ、harm検出もchance付近である。

Ridgeの線形性だけの問題ではないかを確認するため、同じ32品質特徴にLOSO
ExtraTreesを一度だけ適用した。

| condition | ExtraTrees ρ | ExtraTrees harm5 AUROC |
|---|---:|---:|
| pure_lr | -0.054 | 0.585 |
| pure_ud | -0.041 | 0.460 |

UDでは非線形化してもchance未満であり、LRの0.585も再現性を主張できる強さではない。

## 判断

### 撤回

```text
prefixのcovariance / band-power品質を用いれば、危険な電極を無ラベルで特定し、
EA害を避けられる。
```

### 保留

```text
impedanceやline noiseなど、cacheに存在しないhardware quality signalを
追加すれば別の結果になる可能性。
```

ただし現在の公開データだけでそれを主張することはできない。

### 次にしないこと

```text
- hard top-k electrode selection
- target only channel weighting
- G1b patch ablationの全件実行
- dynamic electrode selectionをNeuro-LEMAの本体とする
```

## 研究上の意味

E2とG1aは同じ方向を指す。

```text
target prefixから見える無ラベル分布シフトは、「適応すべきか」「どの表現を使うべきか」
「どの電極を落とすべきか」を十分に判別していない。
```

無ラベルで情報が無いものをさらにgateや重みで処理する方向は止める。
