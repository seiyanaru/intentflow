# 2026-06-01 EA-aware TCFormer: success/failure analysis

## 結論

BCIC2a seed0 のEA-aware再学習は、full EA固定では平均 +0.35pp に止まった。理由は「EAが効く被験者」と「full whiteningが壊す被験者」が混在するため。悪化はランダムな汎化失敗ではなく、S5/S7で特定クラスへの混同増加として出ている。

今回の追加実験では、S5/S7は共分散shrinkageで回復したが、S2は同じshrinkageで崩壊した。したがって次の主軸は「固定EA」ではなく、test-timeの共分散/予測信頼度でEA強度を選ぶ Reliability-Aware Alignment に置くべき。

## Full EA 結果

| subj | source | full EA | delta |
|---:|---:|---:|---:|
| S1 | 85.76 | 87.85 | +2.09 |
| S2 | 72.57 | 75.69 | +3.12 |
| S3 | 93.06 | 92.36 | -0.70 |
| S4 | 81.94 | 83.33 | +1.39 |
| S5 | 77.43 | 72.57 | -4.86 |
| S6 | 70.49 | 73.61 | +3.12 |
| S7 | 90.97 | 83.68 | -7.29 |
| S8 | 84.38 | 90.28 | +5.90 |
| S9 | 87.85 | 88.19 | +0.34 |

Mean: source 82.72, full EA 83.06, delta +0.35pp.

## 追加 ablation

| subj | source | full EA | power=.5, shrink=.1 | power=1, shrink=.03 | power=1, shrink=.1 | best tested |
|---:|---:|---:|---:|---:|---:|---:|
| S2 | 72.57 | 75.69 | 67.36 | 52.78 | 47.22 | full EA |
| S5 | 77.43 | 72.57 | 75.69 | 77.08 | 77.78 | shrink .1 |
| S7 | 90.97 | 83.68 | 92.36 | 92.36 | 93.06 | shrink .1 |
| S8 | 84.38 | 90.28 | 88.19 | 88.89 | 90.28 | full/shrink .1 |

Best-tested oracle mean over 9 subjects, using source/full plus tested variants where available: 84.76%. This is not a publishable claim because it uses oracle selection, but it estimates adaptive alignment headroom at +2.04pp over source and +1.70pp over full EA.

## What broke

S5 full EA:
- Main new errors: c2->c3 17->31, c1->c3 4->15, c0->c3 6->13.
- Interpretation: full EA pushed predictions toward c3. shrink=.1 restored accuracy to 77.78 and reduced c2->c3 to 10.

S7 full EA:
- Main new errors: c2->c3 8->19, c0->c3 3->10.
- shrink=.1 reached 93.06, above source 90.97. This is the clearest case that full EA was over-aligning.

S2 shrinkage:
- full EA is good at 75.69, but shrink=.1 collapses to 47.22 and shrink=.03 to 52.78.
- Bad shrinkage predictions are class-imbalanced: shrink=.1 predicted [169, 25, 13, 81] vs balanced test prior [72, 72, 72, 72].
- Interpretation: S2 needs strong whitening. Uniform shrinkage removes necessary session correction.

S8:
- Robust winner. full EA and shrink=.1 are both 90.28. This is the case we want RAA not to damage.

## Covariance diagnostics

| subj | cov shift | test cond | test diag CV | diag ratio | c2-c3 pre | c2-c3 after EA | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| S2 | 0.164 | 40446 | 0.101 | 1.47 | 0.248 | 0.877 | full EA needed, shrinkage toxic |
| S5 | 0.169 | 15747 | 0.195 | 2.21 | 0.267 | 0.853 | channel variance imbalance high; shrinkage helps |
| S7 | 0.066 | 5144 | 0.056 | 1.31 | 0.785 | 0.658 | full EA reduces c2-c3 separation; shrinkage helps |
| S8 | 0.053 | 11909 | 0.054 | 1.26 | 0.587 | 1.087 | EA increases c2-c3 separation; strong gain |

S5 supports the channel-reliability hypothesis most directly. S7 is more subtle: obvious channel variance imbalanceは小さいが、full EAがdiscriminative subspaceを壊している。RAAは「ノイズ電極だけ」では狭すぎるので、共分散信頼度 + alignment strength制御として定義する方がよい。

## Test-time rejection signal

Prediction prior KL to balanced prior may catch bad alignment:
- S2 shrink=.1: KL 0.365, acc 47.22
- S2 shrink=.03: KL 0.369, acc 52.78
- S5 full EA: KL 0.090, acc 72.57
- S5 shrink=.1: KL 0.029, acc 77.78
- S7 full EA: KL 0.037, acc 83.68
- S7 shrink=.1: KL 0.006, acc 93.06
- S8 full EA: KL 0.005, acc 90.28

ただしprior KLだけではsource vs full/shrinkの最良選択まではできない。役割は「明らかに悪い整列の拒否」に限定する。

## Next

1. Run shrink=.1 on remaining S1/S3/S4/S6/S9 to map whether S2 is the only toxic case.
2. Implement adaptive RAA candidates:
   - full EA reference: `R`
   - shrink reference: `R_lambda = (1-lambda)R + lambda * tr(R)/C * I`
   - weighted reference: `R_w = mean(D_w x x^T D_w)`, `D_w=diag(sqrt(w_c))`
   - candidate rejection by prediction prior KL and confidence collapse.
3. Do not claim gain from oracle selection. The valid claim must be: adaptive rule chosen without labels improves mean and reduces worst-case harm.
