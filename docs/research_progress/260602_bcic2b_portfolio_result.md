# 260602 BCIC2b TCFormer EA/Portfolio Result

## Setup

- Dataset: BCIC-IV 2b (`bcic2b`), seed 0, TCFormer.
- Source baseline: `intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347`
- Full EA: `--ea --ea_shrinkage 0`
- Shrink EA: `--ea --ea_shrinkage 0.1`
- Evaluation: source / fullEA / shrinkEA logits portfolio, subject-level mean accuracy.
- Note: BCIC2b is loaded through MOABB. `moabb==1.2.0` was installed in `conda intentflow`; this downgraded several deps including numpy/scikit-learn.

## Main Result

| Subject | Source | Full EA | Shrink EA | Best single expert |
|---:|---:|---:|---:|---|
| S1 | 78.75 | 81.25 | 81.25 | Full EA / Shrink EA |
| S2 | 70.00 | 71.07 | 69.64 | Full EA |
| S3 | 84.38 | 82.50 | 84.69 | Shrink EA |
| S4 | 98.12 | 98.12 | 98.12 | Tie |
| S5 | 97.81 | 64.06 | 78.44 | Source |
| S6 | 83.75 | 84.06 | 84.69 | Shrink EA |
| S7 | 93.12 | 93.44 | 93.44 | Full EA / Shrink EA |
| S8 | 94.69 | 94.38 | 94.69 | Source / Shrink EA |
| S9 | 89.06 | 88.12 | 89.38 | Shrink EA |
| Mean | 87.74 | 84.11 | 86.04 | - |

## Portfolio Result

| Method | Mean | Worst | Harmed vs Source | Delta vs Source |
|---|---:|---:|---:|---:|
| Source | 87.74 | 70.00 | 0 | +0.00 |
| Full EA | 84.11 | 64.06 | 4 | -3.63 |
| Shrink EA 0.1 | 86.04 | 69.64 | 2 | -1.71 |
| Static 0.45/0.45/0.10 | 86.50 | 70.36 | 1 | -1.25 |
| Safe 0.65/0.25/0.10 | 87.78 | 70.36 | 2 | +0.04 |
| Global best, mean-minus-harm | 87.82 | 70.36 | 0 | +0.07 |
| Global best, mean objective | 88.02 | 70.00 | 2 | +0.28 |
| Global best, worst objective | 86.21 | 71.79 | 1 | -1.54 |
| LOSO, mean-minus-harm | 85.91 | - | 1 | -1.84 |

Global best under `mean_minus_harm` was almost source-only:

```text
p = 0.98 p_source + 0.01 p_fullEA + 0.01 p_shrinkEA
```

Mean-max objective gave:

```text
p = 0.87 p_source + 0.07 p_fullEA + 0.06 p_shrinkEA
```

Worst-case objective gave:

```text
p = 0.22 p_source + 0.34 p_fullEA + 0.44 p_shrinkEA
```

## Interpretation

BCIC2bでは、BCIC2aよりさらに「EAを入れるかどうか」の個体差が大きい。Full EAはS1/S2/S6/S7で少し効くが、S5で -33.75pp と致命的に壊れる。Shrink EAはS5の崩壊を 64.06 -> 78.44 まで戻すが、source 97.81 には遠い。したがって、RAA/EA系を単体手法として強く見せるのは難しい。

一方で、S1/S2/S3/S6/S7/S9にはEA系の微小gainがある。つまり結論は「EAを捨てる」ではなく、「EAを適用してよい状態を検出するselectorが必要」。BCIC2bのS5は、このselectorの失敗例として非常に強い反証ケースになる。

LOSOでもS5をheld-outにすると、訓練側にS5のような危険例がないため、EA混合を選んで 97.81 -> 80.62 に落ちる。これは、subject-idに依存した重み学習では不十分で、trial/sessionの共分散品質や信頼性特徴から危険を検出する必要があることを示す。

## Next

1. BCIC2bでもselector feature tableを作る。
   - source/full/shrinkのconfidence, entropy, margin
   - EA covariance condition number, log-det, trace, off-diagonal correlation
   - train/test covariance distance
   - channel-wise variance / kurtosis / adjacent correlation
2. S5を主なnegative caseとして、EA failure detectorを設計する。
3. Selectorの目的を「平均最大化」ではなく、「source破壊の回避 + 小gain回収」に置く。
4. BCIC2a + BCIC2bをまとめ、datasetを跨いだmeta-selectorとして評価する。

