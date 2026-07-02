# V1/V2: selection ablation and random top-k controls — 2026-06-27

## 結論

**E5bの主張は強化された。**

V1/V2の問いは次だった。

```text
E5bの改善は、本当に longitudinal class-stability score の効果か？
それとも単なる top-k 次元削減 / 正則化 / random subset でも起きるのか？
```

結果:

```text
random top-k は大きく負ける。
source-only Fisher top-k は改善するが、longitudinal score に明確に負ける。
separation without drift も改善するが、longitudinal score に明確に負ける。
drift-only は崩壊する。
```

したがって、現時点で守れる主張はこれ。

```text
top-k subspace restriction is necessary but not sufficient.
The same-class cross-session drift penalty adds measurable value beyond
ordinary source discriminability or separation-only feature selection.
```

日本語では、

```text
単なる次元削減ではなく、同一クラスがセッションを跨いでどれだけ動くかを罰する
縦断的安定性項が、追加の性能改善を生んでいる。
```

## 実装

Script:

```text
intentflow/offline/scripts/analysis/stieger_longitudinal_selection_ablation.py
```

Output:

```text
intentflow/offline/results/research_outputs/260627_stieger_longitudinal_selection_ablation/
```

Command:

```bash
/home/islabshi/anaconda3/envs/intentflow/bin/python \
  intentflow/offline/scripts/analysis/stieger_longitudinal_selection_ablation.py \
  --subjects 1-62 \
  --output-dir intentflow/offline/results/research_outputs/260627_stieger_longitudinal_selection_ablation \
  --fractions 1.0 0.25 0.1 \
  --random-seeds 0 1 2 3 4 \
  --bootstrap 4000 \
  --force \
  --quiet
```

Protocol:

```text
subjects = 62
rows = 1558 session-condition rows
failures = 0
source = session 1
target prefix = first 32 trials, labels unused for EA
evaluation = trial 65 onward
branches = broad_all60 + fb_sensorimotor21_mu_beta
classifier = branch-wise LDA + equal posterior fusion
held-out subject = all sessions excluded from score learning
```

Comparator:

```text
longitudinal_q1p00 = E4a equal posterior + prefix-EA equivalent
```

## Scores compared

### Full longitudinal score

```text
(source class separation + target class separation)
/
(same-class source-target drift + source within-class variance + target within-class variance)
```

### Source-only score

```text
source class separation / source within-class variance
```

### Separation without drift

```text
(source class separation + target class separation)
/
(source within-class variance + target within-class variance)
```

### Target-only score

```text
target class separation / target within-class variance
```

### Drift-only score

```text
1 / (same-class source-target drift + within-class variance)
```

### Random top-k

```text
random seeds = 0, 1, 2, 3, 4
q = 1.0, 0.25, 0.1
```

## V1: random top-k

Primary LR+UD:

| method | acc mean | gain vs full q1.00 |
|---|---:|---:|
| full q1.00 | 65.325 | 0.000 |
| random q0.25, 5 seeds mean | 61.518 | -3.807 |
| random q0.25, best seed | 61.739 | -3.586 |
| random q0.10, 5 seeds mean | 58.881 | -6.444 |
| random q0.10, best seed | 59.221 | -6.104 |
| longitudinal q0.25 | 67.082 | +1.758 |
| longitudinal q0.10 | 67.681 | +2.357 |

Interpretation:

```text
random top-k does not explain the gain.
```

Randomly dropping features destroys accuracy. Therefore the result is not simply
"using fewer dimensions".

## V2: score ablation

### Primary LR+UD

| method | acc | gain vs q1.00 | 95% CI | R10 loss vs q1.00 | P(loss<-5pp) |
|---|---:|---:|---:|---:|---:|
| q1.00 full | 65.325 | 0.000 | [0, 0] | 0.000 | 0.00% |
| source-only outer best | 66.777 | +1.452 | [+0.666, +2.233] | 12.467 | 18.03% |
| sep-no-drift outer best | 66.805 | +1.480 | [+0.683, +2.239] | 12.694 | 18.11% |
| target-only outer best | 65.988 | +0.663 | [+0.073, +1.275] | 11.917 | 12.93% |
| drift-only outer best | 65.325 | 0.000 | [0, 0] | 0.000 | 0.00% |
| longitudinal outer best | **67.734** | **+2.409** | **[+1.507, +3.345]** | 11.266 | 16.25% |

Direct paired comparisons:

| comparison | LR | UD | primary |
|---|---:|---:|---:|
| longitudinal - source-only | +1.129 [+0.389, +1.903] | +0.825 [+0.197, +1.434] | **+0.957 [+0.445, +1.468]** |
| longitudinal - sep-no-drift | +0.964 [+0.286, +1.630] | +0.929 [+0.150, +1.702] | **+0.929 [+0.363, +1.462]** |
| longitudinal - target-only | +1.379 [+0.525, +2.267] | +2.160 [+1.277, +3.088] | **+1.746 [+1.090, +2.400]** |

Interpretation:

- source-only discriminability is a strong baseline.
- source+target separation without drift is also strong.
- But both are significantly below full longitudinal score.
- Therefore the same-class drift penalty is not decorative; it contributes.

## Condition-wise details

### LR

| method | acc | gain vs q1.00 |
|---|---:|---:|
| q1.00 | 66.873 | 0.000 |
| source-only q0.10 | 69.302 | +2.429 |
| sep-no-drift q0.10 | 69.467 | +2.594 |
| target-only q0.10 | 69.052 | +2.179 |
| longitudinal q0.10 | **70.431** | **+3.558** |

LR is where longitudinal stability is strongest.

### UD

| method | acc | gain vs q1.00 |
|---|---:|---:|
| q1.00 | 63.657 | 0.000 |
| source-only q0.25 | 64.168 | +0.512 |
| sep-no-drift q0.25 | 63.919 | +0.262 |
| target-only q0.25 | 63.609 | -0.048 |
| longitudinal q0.25 | **64.848** | **+1.192** |

UD gain is smaller but still positive.

## 守る / 修正 / 撤回

### 守る

```text
Longitudinal class-stability score improves zero-target-label cross-session EEG-MI
beyond ordinary source-only or separation-only feature selection.
```

Evidence:

```text
primary longitudinal - source-only = +0.957pp [0.445, 1.468]
primary longitudinal - sep-no-drift = +0.929pp [0.363, 1.462]
primary longitudinal - target-only = +1.746pp [1.090, 2.400]
```

### 修正

```text
The gain is not purely from longitudinal stability.
```

Source-only and separation-only top-k already improve over q1.00.
Therefore a fair story must say:

```text
feature subspace restriction gives a large regularization gain,
and longitudinal stability gives an additional significant gain.
```

### 撤回

```text
drift-only stability is enough.
```

False. Drift-only q0.25 and q0.10 collapse badly:

```text
primary drift-only q0.25: -8.561pp
primary drift-only q0.10: -10.719pp
```

Stability without class separation selects uninformative dimensions.

### 禁止

```text
random top-k baseline is optional.
```

No. It is mandatory, but now it is passed.

## 研究上の意味

この結果で、主張の骨格はかなり固くなった。

```text
1. broad/neuro fixed fusion gives average gain but lower-tail remains.
2. class-conditional geometry explains branch gap better than unlabeled global shift.
3. source-side top-k subspace selection gives a large accuracy gain.
4. random top-k fails.
5. source-only and sep-only improve, but full longitudinal score is significantly better.
```

したがって、次の主軸はこれでよい。

```text
ソース側縦断データに基づく部分空間選択により、
ターゲットラベルなしセッション間EEG-MIのrisk-utility frontierを改善する。
```

## 次にやるべきこと

次は実装探索ではなく、**論文化に必要な外的妥当性と可視化**に移る。

Priority:

```text
V3 second multi-session MI dataset
V4 risk-utility frontier figure
V5 topomap / selected feature interpretation
```

V3が同方向なら、修士研究としてかなり戦える。

V3で落ちた場合は、Stieger固有の feature-regularization 現象として格下げし、
workshop / 修論主軸に留める。
