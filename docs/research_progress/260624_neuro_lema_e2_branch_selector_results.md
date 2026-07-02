# Neuro-LEMA E2 branch selector results — 2026-06-26

## 結論

E2の初回結果は **weak / mostly negative**。

E1bでは broad branch と neuro branch のoracle complementarityが大きかったが、target labelなしのprefix signalだけでは、そのheadroomをほとんど回収できなかった。

守れる結果：

```text
UDでは label-free ridge selector が best fixed arm に対して +0.54pp。
さらに P(Δ<-5pp) は 18.60% → 13.76% に改善。
```

ただし、bootstrap CIは0を跨ぐ。

```text
UD gain vs best fixed = +0.54pp, 95% CI [-0.80, +1.73]
```

したがって、現時点で

```text
label-free branch selectorでoracle headroomを回収できた
```

とは主張できない。

## 実装

Script:

```text
intentflow/offline/scripts/analysis/stieger_neuro_branch_selector.py
```

Output:

```text
intentflow/offline/results/research_outputs/260624_stieger_neuro_branch_selector/selector_summary.json
intentflow/offline/results/research_outputs/260624_stieger_neuro_branch_selector/selector_predictions.csv
intentflow/offline/results/research_outputs/260624_stieger_neuro_branch_selector/branch_selector_table.csv
```

Command:

```bash
/home/islabshi/anaconda3/envs/intentflow/bin/python \
  intentflow/offline/scripts/analysis/stieger_neuro_branch_selector.py \
  --subjects 1-62 \
  --cache-dir /home/islabshi/workspace-local2/mne_data/stieger_neuro_band_cov_cache \
  --e1-summary-json intentflow/offline/results/research_outputs/260624_stieger_neuro_feature_baseline/summary.json \
  --output-dir intentflow/offline/results/research_outputs/260624_stieger_neuro_branch_selector \
  --bootstrap-instability 0 \
  --outer loso \
  --quiet \
  --force
```

## E2 design

Arms:

```text
broad_all60__source
broad_all60__prefix_ea
fb_sensorimotor21_mu_beta__source
fb_sensorimotor21_mu_beta__prefix_ea
```

Label-free prefix signals:

```text
source-target covariance shift
EA reference shift
prefix confidence / entropy / margin
source vs prefix-EA prediction disagreement
broad vs neuro prediction disagreement
source-session train accuracy
session number / n_eval
```

Evaluation:

```text
outer = leave-one-subject-out
inner = GroupKFold on training subjects
model = Ridge regression predicting each arm's absolute target accuracy
selection = choose arm with highest predicted accuracy
```

Two selectors were evaluated:

```text
ridge_acc_selector:
  4-arm selector including source arms

ridge_prefix_pair_selector:
  2-arm selector between broad_all60__prefix_ea and fb_sensorimotor21_mu_beta__prefix_ea
```

## Main results

The strict comparator is the best fixed arm by absolute adapted accuracy.

### pure_lr

| method | acc | gain vs best fixed | R10 vs broad source | P(Δ<-5pp) |
|---|---:|---:|---:|---:|
| broad prefix EA | 63.99 | -2.62 | 9.79 | 9.89% |
| neuro prefix EA | 66.61 | 0.00 | 10.24 | 9.78% |
| ridge prefix pair | 66.61 | 0.00 | 10.24 | 9.78% |
| ridge 4-arm | 66.61 | 0.00 | 10.24 | 9.78% |
| oracle | 70.19 | +3.58 | 0.00 | 0.00% |

Interpretation:

- LRではselectorは完全に `fb_sensorimotor21_mu_beta__prefix_ea` に退化した。
- 平均精度では妥当だが、oracle headroomはまったく回収していない。

### pure_ud

| method | acc | gain vs best fixed | 95% CI | R10 vs broad source | P(Δ<-5pp) |
|---|---:|---:|---:|---:|---:|
| broad prefix EA | 62.08 | -0.13 | - | 11.86 | 14.84% |
| neuro prefix EA | 62.20 | 0.00 | - | 14.75 | 18.60% |
| ridge prefix pair | 62.71 | +0.51 | [-0.66, 1.74] | 13.40 | 15.65% |
| ridge 4-arm | 62.74 | +0.54 | [-0.80, 1.73] | 12.54 | 13.76% |
| oracle | 67.79 | +5.59 | [4.61, 6.60] | 0.00 | 0.00% |

Interpretation:

- UDだけはselectorが少し上がった。
- 特に4-arm selectorはsource armsを一部選び、P(Δ<-5pp)を下げた。
- しかしCIが0を跨ぐため、強い成功ではない。

### two_d

| method | acc | gain vs best fixed | 95% CI | R10 vs broad source | P(Δ<-5pp) |
|---|---:|---:|---:|---:|---:|
| broad prefix EA | 35.30 | -0.10 | - | 10.40 | 11.99% |
| neuro prefix EA | 35.40 | 0.00 | - | 11.82 | 16.45% |
| ridge prefix pair | 35.39 | -0.01 | [-0.71, 0.63] | 11.51 | 13.92% |
| ridge 4-arm | 35.15 | -0.25 | [-0.94, 0.42] | 11.86 | 14.41% |
| oracle | 39.96 | +4.56 | [4.03, 5.12] | 0.00 | 0.00% |

Interpretation:

- two_dではselectorはbest fixedを超えない。
- oracle headroomは大きいので、branchの相補性はある。
- しかしprefix label-free signalからは勝ちbranchを読めていない。

## 判断

### 守る

```text
broad/neuro branch complementarity is real.
oracle headroom is large.
```

### 修正

```text
E2のlabel-free ridge selectorは、UDで弱い改善を示すが、
LR/2Dではbest fixedを超えない。
```

### 撤回

```text
generic prefix label-free signals are sufficient to recover oracle branch headroom.
```

これは撤回。

## 研究上の意味

今回の結果は、以前のsafe selector失敗と同じ警告を出している。

```text
oracleでbranch complementarityが見えても、
label-free signalだけでは正しく選べない。
```

したがって、ここから単純にselector特徴量を増やすのは危険。

次に進むなら、以下のどちらかに絞るべき。

### Option A: 少数ラベルあり branch calibration

E2の結果からは、完全label-free selectorよりも、少数ラベルでbranchを校正する方が筋が良い。

Test:

```text
k = 4, 8, 16 labeled prefix trials
estimate each branch's prefix validation accuracy
choose branch or shrink toward best fixed
evaluate suffix
```

Success:

```text
best fixed +1pp以上
R10悪化なし
```

### Option B: selectorを捨てて fixed representation を強くする

label-free selectionが難しいなら、sessionごとに選ぶのではなく、

```text
broad + neuro feature concatenation
```

をsource trainingから一体で学習する方が良い。

これは以前の教訓である

```text
入力を変えるなら学習も変える
```

に沿う。

Test:

```text
concat(broad_all60 tangent, fb_sensorimotor21_mu_beta tangent)
train source LDA with shrinkage
source / prefix_ea / shrinked prefix_ea
```

Success:

```text
best fixed branchより +1pp
または same accuracy with lower R10
```

## 次アクション推奨

E2をこれ以上generic label-free selectorとして深掘りしない。

次は **fixed broad+neuro representation** を試すべき。

理由：

- E1でneuro source representationは強かった。
- E1bでbroad/neuroの相補性は大きい。
- E2でlabel-free selectionは弱かった。
- よって、session-level selectionではなく、source training時点で相補性を吸収する方が論理的。
