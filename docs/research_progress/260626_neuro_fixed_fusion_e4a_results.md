# E4a: 固定 broad/neuro 融合 — 2026-06-26

## 結論

**平均精度は pass、safe zero-label adaptation としては fail。**

`broad_all60` と `fb_sensorimotor21_mu_beta` をtarget sessionごとに
選択せず、同一trial上で固定融合した。primary (LR + UD) では、
`prefix-EA` 後の等重み posterior fusion が、LOSOで選んだ最強の固定単一branchに対して
**+1.51 pp [95% CI +0.81, +2.22]** だった。

しかし、同じ比較器に対する下位10%の平均損失は **11.24 pp**、
`P(fusion - fixed < -5pp)` は **16.15%** である。従って、事前に決めた
安全ゲート（平均 +1pp、CI下限 > 0、R10損失 <= 1pp、5pp超損失 <= 5%）は満たさない。

この結果を「安全な無ラベル適応ができた」とは呼ばない。一方、E2の失敗は
「broad/neuroの情報に価値がない」ことではなく、**session単位でwinnerを選ぶ問題設定が
誤っていた**ことを示す。二つの予測をtrial単位で同時に使うと、特にUDで平均精度を
有意に改善できる。

## 実装と因果制約

実装:

```text
intentflow/offline/scripts/analysis/stieger_neuro_fixed_fusion.py
```

出力:

```text
intentflow/offline/results/research_outputs/260626_stieger_neuro_fixed_fusion/
```

プロトコル:

```text
Stieger2021, 62 subjects
source = session 1 labels
target prefix = first 32 trials, labels unused, prefix EAのみ
evaluation = trial 65 onward, n_eval >= 40
outer comparator = leave-one-subject-out
primary = pure_lr + pure_ud
```

target sessionの正解ラベルは、fusion weight、classifier hyperparameter、
branch選択に一切使用していない。

比較したもの:

```text
single branch:
  broad / neuro × source or prefix-EA

fixed posterior fusion:
  log p(y|x) = w log p_broad(y|x) + (1-w) log p_neuro(y|x)
  w = 0.5 (equal) または source S1の5-fold OOFから選択

fixed feature concatenation:
  source S1でblock-wise z-score + block-energy normalization
  Ridge classifier
  broad/neuro weight と ridge alpha は source S1の5-fold OOFから選択
```

ここでのLOSOは、held-out subjectのtarget labelを使わずに最強の固定単一branchを
定めるための比較器選択である。target sessionごとのrouterではない。

## 主結果

| condition | method | gain vs outer best fixed | 95% CI | R10 loss | P(loss < -5pp) |
|---|---|---:|---:|---:|---:|
| LR | equal posterior + prefix-EA | +0.52 | [-0.41, +1.45] | 12.09 | 18.23% |
| UD | equal posterior + prefix-EA | **+2.66** | **[+1.54, +3.85]** | 10.24 | 13.82% |
| 2D diagnostic | equal posterior + prefix-EA | +1.75 | [+1.11, +2.38] | 6.44 | 6.74% |
| LR + UD primary | equal posterior + prefix-EA | **+1.51** | **[+0.81, +2.22]** | **11.24** | **16.15%** |

最強の単一branch比較器は、ほぼ常に
`fb_sensorimotor21_mu_beta + prefix-EA` だった（LRは100%、UDは96.2%）。

source-only fusion は一貫して悪化した。例えばprimaryで、等重みposteriorは
-3.87pp、source-OOF concatは -3.95ppである。今回の平均精度の利益は、
`prefix-EA` を捨てた表現では得られない。

## 何が効かなかったか

source S1で複雑さを足しても、等重みposteriorを超えなかった。

| primary LR + UD | gain vs outer best fixed | 95% CI | R10 loss |
|---|---:|---:|---:|
| posterior equal + prefix-EA | **+1.51** | **[+0.81, +2.22]** | 11.24 |
| posterior S1-OOF weight + prefix-EA | +1.29 | [+0.57, +2.04] | 11.70 |
| concat S1-OOF + prefix-EA | +1.20 | [+0.50, +1.88] | 10.63 |

従って、現時点で守れるのは以下である。

```text
simple fixed posterior fusion has an accuracy effect, mainly in UD.
```

以下は撤回する。

```text
source-only learned fusion improves over a simple fixed fusion.
fixed fusion solves the safety / worst-case problem.
```

## 論理的な更新

E1bのoracle相補性に対し、E2のsession-level label-free selectorは失敗した。
E4aは、その間にもう一つの構造があることを示す。

```text
session-level: どちらのbranchを採用するか
  -> 無ラベルprefixからは読めない (E2)

trial-level: 両branchのclass evidenceを同時に用いる
  -> 平均精度は回収可能 (E4a)
```

ただし、trial-levelの等重み融合も約16%のsessionで固定branchより5pp超悪化する。
これはE2/G1aと整合する。target prefixに見える無ラベルsignalだけでは、
「このsessionでは融合が危険か」をまだ読めない。

## 次の判断

### しない

```text
- dynamic electrode selection / channel gate の再開
- session-level label-free selector の特徴量追加
- equal posterior fusion を新手法として論文化
- safe fusion としての主張
```

### 進める条件付きの本線

平均精度を主目的に残すなら、次は単なるfusionの改良でなく、
**class-conditional longitudinal representation learning** の小規模mechanism pilotに進む。

目的は、学習被験者の複数sessionから

```text
same subject + same class + different session
```

を近づけ、異なるMI classを分離したまま保つ表現を学習することにある。評価被験者の
全sessionはpretrainingから除外し、S1だけでheadを作り、S2+はtarget labelなしで評価する。

E4aの固定融合は、このpilotの強いaccuracy baselineにする。ただし、pilotが

```text
primary +2pp以上, CI下限 > 0, R10を悪化させない
```

を満たさないなら、zero-target-label adaptationの本線は終了する。

安全を主目的に戻すなら、E4aは失敗であり、このデータと現行target-prefix signalだけで
安全な融合を目指す探索は終了する。
