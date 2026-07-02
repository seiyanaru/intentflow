# E5a: class-conditional geometry audit — 2026-06-27

## 結論

**Partial pass for mechanism, fail as a safety/gating signal.**

target evaluation suffix のラベルを使って、source-target の class centroid、margin、
prototype gap、LR lateralization を測った。これは deployable method ではなく、
次に representation learning へ進む根拠があるかを見る機構診断である。

結果として、class-conditional geometry は broad/neuro の branch gap や EA delta を
ある程度説明する。特に primary pooled では、class-conditioned 多変量 LOSO が
branch gap を `|rho|=0.38` まで説明し、global unlabeled geometry の `|rho|=0.04`
より明らかに強い。

一方で、fusion harm の予測は弱い。primary pooled の
`fusion_harm5_vs_outer_best_single` は、最良 univariate AUROC が `0.617`、
class-conditioned 多変量 LOSO AUROC が `0.601` に留まった。従って、

```text
class-conditional geometry を見れば安全に fusion/gate できる
```

とは言えない。

次に進むなら、**gate ではなく source-side longitudinal representation learning の
小規模pilot**に限定する。

## 実装

Script:

```text
intentflow/offline/scripts/analysis/stieger_class_conditional_geometry_audit.py
```

Output:

```text
intentflow/offline/results/research_outputs/260627_stieger_class_conditional_geometry_audit/
```

Command:

```bash
/home/islabshi/anaconda3/envs/intentflow/bin/python \
  intentflow/offline/scripts/analysis/stieger_class_conditional_geometry_audit.py \
  --subjects 1-62 \
  --output-dir intentflow/offline/results/research_outputs/260627_stieger_class_conditional_geometry_audit \
  --quiet
```

Protocol:

```text
Stieger2021
subjects = 62
rows = 1558 session-condition rows
failures = 0
source = session 1
target prefix = first 32 trials, labels unused for EA
evaluation = trial 65 onward
features = broad_all60, fb_sensorimotor21_mu_beta
outcomes = E4a fixed fusion table
diagnostic labels = evaluation suffix labels
```

## 何を測ったか

各 branch / reference について以下を計算した。

```text
reference:
  source_ref
  prefix_ea

class-conditional:
  source-target class centroid shift
  class shift / source margin
  target margin / source margin
  target centroid の nearest source prototype identity
  same-class vs wrong-class prototype gap

unlabeled/global:
  global centroid shift
  feature spread ratio
  log variance shift

LR physiology diagnostic:
  C3-C4 log-power class contrast
  μ / lowβ / highβ sign agreement
```

重要: ここでの class-conditional 指標は target suffix label を使う。
したがって test-time gate ではなく、機構解析である。

## 主結果

### Primary pooled: LR + UD

| target | best univariate | value |
|---|---:|---:|
| broad EA delta | Spearman rho | 0.305 |
| neuro EA delta | Spearman rho | 0.354 |
| branch gap: neuro - broad | Spearman rho | 0.201 |
| fusion gain vs outer best single | Spearman rho | 0.154 |
| broad EA harm5 | AUROC | 0.620 |
| neuro EA harm5 | AUROC | 0.649 |
| fusion harm5 vs outer best single | AUROC | 0.617 |

Interpretation:

- EA delta には class/prototype geometry の信号がある。
- branch gap も弱くはないが、univariate では `rho=0.201` で強くない。
- fusion harm は `0.617` で、安全判断に使える水準ではない。

### Primary pooled: multivariate LOSO

| target | global unlabeled geometry | class-conditional geometry | all geometry |
|---|---:|---:|---:|
| branch gap: neuro - broad | rho 0.040 | **rho 0.380** | rho 0.379 |
| fusion gain vs outer best single | rho 0.020 | **rho 0.205** | rho 0.192 |
| broad EA harm5 | AUROC 0.531 | AUROC 0.561 | AUROC 0.549 |
| neuro EA harm5 | AUROC 0.588 | **AUROC 0.622** | AUROC 0.609 |
| fusion harm5 vs outer best single | AUROC 0.506 | **AUROC 0.601** | AUROC 0.583 |

Interpretation:

- class-conditional geometry は branch gap には明確に効く。
- fusion gain にも少し効くが、`rho=0.205` で論文主張の核には弱い。
- harm 検出は全体に弱い。安全な無ラベル/少ラベルgateの再開根拠にはならない。

## 研究判断

### 守る

```text
branch performance の違いは、global shift だけではなく
class-conditional geometry と関係している。
```

根拠:

- primary branch gap で class-conditioned LOSO `rho=0.380`
- global unlabeled geometry は同じ target で `rho=0.040`
- broad/neuro EA delta も univariate で `rho=0.30-0.35`

### 修正

```text
class-conditional geometry は fusion gain を完全には説明しない。
```

根拠:

- primary fusion gain vs outer best single は class-conditioned LOSO `rho=0.205`
- univariate best も `rho=0.154`

### 撤回

```text
class-conditional geometry を測れば unsafe fusion/harm を十分に検出できる。
```

根拠:

- primary fusion harm5 AUROC は best univariate `0.617`
- class-conditioned LOSO AUROC は `0.601`
- これは safety gate として弱い。

## 次にやるべきこと

次は gate ではない。**E5b0: source-side longitudinal metric pilot** をやる。

目的:

```text
training subjects の multi-session labels だけを使って、
same-class cross-session を近づけ、
different-class margin を保つ表現/metricを作る。

held-out subject は全sessionをmeta-trainingから除外し、
S1 labels だけで head を作り、
S2+ target labels なしで評価する。
```

最初の実装は deep backbone ではなく、tangent feature 上の軽量 metric / block weighting にする。

理由:

- E5aで class-conditioned signal は branch gap に出た。
- しかし safety gate には弱い。
- 従って、sessionごとに選ぶのではなく、選ばなくて済む source-trained representation を作る方が論理的。

Success:

```text
E4a equal posterior + prefix-EA に対して
primary LR+UD +1pp以上, CI下限 > 0
かつ R10悪化なし
```

Strong success:

```text
primary +2pp以上, CI下限 > 0
かつ fusion harm5 / R10 を悪化させない
```

Stop condition:

```text
E5b0 が E4a equal posterior に勝てない、
または lower-tail が悪化するなら、
zero-target-label adaptation を accuracy-first の本線として続けない。
```

## してはいけないこと

```text
- E5a指標で test-time gate を作ったと主張する
- fusion harm AUROC 0.60 程度を安全性の根拠にする
- dynamic electrode/channel gate を再開する
- E4a equal posterior を新手法として押す
- いきなり深層backboneへ飛ぶ
```

今回の結果は、深層化の前に「class-conditioned longitudinal metric が本当に
E4aを超えるか」を小さく試すべき、という結論である。
