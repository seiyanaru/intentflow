# Neuro-LEMA E1/E1b results — 2026-06-24

## 結論

E1は **pass, but not as originally hoped**。

`fb_sensorimotor21_mu_beta` は、source backboneとしては強い。特にLRではsource時点で `broad_all60` より高精度で、prefix EA後の絶対精度も高い。

一方で、`fb_sensorimotor21_mu_beta` のprefix EAは、sourceからの追加改善量やlower-tail riskでは `broad_all60` を一貫して上回らない。したがって、

```text
sensorimotor μ/β に絞ればEAが安全になる
```

という強い主張は撤回する。

代わりに、次の主張へ修正する。

```text
sensorimotor μ/β は強いneurophysiological branchであり、
broad all-channel branchとはsessionごとに勝敗が大きく異なる。
したがって、Neuro-LEMAはhard feature replacementではなく、
broad branch / neuro branch / source anchorをprefix uncertaintyで制御する
feature-branch selection or soft fusionとして設計すべきである。
```

## 入力

- E1 script:
  - `intentflow/offline/scripts/analysis/stieger_neuro_feature_baseline.py`
- E1 output:
  - `intentflow/offline/results/research_outputs/260624_stieger_neuro_feature_baseline/summary.json`
- E1b script:
  - `intentflow/offline/scripts/analysis/stieger_neuro_feature_complementarity.py`
- E1b output:
  - `intentflow/offline/results/research_outputs/260624_stieger_neuro_feature_complementarity/complementarity_summary.json`

Protocol:

```text
Stieger2021
subjects = 1-62
source = session 1
target = session 2+
conditions = pure_lr / pure_ud / two_d
prefix = first 32 target trials
evaluation = trial 65 onward
adapter = prefix_ea unless noted
```

## E1: neuro feature baseline

E1の目的は、新手法の勝利ではなく、神経生理特徴空間がbaselineとして成立するかの生存確認である。

### prefix EA, source-relative utility/risk

| condition | feature | U Δacc pp | R10 | P(Δ<-5pp) | q05 |
|---|---:|---:|---:|---:|---:|
| pure_lr | broad_all60 | +5.49 | 9.79 | 9.89% | -8.70 |
| pure_lr | fb_sensorimotor21_mu_beta | +4.57 | 10.56 | 12.96% | -9.88 |
| pure_ud | broad_all60 | +5.15 | 11.86 | 14.84% | -10.64 |
| pure_ud | fb_sensorimotor21_mu_beta | +4.08 | 11.72 | 13.28% | -10.00 |
| two_d | broad_all60 | +4.32 | 10.40 | 11.99% | -8.82 |
| two_d | fb_sensorimotor21_mu_beta | +3.49 | 10.10 | 13.76% | -9.38 |

Interpretation:

- `fb_sensorimotor21_mu_beta` はUで `broad_all60` から約 -0.8〜-1.1pp。
- R10はUDでは少し改善するが、LR/2Dでは一貫改善ではない。
- よって、hard sensorimotor-only EAを安全化手法として主張するのは弱い。

### prefix EA, absolute adapted accuracy

| condition | feature | source acc | adapted acc | Δ |
|---|---:|---:|---:|---:|
| pure_lr | broad_all60 | 58.50 | 63.99 | +5.49 |
| pure_lr | fb_sensorimotor21_mu_beta | 62.04 | 66.61 | +4.57 |
| pure_ud | broad_all60 | 56.93 | 62.08 | +5.15 |
| pure_ud | fb_sensorimotor21_mu_beta | 58.13 | 62.20 | +4.08 |
| two_d | broad_all60 | 30.97 | 35.30 | +4.32 |
| two_d | fb_sensorimotor21_mu_beta | 31.91 | 35.40 | +3.49 |

Interpretation:

- `fb_sensorimotor21_mu_beta` はsource時点で強い。
- LRではsourceで +3.54pp、prefix EA後で +2.62pp、`broad_all60` を上回る。
- つまり、neuro featureは「EAで大きく伸びる特徴」ではなく「最初から強い表現」である。

## E1b: broad vs neuro complementarity

E1bはラベルを使うoracle診断である。これはdeployable methodではない。

質問：

```text
broad_all60 と fb_sensorimotor21_mu_beta は、同じsessionで勝つのか？
それとも違うsessionで勝つのか？
違うsessionで勝つなら、gating/fusionにheadroomがある。
```

### Pair oracle: broad_all60 vs fb_sensorimotor21_mu_beta

| condition | broad acc | neuro acc | neuro-broad | oracle pair acc | oracle gain vs best single | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| pure_lr | 63.99 | 66.61 | +2.62 | 68.84 | +2.23 | [1.71, 2.79] |
| pure_ud | 62.08 | 62.20 | +0.13 | 66.28 | +4.08 | [2.95, 4.45] |
| two_d | 35.30 | 35.40 | +0.10 | 38.87 | +3.47 | [2.91, 3.66] |

Interpretation:

- pair oracle gainは全conditionで明確に正。
- LRはneuro単体が強いが、それでもbroadと選び分けるとさらに +2.23pp。
- UD/2Dは単体平均ではほぼ同等なのに、oracle gainが +3〜4ppある。これは勝つsessionが相当違うということ。

### Session disagreement

| condition | neuro win | broad win | \|diff\|≥3pp | \|diff\|≥5pp | Spearman acc | Spearman Δ |
|---|---:|---:|---:|---:|---:|---:|
| pure_lr | 59.2% | 33.9% | 73.3% | 57.2% | 0.75 | 0.26 |
| pure_ud | 44.9% | 48.7% | 73.8% | 57.0% | 0.62 | 0.30 |
| two_d | 46.2% | 47.8% | 72.5% | 57.5% | 0.61 | 0.23 |

Interpretation:

- adapted accuracyの相関は中程度。
- Δaccの相関は低い。
- sessionごとの勝敗差は大きく、`|diff|≥5pp` が約57%もある。
- これは「特徴ブランチ選択問題」としてはかなり強いheadroomを示す。

### Harm overlap

| condition | broad harm5 | neuro harm5 | both harm5 | oracle harm5 |
|---|---:|---:|---:|---:|
| pure_lr | 9.89% | 12.96% | 2.10% | 2.10% |
| pure_ud | 14.84% | 13.28% | 3.33% | 3.33% |
| two_d | 11.99% | 13.76% | 3.33% | 3.33% |

Interpretation:

- broadとneuroのharmは重なりが小さい。
- 片方が壊すsessionをもう片方が救える可能性がある。
- これはrisk-aware branch selectionの最も重要な根拠。

## 4-feature oracle

4 features:

```text
broad_all60
broad_sensorimotor21
fb_all60_mu_beta
fb_sensorimotor21_mu_beta
```

| condition | best single | best single acc | oracle acc | oracle gain | 95% CI |
|---|---|---:|---:|---:|---:|
| pure_lr | fb_sensorimotor21_mu_beta | 66.61 | 70.55 | +3.94 | [3.34, 4.58] |
| pure_ud | fb_sensorimotor21_mu_beta | 62.20 | 68.63 | +6.43 | [5.20, 6.78] |
| two_d | fb_sensorimotor21_mu_beta | 35.40 | 40.65 | +5.26 | [4.57, 5.58] |

Winner rates are not concentrated in one branch:

| condition | broad_all60 | broad_sensorimotor21 | fb_all60_mu_beta | fb_sensorimotor21_mu_beta |
|---|---:|---:|---:|---:|
| pure_lr | 24.4% | 23.7% | 23.8% | 44.5% |
| pure_ud | 30.0% | 21.7% | 31.5% | 28.9% |
| two_d | 33.0% | 23.3% | 25.2% | 33.6% |

Interpretation:

- best branchは平均では `fb_sensorimotor21_mu_beta`。
- しかしoracle winnerは全branchに分散している。
- これは「単一のneuro featureに置換する」より、「branch portfolio + learned selector/fusion」に進むべき証拠。

## 研究方針の修正

撤回：

```text
sensorimotor μ/βに絞ったEAが、broad EAより一貫して安全。
```

守る：

```text
sensorimotor μ/β branchはsource representationとして強い。
```

新しく守る：

```text
broad branchとneuro branchのsession-level complementarityは大きい。
harm overlapも小さい。
よって、次に作るべき新手法は hard neuro replacement ではなく、
prefix uncertaintyに基づく branch selection / soft fusion である。
```

## 次の実験

E2は、いきなり複雑なmeta-adapterではなく、まず以下にする。

```text
E2: label-free branch selector / soft fusion

arms:
  broad_all60 source
  broad_all60 prefix_ea
  fb_sensorimotor21_mu_beta source
  fb_sensorimotor21_mu_beta prefix_ea
  optionally broad_sensorimotor21 / fb_all60_mu_beta

input signals:
  source-target covariance distance
  prefix bootstrap instability
  prediction entropy / margin
  broad-vs-neuro prediction disagreement
  EA update magnitude

target:
  nested subject splitで、sessionごとのbest branchまたはsoft weightを予測

primary metric:
  adapted absolute accuracy

risk metric:
  R10 and P(Δ<-5pp) relative to chosen source-anchor baseline
```

判定：

```text
minimum:
  best single branchに対して +1pp以上
  かつ R10悪化なし

strong:
  +2pp以上
  かつ P(Δ<-5pp)を20%以上削減
```

もしE2でlabel-free selectorがoracle gainをほとんど回収できない場合は、Neuro-LEMA本体に進む前に止める。
