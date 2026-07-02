# E26: Lee2019 nested selection predictability audit

Date: 2026-07-01

## 結論

Lee2019で nested source-side selection がうまくいかない理由は、かなり明確になった。

> 被験者ごとに本当のbest candidateは存在するが、target labelなしのsource-validationではそれを予測できない。

したがって、Lee2019では subject-wise selector を主張するより、固定 `source_only_q0p25` を population-level robust policy として扱う方が正しい。

## 入力

E25の full exact nested output を再利用した。

Input:

`intentflow/offline/results/research_outputs/260630_lee2019_exact_nested_subspace_selection_e25_full/merged/`

Script:

`intentflow/offline/scripts/analysis/lee2019_nested_predictability_e26.py`

Output:

`intentflow/offline/results/research_outputs/260701_lee2019_nested_predictability_e26/`

Main files:

- `summary.json`
- `summary_key_metrics.csv`
- `subject_predictability_diagnostics.csv`
- `candidate_pair_predictability.csv`
- `source_validation_choice_bootstrap.csv`

## 主要結果

### 1. Inner source-validation ranking は outer target ranking をほぼ予測しない

Candidate rankingの Spearman 相関:

| Metric | Mean | 95% CI | Interpretation |
|---|---:|---:|---|
| inner mean vs outer gain, non-full candidates | **0.092** | [-0.037, 0.222] | almost no predictive signal |

つまり、source-validation上で良いcandidateが、held-out targetでも良いとは言えない。

### 2. Nested choice が outer oracle と一致するのは 16.7%

| Metric | Mean | 95% CI |
|---|---:|---:|
| nested choice matches outer oracle | **16.7%** | [7.4%, 27.8%] |
| inner top-1 by mean matches outer oracle | **16.7%** | [7.4%, 27.8%] |

7 candidate なので完全ランダムなら約14.3%。  
16.7%は、ほぼランダムに近い。

### 3. Source risk指標は outer harm をほぼ予測しない

Non-full candidateペアで、outer harm (`outer_gain < -5pp`) を予測するAUROC:

| Source-side risk score | AUROC |
|---|---:|
| inner P(gain < -5pp) | 0.510 |
| inner R10 loss | 0.540 |
| -inner q05 | 0.548 |
| -inner mean | 0.464 |

0.5付近なので、source-validation riskはtarget harm予測器として弱い。

### 4. Nested choice の oracle regret は大きい

| Metric | Mean | 95% CI |
|---|---:|---:|
| outer oracle regret | **5.23pp** | [3.98, 6.57] |
| chosen - fixed `source_only_q0p25` | **-0.81pp** | [-1.81, -0.02] |

Oracleはもちろんdeploy不可だが、候補間の本当の差は存在する。  
問題は、その差をsource-validationが当てられないこと。

### 5. `longitudinal_q0p10` 選択時の失敗が主因

Nestedは以下の候補を選んだ:

- `source_only_q0p25`: 39/54
- `longitudinal_q0p10`: 15/54

`longitudinal_q0p10` が選ばれた15人:

| Metric | Mean | 95% CI |
|---|---:|---:|
| outer gain of chosen `longitudinal_q0p10` | -1.83pp | [-5.67, 1.83] |
| inner margin vs `source_only_q0p25` | +0.286pp | [0.160, 0.431] |
| outer margin vs `source_only_q0p25` | **-2.92pp** | **[-5.92, -0.25]** |

重要:

```text
source-validation上では longitudinal_q0p10 が q0p25 より平均 +0.29pp 良く見える。
しかし outer targetでは q0p25 より平均 -2.92pp 悪い。
```

これは「小さなsource-side marginに基づいて、targetで大きく外す」構造。

## Choice bootstrap: source-validation choice 自体もかなり不安定

Inner source subjectsをbootstrapして、同じnested ruleでcandidate選択を再実行した。

| Metric | Mean | 95% CI |
|---|---:|---:|
| bootstrap top-choice rate | 0.575 | [0.546, 0.605] |
| normalized choice entropy | 0.524 | [0.499, 0.549] |

Interpretation:

```text
source-validationによるcandidate選択は、平均的にかなり不確実。
top candidateがbootstrap resampleで約58%しか再現しない。
```

ただし、完全に不安定なだけではない。  
S25やS35のように `longitudinal_q0p10` が比較的安定に選ばれて、それでもouterで大きく壊れる例もある。

つまり失敗は二層:

1. source-validation rankingが不安定
2. 安定に見える場合でもtarget-specific shiftを外す

## Outer oracle candidate は実際にはバラけている

Outer oracle counts:

| Candidate | Count |
|---|---:|
| full_q1p00 | 13 |
| source_only_q0p25 | 11 |
| source_only_q0p10 | 10 |
| longitudinal_q0p10 | 7 |
| sep_no_drift_q0p10 | 5 |
| sep_no_drift_q0p25 | 5 |
| longitudinal_q0p25 | 3 |

This matters.

もし全員のoracleが `source_only_q0p25` なら、単に「固定q0p25が真のbest」で終わる。  
しかし実際にはoracleは被験者ごとにかなり散っている。

したがって本質は:

```text
per-subject heterogeneity exists,
but zero-target-label source-validation cannot identify it reliably.
```

これがLee2019の中心メカニズム。

## 論文用の言語化

> Lee2019 contains transferable covariance information, but zero-label subject-wise model selection is statistically under-identified. Source-side validation provides little rank information about the held-out target candidate ordering and weakly predicts target-side harm. Consequently, nested selection overfits small source-validation margins, especially favoring overly compact longitudinal subspaces, whereas a fixed source-discriminative q=0.25 compact subspace acts as a more robust population-level denoising policy.

日本語:

> Lee2019には転移可能なcovariance情報はある。しかし、target labelなしで被験者ごとにcandidateを選ぶには情報が足りない。source-validationのcandidate順位はheld-out targetの順位とほぼ相関せず、source-side riskもtarget harmを予測しない。その結果、nested selectionはsource上の小さな平均差に過剰反応し、特にcompactな `longitudinal_q0p10` を選んでtail riskを出す。一方、固定 `source_only_q0p25` は広めの有用subspaceを残すpopulation-level denoising policyとして安定に効く。

## 判断

### 守る

固定 `source_only_q0p25` はLee2019で強い。

### 撤回

Lee2019で subject-wise source-side nested selection を主張する。

### 修正

主張は以下にする。

```text
There is real per-subject candidate heterogeneity,
but zero-label source-validation cannot identify it.
Therefore robust fixed source-side compact subspace policies are preferable
to per-subject nested selection in Lee2019-like regimes.
```

## 次アクション

次は unified table を作る。

Rows:

- Lee2019
- Stieger LR
- Stieger UD
- BNCI2014_001

Columns:

- best fixed policy
- nested/validation policy
- mean gain
- q05
- P(gain < -5pp)
- whether source-validation predicts outer candidate ranking
- interpretation of regime

この表ができると、研究の軸が

```text
new selector
```

ではなく

```text
when zero-label source-side compact subspace selection is identifiable,
and when it should collapse to robust fixed policies
```

に変わる。

