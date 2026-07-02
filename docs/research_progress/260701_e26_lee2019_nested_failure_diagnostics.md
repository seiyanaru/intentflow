# E26: Lee2019 nested selection failure diagnostics

Date: 2026-07-01

## 結論

Lee2019で nested source-side selector がうまくいかない理由は、かなり明確に言語化できる。

> Lee2019では、source-validation による subject-wise candidate selection が target outcome を同定できていない。  
> 特に inner source-validation の平均gainは held-out target gain を予測せず、複数candidateではむしろ負相関になる。  
> その結果、selector は一部被験者で過度にcompactな `longitudinal_q0p10` を選び、tailで大きく壊す。

したがって、Lee2019では「被験者ごとにzero-labelで賢く選ぶ」より、**固定 `source_only_q0p25`** の方が強い。

## 入力

E25 full exact nested selection の merged output を使用。

Input:

- `intentflow/offline/results/research_outputs/260630_lee2019_exact_nested_subspace_selection_e25_full/merged/`

Script:

- `intentflow/offline/scripts/analysis/lee2019_nested_failure_diagnostic_e26.py`

Output:

- `intentflow/offline/results/research_outputs/260701_lee2019_nested_failure_diagnostic_e26/`

Main files:

- `candidate_predictability_by_held_subject.csv`
- `predictability_summary.csv`
- `selection_margin_and_outer_outcome.csv`
- `longitudinal_q0p10_selected_cases.csv`
- `summary.json`

## E26a: source-validation predictability

各 held-out target subject について、inner source-validation metrics と outer target gain の対応を見た。

重要なのは `inner_mean_gain` と `outer_gain` の Spearman correlation。

| Candidate | Spearman(inner mean, outer gain) | Bootstrap 95% CI | Outer mean gain | Outer q05 | P(gain < -5pp) |
|---|---:|---:|---:|---:|---:|
| `source_only_q0p25` | -0.096 | [-0.387, +0.217] | +2.176 | -6.250 | 0.074 |
| `source_only_q0p10` | -0.599 | [-0.754, -0.390] | +0.463 | -11.250 | 0.222 |
| `longitudinal_q0p25` | -0.443 | [-0.661, -0.179] | +1.366 | -7.500 | 0.111 |
| `longitudinal_q0p10` | -0.460 | [-0.633, -0.235] | +1.852 | -12.500 | 0.204 |
| `sep_no_drift_q0p25` | -0.471 | [-0.682, -0.209] | +1.227 | -7.938 | 0.185 |
| `sep_no_drift_q0p10` | -0.667 | [-0.798, -0.475] | +0.255 | -10.438 | 0.278 |

解釈:

```text
source-validationで平均gainが高く見えるcandidateほど、
held-out targetでも高gainになる、という前提が崩れている。
```

特に `longitudinal_q0p10` は outer mean だけ見ると +1.852pp あるが、q05=-12.5pp, harm=20.4% とtailが悪い。  
inner mean で選ぶと、このtail riskを拾い損ねる。

## E26b: selectorはouter-bestをほとんど当てられない

Nested selector の選択内訳:

- `source_only_q0p25`: 39/54
- `longitudinal_q0p10`: 15/54

しかし、選ばれたcandidateが実際に outer-best だった割合は:

```text
outer-best match rate = 16.7%
```

また、選択candidateとouter-bestの差:

```text
selected - outer_best = -5.23pp 平均
```

これは「完璧なoracleに比べて悪い」という当たり前の話ではなく、**source-validationがcandidate rankingをかなり外している**という意味。

## E26c: marginが小さいので、選択が統計的に脆い

inner source-validation の top candidate と runner-up の平均差:

```text
mean margin = 0.333pp
median margin = 0.330pp
q05 margin = 0.015pp
```

つまり、多くの held-out subject で candidate間のsource-validation差は 1pp 未満。  
この程度の差で candidate を切り替えると、outer targetでは簡単に外れる。

これは `source_only_q0p25` 固定が強い理由でもある。  
固定policyは、inner validationの小さい揺らぎで family/fraction を切り替えない。

## E26d: `longitudinal_q0p10` 選択が主な失敗源

`longitudinal_q0p10` が選ばれた15人だけを見る。

| Metric | Value |
|---|---:|
| n | 15 |
| outer gain mean | -1.833pp |
| P(gain < -5pp) | 33.3% |
| selected - fixed `source_only_q0p25` | -2.917pp |
| 95% CI of selected - fixed `source_only_q0p25` | [-5.917, -0.250] |
| inner mean advantage over `source_only_q0p25` | +0.286pp |
| outer advantage over `source_only_q0p25` | -2.917pp |

つまり、selector は source-validation上の平均 +0.29pp 程度の小さい優位を信じて `longitudinal_q0p10` を選ぶ。  
しかし outer target では平均 -2.92pp の損になる。

Worst cases:

| Subject | chosen | inner advantage over source q0.25 | outer advantage over source q0.25 | chosen outer gain |
|---:|---|---:|---:|---:|
| 25 | `longitudinal_q0p10` | +0.637 | -17.500 | -17.500 |
| 35 | `longitudinal_q0p10` | +0.024 | -11.250 | -12.500 |
| 15 | `longitudinal_q0p10` | +0.472 | -6.250 | -11.250 |
| 29 | `longitudinal_q0p10` | +1.014 | -7.500 | -7.500 |

この表が一番わかりやすい。

```text
innerで少し良く見える。
でもouterでは大きく壊れる。
```

## なぜLee2019でこの現象が起きるのか

現時点での言語化:

> Lee2019は session0 -> session1 の single-step transfer であり、source subjectsから見たcandidateの平均的な優劣と、held-out target subject のsession shiftの向きが一致しない。  
> そのため、candidate family/fraction をsubjectごとにzero-labelで選ぶには情報が足りない。  
> 一方、`source_only_q0p25` は高次元all-channel tangent representationを適度にdenoiseしつつ、motor/posterior/mixed covariance contextを広く残すため、subject-wise selectionより安定に効く。

短く言うなら:

```text
Lee2019 is not a subject-wise model-selection problem.
It is a robust source-side denoising / compact subspace design problem.
```

## 守る / 撤回 / 修正

### 守る

`source_only_q0p25` は Lee2019 で強い。

- mean gain: +2.176pp
- 95% CI: [+0.509, +3.843]
- q05: -6.250
- P(gain < -5pp): 7.4%

### 撤回

「source-validationでsubjectごとにcandidateを選べば固定policyを超える」

これは E25/E26 で否定。

### 修正

主張は以下に修正する。

> zero-target-label subject-wise selection is under-identified on Lee2019; a fixed source-discriminative compact subspace is more reliable.

日本語:

> Lee2019では、target labelなしの被験者別選択は同定不能に近い。source側で固定的に設計したcompact subspaceの方が信頼できる。

## 次にやるべきこと

次は新しいselectorを足さない。

優先順位は以下。

1. **Stieger / Lee / 第三データセットの regime table を作る**
   - dataset structure
   - best fixed policy
   - nested selection が勝つか
   - risk profile
2. **第三データセットで fixed policy transfer を検証する**
   - Lee型なら `source_only_q0p25` が再現するか
   - Stieger型なら longitudinal / condition-specific が再現するか
3. それでも足りなければ、selectorではなく **regime classifier** にする
   - subject-wiseではなく dataset/session-regime-wise
   - 入力: channel count, sessions, task family, source-pool geometry
   - 出力: fixed policy family/fraction

## 論文上の使い方

このE26はメイン実験ではなく、**なぜadaptive selectionではなく fixed regime policy に行くのか**を説明するための診断実験。

査読で「なぜもっと賢く選ばないのか？」と聞かれたときに使える。

答え:

> We tried exact nested source-side selection. It failed because source-validation candidate rankings did not transfer to held-out targets; the selected candidate matched the target-best candidate only 16.7% of the time, and source-validation mean gains were weakly or negatively correlated with target gains. Thus, subject-wise zero-label model selection is under-identified in Lee2019.

