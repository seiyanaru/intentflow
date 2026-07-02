# E25: Lee2019 full exact nested source-side selection

Date: 2026-06-30

## 結論

Lee2019 54人全体では、**source-side nested selector は勝っていない**。

採用すべき方策は、現時点では **固定 `source_only_q0p25`**。

理由は明確で、固定 `source_only_q0p25` が平均利得・下側リスク・harm率のすべてで nested を上回った。

## 実験設定

- Dataset: Lee2019 MI, 54 subjects
- 評価: session0 -> session1
- 特徴: 既存の source-side / longitudinal subspace score
- 外側評価: held-out target subject
- 内側選択: target subject を除いた source subjects のみ
- risk constraint: `P(gain < -5pp) <= 0.20`
- candidates:
  - `full_q1p00`
  - `source_only_q0p25`
  - `source_only_q0p10`
  - `longitudinal_q0p25`
  - `longitudinal_q0p10`
  - `sep_no_drift_q0p25`
  - `sep_no_drift_q0p10`

実行高速化のため、`lee2019_exact_nested_subspace_selection.py` に `--target-subjects` を追加し、source集合は全54人固定のまま outer subjects のみ8 shardに分割した。

Merged output:

- `intentflow/offline/results/research_outputs/260630_lee2019_exact_nested_subspace_selection_e25_full/merged/exact_nested_summary.csv`
- `intentflow/offline/results/research_outputs/260630_lee2019_exact_nested_subspace_selection_e25_full/merged/nested_vs_fixed_paired_comparisons.csv`

## 主結果

| method | acc mean | gain vs full | 95% CI | q05 | R10 loss | P(gain < -5pp) |
|---|---:|---:|---:|---:|---:|---:|
| nested_risk | 71.04 | +1.37 | [-0.56, +3.22] | -11.69 | 11.88 | 16.7% |
| fixed `source_only_q0p25` | 71.85 | **+2.18** | **[+0.51, +3.84]** | **-6.25** | **7.50** | **7.4%** |
| fixed `longitudinal_q0p10` | 71.53 | +1.85 | [-0.30, +3.94] | -12.50 | 13.13 | 20.4% |
| fixed `longitudinal_q0p25` | 71.04 | +1.37 | [-0.28, +3.03] | -7.50 | 8.33 | 11.1% |
| full | 69.68 | 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.0% |

`nested_risk` の選択内訳:

- `source_only_q0p25`: 39/54
- `longitudinal_q0p10`: 15/54

## Paired comparison

Nested minus fixed `source_only_q0p25`:

- mean: **-0.81pp**
- 95% CI: **[-1.78, -0.05]**
- nested が fixed より +5pp 以上良い被験者: **0%**
- nested が fixed より -5pp 以上悪い被験者: **7.4%**

つまり、Lee2019 では nested selection は「少し賢く選ぶ」どころか、固定 `source_only_q0p25` から有意に近い形で劣化している。

## 失敗原因

Nested が `longitudinal_q0p10` を15人で選んだが、この group の外側平均 gain は **-1.83pp**。

特に壊れた例:

| subject | chosen | nested gain | fixed `source_only_q0p25` gain | fixed `longitudinal_q0p25` gain |
|---:|---|---:|---:|---:|
| 25 | longitudinal_q0p10 | -17.50 | 0.00 | -12.50 |
| 35 | longitudinal_q0p10 | -12.50 | -1.25 | +6.25 |
| 15 | longitudinal_q0p10 | -11.25 | -5.00 | -2.50 |
| 29 | longitudinal_q0p10 | -7.50 | 0.00 | 0.00 |

内側source-validation上では `longitudinal_q0p10` の平均が高く見えることがあるが、外側targetでは tail risk が大きい。

## Risk threshold sweep

`exact_nested_source_validation.csv` から、risk threshold と選択目的を変えて cheap sweep した。

最大平均でも:

- best nested-like policy: +1.55pp
- P(gain < -5pp): 11.1%
- fixed `source_only_q0p25`: +2.18pp, P(gain < -5pp): 7.4%

threshold を厳しくすると full に退化し、平均利得が消える。

したがって、Lee2019 では「threshold調整で nested を救う」方向も弱い。

## 判断

### 撤回する主張

「source-side validation で candidate を選べば固定policyより良くなる」

これは Lee2019 full exact では否定。

### 守る主張

「source側だけで compact subspace を固定設計すると、zero-target-label cross-session MI で平均利得を出せる」

Lee2019 では fixed `source_only_q0p25` が最も強い。

### 修正する主張

旧: adaptive / nested source-side selection  
新: **regime-aware fixed compact subspace policy**

Lee2019 のような session0->session1 transfer では、source-only discriminative stability を上位25%に絞る固定方策が最も堅い。

## 次アクション

1. Lee2019 では nested selector を主役から外す。
2. Stieger では既存 E5b/E10/E16 の condition-specific longitudinal policy と並べて、dataset regime ごとの固定方策として整理する。
3. 次に作るべき表は「dataset regime × best fixed source-side policy × nestedの失敗/不要性」。
4. その後、第三データセットでは selectorではなく、**事前固定 policy がどの regime で転移するか**を検証する。

## 論文上の含意

この結果は地味だが重要。

「target labelなしで毎subject選ぶ」方向は、少なくとも Lee2019 では過剰適応になっている。

勝ち筋は、target側を見ない安全な個別選択ではなく、source側から安定に決まる compact subspace design を dataset/regime単位で選ぶこと。

これは novelty を以下に寄せるべきことを意味する:

- adaptive selector の新規性ではない
- target-label-free compact subspace design の実証
- session/dataset regime によって optimal source-side criterion が変わる、という整理
- 「選択すればよい」という直感の否定

