# 260422 BCIC-IV 2b First Pass Results

- 目的: `2b first pass` の 3 条件 (`source_only / vanilla_both / hybrid_mom001`) の結果を暫定記録として残す。
- 実行日時: `2026-04-22`
- run root: [intentflow/offline/results/phaseC_2b_firstpass_20260422_164615_seed0](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/phaseC_2b_firstpass_20260422_164615_seed0)
- 前提 source checkpoint: [intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347)

---

## 条件

1. `source_only`
   - `adapt_mode=source_only`
   - `enable_otta=false`
2. `vanilla_both`
   - `adapt_mode=bn_stat_clean`
   - `bn_momentum=0.1`
   - `bn_update_target=both`
3. `hybrid_mom001`
   - `adapt_mode=bn_stat_clean`
   - `bn_momentum=0.01`
   - `bn_update_target=shallow_mean_deep_both`

共通設定:

- dataset: `bcic2b`
- seed: `0`
- eval config: [tcformer_otta_bs1.yaml](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/configs/tcformer_otta/tcformer_otta_bs1.yaml)
- test batch: `1`（online OTTA simulation）
- protocol: `MOABB + stop=-0.5 + 0train/1train/2train -> train, 3test/4test -> test`

---

## Summary

| condition | avg acc | avg kappa | avg loss |
|---|---:|---:|---:|
| `source_only` | `87.74 ± 8.89` | `0.755 ± 0.178` | `0.301 ± 0.210` |
| `vanilla_both` | `88.01 ± 8.70` | `0.760 ± 0.174` | `0.310 ± 0.210` |
| `hybrid_mom001` | `87.82 ± 8.88` | `0.756 ± 0.178` | `0.311 ± 0.222` |

結果ファイル:

- [source_only/results.txt](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/phaseC_2b_firstpass_20260422_164615_seed0/source_only/results.txt)
- [vanilla_both/results.txt](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/phaseC_2b_firstpass_20260422_164615_seed0/vanilla_both/results.txt)
- [hybrid_mom001/results.txt](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/phaseC_2b_firstpass_20260422_164615_seed0/hybrid_mom001/results.txt)

---

## Subject-wise Accuracy

| subject | source_only | vanilla_both | hybrid_mom001 | vanilla Δ | hybrid Δ |
|---|---:|---:|---:|---:|---:|
| S1 | 78.75 | 80.94 | 77.81 | +2.19 | -0.94 |
| S2 | 70.00 | 69.29 | 70.36 | -0.71 | +0.36 |
| S3 | 84.38 | 84.38 | 85.31 | +0.00 | +0.94 |
| S4 | 98.12 | 97.81 | 97.19 | -0.31 | -0.94 |
| S5 | 97.81 | 98.44 | 98.75 | +0.63 | +0.94 |
| S6 | 83.75 | 86.87 | 84.06 | +3.12 | +0.31 |
| S7 | 93.12 | 92.50 | 93.12 | -0.62 | +0.00 |
| S8 | 94.69 | 94.38 | 94.69 | -0.31 | +0.00 |
| S9 | 89.06 | 87.50 | 89.06 | -1.56 | +0.00 |

- `worst vanilla Δ = -1.56 pp` (`S9`)
- `worst hybrid Δ = -0.94 pp` (`S1`, `S4`)
- `mean Δ (vanilla) = +0.27 pp`
- `mean Δ (hybrid) = +0.08 pp`

---

## 暫定判定

事前に置いていた 2b first pass の暫定条件:

1. `vanilla-both` で worst-subject `Δ < -3 pp`
2. `hybrid` で worst-subject `Δ > -1 pp`
3. `hybrid` で mean `Δ >= 0 pp`

今回の結果:

- 条件 1: **不成立**
  - `vanilla worst Δ = -1.56 pp`
- 条件 2: **成立**
  - `hybrid worst Δ = -0.94 pp`
- 条件 3: **成立**
  - `hybrid mean Δ = +0.08 pp`

したがって、**2b では `vanilla_both` の negative transfer が 2a ほど強く出ていない**。  
現時点では、`2a と同じ causal story が 2b でもそのまま通った` とまでは言えない。

---

## 現時点の解釈

1. `source_only` の再評価値は source 学習時 baseline (`87.74%`) と一致しており、2b OTTA 評価経路は正常。
2. `vanilla_both` は平均では微増 (`+0.27 pp`) で、worst-case も `-1.56 pp` に留まる。
3. `hybrid_mom001` は safety を保つが、`vanilla` を明確に救う構図にはなっていない。
4. よって 2b first pass の主メッセージは、`2b では vanilla harm が弱い` である。

---

## メモ

- 本メモは結果の固定用。解釈の深掘りや次段階の方針変更は、別ファイルで扱う。
- `source_only / vanilla_both / hybrid_mom001` の 3 条件は今後の比較基準として残す。

