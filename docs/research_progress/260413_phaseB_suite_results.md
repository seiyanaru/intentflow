# 2026-04-13 Phase B Suite Results

**Result dir:** `intentflow/offline/results/phaseB_suite_20260409_seed0`

**Status:** all 5 conditions completed (`plain`, `aug_only_0025`, `aug_only_005`, `aug_shinv_0025`, `aug_shinv_005`)

## gpt

### 結論

Phase B の結論はかなり明確です。

1. **Phase B の best は `aug_only(std=0.025) + hybrid@0.01`**
   - `plain hybrid` に対して `+0.77pp`
   - `source_only` でも `+0.54pp`
2. **ただし `S2` は全条件で悪化**
   - したがって、train-time 側で `S2 harm` の causal finding を取り込めたとはまだ言えない
3. **`shallow invariance` は加点というより damage control**
   - 特に `std=0.05` では overshoot を緩和する
   - しかし `aug_only(std=0.025)` を超える最良条件にはならなかった
4. **現時点の Phase B best は、研究全体の既存 frontier を超えていない**
   - 既存 best: `hybrid@0.01` = `81.98%`
   - 今回 best: `aug_only_0025 + hybrid` = `80.63%`

最も重要なのは、

> 「mild な gain jitter を train-time に入れる価値はある」

一方で

> 「shallow variance の causal finding を、そのまま train-time regularization へ移植すれば勝てる」

とはまだ言えない、という点である。

---

## 1. 実験設定の注意

今回の Phase B suite は `configs/tcformer_aug_shinv/tcformer_aug_shinv.yaml` を使っており、**`interaug: False`** で学習している。

つまり今回の `plain` は

- 既存の最良 source model
- 既存の paper 系 train-time recipe

そのものではない。

したがって、

- **suite 内比較** (`plain` vs `aug_only` vs `aug+shinv`) は妥当
- **過去の最良 run との比較** は参考値

として読むべきである。

---

## 2. 全体結果

### 2.1 Summary table

| condition | source_only | Δ vs plain | hybrid@0.01 | Δ vs plain | hybrid - source |
|---|---:|---:|---:|---:|---:|
| plain | 79.67 | 0.00 | 79.86 | 0.00 | +0.19 |
| aug_only_0025 | 80.21 | +0.54 | 80.63 | +0.77 | +0.42 |
| aug_only_005 | 79.51 | -0.15 | 79.98 | +0.12 | +0.46 |
| aug_shinv_0025 | 80.17 | +0.50 | 80.21 | +0.35 | +0.04 |
| aug_shinv_005 | 79.78 | +0.12 | 79.94 | +0.08 | +0.15 |

### 2.2 読み方

- `aug_only(std=0.025)` が source/hybrid の両方で最良
- `std=0.05` は train-time aug として強すぎる
- `shinv` を入れると source_only は維持できても、**hybrid 側の上積みが縮む**

つまり、今回最も強い経験則は

> **mild aug は効くが、強い aug は overshoot し、shinv はその overshoot を弱める代わりに test-time adaptivity も弱める**

である。

---

## 3. 被験者群ごとの結果

ここでは、事前の proxy validation に基づいて

- majority 群: `S2, S3, S4, S6, S7, S9`
- risk 群: `S1, S8`
- boundary 群: `S5`

に分けて読む。

### 3.1 majority 群

| condition | source_only | Δ vs plain | hybrid | Δ vs plain |
|---|---:|---:|---:|---:|
| plain | 78.18 | 0.00 | 78.59 | 0.00 |
| aug_only_0025 | 79.22 | +1.04 | 80.03 | +1.45 |
| aug_only_005 | 78.53 | +0.35 | 79.40 | +0.81 |
| aug_shinv_0025 | 78.59 | +0.41 | 79.22 | +0.64 |
| aug_shinv_005 | 78.94 | +0.75 | 78.99 | +0.41 |

**解釈**

- majority 群では `aug_only_0025` が最も強い
- この群に対しては **train-time mild gain jitter は確かに有効**
- 一方 `shinv` を入れると、この majority 群での gain が小さくなる

つまり `shinv` は、少なくとも現実装では **majority 群の useful drift まで削っている** 可能性が高い。

### 3.2 risk 群 (`S1, S8`)

| condition | source_only | Δ vs plain | hybrid | Δ vs plain |
|---|---:|---:|---:|---:|
| plain | 85.94 | 0.00 | 85.94 | 0.00 |
| aug_only_0025 | 86.28 | +0.35 | 86.28 | +0.35 |
| aug_only_005 | 85.76 | -0.17 | 86.11 | +0.17 |
| aug_shinv_0025 | 86.63 | +0.69 | 85.59 | -0.35 |
| aug_shinv_005 | 85.07 | -0.87 | 86.11 | +0.17 |

**解釈**

- risk 群は今回も大崩れはしていない
- ただし `shinv_0025` は source_only では良いが、hybrid を足すと逆に下がる
- `shinv_005` は source_only で risk 群を悪化させる

この挙動は、

> shinv が risk 群の「train-time robustness」には寄与しても、test-time adaptation との整合を壊している

可能性を示す。

### 3.3 boundary 群 (`S5`)

| condition | source_only | Δ vs plain | hybrid | Δ vs plain |
|---|---:|---:|---:|---:|
| plain | 76.04 | 0.00 | 75.35 | 0.00 |
| aug_only_0025 | 73.96 | -2.08 | 72.92 | -2.43 |
| aug_only_005 | 72.92 | -3.12 | 71.18 | -4.17 |
| aug_shinv_0025 | 76.74 | +0.69 | 75.35 | +0.00 |
| aug_shinv_005 | 74.31 | -1.74 | 73.26 | -2.08 |

**解釈**

- `S5` は aug-only に最も弱い
- `shinv_0025` だけが `S5` をほぼ救っている

これは重要で、`shinv` が全く無意味なのではなく、

> **特定の boundary / harmful-drift subject に対しては効いている**

という証拠になっている。

---

## 4. S2 canary

`S2` は今回も最重要。

| condition | source_only | hybrid |
|---|---:|---:|
| plain | 62.85 | 63.89 |
| aug_only_0025 | 60.76 | 62.50 |
| aug_only_005 | 56.94 | 60.07 |
| aug_shinv_0025 | 60.07 | 62.15 |
| aug_shinv_005 | 62.15 | 62.50 |

### 読み

- **全条件で `plain` を下回る**
- ただし `shinv_005` は `aug_only_005` より大きく回復
  - source: `+5.21pp`
  - hybrid: `+2.43pp`

### 解釈

これはかなり重要で、

- `shinv` 自体は S2 型 harm を抑える方向には働いている
- しかし現時点では **plain baseline を超えるほど強くはない**

つまり、今回の結果は

> `L_shallow_inv` の方向性は完全に外していないが、まだ blunt で、useful shift も一緒に削っている

と読むのが正確である。

---

## 5. Worst-case / NTR 的な見方

### source_only vs plain

| condition | mean Δ | worst Δ | strict negatives | material negatives |
|---|---:|---:|---:|---:|
| aug_only_0025 | +0.54 | -2.08 | 4/9 | 3/9 |
| aug_only_005 | -0.15 | -5.90 | 4/9 | 4/9 |
| aug_shinv_0025 | +0.50 | -2.78 | 2/9 | 2/9 |
| aug_shinv_005 | +0.12 | -1.74 | 4/9 | 4/9 |

### hybrid vs plain-hybrid

| condition | mean Δ | worst Δ | strict negatives | material negatives |
|---|---:|---:|---:|---:|
| aug_only_0025 | +0.77 | -2.43 | 3/9 | 3/9 |
| aug_only_005 | +0.12 | -4.17 | 4/9 | 4/9 |
| aug_shinv_0025 | +0.35 | -1.74 | 4/9 | 2/9 |
| aug_shinv_005 | +0.08 | -2.08 | 4/9 | 4/9 |

### 解釈

- `aug_only_0025` は **平均最良**
- `aug_shinv_0025` は **平均は落ちるが worst-case がややマシ**

したがって Phase B の現時点は

- **accuracy-first:** `aug_only_0025`
- **safety-leaning:** `aug_shinv_0025`

の 2 本立てで考えるのが自然。

---

## 6. `shinv` は何をしているか

`shinv` の役割は条件によって変わっている。

### std = 0.025

`aug_only_0025 -> aug_shinv_0025`

- source_only: `80.21 -> 80.17` でほぼ同等
- hybrid: `80.63 -> 80.21` で明確に悪化
- 特に `S4, S6, S7` の gain を削る
- 一方で `S5` は `+2.78pp` 回復

**解釈:**  
`shinv` は harmful drift も useful drift も両方抑えており、低 aug 領域では過剰に保守的。

### std = 0.05

`aug_only_005 -> aug_shinv_005`

- source_only: `79.51 -> 79.78`
- hybrid: `79.98 -> 79.94` でほぼ同等
- `S2` を大きく回復
- `S4` を大きく削る

**解釈:**  
高 aug 領域では `shinv` は **overshoot の緩和** として機能している。

したがって、今回の `shinv` の正しい解釈は

> **performance booster ではなく、overshoot regularizer**

である。

---

## 7. 既存 frontier との関係

過去の既存 best は

- `hybrid@0.01`: `81.98%`
  - `intentflow/offline/results/hybrid_vs_meanonly_20260406_205743/hybrid/results.txt`

今回の Phase B best は

- `aug_only_0025 + hybrid`: `80.63%`

であり、**まだ 1.35pp 届いていない**。

ただしこれはそのまま「train-time aug が無効」とは言えない。理由は今回の suite が `interaug: False` で走っており、既存 frontier と train recipe が異なるから。

それでも、研究判断としては次が妥当。

> **Phase B は promising だが、現時点では “既存システムを置き換える” 段階ではない。**

---

## 8. 最終判断

### 今回強く言えること

1. mild gain jitter (`std=0.025`) は train-time に入れる価値がある
2. `std=0.05` は強すぎる
3. `shinv` は万能ではない
4. `shinv` は overshoot 緩和には効く
5. `S2` を baseline 超えさせるにはまだ足りない

### 次にどう考えるべきか

1. `virtual BN` に進む前に、**Phase B を current strong baseline 上で再検証**すべき
   - 特に `interaug: True` の既存学習 recipe に `gain jitter(std=0.025)` を足す方向
2. `shinv` は全体 boost ではなく、**S2/S5 の救済項**として設計を見直すべき
   - 全 sample 一律ではなく、条件付き・弱め・局所的に入れる方が筋がよい
3. Phase C の前提は、
   - `aug_only_0025` を stronger baseline 上で再確認
   - その上で `S2` をどう救うかを train-time で明示化

### 現時点の best recommendation

- **accuracy-first 実験候補:** `aug_only_0025`
- **safety-first 実験候補:** `aug_shinv_0025`
- ただし両方とも、**まずは current strong baseline (`interaug: True`) に統合して再評価**するのが先

### 次の一手

1. `interaug: True` の現行 strongest train recipe に `gain_jitter(std=0.025)` だけを足して再評価する
2. `aug_shinv_0025` は accuracy 改善の本線ではなく、`S2/S5` の damage control 候補として別軸で追う
3. `Phase C (virtual BN)` はまだ早い。少なくとも `S2` を plain baseline 以上に持ち上げる train-time signal が確認できてから進む

---

## claude

Claude の考察欄
