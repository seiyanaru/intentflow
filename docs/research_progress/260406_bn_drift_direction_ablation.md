# 2026-04-06 BN Drift Direction Ablation

## 今日の問い

「S2 の OTTA harm は、BN 更新の magnitude の問題か、direction（何を・どこで更新するか）の問題か」

前セッションまでの確認事項：
- S2 の harm は bs=48 評価アーティファクトだけでは説明できない（bs=1 でも残る）
- momentum を下げると全体の NTR-S/WSD は改善するが S2 の harm は消えない
- S2 の harm は `bn_stat_clean` の純粋 BN stats 更新が原因（TENT との差はほぼなし）

---

## 実験 A：mom=0.01、全 9 被験者（bs=1）

**設定:** `bn_stat_clean`, `bn_momentum=0.01`, `bs=1`, all 9 subjects  
**Checkpoint:** `results/update_op_v2_20260401_125734/source_model`  
**Config:** `configs/tcformer_otta/tcformer_otta_bs1.yaml`

### 結果

| S | source | mom=0.01 | Δ |
|---|--------|----------|---|
| 1 | 84.03% | 85.42% | +1.39% |
| **2** | **67.71%** | **62.15%** | **-5.56%** |
| 3 | 90.97% | 90.97% | 0.00% |
| 4 | 80.56% | 80.21% | -0.35% |
| 5 | 76.74% | 76.04% | -0.70% |
| 6 | 69.44% | 70.83% | +1.39% |
| 7 | 92.36% | 95.14% | +2.78% |
| 8 | 84.72% | 84.38% | -0.34% |
| 9 | 88.19% | 88.89% | +0.70% |
| **平均** | **81.64%** | **81.56%** | **-0.08%** |

- **NTR-S: 4/9 (44.4%) → 2/9 (22.2%)**（mom=0.1 比）
- **WSD: -8.33% → -5.56%**（mom=0.1 比）
- **Mean delta: -0.08%**（ほぼ中立）

### 解釈（観測事実）

1. momentum 制御は NTR-S/WSD 改善に効く。全体設計として mom=0.01 は mom=0.1 より優れる。
2. S2 の harm は mom=0.01 でも -5.56% 残る。**magnitude 制御だけでは S2 は救えない。**
3. S4, S8 は ~0 で実質中立。S5 は -0.70% で軽微な新規害（閾値以下）。
4. **direction が S2 harm の主因である可能性が高い**というのは、この結果から強化される仮説。

---

## 実験 B：S2 direction ablation — mean/var × shallow/deep（各軸）

**設定:** `bn_stat_clean`, `bn_momentum=0.01`, `bs=1`, S2 のみ  
**bn_update_target:** {both, mean_only, var_only, shallow, deep}  
**Config:** `configs/tcformer_otta/tcformer_otta_bs1_s2only.yaml`

### 結果

| 条件 | S2 acc | Δ | entropy | logit_norm |
|------|--------|---|---------|------------|
| source_only   | 67.71% | ref | — | — |
| **both**          | **62.15%** | **-5.56%** | 0.7419 | 3.1725 |
| **mean_only**     | **67.71%** | **0.00%**  | 0.7540 | 3.0212 |
| **var_only**      | **61.11%** | **-6.60%** | 0.7401 | 3.1820 |
| **shallow**       | **62.50%** | **-5.21%** | 0.7583 | 3.0925 |
| **deep**          | **67.71%** | **0.00%**  | 0.7402 | 3.0931 |

### 観測事実（強く言えること）

1. **mean_only = 0.00%**：mean 更新は S2 に対して完全に無害。
2. **deep = 0.00%**：深い BN 層（mix/reduce/TCN）の更新は完全に無害。
3. **var_only = -6.60%**：variance 更新だけで `both` より大きな害が出る。
4. **shallow = -5.21%**：浅い BN 層（conv_block 6 層）だけで `both` の 94% の害を再現。
5. **`both` (-5.56%) が `var_only` (-6.60%) より若干良い**：mean 更新が variance 更新の害を部分的に補正している。

### 最有力仮説（まだ交互作用の直接検証は未実施）

> **S2 の harm は、浅い BN 層の running_var 更新が主に駆動している。**

この仮説は、mean/var 軸と shallow/deep 軸の**各軸アブレーション**では支持されている。  
ただし 2×2 交互作用（shallow_var_only 等）は未検証。

### 言い過ぎになる主張（現時点では仮説止まり）

- 「Shallow × Variance が真因」：2×2 をまだ見ていない。
- 「mean_only が全体設計原則」：S7 の gain を variance 更新が支えている場合に成立しない。

---

## BN 層別 drift norm（S2, 実験 B より）

| 条件 | shallow[0:6] 合計 | deep[6:12] 合計 | 総計 |
|------|-----------------|----------------|------|
| both | 0.0006 | 0.0005 | 0.0011 |
| mean_only | 0.0001 | 0.0003 | 0.0004 |
| var_only | 0.0005 | 0.0002 | 0.0007 |
| shallow | 0.0006 | 0.0000 | 0.0006 |
| deep | 0.0000 | 0.0005 | 0.0005 |

- drift norm の絶対値は 1e-4 オーダーと微小。
- それでも 288 ステップ累積すると ~5% の精度変化が生じる。
- **小さな系統的ズレが方向を持って蓄積する**問題であり、magnitude の問題ではない。

---

## コード変更

### `intentflow/offline/models/pmax_sal_otta.py`

`_update_bn_stats` に 4 つの複合モードを追加：

```
shallow_mean_only  浅い層 × mean のみ
shallow_var_only   浅い層 × var のみ
deep_mean_only     深い層 × mean のみ
deep_var_only      深い層 × var のみ
```

実装方針：既存の `layer_scope` (shallow/deep/all) と `stat_scope` (mean_only/var_only/both) を独立に解析し、compound mode を prefix/suffix で判定。既存 5 モードとの互換性を保持。

### 追加ファイル

| ファイル | 目的 |
|---------|------|
| `configs/tcformer_otta/tcformer_otta_bs1_s27.yaml` | S2/S7 のみ、bs=1 |
| `scripts/run_mom001_all9.sh` | 実験 A 実行スクリプト |
| `scripts/run_direction_ablation.sh` | 実験 B 実行スクリプト |
| `scripts/run_2x2_ablation.sh` | 次の実験（2×2）実行スクリプト |

---

## 次の実験：Option 0（2×2 交互作用アブレーション）

**目的：** shallow/deep × mean/var の交互作用を S2 と S7 で同時に確認し、次に何を全 9 被験者に持っていくかを決める。

**条件（6 run）：** source_only, both, shallow_mean_only, shallow_var_only, deep_mean_only, deep_var_only

**実行コマンド：**
```bash
cd intentflow/offline
./scripts/run_2x2_ablation.sh results/update_op_v2_20260401_125734/source_model 0
```

**解釈分岐：**

| 結果パターン | 意味 | 次の一手 |
|------------|------|---------|
| S2: `shallow_var_only` ≈ `both` かつ S7: `shallow_var_only` ≈ 0 | var-freeze が全体的に安全 | mean_only または hybrid で全 9 |
| S2: `shallow_var_only` ≈ `both` かつ S7: `shallow_var_only` > 0 | variance は被験者依存の両刃 | subject-adaptive rule を検討 |
| `deep_var_only` でも S2 に harm が出る | shallow culprit 説を修正 | 解釈を更新してから再設計 |

---

## 現在地の整理

```
観測済み（強い）
  ├── S2 harm は batch artifact 主因説ではない（bs=1 でも持続）
  ├── momentum 制御は NTR-S/WSD を改善する（4/9→2/9, -8.33→-5.56%）
  ├── S2 では mean 更新は無害、deep BN 更新は無害
  └── S2 では var 更新 / shallow 更新がそれぞれ単独でほぼ全害を再現

最有力仮説（次実験で検証対象）
  └── S2 harm = 浅い BN 層の running_var 更新が主因

未確定
  ├── shallow × var の交互作用（2×2 未実施）
  ├── S7 の gain source が variance 依存かどうか
  └── mean_only が全体最適かどうか（S7 側のコストが不明）
```

---

## 追記（2026-04-08）：Hybrid all-9 fine sweep（5 seeds）

**目的:** 前日の `hybrid = shallow_mean_deep_both @ mom=0.01` が seed 偶然ではないかを確認しつつ、`0.01` と `0.02` の間により良い momentum が存在するかを all-9 × multi-seed で検証する。

**設定:**
- target: `{shallow_mean_deep_both, shallow_mean_deep_mean_only}`
- momentum: `{0.01, 0.0125, 0.015, 0.0175, 0.02}`
- subjects: all 9, `bs=1`, `seed=0..4`
- output: `intentflow/offline/results/hybrid_all9_fine_sweep_20260407_010039`
- launcher log: `intentflow/offline/results/launch_hybrid_all9_fine_sweep_20260407_010039.log`

### 結果サマリー

| target | mom | coverage | meanΔ | worstWSD | safe@0.5 | safe<0 |
|--------|-----|----------|-------|----------|----------|--------|
| shallow_mean_deep_both | 0.01 | 5/5 | **+0.37±0.02** | **-0.35%** | **5/5** | 0/5 |
| shallow_mean_deep_both | 0.0125 | 5/5 | +0.30±0.08 | -0.69% | 4/5 | 0/5 |
| shallow_mean_deep_both | 0.015 | 5/5 | +0.29±0.07 | -1.04% | 2/5 | 0/5 |
| shallow_mean_deep_both | 0.0175 | 5/5 | +0.30±0.09 | -1.04% | 2/5 | 0/5 |
| shallow_mean_deep_both | 0.02 | 5/5 | +0.29±0.06 | -1.39% | 1/5 | 0/5 |
| shallow_mean_deep_mean_only | 0.01 | 5/5 | +0.22±0.02 | -0.69% | 4/5 | 0/5 |
| shallow_mean_deep_mean_only | 0.0125 | 5/5 | +0.31±0.02 | -0.35% | 5/5 | 0/5 |
| shallow_mean_deep_mean_only | 0.015 | 5/5 | +0.30±0.02 | -0.69% | 3/5 | 0/5 |
| shallow_mean_deep_mean_only | 0.0175 | 5/5 | +0.30±0.02 | -0.35% | 5/5 | 0/5 |
| shallow_mean_deep_mean_only | 0.02 | 5/5 | +0.30±0.02 | -0.69% | 3/5 | 0/5 |

**最良条件（material safety 制約下）:** `shallow_mean_deep_both @ mom=0.01`

### 観測事実

1. **全 50 run が完走**し、`Stop early = 0`。今回の結論は途中打ち切りではなく、all-9 × 5 seed の完全なグリッドに基づく。
2. **前日の hybrid@0.01 は再現した。** 単一 seed では `meanΔ=+0.35%`, `WSD=-0.34%` だったが、5 seed でも `+0.37±0.02`, `-0.35%` で安定した。前日のベストは偶然ではなかった。
3. **`0.01` と `0.02` の間に hidden sweet spot があるという仮説は支持されない。** 少なくとも shared scalar momentum では、`shallow_mean_deep_both` の最適点は探索区間の左端 `0.01` にある。
4. **`shallow_mean_deep_both` は momentum 増加に対して安全性が急速に悪化する。** meanΔ はほぼ横ばい（`+0.29 ~ +0.30`）なのに、`safe@0.5` は `5/5 → 4/5 → 2/5 → 2/5 → 1/5` と崩れる。高 momentum は gain を増やすより先に worst-case を壊す。
5. **`shallow_mean_deep_mean_only` はより平坦で頑健だが、平均改善の ceiling が低い。** `0.0125` と `0.0175` は `5/5 material-safe` を維持しつつ `+0.31`, `+0.30` を出すが、`both@0.01` には一貫して及ばない。
6. **strict safety は全条件で未達。** `safe<0 = 0/5` であり、現状の設計空間では micro-harm 完全ゼロよりも `material harm を抑えつつ平均改善を取る` 方が妥当な設計目標である。

### 被験者別の比較（best vs 次点）

`both@0.01` と、mean-only 系で最良だった `mean_only@0.0125` を比べる：

| Subject | both@0.01 | mean_only@0.0125 | 差（both - mean_only） |
|---------|-----------|------------------|------------------------|
| S1 | +1.04 | +1.39 | -0.35 |
| S2 | +0.07 | +0.35 | -0.28 |
| S3 | **+0.69** | **-0.07** | **+0.76** |
| S4 | -0.28 | +0.00 | -0.28 |
| S5 | -0.14 | -0.28 | +0.14 |
| S6 | +0.69 | +0.69 | +0.00 |
| S7 | **+1.04** | **+0.69** | **+0.35** |
| S8 | +0.00 | -0.35 | +0.35 |
| S9 | +0.21 | +0.35 | -0.14 |

### 被験者別に何が起きているか

1. **`both@0.01` の勝ち分は主に S3 と S7 から来る。**
   - S3: `+0.69` vs `-0.07`
   - S7: `+1.04` vs `+0.69`
   - S8 でも `0.00` vs `-0.35` で `both@0.01` が上回る
2. **`mean_only@0.0125` の利点は S1/S2/S4/S9 の安定側にある。**
   - S1: `+1.39`
   - S2: `+0.35`
   - S4: `0.00`
   - S9: `+0.35`
3. **残る micro-harm の被験者分布も異なる。**
   - `both@0.01`: strict harm は主に `S4 (4/5)`, `S5 (2/5)`, `S9 (1/5)`
   - `mean_only@0.0125`: strict harm は主に `S5 (4/5)`, `S8 (5/5)`, `S3 (1/5)`

### momentum に対する反応の論理

#### `shallow_mean_deep_both`

- S7 は momentum を上げるほど伸びる（`+1.04 → +1.74`）
- S9 も高 momentum 側で伸びる（`+0.21 → +0.76`）
- しかしその代償として
  - S3 は `+0.69 → +0.00` に低下
  - S5 は `-0.14 → -0.62` に悪化
  - S8 は `+0.00 → -0.42` に悪化

**解釈:** deep variance 更新は「不要」ではない。むしろ低 momentum では有益だが、shared momentum を上げると S7/S9 の gain を取りに行く過程で S3/S5/S8 を壊す。

#### `shallow_mean_deep_mean_only`

- `0.0125 ~ 0.02` で meanΔ はほぼ一定（`+0.30 ~ +0.31`）
- S1, S6, S7, S9 の gain はほぼ固定
- S5, S8 の micro-harm もほぼ固定

**解釈:** deep mean は shared momentum の微調整に対して鈍感であり、平均改善は安定するが、これだけでは `both@0.01` の S3/S7 利得を取り切れない。

### ここから言えること

1. **前日の主張は multi-seed で補強された。**
   - `hybrid = shallow_mean_deep_both @ 0.01` は、単発偶然ではなく現時点ベストである。
2. **一方で、改善余地のボトルネックも明確になった。**
   - `deep var` を完全に切ると安全だが、S3/S7 側の gain を失う
   - `deep var` を shared momentum で強くすると、S5/S8/S3 側を壊す
3. **したがって次の設計変数は「deep mean と deep var の分離制御」である。**
   - 問題は「var を使うか否か」ではなく、「var をどの強さで混ぜるか」
   - shared scalar momentum はこのトレードオフを吸収できない

### 更新後の現在地

```text
強く支持される観測事実
  ├── S2 harm は shallow var が主因
  ├── shallow var を凍結した hybrid は all-9 で安全性を大きく改善する
  ├── hybrid@0.01 は 5 seed でも現時点ベスト
  └── hidden sweet spot は shared momentum の fine sweep では見つからなかった

新たに明確になったこと
  ├── deep var は低 momentum では net positive
  ├── ただし momentum を上げると S7/S9 gain と引き換えに S3/S5/S8 を壊す
  └── deep mean only は安定だが ceiling が低い

次に検証すべき仮説
  └── deep_mean momentum と deep_var momentum を分離すれば、
      mean_only の安定性を保ったまま both@0.01 の S3/S7 利得を回収できる
```
