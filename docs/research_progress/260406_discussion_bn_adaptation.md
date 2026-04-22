# OTTA BN Adaptation — 因果同定実験の全記録と未解決問題
*2026-04-06 作成。Codex レビュー反映済み（同日）。*

---

## このドキュメントの目的

EEG 運動想起の cross-session OTTA（Online Test-Time Adaptation）において、**BN（Batch Normalization）の running statistics 更新が特定被験者に害を与える問題**を機構レベルで追跡した実験記録と、現時点での仮説・未解決問題をまとめる。

観測事実と機構仮説は意図的に分離して記述する。

---

## 研究の文脈

### タスク設定

- Dataset: BCIC-IV 2a（9 被験者、4 クラス運動想起 EEG、288 試行/被験者）
- Split: session_T（学習）→ session_E（評価）。cross-session covariate shift が主な課題。
- Model: TCFormer（77,840 params）。BN 層 12 個（前半 6: conv_block、後半 6: group attention + TCN）。
- OTTA: test 時に BN running statistics を逐次更新。適応ゲート（pmax > 0.7 AND SAL > 0.5 AND energy ≤ θ_95）を通過した試行のみ更新。
- 評価: accuracy（%）、NTR-S（Negative Transfer Rate）、WSD（Worst Subject Delta）。

### NTR-S の定義（本文書での扱い）

NTR-S には閾値依存性があるため、以下の2段階で区別する：

- **strict NTR-S**（Δ < 0）: adapted acc が source を 1 試行でも下回る被験者の割合
- **NTR-S@0.5pp**（Δ < -0.5%）: material harm（0.5%pp 以上の悪化）を受ける被験者の割合

両者を混在させると「0/9 達成」の意味が変わる。本文書では必ず区別して記載する。

### 出発点となった問題

vanilla OTTA（bn_stat_clean, bn_momentum=0.1, bs=48）では：
- S2 が -8.33% という大きな害を受ける
- strict NTR-S = 4/9、NTR-S@0.5pp = 4/9、WSD = -8.33%

---

## 実験チェーン（実施順）

### Step 1: Batch artifact 仮説の検証（bs=48 → bs=1）

**問い:** S2 の害は bs=48 の評価アーティファクト（適応済み試行と未適応試行が同一バッチに混在）が原因か？

**結果:**

| 条件 | S2 acc | Δ from source |
|------|--------|---------------|
| source_only | 67.71% | ref |
| bn_stat_clean, mom=0.1, bs=48 | 59.38% | -8.33% |
| bn_stat_clean, mom=0.1, bs=1 | 59.72% | -7.99% |

**観測事実:** bs=1 でも -7.99% の害が残る。  
**結論:** batch artifact は実在するが主因ではない（寄与は ~0.3%）。害の本体は BN の累積 drift。

---

### Step 2: Magnitude 制御（momentum sweep）

**問い:** 適応の「強さ」を下げれば S2 の害は消えるか？

**設定:** bn_stat_clean, bs=1, S2/S7/S9、momentum ∈ {0.1, 0.01, 0.001}

**結果（probe sweep、最新数値）:**

| momentum | S2 | S7 | S9 |
|----------|----|----|----|
| source | 67.71 | 92.36 | 88.19 |
| 0.1 | 59.72 (-7.99) | 94.44 (+2.08) | 86.11 (-2.08) |
| 0.01 | 62.15 (-5.56) | 95.14 (+2.78) | 88.89 (+0.70) |
| 0.001 | 66.67 (-1.04) | 93.06 (+0.70) | 88.54 (+0.35) |

**観測事実:**
- S2 の harm は momentum に対して単調に減る
- S7 の gain は mom=0.01 で最大（mom=0.1 で最大ではない）
- mom=0.001 では harm も gain もほぼ消える

**結論:** magnitude 制御は worst-case を緩和するが、S2 の harm を構造的には消せない。したがって causal axis は「更新量」だけではなく「更新方向」にある。

---

### Step 3: Direction ablation（mean/var 軸、shallow/deep 軸）

**問い:** S2 の害は BN の何が（mean か var か）、どの層で（浅いか深いか）更新されることで生じるか？

**設定:** bn_stat_clean, mom=0.01, bs=1, S2 のみ

**結果（精度と drift norm）:**

| bn_update_target | S2 acc | Δ | entropy | logit_norm | shallow drift | deep drift |
|-----------------|--------|---|---------|------------|--------------|------------|
| source_only | 67.71 | ref | — | — | — | — |
| both | 62.15 | -5.56 | 0.742 | 3.173 | 0.0006 | 0.0005 |
| mean_only | 67.71 | 0.00 | 0.754 | 3.021 | 0.0001 | 0.0003 |
| var_only | 61.11 | -6.60 | 0.740 | 3.182 | 0.0005 | 0.0002 |
| shallow | 62.50 | -5.21 | 0.758 | 3.093 | 0.0006 | 0.0000 |
| deep | 67.71 | 0.00 | 0.740 | 3.093 | 0.0000 | 0.0005 |

**観測事実（強度順）:**
1. mean_only = 0.00%：mean 更新は S2 に完全無害
2. deep = 0.00%：深い BN 更新は S2 に完全無害
3. var_only = -6.60%：var 更新だけで both (-5.56%) より大きな害
4. shallow = -5.21%：浅い BN だけで both の 94% の害を再現
5. both が var_only より若干良い（-5.56% vs -6.60%）：mean 更新が var の害を部分補正

**補足:** S2 の適応率は約 5%（288 試行中ゲート通過は ~14 試行）。「更新回数が多すぎる」問題ではなく、「少数の更新でも方向が悪ければ壊れる」という話であり、これは重要な性質である。drift norm は 1e-4 オーダーと微小だが、288 ステップ累積で ~5% の精度変化を生む。magnitude の問題ではなく、方向を持った系統的な蓄積。

---

### Step 4: 2×2 交互作用アブレーション（S2 × S7）

**問い:** "shallow var" という交差項を直接確認する。同時に S7 の gain 源が variance 依存かを検証する。

**設定:** bn_stat_clean, mom=0.01, bs=1, S2/S7

**結果（2×2 テーブル、Δ from source）:**

|  | mean_only | var_only |
|---|---|---|
| **shallow** | S2: -0.35%、S7: +0.35% | S2: **-5.56%**、S7: **+1.74%** |
| **deep** | S2: -0.35%、S7: +0.70% | S2: **0.00%**、S7: +0.35% |

参照: `both` = S2: -5.56%、S7: +2.78%

**観測事実:**

S2 について：
- shallow × var = -5.56% = both（完全一致）
- 他 3 セルは 0.00% または -0.35%（実質無害）
- S2 の観測された harm のほぼ全量は shallow variance 更新で再現される

S7 について：
- shallow × var = +1.74%（both +2.78% の最大単一寄与）
- 4 成分の isolated effect の総和 = +3.14% > both = +2.78%
- したがって寄与は加法的ではない。より正確には、各更新条件が後続試行の適応ゲート通過率と BN trajectory を変えるという **stateful OTTA に由来する非加法性**が存在する（「超加法性」という表現は不正確）

**最重要帰結:**
> 同じ操作（shallow BN running_var の更新）が、S2 では -5.56%（有害）、S7 では +1.74%（有益）として働く。問題は「variance 更新の有無」ではなく「variance 更新が向かう方向の被験者依存性」。

---

### Step 5: Hybrid 設計の全 9 被験者検証

**設計根拠:** S2 の culprit = shallow var → shallow var のみ凍結し、それ以外（shallow mean, deep both）は自由に更新する最小介入設計。

**bn_update_target = `shallow_mean_deep_both`:**
- shallow BN 6 層：running_mean のみ更新（running_var は凍結）
- deep BN 6 層：running_mean + running_var 両方更新

**設定:** bn_stat_clean, mom=0.01, bs=1, 全 9 被験者

**結果（Δ from source）:**

| S | source | both(0.01) | hybrid | mean_only |
|---|--------|-----------|--------|-----------|
| 1 | 84.03 | +1.39 | +1.04 | +1.39 |
| **2** | **67.71** | **-5.56** | **+0.00** | **+0.00** |
| 3 | 90.97 | +0.00 | **+0.70** | **-0.69** |
| 4 | 80.56 | -0.35 | +0.00 | +0.00 |
| 5 | 76.74 | -0.70 | +0.00 | -0.35 |
| 6 | 69.44 | +1.39 | +0.70 | +0.70 |
| 7 | 92.36 | +2.78 | +1.04 | +0.70 |
| 8 | 84.72 | -0.34 | +0.00 | -0.34 |
| 9 | 88.19 | +0.70 | -0.34 | +0.35 |

**サマリー統計（NTR-S は2段階で報告）:**

| 指標 | both(mom=0.1) | both(mom=0.01) | hybrid | mean_only |
|------|--------------|----------------|--------|-----------|
| strict NTR-S (Δ < 0) | 4/9=44.4% | 4/9=44.4% | 1/9=11.1% | 3/9=33.3% |
| NTR-S@0.5pp (Δ < -0.5%) | 4/9=44.4% | 2/9=22.2% | **0/9=0%** | 1/9=11.1% |
| WSD | -8.33% | -5.56% | **-0.34%** | -0.69% |
| mean Δ | -1.52% | -0.08% | **+0.35%** | +0.20% |

**注意:** hybrid の NTR-S@0.5pp=0/9 は「material harm ゼロ」を意味する。strict な意味では S9 に -0.34%（1 試行分）の micro-harm が残っており、これが noise か真のシグナルかは単一 seed では判断できない。

**主な観測事実:**

1. **S2 完全救済**: hybrid で +0.00%（both(0.01) では -5.56% が残っていた）
2. **S3 の新発見**: hybrid = +0.70%、mean_only = -0.69%、差 = 1.39%。deep var 更新が S3 にも有益。mean_only は S3 に対して有害であり、「var を全凍結すれば安全」という保守設計が成立しないことを示す。
3. **hybrid vs mean_only 被験者別勝敗**: hybrid 優位 4/9、mean_only 優位 2/9、同等 3/9
4. **hybrid は mean_only より優位**: strict/material の両方の NTR-S、WSD、mean Δ の全指標で hybrid が上回る

**S2 rescue の全履歴:**

```
source                    : 67.71%  (ref)
both, mom=0.1, bs=48      : 59.38%  (Δ = -8.33%)
both, mom=0.1, bs=1       : 59.72%  (Δ = -7.99%)
both, mom=0.01, bs=1      : 62.15%  (Δ = -5.56%)
hybrid, mom=0.01, bs=1    : 67.71%  (Δ =  0.00%)  ← shallow var 凍結で完全回復
mean_only, mom=0.01, bs=1 : 67.71%  (Δ =  0.00%)  ← 過保守で同等（ただし S3 に有害）
```

---

## 現時点での仮説の整理

### 確定した観測事実（設計判断上、十分な根拠がある）

1. S2 の観測された harm のほぼ全量は `shallow BN running_var` 更新で再現できる
2. mean 更新は S2 において無害（わずかに補正として働く）
3. deep BN 更新は S2 において無害
4. shallow var 更新は被験者依存の両刃：S2 には -5.56%、S7 には +1.74%
5. deep var 更新は S3/S7 に有益（mean_only で凍結すると harm になる）
6. hybrid（shallow var のみ凍結）は NTR-S@0.5pp=0/9 を達成する

### 最有力仮説（実験的根拠が強いが、機構は未直接確認）

**仮説 1: cross-session covariate shift は層によって性質が異なる**

shallow BN（時空間特徴抽出）での分散シフトは被験者依存の方向を持ち、追従が有害にも有益にもなる。deep BN（attention/TCN）での分散シフトへの追従は、調べた範囲では有害方向に動かない。これは入力側の生理的変動（浅い特徴）と抽象的なタスク表現（深い特徴）では session 間変動の性質が異なるという解釈と整合する。ただし特徴マップを直接確認したわけではない。

**仮説 2: shallow var の「方向」は被験者ごとのセッション間 SNR 変化を反映している**

S2 の session_E でノイズ増加または信号減少が生じ、BN variance がそれを取り込む → 特徴スケールが壊れる、という経路が考えられる。S7 では逆にセッション E の安定性が増し、variance 追従が beneficial になるという解釈。これは妥当な仮説だが、entropy/logit_norm/bn_drift_layers の時系列からは現時点では直接確認できていない。

### 未確定・未検証の問い

**問い 1: shallow/deep 分割の実装整合性**

現実装は「BN 層を列挙して前半 6 個を shallow」としている。TCFormer 側は conv_block に 6 個、mix/reduce/TCN に 6 個の BN があり、アーキテクチャの 6+6 分割と一致している（Codex 確認済み）。したがって本実験の解釈を壊すズレは現時点では確認されていない。

ただし実装は「列挙順の前半/後半」に依存しており、将来のモデル変更には脆い。module-name prefix で指定する方が安全。

**問い 2: test wall-time anomaly（S4、S9 の異常 Test Time）**

S4（47s）と S9（106s）で hybrid/mean_only の Test Time が異常に長い。source_only では発生しない。一方、Average Response Time は全条件で 5.6〜5.7ms と安定しており、`trainer.test()` 前後の wall-time 計測（datamodule.setup や set_train_dataloader を含む）が汚染している可能性が高い。純粋なオンライン推論 latency の問題とは区別すべき。ただし原因は未確定であり、調査は要する。

**問い 3: seed 安定性**

全実験が seed=0 の単一 run。1% 未満の差（S9: -0.34%、S1: +1.04% 等）は 288 試行中 1〜3 試行の差であり、seed で符号が変わりうる。Step 3〜4（S2 の因果切り分け）は差が大きくシード耐性が高いと考えられるが、Step 5 の S1/S8/S9 付近の micro 差は多シード確認が必要。

**問い 4: なぜ shallow var が被験者依存の方向を持つか**

機構の直接証明には、shallow 特徴マップの class-wise 分散や logit margin を S2/S7 で保存して比較する実験が必要。現在の entropy/logit_norm/bn_drift_layers だけでは「なぜ逆符号か」の直接説明には届いていない。

**問い 5: S2 型と S7 型を事前に判別できるか**

session_T のデータから、その被験者の shallow var が有害方向に動くかどうかを予測できれば subject-adaptive rule が可能になる。候補特徴量（session_T の class-wise 分散の安定性、logit margin の分布など）は未検討。

---

## 現設計（hybrid）の残存リスク

| リスク | 規模 | 現在の評価 |
|--------|------|-----------|
| S9 micro-harm（-0.34%） | 1 試行分 | noise か真のシグナルか seed 未確認 |
| S4/S9 の Test Time 異常 | wall-time 汚染の可能性大 | 推論 latency の問題とは別に調査要 |
| shallow 定義の実装脆弱性 | 将来のモデル変更に影響 | 現状は整合確認済み、リファクタ推奨 |
| seed=0 単一 run | 統計的信頼性 | Step 3-4 は強い、Step 5 micro 差は要確認 |
| なぜ動くかの機構未解明 | 新被験者への汎化 | 設計の外挿根拠が現時点では薄い |

---

## 研究の現在地を一言で言うと

本研究はもう「BN を更新すると危ないことがある」段階ではなく、「どの統計を、どの深さで、どこまで残すか」という設計論に入っている。

現時点の最も強い主張は：

> 「cross-session OTTA における BN adaptation の失敗は BN 全体の適応に起因するのではなく、shallow running_var の被験者依存 drift に局在している。この特定の操作を凍結する最小介入設計（hybrid）は、material harm を全被験者で排除しつつ、mean_only より多くの positive transfer を保持する。」

機構仮説（SNR 変化の反映、session 間 covariate shift の層依存性）は妥当であるが、現データから直接証明はされていない。

---

## 実験コードの構成（参考）

```
intentflow/offline/
  models/
    pmax_sal_otta.py     # BN 更新コア。bn_update_target で更新対象を制御。
    tcformer_otta.py     # Lightning wrapper。test_step で stats を収集。
  configs/tcformer_otta/
    tcformer_otta_bs1.yaml          # 全 9 被験者、bs=1
    tcformer_otta_bs1_s2only.yaml   # S2 のみ
    tcformer_otta_bs1_s27.yaml      # S2/S7 のみ
    tcformer_otta_bs1_s279.yaml     # S2/S7/S9 のみ
  scripts/
    run_momentum_sweep.sh           # Step 2
    run_direction_ablation.sh       # Step 3
    run_2x2_ablation.sh             # Step 4
    run_hybrid_vs_meanonly.sh       # Step 5
```

**`bn_update_target` 実装済みモード（pmax_sal_otta.py）:**

```
both                   # mean + var, 全層（デフォルト）
mean_only              # mean のみ, 全層
var_only               # var のみ, 全層
shallow                # mean + var, 前半 6 層（conv_block）
deep                   # mean + var, 後半 6 層（mix/reduce/TCN）
shallow_mean_only      # mean のみ, 前半 6 層
shallow_var_only       # var のみ, 前半 6 層
deep_mean_only         # mean のみ, 後半 6 層
deep_var_only          # var のみ, 後半 6 層
shallow_mean_deep_both # 前半: mean のみ（var 凍結）/ 後半: mean + var   ← hybrid
```
