# 現時点のモデル一覧 — 処理フローと精度

> **このドキュメントの目的**
> 4/22 以降に増えた OTTA バリアントを整理し、「どれが何の発展型で、いま精度がどうなっているか」を 1 ページで追えるようにする。

---

## 0. ひとことで

5 系統ある。**論理的にはすべて同じ TCFormer source を共有していて、test-time の "適応の仕方" だけが違う**。

```
TCFormer (source 学習)
  │
  ├── source_only (適応しない、ベースライン)
  │
  ├── tcformer_otta              ← 前回ゼミの vanilla / hybrid
  │   ├── vanilla       : 全 BN を mom=0.1 で更新     [前回 red bar]
  │   ├── hybrid@0.01   : shallow var だけ凍結       [前回 best 81.98%]
  │   └── …             : 他の bn_update_target 派生
  │
  ├── tcformer_proto_otta        ← 案 E'' 素体 (4/29 実装)
  │   └── BN 完全凍結 + deep prototype EMA + Tri-Lock
  │
  ├── tcformer_policy_safe_otta  ← 案 E'' 拡張 (4/29 実装)
  │   ├── policy_safe_default   : RuleBasedPolicy + 即時 SafeCommit + OperatorBank
  │   └── policy_safe_no_shallow: operator pool から shallow_var を除外
  │
  └── tcformer_replay_safe_otta  ← 今回の本命 (5/6 実装)
      ├── replay_safe_uniform   : 二段 SafeCommit (Tier1 即時 + Tier2 replay reward)
      └── replay_h6_weighted    : replay reward を class-inverse 重みで集約
```

---

## 1. 各モデルの処理フロー

### 1-1. source_only（ベースライン）

> 適応を一切しない。test session でも source の重み・BN stat をそのまま使う。

```
EEG x → TCFormer (eval, no_grad) → logits → argmax → pred
```

**役割**: すべての OTTA 変種のリファレンス。Δ vs source_only で評価する。

---

### 1-2. tcformer_otta — vanilla / hybrid (前回ゼミの主役)

> Tri-Lock (pmax × SAL × Energy) で trial を選び、選ばれた trial だけで BN running stats を更新する。
> 前回ゼミでは vanilla（全 BN 一律更新）の害を、hybrid（shallow var 凍結）が消したのが core finding。

```
EEG x
  ↓
TCFormer forward → logits, features
  ↓
Tri-Lock gate: pmax > 0.7  AND  SAL > 0.5  AND  energy ≤ source_thresh
  ↓
pass: BN running_mean / running_var を mom=0.01 で更新
       (ただし bn_update_target で 何を更新するか制御; hybrid は shallow:mean のみ / deep:mean+var)
fail: 何もしない (fail-closed)
  ↓
予測は更新後の BN で再 forward
```

**重要 config**

| 設定 | 意味 |
|---|---|
| `bn_momentum=0.01` | BN stat の更新の強さ |
| `bn_update_target='shallow_mean_deep_both'` | hybrid の core: 浅い層の var を凍結 |
| `pmax_threshold=0.7`, `sal_threshold=0.5`, `energy_quantile=0.95` | Tri-Lock 閾値 |

---

### 1-3. tcformer_proto_otta — 案 E'' 素体

> BN は完全凍結、その代わり **deep prototype の EMA** で test-time 表現を寄せる。
> Tri-Lock は同じ。fusion alpha でクラス分類の補強として混ぜる。

```
EEG x
  ↓
TCFormer forward (BN 凍結) → classifier_logits, deep_features
  ↓
Tri-Lock gate
  ↓
pass: target_prototype[ŷ] を deep_features で EMA 更新 (m=0.05)
       次の forward で logits = (1-α)·classifier_logits + α·proto_logits
fail: 何もしない
```

**hybrid との違い**: 「BN を動かさず、prototype を動かす」逆方向の介入。case では低危険な代替策。

**重要 config**: `fusion_alpha=0.3`, `proto_momentum=0.05`

---

### 1-4. tcformer_policy_safe_otta — Policy + SafeCommit + OperatorBank

> 「単一 operator 強制」じゃなく、**複数の更新候補から policy が並べ、SafeCommit が安全性検査して採否を決める**設計。
> ただし forward-only OTTA では即時 SafeCommit が機能しないことを今回発見した。

```
EEG x
  ↓
StateExtractor: pmax / SAL / energy / margin / shallow_var_drift_risk / ... を抽出
  ↓
RuleBasedPolicy: state を見て candidate operators を順序付きで提案
  例) [shallow_var_update, hybrid_BN_update, deep_BN_update,
       prototype_update, logit_bias_update, no_update, abstain]
  ↓
for each candidate:
   1. snapshot model state
   2. 仮 apply (operator を実行)
   3. SafeCommit.is_safe(): margin / sal / proto_margin / energy / bn_drift / shallow_var_delta を before vs after で比較
        fail → restore して次 candidate へ
        pass → COMMIT、ループ終了
   ↓
すべて reject なら no_update
   ↓
abstain なら出力もしない
```

**重要 config**

| 設定 | 意味 |
|---|---|
| `prototype_fusion_alpha=0.2` | proto を logits に何割混ぜるか |
| `allowed_operators` | policy が選べる operator 集合 (これで `no_shallow` 派生を作る) |
| `pmax_threshold=0.7`, `sal_threshold=0.5` | very_safe 判定の閾値 |
| `margin_tolerance=0.02`, `sal_tolerance=0.05` | Tier1 ガードの許容幅 |

**問題点（今回見つかった）**: `pred_changed=0` が全 trial で起きる（同 trial 内では更新の効果が出ない）。

---

### 1-5. tcformer_replay_safe_otta — 今回の本命

> Tier 1（即時 SafeCommit）の上に、**Tier 2（過去 K trial の replay buffer 上で simulated reward を計算するゲート）** を追加。
> 「commit の真の効果は将来の trial に出る」を replay で前借り評価する。

```
EEG x
  ↓
StateExtractor → RuleBasedPolicy: candidate operators 提案
  ↓
for each candidate:
   1. snapshot
   2. 仮 apply
   3. Tier 1: 既存の即時 SafeCommit (margin / sal / proto_margin / energy / bn_drift)
        fail → restore → 次 candidate
        pass → ↓
   4. Tier 2: replay buffer (32 trial; warmup は source class-balanced) を fused logits で forward
              sim_score = w_acc·Δreplay_acc + w_mar·Δreplay_margin + w_pro·Δreplay_proto_cos
        sim_score ≤ 0 → restore → 次 candidate
        sim_score > 0 → COMMIT
   ↓
現 trial も pmax > 0.85 AND SAL > 0.6 なら buffer に FIFO admit
```

**replay buffer**: capacity=32, source seed=8/class, target session で pmax+SAL 二重ゲートを通った trial を pseudo-label と一緒に admit。

**variant**:
- `replay_safe_uniform`: replay buffer の各 trial を等重みで集約
- `replay_h6_weighted`: pmax × class-inverse-frequency で重み付け（弱 source 救済目的）

---

## 2. 精度比較表

### 2-1. aug-True 9 被験者、1 seed（**いま一番強い結果**）

source: `intentflow/offline/results/c_aug_true_9subj_20260506_004923/`

| variant | mean | Δmean | Δworst | HSC |
|---|---:|---:|---:|---:|
| source_only | 82.72 | — | — | 0/9 |
| policy_safe_no_shallow | 83.14 | +0.42 | −0.35 | 0/9 |
| **replay_safe_uniform** | **83.41** | **+0.69** | −0.35 | **0/9** |
| replay_h6_weighted | 83.22 | +0.50 | −0.35 | 0/9 |
| (前回ゼミ値) hybrid@0.01 5 seed | **81.98** | +0.35 | −0.34 | 0/9 |

**読み**: replay_safe_uniform が前回 best を **+1.43pp**。HSC=0/9 のまま。

### 2-2. aug-False 9 被験者、1 seed（補助）

source: `intentflow/offline/results/replay_h6_9subject_20260506_004038/`

| variant | mean | Δmean | Δworst | HSC |
|---|---:|---:|---:|---:|
| source_only | 79.82 | — | — | 0/9 |
| proto_otta_default | 79.82 | −0.00 | −1.05 | 2/9 |
| policy_safe_default | 79.86 | +0.04 | −3.12 | 3/9 |
| policy_safe_no_shallow | 80.13 | +0.31 | −1.73 | 1/9 |
| replay_safe_uniform | 80.36 | +0.54 | −1.73 | 1/9 |
| replay_h6_weighted | 80.36 | +0.54 | −1.73 | 1/9 |

**読み**: aug-False の弱 source 環境では、**aug 抜きでも replay が頭一つ抜ける**。aug-True と違って worst Δ=−1.73 (S4) が残っているのが弱点。

### 2-3. 5 seed × 4 被験者（seed-stability）

source: `intentflow/offline/results/b_5seed_4subj_20260506_005153/`

| variant | S2 (62) | S4 (75) | S6 (67) | S7 (87) |
|---|---:|---:|---:|---:|
| source_only | 64.93 ± 1.72 | 80.67 ± 1.80 | 66.43 ± 2.36 | 90.51 ± 1.31 |
| policy_safe_no_shallow | 65.16 ± 1.93 | 80.56 ± 2.04 | 66.43 ± 2.31 | 91.32 ± 1.13 |
| replay_safe_uniform | 65.05 ± 1.82 | 80.55 ± 2.21 | 66.32 ± 2.47 | **92.13 ± 0.91** |
| replay_h6_weighted | **65.28 ± 1.98** | 80.67 ± 2.31 | 66.32 ± 2.47 | **92.36 ± 0.85** |

**読み**: S7 で replay が +1.62 ± 0.91、std が 35% 縮む。S2/S4/S6 では誤差範囲。**前回見えた「S2 で shallow_var が害」は seed=0 アーティファクトだった可能性が高い**。

---

## 3. 「迷ったらこれ」ガイド

| やりたいこと | 使うモデル | config |
|---|---|---|
| 前回ゼミの再現 / hybrid baseline | `tcformer_otta` | `bn_update_target='shallow_mean_deep_both'`, `bn_momentum=0.01` |
| 案 E'' 素体（最小 OTTA、BN 動かさない） | `tcformer_proto_otta` | `fusion_alpha=0.3`, `use_energy_gate=true` |
| Policy + 即時 SafeCommit を試したい（今は不採用） | `tcformer_policy_safe_otta` | デフォルト |
| 上記から shallow_var を抜く（今のサブ best） | `tcformer_policy_safe_otta` | `allowed_operators=[..., 'no_update', 'abstain']` から `shallow_var_update` を除外 |
| **今の本命、論文書くならこれ** | **`tcformer_replay_safe_otta`** | `replay_weight_mode='uniform'`, `replay_capacity=32`, `sim_score_tolerance=0.0` |

---

## 4. なぜ replay_safe が "強い" のか（直感）

短く言うと:

1. **Tri-Lock** で trial を選ぶ → vanilla の暴走を止める（前回 hybrid と同じ役割）
2. **OperatorBank で複数選択肢を持つ** → "shallow_var が万能害" ではなく、subject によって有効な operator が違うので
3. **Tier1 で安全圏かを即チェック** → 明らかに壊す候補は弾く
4. **Tier2 で過去 K trial の replay shoot** → 「次に効くか」を simulate して、効かないなら commit しない

**この 4 段の合わせ技** で、(a) 害を出さない, (b) 効くときだけ commit する、を両立している。

---

## 5. 次の不安要素

- aug-True では `replay_safe_uniform` が現状 best だが **1 seed のみ**。5 seed で再現するか未確認。
- 同 regime で hybrid@0.01 を再評価していない（fairness 比較が未完）。
- `S4`（worst Δ=−0.35）は replay でも救えていない。subject-conditional meta-gate が次の論点。
- BCIC2b では未検証。

これらは [260511_seminar_progress.md](260511_seminar_progress.md) の Section 6 にある今後 2 週間の TODO に相当。

---

## 関連ファイル

### コード（`intentflow/offline/models/`）

| モデル名 | 主実装 | 補助実装 |
|---|---|---|
| tcformer_otta | `tcformer_otta.py` | `pmax_sal_otta.py` |
| tcformer_proto_otta | `tcformer_proto_otta.py` | `prototype_ema_otta.py` |
| tcformer_policy_safe_otta | `tcformer_policy_safe_otta.py` | `policy_safe_commit_otta.py` |
| tcformer_replay_safe_otta | `tcformer_replay_safe_otta.py` | `replay_safe_commit_otta.py`, `replay_buffer.py` |

### 主要 config（`intentflow/offline/configs/`）

- `tcformer_otta/` (vanilla / hybrid 系)
- `tcformer_proto_otta/tcformer_proto_otta_s2_smoke.yaml`
- `tcformer_policy_safe_otta/tcformer_policy_safe_otta_s2_smoke.yaml`
- `tcformer_replay_safe_otta/tcformer_replay_safe_otta_s2_smoke.yaml`
- `tcformer_replay_safe_otta/tcformer_replay_safe_otta_h6_weighted.yaml`

### ゼミ進捗 md

- 前回: `docs/research_progress/ゼミ資料/260420_narukawa.pdf`
- 今回: `docs/research_progress/260511_seminar_progress.md`
