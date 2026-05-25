# 260511 第二回班ゼミ — スライド最終構成案

> EEG-MI cross-session OTTA における Replay-SafeCommit の検討
> 副題: 平均精度向上と subject-level harm 抑制の両立に向けて

---

## 全体の読み筋

```
背景:   EEG-MI は session drift で壊れる
問題:   OTTA は有望だが誤更新で harm が出る
前回:   固定ルール hybrid で harm 抑制は見えた
今回:   固定ルールではなく、更新候補を選ぶ Replay-SafeCommit を検討
結果:   source_only より +0.69pp、HSC=0/9
課題:   平均精度の押し上げはまだ限定的
次:     BTTA-DG の Bayesian calibration を candidate operator として追加する
```

---

# Slide 1: タイトル

**EEG-MI cross-session OTTA における Replay-SafeCommit の検討**
副題: 平均精度向上と subject-level harm 抑制の両立に向けて

### 置く内容

- 史研究室 M2
- 生川 聖也
- 日付
- キーワード: EEG-MI / OTTA / Replay-SafeCommit / BTTA-DG

---

# Slide 2: アジェンダ

**今回の進捗: 固定更新から、候補更新を選ぶ OTTA へ**

```
1. 背景・目的
2. 前回までの到達点と今回の問い
3. 提案: Replay-SafeCommit
4. 実験結果
5. 論文調査: BTTA-DG
6. 今後の方針
```

### 右側の一言

> 今回は、複数の更新候補を replay buffer で検証し、trial ごとに安全な更新を選ぶ方針を検討する。

---

# 1. 背景・目的

## Slide 3: EEG-MI は session が変わると分布がずれる

**主張**: 同じ被験者でも、session が変わるだけで EEG 分布はずれる

### 置く内容

- EEG-MI: 脳波から運動想起を分類
- cross-session drift: 学習 session と評価 session の分布ずれ
- 再キャリブレーションなしで使いたいという実用要件

### 図案

```
session_T で学習
        ↓ drift
session_E で評価
        ↓
精度低下
```

### 下部メッセージ

> 実用上の障壁は、同じ被験者でも毎回 EEG 分布が変わる点にある。

---

## Slide 4: OTTA は drift に追従できるが、誤更新で壊れる

**主張**: OTTA は有望だが、誤った trial を使って更新すると Negative Transfer が起きる

### 置く内容

- OTTA: target session の unlabeled stream を見ながら逐次更新
- drift に追従できる可能性がある
- 一方で、EEG では高 pmax が正解を保証しない
- artifact / session drift による高信頼誤分類がある

### 図案

```
高pmaxだが誤分類
       ↓
誤った更新
       ↓
以降の trial も悪化
```

### 下部メッセージ

> OTTA の本質的な難しさは、「更新するかどうか」だけでなく「何を更新するか」にある。

---

## Slide 5: 本研究の評価軸

**主張**: 平均精度だけではなく、誰かを壊していないかを見る

### 置く表

| 指標 | 意味 | 目的 |
|---|---|---|
| mean accuracy | 全被験者平均の精度 | 全体性能 |
| Δ vs source | 適応なしからの改善幅 | OTTA の価値 |
| WSD | 最悪被験者の悪化幅 | worst-case harm |
| **HSC@0.5pp** | 0.5pp 以上悪化した被験者数 | material harm |

### 下部メッセージ

> 目標は、mean Δ を上げつつ HSC=0/9 を維持すること。

### footnote

`HSC@0.5pp = Harm-Subject Count at 0.5pp threshold (前回 NTR-S@0.5pp と呼んでいた指標を改名。定義は不変)`

---

# 2. 前回までの到達点と今回の問い

## Slide 6: 前回 — 固定ルール hybrid で harm 抑制の方向性を確認

**主張**: 前回は BN 更新範囲を固定的に制御し、subject-level harm を抑えた

### 置く内容

- hybrid@0.01 の要約
  - shallow var を凍結
  - deep 側の適応余地は残す
  - HSC = 0/9
- ただし固定ルール

### 表

| method | mean acc | mean Δ | WSD | HSC |
|---|---:|---:|---:|---:|
| hybrid@0.01 | 81.98 | +0.35 | −0.34 | 0/9 |

### 注意書き

> 今回の主比較は hybrid ではなく、**同一実験内の source_only と replay_safe**。

---

## Slide 7: 固定ルールの限界

**主張**: 固定ルールでは、trial / subject ごとの適切な更新を拾いきれない

### 置く内容

- ある subject では BN 更新が効く
- 別の subject では BN 更新が害になる
- ある trial では prototype 更新が安全
- 別の trial では no_update が最善

### 図案

```
固定更新:
   trial → 常に同じ更新

選択型更新:
   trial → 状態を見る → 候補更新から選ぶ
```

### 下部メッセージ

> 固定ルールの次は、trial ごとに更新候補を選ぶ枠組みが必要。

---

## Slide 8: 今回の問い

**主張**: 複数の更新候補から、安全に効くものだけを選べないか

### 置く内容

> 各 trial に対して複数の更新候補を用意し、その時点で安全かつ改善が見込める候補だけを commit できないか。

### 候補例

```
- prototype_update
- logit_bias_update
- deep_BN_update
- hybrid_BN_update
- shallow_var_update
- no_update
```

### 下部メッセージ

> 今回の主題は「新しい単一更新則」ではなく、「**更新候補を選ぶ仕組み**」である。

---

# 3. 提案: Replay-SafeCommit

## Slide 9: 提案概要

**主張**: Replay-SafeCommit は、候補更新を replay buffer で検証してから採用する

### 置く内容

```
1. 現在 trial の状態を読む
2. 更新候補を複数出す
3. 各候補を仮適用する
4. replay buffer 上で再評価する
5. 改善が見込める候補だけ commit
6. 改善しなければ rollback / no_update
```

### 図案

```
trial
  ↓
状態抽出
  ↓
候補更新
  ↓
仮適用
  ↓
replay 検証
  ↓
commit / rollback
```

---

## Slide 10: replay buffer 上で何を評価しているか

**主張**: Replay buffer は、target session 内の小さな検証集合として候補更新を評価する

### 置く内容

- FIFO 32 件
- 高信頼 trial を保存
- pseudo-label 付き
- 候補更新の前後で replay 指標を比較

### 図案

```
candidate update を仮適用
        ↓
   Replay Buffer
過去 K 件の高信頼 trial
        ↓
   更新前後を比較
        ↓
sim_score > 0 ?
   Yes → commit
   No  → rollback
```

### 下部メッセージ

> replay buffer は、未来 trial の代わりに使う **target session 内の小さな検証集合** である。

---

## Slide 11: Replay-SafeCommit の判断基準

**主張**: replay 上の simulated reward が正の候補だけを commit する

### 数式

```
sim_score =
   w_acc · Δreplay_acc
 + w_mar · Δreplay_margin
 + w_pro · Δreplay_proto_cos
```

### 表

| 項 | 何を見ているか | 直感 |
|---|---|---|
| Δreplay_acc | pseudo-label との一致率変化 | 過去の安定 trial を壊していないか |
| Δreplay_margin | top1 − top2 の予測余裕変化 | 予測が明確になったか |
| Δreplay_proto_cos | feature と prototype の整合変化 | target 特徴空間に近づいたか |

### 判定

```
sim_score > 0  → commit
sim_score ≤ 0  → reject & rollback
```

---

## Slide 12: 各 operator は TCFormer のどこを適応するか

**主張**: candidate operator ごとに、TCFormer 内の適応箇所が異なる

### 図案: TCFormer 適応マップ

```
Input EEG
   ↓
Shallow BN
   ↓
Deep BN
   ↓
Feature representation
   ↓
Classifier logits
   ↓
Prediction
```

### 対応付け

```
shallow_var_update    → shallow BN running_var
hybrid_BN_update      → shallow BN running_mean + deep BN running_mean/var
deep_BN_update        → deep BN running_mean/var
prototype_update      → feature representation の target_prototypes
logit_bias_update     → classifier logits の logit_bias
no_update             → 更新なし
```

### 下部メッセージ

> Replay-SafeCommit は、TCFormer の **異なる位置を更新する候補群から選択** する。

---

## Slide 13: operator の要約表

**主張**: Replay-SafeCommit は、作用点の異なる更新候補から trial ごとに選ぶ

### 表

| operator | 適応箇所 | 変えるもの | 主リスク |
|---|---|---|---|
| prototype_update | feature space | target prototypes | prototype 汚染 |
| logit_bias_update | logit space | logit bias | class 偏り |
| deep_BN_update | deep BN | mean / var | 不安定化 |
| hybrid_BN_update | shallow + deep BN | shallow mean, deep mean/var | 固定ルール |
| shallow_var_update | shallow BN | running_var | 高リスク |
| no_update | — | none | 改善機会損失 |

### 下部メッセージ

> 重要なのは「1 つの更新則を固定で使う」のではなく、「**異なる作用点を持つ候補群から trial ごとに選ぶ**」点。

---

# 4. 実験結果

## Slide 14: 実験設定

**主張**: まずは BCIC IV-2a の aug-True 9 被験者で source_only と比較した

### 置く内容

- dataset: BCIC IV-2a
- setting: cross-session
- subjects: 9
- main comparison: source_only vs replay_safe
- metrics: mean acc / Δ vs source / WSD / HSC

### 比較対象

```
source_only
policy_safe_no_shallow
replay_safe_uniform
replay_h6_weighted
```

### 下部メッセージ

> OTTA の価値は「適応しない場合」からどれだけ改善したかで判断する。

---

## Slide 15: 主結果

**主張**: Replay-SafeCommit は source_only より平均精度を改善し、HSC=0/9 を維持した

### 表

| method | mean acc | Δ vs source | worst Δ | HSC |
|---|---:|---:|---:|---:|
| source_only | 82.72 | — | — | 0/9 |
| policy_safe_no_shallow | 83.14 | +0.42 | −0.35 | 0/9 |
| **replay_safe_uniform** | **83.41** | **+0.69** | **−0.35** | **0/9** |
| replay_h6_weighted | 83.22 | +0.50 | −0.35 | 0/9 |

### So what

> replay_safe_uniform は、**平均精度向上と material harm 抑制を同時に満たす現時点の最有力候補**。

---

## Slide 16: 結果の読み取り

**主張**: 今回の価値は「壊さない」だけでなく「選んで伸ばす」方向性が見えたこと

### 置く内容

- replay_safe_uniform は source_only から **+0.69pp** 改善した
- 同時に worst Δ は −0.35pp に収まり、HSC=0/9 を維持した

> これは、単に過保守に no_update しているわけではない。
> replay によって、改善が見込める候補更新を選べている可能性を示す。

### 図案

```
source_only 82.72
        ↓ +0.69pp
replay_safe_uniform 83.41

HSC = 0/9
worst Δ = −0.35pp
```

---

## Slide 17: replay は何をしているか

**主張**: replay は危険な候補を reject し、別の候補に迂回させている

### 置く内容（candidate trace の代表例）

```
shallow_var_update → reject
hybrid_BN_update   → reject
deep_BN_update     → reject
prototype_update   → reject
logit_bias_update  → commit
```

### 図案

```
candidate ごとの sim_score 横棒
        ↓
       0 線
negative side = reject
positive side = commit
```

### 下部メッセージ

> 「**どの候補を止め、どの候補を採用したか**」が見える点が重要。

---

## Slide 18: 現時点で言えること

**主張**: Replay-SafeCommit は、選択型 OTTA の方向性として有望である

### 置く内容

- source_only に対して **+0.69pp** 改善
- **HSC = 0/9** を維持
- fixed update ではなく candidate selection として OTTA を再設計できる
- replay buffer は subject / session 状態を反映する **proxy 評価関数** として使える可能性がある

### 強い一文

> 今回の貢献は「特定の更新ルール」ではなく、「**更新候補を安全に選ぶ枠組み**」である。

---

## Slide 19: 現時点で言えないこと

**主張**: 一方で、最終主張には追加検証が必要である

### 置く内容

- 5 seed 全体で確定したとは言えない
- hybrid@0.01 に完全に勝ったとは言えない
- hard subject を明確に救えたとは言えない
- BCIC IV-2b / BNCI2014001 で成立するとは言えない
- online BCI の遅延制約でそのまま使えるとは言えない

### 下部メッセージ

> 次の実験では、**再現性・fair comparison・他データセット検証** が必要。

---

# 5. 論文調査: BTTA-DG

## Slide 20: 残課題

**主張**: 安全に選ぶ枠組みは見えたが、平均精度をさらに押し上げる候補更新が必要

### 置く内容

> replay_safe_uniform は +0.69pp 改善した。
> これは有望だが、平均精度の押し上げ幅としてはまだ限定的である。
>
> 今後は、Replay-SafeCommit の枠組みを保ちながら、
> **より強い補正器を candidate operator として追加** する必要がある。

### つなぎ

> その参考になるのが BTTA-DG。

---

## Slide 21: BTTA-DG の要点

**主張**: BTTA-DG は Bayesian calibration により EEG-MI TTA の平均精度を押し上げる

### 置く内容

- Dirichlet feature projection
- GMM-driven inference
- gradient-free TTA
- real-time を意識した設計

### 表

| 観点 | BTTA-DG |
|---|---|
| 目的 | 平均精度向上 |
| 補正対象 | deep feature distribution |
| 更新形式 | gradient-free |
| 強み | calibration が強い |
| 課題 | subject-level harm の制御は主眼ではない |

---

## Slide 22: 自分の研究との差分

**主張**: BTTA-DG は「強い補正器」、本研究は「安全な選択器」として位置づけられる

### 比較表

| 観点 | BTTA-DG | 本研究 |
|---|---|---|
| 主眼 | 平均精度向上 | 平均精度 + harm 抑制 |
| 中核 | Bayesian calibration | Replay-SafeCommit |
| 見ているもの | feature density | candidate update の効果 |
| 強み | 補正器が強い | 更新を棄却できる |
| 弱み | harm 機構の説明は薄い | 補正器はまだ弱い |

### So what

> 両者は競合というより、**組み合わせる余地** がある。

---

# 6. 今後の方針

## Slide 23: 今後のモデル案

**主張**: BTTA-DG の Bayesian calibration を、Replay-SafeCommit の candidate operator として追加する

### 置く内容

> BTTA-DG を丸ごと移植するのではなく、中核である **Bayesian feature-density calibration を operator 化** する。

### 追加候補

```
- bayesian_density_fusion
- bayesian_gaussian_update
- dirichlet_uncertainty_gate
```

### 図案

```
Candidate operators
   prototype_update
   logit_bias_update
   deep_BN_update
   Bayesian density fusion   ← new
   Bayesian Gaussian update  ← new
        ↓
   Replay-SafeCommit
        ↓
   commit / rollback
```

---

## Slide 24: 仮説

**主張**: Bayesian operator により平均精度を押し上げ、replay により harm を抑える

### 置く内容

```
H1:
cross-session drift は deep feature 空間の class distribution shift として現れる。

H2:
prototype 1 点では class 内分散や不確実性を表現しきれない。

H3:
Bayesian calibration は平均精度を上げる可能性がある。

H4:
ただし pseudo-label 汚染により harm を起こすため、
replay による commit 判定が必要。
```

### 下部メッセージ

> **攻めの補正器を、安全な選択器で制御する。**

---

## Slide 25: 次の実験計画

**主張**: いきなり統合せず、Bayesian 補正器の単体効果から検証する

### 表

| Step | 実験 | 目的 |
|---|---|---|
| 1 | bayesian_density_fusion only | 更新なし補正が効くか |
| 2 | bayesian_gaussian_update | target distribution 更新が効くか |
| 3 | replay あり/なし比較 | harm を抑えられるか |
| 4 | 既存 operator 群へ統合 | 選択型 OTTA として完成 |
| 5 | 5 seed × 9 subjects | 再現性確認 |
| 6 | BNCI2014001 / BCIC IV-2b | 外部比較 |

---

## Slide 26: 採択基準とまとめ

**主張**: 採択基準は「安全制約を満たした上で mean Δ 最大」

### 採択基準

```
1. HSC@0.5pp = 0/9
2. WSD ≥ −0.5pp
3. mean Δ が replay_safe_uniform 以上
4. abstain / commit rate が過剰でない
```

### まとめ

- EEG-MI cross-session OTTA では、**平均精度だけでなく subject-level harm の抑制** が重要。
- 前回 hybrid は固定ルールで harm 抑制の方向性を確認した。
- 今回 Replay-SafeCommit により、**候補更新を replay buffer 上で検証し、commit / rollback する枠組み** を検討した。
- source_only に対して **+0.69pp、HSC = 0/9** を確認した。
- 次は BTTA-DG の **Bayesian calibration を candidate operator として追加** し、平均精度のさらなる向上を狙う。

---

## 関連 artifact

### 図 (`docs/research_progress/figures/260511_presentation/`)
- `fig_main_source_vs_replay.png` (Slide 15 用)
- `fig_per_subject_delta.png` (Slide 16 用)
- `fig_safety_tradeoff.png` (Slide 18 用)
- `fig_s7_seed_stability.png` (Slide 18 〜 19 補足)
- `fig_replay_candidate_trace.png` (Slide 17 用)
- `fig_method_flow_replay_safecommit.png` (Slide 9 〜 11 用)
- `caption_draft.md` (図キャプション)

### 表 (`docs/research_progress/tables/260511_presentation/`)
- `main_result_table.csv` (Slide 15 用)
- `per_subject_delta.csv` (Slide 16 用)
- `s7_seed_table.csv` (補足用)
- `replay_candidate_trace.csv` (Slide 17 用)

### 関連ドキュメント
- 前回ゼミ資料: `docs/research_progress/ゼミ資料/260420_narukawa.pdf`
- モデル全体図: `docs/research_progress/260508_models_overview.md`
- 詳細ノート: `docs/research_progress/260511_seminar_progress.md`

### 参考論文
- BTTA-DG: Luo, Lu et al. *Bayesian Test-Time Adaptation via Dirichlet Feature Projection and GMM-Driven Inference for Motor Imagery EEG Decoding*, ICLR 2026
