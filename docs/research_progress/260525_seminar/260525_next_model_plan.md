# 260525 ゼミ資料案: DC-Replay の整理と次仮説

## 方針修正

0525 ゼミでは、**Commitless Memory-Corrected OTTA (CMC-OTTA) を提案手法として前面に出さない**。

理由は、CMC-OTTA はまだ単独モデルとして実装・評価しておらず、現時点では DC-Replay の追加検証から出てきた次仮説だからである。

したがって、0525 の主役は以下にする。

> Replay-SafeCommit 後に実行した DC-Replay 検証を整理し、L1/L2/L3 のどこが効いたのか、どこが安全性を壊したのかを説明する。

CMC-OTTA は最後に「DC-Replay の結果から切り出せそうな次候補」として扱う。

---

## 0525での一行結論

DC-Replay は、Replay-SafeCommit の後に「prediction correction」「external memory」「model-state commit」を分けて検証するためのモデルだった。

結果として、平均精度を伸ばすシグナルは出たが、seed 安定性と HSC は悪化した。さらに L3 診断では、best gain が model-state commit 回数では説明できなかった。

したがって、0525 では **DC-Replay は成功した完成モデルではなく、L1/L2 が効いていそうだと分かった責務分解の検証**として説明する。

## 一行結論

次に説明するモデルは、`Replay-SafeCommit` の単純な拡張ではなく、まず **DC-Replay** として整理する。

Replay-SafeCommit は「model-state update を安全に採用する」手法だった。  
しかし、定期ゼミ前の追加検証では、最も強い精度改善は L3 model-state commit ではなく、L1 prediction correction と L2 external memory から出ていた。

したがって、次モデルの中心を CMC-OTTA と断定する前に、DC-Replay で何をしたかを理解する。

> DC-Replay は、model-state commit を主役にしてよいのか、それとも L1/L2 の非破壊補正が主因なのかを切り分ける検証である。

---

## 1. Replay-SafeCommit から自然に出る未解決点

### Replay-SafeCommit が解いたこと

- OTTA の誤更新は後続 trial に悪影響を残す。
- current trial の margin / SAL だけでは commit の効果を測りにくい。
- replay buffer 上の simulated reward で候補更新を検証することで、危険な commit を reject できた。
- 主結果:
  - `source_only`: 82.72%
  - `replay_safe_uniform`: 83.41%
  - gain: +0.69pp
  - HSC@0.5pp: 0/9

### まだ残った問題

- 安全性は高いが、平均改善は +0.69pp に留まった。
- Replay-SafeCommit は「壊さない」設計として強いが、「積極的に補正する」力はまだ弱い。
- 追加検証では、DC 系が 9被験者評価で最大 +1.23pp まで伸びたが、HSC が増えた。
- L3 commit 診断では、best gain が model-state commit 回数では説明できなかった。

### ここからの問い

> model state を更新しなくても、target session の情報を使って予測だけを安全に補正できないか？

---

## 2. 次モデルの名前と主張

### 推奨名

**Commitless Memory-Corrected OTTA**

短く書くなら:

**CMC-OTTA**

### 名前の意味

- Commitless:
  - BN running stats, prototype, logit bias などの model state を基本的に更新しない。
- Memory-Corrected:
  - target session で得た高信頼 evidence を external memory に蓄積し、現在 trial の posterior を補正する。
- OTTA:
  - target label は使わず、online test stream のみで逐次動作する。

### 主張

Replay-SafeCommit は「model-state commit を安全にする」方向だった。  
CMC-OTTA はその前段として、**model-state commit を必要としない補正を第一選択にする**。

---

## 3. 設計思想

### 基本方針

```
trial x_t
  ↓
TCFormer forward
  ↓
raw posterior p_raw(y|x_t)
  ↓
external memory から memory posterior p_mem(y|x_t) を推定
  ↓
reliability に応じて p_raw と p_mem を混合
  ↓
final prediction
  ↓
高信頼なら memory に admit
```

### 重要な分離

| 状態 | 例 | 破壊性 | CMC-OTTAでの扱い |
|---|---|---:|---|
| model state | BN stats, prototype, logit bias | 高 | 原則更新しない |
| external memory | target evidence, pseudo-label, feature/prob/alpha | 中 | reliability gate 付きで更新 |
| ephemeral state | 現 trial の補正量, posterior | 低 | 毎 trial で使い捨て |

この分離により、誤った trial が来ても model 本体を汚染しにくい。

---

## 4. モデル構成

### L0: Base TCFormer

- 既存の TCFormer checkpoint をそのまま使う。
- training path は変更しない。
- 出力:
  - logits
  - softmax posterior
  - feature vector

### L1: Prediction Correction

現在 trial の予測だけを補正する。

候補式:

```text
p_final = (1 - lambda_t) * p_raw + lambda_t * p_mem
```

ここで:

```text
lambda_t = clip(alpha * reliability_t * drift_need_t, 0, lambda_max)
```

- `p_raw`: TCFormer の元予測
- `p_mem`: external memory から推定した posterior
- `reliability_t`: 現 trial と memory prediction の信頼度
- `drift_need_t`: raw prediction だけでは不安定な度合い
- `lambda_max`: 過補正を防ぐ上限

### L2: External Memory

Memory には、model state ではなく、検証可能・破棄可能な target evidence を入れる。

候補:

- feature vector
- raw posterior
- corrected posterior
- pseudo-label
- confidence
- entropy / SAL
- trial index
- correction strength

Memory admission は hard gate ではなく、score-based policy とする。

候補特徴:

- confidence
- entropy / SAL
- prototype consistency
- temporal consistency
- class balance
- density support
- raw-corrected disagreement
- OOD risk

初期実装では、以下のような conservative policy でよい。

```text
admit if:
  max(p_final) > tau_conf
  entropy(p_final) < tau_ent
  memory/replay support is not low
  class buffer is not over-dominant
```

### L3: Model-State Commit

0525時点では主役にしない。

- Replay-SafeCommit の延長として存在は残す。
- ただし CMC-OTTA の主張では、L3 は optional extension。
- controlled drift 条件で必要性が出た場合だけ検証する。

---

## 5. Replay-SafeCommit との違い

| 観点 | Replay-SafeCommit | CMC-OTTA |
|---|---|---|
| 主目的 | model-state update を安全に commit する | model-state update なしで予測を補正する |
| 使う memory | commit 候補を検証する replay buffer | posterior 補正に使う external memory |
| model state | 候補によって更新される | 原則 freeze |
| harm の経路 | 誤 commit による model contamination | memory contamination による誤補正 |
| safety の考え方 | commit 前に replay-safe かを見る | correction strength と memory admission を制御する |
| 新規性の中心 | replay simulated reward による SafeCommit | commitless memory-corrected prediction |

---

## 6. BTTA-DG との関係

BTTA-DG は、Dirichlet feature projection と GMM-driven Bayesian inference により、gradient-free に posterior を補正する手法である。

CMC-OTTA は BTTA-DG と同じく、model state を直接更新しない補正方向に近い。  
ただし本研究では、Replay-SafeCommit の結果から導いた safety-oriented な設計として位置づける。

| 観点 | BTTA-DG | CMC-OTTA |
|---|---|---|
| 補正方法 | Dirichlet + GMM Bayesian posterior | memory posterior と raw posterior の安全混合 |
| memory | Dirichlet alpha memory | feature/prob/alpha を含む external memory |
| 目的 | gradient-free TTA の高精度化 | EEG-MI OTTA の harm 抑制と補正力の両立 |
| 本研究との関係 | 参考にする補正器 | Replay-SafeCommit 後の次モデル |

---

## 7. 0525で示すべき図

### 図1: Replay-SafeCommit から CMC-OTTA への移行

```text
Replay-SafeCommit
  model-state update を replay で検証
        ↓
追加検証
  best gain は L3 commit では説明できない
        ↓
CMC-OTTA
  model state を動かさず memory で posterior 補正
```

### 図2: CMC-OTTA の処理フロー

```text
EEG trial
  ↓
TCFormer
  ↓
p_raw, feature
  ↓
Memory posterior p_mem
  ↓
Reliability-controlled fusion
  ↓
p_final
  ↓
Memory admission
```

### 図3: 状態のリスク分離

```text
ephemeral correction  <  external memory  <  model state
       low risk              medium             high
```

---

## 8. 実験計画

### ゼミコメントを受けた修正

Replay buffer / memory を使うモデルでは、「安全」という言葉を先に定義し、その定義に対応する指標とログを設計する必要がある。

今回のコメントは以下の4点として反映する。

1. 安全性を明示的に定義する。
2. 定義に基づいて評価指標をブラッシュアップする。
3. replay buffer / external memory の途中値を保存する。
4. 複数データセットで適応性能と安全性を確認する。

### 安全性の定義

本研究では、OTTA の安全性を「平均精度が上がるか」ではなく、**source-only と比べて material harm を起こさないこと**として定義する。

ここで重要なのは、安全性をロバスト性や汎用性の言い換えにしないことである。

### 安全性・ロバスト性・汎用性の違い

| 概念 | 問うこと | 基準 | 本研究での例 |
|---|---|---|---|
| 汎用性 / generalization | 学習していない session / subject / dataset でも性能が出るか | target 全体の平均性能 | source-only TCFormer が別 session でどれだけ当たるか |
| ロバスト性 / robustness | ノイズ・分布ずれ・摂動に対して性能が落ちにくいか | 入力変動に対する性能安定性 | artifact や session drift があっても予測が崩れにくいか |
| 安全性 / safety | 適応や補正という介入が、何もしない場合より追加害を出さないか | source-only に対する downside control | OTTA update / memory correction が一部被験者を悪化させないか |

したがって、本研究での安全性は「強いモデルか」ではなく、**適応 policy が harm を発生させないか**である。

より短く言うと以下になる。

```text
generalization = 未知分布でも当たるか
robustness     = ずれても崩れにくいか
safety         = 適応したせいで悪化させないか
```

この区別により、以下のようなケースを分けて議論できる。

- 汎用性は高いが安全ではない:
  - source-only は高精度だが、aggressive OTTA が一部 subject を壊す。
- ロバストだが安全ではない:
  - ノイズには強いが、誤った memory admission により後続 prediction が悪化する。
- 安全だが補正力は弱い:
  - no-update / abstain が多く harm は少ないが、平均精度はあまり伸びない。

本研究の狙いは、単なる robustness や generalization ではなく、**online adaptation による追加害を制御しながら平均性能を上げること**である。

### 数式的な定義

source-only baseline を \(f_0\)、OTTA policy を \(\pi\)、trial \(t\) の損失を \(\ell\) とする。
このとき、適応による追加害を以下で表す。

\[
  h_t^\pi = \ell(\pi_t(x_t), y_t) - \ell(f_0(x_t), y_t)
\]

\(h_t^\pi > 0\) なら、その trial では「適応したせいで source-only より悪い」ことを意味する。

subject / session \(s\) 全体では、精度差から final harm を定義する。

\[
  H_s^\pi = \mathrm{Acc}_0(s) - \mathrm{Acc}_\pi(s)
\]

安全性の最小条件は以下である。

\[
  H_s^\pi \leq \epsilon
\]

を多くの subject / seed / dataset で満たすこと。
ここで \(\epsilon\) は material harm の閾値であり、現在は 0.5pp と 1.0pp を併記する。

ただし final harm だけでは online 中の一時的崩壊が見えないため、window accuracy に基づく max drawdown や online regret も見る。

より具体的には、以下の3階層で定義する。

| 階層 | 定義 | 評価時に見るもの |
|---|---|---|
| Final safety | session 全体の最終性能が source-only から大きく悪化しない | HSC@0.5pp, HSC@1.0pp, worst delta |
| Online safety | stream の途中で大きな累積悪化や一時的崩壊を起こさない | online regret, max drawdown, recovery time |
| Decision safety | 個々の correction / commit / memory admission が後続性能を悪化させない | harmful correction rate, harmful commit rate, memory reliability proxy |

最小の主張としては、まず以下を安全条件にする。

```text
safe if:
  mean_delta > 0
  HSC@0.5pp is low
  worst_delta is not worse than replay_safe baseline
  harmful correction / admission が診断ログで説明可能
```

論文向けには、HSC@0.5pp だけでなく、seed variance を考慮した HSC@1.0pp または confidence interval 付きの harmful count も併記する。

### 指標のブラッシュアップ

安全性の定義に対応して、指標を4群に分ける。

| 指標群 | 指標 | 目的 |
|---|---|---|
| Accuracy | mean accuracy, delta vs source, per-subject delta | 平均性能と被験者別改善を見る |
| Final harm | HSC@0.5pp, HSC@1.0pp, worst delta | 最終的に誰かを壊していないかを見る |
| Online harm | online regret, max drawdown, recovery time | stream 途中の一時的な崩壊を見る |
| Decision / memory | correction strength, harmful correction rate, memory reliability proxy, class entropy | なぜ安全/危険だったかを説明する |

ここで `harmful correction` は、offline 解析時に正解ラベルを使って以下のように定義する。

```text
raw_pred is correct and corrected_pred is wrong
```

逆に `helpful correction` は以下。

```text
raw_pred is wrong and corrected_pred is correct
```

この2つを見ることで、単に accuracy が上がった/下がっただけでなく、補正器がどの trial を救い、どの trial を壊したかを説明できる。

### replay buffer / memory の途中値を保存する

次の実装では、最終 accuracy だけでなく、各 trial で replay buffer / external memory がどの状態だったかを保存する。

最低限保存する列:

| category | columns |
|---|---|
| raw prediction | `trial_idx`, `y_true`, `raw_pred`, `raw_pmax`, `raw_entropy`, `raw_margin` |
| corrected prediction | `corrected_pred`, `corrected_pmax`, `corrected_entropy`, `correction_strength`, `lambda_t` |
| memory posterior | `p_mem_top1`, `p_mem_pmax`, `memory_support`, `knn_agreement`, `prototype_cos` |
| admission | `memory_admitted`, `admission_score`, `admission_reason`, `memory_size`, `memory_class_entropy` |
| replay evaluation | `replay_acc_before`, `replay_acc_after`, `replay_margin_before`, `replay_margin_after`, `sim_score` |
| safety labels offline | `helpful_correction`, `harmful_correction`, `raw_corrected_disagreement` |

特に replay buffer 型モデルでは、以下を時系列で保存する。

- buffer size
- class counts
- class entropy
- pseudo-label stability
- raw/corrected/replay prediction agreement
- admitted sample の confidence / entropy / support
- replay sim_score の分布

これにより、S4/S6 のような悪化 subject で「何が memory に入ったせいで崩れたのか」を追える。

### 複数データセットでの検証

BCIC2a だけで安全性を主張すると、データセット依存の結果に見える。
したがって、段階的に複数データセットへ広げる。

| 優先度 | dataset | 目的 |
|---|---|---|
| 1 | BCIC2a | 既存結果との連続性。4-class / 9 subjects の主検証 |
| 2 | BCIC2b | 2-class でも memory correction が効くか。計算・解釈が軽い |
| 3 | HGD | より多チャンネル・高密度な EEG で頑健性を見る |
| 4 | REH-MI | 実験条件が違う MI データで外的妥当性を見る |

最初から全 dataset で大規模 sweep はしない。
まずは各 dataset で source-only と replay-safe baseline を揃え、その上で CMC minimal を同一指標で評価する。

### 最小比較

| variant | 目的 |
|---|---|
| source_only | 適応なし baseline |
| replay_safe_uniform | 安全な model-state commit baseline |
| correction_static_only | source memory / fixed stats だけで補正できるか |
| CMC-OTTA memory_only | online memory による補正効果 |
| CMC-OTTA conservative | safety 重視の admission / lambda 上限 |
| CMC-OTTA aggressive | 補正力上限を見る |

### 必須指標

- mean accuracy
- delta vs source
- worst delta
- HSC@0.5pp
- correction rate
- correction strength
- harmful correction rate
- memory size
- memory class entropy
- raw-corrected disagreement rate
- corrected-only success / failure

### まず見るべき失敗条件

- S4/S6 で過補正が起きるか
- memory が一部 class に偏るか
- raw と corrected が disagree したときに悪化するか
- confidence が高い誤補正を memory に admit していないか
- correction strength が大きい trial ほど harm が増えていないか

---

## 9. 0525での話し方

### 悪い言い方

> Replay-SafeCommit の次は、さらに候補を増やします。

これは教授コメントに対して弱い。計算量・責務分離の話に戻ってしまう。

### 良い言い方

> Replay-SafeCommit は model-state update を安全にする手法として有効だった。一方で、追加検証から、精度改善の主因は model-state commit ではなく、prediction correction と external memory である可能性が高い。そこで次は、model state を原則更新しない Commitless Memory-Corrected OTTA として、非破壊な補正を主役にする。

### 一番大事なメッセージ

> 次モデルは「より強い更新を足す」のではなく、「更新しないで済む補正を先に行う」モデルである。

---

## 10. 0525スライド構成案

1. 前回までの結論: Replay-SafeCommit は安全だが補正力に余地
2. 追加検証から分かったこと: best gain は L3 commit では説明できない
3. 次の研究仮説: model state を動かさない補正を第一選択にする
4. 提案: Commitless Memory-Corrected OTTA
5. 処理フロー: raw posterior, memory posterior, reliability fusion
6. Memory admission: 何を入れるか、何を入れないか
7. Replay-SafeCommit / BTTA-DG との違い
8. 実験計画: source, replay_safe, static correction, CMC variants
9. 評価指標: accuracy, HSC, memory reliability, harmful correction
10. 失敗条件: S4/S6, class imbalance, disagreement
11. 今週やること

---

## 11. 今週の実装方針

最初から複雑にしない。

### Step 1: CMC minimal

- TCFormer は freeze
- memory は class-balanced FIFO
- `p_mem` は feature kNN or class prototype similarity で作る
- `p_final = (1-lambda)p_raw + lambda p_mem`
- lambda は固定小さめ、または confidence-based

### Step 2: Diagnostics

- correction_strength
- raw_pred
- corrected_pred
- raw_corrected_disagreement
- memory_class_entropy
- memory_admitted
- memory_reliability_proxy

### Step 3: Sweep

- lambda: 0.05, 0.10, 0.20, 0.30
- memory capacity: 16, 32, 64
- admission threshold: conservative / default / aggressive

### Step 4: Decision

- HSC=0/9 で Replay-SafeCommit を超えるか
- seed stability で HSC が増えないか
- S4/S6 の悪化理由が diagnostics で説明できるか

---

## 12. 現時点の結論

0525で出すべき次モデルは、`Deferred-Commit Replay OTTA` ではなく、**Commitless Memory-Corrected OTTA**。

理由:

1. 追加検証で、L3 commit が改善の主因ではないことが示唆された。
2. no-commit / zero-commit variant でも gain が出た。
3. 教授コメントの「責務を分ける」「候補を根拠を持って選ぶ」に答えやすい。
4. BTTA-DG の gradient-free posterior correction とも自然につながる。
5. Replay-SafeCommit を否定せず、安全 baseline として残せる。

したがって、研究の流れは以下で整理する。

```text
Replay-SafeCommit
  = model-state update を安全にする

Commitless Memory-Corrected OTTA
  = model-state update を必要としない補正を第一選択にする

L3 deferred commit
  = 持続 drift が明確な場合の optional extension
```
