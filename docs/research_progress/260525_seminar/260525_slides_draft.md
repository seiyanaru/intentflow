# 260525 ゼミスライド構成案

> 主題: Replay-SafeCommit の課題から DC-Replay を試し、その結果から次に CMC を検討する

---

# 全体の流れ

```text
背景・目的
  ↓
前回の振り返り
  Replay-SafeCommit は安全だが、補正力に課題
  ↓
提案
  DC-Replay: L1/L2/L3 に責務を分けて検証
  ↓
実験結果
  平均精度は伸びるが、HSC と seed 安定性に課題
  L3 commit は改善の主因ではなさそう
  ↓
今後の方針
  safety 定義と途中ログを整える
  L1/L2 を切り出した CMC を次に試す
```

---

## Slide 1. タイトル

**Replay-SafeCommit 後の追加検証: DC-Replay の分解と安全性評価**

副題:

**model-state commit は本当に改善の主因だったのか**

### 一言

> 前回の Replay-SafeCommit の課題を受けて、今回は DC-Replay により correction / memory / commit の責務を分けて検証した。

---

## Slide 2. 今日の構成

1. 背景・目的
2. 前回の振り返り: Replay-SafeCommit と残った課題
3. 提案: DC-Replay
4. 実験結果と分析
5. 今後の方針: safety 定義と CMC

### 下部メッセージ

> 今日は CMC を結果付きの提案として出すのではなく、DC-Replay の分析から次に CMC を試す理由を説明する。

---

# 1. 背景・目的

## Slide 3. 背景: EEG-MI と cross-session drift

**主張**: EEG-MI では、同じ被験者でも session が変わると分布がずれる。

### 置く内容

- EEG-MI は、運動想起中の EEG から意図を分類する Active BCI タスク。
- 同じ被験者でも、日・疲労・集中度・電極状態・ノイズにより EEG 分布が変化する。
- 学習 session のモデルを評価 session にそのまま使うと、性能が低下しやすい。

### 下部メッセージ

> 再キャリブレーションを増やさず、target session に逐次追従したい。

---

## Slide 4. 問題: OTTA は有望だが誤更新で壊れる

**主張**: OTTA は drift に追従できるが、正解ラベルなしで更新するため harm が起きる。

### 置く内容

```text
高信頼だが誤分類
  ↓
誤った pseudo-label / artifact で更新
  ↓
model state が汚染
  ↓
後続 trial も悪化
```

### 評価軸

| 指標 | 意味 |
|---|---|
| mean accuracy | 全体性能 |
| delta vs source | 適応なしからの改善 |
| worst delta | 最悪悪化幅 |
| HSC@0.5pp | 0.5pp以上悪化した subject / unit 数 |

### 下部メッセージ

> 目標は、平均精度を上げつつ、一部被験者を大きく壊さない OTTA を作ること。

---

# 2. 前回の振り返り: Replay-SafeCommit と残った課題

## Slide 5. 前回: Replay-SafeCommit の考え方

**主張**: Replay-SafeCommit は、候補更新を replay buffer 上で検証してから採用する。

### なぜ必要だったか

- forward-only OTTA では、current trial の即時 margin / SAL 比較だけでは commit の効果を測りにくい。
- BN / prototype / logit bias の更新効果は、主に将来の trial に出る。
- そこで、過去の高信頼 trial を replay buffer として保持し、候補更新の前後を simulated reward で比較する。

### 処理の核

```text
candidate update
  ↓ temporary apply
replay buffer で再評価
  ↓
sim_score > 0 の候補だけ commit
```

---

## Slide 6. 前回結果: 安全な基準点は作れた

**主張**: Replay-SafeCommit は、平均精度を改善しつつ HSC=0/9 を達成した。

| method | mean | delta | worst delta | HSC |
|---|---:|---:|---:|---:|
| source_only | 82.72 | +0.00 | +0.00 | 0/9 |
| policy_safe_no_shallow | 83.14 | +0.42 | -0.35 | 0/9 |
| replay_safe_uniform | 83.41 | +0.69 | -0.35 | 0/9 |

### 言えること

- 「適応したのに大きく壊していない」は示せた。
- Replay-SafeCommit は安全な model-state commit の基準点になった。

### 残った課題

> 安全性は高いが、補正力はまだ弱い。

---

## Slide 7. 今回の問い: 補正力をどう上げるか

**主張**: Replay-SafeCommit の次は、単に候補を増やすのではなく、適応の責務を分ける必要がある。

### 教授コメントを踏まえた論点

- 候補を増やすと計算量や遅延が増える。
- 候補を根拠を持って選ぶ必要がある。
- すべての入力に対して model-state update を検討する必要があるのか。
- 安全性の定義を明確にし、指標をブラッシュアップする必要がある。

### 今回の検証の問い

> model-state update をいきなり検討するのではなく、prediction correction / memory / commit に分けると、何が効くのか？

---

# 3. 提案: DC-Replay

## Slide 8. 提案: DC-Replay の設計思想

**主張**: DC-Replay は、OTTA を L1/L2/L3 の3階層に分けて検証するモデル。

| 階層 | 役割 | 状態 | リスク |
|---|---|---|---:|
| L1 Prediction Correction | 現在 trial の予測を補正 | ephemeral | 低 |
| L2 External Memory | target evidence を蓄積 | memory | 中 |
| L3 Model-State Commit | BN/prototype/logit bias を更新 | model state | 高 |

### 下部メッセージ

> 一時的な予測不安定は L1、信頼できる target 情報は L2、持続 drift だけ L3 で扱う。

---

## Slide 9. DC-Replay の処理フロー

**主張**: DC-Replay は、各 trial で補正・memory 更新・commit 判定を順に行う。

### 図

`fig_dc_replay_flow.png` を配置する。

### 説明

```text
EEG trial
  ↓
TCFormer forward
  ↓
L1: prediction correction
  ↓
L2: external memory admission
  ↓
drift / commit 判定
  ↓
L3: replay-safe model-state commit
```

### 下部メッセージ

> DC-Replay の目的は、L3 commit を強くすることではなく、どの責務が効いているかを切り分けること。

---

## Slide 10. 比較した variant

**主張**: L1/L2/L3 の寄与を分けるため、複数の variant を比較した。

| variant | L1 correction | L2 memory | L3 commit | 目的 |
|---|---|---|---|---|
| source_only | なし | なし | なし | baseline |
| replay_safe_uniform | なし | replay buffer | replay-safe | 安全基準 |
| correction_static_only | あり | なし | なし | static 補正だけの効果 |
| correction_memory_no_commit | あり | あり | なし | L1+L2 の効果 |
| random_sparse_commit | あり | あり | random | commit 回数だけの効果 |
| commit_no_replay_gate | あり | あり | あり | replay gate なしの比較 |
| replay_gated | あり | あり | replay-gated | L3 本命 |

---

## Slide 11. 評価条件

**主張**: 平均精度だけでなく、seed を変えて安全性が崩れないかも見る。

| 評価 | 対象 | 目的 |
|---|---|---|
| 9被験者評価 | 9 subjects, seed0 | 横断的な平均精度シグナルを見る |
| seed 安定性評価 | S2/S4/S6/S7 × seeds 1--3 | 結果が seed に依存しないか、harm が増えないかを見る |

### 注意

> 評価名は実験条件であり、モデル名ではない。

---

# 4. 実験結果と分析

## Slide 12. 結果1: 9被験者評価

**主張**: DC 系は平均精度を伸ばすシグナルを出したが、HSC が増えた。

### 図

`fig_dc_replay_results_summary.png` の上段左を配置する。

| method | mean | delta | worst delta | HSC |
|---|---:|---:|---:|---:|
| source_only | 82.72 | +0.00 | +0.00 | 0/9 |
| replay_safe_uniform | 83.41 | +0.69 | -0.35 | 0/9 |
| correction_memory_no_commit | 83.57 | +0.85 | -1.04 | 1/9 |
| DC high mean | 83.95 | +1.23 | -1.38 | 1/9 |

### 読み取り

- DC 系は最大 +1.23pp まで伸びた。
- ただし HSC=1/9 になり、安全制約 HSC=0 では Replay-SafeCommit が堅い。

---

## Slide 13. 結果2: seed 安定性評価

**主張**: seed 安定性込みでは、DC 系は HSC が増えやすい。

### 図

`fig_dc_replay_results_summary.png` の上段右を配置する。

| method | mean | delta | worst delta | HSC |
|---|---:|---:|---:|---:|
| source_only | 75.63 | +0.00 | +0.00 | 0/12 |
| replay_safe_uniform | 76.01 | +0.38 | -0.69 | 1/12 |
| correction_memory_no_commit | 75.90 | +0.26 | -1.04 | 3/12 |
| DC best mean | 76.13 | +0.49 | -1.04 | 4/12 |

### 読み取り

- DC best は平均では少し伸びる。
- しかし HSC=4/12 で seed 安定性が弱い。
- S2/S7 を救う可能性がある一方、S4/S6 で悪化しやすい。

---

## Slide 14. 分析: L3 commit は改善の主因か

**主張**: DC-Replay の best gain は、L3 model-state commit では説明できない。

### 図

`fig_dc_replay_results_summary.png` の下段を配置する。

| 評価 | variant | delta | HSC | model commit |
|---|---|---:|---:|---:|
| 9被験者 | correction_memory_no_commit | +0.85 | 1 | 0 |
| 9被験者 | replay_gated | +0.85 | 1 | 111 |
| 9被験者 | commit_no_replay_gate | +0.85 | 1 | 150 |
| 9被験者 | high mean variant | +1.23 | 1 | 0 |
| seed | correction_memory_no_commit | +0.26 | 3 | 0 |
| seed | replay_gated | +0.26 | 3 | 101 |

### 読み取り

> commit 回数が増えても精度は説明できない。効いているのは L1/L2 の可能性が高い。

---

## Slide 15. DC-Replay から分かったこと

**支持されたこと**

- L1 prediction correction には精度向上シグナルがある。
- L2 external memory は target session 情報の使い方として有望。
- Replay-SafeCommit は安全基準としてまだ強い。

**まだ支持されないこと**

- L3 replay-gated commit が改善の主因である。
- DC-Replay が seed 安定性込みで Replay-SafeCommit を超える。
- aggressive correction をそのまま採用して安全である。

### 下部メッセージ

> DC-Replay は「失敗」ではなく、次に切り出すべき責務を明らかにした検証。

---

# 5. 今後の方針

## Slide 16. 安全性の定義を明確にする

**主張**: safety は「平均精度が上がった」ではなく、source-only と比べて material harm を起こさないこととして定義する。

### ロバスト性・汎用性との違い

| 概念 | 問うこと | 例 |
|---|---|---|
| 汎用性 | 未知 session / dataset でも当たるか | source-only が target で高精度 |
| ロバスト性 | ノイズや drift でも崩れにくいか | artifact があっても予測が安定 |
| 安全性 | 適応したせいで悪化させないか | OTTA update が subject を壊さない |

### 下部メッセージ

> 本研究の safety は、online adaptation policy の downside control として定義する。

---

## Slide 17. 指標とログをブラッシュアップする

**主張**: 最終 accuracy だけでなく、replay buffer / memory の途中値を保存する。

### 保存する値

- raw prediction: `raw_pred`, `raw_pmax`, `raw_entropy`, `raw_margin`
- correction: `corrected_pred`, `corrected_pmax`, `correction_strength`
- memory: `memory_admitted`, `admission_score`, `memory_size`, `memory_class_entropy`
- replay: `replay_acc_before/after`, `replay_margin_before/after`, `sim_score`
- safety labels: `helpful_correction`, `harmful_correction`, `raw_corrected_disagreement`

### 下部メッセージ

> 悪化した subject で、どの trial / memory / commit が harm に繋がったかを追えるようにする。

---

## Slide 18. 次に CMC を試す理由

**主張**: CMC は、DC-Replay の結果から出た次仮説であり、まだ未実行。

### 位置づけ

| 項目 | 状態 |
|---|---|
| Replay-SafeCommit | 実装・評価済み。安全基準 |
| DC-Replay | 実装・評価済み。責務分解の検証 |
| CMC-OTTA | 未実装/未評価。L1/L2 を切り出す次候補 |

### なぜ CMC か

- DC-Replay の改善は L3 commit では説明しにくい。
- no-commit / zero-commit variant でも gain が出た。
- したがって、L1 prediction correction と L2 external memory を独立モデルとして切り出す価値がある。

---

## Slide 19. 今週やること

1. DC-Replay の結果を見直し、variant ごとの意味を整理する。
2. safety 定義を final / online / decision に分ける。
3. Replay buffer / memory の途中値ログを追加する。
4. harmful/helpful correction を offline 解析で出す。
5. S4/S6 の悪化条件をログから追う。
6. BCIC2b への最小移植を確認する。
7. その後、CMC minimal を実装するか判断する。

### 下部メッセージ

> まずは DC-Replay の何が効き、何が危険だったのかを説明可能にする。

---

## Slide 20. まとめ

- 背景目的は、EEG-MI cross-session drift に対して安全な OTTA を作ること。
- 前回の Replay-SafeCommit で +0.69pp, HSC=0/9 の安全な基準点を作った。
- 残った課題は補正力であり、今回は DC-Replay により L1/L2/L3 の責務を分けて検証した。
- DC 系は平均精度を伸ばしたが、HSC と seed 安定性に課題がある。
- L3 commit は改善の主因として支持されず、L1/L2 が効いている可能性が高い。
- 次は safety 定義と途中ログを整えた上で、CMC を試す。
