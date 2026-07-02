# Stieger task-aware prospective EA audit — 2026-06-22

## 結論

**旧解析のLR/2D-horizontal混合は、EA harmの主因ではなかった。**

62人・598 sessionsを、raw MATの`tasknumber`に基づきnative taskへ分離して再評価した。mixed-horizontalとpure-LRの差は、utility、R10、severe-harm率の全てで実質ゼロだった。一方、pure-LRだけでも下側リスクと被験者間異質性は残った。

したがって、

- 「旧結果はtask混合artifactだった」は棄却する。
- 旧mixed full-session EAの結果は再現された。
- 新規性の候補は **control context混合** ではなく、**native taskと無ラベルcalibration budgetによって適応のdownside riskが変わること**へ修正する。

---

## データ修正と検証

Stieger各sessionは次の3 native tasksからなる。

| tasknumber | native task | classes |
|---:|---|---|
| 1 | LR | right / left |
| 2 | UD | both hands / rest |
| 3 | 2D | right / left / both / rest |

旧`LeftRightImagery` cacheはtask番号を保持せず、task 1のLRとtask 3のhorizontal trialsを結合していた。

今回、raw MATからtask-aware covariance cacheを作成した。

- 62 subjects
- 598 sessions（41人×11 sessions、21人×7 sessions）
- 8–30 Hz continuous IIR filter
- epoch 0–3 s
- 250 Hz
- 旧解析と同じordered 60 channels
- `task / target / run / trial / session`を保存
- cache size 2.4 GB

前処理一致検証：

- S1/session 1の旧horizontal cache：205 trials
- 新task-aware cache内horizontal：205 trials
- trial順・label順：完全一致
- covariance relative mean error：`2.18×10^-8`

したがって、新旧差はtask定義だけであり、filterやepoch処理の変更ではない。

---

## 評価

- source：session 1でtaskごとにRiemann tangent-space LDAを学習
- target source：session-1 referenceを使用
- prospective EA：target sessionのtask-specific先頭32 trialsだけからreferenceを推定
- evaluation：trial 33以降
- full-session EA：target session全trialからreferenceを推定するtransductive参考上限
- subject-balanced utility `U`
- lower-tail `R10 = -LCVaR10`
- `P(Δ < -5pp)`
- Δのone-way random-effect ICCとsubject bootstrap CI

主解析はnative tasks：

- pure LR：task 1、2-class
- pure UD：task 2、2-class
- 2D：task 3、4-class

mixed-horizontal/verticalと2D内binary projectionは診断解析に限定する。

---

## native task結果

### 全remaining trialsで評価

| task | adapter | U | R10 | P(Δ<−5pp) | ICC |
|---|---|---:|---:|---:|---:|
| LR | prefix-32 EA | +6.30 | 6.91 | 6.18% | 0.294 [0.186, 0.386] |
| LR | full EA | +7.68 | 5.51 | 3.87% | 0.400 [0.288, 0.489] |
| UD | prefix-32 EA | +5.40 | 9.89 | 11.34% | 0.318 [0.193, 0.419] |
| UD | full EA | +5.95 | 9.66 | 9.25% | 0.345 [0.222, 0.449] |
| 2D | prefix-32 EA | +4.72 | 7.95 | 8.49% | 0.264 [0.156, 0.370] |
| 2D | full EA | +5.50 | 6.22 | 6.02% | 0.319 [0.203, 0.419] |

旧mixed-horizontal full EA：

- U = +7.73pp
- ICC = 0.468、95%CI [0.352, 0.561]

これは旧解析の約+7.8pp・ICC=0.468を再現する。

### evaluation試行数を32に統一した感度解析

| task | U(prefix-32) | R10 | P(Δ<−5pp) |
|---|---:|---:|---:|
| LR | +8.06 | 11.21 | 11.08% |
| UD | +6.36 | 14.25 | 16.45% |
| 2D | +5.29 | 11.29 | 16.61% |

evaluationを32 trialsへ短縮すると全taskでtail riskが増える。これは「適応が急に危険になる」のではなく、session Δの測定誤差が増えるためである。したがってmain risk estimateは全remaining trialsを使用し、equal-32はtask比較の感度解析に限定する。

---

## 仮説検定

### 1. mixed contextが害の原因か

prefix-32 EA、mixed-horizontal − pure-LR：

| 指標 | 差 | subject-bootstrap 95%CI |
|---|---:|---:|
| U | −0.036pp | [−0.795, +0.728] |
| R10 | +0.120pp | [−1.795, +2.208] |
| P(Δ<−5pp) | −0.11 percentage points | [−2.90, +2.69] |

equal-32評価でもU差−0.291、R10差−0.252で、両CIは0を跨いだ。

**判定：mixed context主因説を棄却。** task 1とtask 3を混ぜたことはデータ定義として不透明だったが、旧harm現象を作った原因ではない。

### 2. native taskでriskが変わるか

UD − LR、全remaining trials：

| 指標 | 差 | subject-bootstrap 95%CI |
|---|---:|---:|
| U | −0.90pp | [−2.64, +0.83] |
| R10 | **+2.98pp** | **[+0.67, +5.43]** |
| P(Δ<−5pp) | **+5.16 points** | **[+1.18, +9.78]** |

equal-32ではR10差のCIは0を跨ぐが、P(Δ<−5pp)差は+5.38 points、95%CI [+0.64, +10.49]で残る。

**判定：adaptation safetyはdataset全体の一つの値ではなく、native control taskに依存する。**

### 3. prefix-32はfull-session referenceより悪いか

prefix − full：

| task | ΔU [95%CI] | ΔR10 [95%CI] |
|---|---:|---:|
| LR | **−1.38** [−1.78, −1.00] | **+1.40** [+0.43, +2.43] |
| UD | **−0.55** [−0.99, −0.11] | +0.23 [−1.37, +1.64] |
| 2D | **−0.78** [−1.18, −0.40] | **+1.73** [+0.58, +2.92] |

**判定：prospective EAの主要な弱点は、hidden contextより有限な無ラベルreference推定budgetである。**

---

## 研究方向の修正

### 撤回

- 「LR/2D context混合がnegative transferを作った」
- context-conditioned EAを直ちに新手法として作ること
- 旧risk audit全体をloader artifactとして破棄すること

### 守る

- 平均利得は大きいがdownside tailが残る
- harm/Δには再現可能な被験者間異質性がある
- transductive full-session評価はprospective deploymentを楽観評価する
- riskはtaskとcalibration information量の双方に依存する

### 次に行う唯一の実験

**Native-task calibration-budget frontier**を測る。

- tasks：LR / UD / 2D
- prefix `m ∈ {8, 16, 32, 64, full}`
- main evaluation：prefix後の全remaining trials
- sensitivity：各条件32 evaluation trials
- 指標：U、R10、P(Δ<−5pp)、ICC
- 階層モデル：
  `Δ ~ log(m) × task + sourceAcc + nEval + (1|subject)`

同時に、analysis-only oracleとしてprefix referenceとfull-session referenceのAIRM距離、およびprefix bootstrap instabilityを計算する。これらがharmとbudget効果を説明するかを検証する。

構成的手法へ進む条件：

1. riskがbudget増加で系統的に減る。
2. reference estimation errorがharmを予測する。
3. task差の一部がreference errorで媒介される。

3条件を満たした場合だけ、reference uncertaintyに応じたsource-shrinkageをnested subject splitで作る。標準scalar shrinkage、OAS/Ledoit-Wolf、full EAを必須baselineとする。満たさなければ新手法は作らず、task-aware prospective risk auditとして終了する。

---

## 実装

- `stieger_task_cov_cache.py`
- `stieger_task_aware_riemann_audit.py`
- `stieger_task_context_contrasts.py`
- `stieger_attach_task_metadata.py`
- `stieger_task_context_audit.py`

結果：

- `260622_stieger_task_aware_riemann/summary.json`
- `260622_stieger_task_aware_riemann/contrasts.json`
- `260622_stieger_task_aware_equal32/summary.json`
- `260622_stieger_task_aware_equal32/contrasts.json`

