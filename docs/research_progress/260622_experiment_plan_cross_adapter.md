# 実験計画：cross-adapter subject consistency（修士研究の最終分岐実験）— 2026-06-22

> GPT査読(2往復)を反映した正式な次フェーズ計画。**最終推奨：cross-adapter consistency を修士研究の最終分岐実験にする。これ以外の新しい安全ゲートは、その結果が出るまで実装しない。**
> 背景・1ヶ月の結果は [260622_current_state_for_discussion.md](260622_current_state_for_discussion.md)。

> **最終実行更新（2026-06-22）**：16人engineering pilotは実装面を通過したが、source accuracy median 54.70%（基準60%）とprefix-Tentの実質no-opで停止。続くpopulation-pretrain source repairは平均60.10%まで改善したがmedian 59.41%で基準未達。**本計画は修士期間では終了し、62人本実験へ進まない。** 詳細は [260622_cross_adapter_pilot_result.md](260622_cross_adapter_pilot_result.md)。
>
> 後続のtask-aware監査では、旧LR/2D-horizontal混合はEA harmの主因でないことを確認した。cross-adapter計画は再開せず、native-task calibration-budget frontierへ移る。詳細は [260622_task_aware_stieger_audit.md](260622_task_aware_stieger_audit.md)。

---

## 0. この実験が答える唯一の問い
**「適応の害が被験者に偏る(ICC=0.47)のは、人間のtraitか、adapter固有の相性(person×adapter interaction)か？」**
- trait成立 → 危険被験者を事前screeningする方向(SLAH再昇格)。
- adapter固有 → 普遍的な危険ユーザーは存在しない。害はperson×adapter interaction。
- どちらでも修士論文の有効な結論になる。投稿論文としての強さは、外部再現または機構説明を追加できるかで決まる。

---

## 1. 設計：4セル（+各source-only）
| セル | backbone | adapter |
|---|---|---|
| 1 | Riemann-tangent-LDA | EA（既存） |
| 2 | EEGNet | EA |
| 3 | EEGNet | AdaBN |
| 4 | EEGNet | 第3adapter（source repair後にviability screenで固定） |
| +各backbone source-only |  |  |
- 「同じEAでbackbone変える(1 vs 2)」「同じEEGNetでadapter変える(2,3,4)」を分離できる。
- 当初候補のprefix-Tentは、事前gridで有効な更新量と再現性を両立しなかったため本解析から除外する。第3adapterの第一候補はconfidence-filtered pseudo-label head adaptation。
- ただし4セルは完全交差ではないため、**全セルを一つのモデルに入れてsubject×adapterとsubject×backboneの分散を同時推定しない**。
  - **主解析（未実行）**：EEGNet内のEA/AdaBN/A3でsubject×adapterを推定。
  - **副解析**：EA固定のRiemann-LDA/EEGNetでcross-backbone一致を評価。
- **TCFormerは初回比較から除外**（BN構造が異なり不公平）。当初はjournal拡張でShallowFBCSPNetを足す予定だったが、engineering gate不合格により未実行。

## 2. 公平プロトコル（全条件統一）
- source学習：session 1のみ。前処理：8–30Hz / 250Hz / 同一epoch。
- deep source：session 1の共分散P1で整列して学習。
- target adaptation prefix：各session先頭**32試行・ラベルなし**。評価：**33試行目以降**。
- target sessionごとにcheckpointへreset。**future target試行は使わない**（prospective）。
- source repair通過後の本実験ではEEGNetを**3 seeds**、adapter hyperparameterを**5-fold subject-level nested CV**で評価する。
- 同一EEGNet checkpointから分岐：
  - source：P1でtarget変換 / EA：prefixから計算したPjで変換 / AdaBN：P1入力・BN統計のみ更新 / 第3adapter：P1入力・同一source checkpointから分岐。
- 初期探索値：AdaBN mixing λ∈{0.25,0.5,1.0}。第3adapter候補のpseudo-label head adaptationはconfidence≥0.8、最低8 prefix samples、head-only 5 steps、lr∈{1e-4,1e-3}。
- **adapter設定は training subjects で平均利得最大化により選択**（安全指標でチューニングするとrisk consistencyを人工的に作るので禁止）。

### 2.1 実装仕様（事前固定）
- **sourceの単位**：両backboneとも被験者別。各被験者のsession 1だけで学習し、後続sessionを評価する。
- **EEGNetの学習epoch/optimizer**：outer training subjects内のinner CVで共通設定を選び、held-out subjectではその設定を固定してsession 1全体で学習する。held-out subjectのtarget sessionラベルをearly stoppingやmodel selectionに使わない。
- **outer split**：5-fold subject splitを全cell・seedで共有。各outer fold内でadapter hyperparameterをinner subject CVにより選ぶ。
- **checkpoint共有**：同一subject・seedのEEGNet source checkpointからsource/EA/AdaBN/A3を分岐する。分岐前checkpointのhashを保存する。
- **AdaBN**：prefix 32試行でrunning mean/varianceだけを更新し、重みは更新しない。評価区間では統計・重みを固定する。
- **第3adapterのviability条件**：16人pilotで `mean |session Δ| ≥1pp`、`SD(subject mean Δ) ≥0.5pp`、固定効果除去後の奇偶session split-half `ρ_SB≥0.3` をすべて満たすこと。満たさないadapterを「低相関の証拠」に数えない。
- **prefix-Tent（棄却済み）**：lr∈{1e-4,1e-3}、steps∈{1,3}を同一checkpointで比較したが、再現する設定は平均絶対Δ=0.186ppの実質no-op、更新量を増やすと再現性が消えた。negative controlとしてのみ保存する。
- **session reset**：各target session開始時にsource checkpoint、BN統計、optimizer stateを完全resetする。
- **seedの扱い**：3 seedsを独立被験者として数えない。主解析はseed平均、seed差はrandom effectまたはseedを含むcluster bootstrapで扱う。
- **prospectiveテスト**：target trial 33以降を変更・削除しても、prefix後のadapted stateが変わらないことをunit testで確認する。
- **既存EAセルの扱い**：過去のfull-session transductive EA結果は参考値に留め、Riemann-LDA+EAもprefix 32だけでreferenceを計算して再実行する。

## 3. 一致の統計定義（事前登録）
harm-rateの単純相関は粗いため、主解析は連続Δを用いる。

### 3.1 主解析：EEGNet内のcross-adapter consistency
```
Δ_{s,j,a,r} =
  β_a + f(sourceAcc_{s,j}) + β_session·session_j + β_n·nTrials_{s,j}
  + u_s + v_{s,a} + c_{s,j} + q_r + ε

  a∈{EA, AdaBN, A3}, backbone=EEGNetで固定
  u_s    : adapterを越えた共通subject効果
  v_{s,a}: subject×adapter相互作用
  c_{s,j}: adapter間で共有されるsubject-session効果
  q_r    : seed効果
```
（source精度を調整しないと天井効果を「危険trait」と誤認する）

**EEGNet内の共通subject比**：
```
T_EEGNet = σ²_subject / (σ²_subject + σ²_{subject×adapter})
```
分散成分検定は通常のχ² LRTでなく **subject単位 parametric bootstrap 2,000回**。

pairwise Spearmanは、各adapterで固定効果（source精度・source精度²・session index・試行数）を独立に除いた後の、被験者別平均残差を用いる。共通subject random effectを入れた同一モデルのBLUP同士を相関させない。

### 3.2 副解析：EA固定のcross-backbone consistency
- Riemann-LDA+EAとEEGNet+EAについて、同じ共変量を除いた被験者別平均残差のSpearman相関を測る。
- 2 backboneだけなので、subject×backbone分散を主たる分散成分として強く解釈しない。
- cross-backbone `ρ≥0.3`かつ95%CI下限`>0`を、共通traitの**支持証拠**とする。主判定条件にはしない。

### 3.3 解析成立条件
各adapterについて、以下を先に確認する。
- adapter内split-half reliabilityは、各被験者のtarget sessionsをsession番号の奇数群／偶数群に固定分割し、固定効果除去後の被験者平均残差を相関させる。主値はSpearman-Brown補正後のρとし、補助的に1,000回のbalanced random split分布を報告する。
- 上記split-half reliabilityのpoint estimate `ρ≥0.3`。
- 被験者間分散が0より大きいことをbootstrapで支持。
- ほぼ全被験者を一様に改善／悪化させ、被験者順位が再現しないadapterは「global success/failure」とし、trait相関の有効な比較対象に数えない。

### 3.4 事前登録の主判定基準
**共通traitを支持**：
- EEGNet内3 pairのうち2 pair以上で、cross-adapter Spearman `ρ≥0.4`かつsubject-cluster bootstrap 95%CI下限`>0`。
- `T_EEGNet≥0.5`かつ共通subject分散のbootstrap 95%CI下限`>0`。
- 各比較adapterが上記「解析成立条件」を満たす。

**adapter固有を支持**：
- EEGNet内cross-adapter相関のmedian point estimate `≤0.2`。
- 3 pairのうち2 pair以上で95%CI上限`<0.3`、または再現的な符号不一致。
- `T_EEGNet≤0.3`かつsubject×adapter分散が共通subject分散を上回る。

**それ以外＝混合/結論不能**（無理に二値結論を出さない）。

AUROCとgate転移は主判定に使わず、共有成分が確認された後のsecondary/constructive評価とする。

## 4. Phase 2：eligibility転移（共有成分が確認された場合のみ）
- Phase 1でadapter内再現性とcross-adapter共有成分が確認されなければ、ここへ進まず新しいgateも作らない。
- 最初の2 target sessionsで **adapter A の subject risk score** を推定 → session 4以降の **adapter B を gate**。
- 閾値は外側training subjectsのみで学習、held-out subjectには固定適用。gate reject時はtarget backboneのsourceへ戻す。
- 診断実験：初期sessionの完全Δ使用。実用版：初期2 session×16 probe labels に落とす。
- cross-adapter risk predictionはAUROCとrisk–utility frontierの両方で評価する。AUROC≥0.65は実用候補の目安であり、trait成立の必須条件ではない。
- transferred gateの保持率を次で定義する：
  ```
  retention = (R_always − R_transfer) / (R_always − R_self)
  ```
  `R_self`がalways-adaptより有意に改善し、かつutility制約を満たす場合のみ計算する。分母が0以下なら「評価不能」とする。
- retention≥70%は構成的成功の目安。主たるtrait判定には使わない。

## 5. 検出力（62人）
単一相関：真ρ=.35→80% / .40→90% / .50→99%。**3比較補正後 ρ=.40で約80%**。
→ **中程度以上のtrait判定には足りる。ρ=.2–.3の精密判定には不足**（その場合は「混合」と正直に結論）。
- `ρ≥0.4`に加えてCI下限`>0.2`を要求する設計には十分な検出力がないため、その基準は採用しない。

---

## 6. 評価指標：risk–utility Pareto frontier
**worst-only / harmed(−1pp) は主指標から外す**（−1pp≈2試行で測定誤差と区別困難）。

- **Utility**：subject-balanced mean `U = (1/S)Σ_s mean_j(Δ_sj)`。
- **session重み**：各sessionに `w_sj = 1 / (S·n_s)` を与え、各被験者の総重みを等しくする。
- **Primary risk＝subject均等重み lower-tail CVaR@10%**：上記重み付きΔ分布の下位10%平均を`LCVaR10`、risk表現を`R10=−LCVaR10`とする。
- 副指標：weighted q05 / P(Δ<−5pp) / `source≥70%`だったsessionが適応後70%未満へ落ちる確率 / subject-level CVaR。感度解析 τ=0,3,5,10pp。
- **論文の主軸**：`min R10(π) s.t. U(π) ≥ 0.8·U(always-adapt)`（80%は70/90%も感度解析）。
  - 双対併記：severe-harm率をalways-adaptの50%以下にした条件で平均利得最大化。
  - source-only(U=0,risk=0)は端点として残すが80%制約を満たさず自明解にならない。採用policyに 95%CI lower(U)>0 を要求。
- `U(always-adapt)`は、gate対象となるadapterごとに対応するalways-adaptを基準とする。adapter選択policyを比較する場合は、outer training subjectsで選んだbest-global adapterも別baselineとして置く。
- 単一加重スコア/hypervolumeだけで勝敗を決めない。

---

## 7. 6週間の実行順
1. **risk metric＋causal prefix評価を共通ライブラリ化**（4セルで共有）。
2. **EEGNet engineering pilot**（16人×1 seed）でパイプライン検証。2026-06-22実施済み。
   - 14/16人以上で学習・推論が完走し、NaNやcheckpoint混線がない。
   - binary source accuracyの被験者medianが60%以上。
   - future trial変更に対するprefix state不変、session reset、branch前checkpoint一致のunit testを通す。
   - 結果：16/16完走・reset/test合格、source median 54.70%で不合格。prefix-Tentもviability不合格。
3. **source repair pilot**（2026-06-22実施済み）。
   - 16人を4-foldに分け、training 12人のsession 1だけでpopulation pretrainし、held-out 4人を各自session 1だけでfine-tuneする。
   - Braindecode EEGNet、seed 0。target sessionは評価にのみ使用する。
   - 元の合格条件 `median target source accuracy ≥60%` を維持する。
   - 結果：mean 60.10%、median 59.41%、13/16人改善。**不合格**。
4. source repair不合格のため、第3adapter screen、62人×3 seeds、階層モデル、Phase 2をすべて中止する。
5. 実装と16人結果を探索的appendixとして保存し、修論本線をprospective longitudinal risk auditへ戻す。cross-adapter系のLee2019外部転移とjournal拡張も行わない。

## 8. 6週間後の分岐（事前固定）
| 結果 | 出口 |
|---|---|
| 共通traitが強い | Phase 2のcross-adapter eligibility screeningへ。外部再現後にSLAH再昇格候補 |
| adapter固有 | 「普遍的危険ユーザーはいない。害はperson×adapter interaction」 |
| 共通成分＋interaction | shared vulnerability と adapter相性の二層分解（方法追加しない） |
| adapter内再現性も低い | trait/interaction判定不能。risk auditとして終了 |
| deep adapterが全て負 | gate探索を止め OTTA negative-transfer benchmark として修論化 |

- adapter固有だった場合は次に adapter oracle を計算：subject oracleがbest global adapterを≥2pp上回る→personalized adapter matching検討／headroom<1pp→selector作らず相互作用の特徴づけで終了／nested risk predictor AUROC<0.65→oracle大でも実用selectorへ進まない。
- **深層OTTA本体へ転じるのは、1 deep adapterが「平均Δ95%CI下限>0／3 seedsでsubject順位再現／修正可能なfailure mechanism」を満たす場合のみ**。trait否定だけを理由にTent改良へ飛ばない。

## 9. venue / scope
- cross-adapter trait論文としてのjournal計画は中止。
- 16人結果単独ではtrait/interaction論文にしない。修論appendixまたはworkshopのnegative engineering evidenceに限定する。
- 投稿の本線は、Stiegerのprospective longitudinal risk auditとrisk–utility frontier。journalを狙うなら別途、第2縦断データでrisk現象を再現する必要がある。
- TNSREよりJNEの方が依然適合するが、単一データ・negative auditの現状ではworkshop/修論が現実線。

---

## 10. 規律（5連敗の教訓）
- **新しい安全ゲートと第3adapterは実装しない。**
- 安全指標でadapterをチューニングしない（risk consistencyを人工的に作る）。
- 判定は事前登録基準に従い、無理に二値結論を出さない（混合なら混合と書く）。
- prospective厳守（future target試行・全セッションprior使用をやめる＝過去の "label-leak-free transductive" を是正）。
- 4セル段階ではEEGNet内adapter解析とEA固定backbone解析を分離し、識別できない統合分散を主張しない。
