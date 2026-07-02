# 現状まとめ（GPT議論用）— cross-session MI 安全選択的適応 / 2026-06-22

> この1ヶ月の検証を、データ・モデル・結果・考察・今後の順で超詳細にまとめる。すべて実データに基づく。過去実験の一部は**正解ラベルを評価から分離した label-leak-free transductive 評価**であり、target session 全体の無ラベル入力を使うため厳密な prospective 評価ではない。
> 議論の目的：(1) この結果群から何が言えるか、(2) 新規性を出すには今後どうすべきか。

---

## 0. TL;DR（先に結論）

- **問題**：cross-session EEG運動想起で、新セッションに教師なし適応(EA)すると平均は上がるが、**16%のセッションが過去のharmed基準（Δ<−1pp）に入る**（worst −12pp）。平均利得を保ちながら下側リスクを抑えたい。
- **この1ヶ月の核心結果**：「**最小ラベルでの安全な選択的適応**」を**5つの構成で試し、すべて成立しなかった**。per-session LCB(m=16)はharmedを一桁にできるが利得を+2.5ppまで捨てる。一方、利得を回収しながら真の選択性を得るには64-96 labels/sessionが必要だった。＝**「安全・高利得・低ラベル」を同時に満たす点は、試した設計空間では見つからない**。
- **守れる構造発見**：単一のEA adapterでは適応Δに安定した被験者間異質性がある（ICC(1)=0.468、被験者bootstrap 95%CI≈[0.35,0.56]）。ただし、これはまだ「人間trait」ではなく、**subject×adapter相性**の可能性を排除できない。
- **撤回済み**：「29人は無害」「時系列持続2.4倍」「危険被験者は予測しやすい」「5つの失敗はSLAHの帰結」。SLAH命名も保留する。
- **現在の主軸**：headlineは **“Average gains conceal concentrated downside: a task-aware longitudinal risk audit of cross-session EEG adaptation”**。次の最優先実験はnative taskごとのcalibration-budget frontierとreference estimation errorの分解。
- **2026-06-22実行更新**：16人prospective cross-adapter pilotは16/16完走・reset/test合格。ただしEEGNet source median 54.70%（事前基準60%未達）、prefix-Tentは実質no-op。追加のpopulation-pretrain source repairは平均60.10%・median 59.41%（13/16人改善）まで上げたが基準未達だったため、62人本実験は中止した。EA/AdaBNはadapter内split-halfが高い一方、補正後cross-adapter Spearmanは0.103で、adapter固有相性を示唆する探索信号が出たが、主張には使わない。詳細は [260622_cross_adapter_pilot_result.md](260622_cross_adapter_pilot_result.md)。
- **task-aware再監査（2026-06-22）**：raw MATのtasknumberを保持した62人×598 sessionsのcovariance cacheを作成し、LR / UD / 2Dをnative taskとして再評価した。旧mixed-horizontalとpure-LRのprefix-EA差はU −0.04pp、R10 +0.12ppでCIは0を跨ぎ、**context混合主因説は棄却**。pure-LRでもU +6.30pp、R10 6.91、ICC 0.294が残った。UDはLRよりR10 +2.98pp、P(Δ<−5pp) +5.16 points。prefix-32はfull EAよりLR/2Dでutility低下とrisk増加が有意。詳細は [260622_task_aware_stieger_audit.md](260622_task_aware_stieger_audit.md)。

---

## 1. データセット

### Stieger2021（今回の主戦場）
- 大規模・縦断の運動想起EEG。**62被験者 × 最大11セッション（別日）× 各セッション約200試行**。
- 縦断性（同一被験者を複数日追える）が最大の強み → 「日ごとの適応の効き/壊れ」を多数例で観察でき、被験者ごとの偏りを測れる。
- 今回：左手/右手の2クラス、60ch、計 **約524〜536 (被験者×セッション)** を使用（前処理で有効分のみ）。
- 前処理済みepochをキャッシュ済（`stieger_epochs_cache/`、62人）。MOABBの`get_data`が1被験者233秒と律速なので一度dumpして再利用。

### 補助データ（前回までの検証）
- **BCIC 2a**（9人, 4クラス, 22ch）：前回の主データ。小規模ゆえ統計的確証が弱い。
- **Lee2019**（54人, 2クラス, 62ch）：別系統アダプタ(Riemann)の検証に使用。

---

## 2. モデル / 設定

- **デコーダ**：session-1で学習し**凍結**した **Riemann-tangent-LDA**（共分散→接空間→LDA、shrinkage=auto）。
  - 注：意図的に古典手法（GPU不要）。深層(TCFormer)は今回の安全検証では未使用。
- **適応器(adapter)**：教師なし **EA（Euclidean Alignment）再センタリング**（session-jの平均共分散Rjで白色化し直す）。
  - 前回提案の DA-DC（4判定器合成）でなく、効果がクリーンに出る最小構成としてEAを採用。
- **評価プロトコル（過去実験）**：正解ラベルについては、校正(probe)＝各セッション先頭k試行、評価＝残り試行として分離。ただしEA referenceの一部はtarget session全体の無ラベル入力から計算しており、厳密には **label-leak-free transductive**。次フェーズでは先頭prefixだけを使うprospective評価へ改める。
- **過去の主指標**：平均Δacc（適応−source）、**harmed＝Δ<−1ppのセッション数**、worst（最小Δ）、%oracle、消費ラベル数。
  - oracle＝各セッションでsourceとEAの良い方を選んだ理想（到達不能、天井の基準）。
- **今後の主指標**：subject-balanced mean utility と lower-tail CVaR@10% の risk–utility frontier。harmed閾値とworstは副指標・感度解析へ下げる。

---

## 3. 検証結果（全て実データ。過去評価は主に label-leak-free transductive）

### 3.0 適応の素の挙動（土台）
| policy | meanΔ | harmed | worst | 備考 |
|---|---|---|---|---|
| source（適応なし） | +0.00 | 0 | 0 | 基準 |
| **always-EA（常に適応）** | **+7.8pp** | **87 (16%)** | **−12** | 76%改善 |
| oracle（達成可能上限） | +8.34 | 0 | 0 | always-EAは**oracleの93%** |

→ **今回のsource/EA二択では、always-EAが平均oracleの93%を回収する一方、下側リスクが残る。** 「平均が一般に解けた」のではなく、今回のadapter集合内で追加平均headroomが小さいという意味に限る。

### 3.1 〔棄却①〕コホート EB プーリング（mirage）
- 少数ラベル(k≤16)で「適応すべきセッション」をコホート縮約で判定。
- **k≤16で全採用に退化**（harmed≈81、選択性≈0）。母集団priorが強陽性なので全部採用してしまう。
- 真の選択性はk~64-96で出るが、そこでpoolingのper-session LCB比優位は小。
- 参考（2a/Lee2019）：2a(全員+)はEB-pooledが+4.58/害0、Lee2019(全員−)は全棄却で害0＝**コホート単位の挙動**で、per-session選択性ではない。
- **結論：安いコホート平均利得は買えるが、安いper-session安全は買えない。**

### 3.2 〔棄却②〕label-free シグナルによる veto
87の有害セッション判別 AUROC（n=536）：
| 指標 | AUROC |
|---|---|
| 確信オーバールール H | 0.565 |
| 不一致率 overrule | 0.559 |
| 確信低下 conf_drop | 0.563 |
| エントロピー | 0.605 |
| 分かれ具合 dispersity | 0.607 |
| 共分散ドリフト Riemann | 0.547 |
| **6指標融合 (LOSO)** | **0.661** |

- 単一は全て<0.62、融合でも0.66＝**弱い**（実用水準0.65をかろうじて）。
- **Lee2019で H が ρ=−0.72 だったのは極端なRiemann崩壊regime固有**で、Stiegerの穏やかな尾には転移せず（ρ≈+0.28）。
- **結論：正解なしで危険セッションを当てるのは実データでは不十分。** 前回フィルタ(分かれ具合)の核がここで揺らいだ。

### 3.3 〔棄却③〕異方的（固有方向別）trust 縮約
EAを固有方向ごとにラベルフリーtrustで縮約（危険方向だけsourceへ）。n=536：
| 条件 | meanΔ | harmed | worst | a_mean |
|---|---|---|---|---|
| EA(=full) | +7.75 | 87 | −12.2 | — |
| scalar_0.5（一律縮約） | +5.59 | 75 | −21.7 | — |
| **anisoS（安定性trust）** | +7.96 | 77 | −10.5 | **0.95** |
| **anisoM(β=1)（移動量trust）** | +1.07 | 75 | −8.9 | 0.41 |

- **anisoSは退化**（a_mean0.95＝ほぼ全方向full EA＝実質EAの言い換え、harmed 87→77は誤差）。
- **anisoMはscalar以下**（同harmedで利得が低い）＝**「大きく動く方向＝危険」でなく「大きく動く方向＝利得源」**だった。
- **結論：害と利得が同じ方向に乗っており、方向単位で切り分けられない。棄却。**

### 3.4 〔棄却④〕被験者内 縦断 EB ＋ α床
被験者の過去セッションをpriorにEB縮約、向かない人はα縮約。n=536：
| policy | meanΔ | %ora | harmed | worst | labels |
|---|---|---|---|---|---|
| always-EA | +7.79 | 93 | 85 | −11.0 | 0 |
| per-sess LCB(m16) | +2.54 | 30 | **9** | −6.4 | 8576 |
| subjMean-hard(m16) | +7.23 | 87 | 48 | −11.0 | 8576 |
| **within-EB+floor(提案)** | +7.62 | 91 | **72** | −8.5 | 8576 |
| **subject-oracle(天井)** | +7.93 | 95 | **54** | −9.9 | 0 |

- 提案 harmed 72 は always-EA(85)から13減のみ、**被験者クラスタbootstrapのΔharmed CIが0を跨ぐ**（有意でない）。
- しかも**素朴な subjMean-hard(48) に負け**た（α床0.3が裏目）。
- **決定的：subject-oracle天井ですら harmed 54**＝**被験者を丸ごと最適に採否しても害の大半(54/87)は消せない**（残りは良い被験者内の悪いセッション＝被験者粒度では触れない）。
- **結論：被験者単位の決定は天井が低い。**

### 3.5 〔棄却⑤〕per-session LCB × α床 融合（最後の一手）
LCBが「効くと確信できない」時に棄却(source)せず弱く適応(α床)する。n=536：
| policy | meanΔ | %ora | harmed | worst |
|---|---|---|---|---|
| LCB+floor0.0（棄却型） | +2.54 | 30 | **9** | −6.4 |
| LCB+floor0.2 | +4.24 | 51 | 70 | −9.7 |
| LCB+floor0.3 | +5.04 | 60 | 74 | −11.5 |
| LCB+floor0.5 | +6.48 | 78 | 75 | −20.5 |

- floorを上げると利得は回復(+6.5)するが、**harmedが9→70-75に激増、worstも−20pp**に悪化。
- ＝**LCBが見逃した危険セッションを「弱くても」適応してしまい、harmedが増える**。
- **結論：棄却の代わりにα床で利得回収、は不可能。安全と利得が両立しない。**

### 3.6 〔参考〕被験者履歴集約 policy（探索的・部分的に効くが限界）
| policy | effΔ | harmed | labels |
|---|---|---|---|
| SEQ-prevΔ(m8) | +4.44 | 28 | 4192 |
| **SEQ-subjMean(m8)** | **+6.05** | 34 | 4192 |
| SEQ-subjLCB(m8) | +4.84 | **19** | 4192 |

- 被験者累積で「高利得を保ったまま安く尾を削る」候補点は得た（subjMean +6.05/harmed34を1/4ラベルで）。
- だが**強い安全(harmed一桁)には届かない**（subjLCBで19が限界）。
- これは被験者履歴がpolicy上利用可能であることを示す探索結果であり、**追加の時系列持続効果を示す証拠ではない**。被験者内シャッフル検定ではlag効果は非有意。

---

## 4. 考察（この結果群から言えること）

### 4.1 中心的な負の結果：試した設計空間では安価な強安全は成立しなかった
「最小ラベルで、壊れるセッションだけ避ける」は、**コホートpooling／label-freeシグナル／異方的縮約／被験者内EB／LCB×α床 の5方向すべてで失敗**した。per-session LCB(m=16)はharmedを一桁にできるが利得は+2.5ppに留まり、利得を保ちながら選択性を得るには64-96 labels/sessionを要した。
→ **「安全・高利得・低ラベル」を同時に満たす点は、今回試したadapter・信号・決定則の範囲では見つからない。** 形式的な不可能性や、他adapterへの一般化は主張しない。

### 4.2 なぜ失敗するか（機構の解釈）
1. **per-sessionのΔは推定誤差が大きい**：1セッション200試行の少数(8-16)では「効く」と統計的に確信できない（検定力不足）→ 最小ラベル判定が退化。
2. **label-freeシグナルは穏やかな害を捉えられない**：出力の分かれ具合等は、極端な崩壊(Lee2019)では効くが、Stiegerの worst −12pp程度の穏やかな害とは相関が弱い(0.66)。
3. **害と利得が同方向**：異方的の失敗が示すように、大きく動く（利得を生む）方向ほど危険でもあり、方向で切り分けられない。
4. **単一EA adapter内には被験者間異質性があるが、被験者粒度だけでは不十分**：ICC=0.47は同じ被験者のΔが似ることを示す。しかし人間traitかsubject×EA相性かは未確定であり、subject-oracleでも harmed54が残る。

### 4.3 守れる前向きな構造：単一adapter内の被験者間異質性
- ICC(1)=0.468、被験者bootstrap 95%CI≈[0.35,0.56]、leave-one-subject-out 0.44-0.49。
- harmed閾値を−1から−2/−3/−5ppへ変えても、負のΔが一部被験者へ集中する傾向は残る。
- ただし「観測期間中に害がなかった被験者」を安全と分類しない。6-10セッションでは真の害率に大きな不確実性が残る。
- cross-adapter共有性は16人pilotでは結論不能となった。修士期間では深追いせず、task×calibration-budgetによるrisk変動を先に確定する。

### 4.4 平均utilityと下側riskの非対称
- always-adaptは今回のEA-session oracleの93%を平均で回収する一方、負のΔを持つセッションが残る。
- source-onlyはrisk 0の自明解なので、安全だけを最適化しても意味がない。
- 今後は「平均を捨ててworstを最小化」ではなく、**平均利得を一定割合以上保持する条件でlower-tail CVaRやsevere-harm率を抑える**Pareto問題として評価する。

---

## 5. 新規性・研究成果を出すには（文献精査の結論を反映）

> 先行文献調査と、その後の統計・査読監査を統合した現在の見解。

### 5.0 最重要の結論：論文の正しい形
**「5手法が失敗」論文でも「危険被験者trait」論文でもなく、task-aware prospective risk auditとcalibration-information frontierにする。**
- 推奨headline：**“Average gains conceal concentrated downside: a task-aware longitudinal risk audit of cross-session EEG adaptation.”**
- SLAH命名は、別adapter/backboneでも同じ被験者効果が再現するまで保留。
- label-free予測の弱さと5つの負の結果は、互いに異なるfailure modeを示す支持証拠として扱い、一つの原因へまとめない。
- **"impossibility"とは言わない**。主張は「今回試した設計空間で安全・高利得・低ラベルの同時達成を確認できなかった」に限定する。

### 5.1 先取り済み（手法として主張できない）
- adopt-vs-keep の下側信頼限界判定 = **c-value (Trippe 2021)**
- never-below-baseline 契約 = **Safe Policy Improvement (SPIBB)**
- コホート縮約＋go/no-go = **Bayesian basket trials**
- 同Stiegerでの EA+AdaBN OTTA = **Wimpff 2502.06828**（ただし harmed分布は未報告）
- calibration-free continual = **EDAPT 2508.10474**
- scalar partial/geodesic α recenter = **GOPSA(NeurIPS24)/SPDIM(ICLR25)/TTN**
- per-band filter-bank Riemann shrinkage = **Springer 2024**
- label-free精度推定(不一致) = **AETTA 2404.01351**

### 5.2 防衛可能な新規性の置きどころ
1. **縦断risk audit**：平均Δだけでなく、subject-balanced utility、lower-tail CVaR、severe-harm率、被験者間・被験者内分散を報告する。
2. **task-aware within-cohort比較**：同じ62人・同じsessionsのLR/UD/2Dで、dataset差に交絡されずtask依存riskを測る。
3. **calibration-information frontier**：prefix budgetとfull-session referenceの差をutility/risk両面で定量化する。
4. **再利用可能な測定プロトコル**：causal prefix、task-aware event定義、risk–utility frontierを共通化する。
5. **cross-adapter探索結果は補助**：16人pilotは修論appendixに限定する。

### 5.3 次にやるべき具体実験
- **cross-adapter pilotは実施済み・終了**。16/16人完走したが、subject-specific EEGNet source median 54.70%、population-pretrain repair後も59.41%で事前60%基準に未達。prefix-Tentも有効adapterとして不成立。
- **prospective longitudinal risk auditはtask-awareで完了**。LR / UD / 2Dすべてで平均利得とdownside riskが共存した。
- **次の本線はnative-task calibration-budget frontier**。prefix `{8,16,32,64,full}`でU/R10/P(Δ<−5pp)を測り、task×budget効果とreference estimation errorを分解する。
- 新しいselector、gate、深層adapter、eligibility transferは追加しない。
- cross-adapterの実装と16人結果は修論appendixに保存する。詳細は [260622_cross_adapter_pilot_result.md](260622_cross_adapter_pilot_result.md)。

### 5.4 venue見立て
- **現状**：NeurIPS/ICLRのdistribution-shift/TTA workshop、BCI系workshop、EMBC級が現実的。
- **cross-adapter 16人結果**：単独投稿の中核にはせず、negative engineering evidenceまたはappendix。
- **Stieger prospective risk audit**：修士論文の中核。単一データのままならworkshop/EMBC級が現実線。
- **+第2縦断データでrisk現象を外部再現**：JNEが射程。現状は臨床・支援機器への接続が弱いためTNSREよりJNEが適合する。
- **NeurIPS/ICLR本会議**：形式的下界か新手法が要り、現状の延長では届きにくい。

---

## 5.5 【監査記録】撤回・修正・新方向（2026-06-22）

GPT(査読者役)が一部主張を**再計算して反証**。以下は変更履歴であり、本文0-5.4はこの内容を反映済み。

### 撤回する主張
- **「時系列持続2.4倍」＝撤回**。P(harm|prev-harm)=0.39 は**被験者の基礎リスク差だけで説明できる**（危険な人は毎回壊れやすい＝持続でなく個人差）。被験者内シャッフル帰無検定で追加lag効果は非有意(片側p≈0.23)。＝私の解釈ミス。
- **「29人は無害」＝撤回**。正しくは「観測期間中に害を観測しなかった」。6-10セッションでは真の害率の95%上限が26-39%。
- **「危険被験者は予測しやすい」＝撤回**（持続性が無いなら事前予測の根拠が弱い）。
- **「5つの失敗はSLAHの帰結」＝撤回**。5つは原因が別：①検定力 ②可観測性 ③gain-harm entanglement ④粒度の限界。同一原因ではない。

### 守れる主張（GPTも再計算で支持）
- **ICC(1)=0.468 は安定**。被験者bootstrap 95%CI≈[0.35,0.56]、LOSO 0.44-0.49。閾値を−1→−2/−3/−5ppにしても害の被験者集中は残る。＝**被験者間異質性は本物**。
- 平均利得と下側リスクの非対称（平均では小headroomだが下側に集中した害）。
- 安価なper-session判定の難しさ（検定力・可観測性の壁）。

### 修正する点
- **headlineをSLAHから格下げ**。SLAH(被験者trait)は「別adapterでも同じ被験者が危険」と確認できるまで命名保留。推奨headline：**"Average gains conceal concentrated downside: a longitudinal risk audit of cross-session EEG adaptation"**。
- **worst-only → risk-utility Pareto frontier**。「平均を捨てworstへ」でなく「**平均利得を保つ条件下で下側リスク(harm確率/CVaR)を制約**」する評価に。worst=min・harmed(−1pp)単独依存をやめ、**CVaR/q05・閾値感度**を併記。
- **"leak-free" → "label-leak-free transductive"** に正確化。現EA referenceはセッション全体(未来の評価入力含む)を使用＝厳密なprospectiveでない。within-EBのcohort priorも全セッションprobe平均を固定prior＝prospectiveでない。要修正。

### 過去の最優先実験変更（E1→cross-adapter consistency、後にpilotで終了）
> 以下は当時の監査記録。16人pilotとsource repairが事前基準に届かなかったため、現在の最優先ではない。
GPT指摘：**最大の弱点は「1 adapter・1 backboneで見たランダム効果を"人間のtrait"と解釈していること」**。E1(ICCのCI)はGPTが既に計算し安定確認済＝衛生検査に過ぎない。**生命線は被験者traitの反証可能な検証**：

- **【最優先】cross-adapter / cross-backbone subject consistency**：同じStieger被験者に EA / AdaBN・Tent系 / 別backbone を適用し、
  - 被験者平均Δの順位相関、harm-rateの一致、subject×adapter交互作用、
  - 一方のadapterで学んだeligibilityが別adapterへ転移するか。
  - **一致すれば「危険被験者trait」を支持(SLAH再検討)／不一致なら adapter-specific incompatibility**（その場合は普遍的subject gateを否定し adapter選択・source anchorへ）。どちらでも修士論文の有効な結論になるが、投稿論文としての強さは外部再現または機構説明に依存する。
- **統計・プロトコル修正（先に済ませる衛生）**：ICC CI・閾値感度・CVaR/q05・worst-subjectを報告。時系列効果は被験者異質性を条件付けて再検定。leak表現の修正。
- **AETTA系を強baselineに**：深層モデル投入時のlabel-free baselineとしてAETTA(不一致ベース)が必須。現6指標だけでは「baseline不足」と指摘される。
- **E2(被験者単位ルール)は consistency陽性が前提**。かつ subject-oracle天井 harmed54 のため「強安全保証」でなく「固定ラベル予算下のutility-risk改善」に目的限定。

### Publishability（GPT見立て、現実的）
- 現状：NeurIPS/ICLR workshop・BCI系workshop・EMBC級。
- JNE最低線：完全risk統計＋causal評価＋**2 backbones/2 adapter families**＋subject riskのadapter越え再現＋可能なら第2縦断データ＋外部/nested decision-rule評価。
- 近年のBCI適応研究は2-9データセットで評価(Wimpff2/T-TIME3/EDAPT9)＝**単一パイプライン記述では規模で負ける**。
- NeurIPS/ICLR本会議：一般的な安全TTA定式化・理論限界・複数領域で効く新手法が要る＝EEG内追加実験だけでは厳しい。

---

## 6. 議論したい問い（GPTへ）— 回答済み、上記5.5に反映
1. 5つの負の結果＋偏在の発見は、**それ自体で publishable な貢献**になるか？ なるなら何を足すべきか？
2. 「安く安全は不可能」をどこまで強く（不可能性/下界として）主張できるか？ 単なる"5手法の失敗"を超えるには？
3. 構成的成果が要るなら、5.3のどれが最も筋が良いか？（AETTA / 深層 / 被験者eligibility / 能動ラベル）
4. 「worst-case主軸＋偏在の発見」で TNSRE/JNE クラスは狙えるか？ NeurIPS/ICLR workshop が現実的か？

---

## 付録：再現情報
- 解析スクリプト：`intentflow/offline/scripts/analysis/` の stieger_eb_gate.py / stieger_probs_dump.py / stieger_sequential_veto.py / stieger_anisotropic_align.py / stieger_within_subject_eb.py / stieger_lcb_alpha_fuse.py
- 結果JSON：`intentflow/offline/results/research_outputs/` の 260609〜260622_*.json
- キャッシュ：`stieger_epochs_cache/`（62人前処理済）、260622_lcb_alpha_curves.npz（a-gridカーブ）
- 過去解析は主にCPU・seed固定。評価は正解ラベル分離済みだが、一部はtarget session全体の無ラベル入力を使うtransductive設定。次フェーズでprospective prefixへ統一する。
