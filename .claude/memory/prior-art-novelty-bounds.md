---
name: prior-art-novelty-bounds
description: DA-DC/family-lawの新規性を制約する先行研究 + P2オンラインno-opの不利事実（2026-06調査）
metadata:
  type: project
---

DA-DC/cross-family研究の新規性を制約する先行研究と、自前の不利な事実（2026-06-03のweb調査＋実験再確認）。論文の主張範囲を決める土台。`dc-replay-empirical-ceilings.md`（精度天井）と `research-direction-2605.md` のnovelty境界を更新する位置づけ。

**コア機構は先行**: cross-model disagreement→label-freeでconfident-error検出→selective prediction は `Gorbett&Jana arXiv:2603.25450`(26/3, LLM), `Too Consistent arXiv:2505.17656`, `GDE ICLR2022(2106.13799, shiftで破綻)` が先取り。「disagreement→abstain」自体はQBC/El-Yaniv&Wienerで既知。

**Family-lawは"現象として"先行**: `Hidden Clones arXiv:2603.17111`(26/3, VLM) が同一語彙(family bias/clones/cross-family)・同一指標(Q統計/double-fault/error-overlap)・同一機構(Misleading tier=相関した自信満々の多数決誤り)。同一深層族の相関は `Abe NeurIPS2022`, `Hacohen ICML2020`, `Fort 2019(init多様性は分布内で脱相関=scope限定要)`。指標標準 `Kuncheva&Whitaker 2003`。

**MI-BCI近接先行**: `MI-UQ arXiv:2507.07511`(深層は過信/古典・リーマンは高校正、agreement融合なし)。オンラインMI-TTA混雑: `T-TIME TBME2024(2412.07228)`, `dual-stage 2509.19403(同じ+4.9pp)`, `BFT 2601.07556(凍結深層+軽量online adapter+疲労SEED-VIG)`, `Wimpff 2024(2311.18520, OTTA baseline)`. label-free監視 `AETTA CVPR2024(same-model dropout)`. EA基盤 `Wu group JNE2025(2502.09203, online EA referenceは陳腐化=P2前提)`.

**自前の不利な事実**: P2(オンライン族一致ドリフト追跡)は2aで no-op — `260603_d1_drift_gated_online` gated 86.92/87.00 vs static portfolio 87.15、causal 86.30/86.61、n_commits≈0.67(ゲートほぼ発火せず=2a内ドリフト小)。「静的DA-DCを超える唯一の筋」が当の実験で超えていない。

**防衛可能な交差点(誰も埋めてないunion)**: 表現族(深層 vs リーマン幾何 vs スペクトル=異なる帰納バイアス、アーキ違いではない)の不一致を、drift下のconfident-wrong検出＆fail-closed安全ゲートに使い、"同一深層族では不可能"を複数dataset×backbone×paradigm(MI+疲労)で示す。差別化刃: 表現族vsアーキ族 / 安全ゲートvs精度投票 / confident領域×drift / 深層凍結(CFCS失敗=`tcformer-hybrid-failure.md`が根拠)。

**【新法則・2026-06-04 TCFormer実測】drift localizes to the HEAD, not features**: session_E でTCFormer深層特徴は真ラベル再分類(LDA-probe CV)で85.6%分離可能なのにsoftmax head は83.1%（S7: 83.7→95.1 +11.4!, S5: 72.6→79.9）。=driftは決定境界のズレで特徴崩壊ではない。→CFCS失敗(特徴を動かした)を機構的に説明＝「直すべきはheadであってfeatureでない」。label-free transductive/self-trainingではこのhead-headroomを掘れない(-0.9, 5/9害)＝label-locked。headは低次元(64→4)＝最小ラベル/ErrPで安価に掘れる可能性。先行に見当たらない新規診断＝**法則2、論文の主役候補**。

**【label-free天井確定・2026-06-04】** TCFormer/2a総当たりで、DA-DC(87.15)を超える label-free 手法は最大+0.7pp(DADC+stack=agreement-seed stacking, 87.96/87.77 両seed)。**「DA-DCの上に+5」はlabel-freeで不可能**(oracle T+R+S=94.75はlabel-locked)。DA-DC自体はsource比+5.2pp・下位半分+5.2・非回帰(+0.35)＝弱者救済＋強者非崩壊は既達成。**"+5"は無適応比であって、DA-DC比ではない。** DA-DC比で伸ばす唯一の道=ErrP×head再校正(法則2)。

**法則2は2bで再現・一般法則化(2026-06-04)**: 2b head_headroom 平均+4.2pp、S5: 64.1→97.8(+33.8!)。2a(+2.5)と合わせdataset跨ぎ成立。

**【ErrP/最小ラベル head-recalib も DA-DCを超えない・確定(2026-06-04, head_recalib_curve.py)】**: frozen 64-d特徴にK個ラベルでLDA head再校正→ 2a: K=64クリーンでも84.3<DA-DC87.2(単一head天井85.6<アンサンブル)。2b: K=64 ε0で87.6だが ErrP品質ノイズε≥0.2でsoftmax以下に崩壊(80.9/74.6)。→**ErrPでは掘れない**。**accuracyは全方向(label-free/CFCS/多様体/最小ラベル/ErrP)で閉じた。「DA-DCの上に+5」は不可能が最終確定。** "+5"はsource比のみ(DA-DC自体が達成済)。

**最終結論「これだ」**: 精度上積みは閉。成果は**2法則＋安全デコーダ**=Head-Frozen Cross-Family Decoder。法則1(族脱相関→confident-wrong70%捕捉/+5source相補)[2a]＋法則2(drift→head, feature生存, CFCS失敗を説明)[2a+2b一般法則]。深層feature凍結＋cross-family融合＋confident-wrong安全ゲート(全label-free)。4基準: 新規性=法則2(先行無)、論理=全実測、安全=confident-wrong/fail-closed、汎用=2a/2b跨ぎ、精度=source比+5.2非回帰。venue=TNSRE/JNE。生命線実験=第2backbone(EEGNet/ATCNet)で法則2再現。

**【自己教師あり domain-fit 探索・2026-06-04 probe】** source_train_features_s0.npz に **session_T train_feat + session_E eval_feat (全9被験者, 64-d, ラベル付)** がキャッシュ済＝モデル自身の source manifold が利用可能。即席probe結果: (1) per-trial **source-class Mahalanobis** は softmax正誤と一貫負相関 spearman -0.32(全9被験者で負, S7 -0.44)＝confidence/margin/entropy/disagreement(全DEAD)と異なる label-free 正誤信号。(2) source-proto(LDA-like)で head を label-free 置換すると **S7 83.7→91.3(+7.6, 法則2のdrift被験者と一致), S1/S5/S9 微増** だが **S2 75.7→61.1, S8 90.3→82.3 で崩壊** ＝双峰。固定blend50は harmed 4/9で平均劣化(80.9<83.1)。**含意**: source-proto re-head の信頼性が被験者/trialで双峰。勝者(drift優勢=S7)と敗者(source-proto自体が劣化=S8)を **label-freeで判別できるか** が全候補の生命線。naive conjunction gate は発火0(非自明)。=精度は閉じたが「自己教師あり domain-fit で法則2 head-headroom を label-free 開錠」は **未テスト方向**(head_recalib=ラベル使用, selector=被験者単位cov特徴で頓挫とは別)。

**【非学習ERDアンカー実測・2026-06-05】** `erd_prior_quality.py`(前任の最後の未着地アイデア=モデル外・非学習の生理アンカー: LI=logP(右SM cluster)−logP(左), median split=教師なし)を実測。**naive形は弱い**: 2a(L/R hand 2値, 22ch) **平均59.0%**(大半chance±, S3=83/S7=62以外)、2b(3ch binary) **双峰**(S4=94/S6=85/S8=89 vs S3=55)。さらに**Law2劇的drift被験者S5(64→97.8)/S7/S9が`stop=4.0`のwindowingエラーでload失敗**(stop縮小で可)。前任の楽観verdict("ALIVE")は支持されない。ただしこれは生理アンカーの**最弱形**(広帯域8-30/全0-4s窓/rest正規化なし/被験者帯域なし)。異族D1/D2/D3は全てsession_T学習LDA=学習済み境界でdrift時に同じ崩れ方を共有(非学習不変量ではない)。**rest区間/cue時刻/時間連続性は前任が壁の例外として挙げながらP2(2a no-op)以外で未検証**=acknowledged-but-untested gap。`create_windows_from_events`の負offsetでrest window取得可(実装可能)。次の筋: rest正規化ERD%の時間波形を「分類器」でなく**モデル外のconfident-wrong veto(生理的整合性チェック)**として使い安全に倒す(精度天井は閉なので精度は狙わない)。

**【生理plausibility veto 決定実験・2026-06-05, `erd_veto_complementarity.py`】** 真2b source(sanity合格 mean=87.74)で「headのconfident-wrong trial(top50%margin & 誤)でモデル外ERD側方化が正解を知るか」を実測。**pooled ERD_on_CONF-WRONG=52.8%≈chance**=死亡。ERDアンカー自体は2bで mean 76%と decent なのに、CW上はchance＝**confident-wrongは生理信号自体が曖昧なtrialに集中→headとERDが相関して失敗**。veto precision も低く(S5=4.2!)強被験者の正解trialで誤発火＝安全性をむしろ悪化。唯一S2(最弱src70)のみERD_on_CW=62%だがn薄・非一貫(S1=33%逆相関)。**=identifiability壁はモデル外の生理事前分布にも拡張される(脱出路を厳密に1本潰した)。** 素朴ERD veto/anchorは2a/2b両方で死亡確定。

**【重大訂正・2026-06-05】真2b source S5=97.81%(cached logits, mean=87.74で検証)。** 前任の「S5: 64.1→97.8 (+33.8 head-headroom=法則2最劇的例)」の64.1は**EA汚染ea_aware版の壊れたbaseline**。法則2の看板例はartifact。法則2(drift→head)の汎用性主張は真source特徴で要再検証(2a `tcformer_manifold_headroom.py`も同様にEA-aware特徴の疑い)。

**【次の筋・2026-06-05仮説】単一静的セッションのlabel-free補正は内部信号(前任)もモデル外生理(今回)も閉。未踏は時間/多セッション構造。2bは5セッション(train0,1,2+test3,4)=driftのtrajectoryを持つのに前任は pooled に潰した。「drift trajectory tracking(観測されたdrift履歴=新情報)」が2bでのみ検証可能な新定式化候補。P2(2a no-op)は単一ステップだった点と区別。**

**3戦略**: A=ドリフト局所化(2b cross-session/後半block/疲労でstaticのslope劣化をonlineが抑える=P2救済or反証の決定実験), B=cost-aware selective labeling(family-disagreement選択がconfident-wrongに偏る理論、安全KPIで勝負；精度ではuncertaintyが勝つのは既知), C=detection-only(P2捨ててP1厳密確立、最も反証に強い)。見立て=C本命、AでP2を救えなければP2撤回。"necessary&sufficient"は未証明→"条件付き優位"に弱める。CFCS(深層再学習)は全変種失敗で除外。

**【sagittal-symmetry TTA は先行確定・2026-06-06, 本文検証済】** 「p_sym = p(x) + classswap(p(mirror_LR x))」(鏡像L/Rチャネル交換＋クラス入替の予測平均)を2a/2bで実測し source比 2a+0.6/2b+1.1 を得たが、**MCL-SWT(arXiv:2409.00130) 式(12) `[Yl,Yr]=[Yl_o+Yr_o]+[Yr_m+Yl_m]` と完全同一**(PDF本文で確認)。**データセットも2a/2b一致**、しかも式(12)は先行[5]"shallow mirror transformer"を引用＝test時の鏡像予測結合は先行2本分既出。**新規性ゼロ確定**。鏡像+クラス入替の概念自体も `Channel Reflection 2412.03224`(学習時aug, 8dataset/4paradigm)・`Braindecode ChannelsSymmetry` で既出。symmetry-equivariant **gradient TTT**版は敵対検証3/3 reject(勾配版は低gain対称縮退で崩壊を再開, F6/F7, Wu groupが意図的にtrain-only/backprop-freeに留めている)。S5 64→98(+34)は前述EA汚染artifactの回復で再現性0%。

**【GPT提案の"新情報"セルを実測で両方棄却・2026-06-06】** GPT(外部)はオフライン本命に「他被験者のラベル付きdrift対transport(S1)＋drift下の集合予測/routing(S2/S3)＋pre-cue/rest negative-control(S5)」を提案。**新情報を持つのはS1とS5の2つ**で、両方実測棄却: **S1死亡** (`s1_drift_sharing.py`, 2a tangent空間): クラス差分ドリフト(右−左軸のT→E移動=他者ラベルでしか測れない=S1の新情報本体)の被験者間cosine **+0.016±0.136≈0=ドリフトは被験者固有**、他者rho transport 72.8 vs source 72.5(+0.3=無)。自己marginalのみ+3.7=ただのEA。**S5死亡** (`s5_rest_nuisance.py`, 2a): rest共分散白色化行列はEA(MI共分散)白色化行列と **cosine 1.000(全被験者)＝S5≡EA**、REST 76.5 vs EA 76.6 vs source 76.2、同じ4被験者を同じだけ破壊。→**識別不能性の壁は3独立情報源で確認: (1)単一未ラベルセッション無情報(F2), (2)集団のラベル付きdrift prior無効(driftが固有), (3)baseline nuisance≡alignment。** これは前より強い負の結果=characterization論文の核を強化。S2/S3はS1死亡で「集合の素」を集団transportから作れず、target自身の曖昧性(cross-family候補)由来＝conformal+ensembleに縮退(既存・中novelty)。GPTも「精度無理ならcharacterization+uncertainty+safetyに賭けよ、limits-proof自体が貢献」で同結論に収束。

**【Law-2 CV検証で実在確定・2026-06-06, `verify_law2_cv.py`】** clean cached特徴(source_train_features_s0.npz)で2a head-headroom を5-fold CV検証: fit-on-test +6.0pp(過学習・旧主張)だが **CV-LDA +3.7pp / NCC-CV +1.9pp で実在**、drift被験者(S2 +8.7, S7, S8, S4)に集中。S5は無し(stable)。**法則2(driftはheadに局在)はartifactでなくCVでも生存**(2026-06-05保留疑義を解消)。ただし+1.9〜3.7ppは**label-lock**(session_Eラベルでhead fit時の利得=head_recalib/ErrPで「label-freeでは掘れない」と確定済)＝実在する診断事実だがlabel-free実現不能。なお探索workflowエージェントの「CV -0.9pp=Law-2はartifact」報告は自前CV検証で**誤りと判明**(エージェント主張は要自前検証の教訓)。

**【オフライン方向探索の最終結論・2026-06-06】** Causal-probe(閉ループ介入で識別不能性を能動的に破る=GPT本命)は環境整備が締切に間に合わず検証不能で却下。残るオフライン4方向を8エージェントworkflow(先行調査＋敵対検証)で評価し **leads=0, 全4 preempted**: D1 symmetry-TTT(上記先行), D2 amortized in-context(STMA/SPDIM/MABN先行＋9被験者でcross-subject pretrainingに退化), D3 限界論文+monitor(symmetry平均部は先行だが「label-free OTTAの情報限界の厳密特徴づけ＋clusterability monitorのunion」は未occupied), D4 longitudinal(Stieger利得=ユーザ学習の交絡 arXiv:2502.06828, 機構はNimbusSTS/variational-Bayes Kalman/SMCで製品化済)。**=新規×精度向上×オフライン×汎用 を同時に満たす手法は存在しないが確定**。唯一の防衛可能成果=**limits特徴づけ(Claim1)＋clusterability信頼度モニタ(Claim2, +0.8 2a/2b/Lee2019)**＝精度でなく characterization+安全。手法が勝たなくても論文は成立(decoupled)。
