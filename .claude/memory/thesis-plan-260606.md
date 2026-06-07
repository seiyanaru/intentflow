---
name: thesis-plan-260606
description: M2修士論文の確定方向と半年実行プログラム（オフライン特徴づけ論文：壁＋機構＋monitor）
metadata:
  type: project
---

2026-06-06確定／2026-06-07で枠組み刷新。**修士論文（M2、提出まで約半年＝2026年末頃）はオフライン論文**。精度の壁破り（causal-probe＝閉ループ）は環境コストで**範囲外**。詳細な負の結果・先行・walls は [[prior-art-novelty-bounds]]。

**【枠組み刷新・2026-06-07：2段構え（gate→adapt）に確定】** 「壁を示す（負の結果）」を contribution から外し、ユーザ主導で **positive な2段アーキテクチャ**に再構成：脳波には(A)適応で直せる session drift と(B)そもそも適応できない弱い信号が混在。一律処理は重い＋危険。→ **① 軽量な信頼度フィルタ（clusterability=分かれ具合）で弱い信号を先に・安く弾く（＝提案の核・新規）→ ② 通過したものだけ DA-DC で適応**。効果＝軽い＋安全＋実効精度↑(selective prediction)。**新規性の2条件（両立で初めて立つ）**: (1)フィルタが既存の信頼度指標(conf/entropy/Mahalanobis/Blankertz)より良い＝E4 make-or-break、(2)本当に軽量＝深層forwardでなく浅い特徴(共分散/tangent)でゲートできるか（**未検証・要確認**。Lee2019 tangent版は弱かった-0.26だが弱デコーダ交絡の疑い）。clusterabilityはクラス数kで測る(2a:k=4, 2b:k=2)＝分類数非依存。

**論文の3主張:**
- CLAIM1（壁）: label-free観測型OTTAは識別不能性により情報限界。どのlabel-free信号もDA-DCを>~0.7pp超えない。**多データセットで一般法則として示す**（1データセットの逸話にしない）。
- CLAIM2（機構=なぜ閉じるか）: (a)Law-2=driftはheadに局在（clean 2a 5-fold CV head-headroom +3.7pp LDA/+1.9pp NCC実証、drift被験者に集中、ただしlabel-lock＝label-free実現不能）。(b)S1ドリフト分解=cross-subject driftは幾何的に集団共有(cos~0.62)だが共有成分は精度inert(LOSO transport +0.4pp/oracle gapの3.7%)・精度成分は固有で転送不能＝"identifiable-yet-uncorrectable"。
- CLAIM3（唯一の正の貢献）: clusterability(未ラベルsession_E特徴のsilhouette)→信頼度予測（2a +0.78 / 2b +0.82）＝reliability monitor/abstention gate。**第3データセット汎用性が最大の開問**。

**実行プログラム（GPT＋3 workflow＋自前実測で確定）:**
- データセット: Tier A {2a, Lee2019(n=54), 2b(3ch=EA OFF)} ＋ Tier B {BNCI2015_001(13ch, right-vs-feet)}、ストレッチ {Stieger2021(学習交絡を session-index 回帰で除去), SHU}。backbone TCFormer主・EEGNet副。
- F7回避: **cross-subject(LOSO)事前学習＋軽い特化**（from-scratch per-subjectは58.3%で枯渇＝禁止）。壁の baseline は pooled+DA-DC（fine-tuned decoderで測ると不公平に baseline が上がる＝最大の公平性エラー、禁止）。EA: 2a/Lee2019=ON, 2b=OFF。clean特徴で測る（EA汚染cache禁止）。新config suffix（tcformer_pooled.yaml等）、tcformer.yaml上書き禁止。要コード追加: datamodules/lee2019.py + Lee2019LOSO（bcic4_2a.py を踏襲、_ea_align_tt再利用）。
- 実験順: **E1=最重要go/no-go**（Lee2019 F7-safe再構築→monitor汎用性 n=54＋S1分解 n=54, ~28GPU-h/1週）→ E2 2a再現(健全性) → E3 壁 → E4 monitor vs 既存予測器(Garg/Deng/Blankertz rest-SMR, 差別化) → E5 第2backbone → E6 task一般化 → E7/E8 ストレッチ。

**【E1 PASSED・2026-06-07, `lee2019_e1_monitor_n54.py`】CLAIM3最大リスク解消**。cross-subject事前学習（3-fold全54カバー, EA+interaug, F1=32）で **mean acc 82.5%（F7回避: from-scratch 58%→82.5%）**、**clusterability→acc Spearman +0.638 95%CI[+0.43,+0.78] perm-p=0.0002、partial(source acc制御)+0.514**。3基準(acc≥68/Spearman≥0.4・CI除外0/partial≥0.3)全通過。**monitorは n=54・62ch で確証**＝以前の揺れ(+0.53弱デコーダ/-0.26 2nd backbone)はF7交絡と確定。パイロット(n=6)で+0.94、frozen+LDA(79.7)≈headFT(80.0)＝深層特徴が被験者横断転移＝"オンライン使用可能デコーダ"。結果JSON: docs/research_progress/260606_lee2019_e1_monitor_n54.json。
**残タスク（CLAIM3完全確保）**: E5=EEGNet第2backbone、E4=clusterabilityが既存label-free予測器を AURC で上回ることを示す（相関でなく優位性＝新規性）。SOLID=CLAIM2 head-headroom(2a)＋CLAIM3 monitor(2a/2b/Lee2019 n=54, TCFormer)。

**【deep-research 文献調査・2026-06-08, 23ソース3票検証】新規性は"3点交差"のみ、E4ベースライン確定**。3者(GPT/前回workflow/deep-research)収束。genuinely novel = **(a)silhouette/clusterabilityをゲート信号 ×(b)MI cross-session ×(c)セッション単位abstain-safe selective prediction** の交差のみ。カスケード・label-freeゲート・selective predictionは各々先行(StableSleep arXiv:2509.02982=EEG睡眠entropyゲート→Tent / PonderTTT 2601.00894=LM再構成lossゲート / SoTTA 2310.10074=vision confidenceゲート / TinyTTA NeurIPS24=early-exit)＝incremental。MI性能予測(Blankertz SMR r=0.53 / microstate AUC=0.83)は被験者単位・安静時=粒度違い。silhouette×MIの唯一近接 MSFS(Brain Sci 2026, 16(2):230)は教師あり訓練時正則化=test-timeゲートでない。**E4必須ベースライン(全てEEG転用・誰もsilhouette不使用)**: Nuclear-norm/dispersity(Deng ICML23 arXiv:2302.01094=ATC超え最重要), ATC(Garg ICLR22 2201.04234), MaNo(Xie NeurIPS24 2405.18979), Agreement-on-the-line(Baek NeurIPS22 2206.13089), MaxLogit-pNorm(Cattelan UAI24 2305.15508), entropy/MSP/Mahalanobis。**指標**: AURC/risk-coverage/selective-acc@coverage。**差別化レバー**: ATC/ACは低精度域(50-70%)で崩れる既知弱点＝cross-session MIはこの帯域＝そこでclusterabilityの相対優位を示せれば差別化。**2大リスク**: (1)dispersity(Deng)≈silhouette なら「dispersityのEEG転用」に転落→差分厳密化 or 実測優位が必須(最大の新規性リスク), (2)Q5軽量ゲート空白は不在証明(EEG signal-quality/artifact-rejectionゲート文献の追加探索で確認要)。

**【GPT deep-research・2026-06-08, 4者目で収束＋重要追加】** 私のdeep-researchが取りこぼした先行をGPTが発見:
- **Q5の"空白"は埋まった＝Riemannian Potato(Barachant TOBI2013)/Riemannian Potato Field(Barthélemy IEEE TNSRE2019)** が共分散リーマン距離でonline EEG信号品質ゲート(悪ければfeedback停止)。＝**浅い共分散ゲートはclean gapでない**。Tomida(Active Data Selection IEEE TBME2015), FAAR artifact rejection(2605.12408), EEG Quality Index も同系。
- **最重要の差別化メッセージ（4者で最も鋭い）: 「signal-quality gate ≠ adaptation-worthiness gate」**。Potato/SQIはartifact-richを見る。だがcross-session driftで本当に問題なのは**artifactは少ないのにクラス構造が崩れて適応しても回復しないsession**。clusterabilityが**decoder-relevant structureの有無**を捉え、SQI/Potatoが見逃すこの種を当てれば強い。同振る舞いなら"品質管理の言い換え"に転落。
- **silhouette→適応判断の最近接アナロジー: van Heerden(2021, speaker diarization)** ＝per-file silhouetteで無教師domain adaptation/ハイパラ選択。差別化: diarizationはクラスタリング問題そのもの、我々は固定クラスのsupervised decoderへのpost-hoc gate。
- **gate→adapt先行(一般TTA): EATA(Niu ICML2022, 高entropyを適応除外)/HAMLET(Colomer ICCV2023, domain-shift detectorで適応制御)/AETTA(Lee CVPR2024)**。MI二段カスケード最近接=**Liu Temporal-OOD asynchronous MI(2605.01014)=rest/task gate→ID分類+OOD**（ただし二段目は分類/OODで適応でない）。MI trial-level reject=Ganeshkumar2017。BCI silhouette=Castillo-Garcia(IWAT2015, 特徴選択用)。
- **silhouetteの不安定性(Teng 2025 "When does silhouette work")**: クラス不均衡・弱分離・高次元で過大過小評価→**class-balance shift/one-class collapse/noisy sessionのstress test必須**。主張は「万能」でなく「経験的utility」に置く。
- **必須ベースライン3群**: ①confidence/OOD(MSP/entropy/Mahalanobis/energy) ②label-free精度推定(ATC/Deng-nuclear-norm/AGL/self-training-ensembles/AETTA) ③**EEG浅いゲート(Riemannian Potato/Field/Tomida/FAAR/EEG-QI)**。指標: gate品質(MAE/Spearman/Kendall vs adapt-benefit)＋運用(AURC/risk-coverage/selective-risk@coverage)＋コスト(latency/FLOPs/backward数/適応起動率)。
- **正直なfallback**: 勝てなければ「cross-session MI初の systematic adaptation-worthiness gate benchmark」まで主張を下げる。
- **含意（topomap議論と接続）**: 「worthiness≠quality」より、根拠図は**生パワー(=quality, Potatoが既にやる)でなく判別構造(=decodability)を見せるべき**＝判別信号トポマップが研究の芯に合う。
**【E5第1試行・2026-06-07, `lee2019_monitor_backbone.py` BACKBONE=eegnet】inconclusive**: EEGNet cross-subjectデコーダが **mean acc 60.5%**（F7閾値68%未達＝弱い、TCFormer用recipe流用が原因）。その弱デコーダ上で clusterability Spearman **-0.087(CI[-0.35,+0.19])・partial -0.10＝効かず**。だが出力空間予測器は弱く正(conf +0.31/negent +0.31/negmaha +0.34)。**解釈: backbone非依存の否定ではない**——「モニタは強デコーダでのみ効く」regimeに届いていない（TCFormerも58%弱デコーダでは+0.53/-0.26と揺れた）。EEGNet recipe(lr/epoch/F1)調整で≥68%に届かせて再判定が必要。結果JSON: docs/research_progress/260607_lee2019_monitor_eegnet.json。**注意: E4(モニタ vs 既存予測器)は、弱デコーダでは出力空間予測器が勝ちうる→必ず強デコーダ(TCFormer)で比較すること。**
