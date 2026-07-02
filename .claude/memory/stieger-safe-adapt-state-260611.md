---
name: stieger-safe-adapt-state-260611
description: Stieger2021 安全選択的適応の現状・確定値・走行中スクリプト・次の一手（2026-06-11 引き継ぎ）
metadata:
  type: project
---

cross-session MI の「2段構え安全選択的適応」検証の現状スナップショット（2026-06-11、新チャット引き継ぎ用）。進捗ノート本体は [docs/research_progress/260609_safe_selective_adaptation.md](docs/research_progress/260609_safe_selective_adaptation.md) §1-11。関連: [[thesis-plan-260606]] [[anisotropic-trust-shrinkage-idea]] [[feedback-dont-declare-dead]]。

## 設定（全てleak-free, 実データ）
- データ: **Stieger2021** 62被験者×6-11セッション。**前処理済みepochをキャッシュ済**＝`/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache/S*_epochs.npz`（X,y,sess）。**これで `prm.get_data()`(=1被験者233秒の律速)は二度と不要**。生成器 [stieger_dump_epochs.py](intentflow/offline/scripts/analysis/stieger_dump_epochs.py)。
- decoder: 凍結 Riemann-tangent-LDA（session-1で学習, P1=R1^-1/2, shrinkage=auto）。adapter: 教師なし EA再センタリング（session-jの平均共分散Rjへ）。gate: per-session adopt-adapted vs keep-source。
- cases: [260609_stieger_cases.npz](intentflow/offline/results/research_outputs/260609_stieger_cases.npz)（cs_/ca_=per-trial 0/1正誤, 60被験者524セッション）。signals: [260610_stieger_signals.npy](intentflow/offline/results/research_outputs/260610_stieger_signals.npy)（per-session true_d + 6 label-freeシグナル, 62被験者536セッション）。

## 確定した観測事実（数値）
- **SPREAD**: EA適応Δ mean **+7.85pp**、helped 76% / harmed **16%(87/536)** / worst **−10.3**。net-beneficialだが有害尾あり。oracle天井 +8.34pp。always-adapt=oracleの**93%**＝**平均精度はほぼ解けている、問題はworst-case**。
- **mirage（§9）**: 最小ラベル(k≤16)のEB-pooling gateは**全採用に退化**(harmed81,selectivity0,shrinkW0.07)。真のper-session選択性はk~64-96要、そこでpoolingのper-session LCB比優位は小。**安いコホート平均利得は買えるが安いper-session安全は買えない**。
- **有害の構造（§10）**: 時系列持続 P(harm|prev-harm)=0.39 vs base0.16(**2.4倍**)。被験者集中 84%が上位33%被験者・**29/60は無害**。subjMean veto(m=8)で harmed 81→34 / +6.05pp / 1/4ラベル。だが強安全(harmed数件)には届かず。
- **ICC=0.468**: 有害Δ分散の約半分が被験者trait（被験者間SD6.39 / 被験者内SD6.81）。**7/62人が平均で損**。被験者丸ごとveto天井=harmed 87→56（残56は被験者内セッション残差）。
- **E-A label-free veto（make-or-break, 棄却寄り）**: 6シグナル(H/overrule/conf_drop/entropy/dispersity/riemann)単独AUROC全て**<0.62**、融合LOSO**0.66**。**Lee2019のH ρ=−0.72はあの極端Riemann崩壊regime固有でStiegerに転移せず**(ρ+0.28)。→ **出力空間label-freeシグナルで有害尾を当てるのは決定打にならない**。スクリプト [stieger_probs_dump.py](intentflow/offline/scripts/analysis/stieger_probs_dump.py)（キャッシュ読込版）, 結果 [260610_stieger_signals_auroc.json](intentflow/offline/results/research_outputs/260610_stieger_signals_auroc.json)。
- **scalar-α縮約（route1 unfreeze実測）**: α=1:+7.23/worst−9.7 → α=0.5:+4.09/worst−4.2 → α=0.3:+2.34/worst−4.2/harmed21。**worstを締めると利得も等しく半減**＝scalarはトレードオフ。covariate-driftゲートは有害/有益を分離しない。

## 文献・新規性（確定, judge済）
- 直接競合（同Stieger）= **Wimpff arXiv:2502.06828**（unsupervised EA+AdaBN OTTA, per-session FT, **harmed分布を報告せず・gate無**）。引用値 source~74%LR/best 78.8±12.6%。EDAPT 2508.10474 も縦断personalization(veto無)。
- 先取り済（headlineにしない）: adopt-vs-keep LCB=c-value(Trippe2021); never-below=SPIBB; cohort-pooled go/no-go=Bayesian basket trials; **scalar partial-α geodesic recenter=GOPSA NeurIPS24 + SPDIM ICLR25 + TTN**; BFT2026; T-TIME2024; anytime-valid TTA monitor=Monitoring-Risks-in-TTA 2507.08721+POEM。
- **修正必須**: ATTA引用は ICCV'21でなく **Gui/Li/Ji ICLR'24(SimATTA)**。安全は「1−δ確率的」と明記(ハード床でない)。診断をリードに。venue=TNSRE/JNE現実的・NeurIPS/ICLRはworkshop。

## 次の一手（確定・承認待ちで未実行）
**[[anisotropic-trust-shrinkage-idea]] の決定実験を走らせる**。スクリプト [stieger_anisotropic_align.py](intentflow/offline/scripts/analysis/stieger_anisotropic_align.py) は**書き上げ未実行**（ユーザーが新チャット移行のため保留）。キャッシュから数分・GPU不要。勝利条件: aniso が harmed≪EA(87) かつ meanD≫scalar-α(+4.09)。理想は meanD>EA(+7.85)＆worst>−4 のPareto改善。棄却なら「害と利得は同方向」→FUSE-VETO＋α縮約の妥協に戻る。

## 図（ゼミ用, 生成済）
[docs/research_progress/ゼミ資料/](docs/research_progress/ゼミ資料/) に 260610_F1_spread / F2_mirage / F3_frontier。生成器 [viz_stieger_makeorbreak.py](intentflow/offline/scripts/viz/viz_stieger_makeorbreak.py)。

## エンジニアリング落とし穴
- **Stieger律速は計算でなく `prm.get_data()`=233秒/被験者**。必ず上記epochキャッシュから読む。
- 重い解析は**被験者ごとチェックポイント保存＋resume**（落ちても続き）。`pkill -f <name>` は自分のシェルも巻き込む（grep -v 自PID）。
- Workflowスクリプトのパースは**長いschema description文字列内の記号で壊れる**ことがある→descriptionを削るかASCIIのみに。成功した土台スクリプトをコピーして中身だけ差し替えると確実。
