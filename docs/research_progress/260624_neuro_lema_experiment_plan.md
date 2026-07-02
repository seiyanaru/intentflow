# Neuro-LEMA 実験計画 — 2026-06-24

## 結論

次に試す本線は **Neuro-LEMA: sensorimotor-constrained longitudinal meta-adaptation** とする。

ただし、いきなりmeta-adapter本体を実装しない。最初の実験は **E1: neuro feature baseline** であり、μ/β帯域・sensorimotor ROI・lateralizationに基づく特徴空間が、既存のall-channel Riemann/EA baselineに対して大きく劣化しないかを確認する。

このE1を通らなければ、Neuro-LEMA本体には進まない。

## 背景と方針転換

これまでの安全な選択的適応は、主に次の構造だった。

```text
既存adapter(EA/DA-DC等)
  ↓
label-free signal / LCB / EB / gate
  ↓
採用 or 棄却
```

しかし、Stiegerでは「危険なsessionだけを安く見分ける」方向が成立しなかった。特に、label-free vetoは穏やかな害を十分に捉えられず、異方的trust縮約も「危険方向だけ切る」形にはならなかった。

したがって、次はgateではなく、

```text
適応器そのものを、MI神経生理に沿った自由度に制約する
```

という方向へ移る。

## 主仮説

cross-session EEG-MIのnegative transferは、単なる全体covariance driftではなく、task-relevantなsensorimotor μ/β rhythmと、session/artifact由来のnuisance driftを区別できないことから起きる。

そのため、全脳・全帯域を一律に合わせるadapterより、

- μ/β帯域を明示的に分ける
- C3/C4/Cz周辺のsensorimotor ROIを優先する
- 左右手MIではlateralization signを壊さない
- rest/both-handsではbilateral ERD強度を保存する

ような制約付きadapterの方が、平均utilityを保ちながらlower-tail riskを下げられる可能性がある。

## 実験ステージ

### E1. Neuro feature baseline

目的：神経生理特徴空間がbaselineとして成立するか確認する。

使用データ：

```text
Stieger2021
source = session 1
target = session 2以降
native task = LR / UD / 2D
prefix m = 32
evaluation = trial 65以降固定
```

比較するfeature config：

```text
broad_all60:
  8-30Hz, all 60 channels

broad_sensorimotor21:
  8-30Hz, FC/C/CP sensorimotor ROI 21 channels

fb_all60_mu_beta:
  μ 8-13Hz + lowβ 13-20Hz + highβ 20-30Hz, all 60 channels

fb_sensorimotor21_mu_beta:
  μ 8-13Hz + lowβ 13-20Hz + highβ 20-30Hz, sensorimotor ROI 21 channels
```

各feature configで比較するadapter：

```text
source
prefix_ea
full_ea diagnostic
```

主指標：

```text
U = subject-balanced mean Δacc
R10 = -LCVaR@10%
P(Δ < -5pp)
q05
```

判定：

```text
fb_sensorimotor21_mu_beta が broad_all60 / broad_all60+EA に大きく負けるなら、Neuro-LEMA本体へ進まない。

目安:
  U が plain broad_all60 prefix_ea から -3pp以内
  または R10 が明確に改善
```

E1は手法の成功実験ではなく、次へ進むための生存確認である。

### E2. Generic LEMA

神経制約なしの軽量meta-adapterを作る。

```text
z' = z + U V^T z + b
```

入力はtarget prefix summary：

```text
source-target feature mean shift
covariance shift
prediction entropy / confidence
prefix bootstrap instability
```

目的は、longitudinal episodeから「適応写像を学習する」こと自体にheadroomがあるかを見ること。

### E3. Neuro-LEMA

Generic LEMAに神経生理制約を入れる。

```text
Raw EEG
  ↓
μ/β filter-bank covariance
  ↓
sensorimotor ROI + lateralization features
  ↓
target prefix summary
  ↓
meta-adapter g_phi
  ↓
constrained low-rank correction
  ↓
classifier
```

制約：

```text
μ/β帯域の変換を優先
sensorimotor ROI以外のadapter normを抑制
左右手ではlateralization sign collapseを罰する
rest/both-handsではbilateral ERD強度の過剰変形を罰する
prefix bootstrapで不安定な変換を弱める
```

### E4. Ablation

```text
A0: source-only
A1: plain EA
A2: generic LEMA
A3: band prior only
A4: band + sensorimotor ROI
A5: band + ROI + lateralization regularizer
A6: full Neuro-LEMA
```

ここで、改善が「meta-adaptation」由来なのか、「neuro prior」由来なのかを切り分ける。

## DA-DCの位置づけ

DA-DCは本線の構成要素にしない。

理由は、DA-DCを混ぜると、勝った理由がDA-DCなのか、神経生理制約なのか、meta-adapterなのか分からなくなるため。

扱いは以下に限定する。

```text
primary baseline:
  source-only
  plain EA
  filter-bank Riemann
  generic LEMA

secondary baseline:
  DA-DC, same protocolで再現可能な場合のみ
```

## 成功条件

最低成功：

```text
Neuro-LEMA:
  U >= 0.95 * U_plain_EA
  かつ R10を20%以上削減
```

強い成功：

```text
Neuro-LEMA:
  U >= U_plain_EA + 1pp
  かつ R10悪化なし
```

論文化条件：

```text
Stiegerで成功
Lee2019_MIで同方向
ablationでneuro priorの寄与が残る
topomap / lateralization解析で改善がsensorimotor μ/βに由来することを示す
```

## 直近の実装

1. `stieger_neuro_band_cov_cache.py`
   - raw MATから task-aware band-wise covariance cache を作る。
   - bands: 8-30, 8-13, 13-20, 20-30 Hz。

2. `stieger_neuro_feature_baseline.py`
   - broad all-channel / sensorimotor ROI / filter-bank featureを比較する。
   - source / prefix EA / full EA diagnosticを同一評価suffixで測る。

3. E1 summaryを見て、Neuro-LEMA本体へ進むか判定する。

4. `stieger_neuro_feature_complementarity.py`
   - E1 summaryから broad branch / neuro branch のsession-level complementarityをoracle診断する。
   - 結果メモ: `docs/research_progress/260624_neuro_lema_e1_e1b_results.md`

5. `stieger_neuro_branch_selector.py`
   - E2: target prefixのlabel-free signalだけで branch selection headroomを回収できるかをnested subject評価する。
   - 結果メモ: `docs/research_progress/260624_neuro_lema_e2_branch_selector_results.md`
   - 初回結果はweak/mostly negative。generic label-free selectorの深掘りより、fixed broad+neuro representation または少数ラベルbranch calibrationへ進む。

## 現時点の新規性

弱い主張：

```text
μ/βとC3/C4を使った。
```

これは古い。

強い主張：

```text
cross-session EEG-MI TTAにおいて、MI神経生理に基づいてadapterの自由度を制約し、target prefixからsession-specificな補正写像を生成する。
```

この主張にするには、単に性能を出すだけでなく、lateralization collapseやsensorimotor μ/β relianceの解析が必要である。

## 2026-06-27 update: E5a class-conditional geometry audit

E2/G1a/E4aの結果を受け、次の問いを検証した。

```text
target prefixの無ラベルglobal shiftではなく、
class-conditional geometry が branch gap / EA delta / fusion failure を説明しているのではないか？
```

結果メモ:

```text
docs/research_progress/260627_class_conditional_geometry_e5a_results.md
```

結論:

```text
Partial pass for mechanism, fail as a safety/gating signal.
```

守る:

```text
class-conditional geometry は branch performance の違いに関係している。
primary pooled で branch gap は class-conditioned LOSO rho=0.380、
global unlabeled geometry は rho=0.040。
```

撤回:

```text
class-conditional geometry を測れば unsafe fusion/harm を十分に検出できる。
primary pooled の fusion harm5 は best univariate AUROC=0.617、
class-conditioned LOSO AUROC=0.601 に留まる。
```

したがって、次は gate ではなく **E5b0: source-side longitudinal metric pilot** に限定する。

```text
training subjects の multi-session labels だけを使って、
same-class cross-session を近づけ、
different-class margin を保つ metric / block weighting を学習する。

held-out subject は全sessionをmeta-trainingから除外し、
S1 labelsだけでheadを作り、
S2+ target labelsなしで評価する。
```

最初はdeep backboneではなく、tangent feature上の軽量metricで試す。

判定:

```text
minimum:
  E4a equal posterior + prefix-EA に対して
  primary LR+UD +1pp以上, CI下限 > 0
  かつ R10悪化なし

strong:
  +2pp以上, CI下限 > 0
  かつ fusion harm5 / R10 を悪化させない

stop:
  E5b0がE4a equal posteriorに勝てない、
  またはlower-tailが悪化するなら、
  zero-target-label adaptationをaccuracy-first本線として続けない。
```

## 2026-06-27 update: E5b source-side longitudinal feature selection

E5b0/E5b1の結果メモ:

```text
docs/research_progress/260627_longitudinal_feature_selection_e5b_results.md
```

結論:

```text
Accuracy breakthrough.
Safety is improved vs E4a when measured relative to outer-best-single,
but strict no-harm/safe adaptation is still not solved.
```

最初の diagonal metric scaling はLDAで no-op だった。LDAは可逆な対角スケーリングに
ほぼ不変なので、この方向は捨てる。

代わりに、他被験者の multi-session labels から class-stable feature score を作り、
上位 q% の特徴次元だけを残す source-side feature selection を試した。

```text
longitudinal score =
  (source class separation + target class separation)
  /
  (same-class source-target drift + source within-class variance + target within-class variance)
```

held-out subject は全sessionを score 学習から除外し、S1 labelsだけでLDAを学習、
S2+ は target labelなし、prefix EAのみで評価した。

主結果:

```text
E4a equal posterior + prefix-EA:
  primary acc 65.325
  gain vs outer best single +1.509
  R10 loss 11.241
  P(loss<-5pp) 16.15%

E5b longitudinal outer_best_fraction:
  primary acc 67.734
  gain vs E4a equal +2.409 [1.532, 3.366]
  gain vs outer best single +3.919 [3.061, 4.818]
  R10 loss vs outer best single 10.589
  P(loss<-5pp) 12.55%
```

source-only Fisher top-kも改善したが、longitudinal score はそれをさらに上回った。

```text
source-only outer_best_fraction:
  primary acc 67.002
  gain vs E4a equal +1.677

longitudinal outer_best_fraction:
  primary acc 67.734
  gain vs E4a equal +2.409

direct paired diff:
  longitudinal - source-only = +0.732 [0.199, 1.264]
```

したがって、研究の本線を次へ更新する。

```text
旧:
  target prefix label-free signalでsession/branch/channelを選ぶ。

新:
  training subjects の longitudinal labels で不安定特徴次元を落とし、
  held-out subject はtarget labelなしでprefix-EA + fixed fusionする。
```

現時点の主張候補:

```text
Source-side longitudinal labels identify class-stable EEG-MI feature subspaces
that improve the zero-target-label cross-session risk-utility frontier.
```

次に必須:

```text
V1 random top-k baseline
V2 score ablation
V3 second multi-session MI dataset
V4 risk-utility frontier plot
```

ここまで通れば、修士研究の主軸は

```text
safe selective adaptation
```

から

```text
source-side longitudinal subspace selection for zero-target-label cross-session EEG-MI
```

へ正式に切り替える。

## 2026-06-27 update: V1/V2 selection ablation completed

V1/V2結果メモ:

```text
docs/research_progress/260627_selection_ablation_v1_v2_results.md
```

結論:

```text
E5bの主張は強化された。
random top-kは大きく負ける。
source-only / separation-only top-kは改善するが、full longitudinal scoreに有意に負ける。
```

Primary LR+UD:

```text
q1.00 full:
  acc 65.325

random q0.25:
  acc mean 61.518
  gain -3.807

random q0.10:
  acc mean 58.881
  gain -6.444

source-only outer best:
  acc 66.777
  gain +1.452

sep-no-drift outer best:
  acc 66.805
  gain +1.480

longitudinal outer best:
  acc 67.734
  gain +2.409
```

Direct paired comparisons:

```text
longitudinal - source-only:
  primary +0.957 [0.445, 1.468]

longitudinal - sep-no-drift:
  primary +0.929 [0.363, 1.462]

longitudinal - target-only:
  primary +1.746 [1.090, 2.400]
```

守る主張:

```text
単なる次元削減ではない。
source-only discriminabilityでも全部は説明できない。
同一クラスがセッションを跨いでどれだけ動くかを罰する
longitudinal stability term が追加の改善を生む。
```

修正する主張:

```text
gainの全てがlongitudinal stability由来ではない。
大きな部分はtop-k subspace restrictionによる正則化であり、
longitudinal stabilityはその上に乗る追加効果である。
```

次に進む:

```text
V3 second multi-session MI dataset
V4 risk-utility frontier figure
V5 selected feature / topomap interpretation
```

## 2026-06-27 update: V4/V5 frontier + interpretation completed

V4/V5結果メモ:

```text
docs/research_progress/260627_frontier_interpretation_v4_v5_results.md
```

Artifacts:

```text
intentflow/offline/scripts/analysis/stieger_longitudinal_frontier_and_interpretation.py
intentflow/offline/results/research_outputs/260627_stieger_longitudinal_frontier_interpretation/
```

結論:

```text
source-side longitudinal subspace selection は平均精度改善としては強い。
ただし safe adaptation を解いたとは言えない。
paper story は safety claim ではなく、
zero-target-label cross-session EEG-MI の source-side longitudinal feature selection として立てる。
```

Primary LR+UD:

```text
E5b longitudinal outer best:
  acc 67.734
  gain vs E4a outer-best single +3.919
  R10 loss 10.589
  P(gain < -5pp) 12.55%
  q05 gain -8.824

E4a equal posterior + prefix-EA:
  acc 65.325
  gain +1.509
  R10 loss 11.241
  P(gain < -5pp) 16.15%
```

LR:

```text
longitudinal q0.10:
  acc 70.431
  gain +4.079
  R10 loss 8.562
  P(gain < -5pp) 11.2%
```

UD:

```text
longitudinal q0.25:
  acc 64.848
  gain +3.855
  R10 loss 12.359

UDはまだrisk-utility trade-offが残る。
```

Feature interpretation:

```text
LRのlongitudinal selected subspaceはC/CP中心。
filterbankではhigh betaが多い。
これはmotor imagery physiologyと整合的。

一方UDはposterior/frontal参加も強く、脳科学的説明は弱い。
UDまで過剰主張しない。
```

次に進む:

```text
V3 second multi-session MI dataset
これが再現すればpaper候補。
再現しなければStieger-specific discovery + fragility/negative analysisとして畳む。
```

## 2026-06-27 update: V3 Lee2019 second dataset completed

V3結果メモ:

```text
docs/research_progress/260627_lee2019_v3_results.md
```

Artifacts:

```text
intentflow/offline/scripts/analysis/lee2019_longitudinal_selection_pilot.py
intentflow/offline/results/research_outputs/260627_lee2019_longitudinal_selection_pilot54/
```

Lee2019 full n=54:

```text
full_q1p00:
  acc 69.676

source_only_q0.25:
  acc 71.852
  gain vs full +2.176
  CI [+0.555, +3.866]
  P(gain < -5pp) 7.4%

longitudinal_q0.10:
  acc 71.528
  gain vs full +1.852
  CI [-0.278, +3.935]
  P(gain < -5pp) 20.4%

random_q0.25:
  gain -6.209

random_q0.10:
  gain -9.470
```

Critical interpretation:

```text
第二データセットでは source-side subspace selection は再現した。
しかし longitudinal same-class drift penalty が本体という主張は再現しない。

Stieger:
  longitudinal > source-only

Lee2019:
  source-only q0.25 >= longitudinal
```

修正する研究軸:

```text
旧:
  source-side longitudinal subspace selection

新:
  source-side stable/discriminative subspace selection
  for zero-target-label cross-session EEG-MI
```

守る主張:

```text
target labelsなしで、source側ラベル構造からRiemann tangent subspaceを選ぶと、
full featureより改善し、random top-kには大きく勝つ。
```

撤回または弱める主張:

```text
longitudinal drift penalty is universally necessary.
```

次にやるべきこと:

```text
T1: Stieger + Lee unified ablation table
T2: when longitudinal beats source-only の機構分析
T3: method namingを longitudinal 中心から source-side subspace 中心へ変更
```

## 2026-06-27 update: unified Stieger/Lee ablation table

Unified table:

```text
docs/research_progress/260627_unified_stieger_lee_ablation.md
```

最終的な現時点の結論:

```text
cross-datasetに守れるのは source-side subspace selection。
longitudinal drift penalty はStiegerでは効くが、Lee2019ではsource-onlyに負ける。
```

したがって研究タイトル/主張は以下へ修正する:

```text
Source-Side Tangent Subspace Selection
for Zero-Target-Label Cross-Session EEG Motor Imagery
```

longitudinalは主役ではなく、条件付きにする:

```text
When does longitudinal stability help?
```

## 2026-06-27 update: next-experiment decision after V3

Detailed decision memo:

```text
docs/research_progress/260627_next_experiment_decision_after_v3.md
```

次の最優先実験:

```text
E6: source-side nested model selection
```

理由:

```text
Stiegerでは longitudinal q0.10 が強い。
Lee2019では source-only q0.25 が強い。

このままだと「datasetを見てbestを選んだ」post-hoc批判を受ける。
したがって、held-out target subjectを除外したsource subjectsだけで、
family/qを選ぶnested検証が必須。
```

E6 candidate set:

```text
source-only q0.25
source-only q0.10
longitudinal q0.25
longitudinal q0.10
sep-no-drift q0.25
sep-no-drift q0.10
full q1.00
```

E6 selection rule:

```text
source-validation mean gainを最大化。
ただし P(gain < -5pp) <= 0.20 を満たすこと。
満たすcandidateがなければ full q1.00 にfallback。
```

E6 pass criteria:

```text
Stieger and Lee2019 both:
  nested-selected method beats full by > +1.0 pp
  CI lower bound is not strongly negative
  R10 loss is not worse than best fixed source-side candidate by > 2 pp
```

次点の機構検証:

```text
E7: Stieger session-depth ablation

K=2,3,5,all source sessionsでmetricを学習し、
longitudinal - source-only がKとともに増えるかを見る。
```

E7の狙い:

```text
Lee2019でlongitudinalがsource-onlyに負けた理由が、
2 sessionsしかなくdrift推定が不安定だからなのかを検証する。
```

## 2026-06-27 update: E6 nested selection completed

E6 result memo:

```text
docs/research_progress/260627_e6_nested_selection_results.md
```

Artifacts:

```text
intentflow/offline/scripts/analysis/source_side_nested_selection_from_records.py
intentflow/offline/results/research_outputs/260627_source_side_nested_selection_from_records/

intentflow/offline/scripts/analysis/lee2019_exact_nested_subspace_selection.py
intentflow/offline/results/research_outputs/260627_lee2019_exact_nested_subspace_selection_12/
```

E6-fast result:

```text
Stieger:
  source-side selection chooses longitudinal.
  condition-specific:
    LR -> longitudinal q0.10
    UD -> longitudinal q0.25
  gain +2.409 pp
  CI [+1.540, +3.324]

Lee2019:
  source-side selection chooses source_only q0.25
  gain +2.176 pp
  CI [+0.509, +3.843]
```

Critical caveat:

```text
Stieger E6-fast uses existing LOSO records.
It is not exact double-LOSO.
```

Lee2019 exact nested subset n=12:

```text
nested risk selection:
  gain +4.167 pp
  CI [+0.938, +7.188]

fixed source_only q0.25:
  gain +5.313 pp
  CI [+3.021, +7.708]
```

Interpretation:

```text
source-side validation can identify useful dataset/task-level candidates.
But per-target-subject candidate selection may overfit source-validation noise.

Therefore, do not make nested per-subject candidate selection the main method yet.
The cleaner primary method is source-side tangent subspace selection,
with source_only q0.25 as robust cross-dataset baseline and longitudinal as
Stieger/multi-session conditional variant.
```

Next:

```text
E7 Stieger session-depth ablation
```

Reason:

```text
We need to test whether longitudinal helps only when enough source longitudinal
sessions are available.
```

## 2026-06-27 update: E7 session-depth ablation completed

E7 result memo:

```text
docs/research_progress/260627_e7_session_depth_results.md
```

Artifacts:

```text
intentflow/offline/scripts/analysis/stieger_session_depth_ablation.py
intentflow/offline/results/research_outputs/260627_stieger_session_depth_ablation/
```

Primary result:

```text
K=2:
  longitudinal q0.10 gain +2.209
  source-only q0.25 gain +1.385
  long q0.10 - source q0.25 = +0.824

K=3:
  longitudinal q0.10 gain +1.978
  source-only q0.25 gain +1.399
  long q0.10 - source q0.25 = +0.579

K=5:
  longitudinal q0.10 gain +2.305
  source-only q0.25 gain +1.311
  long q0.10 - source q0.25 = +0.994

K=all:
  longitudinal q0.10 gain +2.357
  source-only q0.25 gain +1.313
  long q0.10 - source q0.25 = +1.044
```

Critical interpretation:

```text
longitudinal advantage does not emerge only at large K.
K=2 already works on Stieger.
Therefore Lee2019でlongitudinalが負けた理由を
「2 sessionsしかないから」とだけ説明するのは無理。
```

LR split:

```text
long q0.10 - source q0.25:
  K=2   +1.249
  K=3   +1.258
  K=5   +1.511
  K=all +1.495

LRではlongitudinalが安定してsource-onlyを上回る。
```

UD split:

```text
long q0.25 - source q0.25:
  K=2   +0.225
  K=3   +0.483
  K=5   +0.744
  K=all +0.680

UDは弱いが、allではわずかにpositive。
```

Decision:

```text
longitudinalを主タイトルにしない。
source-side tangent subspace selectionを主軸にする。
longitudinalはStieger/LRで効くconditional variantとして扱う。
```

Next:

```text
E8 selected-feature geometry:
  Stieger LR longitudinal q0.10
  Stieger LR source-only q0.25
  Lee2019 source-only q0.25
  Lee2019 longitudinal q0.10

これらの選択特徴がsensorimotor covarianceにどう寄るかを見る。
```
