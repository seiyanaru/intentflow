---
name: anisotropic-trust-shrinkage-idea
description: 新規手法案「異方的trust縮約アライメント」— 固有方向ごとにラベルフリーtrustでEA縮約し利得保持と安全を両立（未検証・決定実験待ち）
metadata:
  type: project
---

cross-session MI 安全選択的適応の**新規手法案**（2026-06-11、未検証・決定実験書き上げ済み）。全検証を貫く論理から逆算した本命アイデア。状態は [[stieger-safe-adapt-state-260611]] 参照。

## 統合診断（なぜ既存が全部行き詰まったか）
4つの行き止まりが確定: ①ラベルゲート=mirage(全採用) ②label-free veto=AUROC0.66(弱) ③scalar-α縮約=利得と害が連動半減 ④被験者veto=半分しか取れず残差は被験者内ノイズ。
→ **核心の論理: 全手法が「間違った粒度」で安全を取りに行っている。** ①④はセッション/被験者粒度、③はスカラー粒度。**だがEAの害は"方向(固有モード)粒度"で生まれる**——EAは約60本の共分散シフト方向に判定器を動かすが、有害なのは誤推定された少数方向だけ。scalar-αが利得を半減させるのは「安全な利得方向まで巻き添えで縮約するから」。

## 手法: Anisotropic Trust-Shrinkage EA（異方的trust縮約）
EA再センタリングを固有方向に分解し、**信頼できない方向だけsourceへ縮約・信頼できる利得方向はfull EAのまま残す**。
- `M = R₁^(-1/2) Rⱼ R₁^(-1/2) = U diag(μ) Uᵀ`（μᵢ=1=方向iにシフト無し）
- 異方的基底点 `R(a) = R₁^(1/2) U diag(μ^{aᵢ}) Uᵀ R₁^(1/2)`、**方向ごと aᵢ∈[0,1]**（aᵢ=1=full EA, aᵢ=0=source）。whiten by R(a)^-1/2。
- **ラベルフリーtrust aᵢ**: (M)magnitude `aᵢ=exp(−β|log μᵢ|)`＝大移動(危険)方向ほどsource、bootstrap不要; (S)stability=試行bootstrapでμᵢのCVを測り不安定方向をsource。

## なぜ精度・安全・軽量・新規が同時に立つか
- **精度（むしろ上がりうる）**: EA mean+7.85は自分の有害尾(87セッション平均≈−4pp)に足を引かれている。異方的が有害方向だけ殺し利得方向を残せば **meanはEAを上回りうる**（scalar-αは利得方向も削るのでmean低下）。＝精度↑と安全の両立はこれが唯一の道。
- **安全（構成的下限）**: 危険方向 aᵢ→0=完全source＝方向ごとのsource床。確率的1−δでなく「誤推定方向に動かない」構成的保証。
- **軽量（EA以下）**: 一般化固有分解1回(tangent写像で既に計算)＋スカラー演算。SGD無・ラベル無・再学習無。状態はEAと同じ。
- **新規（文献空白の交差点）**: GOPSA/SPDIM/TTNは全部**スカラーα(全方向一律)**。**方向ごとのラベルフリーtrust縮約は誰もやっていない**。評価をharmed-count/worst-caseにするのも空白(全員mean-acc)。

## 論文の主張（negative→constructive）
「**適応の害は"方向的"現象である**。セッション粒度でもスカラー粒度でも安全は安く買えない(我々の3つのnegative)。**安全は方向粒度でラベルフリーtrustにより構成的に課すべき**」。c-value/SPIBB/basketはゲートの話、GOPSA/SPDIMはスカラーの話＝「方向粒度の安全」は誰のものでもない。

## リスクと決定実験（最優先・未実行）
- **最大リスク**: 「利得方向」と「危険方向」が同じなら異方的でも分離できずscalar-αと同結果（トレードオフ復活）。理屈でなく実測でのみ判明。
- **決定実験**: [stieger_anisotropic_align.py](intentflow/offline/scripts/analysis/stieger_anisotropic_align.py)（書き上げ済・未実行, キャッシュから数分・GPU不要）。source/EA/scalar-α{.3,.5,.7}/aniso-M{β}/aniso-S を536セッション横並び。**勝利条件**: harmed≪EA(87) かつ meanD≫scalar-α(+4.09)。理想 meanD>+7.85＆worst>−4。**棄却条件**: anisoがscalar-α同等→方向粒度仮説棄却→FUSE-VETO＋α縮約の妥協へ。
- これを走らせれば天才的発想が本物か今日中に確定する。**新チャットの第一手はこの実行・解釈**。
