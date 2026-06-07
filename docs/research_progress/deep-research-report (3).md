# EEG-MI 精度重視戦略の再設計レビュー

## エグゼクティブサマリ

純粋な **freeze + posterior/prior correction** 路線は、あなたの内部オラクル結果の時点で主戦場から外れています。BCIC2a では **top2 天井は +11.84pp** ある一方、**class-bias 補正天井は +3.36pp** に留まり、実測 best も **+1.23pp**、multi-seed では **+0.26〜0.49pp** です。つまり「大きく上げる」は **出力補正ではなく、少なくとも一部の表現適応か、top2 を取りに行く賢い reranking** が必要です。fileciteturn3file0L3-L3 fileciteturn4file0L3-L3

外部 reality check でも、**single-trial に近い厳しい online 設定で安定して大きい gain を出す手法は多くない**一方、制約を少し緩めると **T-TIME は +2.9〜6.1pp、5-model ensemble では +5.8〜7.2pp** を出しています。逆に **T3A は EEG-MI で -5〜-9pp 級に壊れる**ので、「凍結特徴に prototype を乗せればよい」は成立しません。citeturn11view1turn12view0turn12view1turn11view2

したがって、いま賭けるべき本命は **alignment-first の selective minimal feature adaptation** です。言い換えると、**常時は軽く、難例だけ小さく更新する**。これなら BFT 2026 の「lightweight EEG-TTA」と真正面からぶつかりつつも、**純 no-BP より精度で上回る余地**があります。BFT の accessible text だけで確実に言えるのは、BFT がすでに **EEG 向け・backprop-free・sample-wise transformations・learning-to-rank aggregation** を占有していることです。そこへ「別の no-BP 集約器」を足すだけでは新規性が弱いです。citeturn39academia0turn0academia2turn19view0

## いま既に否定されている設計空間

あなたの内部結果が一番強い制約です。BCIC2a では **source top1 82.72%** に対して、**top2 oracle 94.56%**、すなわち **+11.84pp** の未回収余地があります。しかし **per-class bias 補正の天井は +3.36pp** に留まり、**static global prior 補正は Δ0.00** です。HGD でも **top2 +5.00pp**、**bias +3.18pp**、BCIC2b では 2-class のため top2 は自明で、**bias 天井は +1.73pp** です。これは「正解が top2 に入っている誤りは多いが、それを class-bias だけでは 1 位にできない」ことを意味します。fileciteturn3file0L3-L3

同じことを実測が裏づけています。現在の DC 系列では、single-seed の最良変種でも **+1.23pp**、seed を入れると **+0.26〜0.49pp** まで縮みます。さらに **L3 model-state commit は no-op** で、BCIC2a では **commit 回数 0 / 33 / 111 / 150 で mean_acc がすべて 83.56556%** と一致しています。改善の主成分は L1+L2 であり、L3 は少なくともいまの設計では価値を生んでいません。fileciteturn4file0L3-L3 fileciteturn5file0L3-L3

ここから出る結論は単純です。**posterior/prior correction を磨いても「大きく上げる」は難しい**。これは推測ではなく、あなたのオラクル上界と実測の両方が示しています。したがって、現時点で commitless correction を捨てた判断は妥当です。fileciteturn3file0L3-L3 fileciteturn4file0L3-L3

一方で、「なら feature adaptation に全振りすればよい」とも言えません。repo の TCFormer Hybrid / TTT 系は、少なくとも現実装・現設定では **TCFormer base に負けています**。README 上では、Hybrid は **BCIC IV-2a で -1.15pp、2b で -1.91pp、HGD で -13.66pp** です。つまり **feature adaptation は必要だが、雑に入れると壊れる**。ここが次の設計で最重要のコーナーケースです。fileciteturn15file0L3-L3

## EEG-MI で精度を大きく上げた手法の現実的な相場

以下は、**EEG-MI cross-session / cross-subject / online TL** に関係し、かつ現時点で精度改善の根拠を比較的はっきり追える手法群です。NR は、今回アクセスできた一次ソースからは数値を確定できなかったものです。

| 手法 | 論文 | 設定 | online / batch | freeze / BP | 軽量性 | 報告 gain | コード |
|---|---|---|---|---|---|---|---|
| Wimpff OTTA | *Calibration-free online test-time adaptation for electroencephalography motor imagery decoding* — Martin Wimpff, Mario Döbler, Bin Yang, 2024, IWCBCI, arXiv:2311.18520 | BCIC IV 2a/2b、cross-session / cross-subject / continual cross-subject | single-instance, buffer 32 | alignment + AdaBN は no-BP、EM は BP | BaseNet 軽量、buffer 必須 | accessible textで確実なのは **continual 2a で almost +2pp**。cross-session / cross-subject は source 超えだが exact pp は今回の可引用範囲では欠落 | 有 citeturn2view0turn13view1turn44academia3 |
| T-TIME | *T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs* — Siyang Li et al., 2024, arXiv:2412.07228 | BNCI2014001/4002/5001、cross-subject | online, **test batch size 8** | freeze ではない。**全パラメータ更新** | EEGNet 基盤。CPU で pre-inference 約 **5.6ms**、post-update **34/61/55ms per model** | source EEGNet に対し **+2.92 / +5.65 / +6.07pp**。5-model ensemble では **+5.78 / +7.20 / +6.52pp** | 有 citeturn11view0turn11view1turn12view0turn12view1turn11view2turn34view0turn34view3 |
| Dual-Stage Alignment + Self-Supervision | *Online Adaptation via Dual-Stage Alignment and Self-Supervision for Fast-Calibration Brain-Computer Interfaces* — Sheng-Bin Duan et al., 2025, arXiv:2509.19403 | 5 public datasets, 7 decoders, unseen subjects | **single online trial update** | BP 要。decoder 更新 | abstract上は軽量寄りだが exact latency/params NR | **MI で平均 +3.6pp**, SSVEP で +4.9pp | 未確認 citeturn19view0 |
| SDDA | *Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework* — Zhengqing Miao et al., 2022, arXiv:2202.09559 | BCIC IV 2a/2b、cross-session | offline DA | BP 要 | 軽量ではない | vanilla EEGNet / ConvNet に対し **2a で +15.2 / +10.2pp、2b で +5.5 / +4.2pp** | 未確認 citeturn45academia0 |
| EEG-DG | *EEG-DG: A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification* — Xiao-Cong Zhong et al., 2023, arXiv:2311.05415 | BCIC IV 2a/2b、cross-subject DG | adaptation なし | N/A | inference-only | **81.79% / 87.12%** の強い非適応 baseline | 有 citeturn15academia0 |
| BFT | *Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based Brain-Computer Interfaces* — Siyang Li et al., 2026, arXiv:2601.07556 | 5 EEG datasets、MI + drowsiness | online sample-wise | **no-BP** | lightweight を主張 | accessible text では **exact pp を確定できず**。ただし EEG 向け no-BP TTA として 5 dataset の effectiveness/efficiency を主張 | 未確認 citeturn39academia0 |

この表から読める reality check は明確です。**「精度を大きく上げる」実例は存在する**。ただし、その多くは **batch size 8、複数モデル ensemble、全パラメータ更新、offline DA** のいずれかを許しています。逆に、**single-trial・strict lightweight・freeze 寄り**では、今のところ外部証拠はかなり弱くなります。Wimpff 2024 が single-instance OTTA で示したのは、少なくとも accessible text 上では **2a continual でほぼ +2pp** 規模です。citeturn13view1turn11view1turn12view0turn12view1turn19view0turn45academia0

だから「**+5pp を凍結 backbone + no-BP + single-trial online** で狙う」のは、現時点では**強い仮説**です。成立しうる条件は限られます。少なくとも、**(a) 2a のように top2 slack が大きい、(b) source 特徴がそこそこ使える、(c) alignment が効く、(d) hard trial だけに局所的な追加処理を入れる** の四つが必要です。HGD のように source が強く、top2 余地が **+5.00pp** しかない領域では、同じ方法で大きく上がる見込みは薄いです。fileciteturn3file0L3-L3 citeturn11view1turn12view0turn12view1turn19view0

## 既存法が埋めた空間と、まだ残る隙間

generic TTA 側の既存法も、すでにかなり埋まっています。**Tent** は BN affine と統計を entropy minimization で online 更新する完全 TTA の代表です。**CoTTA** は continual shift に対して weight-averaged / augmentation-averaged prediction と stochastic restore を導入します。**EATA** は reliable / non-redundant sample selection と Fisher regularization で forgetting を抑えます。**LAME** は出力だけを conservative に補正する parameter-free OTTA、**AdaNPC** は source memory と test memory を用いる non-parametric classifier、**TAST** は nearest-neighbor を使うが adaptation module を学習するため、軽量 freeze-only ではありません。citeturn25academia0turn25academia1turn26academia3turn26academia2turn32academia0turn27academia0

EEG-MI に引き寄せたとき、最重要なのは **generic prototype/memory 系がそのまま効かない** という事実です。T-TIME の EEG benchmark では、**T3A は source EEGNet より大きく悪化**しています。具体的には、BNCI2014001 で **73.52 → 68.23 (-5.29pp)**、BNCI2014002 で **72.61 → 63.67 (-8.94pp)**、BNCI2015001 で **71.68 → 64.88 (-6.80pp)** です。これは「naïve な prototype/classifier adjustment は EEG-MI の cross-subject では壊れやすい」ことのかなり強い実証です。citeturn11view1turn12view0turn12view1

ここに BFT が刺さります。accessible text の範囲で BFT が明示しているコアは、**sample-wise transformations**、**knowledge-guided augmentations または approximate Bayesian inference** による複数 score 生成、そして **learning-to-rank による aggregation** です。しかも EEG-specific、online、no-BP、lightweight を前面に出しています。したがって、**「複数変換をかけて score を集約する no-BP lightweight EEG-TTA」** という設計空間は、すでに BFT に占有されています。citeturn39academia0

そのため、新規性が立つ余地は次の三つに限られます。

| 残る隙間 | なぜ未解決か | 想定 gain | 差別化の核 |
|---|---|---:|---|
| **top2-aware local reranking** | output bias では top2 slack を回収できない一方、prototype 系は壊れやすい | 2a では **+1〜4pp** 程度、HGD/2b は小さめ | **oracle に基づき「top2 を 1 位にする」ことを明示目的化**する点。BFT は transformation aggregation、あなたは local geometry / neighbor consistency で top2 correction を狙う |
| **selective minimal feature adaptation** | 大 gain には feature adaptation が要るが、full BP は重く不安定 | 条件が良ければ **+3〜6pp** も視野 | **全試行ではなく hard trial だけ tiny adapter / head / BN を更新**。平均計算量で lightweight を維持 |
| **alignment-first hybrid routing** | EEG は alignment の有無で prototype 系の成否が変わるが、TTA では end-to-end に整理されていない | **+2〜5pp** の現実帯 | **cheap alignment path と expensive adaptation path の条件付き切替**。BFT より制御論が強く、pure freeze correction より精度余地が大きい |

この表の数値は **推論**です。実証ではありません。実証されているのは、**bias-only ceiling は低い**、**T3A は壊れる**、**T-TIME や dual-stage alignment+self-supervision は feature update を許すと gain が伸びる**、の三点です。fileciteturn3file0L3-L3 citeturn11view1turn12view0turn12view1turn19view0

結論だけ先に言えば、**純粋 no-BP / freeze-only のままでは新規性も精度も厳しい**です。BFT がある以上、そこにもう一つ aggregation を足しても論文として弱い。新規性を残したいなら、**oracle-top2 を使った局所 reranking** か、**条件付き minimal feature adaptation** に踏み込む必要があります。citeturn39academia0turn26academia2turn32academia0

## oracle を踏まえた賭け先の比較

ここは設計判断の中心です。比較対象は二つです。

| 路線 | 強み | 弱み | どの条件で成立するか | 私の判定 |
|---|---|---|---|---|
| **特徴固定 reranking** | no-BP、低レイテンシ、小メモリ。TCFormer 凍結に自然に載る | bias ceiling を超えるには local geometry が要る。naïve prototype は壊れる | BCIC2a のように **top2 slack が大きい** / alignment が効く / low-confidence 試行だけ再順位付け | **サブプランとして有望** |
| **minimal feature adaptation** | ceiling 自体を押し上げられる。T-TIME や dual-stage 系の外部証拠と整合 | 壊れやすい。repo の hybrid/TTT は既に失敗例あり | 更新対象を極小化し、**全試行ではなく難例だけ**、alignment を前段で入れる | **本命** |
| **pure BFT-style no-BP aggregation** | もっとも軽い。edge 適性が高い | BFT 2026 と空間が重なる。精度差別化も新規性差別化も弱い | BFT より明確に強い scoring source を出せる場合のみ | **ベースラインとしては良いが主張の軸には弱い** |

なぜ reranking 単独を本命にしないか。理由は内部オラクルと外部 negative result の組み合わせです。BCIC2a の **+11.84pp top2 ceiling** は魅力的ですが、これは「正解が 2 位にいる」だけで、「局所幾何が十分に整っている」ことまでは意味しません。しかも T-TIME 上では T3A が -5〜-9pp 壊れており、**global prototype / template の単純運用は失敗する**と見たほうが合理的です。したがって reranking は、**EA/RA や AdaBN で feature space をまず整えること**、**top1/top2 の局所支持が強いときだけ発火すること**、**global prototype ではなく local neighborhood で判定すること** が条件になります。これは成立しうるが、簡単ではありません。fileciteturn3file0L3-L3 citeturn11view1turn12view0turn12view1turn44academia3

feature adaptation を本命にする理由は、外部 evidence がそこにあるからです。T-TIME は **全パラメータ更新 + adaptive marginal distribution regularization** で **+2.9〜6.1pp**、dual-stage alignment + self-supervision は **single online trial 更新で MI 平均 +3.6pp** を出しています。逆に pure output correction は LAME 的に conservative で速い一方、あなたの内部 ceiling とも整合して大 gain にはつながりにくいです。citeturn11view1turn12view0turn12view1turn19view0turn26academia2

ただし、repo の TCFormer Hybrid / TTT 系が負けている以上、「feature adaptation に踏み込むべき」は「何でも更新せよ」ではありません。ここで必要なのは **更新対象の制限**です。たとえば BN / classifier head / tiny adapter / low-rank sidecar のどれかに限定し、しかも **uncertainty-triggered** にする。つまり **always update** ではなく **update-on-hard-cases** に変える。これなら精度改善のための feature adaptation を持ち込みつつ、平均レイテンシと壊れやすさを抑えられます。これは repo の 300–500 ms 目標とも整合します。fileciteturn15file0L3-L3

## TCFormer に載せる候補トップ

以下は、**TCFormer を中心に、軽量性と精度向上を両立しやすい順**で並べた候補です。expected gain は **推測**であり、先行論文が TCFormer 上で直接実証した値ではありません。

| 候補 | 実装像 | params / latency / adaptation cost | 想定 gain | 新規性 | 判定 |
|---|---|---|---|---|---|
| **alignment-first selective adapter** | online EA/RA + AdaBN を常時適用し、**低次元 adapter / head / BN affine だけ** hard trial 時に更新 | 追加 params は backbone の **ごく一部**。平常時はほぼ inference-only、難例だけ小規模 BP | **+3〜6pp** が狙い目。特に 2a | **中〜強**。BFT と違い、**平均軽量性を保ちながら feature ceiling を動かす** | **最有望** |
| **oracle-guided local reranker** | TCFormer 特徴を凍結し、EA/RA 後の feature 上で **top1/top2 だけ** kNN / local prototype / consistency で再順位付け | 追加 params ほぼ 0。latency は memory search 分だけ増加。BP なし | **+1〜4pp**。主に 2a、HGD/2b は小さい | **中**。ただし **BFT との差別化は「top2 correction 明示」**が必要 | **第二候補** |
| **BFT-style transformation ranker on TCFormer** | sample-wise transformations + score aggregation + ranking head を TCFormer 出力に追加 | no-BP、追加 params 小。latency は変換数に比例 | **+1〜3pp** 程度はありうるが不確実 | **弱**。BFT と空間が近すぎる | **ベースラインとしては必要、主題には不向き** |

この順位付けの理由は二つです。第一に、**TCFormer 自体が強い backbone**であり、内部資料では BCIC IV-2a / 2b / HGD で **84.79% / 85.54% / 93.8%** クラスのベース能力を持つと整理されています。強い backbone では、出力補正だけで大きく伸ばす余地は相対的に小さくなります。第二に、repo の online/active 目標は **300–500 ms** で、常時重い更新は許されません。だから **「平常時は軽く、難例だけ更新」** が設計的に筋が良いです。fileciteturn12file0L3-L3 fileciteturn15file0L3-L3

T-TIME の計算コストも、この方向を後押しします。T-TIME は Intel Core i5 CPU 上で、**pre-inference 5.3ms の IEA + 0.3ms の SML** に加え、**post-inference update が 34/61/55ms per model** です。これは 1-model なら現実的ですが、ensemble を増やすとそのままコストが増える。edge / Active BCI を考えると、**5-model ensemble で勝負するより、single-backbone + sparse tiny-update** の方が長期的に戦いやすいです。citeturn11view2turn34view3

## BFT を超えられるかと、最終推奨

ここは断定します。

**厳密に「freeze + no-BP + online single-trial + constant-cost/lightweight」のまま BFT を精度で明確に上回ることを、主戦略としては勧めません。**  
理由は三つです。  
第一に、BFT はすでに **EEG-specific / no-BP / lightweight / transformation aggregation** を主張しており、設計空間が近すぎます。第二に、あなたの内部 oracle は **output-side correction の ceiling が低い**ことを示しています。第三に、T3A の EEG negative result が示す通り、純粋な classifier/prototype-side 調整は壊れやすいです。citeturn39academia0 fileciteturn3file0L3-L3 citeturn11view1turn12view0turn12view1

**ただし、「平均計算量として lightweight」でよいなら、BFT を精度で上回る余地はあります。**  
その余地は、**selective minimal feature adaptation** にあります。具体的には、常時は EA/RA + AdaBN + frozen TCFormer で即時予測し、**不確実・top2 接戦・局所支持あり** のときだけ small adapter / head / BN affine を更新する。これなら **常時 no-BP ではない**ものの、**平均コストは低く、精度 ceiling を動かせる**。この点で BFT と差別化できます。根拠は、T-TIME と dual-stage alignment+self-supervision が、EEG-MI で gain を大きく出しているのが **feature update を伴う側**だからです。citeturn11view1turn12view0turn12view1turn19view0

したがって、私の最終推奨はこれです。

### 賭けるべき方向

**alignment-first の selective minimal feature adaptation**

### 理由

pure correction は内部 oracle で ceiling が低い。pure no-BP aggregation は BFT と被る。大 gain の外部 evidence は、制約を少し緩めた **feature update 側**に偏っている。だから、**軽量性は「ゼロ更新」ではなく「平均コストを抑える」ことで守る**のが正しいです。fileciteturn3file0L3-L3 citeturn39academia0turn11view1turn12view0turn12view1turn19view0

### 想定 gain

**実証ではなく推測**ですが、BCIC2a 主体なら **+3〜5pp** は十分に狙う価値があります。  
ただし、HGD は source がすでに高く、top2 余地が小さいため、同じ改善幅は期待しにくいです。dataset 依存性は強いはずです。fileciteturn3file0L3-L3

### 精度↔軽量のトレードオフ

- **strict lightweight** を守るほど BFT に近づき、新規性も gain も痩せる。  
- **sparse tiny-update** を許すと、平均計算量はまだ小さく保てる一方、精度上昇余地が一段増える。  
- 逆に **always full-model update** は、T-TIME 的には強くても、あなたの edge / Active BCI 制約と衝突します。citeturn11view2turn19view0

### 新規性の核

**「EEG-MI における top2 slack を、alignment 条件付き・難例限定の minimal feature adaptation で回収する」**  
これです。  
BFT の核は transformation aggregation。あなたの核は **oracle-guided selective adaptation** に置くべきです。fileciteturn3file0L3-L3 citeturn39academia0

### 最初の実験

1. **online EA/RA + AdaBN only** を TCFormer に載せ、BCIC2a/2b/HGD で single-trial source 比を測る。ここで reranking や adapter をまだ入れない。  
2. その上で、**uncertainty-triggered tiny adapter / head-only update** を追加し、always-update と比べる。  
3. 並行して、**top2-aware local reranker** を frozen 特徴上で実装し、**「no-BP だけでどこまで回収できるか」** を測る。  
この三つで、出力補正 ceiling、reranking ceiling、minimal feature adaptation ceiling を同一 backbone 上で切り分けられます。fileciteturn15file0L3-L3 fileciteturn6file0L3-L3

## 警告と限界

最大の罠は、**精度は出ても論文にならない**構図です。具体的には、  
**BFT に近い no-BP aggregation を追加して少し勝つだけ**、  
**BCIC2a single-seed だけで勝ったと言う**、  
**latency を ensemble や post-update を含めずに主張する**、  
この三つです。どれも reviewer が最初に刺してきます。citeturn39academia0turn11view2turn11view0

もう一つの罠は、**feature adaptation に踏み込んだ瞬間に repo の Hybrid/TTT と同じ失敗を再現する**ことです。したがって、「どこを更新するか」よりも「**いつ更新するか**」を先に決めるべきです。**triggered update** を設計しない feature adaptation は、現時点では再現性のある勝ち筋に見えません。fileciteturn15file0L3-L3

最後に限界を明記します。今回、**BFT 2026 と Wimpff 2024 の PDF/HTML から exact table 数値を完全には可引用形で回収できませんでした**。そのため、BFT の exact pp、Wimpff 2024 の cross-session / cross-subject の exact pp は **意図的に捏造せず、未確定のまま残しています**。一方で、BFT のコア設計、T-TIME の exact gains、dual-stage single-trial update の平均 gain、SDDA の大幅 gain、T3A の EEG-MI failure、そしてあなたの内部 oracle / repo diagnostics については、現時点で十分強い根拠があります。citeturn39academia0turn13view1turn11view1turn12view0turn12view1turn19view0turn45academia0