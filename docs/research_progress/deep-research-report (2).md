# EEG-MI OTTA 研究計画 v0 と BFT 2026 の批判的レビュー

## エグゼクティブサマリ

判定は**小修正**です。理由は単純で、計画 v0 はすでに「大きい gain の新手法」から「何が効いて何が効かないかの理解」と「harm 制約つき評価」へ軸を切り替えており、その方向自体は内部証拠と外部文献の両方に整合しているからです。内部では L3 commit が no-op、L1 ceiling が 2a で +3.36pp、2b で +1.73pp、HGD で +3.18pp であり、しかも multi-seed の gain は +0.26〜+0.49pp に縮んでいます。これは「freeze + no-BP + memory/posterior correction だけで大きく上げる」という期待を支持しません。fileciteturn7file0L3-L3 fileciteturn8file0L3-L3 fileciteturn6file0L3-L3

ただし、**手法新規性はかなり弱い**です。BFT 2026 は EEG 向け no-backprop TTA として既に出ており、T3A・LAME・AdaNPC 系も「学習しない／軽くしか学習しない test-time 補正」の設計空間を埋めています。したがって、あなたの勝ち筋は C1（mechanistic negative result を含む理解）と C2（per-subject harm を主目的に据えた安全評価）であって、C3（Active BCI 評価）は**閉ループ感が弱いままだと主貢献になりません**。citeturn1view0turn3academia1turn3academia2turn47view0turn47view3

外部の reality check も二層です。**あなたの制約に近い領域**では Wimpff 2024 の continual cross-subject 2a が「almost a two percent increase」で、あなたの内部結果も low single-digit 未満です。一方で、**制約を緩めた online EEG TL**では T-TIME が binary MI ベンチで +2.9〜+6.1pp を出しています。つまり「EEG で +5pp は存在する」が、「あなたの admissible operation class では期待しにくい」が正しい読みです。citeturn32view2turn56view0turn51view0turn51view2 fileciteturn6file0L3-L3 fileciteturn8file0L3-L3

## 研究計画 v0 の評価

### RQ・仮説・貢献候補の妥当性

計画 v0 は、内部証拠に基づいてすでに正しい方向へ寄っています。特に、RQ1 を「prior 補正 / external memory / model-state commit / feature adaptation に責務分解して何が効くかを問う」形にしたのは強いです。L3 no-op と L1 ceiling がすでに観測されている以上、**mechanistic decomposition は後付けではなく、中核結果そのもの**です。fileciteturn5file0L3-L3 fileciteturn7file0L3-L3 fileciteturn8file0L3-L3

一方で、仮説の wording はまだ危ういです。v0 は “safe-gain 保証” と書いていますが、現状の設計で得られるのは**理論保証ではなく、実験条件下での経験的制約満足**です。保証という語をそのまま使うと、レビューでは「どの仮定の下で証明したのか」と詰められます。ここは **“constraint-based empirical safety evaluation”** か **“ex-post verified safe policy selection”** に落とすべきです。fileciteturn5file0L3-L3

貢献候補の通りやすさは、現時点では次の順です。

| 順位 | 貢献候補 | 判定 | 理由 |
|---|---|---|---|
| 最上位 | C1 Mechanistic | **通る可能性が最も高い** | 内部で L3 no-op、L1 ceiling、static prior Δ0、single-seed と multi-seed の乖離が既に揃っており、negative result を含む説明力がある。fileciteturn7file0L3-L3 fileciteturn8file0L3-L3 fileciteturn6file0L3-L3 |
| 次点 | C2 Safety framework | **条件付きで通る** | per-subject downside を主指標にした EEG-TTA 評価は薄く、Wimpff 2024 でも subject-wise と usability threshold はあるが、worst-drop / harmed-count を主要最適化対象にしていない。だが、閾値の恣意性・多重比較・“保証”の語は危険。citeturn32view2turn2academia1turn40academia1turn39academia3turn2academia3 |
| 最下位 | C3 Active BCI 評価 | **そのままでは弱い** | いまの計画は offline stream replay を高度化した評価であって、閉ループ BCI そのものではない。少なくとも latency、abstain/coverage、prefix vulnerability、order sensitivity、reset を強く入れないと “Active BCI” は過大主張。fileciteturn5file0L3-L3 citeturn52view0turn53academia1turn53academia2 |

結論は明確です。**論文の主貢献は C1、次に C2、C3 は副読本扱いに落とす**のが妥当です。C3 を主看板にすると、いまの証拠量では過剰です。fileciteturn5file0L3-L3

### 達成基準と harm 閾値

v0 の成功条件は「harm 制約下で replay_safe を有意に上回る」です。この方向自体は正しいです。ただし、そのままだと**評価関数とモデル選択関数が分離していない**のが弱点です。もし sweep の中から harm 制約を満たす best mean を選ぶなら、それは最適化であって評価ではありません。したがって、train/validation/test の役割、あるいは policy-selection split を明確にしないと、後出し最適化と見なされます。fileciteturn5file0L3-L3

HSC@0.5pp / 1.0pp は、**TTA 文献の標準ではありません**。EATA は forgetting を IID 劣化として議論し、SAR は collapse や same-class trivial solution を議論し、TTAB は scenario-level pitfall を議論し、RDumb は collapse と reset の優位を示しますが、**subject-level harmed-count の標準閾値は見当たりません**。Wimpff 2024 には 70% usable threshold を超えた subject 数という運用指標がありますが、これは harm 閾値ではなく usability 指標です。citeturn2academia1turn40academia1turn39academia3turn2academia3turn32view2

したがって、τ と k は単一点で置かず、**frontier として出すべき**です。実務的には以下が妥当です。

| 指標 | 推奨 | 理由 |
|---|---|---|
| worst-subject-drop | 主指標 | 一人でも大きく壊す手法を可視化できる。 |
| HSC@0.5 / 1.0 / 2.0pp | 補助指標 | 閾値依存性を露出できる。標準がない以上、複数閾値で robustness を示すべき。 |
| subject-delta の CVaR 下位 20% | 補助指標 | worst 1 点の不安定性を下げつつ downside tail を測れる。 |
| mean gain | 制約下目的 | これ単独では不十分。 |
| 有意差検定 | 必須 | 単位は**trial ではなく subject×seed**。trial-level 検定は擬似反復で過大有意になる。 |

あなたの current result では single-seed +1.23pp が multi-seed で +0.26〜+0.49pp に縮んでおり、しかも high-mean variant ほど harmed subject が増えています。だからこそ、**成功基準を単一 mean ではなく Pareto-front で定義する**必要があります。fileciteturn6file0L3-L3

### safety を final / online / decision の三層で定義する案

この設計は、既存 TTA 文献よりむしろ**過不足が少ない**です。既存文献の harm は、EATA では forgetting、SAR では collapse、TTAB では scenario-level failure、RDumb では long-horizon collapse と reset 優位として現れます。しかし、それらは多くが domain-average で、**subject-wise downside** を主要評価軸にしていません。あなたの三層化はこの穴を埋めています。citeturn2academia1turn40academia1turn39academia3turn2academia3

ただし、不足もあります。**posterior correction を主成分とするなら calibration 指標が必須**です。accuracy だけで posterior をいじると、正答率が同じでも確信度が壊れている可能性があります。少なくとも Brier score、NLL、ECE を subject-wise に追加すべきです。安全な OTTA を名乗るなら、「当てる」だけでなく「自信の出し方を壊していない」も見ないと片手落ちです。これは特に abstain/reject を扱う decision safety で重要です。fileciteturn10file0L3-L3 citeturn1view2turn41academia4

online regret / max drawdown / recovery time は、**BCI ストリーム評価として妥当**です。むしろ TTA 文献がここを十分に測っていない。RDumb 系が長期崩壊を問題化している以上、drawdown と recovery を導入するのは筋が通っています。加えて、decision safety には coverage、selective risk、AURC、false reject/false accept を入れるべきです。abstain rate だけでは安全性を語れません。citeturn2academia3turn41academia0turn52view0

### 実験計画の穴

最も大きい穴は、**比較対象の射程が literature-facing でまだ足りない**ことです。Wimpff-style OTTA だけでは弱いです。少なくとも、EEG online TL の強い実例として T-TIME 系の比較射程を明示すべきです。T-TIME は binary MI の MOABB 3 データセットで、online EEGNet baseline に対して T-TIME が +2.92pp、+5.65pp、+6.07pp、SAR が +1.28pp、+4.92pp、+2.96pp を出しています。一方で T3A は -5.29pp、-8.94pp、-6.80pp と明確に悪化しています。**この事実は危険で、あなたの “memory / prototype / classifier adjustment は壊れにくいかも” という直感に反例を与えます。** citeturn56view0turn51view0turn51view2turn47view1

二つめの穴は、**cold-start と順序依存性**です。あなたの手法は memory admission と gate に依存するので、最初の数十 trial の質と class order の影響を強く受けます。Wimpff 2024 も single-instance OTTA では input buffer 32 を必須としており、バッファ設計が本質的な trade-off であると明言しています。prefix failure を測らずに安全性を語るのは弱いです。citeturn52view0

三つめの穴は、**データセットの外部比較可能性**です。BCIC2a/2b/HGD は内部主張にはよいですが、外部の online EEG-TTA 文献は BNCI2014001/4002/2015001 も多く使います。特に T-TIME はその 3 つを使っています。2b は binary で ranking-based rescue の情報量が低く、HGD は base が高く ceiling が小さいので gain を議論しづらい。したがって、外部比較の橋渡しとして MOABB 系を少なくとも一部追加する価値があります。citeturn51view0turn51view2 fileciteturn8file0L3-L3

### 計画全体の弱点トップ

弱点は三つです。第一に、**手法新規性が弱いのに、その点を断ち切る言い切りがまだ足りない**ことです。BFT・T3A・LAME・AdaNPC の存在下では、commitless memory correction を “新手法” として押すのは無理があります。citeturn1view0turn3academia1turn3academia2turn47view0

第二に、**効果量が小さすぎて、統計的にも物語的にも負けるリスクが高い**ことです。内部で best single-seed +1.23pp が multi-seed で +0.26〜+0.49pp に縮むなら、method paper としては弱い。ここは最初から method-first を捨て、evaluation-first に倒した方がよいです。fileciteturn6file0L3-L3

第三に、**“Active BCI” と “保証” という語が今の証拠量に対して強すぎる**ことです。前者は閉ループ・非同期・coverage まで要るし、後者は理論保証を連想させます。いずれも reviewer の格好の攻撃点です。fileciteturn5file0L3-L3 citeturn53academia1turn36academia2

## BFT 2026 の精読

### 事実として確認できること

BFT 2026 の arXiv 抄録で確認できる事実は次です。BFT は **Backpropagation-Free Transformations** であり、EEG-BCI の TTA に対して、**knowledge-guided augmentations** または **approximate Bayesian inference** による **sample-wise transformations** を各テスト trial に適用し、そこから得た複数の prediction score を **learning-to-rank module** で重み付け集約する手法です。論文は、既存 TTA の backprop に伴う計算コスト・プライバシー問題・ノイズ感度を批判し、それを回避する no-BP EEG-TTA として BFT を位置づけています。対象は「五つの EEG データセット」にまたがる MI classification と driver drowsiness regression で、resource-constrained devices 向けの軽量性も主張しています。citeturn1view0turn35academia0

ここから確実に言えるのは、BFT が**オンライン EEG ストリームを想定した no-backprop TTA**であり、適応の主体を loss-based gradient update ではなく **sample-level transformation ensemble + prediction aggregation** に置いていることです。これは、あなたが commitless / no-BP / freeze を考えているなら、真正面の競合です。citeturn1view0turn35academia0

### 原典本文から確認できなかった点

この環境では、BFT の PDF/HTML 本文を安定して取得できず、**正確な表・式・実験設定の全文確認はできませんでした**。したがって、以下は**未確認**です。  
BFT の具体的な transformations の列挙、Bayesian approximation の具体形（MC dropout か、別の stochastic approximation か）、ranking aggregation の学習設定、各データセット名、正確な pp 改善、per-subject breakdown、安全性指標、trial-level latency、コード公開リンクです。これらは accessible な抄録ページだけでは出てきませんでした。citeturn1view0turn35academia0

この不足は地味に重要です。なぜなら、**BFT があなたをどこまで subsume するかは、“score aggregation だけなのか / external memory を使うのか / temporal state を持つのか” で変わる**からです。今のソースから言えるのは、少なくとも「no-BP EEG-TTA」「single-trial sample-wise operation」「multi-score aggregation」という大枠が被る、というところまでです。citeturn1view0turn35academia0

### BFT とあなたの commitless OTTA の差分

差分を、**事実**と**推定**を分けて整理すると次です。

| 軸 | BFT 2026 | あなたの commitless memory-corrected OTTA | 判定 |
|---|---|---|---|
| 勾配 | **不要**。no-BP が主張の中核。citeturn1view0turn35academia0 | **不要**。TCFormer は eval + no_grad、L1/L2 は外側制御。fileciteturn10file0L3-L3 fileciteturn12file0L3-L3 | 同系統 |
| 適応単位 | **各 test trial** に sample-wise transformations。citeturn1view0turn35academia0 | single-trial stream。L1 は per-trial、L2 は stream memory。fileciteturn10file0L3-L3 | 同系統 |
| 主たる操作対象 | prediction scores の生成と集約。citeturn1view0turn35academia0 | posterior / logit correction と external memory。fileciteturn10file0L3-L3 fileciteturn12file0L3-L3 | 一部差分あり |
| external memory | accessible source では**未確認** | **明示的に有り**。memory prior と reliability gate。fileciteturn10file0L3-L3 fileciteturn13file0L3-L3 | ここは差別化候補 |
| model-state commit | accessible source では**未確認**。少なくとも BP update はしない。citeturn1view0turn35academia0 | v0 では commitless を志向。内部でも L3 no-op。fileciteturn7file0L3-L3 | あなた側の主張材料 |
| 安全設計 | abstract では robustness / efficiency を主張するが、harm 制約は未確認。citeturn1view0turn35academia0 | harmed-subject / worst-drop / abstain-safe を主目的に置く。fileciteturn5file0L3-L3 fileciteturn10file0L3-L3 | **差別化の本命** |
| 新規性の中心 | no-BP EEG-TTA + transformation/ranking | safe memory correction + mechanistic evaluation | method novelty の競合は強い |

ここからの結論は曖昧ではありません。**アルゴリズム新規性は残りません。** BFT が accessible source の範囲で既に殺しているのは、「EEG向け no-BP test-time adaptation」という看板です。さらに T3A/LAME/AdaNPC が一般形としてその設計空間を埋めています。citeturn1view0turn3academia1turn3academia2turn47view3

ただし、**論文新規性まで完全に消えるわけではありません**。残る軸は三つだけです。  
第一に、**L3 commit が no-op であるという mechanistic negative result**。第二に、**mean accuracy ではなく worst-subject drop / harmed count / online drawdown を軸にした安全評価**。第三に、**posterior correction の ceiling を oracle で先に抑え、その範囲内で何が起きているかを分解すること**です。これは BFT の abstract からは見えないし、少なくとも accessible source の範囲では代替されていません。fileciteturn7file0L3-L3 fileciteturn8file0L3-L3 fileciteturn5file0L3-L3

### BFT から見た、あなたの現方向の弱点

弱点は二つです。  
一つ目は、**BFT が “memory を持たずに 1 試行内で複数 score を作って集約する” 系だとすると、あなたの memory correction は cold-start と error accumulation を背負う**ことです。つまり、BFT が temporal commitment を避けているなら、あなたは逆に temporal state の安全性を証明しないといけません。これは負担です。citeturn1view0turn35academia0 fileciteturn10file0L3-L3

二つ目は、**BFT が no-BP を守りつつ “transformation diversity” を使って uncertainty suppression をするのに対し、あなたは class-prior / memory-prior の correction が中心で、表現多様性を増やしていない**ことです。内部 oracle が示す通り、L1 ceiling は +3.36pp で止まります。したがって、「精度を大きく上げる」という観点では、BFT 型の per-sample diversification の方がまだ夢があります。あなたの現方向はここで負けやすいです。fileciteturn8file0L3-L3 citeturn1view0turn35academia0

## 方向判断

### 「大きく精度を上げる」を外す判断は妥当か

**妥当です。しかもかなり強く妥当です。** 内部 oracle は 2a の L1 上限を +3.36pp、2b を +1.73pp、HGD を +3.18pp としており、しかも static global prior は全部 Δ0 です。さらに実測では best single-seed が +1.23pp、multi-seed では +0.26〜+0.49pp です。これを見てなお “freeze + posterior/memory correction で大きく上げる” を主目標に置くのは、データより願望を優先しています。fileciteturn8file0L3-L3 fileciteturn6file0L3-L3

ただし、一般論としての「EEG online TTA で +5pp は非現実的」までは言えません。T-TIME は binary MI の BNCI2014001/4002/2015001 で、online EEGNet に対して +2.92pp、+5.65pp、+6.07pp を達成しています。SAR も同条件で +1.28pp、+4.92pp、+2.96pp を出しています。したがって、**“EEG online TL 全体” では +5pp は現実にある**。でもそれは、incremental EA、batch size 8、full-parameter fine-tuning、複数モデル ensemble、binary MI という、あなたの current constraint と異なるレジームです。citeturn56view0turn51view0turn51view2

正確な言い方をすると、**“+5pp は EEG ではあり得るが、あなたの admissible operation class では期待しにくい”** です。ここを混同すると進路を誤ります。fileciteturn8file0L3-L3 citeturn56view0turn51view0turn51view2

### 特徴適応に踏み込むべきか

特徴適応に踏み込めば ceiling は上がります。これは内部 oracle でも、L1 ceiling と top2 ceiling のギャップが示しています。2a では +3.36pp と +11.84pp の差が大きく、bias correction では救えない “正解は top2 に入るが bias では動かない” 領域がある。ここは feature adaptation の守備範囲です。fileciteturn8file0L3-L3 fileciteturn9file0L3-L3

しかし、今それをやるべきかというと、**答えは no** です。理由は三つあります。  
第一に、競合が急に強くなる。T-TIME、Wimpff OTTA、Tent、SAR、CoTTA 系と真正面に戦うことになります。citeturn44view0turn52view0turn40academia1turn47view0  
第二に、安全性が難しくなる。EATA・SAR・RDumb が示す通り、勾配更新系は forgetting / collapse / long-horizon failure の問題を抱えます。citeturn2academia1turn40academia1turn2academia3  
第三に、あなたの current evidence は「L3 no-op」「L1/L2 で一部 gain」「harm 出現」という、**理解論文にとって都合がよすぎる状態**です。ここで feature adaptation に乗り換えると、せっかく揃った negative result を捨てることになります。fileciteturn7file0L3-L3 fileciteturn6file0L3-L3

したがって推奨は一つです。**今回の論文は safety / understanding 主導で行くべき**です。feature adaptation は別プロジェクトに切り出す方が合理的です。今やるべきなのは、「どこまでなら壊さずに上がるか」をきちんと境界づけることです。fileciteturn5file0L3-L3 fileciteturn8file0L3-L3

## 修正版の貢献ランキングと次にやるべき実験

### 修正版の貢献ランキング

| 順位 | 貢献の形 | 具体化 |
|---|---|---|
| 最上位 | **Mechanistic negative result** | L3 commit no-op、L1 ceiling、static prior Δ0、single-seed と multi-seed の解離を、同一 backbone・同一 protocol・公平 ablation で示す。fileciteturn7file0L3-L3 fileciteturn8file0L3-L3 fileciteturn6file0L3-L3 |
| 次点 | **Safety evaluation framework** | mean gain ではなく worst-subject drop / HSC / CVaR / online regret / drawdown / recovery / coverage-risk を前面化する。既存 TTA が薄いところ。citeturn2academia1turn40academia1turn39academia3turn2academia3turn32view2 |
| 三番手 | **Commitless policy selection** | “新手法” ではなく、“constraint-satisfying policy selection among lightweight controls” として出す。補正式自体ではなく採用基準で差別化。fileciteturn5file0L3-L3 |
| 最下位 | **Active BCI claim** | 実時間・非同期・閉ループが無い限り副次的主張に留める。citeturn53academia1turn36academia2 |

### 次にやるべき実験

最優先の三つはこれです。

**第一に、multi-seed の safety frontier を固定評価プロトコルで作ること。**  
対象は 2a/2b/HGD の `source_only`、`replay_safe`、L1-only、L1+L2 commitless、L3 on/off、high-mean variant です。出すべきものは mean ではなく、subject×seed 単位の delta 分布、worst-drop、HSC@0.5/1.0/2.0、CVaR、オンライン regret / max drawdown / recovery time です。今の repo にはその材料があり、しかも v0 自身がこれをマイルストーン 2–4 に置いています。fileciteturn5file0L3-L3 fileciteturn6file0L3-L3 fileciteturn7file0L3-L3

**第二に、posterior correction を calibration まで含めて評価すること。**  
commitless をやるなら accuracy だけでは足りません。Brier、NLL、ECE、coverage-risk、AURC を subject-wise に追加してください。もし posterior correction で accuracy が 0.3pp 上がっても calibration が大きく悪化するなら、それは “safe” ではありません。これは current repo の L1/L2 設計と相性がよく、実装コストも低いです。fileciteturn10file0L3-L3 fileciteturn12file0L3-L3

**第三に、cold-start / order sensitivity のストレステストを入れること。**  
memory correction は prefix と stream order に依存します。各 subject で canonical order だけでなく、class-block の偏りや early noisy prefix を模した順序ストレス試験を加え、`memory_not_ready`、admission error、early harm を可視化するべきです。Wimpff 2024 が input buffer を設計中心に置いている以上、ここを落とすのはまずいです。citeturn52view0 fileciteturn10file0L3-L3 fileciteturn13file0L3-L3

## 限界と未解決点

BFT 2026 については、accessible な一次ソースから確認できたのは arXiv 抄録までで、**正確な式・表・実験値・コード公開の有無は本文から最終確認できませんでした**。したがって、BFT 評価のうち「sample-wise transformation」「knowledge-guided augmentations or approximate Bayesian inference」「learning-to-rank aggregation」「no-BP EEG-TTA」という骨格は**事実**ですが、transform の具体名や exact pp gain は**未確認**として扱うのが正確です。citeturn1view0turn35academia0

T3A については、原著タイトルと bibliographic information は T-TIME の reference から追えましたが、今回の収集では原著本文まで遡って説明を厚くするところまでは届いていません。そのため、本報告での T3A の位置づけは「EEG online TL ベンチで strong baseline ではなく、むしろ大きく悪化しうる prototype/classifier adjustment 系」という**T-TIME 上の観測**に依っています。citeturn47view3turn47view1

最終的な一文だけ残すならこうです。**いま追うべきは “大きな gain を出す新手法” ではなく、“どこまでなら壊さずに gain を取れるかを、subject downside まで含めて確定する論文” です。** その線なら、BFT が出てもまだ席があります。逆に、commitless method paper に固執すると席はかなり狭いです。fileciteturn5file0L3-L3 fileciteturn8file0L3-L3 citeturn1view0turn35academia0turn47view1