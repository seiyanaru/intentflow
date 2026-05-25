# EEG-MI の OTTA と commitless memory correction の新規性評価

## エグゼクティブサマリ

外部文献の reality check では、**EEG-MI で本当に online / source-free / calibration-free に近い設定の改善幅は、いま見えている範囲では概ね low single-digit pp が中心**です。これに対して **+5pp 超の改善は、offline DA・few-shot・session-aware fine-tuning ではあり得るが、freeze + single-trial + no-backprop の OTTA で一般的期待値として置くには強すぎます**。citeturn35view0turn37academia1turn16academia0turn39academia1

あなたの案である **commitless memory-corrected OTTA** は、**「モデル状態を更新せず posterior / template / memory で補正する」というコア機構だけを見ると新規性は弱い**です。LAME、AdaNPC、TAST 系に加え、**EEG そのものでも 2026 年に backprop-free TTA が出ており**、アルゴリズム単体では差が立ちにくいです。citeturn17academia3turn17academia0turn17academia1turn39academia1

一方で、**「平均精度を少し上げる」よりも「誰も大きく壊さない」を主目的に据えるなら、まだ勝ち筋はあります**。TTA 文献は collapse / forgetting / instability を論じていますが、**per-subject downside を主要評価軸にした例は薄く**、あなたの HSC 的な指標はむしろ実運用向きです。citeturn28academia0turn30academia0turn30academia2turn29academia1turn35view0

結論は **「現方向はそのままでは弱いが、安全性主導の再定義をすれば通る余地がある」**です。主張を **“large gain” から “safe gain under batch=1 online MI-BCI”** に下げ、**worst-subject / harmed-subject / latency / abstain-reset を正面から評価**する形に修正すべきです。citeturn30academia2turn29academia1turn39academia1turn35view0fileciteturn7file0L1-L3

## Reality check

**事実**として、現時点で見つかる EEG-MI の OTTA / source-free / calibration-free 系文献は、Computer Vision の TTA 文献ほど厚くありません。Wimpff らの 2024 年論文は、自らを **“EEG motor imagery decoding に OTTA を適用した最初の研究”** と位置づけ、BCIC IV 2a/2b で **cross-session、cross-subject、continual cross-subject** を評価しています。そこで使っているのは、online alignment、adaptive BN、entropy minimization であり、single-instance OTTA のために **buffer size 32** を置き、entropy の更新は **buffer が更新されたとき**に行う設計です。citeturn3view0turn35view0

**推定**として、ここから読み取れる現実的レンジは次です。**真に近い online / source-free / calibration-free OTTA の gain は low single-digit pp が中心**で、**offline DA や few-shot を混ぜると +5pp〜+15pp も出る**、という二層構造です。したがって、**freeze + online + single-trial + no-backprop のみで “安定して +5pp” を期待するのはかなり強い仮説**です。反証条件は簡単で、**multi-seed・per-subject で +5pp 級が再現する論文や、自分の設定と同等制約のベンチマークで同水準が出ていること**が必要です。現時点の外部証拠はそこまで強くありません。citeturn35view0turn37academia1turn39academia1turn16academia0

| 代表手法 | 論文情報 | 設定 | model update | 改善幅の読み | コード | 含意 |
|---|---|---|---|---|---|---|
| Calibration-free OTTA | *Calibration-free online test-time adaptation for electroencephalography motor imagery decoding* — Martin Wimpff, Mario Döbler, Bin Yang, 2024, International Winter Conference on BCI, arXiv:2311.18520 | BCIC IV 2a/2b、cross-session / cross-subject / continual cross-subject、single-instance OTTA | alignment・BN 更新、EM は buffer 更新時に BP | accessible text から確実に読めるのは **continual 2a で「almost 2 percent increase」**。cross-session / cross-subject は baseline 超えだが HTML 抽出では表の数値が欠落 | **有**。論文本文で code 公開記載 | 真に近い MI-OTTA の外部 reality check。大幅改善より**小〜中改善**の証拠 citeturn35view0turn36view2 |
| Backprop-free EEG TTA | *Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based Brain-Computer Interfaces* — Siyang Li et al., 2026, arXiv:2601.07556 | 5 EEG datasets、MI と drowsiness、resource-constrained BCI 向け | **No BP**。sample-wise transformation + ranking aggregation | abstract では **effectiveness / robustness / efficiency** を主張するが、accessible abstract に exact pp はない | 未確認 | あなたの方向に最も近い外部競合。**「no-BP EEG-TTA」自体は既に先行あり** citeturn39academia1turn40view0 |
| SDDA | *Priming Cross-Session Motor Imagery Classification with A Universal Deep Domain Adaptation Framework* — Zhengqing Miao et al., 2022, arXiv:2202.09559 | BCIC IV 2a/2b、cross-session、offline DA | 学習時に target を用いる | EEGNet / ConvNet 比で **2a は +15.2pp / +10.2pp、2b は +5.5pp / +4.2pp** | 未確認 | **+5pp 超はある**。ただしこれは **online freeze-only OTTA ではない** citeturn37academia1turn48academia2 |
| EEG-DG | *EEG-DG: A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification* — Xiao-Cong Zhong et al., 2023, arXiv:2311.05415 | BCIC IV 2a/2b、subject-independent DG | test-time update なし | **81.79% / 87.12%** を reported。DA を使わずに一部 DA 法を上回ると主張 | **有** | 強い non-adaptive baseline が成立する。OTTA の上積み余地は baseline 依存 citeturn37academia0 |
| FBCNet | *FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface* — Ravikiran Mane et al., 2021, arXiv:2104.01233 | MI 4 datasets、subject-independent 評価包含 | update なし | BCIC-IV-2a **76.20%**、他 binary dataset では **up to +8%** | **有** | source model 自体が強いほど OTTA の絶対改善余地は縮む citeturn53academia3 |
| あなたの内部 reality check | `intentflow` 内部 OTTA 分析、2026-03-07 | BCIC2a、9 subjects | current OTTA | **平均 +0.20%、標準偏差悪化、最大 -7.99%**、適応率と改善に正相関 | repo 内部資料 | **外部文献と整合的に、gain は小さく harm は顕在** fileciteturn7file0L1-L3 |

**判定**：  
**「+5pp を凍結モデル + オンライン適応で出すのは現実的か」への答えは、“一般解としてはかなり厳しい”** です。外部で大きな gain が出るのは、offline DA・few-shot・fine-tuning を許す場合が多く、**あなたの制約に近い設定では low single-digit pp が相場**とみるのが妥当です。**+5pp を狙うなら、source model がかなり弱い、あるいは alignment / BN / 軽い BP / session-level fine-tune のどれかを足す必要がある**可能性が高いです。citeturn37academia1turn35view0turn16academia0turn39academia1fileciteturn7file0L1-L3

## 新規性ギャップ

**事実**として、一般 TTA にはすでに **「モデル本体を大きく変えず、prediction / classifier / memory を test-time で補正する」** 系列が存在します。LAME は **model output を適応**する conservative, parameter-free OTTA を提案し、Laplacian Adjusted MLE を解くことで **バックボーンを更新しない**方向を取っています。AdaNPC は **source memory に feature-label を保存し、test feature に近いサンプルで vote し、test feature と pseudo-label を memory に追加**する非パラメトリック適応です。TAST は nearest-neighbor 情報を使いますが、**adaptation module を追加し、test-time に学習**するため、純粋な freeze + posterior correction ではありません。citeturn17academia3turn17academia0turn17academia1

**EEG への引き寄せで最重要なのは BFT 2026**です。BFT は EEG-BCI 向けに、**backpropagation を使わず、trial ごとの複数 transformation / Bayesian approximation から複数 score を作り、ranking で重みづけて最終予測を集約**します。つまり、**「no-BP」「lightweight」「online EEG」「posterior aggregation」**の組み合わせは、すでに外部にあります。ここがあなたの構想に対する一番強いカウンターです。citeturn39academia1turn40view0

| 手法 | 論文情報 | 中核アイデア | 勾配 | freeze | memory | EEG-MI online single-trial への適用確認 | あなたの研究への含意 |
|---|---|---|---|---|---|---|---|
| LAME | *Parameter-free Online Test-time Adaptation* — Malik Boudiaf et al., 2022, arXiv:2201.05718 | **model output** を conservative に補正する parameter-free OTTA | 不要 | 実質 yes | 重い replay ではない | 今回の探索では **直接適用例なし** | **直接競合**。単なる posterior correction では弱い citeturn17academia3 |
| AdaNPC | *AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation* — Yi-Fan Zhang et al., 2023, arXiv:2304.12566 | source memory + KNN vote + test sample の memory 追加 | 不要 | ほぼ yes | **有** | 今回の探索では **直接適用例なし** | **memory-based correction の代表競合**。差別化必須 citeturn17academia0 |
| TAST | *Test-Time Adaptation via Self-Training with Nearest Neighbor Information* — Minguk Jang et al., 2022, arXiv:2207.10792 | NN 情報で pseudo-label 分布を作り、test-time に adaptation module を学習 | **要** | extractor は保持、module は更新 | prototype / NN 依存 | 今回の探索では **直接適用例なし** | 直接競合ではないが、「NN と prototype を使う」点は近い citeturn17academia1 |
| BFT | *Backpropagation-Free Test-Time Adaptation for Lightweight EEG-Based Brain-Computer Interfaces* — Siyang Li et al., 2026, arXiv:2601.07556 | sample-wise transformations + ranking aggregation による EEG 向け no-BP TTA | 不要 | 実質 yes | temporal memory は主ではない | **有** | **最重要競合**。EEG 向け・no-BP という軸は既に埋まりつつある citeturn39academia1turn40view0 |
| Wimpff OTTA | *Calibration-free online test-time adaptation for electroencephalography motor imagery decoding* — Martin Wimpff et al., 2024, Winter BCI, arXiv:2311.18520 | online alignment + adaptive BN + optional entropy minimization | BN は不要、EM は要 | 部分更新 | buffer 使用 | **有** | **MI-OTTA の必須比較対象** citeturn3view0turn35view0 |
| T3A | 原典の bibliographic metadata を今回の探索で source-verify できず | 後続文献では **training-free / prototype-template adjustment 系 baseline** として扱われる | 不要系として扱われがち | おそらく yes | template/prototype | EEG-MI の直接適用は今回未確認 | **名前だけ出すのは危険**。原典を再確認してから比較表に入れるべき citeturn21academia0turn24academia0 |
| BTTA-DG | 原典の bibliographic metadata を今回の探索で source-verify できず | 現探索では十分な一次情報が取れず | 未確定 | 未確定 | 未確定 | EEG-MI 未確認 | **このまま主張に組み込むのは危険** |

**結論**：  
**「commitless memory-corrected OTTA」というアルゴリズム記述だけでは、新規性は弱い**です。特に **LAME / AdaNPC / BFT** があるため、**freeze + gradient-free + posterior / memory correction** は既存の設計空間に入っています。citeturn17academia3turn17academia0turn39academia1

**まだ埋まっていない gap**があるとすれば、それはアルゴリズムの骨格ではなく次の三点です。  
第一に、**EEG-MI の batch=1 online stream で、model state をいっさい commit せず、subject-level harm 制約付きで memory correction を動かす**こと。第二に、**mean accuracy ではなく worst-subject drop / harmed-subject count / usability threshold を主目的関数にする**こと。第三に、**commit / replay / reset / abstain を同一プロトコルで比較し、「commitless が no-op commit より安全に勝つ」ことを示す**ことです。これは **評価設計と安全性主張**の新規性であって、**補正式そのものの新規性ではありません**。citeturn30academia2turn29academia1turn35view0fileciteturn7file0L1-L3

## 精度向上と無害化

**事実**として、TTA 文献では harm は主に **collapse、catastrophic forgetting、unstable adaptation** という言い方で扱われます。EATA は、既存 TTA が **OOD では改善しても IID で severe performance degradation を起こす**ことを問題化し、**reliable / non-redundant test sample selection** と **Fisher regularizer** で forgetting を抑えようとしています。SAR は、mixed shifts・small batch・online imbalanced label shifts では TTA が **same-class ばかり出す trivial solution に collapse**し得ると述べ、sharpness-aware な更新と noisy sample 除去で安定化を図っています。citeturn28academia0turn30academia0turn26academia2

**事実**として、TTAB は 10 手法を広く比較し、**ハイパーパラメータ選択が難しいこと、TTA の有効性がモデルと shift 特性に強く依存すること、既存法のどれも全ての一般的 shift に対処できないこと**を示しています。さらに RDumb は continual test-time adaptation を長時間ストリームで評価し、**ほとんどの手法が最終的に collapse して non-adapting model より悪くなる**ことを報告しています。言い換えると、**「平均を上げつつ最悪ケースも守る」ことは、文献上も未解決寄り**です。citeturn45view0turn29academia1

| 研究 | harm の定義 | 何を測るか | per-domain / per-subject downside 指標 | 成功度の評価 |
|---|---|---|---|---|
| EATA | forgetting、IID 劣化、 noisy sample による有害更新 | OOD 改善と forgetting 抑制 | **明示的な worst-domain 指標は薄い** | 部分成功。だが harm の定義はまだ coarse citeturn28academia0 |
| SAR | collapse、same-class trivial solution、small-batch instability | wild scenarios 下の安定精度 | **domain-average 中心** | 従来法より stable と主張 citeturn30academia0turn26academia2 |
| TTAB | shift ごとの failure、model dependency、hyperparameter brittleness | 多手法×多 shift の benchmark | **worst-subject より scenario-average** | 「既存法は万能でない」を示した点が重要 citeturn45view0 |
| RDumb | 長時間 continual adaptation での collapse | asymptotic stream performance | subject 単位ではない | **reset baseline が強い**。安全性に reset が効く証拠 citeturn29academia1 |
| Wimpff 2024 | per-subject variability を前提に usability threshold を重視 | subject-wise result、70% threshold を超える人数 | **subject-wise は有** | 2a で **3/7 subjects** を threshold 超えに押し上げたと報告 citeturn35view0 |

**推定**として、この文脈では **あなたの HSC（大きく悪化する被験者数）や max-subject-drop は、既存 TTA 文献の平均精度よりも運用価値が高い**です。特に Active BCI では、**平均 +0.5pp と引き換えに 1 人を -8pp 壊す**設計は実用上ほぼ不採用です。あなたの内部分析でも、best OTTA が平均 **+0.20%** に留まり、しかも **最大 -7.99%** の subject が出ています。これは、まさに一般 TTA 文献が言う instability / harmful adaptation と同型です。citeturn28academia0turn30academia0turn29academia1fileciteturn7file0L1-L3

**率直な判定**：  
**「平均精度を上げつつ最悪ケースを守る」ことに成功した例はあるか**に対しては、**限定的には yes、強くは no**です。EATA や SAR は安定化の方向で改善を示しますが、**per-subject downside を明示制約として最適化した成功例**は、今回の探索では見当たりませんでした。したがって、**ここはまだ contribution 余地がある**一方、**それだけ難しい**とも言えます。citeturn28academia0turn30academia0turn45view0turn29academia1

## 重い適応手法の現実性

**事実**として、TENT は test entropy minimization による fully test-time adaptation の代表で、**normalization statistics と affine parameters を online で最適化**します。CoTTA は continual shift に対し、**weight-averaged / augmentation-averaged prediction** と **stochastic restore** で forgetting を回避しようとします。EATA/SAR も勾配更新系です。これらは一般 TTA では強いですが、**単発 prediction を即時返す EEG-MI の batch=1 / low-latency stream では計算負荷と不安定性がそのままボトルネック**になります。TTAB も、一般に TTA の成否が shift と model に強く依存すると警告しています。citeturn27academia0turn27academia2turn28academia0turn30academia0turn45view0

Wimpff 2024 の MI-OTTA でも、single-instance OTTA に immediate prediction が必要なので **input buffer** を置き、alignment と BN の統計推定を安定化させています。さらに entropy minimization は **buffer が完全に更新されたときだけ**実行しています。これは、**純粋な single-trial instant BP よりは現実的だが、それでも posterior-only より重い**という意味です。citeturn35view0

**EEG-MI で“もっと大きく上げる”方向**として外部で見つかったのは、OTTA そのものよりも、**longitudinal fine-tuning** や **real-time architecture adaptation** です。Wimpff らの 2025 年 longitudinal study は、**prior subject-specific information を継承する fine-tuning が performance と stability の両方を改善し、OTTA がそれを補完する**と要約しています。RAP 2025 は、**offline-to-online ギャップ、sliding window での計算量増大、学習データ不足**を real-time BCI の主要課題とし、**realtime adaptive pooling と source-free DA** を組み合わせています。これは、**本当に +5pp を取りにいくなら、posterior correction だけでなく前段の表現や pooling を触るほうが自然**という示唆です。citeturn16academia0turn16academia1

**判定**：  
online single-trial・低レイテンシ制約で現実的なのは、現時点では **BFT 型の no-BP 変換集約**、あるいは **あなたの commitless correction 型**です。逆に、**TENT/EATA/SAR/CoTTA をそのまま batch=1 MI-BCI に持ち込んで「大きく、しかも安全に上げる」**のは、文献上の instability 警告と整合しません。**大幅 gain を優先するなら heavier adaptation を許容する必要があり、その瞬間にあなたの現在の“壊さない軽量 OTTA”という研究軸と衝突します。**citeturn39academia1turn27academia0turn28academia0turn30academia0turn16academia1

## Active BCI 制約

**事実**として、Wimpff 2024 は single-instance OTTA の核心制約をかなり明示しています。新しい input sample を受け取ったら **即座に予測しなければならない**一方で、alignment や BN のためには過去試行の buffer が必要であり、buffer size は **performance・memory・distribution shift speed の trade-off**になります。彼らは **FIFO で 32 試行**を保持しています。これは Active BCI の設計論としてかなり重要です。**batch=1 そのものは予測単位であっても、適応は“短い履歴窓”を必要とする**という点です。citeturn35view0

RAP 2025 は、real-time BCI で深層学習が普及しない理由として、**offline model の online 化が不明瞭、sliding windows が計算量を大きく増やす、データ量が少ない**の三点を挙げています。そのうえで、**pooling 層を real-time 用に改造し、source-free adaptation を使って calibration-free operation を狙う**と述べています。つまり、Active BCI の“本丸”は OTTA 単独ではなく、**inference architecture / compute budget / calibration reduction を一体で設計すること**です。citeturn16academia1

Longitudinal online MI の論文も、**複数セッションにまたがる causal setting**で performance と stability を見る必要性を強調しています。これは、あなたが最終的に目指す **online Active BCI** にかなり近いです。逆に言うと、BCIC2a のような短い replay ベース評価だけで安全性を言い切るのは弱く、**session drift・long stream・reset 戦略・user feedback 有無**まで拡張しないと、Active BCI claim は薄くなります。citeturn16academia0turn29academia1

あなたの `intentflow` は、取得→前処理→推論→安定化→adapt→配信の online pipeline を明示し、**目標レイテンシ 300–500 ms**、confidence threshold、EMA、連続一致を実装上の KPI としています。これは文献ではなくあなたの実装目標ですが、**研究評価に latency / abstain / stabilizer を入れる妥当性**を裏づけます。今回の探索では、**abstain / reject を主評価に置いた MI-OTTA 文献は強く見つかっておらず、ここは差別化点になり得ます。** fileciteturn4file0L1-L3

## 標準ベンチマークと評価

**事実**として、EEG-MI の OTTA 比較で今いちばんプロトコルが明確なのは Wimpff 2024 の BCIC IV 2a/2b です。BCIC IV 2a は **9 subjects、22 electrodes、2 sessions、各 288 trials、4 classes**、2b は **9 subjects、3 electrodes、5 sessions、2 classes**です。彼らの OTTA 評価は、2a では **cross-session: same subject の session 1 で train、session 2 で adapt/eval**、**cross-subject: 8 source subjects の session 1 で source training、hold-out subject の session 2 で adapt/eval**、**continual cross-subject: hold-out subject の session 1 と 2 を続けて適応**、という整理です。citeturn3view0turn35view0

OpenBMI / Lee2019 系は subject-independent MI の標準ベンチマークとしてよく使われており、MIN2Net、DADL-Net、DFBRTS でも BCIC2a と並べて扱われています。ただし、**OpenBMI に対する“標準 OTTA split” は、今回の探索では明瞭に定着していません**。つまり、OpenBMI は使われているが、**MI-OTTA の公平比較プロトコルは BCIC2a/2b ほど揃っていない**というのが現状認識です。citeturn1academia2turn53academia1turn53academia0

| データセット | 何が標準か | 現探索で確認できたこと | OTTA 相場の見え方 |
|---|---|---|---|
| BCIC IV 2a | 最重要。4-class、9 subjects、cross-session / cross-subject を切りやすい | Wimpff 2024 が OTTA protocol をかなり明示 | **OTTA は low single-digit pp が中心**、強 baseline だと上積みは小さい citeturn35view0turn53academia3 |
| BCIC IV 2b | 2-class、session 数が多く continual 的考察もしやすい | Wimpff 2024 で first 3 sessions train / last 2 test | 2a より sensor 数が少なく、adaptation gain はさらに限定的な可能性 citeturn3view0turn35view0 |
| OpenBMI / Lee2019 | subject-independent MI の準標準 | MIN2Net, DADL-Net, DFBRTS が使用 | **OTTA より offline subject-independent benchmark として成熟** citeturn1academia2turn53academia1turn53academia0 |
| HGD | 強い MI benchmark として使われる | あなたの repo では paper experiments に含まれる | **本探索では標準 OTTA split を十分確認できず** fileciteturn4file0L1-L3 |

**推奨する公平比較軸**は、もう accuracy 一本では足りません。最低でも **mean accuracy / kappa / per-subject delta / worst-subject delta / harmed-subject count / 70% usability crossing / latency per trial** を並べるべきです。Wimpff 2024 は 70% threshold、TTAB と RDumb は long-horizon failure、あなたの実装系は latency KPI を持っています。この三者を合成した評価系が、むしろこのテーマの不足部分です。citeturn35view0turn45view0turn29academia1fileciteturn4file0L1-L3

## 新規性トップ軸と現方向の評価

### 新規性が立つ最も有望な軸

**第一候補は、安全性主導の OTTA**です。  
主張は **「mean を最大化する adaptation」ではなく、「worst-subject drop と harmed-subject count を制約したうえで mean を改善する adaptation」**に置くべきです。一般 TTA 文献は forgetting や collapse を論じますが、**subject-level downside を主要 EP にしたものは弱い**ので、ここはあなたの HSC 系指標がそのまま contribution 候補になります。citeturn28academia0turn30academia0turn45view0turn29academia1fileciteturn7file0L1-L3

**第二候補は、commit vs commitless の系統的否定実験**です。  
あなたは既に **L3 commit が実質 no-op** であるという強い内部知見を持っています。これを主張の前面に出し、**“EEG-MI online batch=1 では model-state commit は効かず、利益の大半は posterior / memory correction から出る”**と見せられれば、単なる新手法より強いメッセージになります。これは **negative result を伴う mechanistic contribution** です。外部でも RDumb が「賢い適応より reset が強い」ことを示しており、**state update を疑う方向自体は潮流に合う**と言えます。citeturn29academia1fileciteturn7file0L1-L3

**第三候補は、Active BCI 向け評価設計の提案**です。  
具体的には、**single-trial、batch=1、latency budget、abstain / reject、stabilizer、reset** を含んだ MI-OTTA benchmark を出すことです。RAP 2025 が offline-to-online ギャップを問題化し、Wimpff 2024 が buffer trade-off を示している一方で、**安全な online BCI を評価するプロトコルはまだ薄い**です。アルゴリズムより評価系のほうが、今のあなたの手持ち結果と噛み合っています。citeturn16academia1turn35view0fileciteturn4file0L1-L3

### 現方向への批判的評価

**判定は「修正すべき」です。**  
理由は三つあります。第一に、**外部 reality check 上、freeze + online + single-trial + no-BP で “大きく” 上げる期待は強すぎる**こと。第二に、**commitless memory correction 自体のアルゴリズム新規性は弱い**こと。第三に、**harm を抑えたいなら、平均精度の最大化と同時追求は欲張りすぎる**ことです。外部の MI-OTTA と一般 TTA の両方が、それを示しています。citeturn35view0turn39academia1turn17academia3turn17academia0turn30academia2turn29academia1

**修正案**は明確です。  
論文の主張を **「commitless memory correction で大きく上げる」から、「subject-safe online MI-OTTA を定式化し、その制約下で小さくても再現的な gain を達成する」**へ切り替えるべきです。具体的には、目的関数を **mean accuracy↑ subject to max-drop ≤ τ / harmed-subject count ≤ k** 型に変え、比較対象として **source only / reset / BN-only / Wimpff-style OTTA / BFT-style no-BP baseline / あなたの commitless** を並べるのが筋です。これなら新規性は **安全性・評価・設計原理**に立ちます。citeturn39academia1turn35view0turn28academia0turn29academia1fileciteturn7file0L1-L3

**逆に、もし本当に “大きく上げる” を最優先にしたいなら、今の方向は多分違います。**  
その場合は、**alignment + BN + 軽い BP** や **session-level continual fine-tuning**、あるいは **real-time architecture adaptation** 側へ寄るべきです。だがそれは、あなたが今守ろうとしている **commitless / lightweight / 壊さない** という軸をかなり崩します。つまり、**「大きく上げる」と「誰も壊さない」を同時に強く主張するのは、現証拠では両立が薄い**です。片方を主目的、もう片方を制約に落とすべきです。citeturn16academia0turn16academia1turn35view0turn30academia2turn29academia1

**Open questions / limitations**：  
今回の探索では、**Wimpff 2024 と BFT 2026 の exact table 数値を accessible HTML から完全には回収できていません**。また、**T3A と BTTA-DG の原典 bibliographic 詳細は source-verified できていない**ため、そこは断定を避けました。したがって、最終的な related work 表を論文化する段階では、**この二点だけは原論文 PDF を直接再確認する必要があります。** ただし、**結論レベルでは既に十分で、現在の研究方向は“そのまま押し切る”より“安全性主導に再定義する”ほうが強い**という判断は変わりません。citeturn35view0turn39academia1turn30academia2turn29academia1