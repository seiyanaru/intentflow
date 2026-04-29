# EEG 運動想起における BN-free Prototype OTTA と Hierarchical Gating の先行研究調査

## 関連論文リスト

### EEG / BCI に直接関係する OTTA・SFDA・オンライン適応

| 論文 | 著者 | 年 / venue / 査読 | 識別子 | 一行要約 | 関連度 | 出典 |
|---|---|---|---|---|---|---|
| *Calibration-free online test-time adaptation for electroencephalography motor imagery decoding* | entity["people","Martin Wimpff","eeg adaptation researcher"] et al. | 2024 / Winter Conference on Brain-Computer Interface 系会議・arXiv / 査読会議 + preprint | 書誌確認は会議記載ベース | EA/RA と BN 系微調整で MI を校正不要化 | A | citeturn1search0turn43search13 |
| *T-TIME: Test-Time Information Maximization Ensemble for Plug-and-Play BCIs* | entity["people","Siyang Li","hust eeg researcher"] et al. | 2024 / IEEE *Transactions on Biomedical Engineering* / 査読済み | DOI: 10.1109/TBME.2023.3303289 | EEG-BCI 向け初期の本格 OTTA | A | citeturn42search0turn42search7turn42search19 |
| *Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding* | entity["people","Huan Luo","xjtu eeg researcher"] et al. | 2026 / ICLR 2026 poster / 査読済み | OpenReview: VDG6Pv4S3v | BN-free・勾配不要の MI-TTA | A | citeturn38search1turn38search2turn40view0turn40view1 |
| *Latent alignment in deep learning models for EEG decoding* | entity["people","Stylianos Bakas","imperial eeg researcher"] et al. | 2025 / *Journal of Neural Engineering* 22(1) / 査読済み | DOI: 10.1088/1741-2552/adb336 | 深層特徴空間で分布整合 | B | citeturn42search4turn42search16 |
| *Source-free domain adaptation for ssvep-based brain-computer interfaces* | entity["people","Osman Berke Guney","ssvep sfda researcher"] et al. | 2023 / arXiv / preprint | arXiv:2305.17403 | SSVEP の source-free 適応 | B | citeturn4search22 |
| *An Online Adaptation Framework for Enhancing Calibration-Free SSVEP-Based BCI Performance* | 著者情報取得は今回未完了 | 2025 / IEEE JBHI 系 / 査読済み | DOI: 10.1109/JBHI.2025.3644250 | SSVEP のオンライン適応 | B | citeturn4search5 |
| *Test-Time Adaptation for EEG-Based Driver Drowsiness Classification* | entity["people","Geun-Deok Jang","driver drowsiness eeg"] et al. | 2025 / 会議論文メタデータ確認 / 査読状況は一次PDF未確認 | 書誌メタデータのみ確認 | 非 MI だが EEG-TTA に prototype 記述あり | B | citeturn43search4turn36search6 |

**補足**  
BTTA-DG の related work では、**“MI-FTTA (Peng et al., 2025)”** が「teacher-student mutual learning + time-constrained sample selection + BN statistics recalculation + prototype-based contrastive learning」を使う MI-TTA として言及されています。ただし、この論文の**一次ソース書誌は今回の検索では独立確認できませんでした**。したがって、prototype を test-time に使う EEG MI 先例としては**二次引用ベースの弱い証拠**として扱うのが妥当です。citeturn40view3turn41search1

### BN-free / prototype-based / layer-wise TTA の一般領域

| 論文 | 著者 | 年 / venue / 査読 | 識別子 | 一行要約 | 関連度 | 出典 |
|---|---|---|---|---|---|---|
| *Test-time classifier adjustment module for model-agnostic domain generalization* | entity["people","Yusuke Iwasawa","domain generalization"] et al. | 2021 / NeurIPS / 査読済み | T3A | 最終分類器を template で補正 | A | citeturn27search16turn27search4 |
| *Parameter-free Online Test-time Adaptation* | entity["people","Malik Boudiaf","cvpr tta researcher"] et al. | 2022 / CVPR / 査読済み | DOI: 10.1109/CVPR52688.2022.00816 | 出力のみを調整する LAME | A | citeturn30search1turn30search17 |
| *Contrastive Test-Time Adaptation* | entity["people","Dian Chen","vision tta researcher"] et al. | 2022 / CVPR / 査読済み | AdaContrast | contrastive + pseudo-label queue | B | citeturn30search2turn30search10 |
| *Test Time Adaptation via Conjugate Pseudo-labels* | entity["people","Sachin Goyal","test time adaptation"] et al. | 2022 / NeurIPS / 査読済み | Conjugate PL | 疑似ラベル設計を損失側で改善 | B | citeturn27search3turn27search15 |
| *Decoupled Prototype Learning for Reliable Test-Time Adaptation* | entity["people","Guangrui Wang","prototype tta researcher"] et al. | 2024 / arXiv / preprint | arXiv:2401.08703 | prototype 論理で TTA を安定化 | A | citeturn27search14turn28search15 |
| *ProtoTTA: Prototype-Guided Test-Time Adaptation* | entity["people","Mohammad Mahdi Abootorabi","prototypical tta"] et al. | 2026 / arXiv / preprint | arXiv:2604.15494 | prototype-guided TTA を明示化 | A | citeturn27search2turn27search6 |
| *Buffer layers for Test-Time Adaptation* | entity["people","Hyun Kim","buffer layer tta"] et al. | 2025 / arXiv・OpenReview / preprint | arXiv:2510.21271 | BN 依存を避ける buffer 層追加 | B | citeturn32search10turn32search25 |

### Layer-wise / hierarchical gating と、source-side prototype・DG・EEG foundation model 関連

| 論文 | 著者 | 年 / venue / 査読 | 識別子 | 一行要約 | 関連度 | 出典 |
|---|---|---|---|---|---|---|
| *Layer-Wise Auto-Weighting for Non-Stationary Test-Time Adaptation* | entity["people","Junyoung Park","layerwise tta researcher"] et al. | 2024 / WACV / 査読済み | arXiv:2311.05858 | FIM で層ごとに更新強度を変える | A | citeturn31search11turn31search4 |
| *A Layer Selection Approach to Test Time Adaptation* | entity["people","Sabyasachi Sahoo","layer selection tta"] et al. | 2025 / AAAI / 査読済み | GALA | 層選択と unreliable sample 除外 | A | citeturn31search8turn32search6 |
| *Layerwise Early Stopping for Test Time Adaptation* | 同上グループ | 2024 / arXiv / preprint | LEAST | 層ごとに stop 時点を動的決定 | A | citeturn33search0turn33search4 |
| *Hierarchical Adaptive networks with Task vectors for Test-Time Adaptation* | entity["people","Sameer Ambekar","hierarchical tta"] et al. | 2025 / arXiv / preprint | Hi-Vec | dynamic layer selection + gate | A | citeturn31search3turn32search11 |
| *Transfer learning with optimal transportation and frequency mixup for EEG-based motor imagery recognition* | 著者情報取得は今回未完了 | 2022 / 査読論文 / 旧めだが直接関連 | DOI は今回未確認 | MI で frequency mixup を導入 | B | citeturn19search1 |
| *Cross-Subject Motor Imagery Electroencephalogram Decoding with Domain Generalization* | entity["people","Yelong Zheng","eeg dg researcher"] et al. | 2025 / *Bioengineering* / 査読済み | MDPI 12(5):495 | MI-DG を明示的に実装 | B | citeturn35search14 |
| *Feature-aware domain invariant representation learning for cross-subject EEG-based motor imagery recognition* | entity["people","Jian Li","domain invariant eeg"] et al. | 2025 / *Scientific Reports* / 査読済み | DOI は本文ページ | 多尺度 invariant representation | B | citeturn34search11 |
| *Prototype-Driven Multi-Scale Feature Alignment Network for Cross-Session EEG Motor Imagery Decoding* | entity["people","Rui Zhao","prototype mi eeg"] et al. | 2025 / ACM 系会議 / 査読済み | PMANet | source-side prototype で cross-session 整合 | A | citeturn35search5 |
| *Target Oriented Prototype Adaptation for Cross-Subject Motor Imagery EEG Decoding* | entity["people","Sheng Shi","topa mi eeg"] et al. | 2025 / SSRN / preprint | SSRN 5359095 | EEG prototype adaptation を前面化 | A | citeturn11search2 |
| *NeuroTTT: Bridging Pretraining-Downstream Task Misalignment in EEG Foundation Models via Test-Time Training* | entity["people","Suli Wang","eeg fm ttt"] et al. | 2025 / arXiv・ICLR 2026 withdrawn/desk-rejected metaあり / preprint | arXiv:2509.26301 | EEG foundation model に TTT を付加 | B | citeturn37search0turn37search10 |
| *Test-Time Adaptation for EEG Foundation Models* | 著者情報は今回省略 | 2026 / arXiv / preprint | arXiv:2604.16926 | EEG FM 上で TTA を体系比較 | A | citeturn43search1turn34search2 |

## 軸別の調査サマリ

**EEG / BCI 文脈での prototype-based test-time adaptation**  
高信頼で一次ソースまで確認できた範囲では、EEG MI/SSVEP/ERP で**「BN を完全凍結し、代わりに class prototype だけを online EMA 更新する」**論文は見当たりませんでした。直接の MI-TTA 先行は OTTA、T-TIME、BTTA-DG が中心で、OTTA と T-TIME は BN や classifier parameter の更新、BTTA-DG は BN-free だが prototype ではなく Dirichlet/GMM による確率的較正です。prototype が test-time に現れる痕跡は、BTTA-DG に二次引用された MI-FTTA と、driver drowsiness TTA の abstract 上の prototype 記述くらいで、どちらも「prototype-only replacing BN」の形ではありません。したがって、あなたの設計の核は「prototype を使うこと」自体ではなく、**EEG MI の OTTA 状態変数を BN から prototype に置換すること**にあります。citeturn1search0turn42search0turn38search1turn40view0turn40view1turn43search4

**BN-free OTTA の一般領域**  
一般 TTA では、BN 統計や BN affine を触る Tent 系と対照的に、BN を触らない路線はすでに強い系譜があります。T3A は最終分類器の supports/templates を調整し、LAME は出力分布だけを最適化し、Conjugate PL は pseudo-label 設計を損失側から安定化し、ProtoTTA や Decoupled Prototype Learning は prototype を介した適応を前面化します。これらの動機は共通で、**未知の shift での hyperparameter brittleness、small-batch BN の不安定性、破壊的な online fine-tuning**を避けることです。特に EEG foundation-model benchmark は、optimization-free な prototype 系が gradient-based 手法より安定し、T3A が平均 balanced accuracy 改善で唯一プラスだったと報告しています。したがって、EEG への転用可能性は高い一方、レビューでは「T3A を EEG に移植しただけではないか」という反論も受けやすいです。citeturn27search16turn30search1turn27search15turn27search14turn27search2turn43search1

**Hierarchical / multi-level gating in TTA**  
ここは新規性が最も立ちやすい領域です。一般 TTA には layer-wise auto-weighting、GALA、LEAST、Hi-Vec のように、**どの層をどの程度更新するか**を選択する手法はあります。しかし、それらは主に gradient 係数、layer selection、early stopping、agreement gate の設計であって、**shallow prototype と deep prototype を別々に持ち、各層で別 gate 条件を通したときだけ state を更新する**構造とは一致しません。つまり「layer-wise selective TTA」は既出ですが、「hierarchical shallow/deep prototype OTTA with separate gates」は、少なくとも今回確認できた先行群にはありません。Tri-Lock gate 自体も、confidence の単独閾値や gradient 信頼度ではなく、**pmax × SAL × Energy の複合判定**にする限り、一般 TTA でもかなり独自です。citeturn31search11turn31search8turn33search0turn31search3

**EEG 領域の inter-subject mixup / cross-subject augmentation**  
この軸は先例が薄いです。MI では frequency mixup を使った transfer learning が旧めの直接例として見つかり、最近の流れはむしろ contrastive learning、domain alignment、domain-invariant representation に寄っています。つまり、**subject shift を train-time から「擬似 shift」として混合で作る**という発想は完全に前例ゼロではないが、2024–2026 の MI mainstream ではありません。これは二面性があります。利点は新規性を主張しやすいこと。弱点は「なぜ subject mixing が sensorimotor rhythm の class semantics を壊さないのか」を自前で説得しなければならないことです。source-side prototype と組み合わせるなら、mixup 後も prototype の class-conditional compactness が保たれるかを、t-SNE や center distance、intra-class covariance などで示す必要があります。citeturn19search1turn41search4turn35search5

**Domain Generalization for EEG MI**  
2025 年前後の EEG MI DG は、domain-generalized MI decoding、domain-invariant multiscale features、supervised contrastive DG など、**source training 側で shift に強い表現を作る**方向に寄っています。重要なのは、これらの多くが test-time plasticity をほぼ持たず、train-time invariance で終わっている点です。あなたの問題意識である「source model 側が適応を受けやすい表現になっていない」という観測は、DG 文献の論点と整合的です。したがって、**DG で clusterable / prototype-friendly な表現を作り、その上で BN-free prototype OTTA を載せる**という二段構えは、文献上かなり自然です。逆に言えば、OTTA 部分だけを積んでも source 表現が悪ければ伸びない、という反論は文献側からも支持されます。citeturn35search14turn34search11turn37search0

**TCFormer および近縁 EEG transformer への TTA 適用**  
ここは空白が大きいです。TCFormer 自体は 2025 年の MI 向け新アーキテクチャとして確認できましたが、**TCFormer に OTTA を適用した一次ソース**は今回見つかっていません。EEG Conformer は BTTA-DG で baseline として参照され、sEEG speech decoding の TTA work でも比較対象に入りますが、attention module 自体を online に adapt する例は確認できませんでした。EEG foundation model では NeuroTTT と 2026 benchmark があり、CBraMod・LaBraM・BIOT 系への TTT/Tent/T3A などは出始めています。しかし、TCFormer/EEG Conformer の shallow/deep attention 特性に合わせた**hierarchical prototype OTTA**は未確認です。つまり、backbone 側でも空白はあるが、レビューで効くのは「TCFormer 初」ではなく、**TCFormer の shallow/deep feature topology を活かした適応状態設計**を示せるかどうかです。citeturn34search0turn34search7turn38search2turn36search12turn37search0turn43search1

## 最近接研究との差分マトリクス

以下の 3 件を、あなたの設計に最も近い先行とみなします。理由は、OTTA の直接性、BN-free 性、prototype 使用の三要素をそれぞれ最も強く代表しているからです。OTTA は EEG MI の直接比較対象、BTTA-DG は EEG MI の BN-free 比較対象、T3A は prototype/template-based OTTA の最重要祖先です。citeturn1search0turn38search1turn27search16

| 軸 | OTTA *Calibration-free online test-time adaptation for electroencephalography motor imagery decoding* citeturn1search0turn43search13 | BTTA-DG *Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding* citeturn38search1turn40view0 | T3A *Test-time classifier adjustment module for model-agnostic domain generalization* citeturn27search16turn27search4 |
|---|---|---|---|
| train-time | source model ほぼ通常学習＋alignment 前提。あなたの案のような inter-subject mixup や shallow/deep prototype 保存はない | SincAdaptNet を事前学習。Dirichlet/GMM を test-time 側で使うが、source-side dual prototype 保存はない | source model は通常 DG 学習。深浅 2 系統の EEG prototype artifact は持たない |
| test-time | EA/RA と BN 系 entropy-based 更新が中心。あなたの「BN 完全凍結」と真逆 | ネットワーク凍結＋Dirichlet/GMM 較正。BN-free では近いが、prototype EMA 更新ではない | backbone 凍結、classifier supports/templates を調整。prototype 発想は近いが shallow/deep 分離なし |
| gating | 明示的な hierarchical gate なし。更新可否は主に unsupervised loss と batch 統計依存 | 明示的な Tri-Lock なし。履歴分布と Bayesian weighting で間接制御 | 高 confidence support を残すが、層別 gate ではない |
| prototype 使用法 | prototype は主役ではない。二次引用上 MI-FTTA との差異としてむしろ BN 系 | class prototype ではなく Dirichlet parameter distribution を online state にする | final feature space の class templates/supports を online 更新。あなたの deep prototype には最も近いが shallow prototype がない |
| dataset | EEG MI、BCIC 系が主。データ領域は最も近い | BNCI2014001/4002, BNCI2015001, SHU MI など MI 主体 | 画像 DG ベンチ中心。EEG ではない |
| backbone | BN を含む MI backbones + alignment | LayerNorm ベースの SincAdaptNet | generic frozen backbone + linear classifier |
| あなたとの差分の本質 | **direct EEG MIだが BN 系** | **direct EEG MI かつ BN-free だが prototype系ではない** | **prototype系だが EEG MI でも hierarchical でもない** |

## 新規性ギャップ評価

**既に提案済みの要素**  
EEG/BCI での online unlabeled adaptation 自体は新規ではありません。OTTA と T-TIME が MI 向けの直接先行で、BTTA-DG は 2026 時点で BN-free・gradient-free MI-TTA をかなり強く押し出しています。一般 TTA では T3A、LAME、Decoupled Prototype Learning、ProtoTTA が、**「BN や重みを触らず、出力・template・prototype 側を動かす」**路線を確立しています。さらに EEG 側でも PMANet や TopA のように、source-side の prototype / class center を使う cross-subject・cross-session 適応はすでにあります。したがって、**prototype を使うこと**、**BN を嫌うこと**、**layer-wise selective adaptation を考えること**は、それぞれ単体では既出です。citeturn1search0turn42search0turn38search1turn27search16turn30search1turn27search14turn27search2turn35search5turn11search2turn31search11turn33search0

**組み合わせとして新規な要素**  
現時点の高信頼証拠から見ると、以下の結合は未提案です。  
第一に、**EEG MI の OTTA で BN を state variable から外し、class prototype を online state にする**こと。第二に、**shallow/deep の二階層 prototype を別個に保持し、更新も推論も分離する**こと。第三に、**gate 条件を層別に分ける hierarchical gating**。第四に、**source-side で inter-subject mixup を通して prototype-friendly な shift-robust 表現を作り、その prototype を test-time artifact として引き渡す**ことです。一般 TTA に layer selection はありますが、**dual-depth prototype memory + depth-specific gate + BN-free EMA update** という形は確認できませんでした。ここが、あなたの最も強い新規性線です。citeturn31search8turn33search4turn31search3turn35search5turn43search1

**完全未提案と見てよい要素**  
今回の調査で、一次ソースとしては見つからなかったのは次の 4 点です。  
ひとつ、**EEG MI/SSVEP/ERP で BN 更新を完全停止し、prototype EMA のみに置換する OTTA**。  
ふたつ、**shallow prototype と deep prototype を source artifact として保存し、test-time に両方持ち込む設計**。  
みっつ、**pmax × SAL × Energy の三信号を各 depth で別々に用いる Tri-Lock gate**。  
よっつ、**logits と prototype distance を shallow/deep 並列に融合する EEG OTTA**。  
この 4 点は、少なくとも今回確認できた主要文献群では一致物がありません。注意点は、これは「存在しない」の証明ではなく、**2023–2026 の主要検索・主要索引で一次ソースを確認できなかった**という意味です。特に MI-FTTA は二次引用しか取れていないため、今後の再チェックで近い先行が追加される可能性は残ります。citeturn40view3turn41search1turn43search1

**率直な批判**  
最大の反論は 4 つあります。  
第一に、**BN-free の動機が本当に構造的か**です。BTTA-DG は batch size 1 で BN-adapt が劣化し、自法が LayerNorm と確率較正で安定すると報告しています。したがって small-batch streaming なら BN-free の動機は強いです。しかし、もし buffering や micro-batch 化で BN 問題が緩和できるなら、「BN を捨てたから勝った」のではなく「batching 設計の問題だった」と反論されます。citeturn40view1turn40view3  
第二に、**prototype EMA は pseudo-label error に脆い**です。T3A でも high-confidence support 選択が要であり、prototype 系一般は誤更新が蓄積しやすい。Tri-Lock を入れても、stream 初期の低信頼・クラス不均衡・subject-specific phase shift に対しては drift が起きます。citeturn27search16turn27search2turn43search1  
第三に、**shallow prototype が nuisance を溜める危険**です。あなた自身の観測どおり shallow BN var は不安定でした。これは shallow feature が subject-specific / session-specific noise を多く含む可能性を示唆します。BN を凍結しても、shallow prototype を更新すれば別の形で nuisance memory を持ち込むだけ、という反論は成立します。したがって shallow prototype を本当に持つべきかは、deep-only との強い ablation が必要です。citeturn40view3turn43search1  
第四に、**新規性の主張点を間違えると弱い**です。「prototype を使った」「EEG で test-time adaptation」「layer-wise にした」は個別には既出です。主張すべきは、**TCFormer の shallow/deep feature geometry に合わせて、BN state を prototype state に置換し、depth-specific gate と source-side mixup を連結した OTTA system**であることです。ここから外れると、T3A・BTTA-DG・PMANet の寄せ集めに見えます。citeturn27search16turn38search1turn35search5

**Open questions / limitations**  
今回の検索で使った主なクエリは、`EEG prototype test-time adaptation`、`motor imagery prototype online adaptation`、`EEG centroid adaptation`、`source-free domain adaptation EEG`、`layer-wise test-time adaptation`、`inter-subject mixup EEG`、`TCFormer test-time adaptation EEG`、`EEG foundation models test-time adaptation` です。  
未解決なのは 3 点です。  
ひとつ、**MI-FTTA (Peng et al., 2025)** の一次ソース書誌が未回収。  
ふたつ、**Test-Time Adaptation for EEG-Based Driver Drowsiness Classification** の prototype の具体的使い方が abstract metadata 止まり。  
みっつ、**Structured Prototype-Guided Adaptation for EEG Foundation Models** という 2026 年 arXiv 題目は見つかったものの、今回は一次本文を精読していないため結論に使っていません。したがって、**「同一」は未発見だが、「prototype を EEG adaptation に持ち込む」大きな潮流自体は 2025–2026 に立ち上がり始めている**、というのが最も正確な着地です。citeturn43search19turn43search4turn41search1