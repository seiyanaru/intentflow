# 260602 seed1再現: EA portfolioの現状整理と次の方針

## 結論

固定比率 `p = 0.3 source + 0.4 fullEA + 0.3 shrinkEA` は、seed0では強かったがseed1では最適ではない。  
一方で、source / fullEA / shrinkEA の予測は補完的で、EA単体が負けるseed1でも融合はsourceを上回った。  
したがって次に進むべき主軸は、RAA単体の共分散重み付けを強くすることではなく、`Reliability-conditioned adaptation portfolio` として「どのadaptationをどれだけ信じるか」をtest-timeに決める設計である。

## seed別結果

| seed | source | fullEA | shrinkEA(0.1) | static 0.30/0.40/0.30 | static 0.45/0.45/0.10 | grid best | LOSO |
|---|---:|---:|---:|---:|---:|---|---:|
| seed0 | 82.72 | 83.06 | 81.44 | 85.76 | 85.92 | 0.46/0.43/0.11 = 85.96 | 85.61 |
| seed1 | 83.87 | 81.83 | 81.79 | 84.57 | 85.30 | 0.65/0.25/0.10 = 85.84 | 85.61 |

seed1ではEA単体がsourceより -2.04pp 落ちた。それでも `0.45/0.45/0.10` は85.30%、grid bestは85.84%まで上がった。つまり「EAが常に良い」のではなく、「EAが正しいtrial/subjectではsourceと違う正解を拾う」ことが価値。

## seed0+seed1を合わせた解釈

| 目的 | 重み(source/fullEA/shrinkEA) | 平均 | source悪化subject数 | 解釈 |
|---|---|---:|---:|---|
| 精度最大 | 0.48/0.29/0.23 | 85.84 | 3/18 | 精度は高いが被験者によってsourceを下回る |
| source非悪化優先 | 0.65/0.25/0.10 | 85.30 | 0/18 | 保守的。online BCI向き |
| バランス固定 | 0.45/0.45/0.10 | 85.61 | 3/18 | seed0/seed1の中間として妥当 |

ここから、固定比率を論文の主張にするのは弱い。主張すべきは「信頼性に応じてadaptation strengthを割り振る」こと。

## 既存online portfolioの失敗

既存のlabel-free Hedge型更新はまだ弱い。

| seed | online Hedge | uniform融合 |
|---|---:|---:|
| seed0 | 84.07 | 85.73 |
| seed1 | 84.53 | 85.03 |

entropy / disagreement / prior collapseで逐次的にalphaを動かすだけでは、良い重みに寄らない。次は逐次更新そのものより、最初の数trial・共分散・チャネル信頼性から「重み候補」を選ぶselectorを作るべき。

## 次に作る手法案

名前案: Reliability-Conditioned Adaptation Portfolio (RCAP)

専門家を以下のように置く。

`E0 = source`, `E1 = fullEA`, `E2 = shrinkEA`, `E3 = RAA`, `E4 = TTA/BN/prototype系`

最終予測は

`p_t(y|x) = sum_k alpha_{s,t,k} p_k(y|x_t)`

で出す。重要なのは `alpha` の決め方。

まずv1ではtrialごとに自由更新しない。被験者単位で

`alpha_s = g(z_s)`

とする。`z_s` はlabel-free診断量:

- train/test covariance distance
- EA前後のcondition number, diag CV
- channel reliabilityのmin/mean/low_count
- source/fullEA/shrinkEA間のdisagreement
- 最初のN trialのentropy, margin, prediction prior collapse

`g` はまず連続回帰ではなく、候補重みの選択にする。

- safe: `0.65/0.25/0.10`
- balanced: `0.45/0.45/0.10`
- accuracy: `0.48/0.29/0.23`
- EA-heavy: `0.30/0.40/0.30`

評価は、seed0でselector設計、seed1で検証。さらにBCIC2b/HGDに横展開する。test labelで重みを選んだら終わりなので、selector入力は必ずlabel-freeに限定する。

## RAAとの接続

RAAは単体主役にしない。`E3 = RAA expert` としてportfolioに入れる。  
理由は、現状の「共分散を重み付けするだけ」のRAAは改善機構が薄く、EAが壊れる被験者では助ける可能性がある一方、全体平均を押し上げる保証が弱いから。

RAAの価値は、selectorの特徴量にもなる。

- RAAとfullEAの予測差が大きい
- RAAでentropyが下がる
- reliability low channelが多い
- fullEAのcovariance conditionが悪い

この条件でRAA重みを上げる、という形ならTTA拡張にも自然につながる。

## 直近の実験順

1. seed0/seed1のsubject-level診断テーブルを作る  
   入力: cov診断、channel reliability、各expertのentropy/disagreement/prior collapse。labelは評価列だけ。

2. selector v0を作る  
   ルールベースで `safe/balanced/accuracy/EA-heavy` を選ぶ。まずはseed0でルールを作り、seed1で固定検証。

3. RAA expertを追加する  
   ただし単体平均ではなく、portfolio内で「RAAが効く被験者だけ拾えるか」を見る。

4. TTA expertを追加する  
   BN/prototype/logit補正を単独手法として競わせるのではなく、`E4` としてalphaで採用率を制御する。

## 警告

- 固定混合だけだと新規性は弱い。
- seed0/seed1の両方を見てから比率を決めると、BCIC2aへの後付け最適化に見える。
- online Hedgeのような逐次更新は、現状では精度を落としている。
- RAA単体の平均改善を追いすぎると泥沼になる。RAAは「壊れるEAを避ける/補うexpert」として扱う。
- 論文にするなら、selectorがlabel-freeで、別seed/別datasetに移ってもsourceより悪化しにくいことを示す必要がある。

## 実行済み: selector feature table

`docs/research_progress/260602_selector_feature_table_seed0_seed1.csv` を作成した。  
18行(seed0/seed1 x 9 subjects)で、各行にはlabel-free特徴量と評価用targetを入れている。

候補重みのoracle分布:

| candidate | count |
|---|---:|
| safe 0.65/0.25/0.10 | 6 |
| balanced 0.45/0.45/0.10 | 4 |
| accuracy 0.48/0.29/0.23 | 4 |
| EA-heavy 0.30/0.40/0.30 | 2 |
| uniform | 2 |

重要な観察: 同じ被験者でもseedで最良候補が変わる。したがって、covariance / channel reliability だけで割り振るselectorは弱い。`source/fullEA/shrinkEA` の実際の出力から取るentropy, disagreement, prediction-prior collapseを必ず使うべき。
