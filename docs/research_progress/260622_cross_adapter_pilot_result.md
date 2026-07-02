# cross-adapter engineering pilot 結果 — 2026-06-22

## 結論

**62人×3 seedsの本実験には進まない。** prospective実装とrisk評価は成立したが、事前固定したengineering gateを満たさなかった。追加のpopulation-pretrain source repairもmedian 59.41%で、60%基準に届かなかった。

失敗は2点ある。

1. EEGNet source精度の被験者medianが **54.70%** で、事前基準60%に未達。
2. prefix-Tentは事前grid全体で、cross-adapter比較に使える「十分な更新量と被験者内再現性」を同時に示さなかった。

一方、EAとAdaBNでは被験者順位が各adapter内で再現し、両者間では一致しないという重要なpilot信号が得られた。これは現時点では **traitよりperson×adapter interactionを示唆**するが、n=16・1 seed・非nestedの探索結果なので主張には使わない。

---

## 実装したもの

- target session先頭32試行だけを使うprospective adaptation
- trial 33以降を固定stateで評価
- sessionごとのsource checkpoint完全resetとhash照合
- EA / AdaBN / prefix-Tentの同一checkpoint分岐
- subject-balanced mean utility、LCVaR10、R10、q05、閾値別harm率
- future trialを変更してもprefix stateが変わらないunit test
- sharded runの統合、固定効果近似を除いたsubject順位相関、奇偶session split-half
- 保存source checkpointを再利用するadapter-only rerun

主要コード：

- `intentflow/offline/scripts/analysis/cross_adapter_core.py`
- `intentflow/offline/scripts/analysis/stieger_cross_adapter_pilot.py`
- `intentflow/offline/scripts/analysis/merge_cross_adapter_pilot.py`
- `intentflow/offline/scripts/analysis/stieger_population_source_pilot.py`
- `intentflow/offline/scripts/analysis/merge_population_source_pilot.py`
- `tests/test_cross_adapter_core.py`

テスト結果：`4 passed`。16/16人、148 target sessionsが完走し、全sessionでreset hash一致。

---

## 16人pilot（local EEGNet、seed 0）

| adapter | subject-balanced U | R10 | mean \|session Δ\| | SD(subject mean Δ) | residual split-half ρ_SB |
|---|---:|---:|---:|---:|---:|
| EA | +9.013pp | 3.510 | 9.384pp | 6.486pp | 0.887 |
| AdaBN | +1.801pp | 6.467 | 3.350pp | 3.260pp | 0.811 |
| prefix-Tent (lr=1e-3, 1 step) | −0.149pp | 1.446 | 0.510pp | 0.272pp | 0.012 |

source accuracy：

- 被験者median：**54.70%**
- 被験者mean：55.70%
- range：48.95–66.01%
- 事前基準 `median ≥60%`：**不合格**

固定効果（source accuracy、source accuracy²、session index、n_eval）の線形近似をadapterごとに除いた後の、被験者平均残差Spearman：

| pair | ρ | p（探索値） |
|---|---:|---:|
| EA–AdaBN | 0.103 | 0.704 |
| EA–Tent | 0.009 | 0.974 |
| AdaBN–Tent | 0.465 | 0.070 |

EAとAdaBNはadapter内split-halfが高いのに、cross-adapter順位はほぼ無相関だった。単なる測定ノイズだけでは説明しにくく、**adapter固有相性の方向を示すpilot evidence**である。ただし16人なのでCIによるtrait判定はしない。

---

## Tent事前grid

同一source checkpoint、同一prefix、同一evaluation splitで比較した。

| lr | steps | U | R10 | mean \|session Δ\| | SD(subject mean Δ) | residual ρ_SB |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 1 | −0.018 | 0.539 | 0.097 | 0.089 | 0.063 |
| 1e-4 | 3 | −0.063 | 0.772 | 0.186 | 0.154 | 0.674 |
| 1e-3 | 1 | −0.149 | 1.446 | 0.510 | 0.272 | 0.012 |
| 1e-3 | 3 | −0.378 | 3.106 | 0.905 | 0.439 | 0.291 |

`1e-4, 3 steps`だけsplit-halfは高いが、平均絶対変化0.186pp・被験者平均SD 0.154ppで実質no-opに近い。更新量を増やすと再現性が消え、平均も悪化する。

**判定：現実装のprefix-Tentを本解析の3本目に数えない。** no-op adapterとの低相関を「adapter固有」と解釈するのは不正。

---

## source弱さの切り分け

S1–S4で学習不足とEEGNet実装差を確認した。

| source設定 | target source accuracy median |
|---|---:|
| local EEGNet、120 epoch、interaugあり | 56.39% |
| local EEGNet、500 epoch、augmentationなし | 56.30% |
| Braindecode標準EEGNet、500 epoch、augmentationなし | 56.32% |

500 epochでは全条件でsource session学習精度100%だが、target session精度は改善しない。したがって原因は単純なunderfittingやローカル実装差ではなく、**1被験者×session 1だけで学習するdeep source設計の汎化不足**である。

参考として、既存Riemann sourceの同じ16人・trial 33以降の被験者medianは56.70%だった。60%基準自体は厳しいが、結果を見た後で閾値だけ下げることはしない。

### population-pretrain source repair

16人を4 foldsに分け、各foldで他12人のsession 1だけを用いてBraindecode EEGNetをpretrainし、held-out 4人を各自のsession 1だけでfine-tuneした。target session入力・ラベルは学習とmodel selectionに使っていない。

| 指標 | subject-specific source | population-pretrain + subject fine-tune |
|---|---:|---:|
| mean | 55.70% | **60.10%** |
| median | 54.70% | **59.41%** |
| range | 48.95–66.01% | 51.95–73.44% |

- paired mean改善：+4.39pp（subject bootstrap 95%CI [1.46, 7.21]）
- paired median改善：+2.94pp
- 改善した被験者：13/16
- repaired median bootstrap 95%CI：[57.58, 62.10]
- 事前合格条件 `median ≥60%`：**不合格**

修正方向は効いたが、閾値には0.59pp届かなかった。CIが60を跨ぐことは「合格」と同義ではない。seed 0の事前判定値は59.41%なので、ここで止める。

---

## 決定と次アクション

### 守る

- prospective prefix評価
- subject-balanced risk–utility
- cross-adapter consistencyという問い自体と、再開時に使える実装
- EA/AdaBNで見えた「adapter内は再現、adapter間は不一致」という問い

### 撤回

- 現在のsubject-specific EEGNet sourceのまま62人へ拡大すること
- prefix-Tentを有効な3本目として扱うこと
- 16人pilot相関からtrait/interactionを結論すること

### 最終判断

source repairは実施済みで不合格だった。したがって事前分岐どおり、**deep cross-adapter trait実験は修士期間では打ち切る**。

- 62人×3 seedsは実行しない。
- pseudo-label head adaptationを含む第3adapter開発には進まない。
- 16人のEA/AdaBN不一致は、修論の探索的補助結果またはappendixに限定する。
- 主たる出口を、既存Riemann-EAを中心とした **prospective longitudinal risk audit + risk–utility frontier** に戻す。
- 新しいgateや深層OTTA改良を追加せず、再現可能なnegative auditとして結果を固める。

これは「population pretrainingが無意味」という結論ではない。平均+4.39ppの改善は明確である。しかし現在の研究目的はsource model改良ではなく、信頼できるcross-adapter harm trait検定である。59.41%のsourceと有効な第3adapter不在のまま本実験を拡大すると、査読で最も弱い部分に計算資源を投じることになる。
