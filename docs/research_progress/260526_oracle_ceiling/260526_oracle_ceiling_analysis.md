# 260526 精度向上の実現可能性分析(2a中心): 特徴を変えない補正は全滅、EA/特徴適応が唯一の道

> 位置づけ: 研究を「精度を大きく上げる」主軸に転換するにあたり、**TCFormer凍結+予測補正の枠で精度がどこまで上がるか**を、source-only logitsの3分析(oracle天井 → top2の中身 → calibration)で確定。結論は一貫: **特徴を変えない補正では上がらない。EA(特徴整列)/特徴適応が唯一の道。**

---

## 結論(3行)

- **特徴を変えない補正(static prior / per-class bias / 温度 / confidence re-ranking)は精度をほぼ上げられない。** prior天井 +3.36pp(後知恵)、near-tie 18.6%のみ、温度補正はacc不変。
- 誤分類の **68.5% は「正解が2位」**(特徴固定でも理論上届く)だが、その大半は **確信を持った誤り**(margin中央0.334、ECE 5.6%・T*≈0.89で**未calibration由来ではない**)。
- **→ 精度を上げる道は EA / 特徴適応のみ。** 混同は系統的(feet↔tongue, left↔right hand)で、特徴整列に望みがある。

実測の補強: best DC variant は 2a single-seed で +1.23pp([key_result_table.csv](../regular_seminar_2605/tables/key_result_table.csv))、L3 commit は no-op([l3_diagnostics.csv](../regular_seminar_2605/tables/l3_diagnostics.csv))。

---

## 背景・問い

DC-Replay / CMC(TCFormer凍結 + posterior/prior/memory補正)で精度がどこまで上がるかを、願望でなくデータで上から押さえる。すべて **source-only logits の read-only 分析**(GPU不要)。oracle系は **後知恵オラクル=上界**で、オンライン無教師では必ずこれより小さい。

---

## 分析1: Oracle天井 — どこまで届くか

スクリプト [oracle_ceiling.py](../../../intentflow/offline/scripts/analysis/oracle_ceiling.py) / [oracle_ceiling_all.csv](oracle_ceiling_all.csv)

| dataset | クラス | source acc | Δ top2(特徴固定の絶対上限) | Δ per-class bias(L1上限) | Δ static prior |
|---|---:|---:|---:|---:|---:|
| 2a | 4 | 82.72 | **+11.84** | +3.36 | +0.00 |
| 2b | 2 | 87.74 | 自明(100%) | +1.73 | +0.00 |
| HGD | 4 | 94.32 | +5.00 | +3.18 | +0.00 |

- 静的global prior補正は3データセットでΔ0(テストがクラスバランスのため無価値)。
- per-class bias(L1の上限)は3データセットで +1.7〜3.4pp に留まる。**L1天井が低いのは2a特有でなく汎用的。**
- top2(特徴固定の絶対上限)は source精度に逆相関(82.72→+11.84、94.32→+5.00)。

検証: 2a source_acc平均=82.72 が key_result_table の `source_only` と完全一致(順序整合OK)。

## 分析2: top2の中身 — 「届く分」の正体

スクリプト [top2_breakdown.py](../../../intentflow/offline/scripts/analysis/top2_breakdown.py) / [top2_breakdown_2a.csv](top2_breakdown_2a.csv)

- 誤分類 448 / 2592。うち **正解が2位 = 307(誤分類の68.5%、全体の11.84pp)** が特徴固定で理論上届く。
- margin = p(誤1位) − p(正解2位) の **中央値 0.334**。僅差(margin<0.1)は **18.6%のみ**、<0.2 でも 36.2%。
- → **大半(63.8%)はモデルが確信を持って間違えている。** 僅差の取りこぼしではない。
- 系統的混同ペア(救済可能trial、pred→true): tongue→feet(39)、right_hand→feet(36)、left_hand→tongue(33)、left_hand→right_hand(32)。ランダムでなく構造的。

含意: 温度/confidence-basedの安直なre-rankingで取れるのは **≈ +2pp**(near-tie 18.6% × 11.84pp)止まり。

## 分析3: calibration — 確信誤りは本物か

スクリプト [calibration_analysis.py](../../../intentflow/offline/scripts/analysis/calibration_analysis.py) / [calibration_2a.csv](calibration_2a.csv)

| 指標 | 平均 | 読み |
|---|---:|---|
| ECE | 0.056 | 中程度。過剰自信は深刻でない |
| 最適温度 T* | 0.89 | **<1 = 過剰自信ではない**(むしろ自信不足気味。T*>1はS5/S6のみ) |
| ECE@T* | 0.045 | 温度補正で直る分はわずか0.011 |
| acc(温度補正後) | 全被験者 不変 | **温度はargmaxを変えない=精度に無力** |
| 確信誤りtrialのconf | 0.623 | 62%の確信で誤る(chance 25%の2.5倍) |

→ 「margin大=未calibration」仮説は棄却。**確信誤りは本物**。calibration/温度補正では精度は上がらない(acc不変、確定)。

---

## 統合解釈: 精度向上の経路マップ(2a)

| 精度向上の経路 | 2aでの天井/効果 | 判定 |
|---|---|---|
| 静的global prior補正 | Δ0.00 | **死亡** |
| per-class bias (L1) | +3.36(後知恵上界) | 低すぎ |
| 温度 / calibration補正 | acc不変(Δ0) | **無力** |
| confidence-based re-ranking | ~+2pp(near-tieのみ) | 不足 |
| 特徴固定の絶対上限(top2) | +11.84 | だが63.8%は確信誤りで特徴情報が必須 |
| **EA(特徴整列)/ 特徴適応** | top2超えを狙える | **唯一の道** |

3分析が一点に収束: **特徴を変えない限り精度は実質上がらない。** 混同が系統的(クラス対)なので、EA で分布ずれを整えればこの確信誤りを減らせる可能性がある(cross-sessionでnaïve prototype=T3Aが悪化したのは整列前のため、という外部所見とも整合)。

---

## データ品質・限界

- oracle/top2/calibration はすべて **後知恵=上界**。実運用はこれより小さい。
- 2a は最新の充実sweep、2b/HGD は single-seed firstpass(HGDは14中11被験者)。
- best_temp は test label で最適化した上界(過剰自信の判定には十分)。ECEは288 trial/被験者でbin推定にノイズ。
- calibration は精度を上げないが、**制約側(abstain-safe / decision safety)の土台**としては使える(ECE 5.6%)。

## 研究方向への含意

- 精度主軸で進むなら、**次はEA(特徴整列)の実測が第一手**(現状リポジトリにEA未実装)。軽量制約にも最も合う。
- 「特徴を変えない補正の天井 + 確信誤りの実証」自体が、commitless系を捨てる強い根拠であり、先行検証として論文の付録になりうる。

## 再現方法

```bash
conda activate intentflow
python intentflow/offline/scripts/analysis/oracle_ceiling.py       # 天井
python intentflow/offline/scripts/analysis/top2_breakdown.py       # top2の中身
python intentflow/offline/scripts/analysis/calibration_analysis.py # calibration
```
