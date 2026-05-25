# 260526 Oracle天井分析: 「特徴を変えない補正」の理論限界(2a/2b/HGD)

## 結論(3行)

- **L1(prior/logit-bias補正)の後知恵天井は、3データセット全部で +1.7〜3.4pp しかない。** 天井が低いのは2a特有ではなく汎用的。
- **「特徴を変えない補正」の絶対上限(top2 oracle)は source 精度に強く依存**: 2a +11.84pp、HGD +5.00pp、2b は2クラスで自明。source が強いほど余地が縮む。
- **→ commitless posterior/prior correction で「+5ppを汎用的に」上げるのは物理的に不可能。** 「大きく上げる」には特徴適応が必須(=安全性トレードオフ・競合多)。

これは実測の補強知見と整合する: 実測 best DC variant は 2a single-seed で **+1.23pp**([key_result_table.csv](../regular_seminar_2605/tables/key_result_table.csv))= L1天井(+3.36)の37%、top2天井(+11.84)の10%。L3 model-state commit は精度に **no-op**([l3_diagnostics.csv](../regular_seminar_2605/tables/l3_diagnostics.csv))。

---

## 問いと位置づけ

DC-Replay / CMC は「TCFormer本体を凍結したまま posterior / prior / memory で予測を補正する」OTTA。**この枠組みで原理的に何ppまで上げられるか**を、source-only logits + 真ラベルで上から押さえる。願望ではなくデータで天井を決めるのが目的。

得られた天井は **すべて後知恵オラクル(hindsight upper bound)** であり、オンライン無教師の実運用では**必ずこれより小さい**。「ここまでしか伸びない」を示す上界として読む。

---

## 方法

- **入力**: 各被験者の source-only(適応なし)全クラス logits `.npy` と真ラベル(variant npz の `label` 列)。読み取りのみ、GPU不要、再ラン無し。
- **検証**: 2a の source_acc 平均 = 82.72 が key_result_table の `source_only`(82.72)と完全一致 → logits とラベルの順序整合を確認済み。
- **天井の定義**:
  - `top2_oracle`: 真クラスが source top-2 に入る割合。**特徴を変えない任意の re-ranking の絶対上限**(2クラスでは常に100%=無意味)。
  - `prior_shift_oracle`: クラス別定数バイアス b を後知恵で座標降下最適化した argmax 精度。**L1(prior/logit-bias補正)の上限**。
  - `uniform_prior_acc`: 真のテスト事前分布で global prior 補正した精度。バランスドデータでの sanity。
- スクリプト: [oracle_ceiling.py](../../../intentflow/offline/scripts/analysis/oracle_ceiling.py) / 数値: [oracle_ceiling_all.csv](oracle_ceiling_all.csv)

---

## 結果

### クロスデータセット(平均 Δ vs source)

| dataset | クラス | 被験者 | source acc | Δ top2(特徴固定の絶対上限) | Δ prior_shift(L1上限) | Δ uniform |
|---|---:|---:|---:|---:|---:|---:|
| BCIC2a | 4 | 9 | 82.72 | **+11.84** | **+3.36** | +0.00 |
| BCIC2b | 2 | 9 | 87.74 | 自明(100%) | **+1.73** | +0.00 |
| HGD | 4 | 11 | 94.32 | **+5.00** | **+3.18** | +0.00 |

### 被験者別の要点

- **2a**: S6(src 70.49)と S2(src 72.57)が難。特に S6 は top2余地 +18.40 と大きいのに prior_shift は +1.74 = 「正解は2位に入るが定数バイアスでは救えない」=**特徴が混線**。bias補正では届かない被験者。
- **2b**: S4(98.12)/S5(97.81)は既に高精度で余地ほぼゼロ。S2(70.00)は top2 +30 だが prior_shift +1.79。
- **HGD**: S4/S5/S9 は既に **100%**(適応余地ゼロ)。一方 S2/S7/S11 は prior_shift 余地が大(+6.88/+8.12/+6.88)。

---

## 解釈(観測と解釈を分ける)

**観測(データが言っていること):**
1. `uniform prior` は3データセットで Δ0.00。静的グローバルprior補正は無価値。
2. `prior_shift` 天井は 3データセットで +1.7〜3.4pp に収まる。
3. `top2` 天井は source 精度と逆相関(82.72→+11.84、94.32→+5.00)。

**解釈(観測からの推論、確度中):**
1. L1の `static_prior` モードは原理的に効かない。`memory_prior` でオンラインに偏らせても、argmax を動かす力は per-class bias 空間に射影されるので **+1.7〜3.4pp が上界**。
2. 実測 +1.23pp は L1天井の範囲内 = **prototype補正を持ち出さずとも prior 補正だけで説明可能**。+1.23pp の「正体」は posterior の定数シフトで足りる。
3. source が強い設定(HGD)では「特徴を変えない補正」自体の余地が乏しい。汎用性を謳うほど、commitless では伸びしろが消える。

---

## データ品質・限界

- すべて **後知恵オラクル=上界**。実運用のオンライン無教師補正は必ずこれより小さい。
- **2a** は最新の充実 sweep(`c_aug_true_9subj`)。**2b/HGD は single-seed firstpass** で粗い初期推定。HGD は14被験者中11のみ(3被験者は logits/label 欠損)。
- 完全な汎用性検証には、2b/HGD のマルチシード source-only 評価、および OpenBMI/Lee2019 の追加ランが必要(現状データ無し)。

---

## 研究方向への含意

- **commitless correction(L1+L2)を「次の提案手法」として磨くのは、3データセットで天井が低いと確証された以上、筋が悪い。**
- 「大きく上げる」を汎用的に取るなら **特徴適応(TTT/self-supervised/alignment)が必須**。ただし安全性トレードオフ・競合多(EEGでも先行多数)。
- 逆に、**「特徴を変えない補正の天井はこれだけ低い」+「L3 commit は no-op」という negative result 群が、手持ちデータでそのまま主張になる**。新規性を手法ではなく「理解と評価」に置く方向と整合。

---

## 再現方法

```bash
conda activate intentflow
python intentflow/offline/scripts/analysis/oracle_ceiling.py
```
パスは [oracle_ceiling.py](../../../intentflow/offline/scripts/analysis/oracle_ceiling.py) 冒頭の `R_2A / R_2B / R_HGD` で指定。
