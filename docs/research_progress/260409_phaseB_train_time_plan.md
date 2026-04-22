# 2026-04-09 Phase B Train-Time Robustness Plan

## 背景

このメモは、Phase A の proxy validation、BN drift direction ablation、hybrid/dual-momentum の結果を踏まえて、

> 次にどのような train-time 実装を行い、どの順序で検証すべきか

を整理したもの。

主眼は「いきなり複雑な train-time adaptation を入れる」のではなく、

1. まず train-time で **shallow を壊れにくくする**
2. その上で既存ベストの **test-time hybrid@0.01** を足して上積みできるか確認する

という段階的な設計にある。

---

## gpt

### 統合結論

Claude のレビューまで踏まえて結論を一本化すると、次に進むべき方向は次の通り。

1. **Phase B-1 を先にやる**
   - `plain`
   - `aug-only(std=0.025)`
   - `aug-only(std=0.05)`
2. **次に Phase B-2 をやる**
   - `aug + shallow invariance`
3. **まだ virtual BN には進まない**
   - Phase A は partial pass であり、proxy は universal ではない
   - まずは `gain jitter` と `shallow-late invariance` の寄与を分離すべき

この順序にする理由は明確で、現在の strongest finding は

- `S2 harm` の主因が `shallow running_var`
- `hybrid@0.01` が test-time best
- `gain_std=0.05` の proxy が多数派 subject には通る
- ただし `S1/S8` では proxy fail

だからである。

したがって、今作るべきモデルは

> **TCFormerAugShInv = plain TCFormer + mild gain jitter + optional shallow-late invariance**

であり、deep adaptation は引き続き test-time `hybrid@0.01` 側に任せる。

### ひとことで

次に作るべきモデルは、`virtual BN` を持つ複雑な adaptive model ではない。  
まずは **`TCFormerAugShInv`** を作るべきである。

これは

- backbone は既存の `TCFormer`
- 学習時にだけ mild な `channel_gain_jitter`
- 必要に応じて shallow 後段への invariance regularization
- 推論時は既存ベスト `hybrid@0.01`

という、train-time robustness と test-time adaptation を分離した設計である。

---

## 1. 今なにが確定しているか

### 直接確認済み

- `S2 harm` の主因は `shallow running_var`
- `hybrid@0.01 (shallow_mean_deep_both)` が test-time best
- dual momentum は multi-seed で hybrid を安定して超えない
- test-time heuristic 側の改善余地はかなり飽和
- Phase A では `gain jitter(std=0.05)` が `6/9~7/9` で proxy として機能

### 強い推論

- session shift は少なくとも一部被験者では `channel gain` 方向に近い
- ただし universal ではない
- `S1/S8` は gain jitter proxy failure 群
- `shallow` の中でも、効いているのは前段ではなく後段

### まだ未確定

- gain jitter を train-time に入れると本当に精度が上がるか
- `L_shallow_inv` が `S2` 型 harm を減らせるか
- `virtual BN` を入れる価値があるか

---

## 2. 中心仮説

今回の結果をまとめると、train-time に学習させるべき性質は次の 2 つである。

### A. Shallow-late invariant

浅い層全体ではなく、**shallow 後段**が session shift に対して壊れにくいこと。

対象:

- `conv_block.channel_DW_conv.1`
- `conv_block.channel_reduction_2.1`
- `conv_block.temporal_conv_2.1`

理由:

- Phase A で proxy として最も強く反応したのがこの 3 層
- gain が強すぎると真っ先に壊れるのもこの 3 層
- S2 harm の causal finding とも整合

### B. Deep adaptive

deep 側の adaptation 利得は、train-time ではまだ壊さない。  
deep adaptation は **推論時の hybrid@0.01** に任せる。

理由:

- S3/S7 の gain source は deep 側
- しかし train-time から deep adaptation を入れると効果帰属が崩れる
- 今の段階では robustness と adaptivity を同時に最適化しない方がよい

---

## 3. 次に作るべきモデル

### モデル名

`TCFormerAugShInv`

### 設計

- backbone: 既存 `TCFormer`
- train-time augmentation: `channel_gain_jitter`
- optional regularizer: `L_shallow_inv`
- inference-time: 既存 `hybrid@0.01`

### 重要な点

- backbone 構造は変えない
- test-time OTTA 実装も変えない
- まず変えるのは **training_step だけ**

つまり今回は、新しいアーキテクチャを発明するのではなく、

> 既存 TCFormer を、session shift に対して少しだけ頑健に再学習する

ことが目的である。

---

## 4. 実験計画

### Phase B-1: augmentation-only baseline

#### 目的

gain jitter を train-time に混ぜるだけで、raw に効くか確認する。

#### 学習 loss

\[
L = CE(x_{clean}, y) + \lambda_{aug} \, CE(x_{aug}, y)
\]

推奨初期値:

- `lambda_aug = 0.5`

#### augmentation

- `channel_gain_jitter(std=0.025)`
- `channel_gain_jitter(std=0.05)`

ここで `0.10` は試さない。Phase A の時点で被験者依存リスクが増えているため。

#### 比較条件

1. `plain`
2. `aug-only(std=0.025)`
3. `aug-only(std=0.05)`

#### 評価

- `source_only`
- `hybrid@0.01`

#### 見るべき指標

- all-9 平均精度
- majority 群: `S2/S3/S4/S6/S7/S9`
- risk 群: `S1/S5/S8`
- `S2` 単体の delta
- `WSD`

---

### Phase B-2: augmentation + shallow invariance

#### 目的

causal finding を train-time に活かせるか確認する。

#### 学習 loss

\[
L = CE(x_{clean}, y) + \lambda_{aug} \, CE(x_{aug}, y) + \lambda_{sinv} \, L_{shallow\_inv}
\]

推奨初期値:

- `lambda_aug = 0.5`
- `lambda_sinv = 0.1`

#### `L_shallow_inv` の対象

**shallow 全体ではなく、shallow 後段を代表する特徴**にかける。

最初の実装案:

- `conv_block` の出力を使う
- clean 側を anchor (`detach`)
- aug 側を clean に寄せる
- 比較量は `mean + logvar`

式のイメージ:

\[
L_{shallow\_inv}
=
\| \mu_{aug} - \mu_{clean} \|_2^2
+
\| \log \sigma^2_{aug} - \log \sigma^2_{clean} \|_2^2
\]

#### 比較条件

1. `plain`
2. `aug-only(std=0.025)`
3. `aug-only(std=0.05)`
4. `aug+shinv(std=0.025)`
5. `aug+shinv(std=0.05)`

#### 評価

同じく

- `source_only`
- `hybrid@0.01`

---

## 5. 成功条件

### 基本判定

- `B-1 > plain` なら augmentation に価値あり
- `B-2 > B-1` なら shallow invariance に追加価値あり

### より重要な判定

- majority 群で改善するか
- `S1/S5/S8` が大崩れしないか
- `hybrid@0.01` を被せても悪化しないか

### 特に重視するシグナル

- `S2` の source-only delta が改善するか

これは最重要 canary だが、**唯一の判定基準ではない**。  
理由は、今回の proxy 品質が被験者群によって分かれているため。

---

## 6. なぜこの順序がよいか

### 理由 1: 効果帰属ができる

いきなり `virtual BN` や `consistency` を入れると、

- augmentation が効いたのか
- shallow invariance が効いたのか
- train-time adaptation が効いたのか

が分からなくなる。

### 理由 2: Phase A は partial pass に留まっている

今ある proxy は universal ではない。  
したがって、

> まず mild augmentation だけで robustness が作れるか

を見るのが筋であり、いきなり adaptive training に進むのは早い。

### 理由 3: test-time 側の best はすでにある

今の推論時ベストは `hybrid@0.01` でかなり安定している。  
だから train-time の役割は

- deep adaptation の代替

ではなく

- shallow を壊れにくくして hybrid の上に積むこと

である。

---

## 7. まだやらないこと

今回の段階では、以下は入れない方がよい。

- differentiable virtual BN
- learned alpha
- dual momentum の train-time 化
- consistency loss
- 全 shallow 層への一律 invariance

理由:

- どれも因子が増えすぎる
- 今必要なのは「adaptive learning の最大化」ではなく「robustness の足場固め」

---

## 8. Phase C に進む条件

`virtual BN` に進むのは、少なくとも次のいずれかが成り立った後。

1. `aug-only` が plain を超える
2. `aug+shinv` が `aug-only` を超える
3. `S2` が source-only で改善する
4. `hybrid@0.01` を被せても train-time 改善が壊れない

このどれも満たさない場合、Phase C に進んでも複雑化だけが増える可能性が高い。

---

## 9. 実装の最小単位

### 追加・変更候補

- `models/tcformer/classification_module.py`
  - `shared_step` に `aug-only` / `aug+shinv` 分岐を追加
- `offline/utils/augmentations.py` または同等の新規 util
  - `channel_gain_jitter`
- `configs/tcformer_ttta/`
  - `tcformer_aug_only.yaml`
  - `tcformer_aug_shinv.yaml`

### 実装ポリシー

- まずは既存 `TCFormer` を使う
- 新しい backbone は作らない
- train-time adaptation 専用の新しい大きな class はまだ作らない

---

## 10. 最終提案

次の一手として最も筋がよいのは:

1. `Phase B-1`: `plain vs aug-only(std=0.025, 0.05)`
2. `Phase B-2`: `aug+shinv(std=0.025, 0.05)`
3. 各条件を `source_only` と `hybrid@0.01` で評価
4. majority 群 / risk 群 / S2 を分けて読む

この順番で成功した場合にのみ、次に `Phase C` の deep-only virtual BN を検討する。

要するに、今の段階で作るべきものは

> **train-time adaptive model ではなく、train-time robust TCFormer**

である。

---

## claude

### 全体評価

GPT-5.4 の計画は論理的に整合しており、方向性に同意する。特に「train-time adaptive model ではなく train-time robust TCFormer」という位置づけは正確。

以下、同意する点・修正すべき点・追加すべき観点を述べる。

---

### 同意する点

**1. Phase B を B-1 / B-2 に分ける構造**

これは前回の議論で私が強く推した点であり、GPT-5.4 の計画に正しく反映されている。一度に augmentation + consistency + shallow_inv + virtual BN を入れる元の案より遥かに堅実。

**2. consistency loss を Phase B で入れない判断**

gain_std=0.05 では clean と aug の予測差がそもそも小さいため、consistency loss は gradient として有意に機能しない。この判断は正しい。

**3. deep adaptation を推論時に任せる判断**

Phase A で deep 層の proxy 品質は高かったが、train-time で deep adaptation を入れると効果帰属が崩れる。今の段階では shallow robustness に集中し、deep は test-time hybrid@0.01 に委ねるのが正しい。

**4. Phase C へのゲート条件**

「B-1 > plain」「B-2 > B-1」「S2 改善」「hybrid を被せても壊れない」の 4 条件のいずれかを要求する構造は適切。全て fail なら Phase C に進む根拠がない、という判断も正しい。

---

### 修正すべき点

**1. gain_std=0.025 を B-1 に入れる必要があるか**

GPT-5.4 は `std=0.025` と `std=0.05` の 2 条件を提案している。しかし Phase A は `0.025` を実施していない。proxy validation なしで学習に入ると、Phase A → Phase B の因果的な接続が弱くなる。

**私の推奨:** Phase B-1 はまず `std=0.05` の 1 条件だけで走る。

理由：
- Phase A で proxy 品質を確認したのは `0.05` のみ
- `0.025` は「0.05 が強すぎた場合の退路」として意味があるが、Phase A の結果では `0.05` の improvement は既に 1e-4 オーダーと小さい。これより弱い `0.025` は効かない可能性が高い
- 条件数を 2 → 1 に減らすことで、Phase B-1 の実行時間と解釈コストが半減する
- `0.05` で効果が出たが S1/S8 が悪化した場合にのみ `0.025` を追試する、という順序の方が効率的

**2. `L_shallow_inv` のターゲットの精密化**

GPT-5.4 は「conv_block の出力を使う」としているが、conv_block の出力は shallow 全体（層 0–5）を通過した後の特徴である。Phase A で判明したのは：

- 層 0–2（temporal_convs BN）: proxy 効果が弱い（improvement ~1e-5、cos_sim ~0.3）
- 層 3–5（ch_DW_conv, ch_reduce_2, temp_conv_2 BN）: proxy 効果が強い（improvement ~1e-3、cos_sim ~0.5）

conv_block の出力は層 5 の後なので、**shallow 後段の特徴を反映している**。しかし conv_block 出力の mean/logvar は層 0–2 を通過した後の累積的な統計であり、層 3–5 の寄与だけを見ているわけではない。

ここは 2 つの選択肢がある：

- **選択肢 A（GPT-5.4 案）:** conv_block 出力を使う → 実装が単純、1 点で shallow 全体を代表
- **選択肢 B（より精密）:** 層 3–5 の BN 入力に直接 hook を付けて統計を取り、各層で独立に正則化

**Phase B-2 では選択肢 A で始めてよい。** 理由は conv_block 出力が shallow 後段の支配を受けているため、実質的に同じ方向に正則化が効く。層ごとの hook は実装が煩雑になり、Phase B の「最小限の変更」方針と矛盾する。

**3. B-1 と B-2 を同時に走らせるか、逐次にするか**

GPT-5.4 の計画は「B-1 → B-2」と逐次的。しかし実務上は以下を考慮すべき：

- B-1 の学習は ~2 時間
- B-2 の実装は B-1 と独立に可能（shared_step への分岐追加のみ）
- B-1 の結果を待ってから B-2 を走らせると 1 日遅れる

**提案:** B-1 と B-2 の実装は同時に行い、B-1 を先に学習開始する。B-1 の学習中に B-2 の config を準備し、B-1 完了後すぐに B-2 を開始する。ただし **B-2 の結果解釈は B-1 の結果を見てから行う**。「B-1 ≈ plain なのに B-2 > plain」であれば shallow_inv が主要因と言えるが、「B-1 > plain かつ B-2 > B-1」の場合のみ両方に価値があると判断する。

---

### 追加すべき観点

**1. 学習の再現性に関する設計**

plain TCFormer の既存 checkpoint と比較するなら、**学習条件を完全に揃える必要がある**。

確認すべき点：
- 既存の source_model checkpoint（`results/update_op_v2_20260401_125734/source_model`）は seed=0、interaug=True で学習されたか
- Phase B の学習も同じ seed、同じ interaug 設定で行うか
- interaug と gain jitter の交互作用がないか（interaug は temporal chunk shuffle、gain jitter は channel-wise — 独立に作用するはずだが確認は必要）

**推奨:** Phase B-1 の最初の run は `interaug=True`（既存と同条件）+ `gain_jitter=True` とする。interaug を切ると、augmentation の効果なのか interaug の欠如の効果なのかが分離できない。

**2. plain の再学習を Phase B の比較対象にすべきか**

GPT-5.4 の計画では plain は「既存 checkpoint、再学習なし」としている。しかし Phase B で学習条件を変えた場合（例：batch 構成が変わる、forward 2 回で gradient accumulation の挙動が変わる等）、plain との比較が不公平になるリスクがある。

**推奨:** Phase B の比較対象として、plain も同じ training script で seed=0 から再学習する。これにより、training infrastructure の変更による差異を排除できる。ただし既存 checkpoint との差異が出た場合はその原因を特定する必要がある。

**代替案:** 実装上、plain 相当を `gain_jitter_std=0.0, lambda_sinv=0.0` として同じ training script で走らせる。これが最もクリーン。

**3. `lambda_aug = 0.5` の根拠**

GPT-5.4 は `lambda_aug = 0.5` を推奨しているが、根拠が明記されていない。

考慮すべき点：
- `lambda_aug = 0.5` は clean:aug = 1:0.5 の重み比。aug loss が clean の半分の影響力
- `lambda_aug = 1.0` にすると clean と aug が等価になり、effective batch size が 2 倍で学習するのと同等
- `lambda_aug` が大きすぎると、aug 側の gradient が学習を支配し、clean での精度が下がるリスク

**推奨:** `lambda_aug = 0.5` で問題ない。根拠は「aug は proxy であり、real distribution ではないため、clean より低い信頼度で混ぜるべき」という原則。ただし Phase B-1 が「aug-only ≈ plain」だった場合、`lambda_aug = 1.0` に上げて再試する選択肢を持っておく。

**4. 評価時の interaug の扱い**

学習時の interaug はデータ数を 2 倍にする。gain jitter を追加すると、1 step あたりの計算が「clean forward + aug forward」で 2 倍になる。interaug + gain jitter の組み合わせで effective training volume が 4 倍近くになる可能性がある。

これは学習時間への影響だけでなく、**overfitting のリスクにも関わる**。BCIC-IV 2a は 230 training samples しかなく、augmentation を増やしすぎると「augmented distribution に過適合して real eval distribution では改善しない」ことがありうる。

**推奨:** Phase B-1 の学習曲線（train_loss, val_loss, val_acc）を注視する。aug-only で val_loss が plain より悪化するなら、overfitting が疑われる。

---

### 判定ロジックの補足

GPT-5.4 の分岐表に 1 つ追加すべきパターンがある：

| 結果パターン | 意味 | 次の一手 |
|---|---|---|
| B-1 で S2 の source-only が改善し、S1/S8 が悪化しない | **最良のシグナル** | B-2 に進み、shinv の追加価値を確認。B-2 でさらに改善するなら Phase C の根拠が最も強い |
| B-1 で全体平均は横ばいだが S2 のみ改善 | proxy が S2 に特異的に効いている | B-2 の shinv が他の被験者に汎化するか確認 |
| B-2 で S2 が改善しない | **critical failure** | causal finding → 正則化の因果連鎖が成り立たなかった。Phase C に進む前に原因を分析する必要がある |

最後のパターンが最も重要。S2 は proxy 品質が最も高く（cos_sim 0.64）、harm の原因が特定済みで、正則化のターゲットが明確な被験者。**ここで改善しないなら、train-time adaptation の仮説自体を見直す必要がある。**

---

### 最終意見

GPT-5.4 の計画はほぼそのまま採用可能。修正は以下の 3 点のみ：

1. **Phase B-1 は `std=0.05` の 1 条件に絞る**（0.025 は退路として温存）
2. **plain も同じ training script で再学習する**（`gain_jitter_std=0.0` として統一）
3. **interaug との交互作用と overfitting を学習曲線で監視する**

実装着手の優先度：
1. `channel_gain_jitter` を `utils/augmentations.py` に配置
2. `shared_step` に aug 分岐を追加
3. `tcformer_aug_only.yaml` を作成
4. Phase B-1 を seed=0 で走らせる（plain 再学習と並行）
5. B-1 完了を待ちつつ B-2 の `_shallow_invariance_loss` を実装
