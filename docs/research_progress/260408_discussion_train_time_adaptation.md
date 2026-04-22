# 2026-04-08 Discussion: Train-Time Adaptation for TCFormer + OTTA

## 目的

Claude と議論するための整理メモ。

今回の主題は次の 1 点:

> 学習時にも adaptation をある程度組み込み、`plain TCFormer` より強い source model を作った上で、推論時 OTTA をさらに足して性能を上積みできる設計にできないか。

---

## まず現状の確認

今のパイプラインでは、OTTA は**学習時には使っていない**。

- 学習時: 通常の supervised training
- 推論時: 学習済み TCFormer checkpoint をロードし、`on_test_start()` で Pmax-SAL OTTA を初期化
- adaptation の実体: optimizer 学習ではなく、主に BN running stats の online update

つまり現状は

1. source session に対して plain TCFormer を学習
2. test 時だけ OTTA を被せて session shift を吸収

という train-test mismatch の強い設計になっている。

---

## これまでの結果から強く言えること

### 1. S2 harm の主因は shallow running_var

既存の BN direction ablation から、S2 の harm は「BN 全体」ではなく、ほぼ浅い層の running_var 更新に局在している。

- `mean_only`: S2 で無害
- `deep`: S2 で無害
- `var_only`: S2 で強い harm
- `shallow`: `both` に近い harm を再現

したがって、浅い層は「適応できること」より「壊れにくいこと」を優先すべきである。

### 2. deep 側の adaptation は価値がある

all-9 fine sweep と dual-momentum exploration から、deep mean / deep var は完全に不要ではない。

- `both@0.01` は multi-seed でも現時点ベスト
- S3 / S7 / S9 では deep adaptation が gain source になっている
- ただし固定 scalar momentum では gain と harm を同時に最適化しきれない

したがって、深い層は「追従できること」を維持すべきである。

### 3. confidence 系 gate だけでは足りない

S5 のように、

- entropy は下がる
- logit norm は上がる
- しかし精度は落ちる

というケースが残っている。

これは現在の OTTA が「自信が上がる adaptation」と「正解に近づく adaptation」を区別できていないことを示す。

---

## 設計原理

以上から、学習時に作り込むべき性質は次の 2 つに整理できる。

### A. Shallow invariant

浅い表現は session shift に対して不変であるべき。

- channel gain
- additive noise
- SNR shift
- spectral tilt

のような変動で shallow feature / shallow BN sensitive になりすぎないように学習する。

### B. Deep adaptive

深い表現は少量の support を見て素直に追従できるべき。

- 深い BN mean / var の少量更新
- support 後の query 改善

を前提にした表現を学習する。

要するに目標は:

> `shallow invariant + deep adaptive`

である。

---

## 提案: Train-Time Simulated OTTA

### コアアイデア

学習時に OTTA の support/query 構造を模擬する。

単なる source CE ではなく、

1. clean batch で通常予測
2. batch の一部を support として使い、test 時と同じ BN adaptation を模擬
3. 更新後の状態で query を推論
4. query の正解率と安定性を loss に入れる

という training を行う。

### 1 step のイメージ

1. batch を `support` と `query` に分割
2. `support` に session-shift augmentation を適用
3. `support` を流して shallow/deep の BN stats を online update
4. 更新後のモデル状態で `query` を推論
5. clean query と adapted query の両方で loss を取る

### 提案 loss

\[
L = L_{ce}^{clean} + \lambda_1 L_{ce}^{adapted} + \lambda_2 L_{consistency} + \lambda_3 L_{shallow\_inv}
\]

それぞれの意味:

- `L_ce^clean`
  - 通常の source supervised CE
- `L_ce^adapted`
  - support adaptation 後の query CE
  - 「adapt 後でも当たる」ことを直接学習させる
- `L_consistency`
  - clean query と adapted query の JS / KL
  - adaptation で予測が不必要に暴れないようにする
- `L_shallow_inv`
  - shallow feature の shift 感度を抑える
  - 今回の causal finding に最も対応する正則化

---

## 具体的にどう学習されるべきか

### 学習目標 1: plain TCFormer より強い source model

OTTA を使わなくても、もともとの TCFormer より shift-robust な表現を作る。

これは主に

- `L_ce^clean`
- mild augmentation
- shallow invariance

で達成する。

### 学習目標 2: OTTA を足すとさらに伸びるモデル

推論時に support を見たら、その更新で query loss が実際に下がるようにする。

これは主に

- `L_ce^adapted`
- support/query simulation
- deep adaptive design

で達成する。

この 2 段目を入れないと、「train 時は static、test 時だけ dynamic」という mismatch が残る。

---

## なぜこの方針が妥当か

### empirical side

今の結果は、test-time heuristics だけで performance frontier を少し動かせることは示した。
一方で dual-momentum の seed 安定性は弱く、test-time 制御だけで最適化し続けるのは限界がある。

### modeling side

今の問題は test-time に初めて adaptation が現れることにある。
ならば、学習時から

- support による shallow/deep stat change
- adaptation 後の query behavior
- harmful overconfidence

を見せておくべきである。

### systems side

この方針なら軽量化もしやすい。

- inner-loop optimizer を持たない
- support で BN buffer だけ更新
- test-time も optimizer state 不要
- deep-only adaptation へ寄せれば、更新対象もさらに減らせる

---

## 実装方針

### Phase 1: 最小実装

まずは一番 conservative に始める。

#### backbone

- 既存の `TCFormer` / `TCFormerOTTA` を使う

#### train-time adaptation rule

- まずは `shared_hybrid@0.01` 相当を学習時模擬に使う
- いきなり dual-momentum を学習に入れない

理由:

- いま最も安定している実験条件だから
- train-time adaptation と test-time adaptation を同時に増やすと原因分離できなくなるから

#### augmentation

- channel-wise gain jitter
- small additive Gaussian noise
- optional spectral tilt

目的:

- session shift の proxy を source 内で作る

#### loss

- `CE(clean)`
- `CE(adapted query)`
- `JS(clean query, adapted query)`

この 3 つから始める。

### Phase 2: shallow invariance を追加

次に shallow feature map の shift 感度を直接抑える。

候補:

- clean / aug で shallow feature の L2 distance を抑える
- shallow feature variance の差を抑える
- shallow BN 入力の統計差を抑える

### Phase 3: benefit-aware gate

必要なら gate head を追加する。

入力候補:

- pmax
- entropy
- energy
- neuro_score
- shallow/deep BN drift proxy

教師候補:

- adapt 後 query CE が adapt 前より下がるか

目的:

- 「adapt すると confidence は上がるが間違う」ケースを学習時に抑える

---

## 実装の最小単位

最初の実装は新しいモデルを増やしすぎない方がよい。

### 変更先候補

- `intentflow/offline/models/tcformer/classification_module.py`
  - support/query split と train-time simulated adaptation を入れる
- `intentflow/offline/models/tcformer_otta.py`
  - 既存 OTTA の shallow/deep BN update を train-time にも使い回せるよう整理
- 新規 config
  - train-time simulated OTTA 用 config を追加

### 実験の比較軸

最低限、次の 4 条件を比べるべき。

1. plain `TCFormer`
2. `TCFormer + train-time augmentation only`
3. `TCFormer + train-time simulated OTTA`
4. `3 + test-time OTTA`

ここで見たいのは:

- `3` が `1` より強いか
- `4` が `3` よりさらに強いか

この 2 段改善が出れば、今回の方向性はかなり強い。

---

## 期待できるメリット

### 精度向上

- source-only 精度の底上げ
- OTTA ありの gain の安定化
- dual-momentum のような seed-sensitive gain を学習側で吸収

### 安全性

- shallow variance sensitivity を train-time に抑える
- S5 型の harmful overconfidence を減らす

### 軽量化

- optimizer inner-loop 不要
- BN stats update のみ
- shallow を固定し deep のみ更新すれば test-time コストも小さい

---

## 現時点の主張

今回の結果から自然に導かれるのは、

> test-time OTTA を後付けするだけでは限界があり、  
> `shallow invariant + deep adaptive` を train-time から作る必要がある

という方向性である。

より具体的には、

> 学習時に support/query 構造で simulated OTTA を行い、  
> adaptation 後の query 性能を直接最適化する

のが次の本命になる。

---

## Claude と議論したい論点

1. `support/query simulated OTTA` の最小実装をどこに入れるべきか
2. `L_shallow_inv` は shallow feature 距離と shallow variance penalty のどちらが良いか
3. augmentation は gain/noise/spectral tilt のどこまで必要か
4. 最初の train-time adaptation rule は `shared_hybrid@0.01` で十分か
5. benefit-aware gate を train 初期から入れるべきか、後段で良いか
6. 「精度向上」と「軽量化」の両立を最も素直に示せる ablation は何か

---

## 暫定結論

- shallow は invariant に学習させるべき
- deep は adaptive に学習させるべき
- train-time から OTTA を模擬しないと、test-time heuristic の不安定さが残る
- 最初の実装は
  - support/query split
  - BN stats adaptation simulation
  - adapted query CE + consistency
  - shallow invariance regularization
  の順で入れるのが妥当

---

## 追記（2026-04-08 夜）：critique を受けた計画修正

外部 critique により、元の案にはまだ 2 つの論理的な弱点があることが明確になった。

### 修正ポイント 1

元の Phase C は、そのままでは

> `adaptation を学ぶ`

よりも

> `adaptation に耐える`

を学ぶ可能性が高い。

理由は、元案の support/query 構成が

- support = augmented
- query = clean

に近く、support 統計で query が壊れないように特徴を頑健化する勾配が主になるからである。

これは有用ではあるが、意味は

- source-only robustness の学習
- test-time OTTA gain の直接学習

とは異なる。

### 修正ポイント 2

元の Phase A は marginal proximity を主に見ていたが、それだけでは不十分。

重要なのは

- augmentation が eval session に marginal で近いか

だけでなく、

- augmentation によって誘導される BN drift direction が eval 時の drift direction に近いか

である。

augmentation が「見た目は近いが drift 方向は逆」なら、Phase C に進む前提が崩れる。

---

## 改訂後の 4 Phase

### Phase A: augmentation proxy の妥当性確認

#### A-1: marginal proximity

各 BN layer の shallow/deep 入力統計について、

- clean training data
- augmented training data
- eval session data

の mean / logvar を保存し、aug が eval に近づくかを見る。

指標例:

- channel-wise Gaussian 近似 KL
- mean distance
- logvar distance

これは必要条件。

#### A-2: drift direction consistency

さらに、各 BN layer で drift vector の方向一致を確認する。

定義例:

- `drift_aug = stats(aug_T) - stats(clean_T)`
- `drift_eval = stats(E) - stats(source_model_running_stats)`

ここで `stats` は layer-wise mean と logvar の連結ベクトル。

見る指標:

- cos similarity `cos(drift_aug, drift_eval)`
- shallow / deep 別の layer-wise 分布

**go/no-go**

- shallow 層で `A-1` が改善し、かつ `A-2` が正の方向一致を示すなら次へ進む
- shallow 層で `A-2 < 0` が多いなら augmentation 設計を見直す

これは実装変更なしで先に確認できる。

---

### Phase B: augmentation-only robustness

ここでは adaptation は入れず、

- session-shift augmentation
- `CE(clean)`
- `CE(aug)`
- consistency
- shallow invariance regularization

だけで plain TCFormer を超えるか確認する。

この Phase の意味は

- train-time shift robustness だけでどこまで行けるか
- OTTA なしの土台性能をどこまで上げられるか

の確認である。

---

### Phase C: train-time virtual BN

ここは **2 つに分けて考える必要がある。**

#### Phase C-R: robustness-oriented virtual BN

support と query を

- support = augmented pseudo-domain
- query = clean

のように置いた virtual BN。

これは主に

- adaptation perturbation に対する耐性
- shallow sensitivity の緩和

を学ぶ。

したがって success criterion は

- 条件 3 が augmentation-only より source-only 精度で伸びる
- 条件 3 + OTTA が augmentation-only + OTTA より悪化しない

でよく、`4 > 3` を必須条件にしない。

#### Phase C-A: true adaptive virtual BN

真に「adaptation すると query が良くなる」ことを学びたいなら、

- support と query が**同じ shifted pseudo-domain**を共有し、
- clean branch とは異なる

必要がある。

つまり 1 episode の中で、

- clean anchor batch `x`
- shifted support `S_phi`
- shifted query `Q_phi`

を作る。ここで `S_phi` と `Q_phi` は**同じ augmentation parameter `phi`** を共有する。

このとき adaptation は

- `S_phi` で BN virtual stats を計算
- その stats で `Q_phi` を正規化

として行う。

これなら勾配は

> この pseudo-domain shift では support を見ると query loss が下がる

ことを直接学べる。

---

## Phase C-A の推奨 loss

元の

\[
L = L_{ce}^{clean} + \lambda_1 L_{ce}^{adapted} + \lambda_2 L_{consistency} + \lambda_3 L_{shallow\_inv}
\]

に加えて、**adaptation benefit を直接比較する loss** を入れる方が良い。

### 定義

- `L_noadapt = CE(f(Q_phi), y_q)`
- `L_adapt = CE(f(Q_phi | S_phi), y_q)`

追加候補:

\[
L_{gain} = \max(0, L_{adapt} - L_{noadapt} + m)
\]

ここで `m >= 0` は margin。

最終候補:

\[
L = L_{ce}^{clean}
  + \lambda_1 L_{adapt}
  + \lambda_2 L_{gain}
  + \lambda_3 L_{consistency}
  + \lambda_4 L_{shallow\_inv}
\]

これにより、Phase C-A は単なる robustness ではなく、

> adaptation 後の方が同一 pseudo-domain query に対して良い

ことを直接最適化できる。

---

## α の設計

critique の通り、Phase C では α を学習可能にしない方が良い。

理由:

- α を学習可能にすると gate と切り分けにくい
- Phase D の learned gate / benefit predictor と役割が被る
- まずは causal finding に沿った固定設計で identifiability を保つべき

### 推奨

Phase C 開始時点では fixed α。

#### conservative variant

- `alpha_shallow_mean = 0`
- `alpha_shallow_logvar = 0`
- `alpha_deep_mean = alpha_deep_logvar = {0.1, 0.3, 0.5}` sweep

これは最も識別しやすい。

#### hybrid-consistent variant

既存の test-time best が `shallow_mean + deep_both` である点を踏まえると、

- `alpha_shallow_mean = small constant`
- `alpha_shallow_logvar = 0`
- `alpha_deep_mean = alpha_deep_logvar = sweep`

という variant も検討余地がある。

ただし最初の実装は conservative variant の方が解釈しやすい。

---

## 修正後の成功基準

### Phase B 成功

- plain TCFormer より source-only 精度が上がる
- OTTA を足しても悪化しない

### Phase C-R 成功

- augmentation-only より source-only 精度が上がる
- OTTA 併用時の NTR-S / WSD が悪化しない

### Phase C-A 成功

- shifted pseudo-domain 上で `adapt > no-adapt`
- source-only でも plain を上回る
- test-time OTTA 追加でさらに gain が出る

ここで初めて

> train-time adaptation + test-time adaptation の二段構え

が強く主張できる。

---

## 現在の最小アクション

実装に入る前に最も安全なのは、まず **Phase A-1 + A-2** をやること。

やることは次の通り。

1. source model の BN running stats を保存
2. clean training data を流したときの BN 入力 mean / logvar を保存
3. augmented training data を流したときの BN 入力 mean / logvar を保存
4. eval session data を流したときの BN 入力 mean / logvar を保存
5. A-1: marginal 距離を比較
6. A-2: layer-wise drift cosine を比較

これにより、

- augmentation が session shift の proxy になっているか
- どの層で proxy として使えるか

が決まり、Phase B/C に進む根拠ができる。

---

## 修正後の暫定結論

- 元案は方向として正しかった
- ただし Phase C の意味は robustness と adaptivity を分けて定義し直す必要がある
- Phase A は marginal だけでなく drift direction を確認すべき
- α は固定値で始めるべき
- 最初の実装前に Phase A-1 + A-2 を先にやるのが最も安全
