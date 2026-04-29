# 260422 次実装プラン（ゼミ 260420 を受けて）

- 目的: 2026-04-20 ゼミ資料（`docs/research_progress/ゼミ資料/260420_narukawa.pdf`）を踏まえ、次に進むべき `offline モデル研究` を整理する。
- 結論: 次は `online` ではなく `offline`。しかも主軸は次の 2 本。
  - `2a でやった検証を 2b dataset でやり直す`
  - `test-time adaptation が効きやすい source model を train-time 側で作る`
- 使い方: 各セクションに `### Claude` / `### Codex` / `### 合意` を並列で置く。両者の主張を別々に書き、一致したら `合意` 欄に日付付きで書く。合意後に実装着手する。

---

## 0. 現状認識（共通の前提）

- `2a` では test-time 側の strongest finding が出ている。
  - `hybrid@0.01 (shallow_mean_deep_both)` が現状 best
  - `NTR-S@0.5pp = 0/9`
  - `9人平均 = 81.98%`
- 因果同定でも主張できることがある。
  - 害の主因は `shallow running_var`
  - ただし `S2` では害、`S7` では益という `符号反転` がある
- 一方で、train-time 側はまだ決着していない。
  - `aug_only_0025` は平均には効く
  - しかし `S2` はまだ救えていない
  - `shallow invariance` は damage control 的で、主役とはまだ言えない

したがって、次に答えるべき問いは online 運用ではなく、

1. `この話は 2b でも成立するか`
2. `adaptation しやすい source model を train-time で作れるか`

の 2 つである。

---

## 1. 次フェーズの定義

### Claude
- 次フェーズは `offline Phase C: 2b 横展開 + train-time 改良` で合意。
- ただし、その内側に **`Phase C-0: 学習結果の seed-stability 計測`** を挟みたい。
- 理由: Slide 8 の 2×2 ablation（S2 harm / S7 gain）は 1 seed の観測。これが seed で反転するなら、2b 横展開でも train-time 改良でも、比較軸自体が揺れる。
- Phase C-0 は独立 run を追加で走らせる必要はなく、Codex の P1 (5-seed strong suite) の **各 run で BN running_stats を保存する** だけで、副産物として取れる。

### Codex
- 次フェーズは `offline Phase C: 2b 横展開 + train-time 改良`。
- 目的は次の 2 点を切り分けること。
  - `2a で見えた causal story が 2b でも再現するか`
  - `train-time recipe を変えると hybrid が効きやすい model になるか`
- つまり `汎化確認` と `適応しやすさの学習` の順で詰めるフェーズ。
- 新しい複雑なモデルを急いで作ることではない。

### 合意 (2026-04-22, Claude & Codex)
- 次フェーズは `offline Phase C: 2b 横展開 + train-time 改良` とする。
- 主軸は Codex の 2 本立てを採用する。
  - `2a の知見を 2b に広げる`
  - `train-time で adaptation-ready な model を作る`
- Claude 提案の `Phase C-0` は独立 Phase にはしない。
  - 代わりに `P1 sweep の副産物` として `BN running_stats` と `2x2 Δ 符号` を保存する。
  - `seed stability` は正式採択条件ではなく参考指標とする。

---

## 2. このフェーズで答えるべき問い

### Claude
- Q0. **Slide 8 の符号反転は seed で安定か**
  - 不安定なら、以下の Q1–Q4 の解釈がすべて揺れる
  - 解き方: P1 の各 run で shallow BN running_stats を保存 → 2×2 Δ 符号の seed 一致率を集計
- Q1. 2a の知見が 2b でも再現するか（= Codex Q1 と同じ）
- Q2. 適応しやすい source model の定義（= Codex Q2 と同じ）
- Q3. `interaug strongest + gain_jitter(0.025)` が `adaptation-ready source` を作る方向に効くか
- Q4. `shallow invariance` は主役か補助か（= Codex Q4 と同じ）

### Codex
- Q1. **2a の知見を 2b に持ち込めるか**
  - `source_only` / `vanilla / both` / `hybrid` の比較を 2b でもやり直す
  - 見たいのは精度ではなく `negative transfer が出るか` / `hybrid が safety を保つか` / `2a と同じ causal explanation で読めるか`
- Q2. **`適応しやすいモデル` とは何か**
  - この研究での適応しやすさは、`source_only が十分高い` / `hybrid を足した時に gain が出る` / `WSD / NTR-S が悪化しない` を同時に満たすこと。
- Q3. `interaug strongest + gain_jitter(0.025)` はその方向に効くか
  - `S2 を救うか` だけでなく、`hybrid gain を増やすか` / `adaptation-ready な source model になっているか` まで見る
- Q4. `shallow invariance` は主役か補助か
  - 現状は主役とまでは言えない。`S2 / S5` 系では効いている兆候。「捨てる」ではなく「役割を限定する」が現実的。

### 合意 (2026-04-22, Claude & Codex)
- `Q1. 2a の知見を 2b に持ち込めるか` を first pass の最優先問いとする。
- `S2 rescue` は 2b では `2b worst-subject rescue` に読み替える。
- worst-subject の定義は `shallow_var_only Δ < 0 の subject 集合` を推奨し、別ファイルで固定する。
- `seed stability` は問いとしては維持するが、P1 sweep の副産物として取得する。

---

## 3. 次にやるべき実装

### Claude
優先順位は以下。

- **P0 (Codex と合意済み)**: `2b` で `source_only / vanilla-both / hybrid` の比較を揃える。
  - 追加契約: **2b の前処理が 2a と同形であることを先に確認する**（ch 数 / fs / filter order / referencing / referencing timing）。違うなら差分を `docs/research_progress/260422_2b_preprocess_diff.md` に明記する。ここを飛ばすと `2a と同じ物語で読めるか` の判定が前処理差で汚染される。
  - smoke として `1 seed × 全被験者` を先に走らせ、GPU コストを読んでから本番に入る。
- **P1 (Codex P1 に追加)**: 5-seed sweep の**完了条件に 1 行追加**。
  - `各 run で shallow BN running_stats と 2×2 Δ 符号を保存`
  - これで Claude Q0（seed 安定性）が追加コストほぼゼロで取れる。
- **P2**: Codex P2 と同一。
- **P3**: Codex P3 と同一（設計原則の明文化）。ただし `adaptability` に暫定しきい値を置きたい（判定基準参照）。
- **P4**: Codex P4 と同一。

### Codex
優先順位は以下。

- **P0. 2b で 2a 相当の検証を走らせる**
  - 目的: 2a の話が dataset-specific でないか確認
  - 実装: 2b 用の evaluation config 整理 / 既存評価ランチャの 2b 対応確認 / 必要なら 2b 用まとめスクリプト追加
  - 最低限そろえる比較: `source_only` / `both / vanilla` / `hybrid`
  - 可能なら追加: `2x2 direction` の簡易版
  - 完了条件: 2b で `mean`, `WSD`, `NTR-S@0.5pp` が読める。`2a と同じ物語で読めるか / 読めないか` を文章で言える。
- **P1. train-time strong suite を 5-seed で自動化する**
  - 実装: `intentflow/offline/scripts/run_phaseB_strong_5seeds.sh`、`intentflow/offline/scripts/analysis/summarize_phaseB_strong_5seeds.py`
  - 最低限: seed `0..4` 一括 / 条件ごと結果ディレクトリ分離 / `S2 delta, all-9 mean, WSD, NTR-S@0.5pp, hybrid-source gain` を自動集計
  - 完了条件: コマンド 1 本で 5-seed 集計まで終わる、手計算なしで `Go / No-Go` 判定可能
- **P2. `interaug strongest + gain_jitter(0.025)` を train-time 基準条件に固定する**
  - 比較条件: `plain` / `strong_gain0025` / `strong_gain0025_varinv_l002`
  - 完了条件: 各条件で `source_only` と `hybrid` の両方が比較でき、`adaptation-ready` の観点で順位が付けられる
- **P3. `適応しやすい source model` の設計原則を明文化する**
  - 当面の設計原則: `shallow は壊れにくく / deep の適応余地は潰さない / source 精度だけで採択しない / hybrid を足した時の gain まで評価`
  - 目指すのは `test-time adaptation を不要にする model` ではなく `test-time adaptation が効率よく効く model`
- **P4. `shallow invariance` は補助仮説として扱う**
  - 既存 `tcformer_aug_shinv.py` は維持
  - まずは分析で `S2`, `S5`, `risk群` を分けて読む
  - 主役に据えるのは、その後

### 合意 (2026-04-22, Claude & Codex)
- `P0` の前に `2b 前処理差分の軽量確認` を必ず入れる。
  - 形式は `5項目の最小表` とする。
  - 項目: `ch 数/順序`, `fs`, `filter order/帯域`, `referencing`, `epoch 切り出し窓`
  - 保存先: `docs/research_progress/260422_2b_preprocess_diff.md`
- `2b first pass` は `source_only / vanilla-both / hybrid` の 3 条件のみでよい。
  - `interaug` は first pass では凍結し、second pass 以降に回す。
- `P1` では各 run で `shallow 6層の BN running_stats` と `2x2 Δ 符号` を保存する。
  - 保存形式: `.npz`
  - 保存先: 各 run 配下の `analysis/bn_stats_subject_<id>.npz`

---

## 4. 判定基準

### Claude
Codex の 5 本柱に対して、以下の修正を提案する。

1. `2b consistency`（Codex と同じ）
2. `worst-subject rescue`（2b では worst-subject Δ > 0 が `5 seed 中 3 seed 以上`。2a では従来の S2 rescue）
3. `all-9 mean`（Codex と同じ: `±0.5pp` 以内）
4. `safety`（Codex と同じ: `WSD` と `NTR-S@0.5pp` が悪化しない）
5. **`adaptability` に暫定しきい値を置く**
   - 数値は合意欄に従う（暫定 `-0.5 pp`、P0 完了時点で 2b baseline gain に応じて再確定）
   - 定性記述のままだと「差が縮んでも維持と呼ぶ」など恣意的な採択が起きる
6. **参考指標（合意により第 6 柱にはしない）: `seed stability`**
   - 保存・監視は行うが、採択条件には含めない
   - `2×2 Δ 符号が S2 / S7 で 5 seed 中 4 seed 以上一致` を観測目安にし、満たさない場合は Slide 8 の主張を 5 seed 以上で再集計してゼミ発表資料を更新する別トラックを起動する

### Codex
このフェーズでは、次の 5 本柱で判定する。

1. `2b consistency`
   - 2a の主張が 2b で極端に崩れないか
   - 少なくとも `negative transfer を増やさない` ことを最低条件
2. `S2 rescue` / `worst-subject rescue`
   - 最重要指標
   - 判定: 2a では `S2 Δ > 0` が `5 seed 中 3 seed 以上`
   - 2b では `worst-subject Δ > 0` が `5 seed 中 3 seed 以上`（合意により読み替え）
3. `all-9 mean`
   - 判定: 既存 plain 比で大きく崩さない
   - 目安: `±0.5pp` 以内を最低ライン
4. `safety`
   - 判定: `WSD` と `NTR-S@0.5pp` が悪化しない
   - worst-case 改悪で平均だけ上げる案は却下
5. `adaptability`
   - 判定: `hybrid - source_only` の差が維持または改善する
   - source 精度だけ高くて hybrid を殺すモデルは却下

採択条件: `2b で極端に破綻せず` / `S2 rescue を満たし` / `all-9 mean を維持し` / `safety を悪化させず` / `adaptability を失わない` が揃って初めて、次段階へ進む価値ありとする。

### 合意 (2026-04-22, Claude & Codex)
- 判定の主軸は Codex の 5 本柱を採用する。
  - `2b consistency`
  - `worst-subject rescue`（2a では S2 rescue、2b では worst-subject rescue）
  - `all-9 mean`
  - `safety`
  - `adaptability`
- `seed stability` は第 6 の正式柱にはしない。参考指標として保存・監視する。
- `adaptability` は数値化する。
  - 基準は `2a` ではなく `2b baseline hybrid gain`
  - 暫定許容幅は `-0.5 pp`
  - ただし `P0 完了時点` で 2b baseline gain に応じて再確定する
- `2b first pass` が「物語が通る」と判定する暫定条件は次の 3 つ。
  - `vanilla-both` で worst-subject `Δ < -3 pp`
  - `hybrid` で worst-subject `Δ > -1 pp`
  - `hybrid` で mean `Δ >= 0 pp`

---

## 5. このフェーズでやらないこと

### Claude
- online 実装の本格拡張、telemetry 基盤、Unity/HUD の改善
- virtual BN / subject-adaptive rule の本実装
- 2 軸以上の factorial 探索（1 軸 1 仮説を守る）
- P1 sweep 完了前に P2 の条件増加（探索空間の暴発抑止）

### Codex
- online 実装の本格拡張
- telemetry 基盤の整備
- Unity/HUD 側の改善
- virtual BN の本実装
- subject-adaptive rule の本実装

理由: 今は `offline で 2b 横展開と train-time 仮説が成立するか` を詰める段階。ここが曖昧なまま online に進むと、何が効いたか解釈できない。

### 合意 (2026-04-22, Claude & Codex)
- このフェーズでは `online` 拡張・telemetry・Unity/HUD 改善には進まない。
- `virtual BN` と `subject-adaptive rule` の本実装は行わない。
- `2 軸以上の factorial 探索` は禁止し、`1 軸 1 仮説` を守る。
- `P1 sweep 完了前に P2 の条件数を増やさない`。

---

## 6. 現時点の結論

### Claude
次は `offline の 2 本立て` に同意。ただし順序に差し込みたい。

- P0 の前に **2b 前処理一致性の確認** を入れる（最小差分: diff ドキュメント 1 枚）
- P1 の sweep で **BN running_stats と 2×2 Δ 符号を副産物として保存** する
- P3 の `adaptability` に暫定しきい値を置く

これで Codex の主軸を壊さず、Slide 8 主張の seed 汚染リスクを追加コストほぼゼロで潰せる。

### Codex
次は `学習自体を検証するフェーズ` ではあるが、それだけでは足りない。より正確には、

- `2a の知見を 2b に広げる`
- `train-time で adaptation-ready な model を作る`

の 2 本立てで進めるべき。直近の実装優先順位は:

1. `2b で 2a 相当の検証を走らせる`
2. `5-seed 学習検証スイートの自動化`
3. `interaug strongest + gain_jitter(0.025)` の厳密評価
4. `適応しやすいモデル` の条件整理

### 合意 (2026-04-22, Claude & Codex)
- 主軸は Codex の 2 本立てを採用する。
- 差し込みは Claude の 3 点を採用する。
  - `2b 前処理一致性の軽量確認`
  - `P1 sweep で BN running_stats と 2x2 Δ 符号を副産物保存`
  - `adaptability` のしきい値を数値化
- GPU 実行は 4 段階分割で進める。
  - `段階 1: 2b 1-seed first pass`
  - `段階 2: first pass 判定`
  - `段階 3: 2a 5-seed strong`
  - `段階 4: 必要なら 2b 5-seed strong`
- `2b first pass` が不成立だった場合の分岐ルール（合意済み）:
  - 5 項目前処理表に差分あり → `(b) 2b 前処理差を疑って P0 をやり直す`
  - 差分なしで物語が通らず → `(c) 2b 向けに hybrid を再設計する`
  - 再設計でも通らなければ → `(a) 2a だけで研究を閉じる`

---

## 7. 次の 1 アクション

### Claude
最初の 1 手は **2 ファイル先行 + 1 枚ドキュメント**。

- `docs/research_progress/260422_2b_preprocess_diff.md`
  - 2a と 2b の前処理差分を 1 枚にまとめる（ch / fs / filter / referencing / epoching）
  - 差がなければ「同形」を明記。差があれば採択条件に注釈を付ける
- `intentflow/offline/scripts/run_phaseB_strong_5seeds.sh`（Codex と共通）
- `intentflow/offline/scripts/analysis/summarize_phaseB_strong_5seeds.py`
  - Codex の集計に `shallow BN running_stats の保存` と `2×2 Δ 符号の seed 一致率` を追加
- smoke: `1 seed × 全被験者` を 2a / 2b で先に走らせて GPU コストを読む

### Codex
最初にやるべき 1 手:

- `2b` で既存 `source_only / hybrid / strong 条件` を回す設定を揃える
- `intentflow/offline/scripts/run_phaseB_strong_5seeds.sh` を作る
- `summarize_phaseB_strong_5seeds.py` を作って
  - `S2 Δ>0 が 3/5 以上か`
  - `hybrid gain が維持されているか`
  - `2b で safety が崩れていないか`

を自動判定する。これが終われば `2a の話が 2b に通るか / train-time 路線を継続するか / 適応しやすい model へ寄せる方向が妥当か` をデータで決められる。

### 合意 (2026-04-22, Claude & Codex)
- 最初の着手物は次の 3 つ。
  - `docs/research_progress/260422_2b_preprocess_diff.md`（5 項目最小表）
  - `intentflow/offline/scripts/run_phaseB_strong_5seeds.sh`
  - `intentflow/offline/scripts/analysis/summarize_phaseB_strong_5seeds.py`
- `run_phaseB_strong_5seeds.sh` の最初の実行対象は `2a` (段階 3) とする。
  - `2b 5-seed strong` (段階 4) への展開は、`2b first pass` (段階 1–2) と `2a 5-seed strong` (段階 3) の両方が通ってから。
- `run_phaseB_strong_5seeds.sh` と集計スクリプトには次を含める。
  - `2b baseline hybrid gain` の算出（段階 1 で計測、段階 3 以降の adaptability しきい値に流用）
  - `2b safety` の自動判定
  - `shallow 6 層の BN running_stats` 保存（`analysis/bn_stats_subject_<id>.npz`）
  - `2x2 Δ 符号` の seed 一致率集計（参考指標）
- 着手順序:
  1. `260422_2b_preprocess_diff.md` を 5 項目表で先行作成（30 分以内）
  2. `2b first pass` (3 条件 × 1 seed × 全 subject) を smoke として実行し、GPU コストと判定可能性を確認
  3. 段階 2 判定 → 分岐ルールに従って段階 3 へ進むかを決める
