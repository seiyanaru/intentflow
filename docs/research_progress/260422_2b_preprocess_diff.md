# 260422 BCIC-IV 2a / 2b 前処理差分（5 項目最小表）

- 目的: 2b first pass (3 条件 × 1 seed × 全 subject) を走らせる前に、2a と 2b の前処理が同形であることを確認する。
- 合意済みの契約: 差分があれば「2a と同じ物語で読めるか」の判定に注釈を付ける（`260422_next_implementation_plan.md` 合意欄）。
- 参照実装:
  - [intentflow/offline/utils/load_bcic4.py](intentflow/offline/utils/load_bcic4.py)
  - [intentflow/offline/datamodules/bcic4_2a.py](intentflow/offline/datamodules/bcic4_2a.py)
  - [intentflow/offline/datamodules/bcic4_2b.py](intentflow/offline/datamodules/bcic4_2b.py)
  - [intentflow/offline/configs/tcformer_otta/tcformer_otta.yaml](intentflow/offline/configs/tcformer_otta/tcformer_otta.yaml)
- 共通条件: `z_scale=True`（train 統計で fit、val/test は transform のみ — `base.py:121-132`）、`scale(factor=1e6)` で µV→V スケール補正は共通。

---

## 5 項目差分表

| 項目 | 2a (BCIC-IV 2a) | 2b (BCIC-IV 2b) | 同形判定 |
|---|---|---|---|
| **1. ch 数 / 順序** | 22 ch EEG (`raw.pick(range(22))`) | 3 ch EEG (C3/Cz/C4, `raw.pick(range(3))`) | **異** |
| **2. fs (サンプリングレート)** | 250 Hz (`resample sfreq=250`) | 250 Hz (`resample sfreq=250`) | **同** |
| **3. filter order / 帯域** | `low_cut=null, high_cut=null` → **bandpass なし**（resample のみ。GDF の元フィルタに依存） | `low_cut=null, high_cut=null` → **bandpass なし**（同上） | **同** |
| **4. referencing** | raw GDF のまま（common average 等の再 referencing **なし**） | raw GDF のまま（同上） | **同** |
| **5. epoch 切り出し窓** | `start=0.0, stop=4.0` → **4.0 s = 1000 サンプル** (cue onset 基準、`bcic4_2a.py:62` で `1000` に強制 crop/pad) | `start=0.0, stop=3.0` → **3.0 s = 750 サンプル** (`bcic4_2b.py:49` で `expected_length=750`) | **異** |

---

## 総合判定

**非同形**。項目 1 (ch 数) と項目 5 (窓長 / サンプル数) で 2a と 2b は構造的に異なる。項目 2–4 は同形。

### この差分が引き起こす影響

1. **モデル入力形状の不一致**:
   - 2a: `(B, 22, 1000)` — source model (TCFormer) の入力契約はこの形
   - 2b: `(B, 3, 750)` — 2b で再学習するか、2a 事前学習モデルを使う場合は**強制 zero-pad / truncate** が必要（`run_unicorn_live.sh` でも指摘あり）
   - → **2b では別途 source model を学習する必要がある**。2a の hybrid@0.01 重みをそのまま 2b に適用してはいけない。

2. **クラス数も違う**:
   - 2a: 4 クラス (feet / L-hand / R-hand / tongue)
   - 2b: 2 クラス (L-hand / R-hand)
   - → softmax の自由度が違うので、`pmax` や entropy 閾値はそのまま流用不可。**2b では τ_pmax を再キャリブレーション**する必要がある。

3. **セッション契約も違う**:
   - 2a: `session_T` (train) / `session_E` (test) の 2 セッション
   - 2b: `session_0–2` (train) / `session_3–4` (test) の 5 セッション
   - → 段階 1 の 2b first pass では、`datamodule の現実装どおり` (train=0–2, test=3–4) を使うのが無難。

---

## first pass 着手前の判断

- **中止ではなく、注釈付きで継続**。2b は「2a の話がそのまま通るか」を見る対象ではなく、「**2a の causal story (shallow×var が害の主因 / S2 型符号反転)** が、別 dataset / 別 ch 数 / 別クラス数でも再現するか」を見る対象。
- したがって判定条件の読み替えは以下:
  - `2b consistency` の評価は「2b で vanilla-both が worst-subject を壊す」「hybrid で救える」が両方成り立つかのみ見る。精度絶対値は比較しない。
  - `worst-subject rescue` は 2b で新たに定義する (vanilla-both Δ が最小 or `shallow_var_only Δ < 0` の subject 集合)。
  - `adaptability` は **2b で新たに測る baseline hybrid gain** を基準にする（合意済み）。

## 次アクションへの注釈

1. 2b 用 source model を学習する必要がある。2a hybrid@0.01 の重みは使えない。
2. ch 数と窓長の違いは TCFormer の `F1`, `pool_length_*`, `temp_kernel_lengths` 設定に影響する可能性。既存 `tcformer_otta.yaml` は 2a 前提のハイパラなので、2b で動くかは **smoke 実行でまず確認**する。
3. `pmax_threshold=0.7`, `sal_threshold=0.5`, `energy_quantile=0.95` は 2a でキャリブレーション済。2b では source model の confidence 分布が違うので、first pass ではこれらを流用しつつ**棄権率と pmax 分布を観測する**。
4. 2b の worst-subject は first pass 結果を見てから確定する。事前に番号は固定しない。

---

## 追記 (2026-04-22 13:50 JST) — 論文プロトコル乖離の発見と修正

### 発端
2b source model 学習を最初に走らせた時点で、ログに 2 つの赤信号:

- `Warning: No evaluation sessions (3, 4) found. Will split training data.`
- `WARNING: val_dataset not found. Returning test_dataloader (DATA LEAKAGE!)`

学習は停止して原因を調査した。

### 判明した現実装の違反 3 点（vs TCFormer 論文 + 公式 repo `github.com/altaheri/TCFormer`）

| 項目 | 本 repo 旧実装 | TCFormer 公式 / 論文 | 結果 |
|---|---|---|---|
| **データ source** | 自前 GDF + label injection。ただし **2a にだけ** `_load_bcic2a_eval_labels` が実装されており、**2b 用は未実装**。E session の raw GDF は cue=`783` のみで 769/770 無し、target_events チェックで skip | `MOABBDataset("BNCI2014004")` 経由で label を自動解決 | session_3-4 が読み込めず fallback で train を train_test_split → **正規 test 320 trial を使わず、train 分割 80 trial を擬似 test にしていた** |
| **窓長 / stop offset** | `tcformer.yaml` の `bcic2b: start=0.0, stop=3.0`（3 s / 750 サンプル） | `start=0.0, stop=-0.5`（MOABB trial_duration 4.5 s から末尾 0.5 s を切り詰め → **4 s / 1000 サンプル**、sessions 1-2 / 3-5 の 4 s vs 4.5 s 差分は stop=-0.5 で吸収） | **MI 時刻帯の末尾 1 s を捨てていた**。サンプル数も論文と不一致 |
| **val/test 分離** | `BCICIV2b.setup()` が `val_dataset` を作らず、base.py が `test_dataloader()` を返すフォールバック → `WARNING: DATA LEAKAGE` | `val == test`（明示設計、last-epoch 評価、early stopping や best-on-val checkpoint 選択を使わない） | 意味的には同じだが契約が曖昧。本 repo の `train_pipeline.py:95-104` は `enable_checkpointing=False` + 最終 epoch で手動保存のため、**val=test でも "best-on-val" リークは構造的に起きない**（val は曲線プロットの監視用のみ） |

### 採った修正（A3: TCFormer 公式準拠）
ユーザー判断 A3 = 公式 repo プロトコルに合わせる。最小差分 4 箇所:

1. **[tcformer.yaml:27-33](intentflow/offline/configs/tcformer/tcformer.yaml#L27-L33)**: `bcic2b.stop: 3.0 → -0.5`
2. **[train_pipeline.py:300-305](intentflow/offline/train_pipeline.py#L300-L305)**: 2b のとき `data_path=None` を強制 → `load_bcic4` の MOABB 経路に入る
3. **[train_pipeline.py:62-65](intentflow/offline/train_pipeline.py#L62-L65)**: `data_path is None` フォールバックの 2a 固定上書きに `dataset_name != "bcic2b"` ガード追加（2b の None を 2a の GDF path で上書きしていたバグ）
4. **[bcic4_2b.py:25-76](intentflow/offline/datamodules/bcic4_2b.py#L25-L76)**: `BCICIV2b.setup()` を書き直し
   - MOABB の session キー `0train/1train/2train/3test/4test` を認識（ローカル GDF 経路 `session_0..4` との両対応）
   - session_0-2 → train、session_3-4 → test
   - `val_dataset = test_dataset.copy()` を明示（TCFormer 公式の val=test 設計に合わせ、`DATA LEAKAGE` 警告を構造的に解消）
   - `expected_length=1000`（4 s @ 250 Hz）に統一、旧 750 へのクロップ/パッド削除
   - `train_test_split` フォールバック（session 3-4 不在時）を削除

### スモークテスト結果 (subject 1)
```
Available sessions for subject 1: ['0train', '1train', '2train', '3test', '4test']
Loaded 0train run 0: X shape (120, 3, 1000)
Loaded 1train run 0: X shape (120, 3, 1000)
Loaded 2train run 0: X shape (160, 3, 1000)
Loaded 3test run 0: X shape (160, 3, 1000)
Loaded 4test run 0: X shape (160, 3, 1000)
Train: X (400, 3, 1000), y (400,); Test: X (320, 3, 1000), y (320,)
train labels unique: [0, 1]; test labels unique: [0, 1]
```

**論文プロトコル完全一致**: 400 trials train (sessions 1-3) / 320 trials test (sessions 4-5) / 1000 サンプル / 2 クラス。

### interaug の挙動確認
[utils/interaug.py:10-16](intentflow/offline/utils/interaug.py#L10-L16) で `T == 750` のとき n_chunks=6 という旧 2b 用特別分岐があるが、窓長修正後 T=1000 になるので **else 分岐 `n_chunks = 8 if T % 8 == 0 else 7` → 8** に入る。これは論文 S&R の Ns=8（2b 用）と一致。旧 750 分岐は dead code 化するが残置（他 dataset で T=750 のケースが将来再発した場合の防御として）。

### 現在進行中の学習
```
pid=2331305  conda=intentflow  gpu=0
log=intentflow/offline/logs/train_tcformer_bcic2b_seed0_20260422_134722.log
results dir=intentflow/offline/results/tcformer_bcic2b_seed-0_aug-True_GPU0_<ts>/
config: tcformer.yaml（素 TCFormer、aug=interaug=True、seed=0、max_epochs_2b=500）
protocol: TCFormer 公式準拠（MOABB + stop=-0.5 + val=test + last-epoch save）
```

見積もり: 1 subject ≈ 500 epoch × ~2 sec ≈ 17 min、9 subjects で **~2.5 時間**。

### 追記 (2026-04-22 16:40 JST) — 2b source model 完走

学習は完走した。`pid=2331305` は終了済みで、`max_epochs=500` 到達後に subject 9 の test まで完了している。

- results dir: [intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347)
- summary: [results.txt](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347/results.txt)
- log: [intentflow/offline/logs/train_tcformer_bcic2b_seed0_20260422_134722.log](/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/logs/train_tcformer_bcic2b_seed0_20260422_134722.log)

主要結果:

- Average Test Accuracy: `87.74 ± 8.89`
- Average Test Kappa: `0.755 ± 0.178`
- Total Training Time: `169.96 min`

subject 別 accuracy:

- S1 `78.75`
- S2 `70.00`
- S3 `84.38`
- S4 `98.12`
- S5 `97.81`
- S6 `83.75`
- S7 `93.12`
- S8 `94.69`
- S9 `89.06`

解釈:

1. 2b source model の baseline は確保できた。段階 1 (`source_only / vanilla-both / hybrid`) に進める。
2. 旧 2b 結果は前処理修正前の可能性があるため、この run を新しい基準とする。
3. 2b first pass はこの checkpoint 群を `checkpoint_dir` として使えばよい。

### 合意済み段階 1 (P0) との関係
- `260422_next_implementation_plan.md` の段階 1「2b first pass（source_only / vanilla-both / hybrid）」の **前提 source model を今このタイミングで作っている**。
- 段階 1 の 3 条件比較は、この source checkpoint が完走してから着手。
- **interaug は source 側で有効（TCFormer 公式デフォルトに合わせる）**。first pass 評価側の interaug は従来通り凍結。

### Codex への申し送り
1. 本 repo の 2b 側は **今日までバグ同然だった**（正規 test session を一度も使っていなかった）。既存の 2b 実験結果があれば全て無効。
2. 修正後プロトコルは TCFormer 公式準拠。**2a 側は不変**（既存 baseline 保護、CLAUDE.md 合意）。
3. 2a vs 2b の非対称が 1 点残る: **2a は 80/20 val split**（`BCICIV2a.setup()` で session_T を分割）、**2b は val=test**。ただし `train_pipeline.py` は val を選択ロジックに使っていないので、**test 汚染による checkpoint 選択リークは発生しない**。val 曲線の意味だけが両 dataset で異なる。
4. worst-subject 判定、adaptability 閾値など段階 1 以降の合意は変更なし。source model が完走次第、first pass スクリプト作成に進む。
