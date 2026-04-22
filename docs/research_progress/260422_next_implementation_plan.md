# 260422 次実装プラン（ゼミ 260420 を受けて）

- 目的: 2026-04-20 ゼミ資料（`docs/research_progress/ゼミ資料/260420_narukawa.pdf`）で浮いた未決事項を、次に書くコード差分まで落とし込む。
- 統合元: `260422_post_seminar_implementation_plan.md` の論点も本ファイルへ統合済み。
- 使い方: 各セクションは `Claude` ブロックと `Codex` ブロックに分けて並べる。片方が空でも両方を残す。結論が一致したら `合意` 行を追記する。合意後に配下の実装へ進む。
- 対象ブランチ: `main`（現状 WIP 多数。新規配置は config / script / 新モジュール追加で、既存最良 `hybrid@0.01` は保護）。

---

## 0. 現状サマリ（共通の前提）

- オフライン最良: `tcformer_otta` の `shallow_mean_deep_both` × `mom=0.01` × Tri-Lock で 9 人平均 **81.98 %**、害 (NTR-S@0.5pp) **0/9**、WSD **-0.34 pp**。
- 最悪被験者 **S2** は hybrid で +0.00 pp に戻せたが、学習時アプローチ（試作 B）ではまだ素の学習より悪化する条件が残る。
- オンライン: Unicorn → UDP → `ttt_broadcaster` → WebSocket → Unity HUD が通ったが、遅延 / 判定棄権率 / 関門通過分布は未計測。
- 因果同定のコア所見: 害は `shallow × running_var` に局在し、**S2 型 / S7 型で符号が反転**する被験者依存 drift。
- 絶対守るもの: `hybrid@0.01` の config / 結果を壊さない。比較用派生は別 suffix。

---

## 1. Phase 1（今週〜04-29）: オンライン実被験者 1 セッションの定量指標取得

### Claude
- **仮説**: オフライン hybrid をそのままオンラインに流しても、`(遅延 < 250 ms, 判定棄権率 10–40 %, 関門通過分布がオフラインと同形)` が同時に満たせる。満たせない軸があれば、そこがオンライン化の最初のボトルネック。
- **やること（優先順）**:
  1. `intentflow/online/ttt_broadcaster.py` に **ログスキーマを 1 本追加**。`window_id, t_recv_udp, t_infer_start, t_infer_end, t_ws_emit, pmax, sal, energy, gate_pass, abstain_reason, intent, confidence, protocol_version` を JSONL で `intentflow/online/results/<session_id>.jsonl` に吐く。既存の broadcast は触らない（契約保護）。
  2. Unity 側は観測のみ。HUD のメッセージ契約は変更しない。
  3. セッション条件を固定化: `configs/online/unicorn_live_s0.yaml` を新規作成（8 ch, 250 Hz, 窓 4.0 s / hop 0.25 s, hybrid@0.01 重みのパス固定）。既存 `online.yaml` は維持。
  4. 被験者は `s0`（実装者本人）で開始。左右運動想起 × 20 試行 × 2 ブロック。事前に resting 30 s を記録し、`StreamNormalizer` の warm 状態を確認。
  5. 解析は `scripts/analysis/plot_online_session_s0.py`（新規）で、(a) end-to-end 遅延の histogram、(b) 時間軸上の関門通過フラグ、(c) pmax/sal/energy の分布をオフライン BCIC-IV 2a `session_E` と重ね描き。
- **判定（事前に置く）**:
  - 合格: 遅延中央値 < 250 ms かつ 95 %tile < 400 ms、棄権率 10–40 %、関門通過分布がオフラインと KS 検定で同形（暫定 p > 0.05）。
  - 不合格 A（遅延超過）: `ttt_broadcaster` の `torch.no_grad` / batch=1 固定 / CPU⇔GPU 往復の削減から着手。
  - 不合格 B（棄権率が 0 または 100 に張り付く）: 閾値 `τ_pmax, τ_sal, energy_range` を実被験者分布でキャリブレーション。**ただし hybrid オフライン結果を壊す変更は別 config に分離**。
  - 不合格 C（分布がオフラインと大きく乖離）: 前処理一致性の再確認（referencing, filter order, サンプルレート変換）。モデル側は触らない。
- **リスク**: 実被験者データの前処理が `session_T/E` と同形でないと、オフラインでの因果同定結果が意味を失う。最初の 1 セッションは必ず「前処理一致性の証明」を優先し、推論性能の議論はその後。
- **Non-goal**: hybrid を置き換える学習時手法の試作は Phase 2 まで手を出さない。

### Codex
- **仮説**: Phase 1 の目的は「モデル改善」ではなく「計測可能性の確立」。現行 live 経路は `confidence_threshold` 主体で、オフライン `hybrid@0.01` と完全同値ではないため、まずはオンラインで何が起きているかを観測可能にする。
- **やること（優先順）**:
  1. `intentflow/online/server/ttt_broadcaster.py` に telemetry 出力を追加。最小必須キーは `window_id, recv_ts, infer_start_ts, infer_end_ts, send_ts, conf, pred_idx, abstained, abstain_reason`。
  2. 同ファイルで `--telemetry_path` / `--metrics_every_n` を CLI 引数として追加し、標準出力だけに依存しない。
  3. `intentflow/online/recorder/unicorn_udp_reader.py` の `stats()` を定期的に採取し、`dropped_packets` と遅延指標を同じログに束ねる。
  4. 解析は新規 `intentflow/online/scripts/analyze_session_log.py` を作成し、遅延・棄権率・意図送信頻度を自動集計する。
  5. Tri-Lock の完全移植は Phase 1 後半で実施。先に計測を固めてから `pmax/sal/energy` を live 経路へ段階導入する。
- **判定（事前に置く）**:
  - 合格: 1 セッションで `recv->infer`, `infer->send`, `abstain_rate`, `drop_rate` の 4 指標が自動で出る。
  - 不合格 A（ログ欠損）: 例外時にも 1 レコード残す finally ログを入れる。
  - 不合格 B（遅延評価不能）: window 生成時刻 `recv_ts` が取れていないので reader 側で採時して渡す。
  - 不合格 C（閾値議論が空転）: abstain 理由の粒度を `low_conf` / `gate_fail` / `other` に固定する。
- **リスク**:
  - 現行 `run_unicorn_live.sh` は 8ch 入力を 22ch モデルへ zero-pad/truncate するため、精度議論は危険。Phase 1 は transport/latency 評価に限定する。
  - `ttt_broadcaster` と FastAPI (`online/server/app.py`) の二系統が並立しており、後で指標定義がずれやすい。まず ttt_broadcaster を基準実装に固定する。
- **Non-goal**:
  - Phase 1 での学習済み重み更新、モデル構造変更、UI 改造はしない。

### 合意
<!-- Claude と Codex の結論が一致したら日付付きで書く。例: 2026-04-23 両者: 仕様 OK、実装着手。 -->

---

## 2. Phase 2（〜05-06）: 学習時に S2 を救う試作 B の最終確認

### Claude
- **仮説**: `interaug_strongest + gain_jitter(0.025)` の二軸追加で、S2 の素の学習比 Δ が 5 seed 中 3 seed 以上で正になる。届かなければ学習時路線を一度凍結し、`試作 C (virtual BN)` の検討に移る。
- **やること**:
  1. `intentflow/offline/configs/tcformer_aug_shinv/` に `tcformer_aug_interaug_gainjitter_0025.yaml` を新規追加（既存 `aug_shinv` 系とは suffix で区別）。
  2. 学習ループは `intentflow/offline/models/tcformer_aug_shinv.py` を基に最小拡張。新しい augmentation は `augment` サブモジュール側に閉じる。**モデル forward / 既存 BN 構造は触らない**。
  3. ランチャーは `scripts/run_phaseB_interaug_gain.sh` を新規追加。`--seed 0..4`、`--gpu_id 0` 固定。既存 `run_bs1_*` 系と干渉しない命名に。
  4. 評価は 9 人全員 × `session_E`、指標は 9 人平均 Δ、S2 Δ、NTR-S@0.5pp、WSD。hybrid@0.01 との二段比較（学習単体 / 学習 + hybrid）を出す。
  5. 可視化は `scripts/analysis/plot_phaseB_interaug_gain.py` を新規。被験者別 Δ、S2 の seed 分布、interaug 強度 × gain_jitter の 2D ヒートマップ。
- **判定**:
  - 成功: S2 Δ(aug/plain) > 0 が 3/5 seed 以上、かつ 9 人平均は既存 plain の ±0.5 pp 内（全体を崩さない）。→ 試作 C に接続する根拠とする。
  - 失敗: **学習時路線は一度凍結**し、進捗ノートに凍結理由を明記。凍結後は Phase 1 の実被験者セッション × hybrid の範囲で様子を見る。
- **リスク**:
  - 5 seed × 9 被験者 × 数 aug の組み合わせを広げすぎると、探索空間が一気に膨れて S2 救済が confounder で汚染される。**操作因子は augmentation の強度 1 軸 + gain_jitter の 2 軸に限定**。それ以上は別 config でしか入れない。
  - `interaug` は被験者間でサンプルを混ぜるため、**cross-subject split の契約が崩れていないかの確認が必須**。少なくとも同一 fold 内で train/val を混ぜない。
- **Non-goal**: 試作 C（virtual BN）のコードは書かない。ドキュメントと接続点だけ用意する。

### Codex
- **仮説**: 既存 `run_phaseB_strong_suite.sh` の拡張だけで、S2 判定に必要な再現性は十分に取れる。新しい学習アーキテクチャは不要。
- **やること**:
  1. `intentflow/offline/scripts/run_phaseB_strong_suite.sh` を 5-seed 対応に拡張するか、薄いラッパー `run_phaseB_strong_5seeds.sh` を新規作成。
  2. 条件は固定して増やしすぎない。`strong_plain`, `strong_gain0025`, `strong_gain0025_varinv_l002` を主比較軸にする。
  3. `intentflow/offline/scripts/analysis/summarize_phaseB_strong_5seeds.py` を追加し、`S2 Δ>0 が 3/5 以上` を機械判定する。
  4. 出力は `CSV + Markdown` の 2 形式で保存し、`NTR-S@0.5pp` と `WSD` も同時に出す。
  5. `interaug` 条件は `tcformer_aug_shinv_interaug.yaml` 系に分離し、既存最良設定を上書きしない。
- **判定**:
  - 成功: S2 が 3/5 seed 以上で改善し、かつ worst-case（WSD）が plain より悪化しない。
  - 失敗: S2 が未改善または WSD 悪化。学習時路線は凍結し、オンライン側の計測・安全性改善へ集中。
- **リスク**:
  - 探索軸を増やすと解釈不能になる。Phase 2 は「S2 救済可否の判定実験」に限定する。
  - 実行失敗時の再開戦略がないと 5-seed が止まる。seed ごと独立結果ディレクトリを必須化する。
- **Non-goal**:
  - virtual BN 実装、複雑なドメイン生成 augmentation の導入はしない。

### 合意
<!-- 合意したら日付付きで書く。 -->

---

## 3. Phase 3（中期, 〜1 ヶ月）: `L_shallow_inv` の shallow 後段局所化

### Claude
- **仮説**: shallow BN の害は 6 層すべてに広がっているのではなく、`ch_DW_conv / ch_reduce_2 / temp_conv_2` の後段 3 層に局在している。局所化した不変性損失だけで S2 の shallow var drift を抑えられる。
- **やること**:
  1. まず解析のみ。`intentflow/offline/scripts/analysis/plot_shallow_layerwise_drift.py`（新規）で、shallow 6 層の `running_var` を `session_T / session_E` で per-subject に出し、S2 / S7 型で drift のどの層が支配的かを可視化。
  2. 解析結果に基づいて、`intentflow/offline/models/tcformer/tcformer.py` の BN 層に `layer_group` タグを付ける軽い改修（forward は変えない）。
  3. 学習時 `L_shallow_inv` は新しい `tcformer_shinv_local.py` を別ファイルで追加。既存 `tcformer_aug_shinv.py` には触らない。
- **判定**: 解析 Phase の結果が 9 人中 5 人以上で「後段 3 層に drift が集中」なら実装に進む。そうでなければ「6 層均等寄与」仮説に戻し、中期計画を再設計。
- **リスク**: 局所化の根拠が薄いまま学習時損失を増やすと、hybrid が既に達成した安全性を逆に壊す可能性。必ず解析 → 局所化 → 損失 の順。

### Codex
- **仮説**: 局所化は有望だが、先に「どの層の何が壊しているか」を統計で示す必要がある。モデルコードの構造変更は後回しでよい。
- **やること**:
  1. 解析優先。`shallow 6 層` の `running_mean/var drift`, `entropy`, `logit_norm` を被験者別に可視化。
  2. 実装初手は `tcformer.py` の改造ではなく、既存 module 名へのフックで計測する。
  3. 局所 `L_shallow_inv` は別モデルファイルで実装し、`tcformer_aug_shinv.py` を保護する。
  4. 評価は必ず `mean` と `safety` を同時報告（平均改善のみで採択しない）。
- **判定**:
  - Go: 局所層で drift が一貫して支配的で、かつ S2 改善に寄与する証拠が出る。
  - No-Go: 寄与が分散している、または安全性指標が悪化する。
- **リスク**:
  - 局所化で S3/S7 の deep 側益を消す可能性がある。`hybrid` と二段比較で必ず確認する。

### 合意
<!-- -->

---

## 4. Phase 4（長期, M2 後半〜）: Subject-adaptive rule と Active BCI 統合

### Claude
- **仮説**: S2 型（shallow var 更新で害）と S7 型（同操作で益）は、**学習セッション（session_T）の shallow BN 統計と source 統計のズレ方向**で事前に判別できる。できるなら、オンライン推論開始時に判別器を通して `shallow_var` を凍結するか更新するかを切り替えられる。
- **やること（原則として研究計画の言語化まで）**:
  1. `docs/research_progress/260422_subject_adaptive_rule_design.md` を別途用意し、判別器の入力候補（per-channel impedance proxy、resting α power、`session_T` の shallow running_stats）を列挙。
  2. オフラインで 9 人を S2 型 / S7 型に仮分類し、各群内で hybrid の挙動を分離評価。群内でさらに分散があるなら、単純な二値判別では足りない証拠。
  3. Active BCI 統合は、オンライン処理に OTTA hybrid を組み込み、ERRP ループで意図 → 制御 → 修正のフィードバックを作る。ERRP 検出は別研究トラックなので、hybrid の `abstain` と ERRP の `reject` が競合しないプロトコル設計を先に決める（fail-closed 優先）。
- **判定**: 仮分類で S2 型 / S7 型の群内分散が群間分散より小さい（F 統計量 > 1）なら判別器に進む根拠。そうでなければ、判別器ではなく「全被験者 safe-default + 個別にだけ unfreeze」という保守設計に倒す。
- **リスク**: 長期項は短期項の結果次第で意味が変わる。Phase 1–3 の結論が出るまで設計を詰めすぎない。

### Codex
- **仮説**: Subject-adaptive rule は「分類器を作る」より先に「安全な運用ポリシー」を作るべき。最初はルールベースで十分。
- **やること**:
  1. 初期案は二段階ポリシーにする。
     - default: `shallow_var freeze`（safe-default）
     - 例外: 事前特徴量が一定条件を満たす場合のみ部分 unfreeze
  2. 判別特徴量は `session_T` のみで算出可能なものに限定（運用時に追加収録を要求しない）。
  3. Active BCI 連携では `abstain` と `reject(ERRP)` の優先順位を仕様化して衝突を防ぐ。
- **判定**:
  - 9 被験者の群分離が不十分なら、adaptive switch は導入しない。
- **リスク**:
  - 被験者数が少ないまま複雑な判別器へ進むと過学習する。長期は保守設計優先。

### 合意
<!-- -->

---

## 5. 横断リスクと運用ルール

### Claude
- `hybrid@0.01` の config・重み・結果は保護する。Phase 2 / 3 はすべて別 suffix で派生させる。
- Phase 1 のオンラインログは PII を含まない（被験者 ID のみ）ことを前処理段で確認してから記録する。
- Phase 2 の `interaug` は cross-subject split の契約を壊さないことを test で守る（最低 1 つ、`tests/` 配下に pytest を追加する）。
- すべての Phase で、GPU 実行（`conda activate intentflow` + `--gpu_id 0`）を既定とし、CPU 暗黙フォールバックは禁止。
- 成果物配置は CLAUDE.md の規約に従う（configs / scripts / results / docs）。`.md` を repo root に散らさない。

### Codex
- `hybrid@0.01` の再現性担保として、すべての新規実験に `git hash / config dump / seed` を保存する。
- `protocol_version` が 1 と 3 で混在しているため、オンラインログは互換パーサを標準化する（読み手で壊れない形にする）。
- 追加テストを最低 2 本入れる。
  - `tests/test_ws_protocol.py` に telemetry 付きメッセージの後方互換確認。
  - 新規で分析スクリプトの smoke test（ファイル存在時に集計が落ちないこと）。
- オンライン経路の一次基準は `ttt_broadcaster` に置き、FastAPI 経路は追従させる。二重実装を同時に進めない。

### 合意
<!-- -->

---

## 6. 次の 1 アクション（今から着手するもの）

### Claude
- **Phase 1 の 1 タスクだけ先に走らせる**: `ttt_broadcaster.py` にログスキーマを追加 → `s0` 被験者で 1 セッション収録 → 遅延 / 棄権率 / 関門通過分布の可視化まで。ここを終えてから Phase 2 の学習時 sweep を起動する（GPU 時間と注意力の集中のため）。
- 着手差分は最小で 3 ファイル: `ttt_broadcaster.py` にロガー差し込み、`configs/online/unicorn_live_s0.yaml` 新規、`scripts/analysis/plot_online_session_s0.py` 新規。

### Codex
- **直近の 1 手**: `ttt_broadcaster.py` の telemetry 実装を最初に着手する。
- **最小差分**:
  - `intentflow/online/server/ttt_broadcaster.py`: ログ書き出しとメトリクス集計
  - `intentflow/online/scripts/analyze_session_log.py`: 解析
  - `configs/online/unicorn_live_s0.yaml`: 実験条件固定
- **完了定義**:
  - 1 回のセッション実行で JSONL が生成され、解析スクリプトで遅延・棄権率が数値化される。

### 合意
<!-- -->
