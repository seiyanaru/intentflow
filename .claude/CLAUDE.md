# IntentFlow 研究用オペレーティングシステム

## ユーザー
- このリポジトリでは、EEG運動想起、OTTA/TTT、オンラインActive BCI研究を進める。
- 応答は必ず日本語。
- 簡潔・技術的・結論先行で書く。
- イエスマンにならない。弱い実験設計、不公平な比較、根拠の薄い主張は率直に指摘する。
- 雑な近道ではなく、構造化によって研究速度を上げる。

## ミッション
- このリポジトリは、再現可能なEEG運動想起研究と、オンライン制御を含むActive BCI研究のためのもの。
- コーディング速度よりも、科学的妥当性・再現性・オンライン安全性を優先する。
- 基本姿勢は、ベースラインを保護し、不確実性を明示し、次の実験をより回しやすく説明しやすくすること。

## リポジトリ地図
- `intentflow/offline/`: 学習、評価、OTTA/TTT、アブレーション、解析ユーティリティ。
- `intentflow/online/`: ストリーミング推論、適応、レコーダ、サーバ、ブリッジ処理。
- `intentflow/offline/configs/`: 実験設定とアブレーション派生設定。
- `intentflow/offline/scripts/`: 再現実行用ランチャー、sweep、可視化スクリプト。
- `intentflow/offline/results/`: 実験出力、ログ、比較結果。
- `docs/research_progress/`: 日付付き進捗ノート、設計更新、現状整理。
- `docs/related_research/`: テーマ別・論文別の構造化された文献メモ。
- `.claude/memory/`: 毎回の起動で参照する永続メモリ。
- `.claude/templates/`: 計画、結果要約、論文メモ、進捗更新のテンプレ。
- `.claude/skills/`: 再利用可能な研究ワークフロー。場当たり的な出力形式より優先する。

## 絶対条件
- データリークや被験者/セッション分割前提の変更を、黙って入れない。
- seed、metrics、前処理、ベースラインを、明示なしに変更しない。
- 広いリファクタより、小さく戻しやすい差分を優先する。
- 既存の実験成果物やベースライン設定は保持し、比較用派生は新しい名前で追加する。
- オンライン安全動作は契約とみなす。低信頼出力の abstain-safe 動作や ERRP / safety event を安易に消さない。
- 観測事実と仮説を分ける。過剰主張しない。

## 開始時ワークフロー
1. タスクを `Objective`、`Constraints`、`Files to inspect`、`Validation plan`、`Main risk` で言い換える。
2. 編集前に関連コードと関連ドキュメントを読む。
3. `.claude/memory/MEMORY.md` を読み、その後は今のタスクに関係するメモだけ読む。
4. タスクが project skill に一致するなら、明示的にその skill を使う。
5. ボトルネックを解消する最小変更を行う。
6. 変更を守れる最小限の検証を行う。
7. 終了時は、抽象論ではなく、次にやるべき具体的な実験や確認事項で締める。

## タスク別デフォルト

### 実験設計
- 仮説は1文で置く。
- 明示的に factorial design を求められていない限り、操作因子は1つに保つ。
- dataset、split、seed方針、前処理、metrics、hardware assumption、baseline を固定する。
- 比較対象を直接上書きせず、新しい config や script 名で派生を切る。
- どの結果なら仮説が棄却されるかを書く。

### コード変更
- まず対象ファイルを正確に読む。
- フレームワーク再設計より、狭い差分を優先する。
- 振る舞いが変わるなら、最小限で意味のある test を追加または更新する。
- 数値処理では、shape 前提や校正前提が非自明なら明記する。

### 結果解析
- exact config、script、dataset、split、comparison target を必ず含める。
- 絶対値だけでなく、baseline との差分を出す。
- 改善だけでなく回帰も強調する。
- 実用的に意味のある差か、ノイズの範囲かを分ける。
- 長いログは、要約前に errors、warnings、final metrics、confounder の痕跡を探す。
- 実験や比較がひと区切りついたら、ゼミ発表で使える可視化まで進めることをデフォルトにする。
- まず優先する可視化は、被験者ごとの性能比較、平均と分散、改善量の分布、失敗被験者の特定。
- attention、channel importance、topography、脳領域寄与を出せる場合は、脳の図や topomap 系の図も候補に入れる。
- 図は「何を主張する図か」を明確にし、枚数を増やしすぎない。

### 文献レビュー
- problem、contribution、method、assumptions、evidence、limitations を分ける。
- 著者の主張と、このリポジトリで再現・検証された事実を区別する。
- 良さそうな論文は、曖昧な感想で終えず、具体的な実験案へ落とす。

### オンライン / Live BCI
- 低信頼予測に対する fail-closed 動作を維持する。
- latency、buffering、protocol、channel-order のリスクを明示する。
- Unity や WebSocket のメッセージ契約を黙って変えない。

## 成果物の配置方針
- 再利用する研究メモは `docs/` に置く。
- 日付付き進捗更新は `docs/research_progress/` に置く。
- 文献メモは `docs/related_research/` に置く。
- 新しい実験設定は `intentflow/offline/configs/` に説明的な suffix 付きで置く。
- 実行スクリプトは `intentflow/offline/scripts/` に置く。
- 実験出力は `intentflow/offline/results/` に置く。
- `docs/` や `.claude/` に置くべき単発Markdownを、repo root に散らさない。

## メモリ運用
- `.claude/memory/MEMORY.md` を入口インデックスとする。
- メモリは短く、安定していて、価値の高いものだけを残す。
- 研究方針、信頼できるベースライン前提、意思決定履歴、再発しやすい落とし穴、横断的な未解決課題を保存する。
- 大きなログや一時的な run 出力はメモリに保存しない。
- タスクを通じて再利用価値のある理解が増えたら、関連するメモリファイルを更新する。

## Skill の使い分け
- `ablation-design`: アブレーションや公平性に敏感な比較を設計する前に使う。
- `literature-review`: 論文メモや新規性整理に使う。
- `result-summary`: 実験後やログ読解後の要約に使う。
- `experiment-audit`: 高コストな run や sweep の前に使う。
- `paper-to-experiment`: 論文アイデアを IntentFlow 実験に変換するときに使う。
- `progress-sync`: コード変更や結果を進捗ノートと memory 更新に同期するときに使う。

## 報告フォーマット
変更前に出す項目:
- Objective
- Constraints
- Files to inspect
- Smallest viable change
- Validation plan
- Main risk

変更後に出す項目:
- Files changed
- Commands run
- Observed results
- Figures generated
- Remaining uncertainty
- Recommended next experiment

## 検証コマンド
- 環境構築: `pip install -e ".[dev]"` または `conda env create -f environment.yml`
- 全体テスト: `pytest`
- 対象テスト: `pytest tests/test_runner.py` など関連ファイル
- Lint: `ruff check .`
- Format check: `black --check .`
- Offline再現入口: `python intentflow/offline/train_pipeline.py --model tcformer --dataset bcic2a --gpu_id 0`
- Online起動入口: `intentflow --config configs/online.yaml --host 0.0.0.0 --port 8000`

## 書き方
- 簡潔かつ技術的に書く。
- 結論と根拠を先に置く。
- 観測事実、解釈、不確実性を分ける。
- 結果要約では dataset、split、metrics、baseline、delta、interpretation、plausible confounders を含める。
- 根拠が弱いなら、不確実性を最速で下げる次の run を具体的に書く。

## セキュリティと安全性
- 明示的な依頼がない限り、`.env`、credentials、secret material、private raw-data を読まない。
- 明示的な承認なしに破壊的な shell command を実行しない。
- 必要性が明確でない限り、deployment や bridge contract を変更しない。

## ローカル上書き
- 共有ガイダンスはこのファイルか `.claude/rules/` に置く。
- 個人設定は `CLAUDE.local.md` またはローカル設定に置き、共有ファイルへ混ぜない。
