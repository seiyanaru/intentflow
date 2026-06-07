# IntentFlow メモリ索引

最初にこのファイルを読み、その後は今のタスクに関係する項目だけ開くこと。

## 基本メモリ
- `project-overview.md`: リポジトリの目的、主要研究トラック、主な作業場所。
- `working-conventions.md`: 命名、成果物の配置、報告期待値、実験衛生。
- `active-questions.md`: 現在の研究課題、詰まりやすい論点、再訪すべき意思決定。

## 実験知見・研究方向（2605）
- `dc-replay-empirical-ceilings.md`: TCFormer凍結+予測補正型OTTAの精度天井（L1 ≲ +3.4pp、3データセット）と L3 model-state commit の no-op。
- `research-direction-2605.md`: 精度主軸へ確定。alignment-first selective minimal feature adaptation が本命(EA + Base凍結)。
- `tcformer-hybrid-failure.md`: triggered feature adaptation は repo で失敗済み(Hybrid HGD -13.66)。差分は EA + Base凍結。
- `prior-art-novelty-bounds.md`: DA-DC/family-lawを先取りする先行研究(Hidden Clones/Gorbett&Jana/T-TIME等)とP2オンラインno-op。防衛可能な交差点と3戦略(A局所化/B選択ラベル/C検出専用)。label-free精度の全方向閉鎖・symmetry/S1/S5棄却・Law-2 CV検証も追記。
- `thesis-plan-260606.md`: M2修論の確定方向（オフライン特徴づけ：壁CLAIM1＋機構CLAIM2＋monitor CLAIM3）と半年実行プログラム（E1=Lee2019 monitor汎用性がgo/no-go）。

## メモリに入れるもの
- 安定した研究方針
- 信頼できるベースライン前提
- 重要な設計判断
- 繰り返し発生する落とし穴
- 複数タスクにまたがる未解決課題

## メモリに入れないもの
- 生ログ
- 完全な実験出力
- 単発セッション用の一時TODO
- すでに `docs/related_research/` にある長い文献要約

## 更新ルール
タスクを通じて再利用価値のある理解が増えたら、終了前に最小の関連メモリファイルを更新する。
