# 作業上の約束

## 命名と配置
- 新しい実験 config は、baseline を上書きせず、説明的な suffix を付ける。
- 新しい run 補助スクリプトは `intentflow/offline/scripts/` に置く。
- 新しい研究メモは repo root ではなく、`docs/research_progress/` または `docs/related_research/` に置く。
- 新しい結果要約には、それを生んだ exact config、script、output path を必ず書く。

## 実験衛生
- 明示的に grid や sweep を回す場合を除き、変更因子は1つにする。
- dataset、split、seed方針、前処理、metrics、hardware assumption を書く。
- 改善と同じくらい、回帰も明確に報告する。
- 根拠が弱いときは、不確実性を最も小さくできる最小 run を提案する。

## 解析衛生
- 観測事実と解釈を分ける。
- 再現性のため、正確な file path と command を書く。
- 長いログを読むときは、summary 前に failure signal、final metrics、confounder の痕跡を探す。

## オンライン安全性
- 低信頼予測に対する abstention 動作を維持する。
- オンラインコード変更時は、latency、buffering、protocol 互換性のリスクを必ず明示する。
