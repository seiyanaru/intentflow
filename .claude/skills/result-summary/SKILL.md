---
name: result-summary
description: このリポジトリの実験出力、ログ、解析ノートブックを、研究報告に使える要約へまとめるときに使う。
---

# 結果要約

実験、解析、検証 run の後に使う。

## チェックリスト
- 結果を生んだ exact config または code path を含める。
- dataset、split、metrics、baseline comparison を含める。
- 絶対値だけでなく、baseline との差分を報告する。
- その変化が実用上 meaningful か、統計的に convincing か、まだ uncertain かを書く。
- ゼミ発表で使える figure が必要なら、結果要約だけで終わらず可視化候補まで出す。
- 最低限の figure 候補は、subject-wise 比較、平均と分散、delta 分布、代表的失敗例。
- attention や channel importance があるなら、脳領域の寄与を示す図も候補に入れる。
- 残る不確実性を最も直接的に減らす次の実験を提案する。

## 出力フォーマット
- Exact config
- Dataset and split
- Metrics
- Delta from baseline
- Interpretation
- Figures to generate
- Remaining uncertainty
- Recommended next experiment
