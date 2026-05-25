# 2026年度 前期定期ゼミ資料

## Files

- `regular_seminar_2605_slides.tex`: 20分発表用 Beamer スライド
- `regular_seminar_2605_report.tex`: 配布・記録用の本文 TeX
- `analysis_summary.md`: 今回の 72h sweep の要点
- `make_regular_seminar_assets.py`: 結果 JSON/NPZ から図表を再生成するスクリプト
- `figures/`: スライド・本文で使う図
- `tables/`: 主要数値の CSV

## Build

LuaLaTeX が入っている環境で、このディレクトリから以下を実行します。

```bash
lualatex regular_seminar_2605_slides.tex
lualatex regular_seminar_2605_report.tex
```

図表を再生成する場合は、リポジトリルートから以下を実行します。

```bash
MPLCONFIGDIR=/tmp/matplotlib-intentflow python docs/research_progress/regular_seminar_2605/make_regular_seminar_assets.py
```

## Main Message

現時点の安全な主結果は Replay-SafeCommit。DC-Replay 系は平均精度を伸ばすシグナルを示したが、seed 安定性と HSC が悪化した。さらに best gain は L3 model-state commit ではなく、L1/L2 の非破壊な prediction correction と external memory で説明される。
