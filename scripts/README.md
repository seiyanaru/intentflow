# Scripts

実験実行・分析用スクリプト

## フォルダ構造

```
scripts/
├── experiments/     # 実験実行シェルスクリプト
│   ├── run_analysis_only.sh
│   ├── run_experiment.sh
│   ├── run_grid_search.sh
│   ├── run_hybrid_experiment.sh
│   ├── run_strategies.sh
│   ├── run_ablation_experiments.sh
│   ├── run_bcic2b_experiments.sh
│   ├── run_bcic2b_hybrid_only.sh
│   └── run_hgd_experiments.sh
├── analysis/        # 分析・可視化スクリプト
│   ├── analyze_hgd_failure.py
│   ├── visualize_*.py
│   ├── plot_*.py
│   └── seminar_figures.py
└── setup/           # セットアップスクリプト
    ├── setup_production.sh
    └── setup_scaffold.sh
```

## 使用方法

### 実験実行
```bash
bash scripts/experiments/run_experiment.sh
```

### 分析実行
```bash
python scripts/analysis/visualize_all_datasets.py
```
