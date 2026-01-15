#!/usr/bin/env python3
"""
ゼミ発表用の図を生成するスクリプト
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# 実験結果データ
data = {
    "BCIC 2a": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Base": [91.4, 63.8, 89.7, 84.5, 74.1, 86.2, 94.8, 86.2, 91.4],
        "Hybrid": [81.0, 75.9, 87.9, 79.3, 79.3, 81.0, 94.8, 87.9, 84.5],
    },
    "BCIC 2b": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Base": [82.5, 81.3, 73.8, 98.8, 79.8, 87.5, 80.0, 83.0, 77.5],
        "Hybrid": [77.5, 80.0, 75.0, 100.0, 81.0, 82.5, 80.0, 78.4, 72.5],
    },
    "HGD": {
        "subjects": list(range(1, 15)),
        "Base": [90.0, 83.8, 97.5, 100.0, 100.0, 95.6, 86.9, 96.3, 100.0, 95.0, 92.5, 98.8, 90.0, 75.0],
        "Hybrid": [77.5, 64.4, 92.5, 98.1, 75.0, 80.6, 51.3, 68.1, 99.4, 87.5, 93.1, 93.1, 68.1, 61.3],
    },
}

summary = {
    "BCIC 2a": {"Base": (84.67, 9.25), "Hybrid": (83.52, 5.52)},
    "BCIC 2b": {"Base": (82.67, 6.73), "Hybrid": (80.76, 7.39)},
    "HGD": {"Base": (92.95, 7.01), "Hybrid": (79.29, 14.61)},
}

training_time = {
    "BCIC 2a": {"Base": 43.3, "Hybrid": 23.7},
    "BCIC 2b": {"Base": 131.3, "Hybrid": 12.1},
    "HGD": {"Base": 212.2, "Hybrid": 79.5},
}

def create_main_comparison():
    """メイン比較図"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'Base': '#2E86AB', 'Hybrid': '#E94F37'}
    
    # (A) 精度比較
    ax = axes[0, 0]
    datasets = list(summary.keys())
    x = np.arange(len(datasets))
    width = 0.35
    
    base_acc = [summary[d]["Base"][0] for d in datasets]
    base_std = [summary[d]["Base"][1] for d in datasets]
    hyb_acc = [summary[d]["Hybrid"][0] for d in datasets]
    hyb_std = [summary[d]["Hybrid"][1] for d in datasets]
    
    bars1 = ax.bar(x - width/2, base_acc, width, yerr=base_std, label='Base (TCFormer)', 
                   color=colors['Base'], capsize=5)
    bars2 = ax.bar(x + width/2, hyb_acc, width, yerr=hyb_std, label='Hybrid Dynamic', 
                   color=colors['Hybrid'], capsize=5)
    
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(A) 精度比較: 全データセット')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(60, 100)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 差分を表示
    for i, (b, h) in enumerate(zip(base_acc, hyb_acc)):
        diff = h - b
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'{diff:+.1f}%', xy=(i + width/2, h + hyb_std[i] + 2), 
                   ha='center', fontsize=11, color=color, fontweight='bold')
    
    # (B) 訓練時間比較
    ax = axes[0, 1]
    base_time = [training_time[d]["Base"] for d in datasets]
    hyb_time = [training_time[d]["Hybrid"] for d in datasets]
    
    bars1 = ax.bar(x - width/2, base_time, width, label='Base', color=colors['Base'])
    bars2 = ax.bar(x + width/2, hyb_time, width, label='Hybrid', color=colors['Hybrid'])
    
    ax.set_ylabel('Training Time (min)')
    ax.set_title('(B) 訓練時間比較')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 短縮率を表示
    for i, (b, h) in enumerate(zip(base_time, hyb_time)):
        speedup = b / h
        ax.annotate(f'{speedup:.1f}x faster', xy=(i + width/2, h + 5), 
                   ha='center', fontsize=10, color='green', fontweight='bold')
    
    # (C) HGD被験者別（問題の可視化）
    ax = axes[1, 0]
    subjects = data["HGD"]["subjects"]
    x_hgd = np.arange(len(subjects))
    
    ax.bar(x_hgd - 0.2, data["HGD"]["Base"], 0.4, label='Base', color=colors['Base'], alpha=0.8)
    ax.bar(x_hgd + 0.2, data["HGD"]["Hybrid"], 0.4, label='Hybrid', color=colors['Hybrid'], alpha=0.8)
    
    # 問題のある被験者をハイライト
    problem_subjects = [5, 7, 8, 13]  # 0-indexed: 4, 6, 7, 12
    for ps in problem_subjects:
        ax.axvspan(ps - 1 - 0.5, ps - 1 + 0.5, alpha=0.2, color='red')
    
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('(C) HGD: 被験者別精度 (赤領域=大幅低下)')
    ax.set_xticks(x_hgd)
    ax.set_xticklabels(subjects)
    ax.legend()
    ax.set_ylim(40, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # (D) 精度低下の分析
    ax = axes[1, 1]
    
    degradation = []
    for ds in datasets:
        base = summary[ds]["Base"][0]
        hyb = summary[ds]["Hybrid"][0]
        degradation.append(base - hyb)
    
    colors_bar = ['#4CAF50' if d < 2 else '#FFC107' if d < 10 else '#F44336' for d in degradation]
    bars = ax.bar(datasets, degradation, color=colors_bar, edgecolor='black')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='許容範囲 (2%)')
    
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('(D) Baseからの精度低下')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 数値を表示
    for bar, d in zip(bars, degradation):
        height = bar.get_height()
        ax.annotate(f'-{d:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height + 0.5),
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_problem_analysis():
    """問題分析図"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # (A) Base精度 vs 精度低下の関係
    ax = axes[0]
    
    base_accs = [summary[d]["Base"][0] for d in summary.keys()]
    drops = [summary[d]["Base"][0] - summary[d]["Hybrid"][0] for d in summary.keys()]
    
    ax.scatter(base_accs, drops, s=200, c=['blue', 'orange', 'red'], edgecolors='black', linewidth=2)
    
    for i, ds in enumerate(summary.keys()):
        ax.annotate(ds, (base_accs[i], drops[i]), xytext=(10, 10), 
                   textcoords='offset points', fontsize=12)
    
    # 傾向線
    z = np.polyfit(base_accs, drops, 1)
    p = np.poly1d(z)
    x_line = np.linspace(80, 95, 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.5, label=f'Trend: higher base → larger drop')
    
    ax.set_xlabel('Base Model Accuracy (%)')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('(A) Base精度が高いほどTTTで劣化')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # (B) 被験者ごとの改善/悪化
    ax = axes[1]
    
    all_diffs = []
    all_labels = []
    
    for ds in data.keys():
        for i, s in enumerate(data[ds]["subjects"]):
            diff = data[ds]["Hybrid"][i] - data[ds]["Base"][i]
            all_diffs.append(diff)
            all_labels.append(f'{ds[:4]}-S{s}')
    
    colors_diff = ['green' if d > 0 else 'red' for d in all_diffs]
    
    improved = sum(1 for d in all_diffs if d > 0)
    degraded = sum(1 for d in all_diffs if d < 0)
    unchanged = sum(1 for d in all_diffs if d == 0)
    
    ax.pie([improved, degraded, unchanged], 
           labels=[f'Improved\n({improved})', f'Degraded\n({degraded})', f'Unchanged\n({unchanged})'],
           colors=['#4CAF50', '#F44336', '#9E9E9E'],
           autopct='%1.1f%%', startangle=90, explode=[0.05, 0.05, 0])
    
    ax.set_title(f'(B) 被験者別の改善/悪化比率\n(全{len(all_diffs)}被験者)')
    
    # (C) TTTの問題点の概念図
    ax = axes[2]
    ax.axis('off')
    
    text = """
    【TTTがうまくいかない理由】
    
    1. 訓練/テストの挙動不一致 ⭐最重要
       ┌─────────────────────────────┐
       │ 訓練時: 静的α（常にTTT適用）│
       │ テスト: 動的α（エントロピー）│
       │ → 学習と評価で挙動が異なる │
       └─────────────────────────────┘
    
    2. 高品質データでの問題
       ┌─────────────────────────────┐
       │ HGD: Base = 93% (最適解)    │
       │ → 不確実な予測が少ない     │
       │ → TTTの学習機会が不足      │
       └─────────────────────────────┘
    
    3. 学習不足 (Early Stopping)
       ┌─────────────────────────────┐
       │ Base: 長時間学習 (収束)     │
       │ Hybrid: 早期停止 (未収束)   │
       │ TTTアダプターが未学習      │
       └─────────────────────────────┘
    """
    
    ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_title('(C) 問題点のまとめ')
    
    plt.tight_layout()
    return fig


def create_improvement_proposal():
    """改善提案図"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    proposal = """
╔══════════════════════════════════════════════════════════════════════════╗
║                        改善提案ロードマップ                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  【短期】即座に実装可能                                                   ║
║  ├── 1. 訓練時もEntropy Gatingを有効化 ⭐最優先                          ║
║  │       entropy_gating_in_train: True                                  ║
║  │                                                                       ║
║  ├── 2. エントロピー閾値の調整 (0.95 → 0.7)                              ║
║  │                                                                       ║
║  ├── 3. TTT学習率の低減 (0.01 → 0.001)                                   ║
║  │                                                                       ║
║  └── 4. max_epochs増加 (十分な学習を保証)                                ║
║                                                                          ║
║  【中期】論文レベルの貢献                                                 ║
║  ├── 5. データセット適応的ゲーティング                                   ║
║  │                                                                       ║
║  └── 6. TTTアダプターの事前学習                                          ║
║                                                                          ║
║  【期待される効果】                                                       ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │ 現状:  精度低下 1-14%,  訓練時間 2-11x短縮                       │    ║
║  │ 目標:  精度維持 ±1%,   訓練時間 3-10x短縮                       │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    
    ax.text(0.5, 0.5, proposal, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    return fig


if __name__ == "__main__":
    output_dir = "/workspace-cloud/seiya.narukawa/intentflow/scripts/"
    
    print("Creating main comparison figure...")
    fig1 = create_main_comparison()
    fig1.savefig(f"{output_dir}seminar_main_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: seminar_main_comparison.png")
    
    print("Creating problem analysis figure...")
    fig2 = create_problem_analysis()
    fig2.savefig(f"{output_dir}seminar_problem_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved: seminar_problem_analysis.png")
    
    print("Creating improvement proposal figure...")
    fig3 = create_improvement_proposal()
    fig3.savefig(f"{output_dir}seminar_improvement_proposal.png", dpi=150, bbox_inches='tight')
    print(f"Saved: seminar_improvement_proposal.png")
    
    print("\nAll seminar figures generated!")

