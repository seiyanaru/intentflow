#!/usr/bin/env python3
"""
3データセット（BCIC 2a, 2b, HGD）の比較可視化スクリプト
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# 結果データ（手動入力 - results.txtから抽出）
# 注: 2aのBaseは interaug=True での結果
results = {
    "BCIC IV 2a": {
        "Base": {"acc": 84.67, "std": 9.25, "time": 45.0, "kappa": 0.796},  # TCFormer with interaug
        "Hybrid Static": {"acc": 82.57, "std": 5.89, "time": 24.0, "kappa": 0.768},  # best static
        "Hybrid Dynamic": {"acc": 83.52, "std": 5.52, "time": 24.0, "kappa": 0.780},  # best dynamic
    },
    "BCIC IV 2b": {
        "Base": {"acc": 82.67, "std": 6.73, "time": 131.31, "kappa": 0.653},
        "Hybrid Static": {"acc": 79.92, "std": 7.98, "time": 11.17, "kappa": 0.598},
        "Hybrid Dynamic": {"acc": 80.76, "std": 7.39, "time": 12.08, "kappa": 0.615},
    },
    "HGD": {
        "Base": {"acc": 92.95, "std": 7.01, "time": 212.19, "kappa": 0.906},
        "Hybrid Static": {"acc": 81.25, "std": 10.06, "time": 67.95, "kappa": 0.750},
        "Hybrid Dynamic": {"acc": 79.29, "std": 14.61, "time": 79.54, "kappa": 0.724},
    },
}

# 被験者ごとの詳細データ
subject_data = {
    "BCIC IV 2b": {
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Base": [82.50, 81.25, 73.75, 98.81, 79.76, 87.50, 80.00, 82.95, 77.50],
        "Hybrid Static": [77.50, 78.75, 66.25, 97.62, 84.52, 81.25, 81.25, 78.41, 73.75],
        "Hybrid Dynamic": [77.50, 80.00, 75.00, 100.0, 80.95, 82.50, 80.00, 78.41, 72.50],
    },
    "HGD": {
        "subjects": list(range(1, 15)),
        "Base": [90.0, 83.75, 97.50, 100.0, 100.0, 95.63, 86.87, 96.25, 100.0, 95.0, 92.50, 98.75, 90.0, 75.0],
        "Hybrid Static": [79.37, 75.0, 75.63, 98.12, 98.12, 80.0, 66.87, 74.37, 96.88, 85.0, 71.88, 88.75, 75.0, 72.50],
        "Hybrid Dynamic": [77.50, 64.38, 92.50, 98.12, 75.0, 80.62, 51.25, 68.12, 99.37, 87.50, 93.12, 93.12, 68.12, 61.25],
    },
}

def create_comprehensive_figure():
    """包括的な比較図を作成"""
    fig = plt.figure(figsize=(20, 16))
    
    # カラーパレット
    colors = {'Base': '#2E86AB', 'Hybrid Static': '#A23B72', 'Hybrid Dynamic': '#F18F01'}
    
    # 1. 全データセット精度比較 (メイングラフ)
    ax1 = fig.add_subplot(2, 3, 1)
    datasets = list(results.keys())
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, (model, color) in enumerate(colors.items()):
        accs = [results[ds][model]["acc"] for ds in datasets]
        stds = [results[ds][model]["std"] for ds in datasets]
        bars = ax1.bar(x + i*width, accs, width, label=model, color=color, yerr=stds, capsize=3)
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('(A) Accuracy Comparison Across Datasets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(datasets, fontsize=10)
    ax1.legend(loc='lower right')
    ax1.set_ylim(60, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 2. 訓練時間比較
    ax2 = fig.add_subplot(2, 3, 2)
    for i, (model, color) in enumerate(colors.items()):
        times = [results[ds][model]["time"] for ds in datasets]
        ax2.bar(x + i*width, times, width, label=model, color=color)
    
    ax2.set_ylabel('Training Time (min)', fontsize=12)
    ax2.set_title('(B) Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(datasets, fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 3. 精度 vs 訓練時間 トレードオフ
    ax3 = fig.add_subplot(2, 3, 3)
    markers = {'BCIC IV 2a': 'o', 'BCIC IV 2b': 's', 'HGD': '^'}
    
    for ds in datasets:
        for model, color in colors.items():
            ax3.scatter(results[ds][model]["time"], results[ds][model]["acc"],
                       c=color, marker=markers[ds], s=150, alpha=0.8,
                       label=f'{ds} - {model}' if ds == 'BCIC IV 2a' else '')
    
    # 凡例用のダミー
    for ds, marker in markers.items():
        ax3.scatter([], [], c='gray', marker=marker, s=100, label=ds)
    for model, color in colors.items():
        ax3.scatter([], [], c=color, marker='o', s=100, label=model)
    
    ax3.set_xlabel('Training Time (min)', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('(C) Accuracy vs Training Time Trade-off', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='lower right', fontsize=8, ncol=2)
    
    # 4. BCIC 2b 被験者別比較
    ax4 = fig.add_subplot(2, 3, 4)
    subjects_2b = subject_data["BCIC IV 2b"]["subjects"]
    x_2b = np.arange(len(subjects_2b))
    
    for model, color in colors.items():
        ax4.plot(x_2b, subject_data["BCIC IV 2b"][model], 'o-', color=color, label=model, markersize=8)
    
    ax4.set_xlabel('Subject ID', fontsize=12)
    ax4.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax4.set_title('(D) BCIC IV 2b: Per-Subject Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_2b)
    ax4.set_xticklabels(subjects_2b)
    ax4.legend(loc='lower right')
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.set_ylim(60, 105)
    
    # 5. HGD 被験者別比較
    ax5 = fig.add_subplot(2, 3, 5)
    subjects_hgd = subject_data["HGD"]["subjects"]
    x_hgd = np.arange(len(subjects_hgd))
    
    for model, color in colors.items():
        ax5.plot(x_hgd, subject_data["HGD"][model], 'o-', color=color, label=model, markersize=6)
    
    ax5.set_xlabel('Subject ID', fontsize=12)
    ax5.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax5.set_title('(E) HGD: Per-Subject Accuracy', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_hgd)
    ax5.set_xticklabels(subjects_hgd)
    ax5.legend(loc='lower right')
    ax5.grid(True, linestyle='--', alpha=0.3)
    ax5.set_ylim(40, 105)
    
    # 6. 性能低下分析
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Baseからの精度低下を計算
    degradation_static = []
    degradation_dynamic = []
    for ds in datasets:
        base_acc = results[ds]["Base"]["acc"]
        degradation_static.append(base_acc - results[ds]["Hybrid Static"]["acc"])
        degradation_dynamic.append(base_acc - results[ds]["Hybrid Dynamic"]["acc"])
    
    x_deg = np.arange(len(datasets))
    ax6.bar(x_deg - 0.2, degradation_static, 0.35, label='Hybrid Static', color=colors['Hybrid Static'])
    ax6.bar(x_deg + 0.2, degradation_dynamic, 0.35, label='Hybrid Dynamic', color=colors['Hybrid Dynamic'])
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('Accuracy Drop from Base (%)', fontsize=12)
    ax6.set_title('(F) Performance Degradation Analysis', fontsize=14, fontweight='bold')
    ax6.set_xticks(x_deg)
    ax6.set_xticklabels(datasets, fontsize=10)
    ax6.legend()
    ax6.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 注釈追加
    for i, (ds, deg_s, deg_d) in enumerate(zip(datasets, degradation_static, degradation_dynamic)):
        if deg_s > 0:
            ax6.annotate(f'-{deg_s:.1f}%', (i-0.2, deg_s+0.5), ha='center', fontsize=9)
        else:
            ax6.annotate(f'+{-deg_s:.1f}%', (i-0.2, deg_s-1.5), ha='center', fontsize=9)
        if deg_d > 0:
            ax6.annotate(f'-{deg_d:.1f}%', (i+0.2, deg_d+0.5), ha='center', fontsize=9)
        else:
            ax6.annotate(f'+{-deg_d:.1f}%', (i+0.2, deg_d-1.5), ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_efficiency_analysis():
    """効率性分析図を作成"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = {'Base': '#2E86AB', 'Hybrid Static': '#A23B72', 'Hybrid Dynamic': '#F18F01'}
    datasets = list(results.keys())
    
    # 1. 訓練時間短縮率
    ax1 = axes[0]
    speedup_static = []
    speedup_dynamic = []
    for ds in datasets:
        base_time = results[ds]["Base"]["time"]
        speedup_static.append(base_time / results[ds]["Hybrid Static"]["time"])
        speedup_dynamic.append(base_time / results[ds]["Hybrid Dynamic"]["time"])
    
    x = np.arange(len(datasets))
    ax1.bar(x - 0.2, speedup_static, 0.35, label='Hybrid Static', color=colors['Hybrid Static'])
    ax1.bar(x + 0.2, speedup_dynamic, 0.35, label='Hybrid Dynamic', color=colors['Hybrid Dynamic'])
    
    for i, (ss, sd) in enumerate(zip(speedup_static, speedup_dynamic)):
        ax1.annotate(f'{ss:.1f}x', (i-0.2, ss+0.2), ha='center', fontsize=10, fontweight='bold')
        ax1.annotate(f'{sd:.1f}x', (i+0.2, sd+0.2), ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Speedup Factor (x)', fontsize=12)
    ax1.set_title('Training Speedup vs Base', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 2. 効率性スコア (精度/時間)
    ax2 = axes[1]
    efficiency = {}
    for model in ['Base', 'Hybrid Static', 'Hybrid Dynamic']:
        efficiency[model] = [results[ds][model]["acc"] / results[ds][model]["time"] for ds in datasets]
    
    for i, (model, color) in enumerate(colors.items()):
        ax2.bar(x + (i-1)*0.25, efficiency[model], 0.25, label=model, color=color)
    
    ax2.set_ylabel('Efficiency (Acc% / min)', fontsize=12)
    ax2.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 3. パレート最適性分析
    ax3 = axes[2]
    markers = {'BCIC IV 2a': 'o', 'BCIC IV 2b': 's', 'HGD': '^'}
    
    all_points = []
    for ds in datasets:
        for model, color in colors.items():
            time = results[ds][model]["time"]
            acc = results[ds][model]["acc"]
            all_points.append((time, acc, ds, model, color, markers[ds]))
            ax3.scatter(time, acc, c=color, marker=markers[ds], s=150, alpha=0.8)
            ax3.annotate(f'{model[:3]}', (time, acc), textcoords="offset points", 
                        xytext=(5, 5), fontsize=8)
    
    ax3.set_xlabel('Training Time (min)', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Pareto Frontier Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # パレート最適線を描画
    sorted_points = sorted(all_points, key=lambda p: p[0])
    pareto_front = []
    max_acc = 0
    for p in sorted_points:
        if p[1] > max_acc:
            pareto_front.append(p)
            max_acc = p[1]
    
    if len(pareto_front) > 1:
        pareto_times = [p[0] for p in pareto_front]
        pareto_accs = [p[1] for p in pareto_front]
        ax3.plot(pareto_times, pareto_accs, 'g--', linewidth=2, alpha=0.7, label='Pareto Front')
        ax3.legend()
    
    plt.tight_layout()
    return fig

def create_summary_table():
    """結果サマリーテーブル"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # テーブルデータ作成
    headers = ['Dataset', 'Model', 'Accuracy (%)', 'Std (%)', 'Time (min)', 'Kappa', 'Speedup', 'Acc Drop']
    table_data = []
    
    for ds in results.keys():
        base_time = results[ds]["Base"]["time"]
        base_acc = results[ds]["Base"]["acc"]
        
        for model in ['Base', 'Hybrid Static', 'Hybrid Dynamic']:
            r = results[ds][model]
            speedup = f"{base_time / r['time']:.1f}x" if model != 'Base' else '-'
            acc_drop = f"{base_acc - r['acc']:.2f}" if model != 'Base' else '-'
            table_data.append([
                ds if model == 'Base' else '',
                model,
                f"{r['acc']:.2f}",
                f"±{r['std']:.2f}",
                f"{r['time']:.1f}",
                f"{r['kappa']:.3f}",
                speedup,
                acc_drop
            ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # ヘッダーの色付け
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 行の色付け
    colors_row = ['#D6EAF8', '#FADBD8', '#D5F5E3']
    for row_idx in range(1, len(table_data) + 1):
        color_idx = (row_idx - 1) // 3
        for col_idx in range(len(headers)):
            table[(row_idx, col_idx)].set_facecolor(colors_row[color_idx % 3])
    
    ax.set_title('Comprehensive Results Summary', fontsize=16, fontweight='bold', pad=20)
    
    return fig

# メイン実行
if __name__ == "__main__":
    output_dir = Path("/workspace-cloud/seiya.narukawa/intentflow/scripts/")
    
    print("Creating comprehensive comparison figure...")
    fig1 = create_comprehensive_figure()
    fig1.savefig(output_dir / "all_datasets_comparison.png", dpi=150, bbox_inches='tight')
    fig1.savefig(output_dir / "all_datasets_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'all_datasets_comparison.png'}")
    
    print("Creating efficiency analysis figure...")
    fig2 = create_efficiency_analysis()
    fig2.savefig(output_dir / "efficiency_analysis.png", dpi=150, bbox_inches='tight')
    fig2.savefig(output_dir / "efficiency_analysis.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir / 'efficiency_analysis.png'}")
    
    print("Creating summary table...")
    fig3 = create_summary_table()
    fig3.savefig(output_dir / "results_summary_table.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'results_summary_table.png'}")
    
    print("\nAll figures generated successfully!")
    plt.show()

