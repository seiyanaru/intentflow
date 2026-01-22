#!/usr/bin/env python3
"""アブレーション実験の被験者ごとの精度を可視化"""

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 10

# =============================================================================
# 実験結果データ（results.txtから抽出）
# =============================================================================

# Exp A: feature_stats gating (entropy OFF)
exp_a = {
    'bcic2a': {1: 79.31, 2: 75.86, 3: 89.66, 4: 75.86, 5: 84.48, 6: 84.48, 7: 94.83, 8: 86.21, 9: 84.48},
    'bcic2b': {1: 77.50, 2: 80.00, 3: 75.00, 4: 100.00, 5: 80.95, 6: 82.50, 7: 80.00, 8: 78.41, 9: 72.50},
    'hgd': {1: 79.37, 2: 66.25, 3: 89.38, 4: 85.62, 5: 100.00, 6: 77.50, 7: 66.87, 8: 73.75, 
            9: 93.75, 10: 75.63, 11: 75.63, 12: 96.88, 13: 74.37, 14: 47.50},
}

# Exp B: Normalized entropy + threshold=0.7
exp_b = {
    'bcic2a': {1: 79.31, 2: 75.86, 3: 89.66, 4: 75.86, 5: 84.48, 6: 84.48, 7: 94.83, 8: 86.21, 9: 84.48},
    'bcic2b': {1: 77.50, 2: 80.00, 3: 75.00, 4: 100.00, 5: 80.95, 6: 82.50, 7: 80.00, 8: 78.41, 9: 72.50},
    'hgd': {1: 79.37, 2: 66.25, 3: 90.00, 4: 85.00, 5: 100.00, 6: 77.50, 7: 66.87, 8: 73.75, 
            9: 93.75, 10: 75.63, 11: 75.63, 12: 96.88, 13: 74.37, 14: 47.50},
}

# Exp C: entropy_gating_in_train=True + threshold=0.7
exp_c = {
    'bcic2a': {1: 84.48, 2: 72.41, 3: 89.66, 4: 70.69, 5: 77.59, 6: 87.93, 7: 96.55, 8: 81.03, 9: 84.48},
    'bcic2b': {1: 82.50, 2: 81.25, 3: 72.50, 4: 98.81, 5: 79.76, 6: 78.75, 7: 82.50, 8: 80.68, 9: 76.25},
    'hgd': {1: 75.63, 2: 78.12, 3: 93.12, 4: 88.13, 5: 99.37, 6: 76.88, 7: 63.13, 8: 74.37,
            9: 90.00, 10: 76.88, 11: 71.88, 12: 85.62, 13: 74.37, 14: 68.75},
}

# Exp D: TTT Drop (0.2) + entropy_gating_in_train=True
exp_d = {
    'bcic2a': {1: 81.03, 2: 68.97, 3: 82.76, 4: 75.86, 5: 68.97, 6: 81.03, 7: 93.10, 8: 91.38, 9: 86.21},
    'bcic2b': {1: 83.75, 2: 81.25, 3: 77.50, 4: 97.62, 5: 80.95, 6: 83.75, 7: 76.25, 8: 78.41, 9: 80.00},
    'hgd': {1: 72.50, 2: 55.00, 3: 90.62, 4: 79.37, 5: 97.50, 6: 76.88, 7: 58.75, 8: 75.00,
            9: 98.12, 10: 80.00, 11: 97.50, 12: 98.75, 13: 78.12, 14: 57.50},
}

# Base結果（比較用）
base_results = {
    'bcic2a': {1: 82.76, 2: 79.31, 3: 91.38, 4: 74.14, 5: 82.76, 6: 87.93, 7: 94.83, 8: 91.38, 9: 77.59},  # 84.67% avg
    'bcic2b': {1: 82.50, 2: 85.00, 3: 72.50, 4: 100.00, 5: 83.33, 6: 85.00, 7: 82.50, 8: 78.41, 9: 75.00},  # 82.67% avg
    'hgd': {1: 90.00, 2: 83.75, 3: 97.50, 4: 100.00, 5: 100.00, 6: 95.63, 7: 86.87, 8: 96.25,
            9: 100.00, 10: 95.00, 11: 92.50, 12: 98.75, 13: 90.00, 14: 75.00},  # 92.95% avg
}

experiments = {
    'Exp A: feature_stats gating\n(entropy OFF)': exp_a,
    'Exp B: Normalized entropy\n(threshold=0.7)': exp_b,
    'Exp C: entropy_gating_in_train\n(threshold=0.7)': exp_c,
    'Exp D: TTT Drop (0.2)\n+ train entropy': exp_d,
}

dataset_colors = {
    'bcic2a': '#3498db',  # 青
    'bcic2b': '#2ecc71',  # 緑
    'hgd': '#e74c3c',     # 赤
}

dataset_labels = {
    'bcic2a': 'BCIC IV 2a (9 subjects, 4 classes)',
    'bcic2b': 'BCIC IV 2b (9 subjects, 2 classes)',
    'hgd': 'HGD (14 subjects, 2 classes)',
}

output_dir = 'intentflow/offline/results/ablation_experiments/per_subject_figures'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 各実験ごとに図を作成
# =============================================================================
for exp_idx, (exp_name, exp_data) in enumerate(experiments.items()):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(exp_name, fontsize=16, fontweight='bold', y=0.98)
    
    for ds_idx, (dataset, color) in enumerate(dataset_colors.items()):
        ax = axes[ds_idx]
        
        subjects = sorted(exp_data[dataset].keys())
        x = np.arange(len(subjects))
        
        # 実験結果
        exp_accs = [exp_data[dataset][s] for s in subjects]
        # Base結果
        base_accs = [base_results[dataset][s] for s in subjects]
        
        width = 0.35
        
        # Baseの棒グラフ
        bars_base = ax.bar(x - width/2, base_accs, width, label='Base (TCFormer)', 
                          color='#95a5a6', alpha=0.7, edgecolor='black')
        
        # 実験結果の棒グラフ
        # 色分け: Baseより良い=緑、同等=青、悪い=赤
        bar_colors = []
        for i, (exp_acc, base_acc) in enumerate(zip(exp_accs, base_accs)):
            diff = exp_acc - base_acc
            if diff > 2:
                bar_colors.append('#27ae60')  # 緑（改善）
            elif diff < -5:
                bar_colors.append('#e74c3c')  # 赤（悪化）
            else:
                bar_colors.append(color)  # 元の色（同等）
        
        bars_exp = ax.bar(x + width/2, exp_accs, width, label='Ablation Exp', 
                         color=bar_colors, alpha=0.85, edgecolor='black')
        
        # 精度差を表示
        for i, (exp_acc, base_acc) in enumerate(zip(exp_accs, base_accs)):
            diff = exp_acc - base_acc
            color_text = 'green' if diff > 0 else 'red' if diff < 0 else 'gray'
            if abs(diff) >= 3:
                ax.text(x[i] + width/2, exp_acc + 1, f'{diff:+.1f}', 
                       ha='center', va='bottom', fontsize=8, color=color_text, fontweight='bold')
        
        # 平均精度を計算
        avg_exp = np.mean(exp_accs)
        avg_base = np.mean(base_accs)
        
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{dataset_labels[dataset]}\nAvg: Base={avg_base:.1f}%, Exp={avg_exp:.1f}% (Δ={avg_exp-avg_base:+.1f}%)', 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=10)
        ax.set_ylim(40, 105)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
        
        # 50%ラインのラベル
        if dataset == 'hgd':
            ax.text(len(subjects)-0.5, 52, 'Random', fontsize=8, color='gray', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # ファイル名を作成（スペースと改行を置換）
    filename = f"exp_{chr(ord('a')+exp_idx)}_per_subject.png"
    filepath = f'{output_dir}/{filename}'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.savefig(filepath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

# =============================================================================
# 全実験の比較図（HGDのみ、詳細版）
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 8))

subjects_hgd = list(range(1, 15))
x = np.arange(len(subjects_hgd))
width = 0.15

# Base
base_accs_hgd = [base_results['hgd'][s] for s in subjects_hgd]
ax.bar(x - 2*width, base_accs_hgd, width, label='Base (TCFormer)', color='#95a5a6', alpha=0.8, edgecolor='black')

# 各実験
colors_exp = ['#3498db', '#9b59b6', '#f39c12', '#1abc9c']
for i, (exp_name, exp_data) in enumerate(experiments.items()):
    exp_accs = [exp_data['hgd'][s] for s in subjects_hgd]
    label = exp_name.split(':')[0]  # "Exp A" etc.
    ax.bar(x + (i-1)*width, exp_accs, width, label=label, color=colors_exp[i], alpha=0.85, edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Subject ID', fontsize=12)
ax.set_title('HGD Dataset: Comparison of All Ablation Experiments per Subject', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'S{s}' for s in subjects_hgd], fontsize=10)
ax.set_ylim(40, 105)
ax.legend(loc='lower right', fontsize=9, ncol=3)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{output_dir}/hgd_all_experiments_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{output_dir}/hgd_all_experiments_comparison.pdf', bbox_inches='tight')
print(f"Saved: {output_dir}/hgd_all_experiments_comparison.png")
plt.close()

# =============================================================================
# サマリーテーブル出力
# =============================================================================
print("\n" + "=" * 80)
print("ABLATION EXPERIMENTS - PER SUBJECT ACCURACY SUMMARY")
print("=" * 80)

for exp_name, exp_data in experiments.items():
    print(f"\n{exp_name}")
    print("-" * 60)
    for dataset in ['bcic2a', 'bcic2b', 'hgd']:
        subjects = sorted(exp_data[dataset].keys())
        accs = [exp_data[dataset][s] for s in subjects]
        base_accs = [base_results[dataset][s] for s in subjects]
        avg_exp = np.mean(accs)
        avg_base = np.mean(base_accs)
        print(f"  {dataset}: {avg_exp:.2f}% ± {np.std(accs):.2f} (Base: {avg_base:.2f}%, Δ={avg_exp-avg_base:+.2f}%)")

print("\n" + "=" * 80)
print(f"Figures saved to: {output_dir}/")
print("=" * 80)

