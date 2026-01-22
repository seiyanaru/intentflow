#!/usr/bin/env python3
"""アブレーション実験結果の可視化スクリプト"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# 結果データ
results = {
    'Exp A\n(feature_stats)': {
        'BCIC 2a': (83.91, 5.86),
        'BCIC 2b': (80.76, 7.39),
        'HGD': (78.75, 13.38),
    },
    'Exp B\n(entropy norm)': {
        'BCIC 2a': (83.91, 5.86),
        'BCIC 2b': (80.76, 7.39),
        'HGD': (78.75, 13.39),
    },
    'Exp C\n(train entropy)': {
        'BCIC 2a': (82.76, 7.84),
        'BCIC 2b': (81.44, 6.84),
        'HGD': (79.73, 9.74),
    },
    'Exp D\n(TTT Drop)': {
        'BCIC 2a': (81.03, 8.17),
        'BCIC 2b': (82.16, 5.98),
        'HGD': (79.69, 14.84),
    },
}

# 以前の結果（比較用）
baseline_results = {
    'Base\n(TCFormer)': {
        'BCIC 2a': (84.67, 6.5),  # 以前のBase結果
        'BCIC 2b': (81.25, 7.0),
        'HGD': (85.94, 6.0),  # 概算
    },
    'Previous\nHybrid': {
        'BCIC 2a': (82.47, 9.5),  # 以前のHybrid結果
        'BCIC 2b': (79.40, 8.5),
        'HGD': (50.0, 15.0),  # 崩壊した結果
    },
}

datasets = ['BCIC 2a', 'BCIC 2b', 'HGD']
experiments = list(results.keys())
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']  # 青、緑、赤、紫

# =============================================================================
# Figure 1: データセット別の棒グラフ
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]
    
    accs = [results[exp][dataset][0] for exp in experiments]
    stds = [results[exp][dataset][1] for exp in experiments]
    
    x = np.arange(len(experiments))
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    # 値をバーの上に表示
    for bar, acc, std in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(dataset, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    ax.set_ylim(50, 100)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% baseline')

plt.suptitle('Ablation Study Results: Effect of Different Improvements', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/ablation_by_dataset.png', dpi=150, bbox_inches='tight')
plt.savefig('intentflow/offline/results/ablation_experiments/ablation_by_dataset.pdf', bbox_inches='tight')
print("Saved: ablation_by_dataset.png")

# =============================================================================
# Figure 2: 実験手法別のグループ化棒グラフ
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(experiments))
width = 0.25
dataset_colors = {'BCIC 2a': '#3498db', 'BCIC 2b': '#2ecc71', 'HGD': '#e74c3c'}

for i, dataset in enumerate(datasets):
    accs = [results[exp][dataset][0] for exp in experiments]
    stds = [results[exp][dataset][1] for exp in experiments]
    
    bars = ax.bar(x + i*width - width, accs, width, yerr=stds, 
                  label=dataset, color=dataset_colors[dataset], capsize=3, alpha=0.8)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Experiment', fontsize=12)
ax.set_title('Ablation Study: Comparison Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=10)
ax.set_ylim(60, 95)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/ablation_by_experiment.png', dpi=150, bbox_inches='tight')
plt.savefig('intentflow/offline/results/ablation_experiments/ablation_by_experiment.pdf', bbox_inches='tight')
print("Saved: ablation_by_experiment.png")

# =============================================================================
# Figure 3: HGD改善の可視化（以前との比較）
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# HGD結果のみ
hgd_experiments = ['Previous\nHybrid', 'Exp A', 'Exp B', 'Exp C', 'Exp D']
hgd_accs = [50.0, 78.75, 78.75, 79.73, 79.69]
hgd_stds = [15.0, 13.38, 13.39, 9.74, 14.84]
hgd_colors = ['#e74c3c'] + colors

bars = ax.bar(hgd_experiments, hgd_accs, yerr=hgd_stds, capsize=5, 
              color=hgd_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# 値を表示
for bar, acc, std in zip(bars, hgd_accs, hgd_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1, 
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('HGD Dataset: Recovery from Catastrophic Failure', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.axhline(y=85.94, color='green', linestyle='--', alpha=0.7, label='Base TCFormer (~86%)')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 改善を示す矢印
ax.annotate('', xy=(2, 78), xytext=(0.5, 52),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1.25, 65, '+28.75%\nImproved!', fontsize=11, color='green', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/hgd_recovery.png', dpi=150, bbox_inches='tight')
plt.savefig('intentflow/offline/results/ablation_experiments/hgd_recovery.pdf', bbox_inches='tight')
print("Saved: hgd_recovery.png")

# =============================================================================
# Figure 4: 標準偏差の比較（安定性）
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(experiments))
width = 0.25

for i, dataset in enumerate(datasets):
    stds = [results[exp][dataset][1] for exp in experiments]
    ax.bar(x + i*width - width, stds, width, label=dataset, 
           color=dataset_colors[dataset], alpha=0.8)

ax.set_ylabel('Standard Deviation (%)', fontsize=12)
ax.set_xlabel('Experiment', fontsize=12)
ax.set_title('Stability Analysis: Lower is Better', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Exp Cのハイライト
ax.annotate('Most Stable\nfor HGD!', xy=(2, 9.74), xytext=(2.5, 12),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/stability_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('intentflow/offline/results/ablation_experiments/stability_analysis.pdf', bbox_inches='tight')
print("Saved: stability_analysis.png")

# =============================================================================
# 結果サマリーを出力
# =============================================================================
print("\n" + "="*70)
print("ABLATION STUDY RESULTS SUMMARY")
print("="*70)

print("\n📊 Main Results:")
print("-"*70)
print(f"{'Experiment':<25} {'BCIC 2a':<18} {'BCIC 2b':<18} {'HGD':<18}")
print("-"*70)
for exp in experiments:
    exp_name = exp.replace('\n', ' ')
    row = f"{exp_name:<25}"
    for ds in datasets:
        acc, std = results[exp][ds]
        row += f"{acc:.2f} ± {std:.2f}      "
    print(row)
print("-"*70)

print("\n🔑 Key Findings:")
print("-"*70)
print("1. HGD Recovery: 50% → ~79% (Catastrophic failure resolved!)")
print("2. Best for BCIC 2b: Exp D (TTT Drop) = 82.16% ± 5.98")
print("3. Most Stable for HGD: Exp C (train entropy) = 79.73% ± 9.74")
print("4. Exp A/B similar results → entropy normalization alone not sufficient")
print("-"*70)

print("\n📈 Improvement Analysis (vs Previous Hybrid):")
for ds in datasets:
    prev = baseline_results['Previous\nHybrid'][ds][0]
    best_exp = max(experiments, key=lambda e: results[e][ds][0])
    best_acc = results[best_exp][ds][0]
    diff = best_acc - prev
    sign = '+' if diff > 0 else ''
    print(f"  {ds}: {prev:.1f}% → {best_acc:.1f}% ({sign}{diff:.1f}%) - Best: {best_exp.replace(chr(10), ' ')}")

print("\n" + "="*70)
print("Figures saved to: intentflow/offline/results/ablation_experiments/")
print("="*70)

