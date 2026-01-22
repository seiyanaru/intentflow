#!/usr/bin/env python3
"""
3モデル比較グラフ生成スクリプト
Base vs Hybrid (静的α) vs Hybrid + Entropy Gating
"""

import matplotlib.pyplot as plt
import numpy as np

# データ設定
subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

# 各モデルの被験者別精度 (%)
base_acc = [91.38, 63.79, 89.66, 84.48, 74.14, 86.21, 94.83, 86.21, 91.38]
hybrid_static_acc = [77.59, 74.14, 86.21, 81.03, 74.14, 84.48, 93.10, 89.66, 82.76]
hybrid_entropy_acc = [79.31, 75.86, 89.66, 75.86, 84.48, 84.48, 94.83, 86.21, 84.48]

# 平均と標準偏差
avg_base = np.mean(base_acc)
avg_hybrid_static = np.mean(hybrid_static_acc)
avg_hybrid_entropy = np.mean(hybrid_entropy_acc)

std_base = np.std(base_acc)
std_hybrid_static = np.std(hybrid_static_acc)
std_hybrid_entropy = np.std(hybrid_entropy_acc)

print(f"Base: {avg_base:.2f}% ± {std_base:.2f}")
print(f"Hybrid (Static): {avg_hybrid_static:.2f}% ± {std_hybrid_static:.2f}")
print(f"Hybrid + Entropy: {avg_hybrid_entropy:.2f}% ± {std_hybrid_entropy:.2f}")

# グラフ設定
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(subjects))
width = 0.25

# バーの描画
bars1 = ax.bar(x - width, base_acc, width, label='Base', color='#7ECFEF', edgecolor='white', linewidth=0.7)
bars2 = ax.bar(x, hybrid_static_acc, width, label='Hybrid (Static)', color='#FFB347', edgecolor='white', linewidth=0.7)
bars3 = ax.bar(x + width, hybrid_entropy_acc, width, label='Hybrid + Entropy', color='#FF6B6B', edgecolor='white', linewidth=0.7)

# 軸設定
ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Comparison per Subject', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(subjects, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))

# 凡例
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# グリッド
ax.yaxis.grid(True, linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

# レイアウト調整
plt.tight_layout()

# 保存
output_path = '/workspace-cloud/seiya.narukawa/intentflow/scripts/model_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved to: {output_path}")

# 追加: 平均精度のサマリーグラフも作成
fig2, ax2 = plt.subplots(figsize=(8, 5))

models = ['Base\n(TCFormer)', 'Hybrid\n(Static)', 'Hybrid +\nEntropy Gating']
averages = [avg_base, avg_hybrid_static, avg_hybrid_entropy]
stds = [std_base, std_hybrid_static, std_hybrid_entropy]
colors = ['#7ECFEF', '#FFB347', '#FF6B6B']

bars = ax2.bar(models, averages, yerr=stds, capsize=5, color=colors, edgecolor='white', linewidth=1.5)

# 値のラベル
for bar, avg, std in zip(bars, averages, stds):
    height = bar.get_height()
    ax2.annotate(f'{avg:.1f}%\n±{std:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylim(0, 100)
ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
ax2.set_axisbelow(True)

plt.tight_layout()

output_path2 = '/workspace-cloud/seiya.narukawa/intentflow/scripts/model_comparison_summary.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to: {output_path2}")

plt.show()

