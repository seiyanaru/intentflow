#!/usr/bin/env python3
"""
Plot comparison between buggy baseline and clean baseline
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
subjects = list(range(1, 10))
old_acc = [85.42, 54.51, 93.75, 87.50, 75.00, 68.75, 93.75, 85.42, 84.72]
new_acc = [87.50, 71.88, 92.01, 81.25, 76.74, 69.44, 92.71, 84.72, 88.89]

old_mean = 80.98
new_mean = 82.79
old_std = 12.05
new_std = 8.05
paper_target = 84.79

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TCFormer Baseline Comparison: Before vs After Bug Fixes', fontsize=16, fontweight='bold')

# Plot 1: Subject-wise accuracy
ax1 = axes[0, 0]
x = np.arange(len(subjects))
width = 0.35

bars1 = ax1.bar(x - width/2, old_acc, width, label='Old (with bugs)', alpha=0.8, color='#e74c3c')
bars2 = ax1.bar(x + width/2, new_acc, width, label='New (clean)', alpha=0.8, color='#3498db')

ax1.axhline(y=paper_target, color='green', linestyle='--', linewidth=2, label='Paper target (84.79%)')
ax1.set_xlabel('Subject ID', fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontweight='bold')
ax1.set_title('Subject-wise Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(subjects)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Improvement delta
ax2 = axes[0, 1]
delta = [new - old for new, old in zip(new_acc, old_acc)]
colors = ['green' if d > 0 else 'red' for d in delta]
bars = ax2.bar(subjects, delta, color=colors, alpha=0.7)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Subject ID', fontweight='bold')
ax2.set_ylabel('Δ Accuracy (%)', fontweight='bold')
ax2.set_title('Improvement per Subject (Positive = Better)')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, delta)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:+.1f}%',
             ha='center', va='bottom' if val > 0 else 'top', fontsize=8)

# Plot 3: Distribution comparison
ax3 = axes[1, 0]
data_to_plot = [old_acc, new_acc]
bp = ax3.boxplot(data_to_plot, labels=['Old (bugs)', 'New (clean)'],
                  patch_artist=True, widths=0.6)

bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#3498db')

ax3.axhline(y=paper_target, color='green', linestyle='--', linewidth=2, label='Paper target')
ax3.set_ylabel('Test Accuracy (%)', fontweight='bold')
ax3.set_title('Distribution Comparison (Stability Improvement)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add mean and std annotations
ax3.text(1, old_mean, f'μ={old_mean:.2f}%\nσ={old_std:.2f}%',
         ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax3.text(2, new_mean, f'μ={new_mean:.2f}%\nσ={new_std:.2f}%',
         ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 4: Summary statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
SUMMARY STATISTICS
{'='*50}

Average Accuracy:
  Old (with bugs):  {old_mean:.2f}% ± {old_std:.2f}%
  New (clean):      {new_mean:.2f}% ± {new_std:.2f}%
  Δ Improvement:    +{new_mean - old_mean:.2f}%

Standard Deviation:
  Old:  {old_std:.2f}%
  New:  {new_std:.2f}%
  Δ Reduction:  -{old_std - new_std:.2f}% (-{(old_std - new_std) / old_std * 100:.1f}%)

Gap to Paper (84.79%):
  Old:  -{paper_target - old_mean:.2f}%
  New:  -{paper_target - new_mean:.2f}%
  Δ Gap Closure:  +{(paper_target - old_mean) - (paper_target - new_mean):.2f}%

Subject Results:
  Improved:   5/9 subjects
  Degraded:   4/9 subjects
  Best Δ:     +17.37% (Subject 2)
  Worst Δ:    -6.25% (Subject 4)

KEY INSIGHTS:
✓ Stability greatly improved (SD -33.2%)
✓ Low-performers boosted significantly
✓ High-performers slightly decreased
⚠ Still 2.00% from paper target
→ Next: Window timing adjustment (0.5-4.5s)
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/comparison_before_after.png',
            dpi=300, bbox_inches='tight')
print("Plot saved to: results/comparison_before_after.png")
plt.show()
