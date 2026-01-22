#!/usr/bin/env python3
"""HGD精度低下の詳細分析スクリプト"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# =============================================================================
# 正確な実験結果データ
# =============================================================================

# 以前の実験結果（画像から抽出）
previous_results = {
    'BCIC 2a': {
        'Base': (84.67, 9.25),
        'Hybrid Static': (82.57, 5.89),
        'Hybrid Dynamic': (83.52, 5.52),
    },
    'BCIC 2b': {
        'Base': (82.67, 6.73),
        'Hybrid Static': (79.92, 7.98),
        'Hybrid Dynamic': (80.76, 7.39),
    },
    'HGD': {
        'Base': (92.95, 7.01),
        'Hybrid Static': (81.25, 10.06),
        'Hybrid Dynamic': (79.29, 14.61),
    },
}

# アブレーション実験結果
ablation_results = {
    'BCIC 2a': {
        'Exp A (feature_stats)': (83.91, 5.86),
        'Exp B (entropy norm)': (83.91, 5.86),
        'Exp C (train entropy)': (82.76, 7.84),
        'Exp D (TTT drop)': (81.03, 8.17),
    },
    'BCIC 2b': {
        'Exp A (feature_stats)': (80.76, 7.39),
        'Exp B (entropy norm)': (80.76, 7.39),
        'Exp C (train entropy)': (81.44, 6.84),
        'Exp D (TTT drop)': (82.16, 5.98),
    },
    'HGD': {
        'Exp A (feature_stats)': (78.75, 13.38),
        'Exp B (entropy norm)': (78.75, 13.39),
        'Exp C (train entropy)': (79.73, 9.74),
        'Exp D (TTT drop)': (79.69, 14.84),
    },
}

# HGD被験者ごとの結果
hgd_subjects = {
    'Base': {
        1: 90.0, 2: 83.8, 3: 97.5, 4: 100.0, 5: 100.0, 6: 95.6, 7: 86.9,
        8: 96.2, 9: 100.0, 10: 95.0, 11: 92.5, 12: 98.8, 13: 90.0, 14: 75.0
    },
    'Hybrid Dynamic': {
        1: 77.5, 2: 64.4, 3: 92.5, 4: 98.1, 5: 75.0, 6: 80.6, 7: 51.2,
        8: 68.1, 9: 99.4, 10: 87.5, 11: 93.1, 12: 93.1, 13: 68.1, 14: 61.3
    },
    'Exp C': {
        1: 75.6, 2: 78.1, 3: 93.1, 4: 88.1, 5: 99.4, 6: 76.9, 7: 63.1,
        8: 74.4, 9: 90.0, 10: 76.9, 11: 71.9, 12: 85.6, 13: 74.4, 14: 68.8
    },
}

# =============================================================================
# Figure 1: 全データセットのBase vs Hybrid比較
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (dataset, data) in enumerate(previous_results.items()):
    ax = axes[idx]
    models = list(data.keys())
    accs = [data[m][0] for m in models]
    stds = [data[m][1] for m in models]
    
    colors = ['#27ae60', '#3498db', '#e74c3c']  # Base=緑, Static=青, Dynamic=赤
    x = np.arange(len(models))
    bars = ax.bar(x, accs, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor='black')
    
    for bar, acc, std in zip(bars, accs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 精度低下を示す矢印
    if dataset == 'HGD':
        ax.annotate('', xy=(2, accs[2]), xytext=(0, accs[0]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.text(1, (accs[0]+accs[2])/2 + 3, f'-{accs[0]-accs[2]:.1f}%',
                fontsize=12, color='red', fontweight='bold', ha='center')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(dataset, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Base', 'Hybrid\nStatic', 'Hybrid\nDynamic'], fontsize=10)
    ax.set_ylim(50, 105)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Previous Results: Base vs Hybrid Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/previous_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: previous_comparison.png")

# =============================================================================
# Figure 2: HGD被験者ごとの比較
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

subjects = list(range(1, 15))
x = np.arange(len(subjects))
width = 0.25

base_accs = [hgd_subjects['Base'][s] for s in subjects]
hybrid_accs = [hgd_subjects['Hybrid Dynamic'][s] for s in subjects]
expc_accs = [hgd_subjects['Exp C'][s] for s in subjects]

bars1 = ax.bar(x - width, base_accs, width, label='Base (TCFormer)', color='#27ae60', alpha=0.85)
bars2 = ax.bar(x, hybrid_accs, width, label='Hybrid Dynamic', color='#e74c3c', alpha=0.85)
bars3 = ax.bar(x + width, expc_accs, width, label='Exp C (train entropy)', color='#9b59b6', alpha=0.85)

# 大幅低下した被験者をハイライト
for i, s in enumerate(subjects):
    diff = hybrid_accs[i] - base_accs[i]
    if diff < -15:
        ax.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.1)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Subject ID', fontsize=12)
ax.set_title('HGD: Per-Subject Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=10)
ax.set_ylim(40, 105)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')

# 凡例に注記
ax.text(0.02, 0.02, '🔴 Highlighted: >15% drop from Base', transform=ax.transAxes,
        fontsize=9, color='red', alpha=0.8)

plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/hgd_per_subject.png', dpi=150, bbox_inches='tight')
print("Saved: hgd_per_subject.png")

# =============================================================================
# Figure 3: 精度低下の分布
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左: Hybrid Dynamicの低下分布
ax1 = axes[0]
drops_hybrid = [base_accs[i] - hybrid_accs[i] for i in range(len(subjects))]
colors_h = ['#e74c3c' if d > 15 else '#f39c12' if d > 5 else '#27ae60' for d in drops_hybrid]
bars = ax1.bar([f'S{s}' for s in subjects], drops_hybrid, color=colors_h, alpha=0.85, edgecolor='black')
ax1.axhline(y=0, color='black', linewidth=1)
ax1.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
ax1.set_ylabel('Accuracy Drop (%)', fontsize=12)
ax1.set_title('Hybrid Dynamic: Accuracy Drop from Base', fontsize=12, fontweight='bold')
ax1.set_ylim(-5, 40)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 右: Exp Cの低下分布
ax2 = axes[1]
drops_expc = [base_accs[i] - expc_accs[i] for i in range(len(subjects))]
colors_c = ['#e74c3c' if d > 15 else '#f39c12' if d > 5 else '#27ae60' for d in drops_expc]
bars = ax2.bar([f'S{s}' for s in subjects], drops_expc, color=colors_c, alpha=0.85, edgecolor='black')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Critical threshold')
ax2.set_ylabel('Accuracy Drop (%)', fontsize=12)
ax2.set_title('Exp C: Accuracy Drop from Base', fontsize=12, fontweight='bold')
ax2.set_ylim(-5, 40)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('HGD: Distribution of Accuracy Drops', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/hgd_drop_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: hgd_drop_distribution.png")

# =============================================================================
# Figure 4: アブレーション実験サマリー
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

datasets = ['BCIC 2a', 'BCIC 2b', 'HGD']
experiments = list(ablation_results['BCIC 2a'].keys())
x = np.arange(len(experiments))
width = 0.25
ds_colors = {'BCIC 2a': '#3498db', 'BCIC 2b': '#2ecc71', 'HGD': '#e74c3c'}

for i, ds in enumerate(datasets):
    accs = [ablation_results[ds][exp][0] for exp in experiments]
    stds = [ablation_results[ds][exp][1] for exp in experiments]
    ax.bar(x + i*width - width, accs, width, yerr=stds, label=ds, 
           color=ds_colors[ds], capsize=3, alpha=0.85)

# Base結果を水平線で表示
ax.axhline(y=84.67, color='#3498db', linestyle='--', alpha=0.5, linewidth=2)
ax.axhline(y=82.67, color='#2ecc71', linestyle='--', alpha=0.5, linewidth=2)
ax.axhline(y=92.95, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=2)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Experiment', fontsize=12)
ax.set_title('Ablation Study Results (Dashed lines = Base accuracy)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([e.replace(' (', '\n(') for e in experiments], fontsize=9)
ax.set_ylim(70, 100)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('intentflow/offline/results/ablation_experiments/ablation_summary.png', dpi=150, bbox_inches='tight')
print("Saved: ablation_summary.png")

# =============================================================================
# 分析サマリー出力
# =============================================================================
print("\n" + "=" * 80)
print("HGD精度低下の詳細分析")
print("=" * 80)

print("\n📊 【数値サマリー】")
print("-" * 80)
print(f"{'Model':<25} {'Accuracy':<15} {'Std':<10} {'Δ from Base':<15}")
print("-" * 80)
print(f"{'Base (TCFormer)':<25} {'92.95%':<15} {'±7.01':<10} {'-':<15}")
print(f"{'Hybrid Static':<25} {'81.25%':<15} {'±10.06':<10} {'-11.70%':<15}")
print(f"{'Hybrid Dynamic':<25} {'79.29%':<15} {'±14.61':<10} {'-13.66%':<15}")
print("-" * 80)
print(f"{'Exp A (feature_stats)':<25} {'78.75%':<15} {'±13.38':<10} {'-14.20%':<15}")
print(f"{'Exp B (entropy norm)':<25} {'78.75%':<15} {'±13.39':<10} {'-14.20%':<15}")
print(f"{'Exp C (train entropy)':<25} {'79.73%':<15} {'±9.74':<10} {'-13.22%':<15}")
print(f"{'Exp D (TTT drop)':<25} {'79.69%':<15} {'±14.84':<10} {'-13.26%':<15}")
print("-" * 80)

print("\n⚠️ 【問題の核心】")
print("-" * 80)
print("1. HGDでは全てのHybridモデルがBaseより約13-14%低下")
print("2. 特定被験者（S7: -35.6%, S8: -28.1%, S5: -25.0%）で壊滅的低下")
print("3. アブレーション実験でも改善なし → 根本的な問題が存在")
print("4. 標準偏差が大きい（14%超）→ 被験者間変動が激しい")

print("\n🔍 【原因分析】")
print("-" * 80)
print("""
1. **データセット特性の問題**
   - HGDは14人の被験者、2クラス（左手/右手）
   - BaseのAcc: 75%〜100%と被験者間変動が大きい
   - 元々難しい被験者（S14: 75%）はさらに悪化

2. **TTT適応の過剰適応**
   - テスト時のTTT更新が、特定の被験者でノイズを増幅
   - 特に高精度被験者（S5: 100%→75%, S8: 96%→68%）で顕著
   - 「既に良い」被験者に対してTTTが悪影響を与えている

3. **訓練-テスト時の動作乖離**
   - 訓練時: feature_stats gating（固定的）
   - テスト時: entropy gating + 2-pass（動的）
   - この不一致がHGDで特に問題になる可能性

4. **2クラス vs 4クラスの違い**
   - BCIC 2a/2b: 4クラス（エントロピー範囲が広い）
   - HGD: 2クラス（エントロピーの分散が小さい）
   - エントロピーベースのゲーティングが2クラスで効果的に機能しない

5. **モデル容量と過学習**
   - HGDはサンプル数が多い → 過学習しやすい
   - TTT層の追加パラメータが過学習を助長
""")

print("\n💡 【推奨される次のステップ】")
print("-" * 80)
print("""
1. **TTTをデータセット別に調整**
   - HGDではTTT学習率を大幅に下げる（0.1 → 0.01）
   - または、HGDではTTTを完全に無効化

2. **被験者適応戦略の見直し**
   - 高精度被験者ではTTTを抑制（α=0に近づける）
   - 低精度被験者でのみTTTを積極活用

3. **エントロピー閾値の再調整**
   - 2クラスデータセット用の低い閾値を設定
   - または、正規化エントロピーの使用を強制

4. **Baseモデルの強化**
   - HGDでは、そもそもBaseが92.95%と高い
   - Hybridよりも、Baseの改善に注力する方が有効かもしれない
""")

print("\n" + "=" * 80)
print("Figures saved to: intentflow/offline/results/ablation_experiments/")
print("=" * 80)

