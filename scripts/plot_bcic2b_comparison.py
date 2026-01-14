#!/usr/bin/env python3
"""
BCI Competition IV 2b - 3モデル比較可視化
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 結果ディレクトリ
RESULTS_BASE = Path("intentflow/offline/results/paper_experiments/bcic2b/20260113_110603")

# モデル名と表示名
MODELS = {
    "base": "Base (TCFormer)",
    "hybrid_static": "Hybrid (Static α)",
    "hybrid_entropy": "Hybrid (Dynamic α)"
}

# 被験者ID
SUBJECTS = list(range(1, 10))

def load_results():
    """各モデルの結果を読み込む"""
    results = {}
    
    for model_key, model_name in MODELS.items():
        model_dir = RESULTS_BASE / model_key
        acc_file = list(model_dir.glob("final_acc_*.json"))[0]
        
        with open(acc_file) as f:
            data = json.load(f)
        
        # 被験者ごとの精度を取得
        accuracies = []
        for s in SUBJECTS:
            key = f"Subject_{s}"
            if key in data:
                accuracies.append(data[key] * 100)  # パーセントに変換
        
        results[model_key] = {
            "name": model_name,
            "accuracies": np.array(accuracies),
            "mean": np.mean(accuracies),
            "std": np.std(accuracies)
        }
    
    return results

def plot_comparison(results):
    """比較グラフを作成"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 色の設定
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # --- 左: 被験者ごとの精度 ---
    ax1 = axes[0]
    x = np.arange(len(SUBJECTS))
    width = 0.25
    
    for i, (model_key, data) in enumerate(results.items()):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, data["accuracies"], width, 
                       label=data["name"], color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('Subject', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('BCI Competition IV 2b - Per-Subject Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{s}' for s in SUBJECTS])
    ax1.set_ylim(60, 105)
    ax1.legend(loc='upper right')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.grid(axis='y', alpha=0.3)
    
    # --- 右: 平均精度の比較 ---
    ax2 = axes[1]
    model_names = [results[k]["name"] for k in MODELS.keys()]
    means = [results[k]["mean"] for k in MODELS.keys()]
    stds = [results[k]["std"] for k in MODELS.keys()]
    
    bars = ax2.bar(model_names, means, yerr=stds, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 値をバーの上に表示
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Average Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(60, 100)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = RESULTS_BASE / "comparison_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # PDF版も保存
    plt.savefig(RESULTS_BASE / "comparison_plot.pdf", bbox_inches='tight', facecolor='white')
    print(f"Saved: {RESULTS_BASE / 'comparison_plot.pdf'}")
    
    plt.show()

def print_summary(results):
    """サマリーを表示"""
    print("\n" + "="*60)
    print(" BCI Competition IV 2b - Results Summary")
    print("="*60)
    print(f"{'Model':<25} {'Mean Acc':<12} {'Std':<10}")
    print("-"*60)
    
    for model_key, data in results.items():
        print(f"{data['name']:<25} {data['mean']:>8.2f}%    ±{data['std']:.2f}%")
    
    print("="*60)
    
    # 最良モデルを特定
    best_model = max(results.items(), key=lambda x: x[1]["mean"])
    print(f"\n🏆 Best Model: {best_model[1]['name']} ({best_model[1]['mean']:.2f}%)")

def plot_radar(results):
    """レーダーチャート（被験者ごとのパフォーマンス）"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2*np.pi, len(SUBJECTS), endpoint=False).tolist()
    angles += angles[:1]  # 閉じる
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (model_key, data) in enumerate(results.items()):
        values = data["accuracies"].tolist()
        values += values[:1]  # 閉じる
        ax.plot(angles, values, 'o-', linewidth=2, label=data["name"], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'S{s}' for s in SUBJECTS])
    ax.set_ylim(50, 100)
    ax.set_title('Per-Subject Accuracy Comparison\n(BCI Competition IV 2b)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    output_path = RESULTS_BASE / "radar_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    import os
    os.chdir("/workspace-cloud/seiya.narukawa/intentflow")
    
    results = load_results()
    print_summary(results)
    plot_comparison(results)
    plot_radar(results)

