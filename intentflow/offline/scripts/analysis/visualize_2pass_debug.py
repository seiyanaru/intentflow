#!/usr/bin/env python3
"""
2-Pass Debug結果の可視化

生成する図:
1. エントロピー分布（before/after）＋「正誤で色分け」
2. delta_KL/delta_logits のヒストグラム（4群比較）
3. Confusion matrix の差分（after - before）
4. Flip解析サマリ図
5. 被験者間比較表
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """Load 2-pass debug results."""
    results_dir = Path(results_dir)
    
    with open(results_dir / "detailed_2pass_debug.json") as f:
        all_results = json.load(f)
    
    with open(results_dir / "summary_table.json") as f:
        summary_table = json.load(f)
    
    return all_results, summary_table


def plot_entropy_distribution_by_correctness(sample_data, subject_id, output_dir):
    """
    エントロピー分布（before/after）を正誤で色分け
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Categorize samples
    groups = {
        "stayed_correct": [s for s in sample_data if s["flip_status"] == "stayed_correct"],
        "stayed_wrong": [s for s in sample_data if s["flip_status"] == "stayed_wrong"],
        "flip_to_correct": [s for s in sample_data if s["flip_status"] == "flip_to_correct"],
        "flip_to_wrong": [s for s in sample_data if s["flip_status"] == "flip_to_wrong"],
    }
    
    colors = {
        "stayed_correct": "#2E86AB",   # blue
        "stayed_wrong": "#A23B72",     # magenta
        "flip_to_correct": "#2CA58D",  # green
        "flip_to_wrong": "#F18F01",    # orange
    }
    
    # Plot 1: Entropy BEFORE (histogram by group)
    ax = axes[0, 0]
    for group_name, group_samples in groups.items():
        if group_samples:
            entropies = [s["entropy_before"] for s in group_samples]
            ax.hist(entropies, bins=30, alpha=0.6, label=f"{group_name} (n={len(group_samples)})", 
                   color=colors[group_name], density=True)
    ax.set_xlabel("Entropy (before TTT)")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id}: Entropy BEFORE by Flip Status")
    ax.legend(fontsize=8)
    ax.axvline(np.log(4), color='gray', linestyle='--', alpha=0.5, label='max entropy (ln4)')
    
    # Plot 2: Entropy AFTER (histogram by group)
    ax = axes[0, 1]
    for group_name, group_samples in groups.items():
        if group_samples:
            entropies = [s["entropy_after"] for s in group_samples]
            ax.hist(entropies, bins=30, alpha=0.6, label=f"{group_name} (n={len(group_samples)})", 
                   color=colors[group_name], density=True)
    ax.set_xlabel("Entropy (after TTT)")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id}: Entropy AFTER by Flip Status")
    ax.legend(fontsize=8)
    ax.axvline(np.log(4), color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Delta Entropy (before → after)
    ax = axes[1, 0]
    for group_name, group_samples in groups.items():
        if group_samples:
            deltas = [s["delta_entropy"] for s in group_samples]
            ax.hist(deltas, bins=30, alpha=0.6, label=f"{group_name}", 
                   color=colors[group_name], density=True)
    ax.set_xlabel("Δ Entropy (after - before)")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id}: Entropy Change")
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)
    
    # Plot 4: Scatter (entropy_before vs delta_entropy), colored by flip status
    ax = axes[1, 1]
    for group_name, group_samples in groups.items():
        if group_samples:
            e_before = [s["entropy_before"] for s in group_samples]
            delta_e = [s["delta_entropy"] for s in group_samples]
            ax.scatter(e_before, delta_e, alpha=0.5, s=15, label=group_name, color=colors[group_name])
    ax.set_xlabel("Entropy (before)")
    ax.set_ylabel("Δ Entropy")
    ax.set_title(f"Subject {subject_id}: Entropy Before vs Change")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"entropy_distribution_s{subject_id}.png", dpi=150)
    plt.close()


def plot_delta_distribution_4groups(sample_data, subject_id, output_dir):
    """
    delta_KL / delta_logits のヒストグラム（4群比較）
    - stayed_correct, stayed_wrong, flip_to_correct, flip_to_wrong
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    groups = {
        "stayed_correct": [s for s in sample_data if s["flip_status"] == "stayed_correct"],
        "stayed_wrong": [s for s in sample_data if s["flip_status"] == "stayed_wrong"],
        "flip_to_correct": [s for s in sample_data if s["flip_status"] == "flip_to_correct"],
        "flip_to_wrong": [s for s in sample_data if s["flip_status"] == "flip_to_wrong"],
    }
    
    colors = {
        "stayed_correct": "#2E86AB",
        "stayed_wrong": "#A23B72",
        "flip_to_correct": "#2CA58D",
        "flip_to_wrong": "#F18F01",
    }
    
    # Plot 1: delta_KL
    ax = axes[0]
    data_for_box = []
    labels = []
    for group_name, group_samples in groups.items():
        if group_samples:
            data_for_box.append([s["delta_kl"] for s in group_samples])
            labels.append(f"{group_name}\n(n={len(group_samples)})")
    
    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[g] for g in groups.keys() if groups[g]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("KL Divergence (before || after)")
    ax.set_title(f"Subject {subject_id}: Distribution Shift by Flip Status")
    ax.set_yscale('log')
    
    # Plot 2: delta_logits
    ax = axes[1]
    data_for_box = []
    labels = []
    for group_name, group_samples in groups.items():
        if group_samples:
            data_for_box.append([s["delta_logits"] for s in group_samples])
            labels.append(f"{group_name}\n(n={len(group_samples)})")
    
    bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], [colors[g] for g in groups.keys() if groups[g]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("||logits_after - logits_before||")
    ax.set_title(f"Subject {subject_id}: Logit Change by Flip Status")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"delta_distribution_s{subject_id}.png", dpi=150)
    plt.close()


def plot_confusion_matrix_diff(sample_data, subject_id, output_dir, n_classes=4):
    """
    Confusion matrix の差分（after - before）
    どのクラスがどのクラスに吸い込まれたか
    """
    # Build confusion matrices
    cm_before = np.zeros((n_classes, n_classes), dtype=int)
    cm_after = np.zeros((n_classes, n_classes), dtype=int)
    
    for s in sample_data:
        true_label = s["true_label"]
        cm_before[true_label, s["pred_before"]] += 1
        cm_after[true_label, s["pred_after"]] += 1
    
    cm_diff = cm_after - cm_before
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Before
    ax = axes[0]
    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id}: Confusion Matrix (BEFORE)")
    
    # After
    ax = axes[1]
    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id}: Confusion Matrix (AFTER)")
    
    # Diff
    ax = axes[2]
    # Use diverging colormap for diff
    vmax = max(abs(cm_diff.min()), abs(cm_diff.max()))
    sns.heatmap(cm_diff, annot=True, fmt='+d', cmap='RdBu_r', ax=ax, center=0, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id}: Confusion Matrix DIFF (After - Before)")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_diff_s{subject_id}.png", dpi=150)
    plt.close()


def plot_flip_analysis_summary(summary_table, output_dir):
    """
    Flip解析サマリ：被験者ごとの flip_to_correct / flip_to_wrong
    """
    subjects = [row["subject"] for row in summary_table]
    flip_to_correct = [row["flip_to_correct"] for row in summary_table]
    flip_to_wrong = [row["flip_to_wrong"] for row in summary_table]
    acc_change = [row["acc_change"] for row in summary_table]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Stacked bar of flip counts
    ax = axes[0]
    x = np.arange(len(subjects))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, flip_to_correct, width, label='Flip to Correct (+)', color='#2CA58D', alpha=0.8)
    bars2 = ax.bar(x + width/2, flip_to_wrong, width, label='Flip to Wrong (−)', color='#F18F01', alpha=0.8)
    
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Flip Analysis: TTT Impact per Subject")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.legend()
    
    # Add net flip text
    for i, (c, w) in enumerate(zip(flip_to_correct, flip_to_wrong)):
        net = c - w
        color = '#2CA58D' if net > 0 else '#F18F01' if net < 0 else 'gray'
        ax.text(i, max(c, w) + 2, f"net: {net:+d}", ha='center', fontsize=9, color=color)
    
    # Plot 2: Accuracy change bar
    ax = axes[1]
    colors = ['#2CA58D' if a > 0 else '#F18F01' if a < 0 else 'gray' for a in acc_change]
    ax.bar(x, [a * 100 for a in acc_change], color=colors, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy Change (%)")
    ax.set_title("TTT Impact: Accuracy Δ")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects])
    
    # Add value labels
    for i, a in enumerate(acc_change):
        ax.text(i, a * 100 + (1 if a >= 0 else -2), f"{a*100:+.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "flip_analysis_summary.png", dpi=150)
    plt.close()


def plot_entropy_pmax_scatter(sample_data, subject_id, output_dir):
    """
    Entropy vs pmax scatter (before/after), colored by flip status.
    Shows decision confidence landscape.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    groups = {
        "stayed_correct": [s for s in sample_data if s["flip_status"] == "stayed_correct"],
        "stayed_wrong": [s for s in sample_data if s["flip_status"] == "stayed_wrong"],
        "flip_to_correct": [s for s in sample_data if s["flip_status"] == "flip_to_correct"],
        "flip_to_wrong": [s for s in sample_data if s["flip_status"] == "flip_to_wrong"],
    }
    
    colors = {
        "stayed_correct": "#2E86AB",
        "stayed_wrong": "#A23B72",
        "flip_to_correct": "#2CA58D",
        "flip_to_wrong": "#F18F01",
    }
    
    # Before
    ax = axes[0]
    for group_name, group_samples in groups.items():
        if group_samples:
            entropy = [s["entropy_before"] for s in group_samples]
            pmax = [s["pmax_before"] for s in group_samples]
            ax.scatter(entropy, pmax, alpha=0.5, s=15, label=group_name, color=colors[group_name])
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Max Probability (pmax)")
    ax.set_title(f"Subject {subject_id}: Confidence Landscape (BEFORE)")
    ax.legend(fontsize=8)
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5)  # random chance for 4-class
    
    # After
    ax = axes[1]
    for group_name, group_samples in groups.items():
        if group_samples:
            entropy = [s["entropy_after"] for s in group_samples]
            pmax = [s["pmax_after"] for s in group_samples]
            ax.scatter(entropy, pmax, alpha=0.5, s=15, label=group_name, color=colors[group_name])
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Max Probability (pmax)")
    ax.set_title(f"Subject {subject_id}: Confidence Landscape (AFTER)")
    ax.legend(fontsize=8)
    ax.axhline(0.25, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"entropy_pmax_scatter_s{subject_id}.png", dpi=150)
    plt.close()


def plot_subject_comparison_table(all_results, summary_table, output_dir):
    """
    被験者間比較テーブル（論文用）
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Build table data
    columns = [
        "Subject",
        "Acc Before",
        "Acc After",
        "Δ Acc",
        "→Correct",
        "→Wrong",
        "Net",
        "H̄ Before",
        "H̄ After",
        "Δ KL̄",
    ]
    
    cell_data = []
    for row in summary_table:
        cell_data.append([
            f"S{row['subject']}",
            f"{row['acc_before']:.3f}",
            f"{row['acc_after']:.3f}",
            f"{row['acc_change']:+.3f}",
            f"{row['flip_to_correct']}",
            f"{row['flip_to_wrong']}",
            f"{row['net_flip']:+d}",
            f"{row['entropy_before']:.3f}",
            f"{row['entropy_after']:.3f}",
            f"{row['delta_kl']:.4f}",
        ])
    
    # Color cells based on improvement/degradation
    cell_colors = []
    for row in summary_table:
        row_colors = ['white'] * len(columns)
        # Acc change color
        if row['acc_change'] > 0.02:
            row_colors[3] = '#c8e6c9'  # green
        elif row['acc_change'] < -0.02:
            row_colors[3] = '#ffcdd2'  # red
        # Net flip color
        if row['net_flip'] > 5:
            row_colors[6] = '#c8e6c9'
        elif row['net_flip'] < -5:
            row_colors[6] = '#ffcdd2'
        cell_colors.append(row_colors)
    
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=['#e3f2fd'] * len(columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title("2-Pass TTT Analysis: Subject Comparison", fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / "subject_comparison_table.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Path to 2-pass debug results directory")
    parser.add_argument("--n_classes", type=int, default=4)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    all_results, summary_table = load_results(results_dir)
    
    print(f"Generating visualizations...")
    
    # Subject comparison
    plot_flip_analysis_summary(summary_table, output_dir)
    plot_subject_comparison_table(all_results, summary_table, output_dir)
    print("  [✓] Summary figures")
    
    # Per-subject figures
    for subject_key, subject_data in all_results.items():
        subject_id = subject_key.replace("subject_", "")
        sample_data = subject_data["samples"]
        
        print(f"  Subject {subject_id}...")
        plot_entropy_distribution_by_correctness(sample_data, subject_id, output_dir)
        plot_delta_distribution_4groups(sample_data, subject_id, output_dir)
        plot_confusion_matrix_diff(sample_data, subject_id, output_dir, args.n_classes)
        plot_entropy_pmax_scatter(sample_data, subject_id, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()


