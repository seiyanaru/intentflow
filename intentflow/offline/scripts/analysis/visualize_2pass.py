#!/usr/bin/env python3
"""
2-Pass Entropy Gating 結果の可視化

生成する図:
1. エントロピー分布（Pass1/Pass2）＋「正誤で色分け」
2. Flip解析サマリ図（被験者比較）
3. Alpha/lr_scale vs Entropy scatter
4. Confusion matrix の差分（Pass2 - Pass1）
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
    """Load 2-pass debug results from the new format."""
    results_dir = Path(results_dir)
    
    all_data = {}
    summary = []
    
    # Find all debug files
    for debug_file in results_dir.glob("debug_s*_TCFormer_Hybrid.json"):
        subject_id = debug_file.stem.split("_")[0].replace("debug_s", "")
        
        with open(debug_file) as f:
            debug = json.load(f)
        
        # Load sample-level data if exists
        twopass_file = results_dir / f"twopass_s{subject_id}_TCFormer_Hybrid.json"
        samples = []
        if twopass_file.exists():
            with open(twopass_file) as f:
                samples = json.load(f)
        
        all_data[f"subject_{subject_id}"] = {
            "debug": debug,
            "samples": samples,
        }
        
        if "two_pass_analysis" in debug:
            tpa = debug["two_pass_analysis"]
            summary.append({
                "subject": int(subject_id),
                "acc_pass1": tpa.get("acc_pass1", 0),
                "acc_pass2": tpa.get("acc_pass2", 0),
                "acc_change": tpa.get("acc_change", 0),
                "flip_to_correct": tpa.get("n_flip_to_correct", 0),
                "flip_to_wrong": tpa.get("n_flip_to_wrong", 0),
                "net_flip": tpa.get("net_flip", 0),
                "entropy_pass1": tpa.get("entropy_pass1_mean", 0),
                "entropy_pass2": tpa.get("entropy_pass2_mean", 0),
            })
    
    # Sort by subject ID
    summary = sorted(summary, key=lambda x: x["subject"])
    
    return all_data, summary


def plot_entropy_distribution(samples, subject_id, output_dir):
    """
    エントロピー分布（Pass1/Pass2）を正誤で色分け
    """
    if not samples:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Categorize samples
    groups = {
        "stayed_correct": [s for s in samples if s.get("flip_status") == "stayed_correct"],
        "stayed_wrong": [s for s in samples if s.get("flip_status") == "stayed_wrong"],
        "flip_to_correct": [s for s in samples if s.get("flip_status") == "flip_to_correct"],
        "flip_to_wrong": [s for s in samples if s.get("flip_status") == "flip_to_wrong"],
    }
    
    colors = {
        "stayed_correct": "#2E86AB",   # blue
        "stayed_wrong": "#A23B72",     # magenta
        "flip_to_correct": "#2CA58D",  # green
        "flip_to_wrong": "#F18F01",    # orange
    }
    
    # Plot 1: Entropy Pass1 (histogram by group)
    ax = axes[0, 0]
    for group_name, group_samples in groups.items():
        if group_samples and all("entropy_pass1" in s for s in group_samples):
            entropies = [s["entropy_pass1"] for s in group_samples]
            ax.hist(entropies, bins=30, alpha=0.6, label=f"{group_name} (n={len(group_samples)})", 
                   color=colors[group_name], density=True)
    ax.set_xlabel("Entropy (Pass 1: Attention-only)")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id}: Entropy PASS 1 by Flip Status")
    ax.legend(fontsize=8)
    ax.axvline(np.log(4), color='gray', linestyle='--', alpha=0.5, label='max entropy (ln4)')
    
    # Plot 2: Entropy Pass2 (histogram by group)
    ax = axes[0, 1]
    for group_name, group_samples in groups.items():
        if group_samples and all("entropy_pass2" in s for s in group_samples):
            entropies = [s["entropy_pass2"] for s in group_samples]
            ax.hist(entropies, bins=30, alpha=0.6, label=f"{group_name} (n={len(group_samples)})", 
                   color=colors[group_name], density=True)
    ax.set_xlabel("Entropy (Pass 2: with TTT)")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id}: Entropy PASS 2 by Flip Status")
    ax.legend(fontsize=8)
    ax.axvline(np.log(4), color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Delta Entropy (Pass2 - Pass1)
    ax = axes[1, 0]
    for group_name, group_samples in groups.items():
        if group_samples and all("entropy_pass1" in s and "entropy_pass2" in s for s in group_samples):
            deltas = [s["entropy_pass2"] - s["entropy_pass1"] for s in group_samples]
            ax.hist(deltas, bins=30, alpha=0.6, label=f"{group_name}", 
                   color=colors[group_name], density=True)
    ax.set_xlabel("Δ Entropy (Pass2 - Pass1)")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id}: Entropy Change by TTT")
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)
    
    # Plot 4: Scatter (entropy_pass1 vs alpha), colored by flip status
    ax = axes[1, 1]
    for group_name, group_samples in groups.items():
        if group_samples and all("entropy_pass1" in s and "alpha" in s for s in group_samples):
            e_p1 = [s["entropy_pass1"] for s in group_samples]
            alpha = [s["alpha"] for s in group_samples]
            ax.scatter(e_p1, alpha, alpha=0.5, s=15, label=group_name, color=colors[group_name])
    ax.set_xlabel("Entropy (Pass 1)")
    ax.set_ylabel("Alpha (gating strength)")
    ax.set_title(f"Subject {subject_id}: Entropy vs Alpha")
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"entropy_distribution_s{subject_id}.png", dpi=150)
    plt.close()


def plot_flip_analysis_summary(summary, output_dir):
    """
    Flip解析サマリ：被験者ごとの flip_to_correct / flip_to_wrong
    """
    if not summary:
        return
    
    subjects = [row["subject"] for row in summary]
    flip_to_correct = [row["flip_to_correct"] for row in summary]
    flip_to_wrong = [row["flip_to_wrong"] for row in summary]
    acc_change = [row["acc_change"] for row in summary]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Stacked bar of flip counts
    ax = axes[0]
    x = np.arange(len(subjects))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, flip_to_correct, width, label='Flip to Correct (+)', color='#2CA58D', alpha=0.8)
    bars2 = ax.bar(x + width/2, flip_to_wrong, width, label='Flip to Wrong (−)', color='#F18F01', alpha=0.8)
    
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of Samples")
    ax.set_title("2-Pass TTT: Flip Analysis per Subject")
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
    colors_bar = ['#2CA58D' if a > 0 else '#F18F01' if a < 0 else 'gray' for a in acc_change]
    ax.bar(x, [a * 100 for a in acc_change], color=colors_bar, alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy Change (%)")
    ax.set_title("2-Pass TTT: Accuracy Δ (Pass2 - Pass1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects])
    
    # Add value labels
    for i, a in enumerate(acc_change):
        ax.text(i, a * 100 + (1 if a >= 0 else -2), f"{a*100:+.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "flip_analysis_summary.png", dpi=150)
    plt.close()


def plot_confusion_matrix_diff(samples, subject_id, output_dir, n_classes=4):
    """
    Confusion matrix の差分（Pass2 - Pass1）
    """
    if not samples:
        return
    
    # Build confusion matrices
    cm_pass1 = np.zeros((n_classes, n_classes), dtype=int)
    cm_pass2 = np.zeros((n_classes, n_classes), dtype=int)
    
    for s in samples:
        if "true_label" not in s:
            continue
        true_label = s["true_label"]
        cm_pass1[true_label, s["pred_pass1"]] += 1
        cm_pass2[true_label, s["pred_pass2"]] += 1
    
    cm_diff = cm_pass2 - cm_pass1
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pass 1
    ax = axes[0]
    sns.heatmap(cm_pass1, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id}: Confusion Matrix (PASS 1)")
    
    # Pass 2
    ax = axes[1]
    sns.heatmap(cm_pass2, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id}: Confusion Matrix (PASS 2)")
    
    # Diff
    ax = axes[2]
    vmax = max(abs(cm_diff.min()), abs(cm_diff.max())) if cm_diff.any() else 1
    sns.heatmap(cm_diff, annot=True, fmt='+d', cmap='RdBu_r', ax=ax, center=0, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id}: Confusion Matrix DIFF (P2 - P1)")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_diff_s{subject_id}.png", dpi=150)
    plt.close()


def plot_alpha_entropy_scatter(samples, subject_id, output_dir):
    """
    Alpha vs Entropy (Pass1) scatter with correctness coloring
    """
    if not samples:
        return
    
    # Check if alpha data exists
    if not any("alpha" in s for s in samples):
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    groups = {
        "stayed_correct": [s for s in samples if s.get("flip_status") == "stayed_correct"],
        "stayed_wrong": [s for s in samples if s.get("flip_status") == "stayed_wrong"],
        "flip_to_correct": [s for s in samples if s.get("flip_status") == "flip_to_correct"],
        "flip_to_wrong": [s for s in samples if s.get("flip_status") == "flip_to_wrong"],
    }
    
    colors = {
        "stayed_correct": "#2E86AB",
        "stayed_wrong": "#A23B72",
        "flip_to_correct": "#2CA58D",
        "flip_to_wrong": "#F18F01",
    }
    
    for group_name, group_samples in groups.items():
        if group_samples and all("entropy_pass1" in s and "alpha" in s for s in group_samples):
            e = [s["entropy_pass1"] for s in group_samples]
            a = [s["alpha"] for s in group_samples]
            ax.scatter(e, a, alpha=0.6, s=30, label=f"{group_name} (n={len(group_samples)})", color=colors[group_name])
    
    ax.set_xlabel("Entropy (Pass 1)", fontsize=12)
    ax.set_ylabel("Alpha (TTT gating strength)", fontsize=12)
    ax.set_title(f"Subject {subject_id}: Entropy-to-Alpha Gating", fontsize=14)
    ax.legend(loc='upper left')
    
    # Mark threshold region
    ax.axvline(np.log(4) * 0.85, color='red', linestyle='--', alpha=0.5, label='threshold (0.85 * max)')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"alpha_entropy_scatter_s{subject_id}.png", dpi=150)
    plt.close()


def plot_delta_distribution_4groups(samples, subject_id, output_dir):
    """
    delta_KL / delta_logits のボックスプロット（4群比較）
    - stayed_correct, stayed_wrong, flip_to_correct, flip_to_wrong
    """
    if not samples:
        return
    
    # Check if delta data exists
    if not any("delta_kl" in s for s in samples):
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    groups = {
        "stayed_correct": [s for s in samples if s.get("flip_status") == "stayed_correct"],
        "stayed_wrong": [s for s in samples if s.get("flip_status") == "stayed_wrong"],
        "flip_to_correct": [s for s in samples if s.get("flip_status") == "flip_to_correct"],
        "flip_to_wrong": [s for s in samples if s.get("flip_status") == "flip_to_wrong"],
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
    box_colors = []
    for group_name, group_samples in groups.items():
        if group_samples and all("delta_kl" in s for s in group_samples):
            data_for_box.append([s["delta_kl"] for s in group_samples])
            labels.append(f"{group_name}\n(n={len(group_samples)})")
            box_colors.append(colors[group_name])
    
    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("KL Divergence (Pass1 || Pass2)")
        ax.set_title(f"Subject {subject_id}: Distribution Shift by Flip Status")
        ax.set_yscale('log')
    
    # Plot 2: delta_logits
    ax = axes[1]
    data_for_box = []
    labels = []
    box_colors = []
    for group_name, group_samples in groups.items():
        if group_samples and all("delta_logits" in s for s in group_samples):
            data_for_box.append([s["delta_logits"] for s in group_samples])
            labels.append(f"{group_name}\n(n={len(group_samples)})")
            box_colors.append(colors[group_name])
    
    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("||logits_pass2 - logits_pass1||")
        ax.set_title(f"Subject {subject_id}: Logit Change by Flip Status")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"delta_distribution_s{subject_id}.png", dpi=150)
    plt.close()


def plot_update_analysis(samples, subject_id, output_dir):
    """
    Update analysis: accuracy when updated vs not updated
    """
    if not samples:
        return
    
    # Check if update_on data exists
    if not any("update_on" in s for s in samples):
        return
    
    updated = [s for s in samples if s.get("update_on") == True]
    not_updated = [s for s in samples if s.get("update_on") == False]
    
    if not updated or not not_updated:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy comparison
    ax = axes[0]
    categories = ["Updated", "Not Updated"]
    acc_p1 = [
        np.mean([s["correct_pass1"] for s in updated if "correct_pass1" in s]),
        np.mean([s["correct_pass1"] for s in not_updated if "correct_pass1" in s])
    ]
    acc_p2 = [
        np.mean([s["correct_pass2"] for s in updated if "correct_pass2" in s]),
        np.mean([s["correct_pass2"] for s in not_updated if "correct_pass2" in s])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, [a * 100 for a in acc_p1], width, label='Pass 1 (no TTT)', color='#2E86AB', alpha=0.7)
    bars2 = ax.bar(x + width/2, [a * 100 for a in acc_p2], width, label='Pass 2 (with TTT)', color='#F18F01', alpha=0.7)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Subject {subject_id}: Accuracy by Update Status')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={len(updated) if i==0 else len(not_updated)})" for i, c in enumerate(categories)])
    ax.legend()
    
    # Add change labels
    for i, (a1, a2) in enumerate(zip(acc_p1, acc_p2)):
        change = (a2 - a1) * 100
        color = '#2CA58D' if change > 0 else '#F18F01' if change < 0 else 'gray'
        ax.text(i, max(a1, a2) * 100 + 2, f"Δ: {change:+.1f}%", ha='center', fontsize=10, color=color)
    
    # Plot 2: Entropy distribution by update status
    ax = axes[1]
    if all("entropy_pass1" in s for s in updated + not_updated):
        ent_updated = [s["entropy_pass1"] for s in updated]
        ent_not_updated = [s["entropy_pass1"] for s in not_updated]
        
        ax.hist(ent_updated, bins=30, alpha=0.6, label=f'Updated (n={len(updated)})', color='#F18F01', density=True)
        ax.hist(ent_not_updated, bins=30, alpha=0.6, label=f'Not Updated (n={len(not_updated)})', color='#2E86AB', density=True)
        ax.set_xlabel("Entropy (Pass 1)")
        ax.set_ylabel("Density")
        ax.set_title(f"Subject {subject_id}: Entropy Distribution by Update Status")
        ax.legend()
        ax.axvline(np.log(4) * 0.85, color='red', linestyle='--', alpha=0.5, label='threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"update_analysis_s{subject_id}.png", dpi=150)
    plt.close()


def plot_subject_comparison_table(summary, output_dir):
    """
    被験者間比較テーブル（論文用）
    """
    if not summary:
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    columns = [
        "Subject",
        "Acc P1",
        "Acc P2",
        "Δ Acc",
        "→Correct",
        "→Wrong",
        "Net",
        "H̄ P1",
        "H̄ P2",
    ]
    
    cell_data = []
    for row in summary:
        cell_data.append([
            f"S{row['subject']}",
            f"{row['acc_pass1']:.3f}",
            f"{row['acc_pass2']:.3f}",
            f"{row['acc_change']:+.3f}",
            f"{row['flip_to_correct']}",
            f"{row['flip_to_wrong']}",
            f"{row['net_flip']:+d}",
            f"{row['entropy_pass1']:.3f}",
            f"{row['entropy_pass2']:.3f}",
        ])
    
    # Color cells
    cell_colors = []
    for row in summary:
        row_colors = ['white'] * len(columns)
        if row['acc_change'] > 0.02:
            row_colors[3] = '#c8e6c9'
        elif row['acc_change'] < -0.02:
            row_colors[3] = '#ffcdd2'
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
    parser.add_argument("results_dir", type=str, help="Path to 2-pass results directory")
    parser.add_argument("--n_classes", type=int, default=4)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {results_dir}")
    all_data, summary = load_results(results_dir)
    
    if not all_data:
        print("No data found!")
        return
    
    print(f"Found {len(all_data)} subjects")
    print(f"Generating visualizations...")
    
    # Summary figures
    plot_flip_analysis_summary(summary, output_dir)
    plot_subject_comparison_table(summary, output_dir)
    print("  [✓] Summary figures")
    
    # Per-subject figures
    for subject_key, subject_data in all_data.items():
        subject_id = subject_key.replace("subject_", "")
        samples = subject_data["samples"]
        
        if not samples:
            print(f"  Subject {subject_id}: No sample data, skipping per-sample figures")
            continue
        
        print(f"  Subject {subject_id}...")
        plot_entropy_distribution(samples, subject_id, output_dir)
        plot_confusion_matrix_diff(samples, subject_id, output_dir, args.n_classes)
        plot_alpha_entropy_scatter(samples, subject_id, output_dir)
        plot_delta_distribution_4groups(samples, subject_id, output_dir)
        plot_update_analysis(samples, subject_id, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()

