import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def plot_pmax_sal_scatter(npz_path, output_path, subject_id="Unknown"):
    """
    Pmax vs SAL の散布図を描画し、ゲーティングの様子を可視化する。
    """
    data = np.load(npz_path)
    
    # Extract data
    pmax = data['pmax']
    sal = data['sal']
    # adapted flag (1.0 = adapted, 0.0 = skipped)
    adapted = data['adapted'] > 0.5
    
    # Correctness (pred == label)
    # Note: 'pred' and 'label' might not be available in earlier versions of debug_otta.py
    # But run_multiset_otta.py should have saved them.
    if 'pred' in data and 'label' in data:
        correct = data['pred'] == data['label']
    else:
        correct = None

    # Plot settings
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Thresholds
    TH_PMAX = 0.7
    TH_SAL = 0.98
    
    # Filter indices based on LOGIC (since 'adapted' flag in npz might be all True due to logging bug)
    # The experiment used Pmax > 0.7 AND SAL > 0.98
    
    # Recalculate status based on the actual algorithm logic
    is_high_pmax = pmax > TH_PMAX
    is_high_sal = sal > TH_SAL
    
    # Logic: Adapt only if BOTH are true
    idx_adapted = is_high_pmax & is_high_sal
    idx_skipped = ~idx_adapted
    
    # Plot Skipped first (Background)
    plt.scatter(sal[idx_skipped], pmax[idx_skipped], 
                c='gray', marker='x', alpha=0.6, label='Skipped (Rejected)', s=40)
    
    # Plot Adapted (Foreground)
    plt.scatter(sal[idx_adapted], pmax[idx_adapted], 
                c='#007acc', marker='o', alpha=0.9, label='Adapted (Accepted)', s=60, edgecolors='white')
    
    # Analyze Overconfidence (High Pmax, Low SAL)
    # Evidence: Counts
    n_total = len(pmax)
    n_adapted = np.sum(adapted)
    n_skipped = n_total - n_adapted
    print(f"Total: {n_total}, Adapted: {n_adapted}, Skipped: {n_skipped}")
    
    # Visualize Thresholds
    plt.axhline(y=TH_PMAX, color='r', linestyle='--', linewidth=2, label=f'Pmax Threshold ({TH_PMAX})')
    plt.axvline(x=TH_SAL, color='g', linestyle='--', linewidth=2, label=f'SAL Threshold ({TH_SAL})')
    
    # Highlight "Overconfidence Zone" (High Pmax, Low SAL)
    plt.fill_betweenx([TH_PMAX, 1.02], 0, TH_SAL, color='red', alpha=0.1, label='Overconfidence Zone\n(High Confidence, Low Align)')

    plt.title(f"Algorithm Behavior Analysis: Subject {subject_id}\n(Evidence: {n_skipped} samples correctly rejected)", fontsize=15)
    plt.xlabel("Source Alignment Level (SAL) -> 'Similarity to Source'", fontsize=14)
    plt.ylabel("Internal Confidence (Pmax) -> 'Self-Confidence'", fontsize=14)
    plt.xlim(0.0, 1.05)
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved scatter plot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True, help="Path to .npz file")
    parser.add_argument("--output_dir", default="vis_results", help="Output directory")
    parser.add_argument("--subject_id", default="?", help="Subject ID")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"scatter_s{args.subject_id}.png"
    
    plot_pmax_sal_scatter(args.npz_path, output_path, args.subject_id)
