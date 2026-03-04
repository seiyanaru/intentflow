import argparse
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to sys.path to allow importing from core
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from intentflow.offline.analysis.core import OTTAAnalyzer

def plot_scatter(df, output_dir, pmax_thresh=0.7, sal_thresh=0.5):
    """Plot Pmax vs SAL scatter for all subjects."""
    for subject, group in df.groupby('subject'):
        plt.figure(figsize=(10, 8))
        
        adapted = group[group['adapted']]
        skipped = group[~group['adapted']]
        
        plt.scatter(adapted['pmax'], adapted['sal'], c='green', alpha=0.6, label='Adapted', s=50)
        plt.scatter(skipped['pmax'], skipped['sal'], c='red', alpha=0.6, label='Skipped', s=50, marker='x')
        
        plt.axvline(x=pmax_thresh, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=sal_thresh, color='k', linestyle='--', alpha=0.5)
        
        plt.title(f"Pmax vs SAL Distribution (Subject {subject})")
        plt.xlabel("Pmax (Confidence)")
        plt.ylabel("SAL (Source Alignment Level)")
        plt.xlim(0, 1.05)
        plt.ylim(-1.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(output_dir, f"scatter_s{subject}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

def plot_failure_mode(df, output_dir):
    """Plot Pmax vs SAL highlighting positive and negative transfer."""
    if df['label'].isnull().all():
        print("Cannot plot failure modes: Ground truth labels are missing.")
        return
        
    for subject, group in df.groupby('subject'):
        plt.figure(figsize=(10, 8))
        
        # Determine transfer types
        is_orig_correct = group['was_correct_orig']
        is_new_correct = group['is_correct']
        
        pos_transfer = (~is_orig_correct) & is_new_correct
        neg_transfer = is_orig_correct & (~is_new_correct)
        
        adapted_mask = group['adapted']
        
        # Plot all adapted samples
        plt.scatter(group.loc[adapted_mask, 'pmax'], group.loc[adapted_mask, 'sal'], 
                    c='gray', alpha=0.3, label='Adapted Samples', s=20)
                    
        # Highlight transfers
        plt.scatter(group.loc[neg_transfer, 'pmax'], group.loc[neg_transfer, 'sal'], 
                    c='red', marker='x', s=100, label='Negative Transfer', linewidth=2)
                    
        plt.scatter(group.loc[pos_transfer, 'pmax'], group.loc[pos_transfer, 'sal'], 
                    c='green', marker='o', s=100, label='Positive Transfer', edgecolors='k')

        plt.title(f"Transfer Analysis (Subject {subject})\nPos: {pos_transfer.sum()}, Neg: {neg_transfer.sum()}")
        plt.xlabel("Pmax (Confidence)")
        plt.ylabel("SAL (Source Alignment Level)")
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(output_dir, f"transfer_analysis_s{subject}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate visual plots for OTTA experiments.")
    parser.add_argument("results_dir", type=str, nargs="?", default=None,
                        help="Path to the results directory. If omitted, finds the latest.")
    parser.add_argument("--type", type=str, choices=["scatter", "failure", "all"], default="all",
                        help="Type of plots to generate.")
                        
    args = parser.parse_args()
    
    try:
        analyzer = OTTAAnalyzer(args.results_dir)
        df_full = analyzer.load_data()
        
        output_dir = os.path.join(analyzer.results_dir, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating plots in: {output_dir}\n")
        
        if args.type in ["scatter", "all"]:
            plot_scatter(df_full, output_dir)
            
        if args.type in ["failure", "all"]:
            plot_failure_mode(df_full, output_dir)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
