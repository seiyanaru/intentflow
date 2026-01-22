
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def analyze_subject(subject_id, results_dir, pmax_threshold=0.7, sal_threshold=0.5):
    """Load and analyze stats for a single subject."""
    filepath = os.path.join(results_dir, f"otta_stats_s{subject_id}_tcformer_otta.npz")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    data = np.load(filepath)
    pmax = data['pmax']
    sal = data['sal']
    adapted = data['adapted']
    # Check if pred/label exist (backward compatibility)
    if 'pred' in data and 'label' in data:
        pred = data['pred']
        label = data['label']
        
        # Calculate Pseudo-Label Accuracy (Accuracy of triggers)
        # We want to know: When we adapted, was the prediction correct?
        if np.sum(adapted) > 0:
            adapted_correct = (pred[adapted==1] == label[adapted==1])
            pl_acc = np.mean(adapted_correct)
        else:
            pl_acc = 0.0
    else:
        pl_acc = None
    
    # Calculate stats
    total = len(pmax)
    n_adapted = np.sum(adapted)
    adaptation_rate = n_adapted / total
    
    # Categorize samples
    # High Pmax, High SAL (Adapt)
    c1 = (pmax > pmax_threshold) & (sal > sal_threshold)
    # High Pmax, Low SAL (Overconfident - Skip)
    c2 = (pmax > pmax_threshold) & (sal <= sal_threshold)
    # Low Pmax, High SAL (Cautious Adapt)
    c3 = (pmax <= pmax_threshold) & (sal > sal_threshold)
    # Low Pmax, Low SAL (Unreliable - Skip)
    c4 = (pmax <= pmax_threshold) & (sal <= sal_threshold)
    
    stats = {
        'subject_id': subject_id,
        'total': total,
        'adapted': n_adapted,
        'rate': adaptation_rate,
        'pl_acc': pl_acc,  # NEW
        'mean_pmax': np.mean(pmax),
        'mean_sal': np.mean(sal),
        'c1_trust': np.sum(c1),
        'c2_overconf': np.sum(c2),
        'c3_cautious': np.sum(c3),
        'c4_unreliable': np.sum(c4)
    }
    
    return stats, pmax, sal, adapted

def plot_trigger_distribution(s3_data, s6_data, output_dir, pmax_thresh=0.7, sal_thresh=0.5):
    """Plot Scatter of Pmax vs SAL for S3 and S6."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    subjects = [('S3 (Success: +12%)', s3_data), ('S6 (Fail: -3.5%)', s6_data)]
    
    for ax, (title, (stats, pmax, sal, adapted)) in zip(axes, subjects):
        # Scatter plot
        # Color by category
        
        # Plot points
        # Adapted (Green) vs Skipped (Red)
        ax.scatter(pmax[adapted==1], sal[adapted==1], c='green', alpha=0.6, label='Adapted', s=50)
        ax.scatter(pmax[adapted==0], sal[adapted==0], c='red', alpha=0.6, label='Skipped', s=50, marker='x')
        
        # Threshold lines
        ax.axvline(x=pmax_thresh, color='k', linestyle='--', alpha=0.5, label='Pmax Threshold')
        ax.axhline(y=sal_thresh, color='k', linestyle='--', alpha=0.5, label='SAL Threshold')
        
        # Regions
        # Top-Right: Trust
        ax.text(0.98, 0.98, f"Trust (Adapt)\n{stats['c1_trust']} samples", 
                transform=ax.transAxes, ha='right', va='top', bbox=dict(facecolor='green', alpha=0.1))
        
        # Bottom-Right: Overconfident
        ax.text(0.98, 0.02, f"Overconfident (Skip)\n{stats['c2_overconf']} samples", 
                transform=ax.transAxes, ha='right', va='bottom', bbox=dict(facecolor='red', alpha=0.1))
        
        # Top-Left: Cautious
        ax.text(0.02, 0.98, f"Cautious (Adapt)\n{stats['c3_cautious']} samples", 
                transform=ax.transAxes, ha='left', va='top', bbox=dict(facecolor='yellow', alpha=0.1))
        
        # Bottom-Left: Unreliable
        ax.text(0.02, 0.02, f"Unreliable (Skip)\n{stats['c4_unreliable']} samples", 
                transform=ax.transAxes, ha='left', va='bottom', bbox=dict(facecolor='grey', alpha=0.1))

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Pmax (Confidence)', fontsize=12)
        ax.set_ylabel('SAL (Source Alignment)', fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-1.05, 1.05) # Cosine sim is -1 to 1
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center right')
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'otta_scatter_s3_vs_s6.png')
    plt.savefig(save_path)
    print(f"Saved scatter plot to {save_path}")


def main():
    RESULTS_DIR = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/tcformer_otta_bcic2a_seed-0_aug-True_GPU0_20260122_1704"
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze S3 and S6
    print("Analyzing Subject 3 (Success)...")
    s3_data = analyze_subject(3, RESULTS_DIR)
    
    print("Analyzing Subject 6 (Failure)...")
    s6_data = analyze_subject(6, RESULTS_DIR)
    
    if s3_data and s6_data:
        # Print comparative stats
        print("\n--- Comparative Statistics ---")
        print(f"{'Metric':<20} | {'Subject 3':<10} | {'Subject 6':<10}")
        print("-" * 50)
        s3_stats = s3_data[0]
        s6_stats = s6_data[0]
        
        print(f"{'Total Samples':<20} | {s3_stats['total']:<10} | {s6_stats['total']:<10}")
        print(f"{'Adaptation Rate':<20} | {s3_stats['rate']*100:.1f}%      | {s6_stats['rate']*100:.1f}%")
        if s3_stats['pl_acc'] is not None:
             print(f"{'Pseudo-Label Acc':<20} | {s3_stats['pl_acc']*100:.1f}%      | {s6_stats['pl_acc']*100:.1f}%")
        print(f"{'Mean Pmax':<20} | {s3_stats['mean_pmax']:.3f}      | {s6_stats['mean_pmax']:.3f}")
        print(f"{'Mean SAL':<20} | {s3_stats['mean_sal']:.3f}      | {s6_stats['mean_sal']:.3f}")
        print(f"{'Overconfident (Skip)':<20} | {s3_stats['c2_overconf']:<10} | {s6_stats['c2_overconf']:<10}")
        print(f"{'Unreliable (Skip)':<20} | {s3_stats['c4_unreliable']:<10} | {s6_stats['c4_unreliable']:<10}")
        
        # Plot
        plot_trigger_distribution(s3_data, s6_data, OUTPUT_DIR)
        
if __name__ == "__main__":
    main()
