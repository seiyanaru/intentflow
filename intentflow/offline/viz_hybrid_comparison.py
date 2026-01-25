import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_hybrid_comparison(dataset_name, subjects, base, hybrid_static, hybrid_entropy, output_path):
    x = np.arange(len(subjects))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Colors matching the user's uploaded image style
    c_base = '#00a8e1'    # Cyan-ish Blue
    c_stat = '#ffb142'    # Orange
    c_entr = '#ff5252'    # Red/Pink
    
    rects1 = ax.bar(x - width, base, width, label='Base', color=c_base)
    rects2 = ax.bar(x, hybrid_static, width, label='Hybrid (Static)', color=c_stat)
    rects3 = ax.bar(x + width, hybrid_entropy, width, label='Hybrid + Entropy', color=c_entr)
    
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Accuracy Comparison per Subject: {dataset_name}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=12)
    ax.set_ylim(40, 105) 
    ax.legend(loc='lower left', fontsize=12, ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")

def main():
    output_dir = Path("vis_results")
    output_dir.mkdir(exist_ok=True)
    
    # --- BCIC 2b Data ---
    subjects_2b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Baseline from bcic2b/20260113_105901/base/results.txt
    # [82.50, 81.25, 73.75, 98.81, 79.76, 87.50, 80.00, 82.95, 77.50]
    base_2b = [82.50, 81.25, 73.75, 98.81, 79.76, 87.50, 80.00, 82.95, 77.50]
    
    # Hybrid Static from Step 1983
    # [77.50, 78.75, 66.25, 97.62, 84.52, 81.25, 81.25, 78.41, 73.75]
    stat_2b = [77.50, 78.75, 66.25, 97.62, 84.52, 81.25, 81.25, 78.41, 73.75]
    
    # Hybrid Entropy from Step 1861
    # [77.50, 80.00, 75.00, 100.0, 80.95, 82.50, 80.00, 78.41, 72.50]
    entr_2b = [77.50, 80.00, 75.00, 100.0, 80.95, 82.50, 80.00, 78.41, 72.50]
    
    plot_hybrid_comparison("TCFormer Hybrid in BCIC 2b", subjects_2b, base_2b, stat_2b, entr_2b, output_dir / "hybrid_comp_bcic2b.png")

    # --- HGD Data ---
    subjects_hgd = list(range(1, 15))
    # Baseline from hgd/20260114_213132/base/results.txt
    # [90.00, 83.75, 97.50, 100.00, 100.00, 95.63, 86.87, 96.25, 100.00, 95.00, 92.50, 98.75, 90.00, 75.00]
    base_hgd = [90.00, 83.75, 97.50, 100.00, 100.00, 95.63, 86.87, 96.25, 100.00, 95.00, 92.50, 98.75, 90.00, 75.00]
    
    # Hybrid Static from Step 1984
    # [79.37, 75.00, 75.63, 98.12, 98.12, 80.00, 66.87, 74.37, 96.88, 85.00, 71.88, 88.75, 75.00, 72.50]
    stat_hgd = [79.37, 75.00, 75.63, 98.12, 98.12, 80.00, 66.87, 74.37, 96.88, 85.00, 71.88, 88.75, 75.00, 72.50]
    
    # Hybrid Entropy from Step 1860
    # [77.50, 64.38, 92.50, 98.12, 75.00, 80.62, 51.25, 68.12, 99.37, 87.50, 93.12, 93.12, 68.12, 61.25]
    entr_hgd = [77.50, 64.38, 92.50, 98.12, 75.00, 80.62, 51.25, 68.12, 99.37, 87.50, 93.12, 93.12, 68.12, 61.25]
    
    plot_hybrid_comparison("TCFormer Hybrid in HGD", subjects_hgd, base_hgd, stat_hgd, entr_hgd, output_dir / "hybrid_comp_hgd.png")

if __name__ == "__main__":
    main()
