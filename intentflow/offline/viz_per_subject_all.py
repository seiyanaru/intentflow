
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os

# --- Configuration ---
OUTPUT_DIR = "vis_results_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File Paths
FILE_2A_BASE = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260121_2342/results.txt"
FILE_2A_OTTA = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/tcformer_otta_bcic2a_seed-0_aug-True_GPU0_20260122_1704/results.txt"
FILE_2B_SUMMARY = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/multiset_otta_experiment_20260122_2002/summary_results.txt"
FILE_HGD_SUMMARY = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/multiset_otta_experiment_hgd_20260122_2221/summary_results.txt"

# Paper Values (from images)
PAPER_SCORES = {
    'BCIC 2a': 84.8,
    'BCIC 2b': 87.7,
    'HGD': 96.3
}

# --- Helpers ---
def parse_results_txt(filepath):
    subjects = []
    accs = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'Subject (\d+) => .*Test Acc: ([\d\.]+)', line)
            if match:
                subjects.append(int(match.group(1)))
                accs.append(float(match.group(2)) * 100)
    return subjects, accs

def parse_summary_txt(filepath):
    df = pd.read_csv(filepath)
    subjects = df['Subject'].tolist()
    base_accs = (df['Baseline_Acc'] * 100).tolist()
    otta_accs = (df['OTTA_Acc_0.98'] * 100).tolist()
    return subjects, base_accs, otta_accs

# --- Data Loading ---
print("Loading data...")

# BCIC 2a
sub_2a, acc_2a_base = parse_results_txt(FILE_2A_BASE)
_, acc_2a_otta = parse_results_txt(FILE_2A_OTTA)
avg_2a_base = np.mean(acc_2a_base)
avg_2a_otta = np.mean(acc_2a_otta)

# BCIC 2b
sub_2b, acc_2b_base, acc_2b_otta = parse_summary_txt(FILE_2B_SUMMARY)
avg_2b_base = np.mean(acc_2b_base)
avg_2b_otta = np.mean(acc_2b_otta)

# HGD
sub_hgd, acc_hgd_base, acc_hgd_otta = parse_summary_txt(FILE_HGD_SUMMARY)
avg_hgd_base = np.mean(acc_hgd_base)
avg_hgd_otta = np.mean(acc_hgd_otta)


# --- Plotting Functions ---

def plot_average_comparison():
    datasets = ['BCIC 2a', 'BCIC 2b', 'HGD']
    
    paper_vals = [PAPER_SCORES['BCIC 2a'], PAPER_SCORES['BCIC 2b'], PAPER_SCORES['HGD']]
    base_vals = [avg_2a_base, avg_2b_base, avg_hgd_base]
    otta_vals = [avg_2a_otta, avg_2b_otta, avg_hgd_otta]
    
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    c_paper = '#95a5a6' # Gray
    c_base = '#3498db'  # Blue
    c_otta = '#e74c3c'  # Red
    
    rects1 = ax.bar(x - width, paper_vals, width, label='TCFormer (Paper SOTA)', color=c_paper)
    rects2 = ax.bar(x, base_vals, width, label='Our Baseline', color=c_base)
    rects3 = ax.bar(x + width, otta_vals, width, label='Our Method (Pmax-SAL)', color=c_otta)
    
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Average Performance Comparison across Datasets', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylim(60, 100)
    ax.legend(loc='lower center', fontsize=12, ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_bar_avg.png", dpi=300)
    print(f"Saved {OUTPUT_DIR}/comparison_bar_avg.png")
    plt.close()

def plot_subject_comparison(dataset_name, subjects, base_accs, otta_accs):
    x = np.arange(len(subjects))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    c_base = '#3498db'
    c_otta = '#e74c3c'
    
    rects1 = ax.bar(x - width/2, base_accs, width, label='Baseline', color=c_base)
    rects2 = ax.bar(x + width/2, otta_accs, width, label='Pmax-SAL OTTA', color=c_otta)
    
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'{dataset_name}: Per-Subject Accuracy Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects], fontsize=12)
    
    # Dynamic ylim
    min_val = min(min(base_accs), min(otta_accs))
    ax.set_ylim(max(0, min_val - 10), 100)
    
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate improvements
    for i, (b, o) in enumerate(zip(base_accs, otta_accs)):
        diff = o - b
        if abs(diff) > 0.1:
            color = 'green' if diff > 0 else 'red'
            sign = '+' if diff > 0 else ''
            # ax.annotate(f'{sign}{diff:.1f}%',
            #             xy=(x[i], max(b, o)),
            #             xytext=(0, 5),
            #             textcoords="offset points",
            #             ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    filename = f"subject_comparison_{dataset_name.replace(' ', '').lower()}.png"
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    print(f"Saved {OUTPUT_DIR}/{filename}")
    plt.close()

# --- Execution ---
plot_average_comparison()
plot_subject_comparison("BCIC 2a", sub_2a, acc_2a_base, acc_2a_otta)
plot_subject_comparison("BCIC 2b", sub_2b, acc_2b_base, acc_2b_otta)
plot_subject_comparison("HGD", sub_hgd, acc_hgd_base, acc_hgd_otta)
