
import os
import glob
import pandas as pd
import re

def get_avg_from_results_txt(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r'Average Test Accuracy[:\s]+([\d\.]+)', content)
            if match:
                return float(match.group(1))
    except:
        pass
    return None

def check_summary_file(filepath):
    try:
        df = pd.read_csv(filepath)
        baseline_avg = df['Baseline_Acc'].mean() * 100
        otta_avg = df['OTTA_Acc_0.98'].mean() * 100
        print(f"File: {os.path.basename(filepath)}")
        print(f"  Dataset: {df['Dataset'].iloc[0]}")
        print(f"  Baseline Avg: {baseline_avg:.2f}%")
        print(f"  OTTA Avg:     {otta_avg:.2f}%")
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")

# Check 2b candidate
print("--- Checking BCIC 2b Candidate ---")
check_summary_file("/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/multiset_otta_experiment_20260122_2002/summary_results.txt")

# Check HGD candidate
print("\n--- Checking HGD Candidate ---")
check_summary_file("/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/multiset_otta_experiment_hgd_20260122_2221/summary_results.txt")

# Check 2a Baseline Candidate
print("\n--- Checking BCIC 2a Baseline Candidate ---")
p = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/TCFormer_bcic2a_seed-0_aug-True_GPU0_20260121_2342/results.txt"
avg = get_avg_from_results_txt(p)
print(f"File: {os.path.basename(p)}")
print(f"  Average Acc: {avg}")
