import argparse
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing from core
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from intentflow.offline.analysis.core import OTTAAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Generate tabular reports for OTTA experiments.")
    parser.add_argument("results_dir", type=str, nargs="?", default=None,
                        help="Path to the results directory. If omitted, finds the latest.")
    parser.add_argument("--mode", type=str, choices=["basic", "detailed", "gating", "correlation", "all"], default="all",
                        help="Type of report to generate.")
    
    args = parser.parse_args()
    
    try:
        analyzer = OTTAAnalyzer(args.results_dir)
        df_summary = analyzer.get_subject_metrics()
        
        print(f"\nAnalyzing: {analyzer.results_dir}\n")
        
        if args.mode in ["basic", "all"]:
            print("=== Basic Performance ===")
            cols = ['Subject', 'Total', 'Acc (%)', 'Orig Acc (%)', 'Adapt Rate (%)']
            print(df_summary[cols].to_string(index=False, float_format="%.2f"))
            print("-" * 40)
            print(f"Average Accuracy: {df_summary['Acc (%)'].mean():.2f}%")
            print(f"Average Adapt Rate: {df_summary['Adapt Rate (%)'].mean():.2f}%\n")
            
        if args.mode in ["gating", "all"]:
            print("=== Gating Impact (Flips) ===")
            cols = ['Subject', 'Total', 'Flip (%)', 'Enc (N->A)', 'Sup (A->N)']
            print(df_summary[cols].to_string(index=False, float_format="%.2f"))
            print("-" * 40)
            print(f"Total Suppressed (Filtered Noise): {df_summary['Sup (A->N)'].sum()}")
            print(f"Total Encouraged (Motor Focus): {df_summary['Enc (N->A)'].sum()}\n")
            
        if args.mode in ["correlation", "all"]:
            print("=== Neuro-Score Correlation & Stats ===")
            cols = ['Subject', 'Mean NS', 'Mean SAL', 'NS-Acc Corr']
            print(df_summary[cols].to_string(index=False, float_format="%.4f"))
            print("-" * 40)
            
            # Global correlation
            df_full = analyzer.load_data()
            if df_full['neuro_score'].std() > 1e-6 and df_full['label'].notnull().all():
                global_corr = df_full['neuro_score'].corr(df_full['is_correct'])
                print(f"Global Correlation (NS v. Correct): {global_corr:.4f}\n")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
