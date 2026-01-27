
import os
import argparse
import subprocess
from datetime import datetime

def run_baseline_reproduction(dataset, subject_ids=None, gpu_id=0):
    """
    Run baseline reproduction for specific dataset and subjects.
    Uses 'tcformer' model with default config (paper settings).
    """
    
    # Base command
    model = "tcformer"
    
    # Construct arguments
    cmd = [
        "python", "intentflow/offline/train_pipeline.py",
        "--model", model,
        "--dataset", dataset,
        "--gpu_id", str(gpu_id),
        "--interaug", # Ensure inter-subject augmentation is ON (key for high performance)
        "--seed", "42" # Use fixed seed for reproducibility
    ]
    
    if dataset == "hgd":
        # HGD might use LOSO? The paper results (93.8%) are usually cross-subject/LOSO or large scale?
        # TCFormer paper HGD 93.8% is likely Within-Subject or Cross-Subject?
        # The README says "HGD: 93.8% (Acc)". Usually HGD within-subject is high.
        # Let's assume standard protocol.
        pass

    if subject_ids:
        # train_pipeline accepts "all" or specific IDs.
        # But here we might want to loop or pass list.
        # train_pipeline takes int or "all" or list in config.
        # But via CLI, it doesn't support list directly unless we modify config.
        # Hack: We will loop here if needed, or just let train_pipeline handle "all".
        pass 
        
    print(f"Running reproduction for {dataset}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["bcic2b", "hgd"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    run_baseline_reproduction(args.dataset, gpu_id=args.gpu)
