
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.metrics import accuracy_score, confusion_matrix

def analyze_failure(npz_path, output_dir=None):
    """
    Analyze OTTA failure cases from .npz file.
    """
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found.")
        return

    data = np.load(npz_path)
    pmax = data['pmax']
    sal = data['sal']
    adapted = data['adapted']
    # adapt_weight = data['adapt_weight']
    pred = data['pred']
    label = data['label']
    
    # Check if original_pred exists (it might not in old files)
    if 'original_pred' in data:
        original_pred = data['original_pred']
        has_orig = True
    else:
        print("Warning: 'original_pred' not found in .npz. Identifying negative transfer is limited.")
        original_pred = pred # Fallback (assumes no change if not adapted, but for adapted we don't know)
        has_orig = False

    # Metrics
    acc_otta = accuracy_score(label, pred)
    acc_orig = accuracy_score(label, original_pred) if has_orig else "N/A"
    
    print(f"Analysis for {os.path.basename(npz_path)}")
    print(f"Original Acc: {acc_orig}")
    print(f"OTTA Acc:     {acc_otta:.4f}")
    
    if not has_orig:
        return

    # Identify Transfer Types
    # Positive: Orig Wrong -> New Correct
    # Negative: Orig Correct -> New Wrong
    # Preserved Correct: Orig Correct -> New Correct
    # Preserved Wrong: Orig Wrong -> New Wrong
    
    is_orig_correct = (original_pred == label)
    is_new_correct = (pred == label)
    
    pos_transfer = (~is_orig_correct) & is_new_correct
    neg_transfer = is_orig_correct & (~is_new_correct)
    
    print(f"Positive Transfer: {pos_transfer.sum()} samples")
    print(f"Negative Transfer: {neg_transfer.sum()} samples (FAILURE CASES)")
    
    if output_dir is None:
        output_dir = os.path.dirname(npz_path)
    
    # Visualization: Pmax vs SAL scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot all adapted samples (background)
    adapted_mask = adapted.astype(bool)
    plt.scatter(pmax[adapted_mask], sal[adapted_mask], c='gray', alpha=0.3, label='Adapted Samples', s=20)
    
    # Highlight Negative Transfer (Red)
    # Only adapted samples can cause negative transfer? 
    # Yes, if not adapted, pred == original_pred, so logic holds.
    plt.scatter(pmax[neg_transfer], sal[neg_transfer], c='red', marker='x', s=100, label='Negative Transfer', linewidth=2)
    
    # Highlight Positive Transfer (Green)
    plt.scatter(pmax[pos_transfer], sal[pos_transfer], c='green', marker='o', s=100, label='Positive Transfer', edgecolors='k')

    plt.xlabel('Pmax (Confidence)')
    plt.ylabel('SAL (Source Alignment Level)')
    plt.title(f'Negative Transfer Analysis\nPos: {pos_transfer.sum()}, Neg: {neg_transfer.sum()}')
    plt.legend()
    plt.grid(True)
    
    # Threshold lines
    # Can't easily get threshold from .npz, hardcode standard or pass as arg?
    plt.axvline(x=0.7, color='k', linestyle='--', alpha=0.5, label='Pmax Thresh (0.7)')
    plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='SAL Thresh (0.5)')
    
    save_path = os.path.join(output_dir, f"failure_analysis_{os.path.basename(npz_path).replace('.npz', '.png')}")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .npz file")
    args = parser.parse_args()
    
    analyze_failure(args.file)
