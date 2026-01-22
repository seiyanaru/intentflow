#!/usr/bin/env python3
"""
BCIC 2a 2-Pass Entropy Gating 実験

Purpose: 
- Pass 1: Attention-only (TTT OFF) → Entropy計算
- Entropy Gating: H > threshold なら Pass 2 へ
- Pass 2: TTT ON with gated α, lr_scale → 最終予測

目的:
- 被験者1, 2, 5 で精度変化が起きる原因を特定
- flip解析（正誤が変わった試行の内訳）
- 更新量解析（どれだけ動かしたか）

出力:
- debug_s{subject}_TCFormer_Hybrid.json: 2-pass flip analysis
- twopass_s{subject}_TCFormer_Hybrid.json: サンプルごとの詳細データ
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from train_pipeline import train_and_test


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="1,2,5", help="Comma-separated subject IDs")
    parser.add_argument("--all_subjects", action="store_true", help="Run all 9 subjects")
    parser.add_argument("--entropy_threshold", type=float, default=0.85)
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--lr_scale_max", type=float, default=0.5)
    args = parser.parse_args()
    
    if args.all_subjects:
        subject_ids = list(range(1, 10))
    else:
        subject_ids = [int(s) for s in args.subjects.split(",")]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/bcic2a_2pass/{timestamp}"
    
    # Config: Best settings from grid search + entropy gating (2-pass)
    config = {
        "model": "TCFormer_Hybrid",
        "dataset_name": "bcic2a",
        "seed": 42,
        "max_epochs": 1000,
        "gpu_id": args.gpu_id,
        "subject_ids": subject_ids,
        "results_dir": results_dir,
        "preprocessing": {
            "sfreq": 250,
            "low_cut": None,
            "high_cut": None,
            "start": 0.0,
            "stop": 4.0,
            "batch_size": 48,
            "interaug": True,
            "z_scale": True,
            "data_path": "/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/",
        },
        "model_kwargs": {
            "F1": 32,
            "temp_kernel_lengths": [20, 32, 64],
            "d_group": 16,
            "D": 2,
            "pool_length_1": 8,
            "pool_length_2": 7,
            "dropout_conv": 0.4,
            "use_group_attn": True,
            "q_heads": 4,
            "kv_heads": 4,
            "trans_depth": 2,
            "lr": 0.001,
            "weight_decay": 0.05,
            
            # TTT config (best from grid search)
            "ttt_config": {
                "layer_type": "linear",
                "base_lr": 0.05,
                "mini_batch_size": 16,
                "share_qk": False,
                "use_dual_form": True,
                "learnable_init_state": False,
                "ttt_reg_lambda": 0.05,
                "ttt_anchor_scale_mode": "none",
                "ttt_loss_scale": 0.05,
                "ttt_grad_clip": 1.0,
            },
            
            # ===== KEY: 2-Pass Entropy Gating =====
            "use_dynamic_gating": True,
            "gating_mode": "entropy",
            "entropy_threshold": args.entropy_threshold,
            "alpha_max": args.alpha_max,
            "lr_scale_max": args.lr_scale_max,
            "entropy_gating_in_train": False,  # 2-pass is ONLY at test time
            "entropy_alpha_init_w": 2.0,
            "entropy_alpha_init_b": -3.0,
            "entropy_lr_init_w": 2.0,
            "entropy_lr_init_b": -2.0,
        }
    }
    
    print("=" * 70)
    print("BCIC 2a 2-Pass Entropy Gating Experiment")
    print("=" * 70)
    print(f"Subjects: {subject_ids}")
    print(f"Results dir: {results_dir}")
    print(f"Gating mode: entropy (2-pass)")
    print(f"  entropy_threshold: {args.entropy_threshold}")
    print(f"  alpha_max: {args.alpha_max}")
    print(f"  lr_scale_max: {args.lr_scale_max}")
    print("=" * 70)
    
    # Save config
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run experiment
    train_and_test(config)
    
    print("\n" + "=" * 70)
    print("Experiment completed!")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)
    
    # Summarize 2-pass results
    print("\n=== 2-Pass Flip Analysis Summary ===")
    for subj in subject_ids:
        debug_path = os.path.join(results_dir, f"debug_s{subj}_TCFormer_Hybrid.json")
        if os.path.exists(debug_path):
            with open(debug_path) as f:
                debug = json.load(f)
            
            if "two_pass_analysis" in debug:
                tpa = debug["two_pass_analysis"]
                print(f"\nSubject {subj}:")
                print(f"  Acc: {tpa['acc_pass1']:.4f} (P1) → {tpa['acc_pass2']:.4f} (P2) | Δ = {tpa['acc_change']:+.4f}")
                print(f"  Flips: {tpa['n_flip_to_correct']} to_correct, {tpa['n_flip_to_wrong']} to_wrong | net = {tpa['net_flip']:+d}")
                if "entropy_p1_flip_to_correct" in tpa:
                    print(f"  Entropy (P1) flip_to_correct: {tpa['entropy_p1_flip_to_correct']:.4f}")
                if "entropy_p1_flip_to_wrong" in tpa:
                    print(f"  Entropy (P1) flip_to_wrong: {tpa['entropy_p1_flip_to_wrong']:.4f}")
                if "alpha_flip_to_correct" in tpa:
                    print(f"  Alpha flip_to_correct: {tpa['alpha_flip_to_correct']:.4f}")
                if "alpha_flip_to_wrong" in tpa:
                    print(f"  Alpha flip_to_wrong: {tpa['alpha_flip_to_wrong']:.4f}")


if __name__ == "__main__":
    main()

