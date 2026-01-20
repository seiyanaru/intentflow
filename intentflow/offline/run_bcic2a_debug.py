#!/usr/bin/env python3
"""
BCIC 2a 動的ゲーティング詳細デバッグ実験

目的: 被験者1, 2, 5 で精度変化が起きる原因を特定する

出力:
- サンプルごとの entropy, alpha, lr_scale, 予測確率, 正解/不正解
- 被験者ごとの統計（正解/不正解別の entropy 分布など）
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything
from utils.metrics import MetricsCallback


def run_debug_experiment(config, results_dir, subject_ids=None):
    """Run experiment with detailed per-sample debugging."""
    
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    model_cls = get_model_cls(config["model"])
    datamodule_cls = get_datamodule_cls(config["dataset_name"])
    
    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes
    
    if subject_ids is None:
        subject_ids = datamodule_cls.all_subject_ids
    
    all_results = {}
    
    for subject_id in subject_ids:
        print(f"\n{'='*60}")
        print(f"Subject {subject_id}")
        print(f"{'='*60}")
        
        seed_everything(config["seed"])
        
        # Create datamodule and model
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(
            **config["model_kwargs"],
            subject_id=subject_id,
            model_name=config["model"],
            results_dir=str(results_dir)
        )
        
        # Training
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices=[config["gpu_id"]],
            accelerator="auto",
            logger=False,
            enable_checkpointing=True,
            callbacks=[
                MetricsCallback(),
                EarlyStopping(monitor="val_loss", mode="min", patience=50)
            ],
            default_root_dir=str(results_dir / "checkpoints")
        )
        
        trainer.fit(model, datamodule=datamodule)
        
        # Testing with detailed logging
        print(f"\n--- Testing Subject {subject_id} ---")
        
        model.eval()
        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()
        
        # Collect per-sample data
        sample_data = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                x = x.to(model.device)
                y = y.to(model.device)
                
                # Forward pass
                logits = model.model(x)
                
                # Get debug info from model
                get_dbg = getattr(model.model, "get_debug_batch", None)
                dbg = get_dbg() if callable(get_dbg) else {}
                
                # Compute per-sample metrics
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                correct = (preds == y).float()
                
                # Entropy
                entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
                
                # Max prob (confidence)
                max_prob = probs.max(dim=-1).values
                
                # Store per-sample data
                batch_size = x.shape[0]
                for i in range(batch_size):
                    sample_info = {
                        "batch_idx": batch_idx,
                        "sample_idx": batch_idx * test_loader.batch_size + i,
                        "true_label": int(y[i].item()),
                        "pred_label": int(preds[i].item()),
                        "correct": bool(correct[i].item()),
                        "entropy": float(entropy[i].item()),
                        "max_prob": float(max_prob[i].item()),
                        "probs": [float(p) for p in probs[i].cpu().numpy()],
                    }
                    
                    # Add alpha/lr_scale if available
                    if dbg.get("alpha") is not None:
                        alpha = dbg["alpha"]
                        if alpha.ndim > 0 and i < len(alpha):
                            sample_info["alpha"] = float(alpha[i].item())
                    
                    if dbg.get("gate_entropy") is not None:
                        gate_ent = dbg["gate_entropy"]
                        if gate_ent.ndim > 0 and i < len(gate_ent):
                            sample_info["gate_entropy"] = float(gate_ent[i].item())
                    
                    if dbg.get("ttt_lr_scale") is not None:
                        lr_scale = dbg["ttt_lr_scale"]
                        if lr_scale.ndim > 0 and i < len(lr_scale):
                            sample_info["ttt_lr_scale"] = float(lr_scale[i].item())
                    
                    sample_data.append(sample_info)
        
        # Compute subject-level statistics
        correct_samples = [s for s in sample_data if s["correct"]]
        incorrect_samples = [s for s in sample_data if not s["correct"]]
        
        subject_stats = {
            "accuracy": len(correct_samples) / len(sample_data),
            "n_samples": len(sample_data),
            "n_correct": len(correct_samples),
            "n_incorrect": len(incorrect_samples),
            
            # Overall entropy stats
            "entropy_mean": np.mean([s["entropy"] for s in sample_data]),
            "entropy_std": np.std([s["entropy"] for s in sample_data]),
            
            # Entropy by correctness
            "entropy_correct_mean": np.mean([s["entropy"] for s in correct_samples]) if correct_samples else None,
            "entropy_incorrect_mean": np.mean([s["entropy"] for s in incorrect_samples]) if incorrect_samples else None,
            
            # Alpha stats (if available)
            "alpha_mean": np.mean([s["alpha"] for s in sample_data if "alpha" in s]) if any("alpha" in s for s in sample_data) else None,
            "alpha_correct_mean": np.mean([s["alpha"] for s in correct_samples if "alpha" in s]) if any("alpha" in s for s in correct_samples) else None,
            "alpha_incorrect_mean": np.mean([s["alpha"] for s in incorrect_samples if "alpha" in s]) if any("alpha" in s for s in incorrect_samples) else None,
            
            # Per-class accuracy
            "per_class_acc": {},
        }
        
        # Per-class accuracy
        for cls in range(config["model_kwargs"]["n_classes"]):
            cls_samples = [s for s in sample_data if s["true_label"] == cls]
            if cls_samples:
                cls_correct = [s for s in cls_samples if s["correct"]]
                subject_stats["per_class_acc"][f"class_{cls}"] = len(cls_correct) / len(cls_samples)
        
        # Save results
        all_results[f"subject_{subject_id}"] = {
            "stats": subject_stats,
            "samples": sample_data,
        }
        
        print(f"  Accuracy: {subject_stats['accuracy']:.4f}")
        print(f"  Entropy (correct): {subject_stats['entropy_correct_mean']:.4f}")
        print(f"  Entropy (incorrect): {subject_stats['entropy_incorrect_mean']:.4f}")
        if subject_stats["alpha_mean"]:
            print(f"  Alpha mean: {subject_stats['alpha_mean']:.4f}")
    
    # Save all results
    output_path = results_dir / "detailed_debug.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="entropy", choices=["entropy", "no_ttt", "feature_stats"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--subjects", type=str, default="1,2,5", help="Comma-separated subject IDs")
    args = parser.parse_args()
    
    subject_ids = [int(s) for s in args.subjects.split(",")]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base config
    config = {
        "model": "TCFormer_Hybrid",
        "dataset_name": "bcic2a",
        "seed": 42,
        "max_epochs": 1000,
        "gpu_id": args.gpu_id,
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
        }
    }
    
    # Mode-specific settings
    if args.mode == "entropy":
        config["model_kwargs"]["use_dynamic_gating"] = True
        config["model_kwargs"]["gating_mode"] = "entropy"
        config["model_kwargs"]["entropy_threshold"] = 0.85
        config["model_kwargs"]["alpha_max"] = 0.5
        config["model_kwargs"]["lr_scale_max"] = 0.5
        config["model_kwargs"]["entropy_gating_in_train"] = False
        results_dir = f"results/bcic2a_debug/{timestamp}/entropy_gating"
        
    elif args.mode == "no_ttt":
        config["model_kwargs"]["use_dynamic_gating"] = False
        config["model_kwargs"]["gating_mode"] = "feature_stats"
        config["model_kwargs"]["safe_tta_disable_ttt"] = True  # Disable TTT completely
        results_dir = f"results/bcic2a_debug/{timestamp}/no_ttt"
        
    elif args.mode == "feature_stats":
        config["model_kwargs"]["use_dynamic_gating"] = True
        config["model_kwargs"]["gating_mode"] = "feature_stats"  # Effectively disables adaptive TTT
        results_dir = f"results/bcic2a_debug/{timestamp}/feature_stats"
    
    print(f"Mode: {args.mode}")
    print(f"Subjects: {subject_ids}")
    print(f"Results dir: {results_dir}")
    
    run_debug_experiment(config, results_dir, subject_ids)


if __name__ == "__main__":
    main()


