
import os
import argparse
import sys
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

sys.path.append(os.getcwd())

from utils.get_model_cls import get_model_cls
from utils.get_datamodule_cls import get_datamodule_cls

def run_ablation(
    checkpoint_dir,
    dataset="bcic2a",
    gpu_id=0
):
    """
    Run Ablation Study on SAL Thresholds.
    Target Thresholds: [0.5, 0.9, 0.92, 0.95, 0.98]
    """
    # Load base config
    config_path = "configs/tcformer_otta/tcformer_otta.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    config["preprocessing"] = config["preprocessing"][dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]
    config["preprocessing"]["data_path"] = config.get("data_path", None)
    config["preprocessing"]["interaug"] = True
    
    # Setup devices
    if gpu_id != -1:
        accelerator = "gpu"
        devices = [gpu_id]
    else:
        accelerator = "cpu"
        devices = "auto"
        
    model_name = "tcformer_otta"
    model_cls = get_model_cls(model_name)
    datamodule_cls = get_datamodule_cls("bcic2a") 
    
    subjects = range(1, 10)
    
    # List of thresholds to test
    thresholds = [0.5, 0.9, 0.92, 0.95, 0.98]
    
    results_file = os.path.join(checkpoint_dir.replace("/checkpoints", ""), "ablation_results.txt")
    with open(results_file, "w") as f:
        f.write("Subject,Baseline,SAL_0.5,SAL_0.9,SAL_0.92,SAL_0.95,SAL_0.98\n")
    
    print(f"Starting Ablation Study with checkpoints from: {checkpoint_dir}")
    
    for subject_id in subjects:
        print(f"\n>>> Processing Subject {subject_id}")
        
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        
        # Load Model Wrapper (used to load weights)
        model_kwargs = config["model_kwargs"]
        
        model = model_cls(
            n_classes=4,
            **model_kwargs,
            max_epochs=config["max_epochs"],
            subject_id=subject_id,
            model_name=model_name,
            results_dir=checkpoint_dir.replace("/checkpoints", "")
        )
        
        ckpt_path = os.path.join(checkpoint_dir, f"subject_{subject_id}_model.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping...")
            continue
            
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False) 
        
        # Initialize OTTA and Compute Prototypes
        # We need to do this FIRST to ensure model.otta is initialized
        print("[Ablation] Setting up OTTA & Computing Prototypes...")
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        
        # Manually initialize if needed, or rely on on_test_start logic
        # But for Baseline, we want OTTA disabled.
        # Let's manually trigger initialization so we can then disable it.
        model.train_dataloader_ref = train_loader
        # We need to force initialization. TCFormerOTTA initializes in on_test_start.
        # But we want to manipulate it before test.
        # Let's just create it manually if it's None.
        if model.otta is None:
             from models.pmax_sal_otta import PmaxSAL_OTTA
             model.otta = PmaxSAL_OTTA(
                 model=model.model,
                 n_classes=model.n_classes,
                 pmax_threshold=model.pmax_threshold,
                 sal_threshold=model.sal_threshold,
                 enable_adaptation=True
             )
             # Compute prototypes
             device = torch.device(f"cuda:{gpu_id}" if gpu_id != -1 else "cpu")
             model.to(device)
             model.otta.compute_source_prototypes(train_loader, device=device)

        # 1. BASELINE (No OTTA)
        print(f"--- Evaluating Baseline (No OTTA) ---")
        model.enable_otta = False
        model.otta.enable_adaptation = False
        
        trainer_base = Trainer(accelerator=accelerator, devices=devices, logger=False, enable_checkpointing=False)
        base_res = trainer_base.test(model, datamodule=datamodule, verbose=False)
        base_acc = base_res[0]['test_acc']
        print(f"[Baseline] Acc: {base_acc:.4f}")
        
        # Prepare for OTTA
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        model.train_dataloader_ref = train_loader
        model.enable_otta = True
        model.otta.enable_adaptation = True
        
        subject_results = [base_acc]
        
        # 2. Iterate Thresholds
        for thresh in thresholds:
            print(f"--- Evaluating SAL Threshold: {thresh} ---")
            
            # Update threshold
            model.otta.sal_threshold = thresh
            model.sal_threshold = thresh # Update wrapper too just in case
            
            # Reset OTTA stats just in case
            model.otta.reset_stats()
            # Need to re-compute source prototypes? 
            # No, prototypes are fixed (computed once in on_test_start).
            # But wait, trainer.test() calls on_test_start each time.
            # And on_test_start calls compute_source_prototypes.
            # So it will re-compute them each time. This is fine, but slightly inefficient.
            # However, since we re-create Trainer each time? No, we reuse model instance.
            # Reusing model instance with trainer.test() is fine.
            
            trainer_otta = Trainer(accelerator=accelerator, devices=devices, logger=False, enable_checkpointing=False)
            otta_res = trainer_otta.test(model, datamodule=datamodule, verbose=False)
            otta_acc = otta_res[0]['test_acc']
            print(f"[SAL {thresh}] Acc: {otta_acc:.4f}")
            subject_results.append(otta_acc)
            
        # Save Row
        res_str = f"{subject_id}," + ",".join([f"{x:.4f}" for x in subject_results]) + "\n"
        with open(results_file, "a") as f:
            f.write(res_str)

if __name__ == "__main__":
    RESULT_DIR = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/tcformer_otta_bcic2a_seed-0_aug-True_GPU0_20260122_1704/checkpoints"
    run_ablation(RESULT_DIR)
