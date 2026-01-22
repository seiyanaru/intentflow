
import os
import argparse
import sys
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Add current directory to path
sys.path.append(os.getcwd())

from utils.get_model_cls import get_model_cls
from utils.get_datamodule_cls import get_datamodule_cls

def debug_otta(
    checkpoint_dir,
    dataset="bcic2a",
    gpu_id=0,
    seed=0
):
    """
    Run OTTA debugging using existing checkpoints.
    """
    # Load base config
    config_path = "configs/tcformer_otta/tcformer_otta.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load preprocessing config (simplified logic from train_pipeline.py)
    # We assume bcic2a for now as per requirements
    # Ideally we'd load defaults from train_pipeline logic, but manual is faster here
    config["preprocessing"] = config["preprocessing"][dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]
    config["preprocessing"]["data_path"] = config.get("data_path", None)
    config["preprocessing"]["interaug"] = True # Force interaug as per previous run
    
    # Setup devices
    if gpu_id != -1:
        accelerator = "gpu"
        devices = [gpu_id]
    else:
        accelerator = "cpu"
        devices = "auto"
        
    # Get classes
    # We know the model is tcformer_otta
    model_name = "tcformer_otta"
    model_cls = get_model_cls(model_name)
    
    # Dataset class logic from train_pipeline.py
    # "bcic2a" -> BCIC2aDataModule
    # We can use get_datamodule_cls logic if available, or import directly
    # Reuse utils if possible
    datamodule_cls = get_datamodule_cls("bcic2a") 
    
    subjects = range(1, 10) # 1 to 9
    
    print(f"Starting Debug OTTA with checkpoints from: {checkpoint_dir}")
    
    for subject_id in subjects:
        print(f"\n>>> Processing Subject {subject_id}")
        
        # 1. Initialize DataModule
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        
        # 2. Initialize Model (Fresh)
        # We need to construct it with same args
        model_kwargs = config["model_kwargs"]
        # Override OTTA params if needed (ensure they are enabled)
        model_kwargs["enable_otta"] = True
        model_kwargs["pmax_threshold"] = 0.7
        model_kwargs["sal_threshold"] = 0.5
        
        model = model_cls(
            n_classes=4, # Hardcoded for BCIC 2a
            **model_kwargs,
            max_epochs=config["max_epochs"],
            subject_id=subject_id,
            model_name=model_name,
            results_dir=checkpoint_dir.replace("/checkpoints", "") # Save to same results dir
        )
        
        # 3. Load Checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"subject_{subject_id}_model.ckpt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping...")
            continue
            
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False) 
        
        # --- BASELINE EVALUATION (No OTTA) ---
        print(f"[DebugOTTA] Evaluating Baseline (No OTTA)...")
        model.enable_otta = False # Disable OTTA
        model.otta.enable_adaptation = False
        
        trainer_base = Trainer(accelerator=accelerator, devices=devices, logger=False, enable_checkpointing=False)
        base_results = trainer_base.test(model, datamodule=datamodule, verbose=False)
        base_acc = base_results[0]['test_acc']
        print(f"[Baseline] Subject {subject_id} Acc: {base_acc:.4f}")
        
        
        # --- OTTA EVALUATION ---
        print("[DebugOTTA] Setting train_dataloader_ref for prototype computation and Enabling OTTA...")
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        model.train_dataloader_ref = train_loader
        model.enable_otta = True
        model.otta.enable_adaptation = True
        
        # 4. Trainer
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
            enable_checkpointing=False
        )
        
        # 5. Test
        otta_results = trainer.test(model, datamodule=datamodule, verbose=False)
        otta_acc = otta_results[0]['test_acc']
        print(f"[OTTA] Subject {subject_id} Acc: {otta_acc:.4f}")
        print(f"[Diff] Improvement: {otta_acc - base_acc:+.4f}")
        
        # Save comparison results
        comp_file = os.path.join(checkpoint_dir.replace("/checkpoints", ""), "comparison_seed0.txt")
        with open(comp_file, "a") as f:
            f.write(f"Subject {subject_id}: Baseline={base_acc:.4f}, OTTA={otta_acc:.4f}, Diff={otta_acc - base_acc:+.4f}\n")


if __name__ == "__main__":
    # Hardcoded path to the specific result dir we want to reuse
    RESULT_DIR = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/results/tcformer_otta_bcic2a_seed-0_aug-True_GPU0_20260122_1704/checkpoints"
    
    debug_otta(RESULT_DIR)
