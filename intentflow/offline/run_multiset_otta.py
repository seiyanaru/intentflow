
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

import sys
sys.path.append(os.getcwd())

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.metrics import MetricsCallback
from utils.seed import seed_everything
from models.pmax_sal_otta import PmaxSAL_OTTA

# Datasets to run
DATASETS = ["hgd"]
TARGET_SAL_THRESHOLD = 0.98

def run_multiset_experiment():
    # Base config path
    config_path = "configs/tcformer_otta/tcformer_otta.yaml"
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)
    
    # Create main result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    main_result_dir = Path(f"results/multiset_otta_experiment_hgd_{timestamp}")
    main_result_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = main_result_dir / "summary_results.txt"
    with open(summary_file, "w") as f:
        f.write("Dataset,Subject,Baseline_Acc,OTTA_Acc_0.98,Improvement\n")
        
    for dataset_name in DATASETS:
        print(f"\n\n{'='*40}")
        print(f"Starting Experiment for Dataset: {dataset_name}")
        print(f"{'='*40}")
        
        # Prepare Dataset Specific Config
        config = base_config.copy()
        
        # 1. Dataset Class & Params
        datamodule_cls = get_datamodule_cls(dataset_name)
        
        # 2. Config Overrides (logic from train_pipeline.py)
        config["dataset_name"] = dataset_name
        config["model_kwargs"]["n_channels"] = datamodule_cls.channels
        config["model_kwargs"]["n_classes"] = datamodule_cls.classes
        
        # Preprocessing
        config["preprocessing"] = base_config["preprocessing"][dataset_name]
        config["preprocessing"]["z_scale"] = base_config["z_scale"]
        
        # Disable interaug for HGD and BCIC 2b as per scripts
        config["preprocessing"]["interaug"] = False 
        
        # Data Path
        if dataset_name == "bcic2b":
            config["preprocessing"]["data_path"] = base_config.get("data_path_2b", base_config.get("data_path"))
        else: # hgd
             # Use None to let MOABB/loader handle it default
            config["preprocessing"]["data_path"] = None 
            
        # Max Epochs override
        if dataset_name == "bcic2b":
            config["max_epochs"] = base_config["max_epochs_2b"]
        elif dataset_name == "hgd":
             # HGD uses 'max_epochs' default (500) ? 
             # train_pipeline says: config["max_epochs"] = config["max_epochs_loso_hgd"] IF LOSO.
             # but we are NOT in LOSO mode (Within-Subject).
             # So use default 500.
             pass
             
        # Subjects
        if dataset_name == "bcic2b":
            subjects = range(1, 10) # 1-9
        elif dataset_name == "hgd":
            subjects = range(1, 15) # 1-14
        
        # Model Class
        model_cls = get_model_cls("tcformer_otta")
        
        for subject_id in subjects:
            print(f"\n>>> Processing {dataset_name} - Subject {subject_id}")
            
            # Directory for this subject
            sub_res_dir = main_result_dir / dataset_name / f"subject_{subject_id}"
            sub_res_dir.mkdir(parents=True, exist_ok=True)
            
            # Seed
            seed_everything(0) # Use seed 0
            
            # --- TRAINING ---
            datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
            
            model = model_cls(
                **config["model_kwargs"],
                max_epochs=config["max_epochs"],
                subject_id=subject_id,
                model_name="tcformer_otta",
                results_dir=sub_res_dir
            )
            
            # Callbacks
            metrics_callback = MetricsCallback()
            early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=False)
            
            trainer = Trainer(
                max_epochs=config["max_epochs"],
                devices=1,
                accelerator="gpu",
                logger=False,
                enable_checkpointing=True, # Need to save best model
                default_root_dir=sub_res_dir,
                callbacks=[metrics_callback, early_stopping]
            )
            
            print("Training...")
            trainer.fit(model, datamodule=datamodule)
            
            # Load best checkpoint for testing
            best_ckpt_path = trainer.checkpoint_callback.best_model_path
            print(f"Loading best checkpoint: {best_ckpt_path}")
            
            # --- TEST 1: BASELINE (No OTTA) ---
            print("Testing Baseline (No OTTA)...")
            
            # Re-load model from checkpoint to ensure clean state
            # (Though Trainer.test loads best weights automatically if we pass ckpt_path='best')
            # But we need to modify model properties (enable_otta=False)
            
            # We can use the existing 'model' object, load weights, and mod properties.
            checkpoint = torch.load(best_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            # Initialize OTTA explicitly if None (though fit usually initializes it? No, PmaxSAL_OTTA likely init in __init__)
            # But compute_source_prototypes is needed. It runs in on_test_start.
            # However, for Baseline we disable OTTA, so prototypes don't matter much, but let's be safe.
            model.enable_otta = False
            if model.otta is not None:
                model.otta.enable_adaptation = False
            
            trainer_base = Trainer(accelerator="gpu", devices=1, logger=False, enable_checkpointing=False)
            base_res = trainer_base.test(model, datamodule=datamodule)
            base_acc = base_res[0]['test_acc']
            print(f"Baseline Acc: {base_acc:.4f}")
            
            # --- TEST 2: OTTA (SAL=0.98) ---
            print(f"Testing OTTA (SAL={TARGET_SAL_THRESHOLD})...")
            
            # Reload to reset any state (BN stats shouldn't change in Baseline test, but good practice)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            # Setup OTTA
            model.enable_otta = True
            model.sal_threshold = TARGET_SAL_THRESHOLD
            
            # We rely on on_test_start to calculate prototypes.
            # But we must ensure train_dataloader_ref is set!
            datamodule.setup()
            model.train_dataloader_ref = datamodule.train_dataloader()
            
            # If model.otta is already init, update params
            if model.otta is not None:
                model.otta.sal_threshold = TARGET_SAL_THRESHOLD
                model.otta.enable_adaptation = True
                model.otta.reset_stats()
            
            trainer_otta = Trainer(accelerator="gpu", devices=1, logger=False, enable_checkpointing=False)
            otta_res = trainer_otta.test(model, datamodule=datamodule)
            otta_acc = otta_res[0]['test_acc']
            print(f"OTTA Acc: {otta_acc:.4f}")
            
            # Save Result
            res_str = f"{dataset_name},{subject_id},{base_acc:.4f},{otta_acc:.4f},{otta_acc-base_acc:+.4f}\n"
            with open(summary_file, "a") as f:
                f.write(res_str)
                
    print(f"\nAll experiments completed. Results saved to {summary_file}")

if __name__ == "__main__":
    run_multiset_experiment()
