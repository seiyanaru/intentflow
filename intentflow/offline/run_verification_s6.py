
import os
import yaml
import sys
from pathlib import Path

# Ensure local imports work same as train_pipeline.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(PROJECT_ROOT, "intentflow/offline"))

from train_pipeline import train_and_test, CONFIG_DIR

def run_s1():
    model_name = "tcformer_otta"
    config_path = os.path.join(CONFIG_DIR, model_name, f"{model_name}.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Override for verification
    config["subject_ids"] = [6] # Verification for Subject 6
    
    # Set dataset name (required by pipeline logic if not in yaml root correctly or overwritten)
    # The config has "dataset_name: """ at top level, but train_pipeline logic expects it to be populated.
    # In run.py it sets it from CLI args. We must do same here.
    config["dataset_name"] = "bcic2a"
    
    # Also need to make sure preprocessing config is set correctly (pipeline does this)
    # The pipeline line 268: config["preprocessing"] = config["preprocessing"][args.dataset]
    # We must start with the full config, so pipeline can select the right sub-config.
    # Actually wait, train_pipeline.py:train_and_test() expects flattened config if it was already processed?
    # No, train_and_test receives 'config'.
    # BUT, train_pipeline.py:run() does some pre-processing of 'config' before calling train_and_test().
    # Specifically:
    # 1. config["dataset_name"] = args.dataset
    # 2. config["preprocessing"] = config["preprocessing"][args.dataset]
    # 3. config["preprocessing"]["z_scale"] = config["z_scale"]
    # 4. config["preprocessing"]["data_path"] ...
    
    # We need to replicate this setup logic manually since we are calling train_and_test directly.
    dataset_name = "bcic2a"
    config["dataset_name"] = dataset_name
    
    # Handle preprocessing config selection
    if dataset_name in config["preprocessing"]:
        raw_preproc = config["preprocessing"][dataset_name]
        # Copy global settings into it if needed
        raw_preproc["z_scale"] = config.get("z_scale", True)
        raw_preproc["interaug"] = config.get("interaug", True)
        raw_preproc["data_path"] = config.get("data_path", None) # Default
        
        config["preprocessing"] = raw_preproc
    
    # Initialize "model_kwargs" defaults (n_channels/n_classes) are done inside train_and_test via datamodule_cls
    
    # Use a specific results directory
    timestamp = "verification_s1"
    config["results_dir"] = os.path.join(PROJECT_ROOT, "intentflow/offline/results/tcformer_otta_verification_s6")
    
    # Ensure correct GPU
    config["gpu_id"] = 0
    
    # Printing config for debug
    print(f"Dataset Name: {config['dataset_name']}")
    
    print(">>> Running Verification Experiment for Subject 1 (Neuro-Gated OTTA Phase 7) <<<")
    train_and_test(config)

if __name__ == "__main__":
    run_s1()
