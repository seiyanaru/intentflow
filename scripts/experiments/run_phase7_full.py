
import os
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Ensure local imports work same as train_pipeline.py
# Assuming this script is run from the project root or script dir
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(PROJECT_ROOT, "intentflow/offline"))

from train_pipeline import train_and_test, CONFIG_DIR

def run_full_experiment():
    model_name = "tcformer_otta"
    dataset_name = "bcic2a"
    
    # Path to config
    config_path = os.path.join(CONFIG_DIR, model_name, f"{model_name}.yaml")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Override for Full Experiment
    config["subject_ids"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["dataset_name"] = dataset_name
    
    # Handle preprocessing config selection logic manually as in pipeline
    if dataset_name in config["preprocessing"]:
        raw_preproc = config["preprocessing"][dataset_name]
        raw_preproc["z_scale"] = config.get("z_scale", True)
        raw_preproc["interaug"] = config.get("interaug", True)
        raw_preproc["data_path"] = config.get("data_path", None)
        config["preprocessing"] = raw_preproc

    # Set Results Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir_name = f"phase7_conservative_{timestamp}"
    config["results_dir"] = os.path.join(PROJECT_ROOT, "intentflow/offline/results", results_dir_name)
    
    config["gpu_id"] = 0
    
    print(f">>> Starting Phase 7 Full Experiment (Subjects 1-9) <<<")
    print(f"Results will be saved to: {config['results_dir']}")
    
    train_and_test(config)

if __name__ == "__main__":
    run_full_experiment()
