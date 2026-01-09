import os, time, yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping
# from torchviz import make_dot  # optional for graph visualization

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics  import MetricsCallback, write_summary
from utils.latency  import measure_latency
from utils.misc     import visualize_model_graph, show_gpu_info

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

# Set visible GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Define the path to the configuration directory
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# Main training and testing pipeline
def train_and_test(config):
     # Create result and checkpoints directories
    model_name = config["model"]
    dataset_name = config["dataset_name"]
    
    if "results_dir" in config and config["results_dir"]:
        result_dir = Path(config["results_dir"])
    else:
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") # Format: YYYYMMDD_HHMM (e.g., 20250517_1530)
    aug_flag = config['preprocessing']['interaug']
    gpu_id = config['gpu_id']
    dir_name = "results/{}_{}_seed-{}_aug-{}_GPU{}_{}".format(
        model_name, dataset_name, seed, aug_flag, gpu_id, timestamp
    )
    result_dir = Path(__file__).resolve().parent / dir_name

    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves"]: (result_dir / sub).mkdir(parents=True, exist_ok=True)

    # Save config to the result directory
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Retrieve model and datamodule classes
    model_cls = get_model_cls(model_name)
    datamodule_cls = get_datamodule_cls(dataset_name)

    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes
    
    # Update data_path to absolute path if not set (fallback)
    if config["preprocessing"].get("data_path") is None:
        config["preprocessing"]["data_path"] = "/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"

    # Parse subject IDs from config
    subj_cfg = config["subject_ids"]
    if subj_cfg == "all":
        subject_ids = datamodule_cls.all_subject_ids
    elif isinstance(subj_cfg, int):
        subject_ids = [subj_cfg]
    else:
        subject_ids = subj_cfg
  
    # Initialize containers for tracking metrics across subjects
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times = [], [], []
    all_confmats = []

    # Loop through each subject ID for training and testing   
    for subject_id in subject_ids:
        print(f"\n>>> Training on subject: {subject_id}")

        # Set seed for reproducibility
        seed_everything(config["seed"])
        metrics_callback = MetricsCallback()
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=50,
            verbose=True
        )
   
        # Initialize PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            devices = -1 if config.get("gpu_id", 0) == -1 else \
                [config.get("gpu_id", 0)],
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy = "auto" if config.get("gpu_id", 0) != -1 
                else DDPStrategy(find_unused_parameters=True), 
            logger=False,
            enable_checkpointing=False,
            callbacks=[metrics_callback, early_stopping]
        )

        # Instantiate datamodule and model
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(
            **config["model_kwargs"], 
            max_epochs=config["max_epochs"],
            subject_id=subject_id,          # Pass subject ID for file naming
            model_name=model_name,          # Pass model name
            results_dir=result_dir          # Pass directory for saving analysis files
        )
        
        # Count total number of model parameters
        param_count = sum(p.numel() for p in model.parameters())

        # ---------------- TRAIN ----------------
        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60) # minutes

        # ---------------- TEST -----------------
        st_test = time.time()
        test_results = trainer.test(model, datamodule)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        # ---------------- LATENCY --------------
        # Deduce input shape from one sample of the test dataset
        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)  # prepend batch dim
        # device_str = "cuda:" + str(config["gpu_id"]) if config["gpu_id"] != -1 else "cpu"
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
        response_times.append(lat_ms)  # convert to seconds for summary helper

        # ---------------- METRICS --------------
        test_accs.append(test_results[0]["test_acc"])
        test_losses.append(test_results[0]["test_loss"])
        test_kappas.append(test_results[0]["test_kappa"])

        # compute & store this subject's confusion matrix
        cm = model.test_confmat.numpy()
        all_confmats.append(cm)

        # plot per-subject if requested
        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"Confusion Matrix – Subject {subject_id}",
            )            

        # Plot and save loss and accuracy curves if available
        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                        "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                        "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")

        # Optionally save the trained model's weights
        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)
   
    # Summarize and save final results
    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
        test_accs, test_losses, test_kappas, train_times, test_times, response_times)
    
    # Save Final Accuracy Dictionary for plotting
    import json
    final_acc_dict = {f"Subject_{sid}": acc for sid, acc in zip(subject_ids, test_accs)}
    final_acc_dict["Average"] = np.mean(test_accs)
    final_acc_path = result_dir / f"final_acc_{model_name}.json"
    with open(final_acc_path, 'w') as f:
        json.dump(final_acc_dict, f, indent=4)
    
    # plot the average if requested
    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names= datamodule_cls.class_names,
            title="Average Confusion Matrix",
        )     


# Command-line argument parsing
def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="tcformer",
        help = "Name of the model to use. Options:\n"
               "tcformer, atcnet, d-atcnet, atcnet_2_0, eegnet, shallownet, basenet\n"
                "eegtcnet, eegconformer, tsseffnet, eegdeformer, sst_dpn, ctnet, mscformer"
    )        
    parser.add_argument("--dataset", type=str, default="bcic2a", 
        help="Name of the dataset to use."
                        "Options: bcic2a, bcic2b, hgd, reh_mi, bcic3"
    )
    parser.add_argument("--loso", action="store_true", default=False, 
        help="Enable subject-independent (LOSO) mode"
    )
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU device ID to use")
    
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed value (overrides config if specified)")
    parser.add_argument("--interaug", action="store_true", 
                        help="Enable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--no_interaug", action="store_true", 
                        help="Disable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help="JSON string to override model_kwargs (e.g. '{\"ttt_config\": {\"base_lr\": 0.1}}')")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory to save results (overrides default timestamp-based directory)")
    return parser.parse_args()

# ----------------------------------------------
# Main function to run the training and testing pipeline
# ----------------------------------------------
def run():
    # show_gpu_info()     # Uncomment to display GPU info
    args = parse_arguments()
     
    # load config
    config_path = os.path.join(CONFIG_DIR, args.model, f"{args.model}.yaml")
    if not os.path.exists(config_path):
    config_path = os.path.join(CONFIG_DIR, f"{args.model}.yaml") 
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found for model '{args.model}'. Tried: {config_path}")
             
    with open(config_path) as f:    
        config = yaml.safe_load(f)

    # Merge CLI model_kwargs overrides
    if args.model_kwargs:
        import json
        try:
            overrides = json.loads(args.model_kwargs)
            # Deep merge helper
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            
            deep_update(config["model_kwargs"], overrides)
            print(f"Overridden model_kwargs: {overrides}")
        except json.JSONDecodeError as e:
            print(f"Error parsing --model_kwargs: {e}")
            exit(1)

    # Adjust training parameters based on LOSO setting
    if args.loso:
        config["dataset_name"] = args.dataset + "_loso" 
        config["max_epochs"] = config["max_epochs_loso_hgd"] if args.dataset == "hgd" else config["max_epochs_loso"]
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"]["warmup_epochs_loso"]
    else:
        config["dataset_name"] = args.dataset
        config["max_epochs"] = config["max_epochs_2b"] if args.dataset == "bcic2b" else config["max_epochs"]

    config["preprocessing"] = config["preprocessing"][args.dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]
    # Select appropriate data path based on dataset
    if args.dataset == "bcic2b":
        config["preprocessing"]["data_path"] = config.get("data_path_2b", config.get("data_path", None))
    else:
    config["preprocessing"]["data_path"] = config.get("data_path", None)
    # Override interaug if specified
    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]
    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id
    # Override seed if specified
    if args.seed is not None:
        config["seed"] = args.seed

    # set to True to plot confusion matrices
    config["plot_cm_per_subject"] = True # set to True to plot per-subject confusion matrices
    config["plot_cm_average"]     = True # set to True to plot average confusion matrix

    # Set results_dir from args if provided
    if args.results_dir:
        config["results_dir"] = args.results_dir

    train_and_test(config)

if __name__ == "__main__":
    run()
