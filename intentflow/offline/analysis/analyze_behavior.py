"""
Analyze behavior of TCFormer and TCFormer_TTT models.
Visualizes feature distribution (t-SNE) and entropy.
"""
import os
import sys
import argparse
import glob
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import entropy
from pathlib import Path

# Adjust path to allow imports from intentflow/offline
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import directly from models using intentflow structure
# Assumes we are running from root or intentflow is in path
try:
    from intentflow.offline.models.tcformer.tcformer import TCFormer
    from intentflow.offline.models.tcformer_ttt.tcformer_ttt import TCFormerTTT
    from intentflow.offline.utils.get_datamodule_cls import get_datamodule_cls
except ImportError:
    # Fallback if running from a different context where intentflow is not a package
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.tcformer.tcformer import TCFormer
    from models.tcformer_ttt.tcformer_ttt import TCFormerTTT
    from utils.get_datamodule_cls import get_datamodule_cls

def load_config(result_dir):
    config_path = os.path.join(result_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def find_checkpoint(result_dir, subject_id):
    # Checkpoints are usually in result_dir/checkpoints/subject_{id}_model.ckpt
    # Or just *.ckpt search
    ckpt_pattern = os.path.join(result_dir, "checkpoints", f"subject_{subject_id}*.ckpt")
    ckpts = glob.glob(ckpt_pattern)
    if not ckpts:
        # Try recursive search
        ckpts = glob.glob(os.path.join(result_dir, "**", f"subject_{subject_id}*.ckpt"), recursive=True)
    
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for subject {subject_id} in {result_dir}")
    return ckpts[0] # Return first match

def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    
    all_features = []
    all_logits = []
    all_labels = []
    
    # Define hook to capture features
    features_storage = []
    def hook(module, input, output):
        # input is a tuple (x,)
        # x shape: (B, d_model, 1) usually
        feat = input[0].detach().cpu()
        if feat.ndim == 3:
            feat = feat.squeeze(-1) # (B, d_model)
        features_storage.append(feat)
        
    # Register hook on the classifier
    if hasattr(model.model, 'tcn_head'):
        handle = model.model.tcn_head.classifier.register_forward_hook(hook)
    elif hasattr(model.model, 'classifier'):
         handle = model.model.classifier.register_forward_hook(hook)
    else:
        # Fallback: hook the last module
        handle = list(model.model.children())[-1].register_forward_hook(hook)

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            
            all_logits.append(logits.detach().cpu())
            all_labels.append(y)
            
    handle.remove()
    
    all_features = torch.cat(features_storage, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate probabilities and entropy
    probs = torch.softmax(all_logits, dim=1).numpy()
    ents = entropy(probs, axis=1)
    
    return all_features, ents, all_labels

def main():
    parser = argparse.ArgumentParser(description="Analyze behavior of Base vs TTT models")
    parser.add_argument("--subject_id", type=int, default=6, help="Subject ID to analyze")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing Base model results")
    parser.add_argument("--ttt_dir", type=str, required=True, help="Directory containing TTT model results")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--out_dir", type=str, default="analysis_outputs", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"Analyzing Subject {args.subject_id}...")
    
    # 1. Load Configurations
    base_config = load_config(args.base_dir)
    ttt_config = load_config(args.ttt_dir)
    
    # Get Datamodule class to infer channels/classes
    datamodule_cls = get_datamodule_cls(base_config["dataset_name"])
    n_channels = datamodule_cls.channels
    n_classes = datamodule_cls.classes

    # Instantiate Base Model
    print("Loading Base Model...")
    base_kwargs = base_config["model_kwargs"]
    base_kwargs["n_channels"] = n_channels
    base_kwargs["n_classes"] = n_classes
    
    model_base = TCFormer(**base_kwargs)
    ckpt_base = find_checkpoint(args.base_dir, args.subject_id)
    checkpoint = torch.load(ckpt_base, map_location="cpu")
    model_base.load_state_dict(checkpoint["state_dict"])
    
    # Instantiate TTT Model
    print("Loading TTT Model...")
    ttt_kwargs = ttt_config["model_kwargs"]
    ttt_kwargs["n_channels"] = n_channels
    ttt_kwargs["n_classes"] = n_classes
    
    model_ttt = TCFormerTTT(**ttt_kwargs)
    ckpt_ttt = find_checkpoint(args.ttt_dir, args.subject_id)
    checkpoint = torch.load(ckpt_ttt, map_location="cpu")
    model_ttt.load_state_dict(checkpoint["state_dict"])
    
    # 2. Prepare Data
    print("Loading Data...")
    # Use preprocessing config from base
    datamodule = datamodule_cls(base_config["preprocessing"], subject_id=args.subject_id)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    
    # 3. Run Inference
    print("Running Inference on Base...")
    feat_base, ent_base, labels = run_inference(model_base, test_loader, device)
    
    print("Running Inference on TTT...")
    feat_ttt, ent_ttt, _ = run_inference(model_ttt, test_loader, device)
    
    # 4. Visualization
    
    # t-SNE
    print("Computing t-SNE (Individually due to potential dimension mismatch)...")
    
    # Base Model t-SNE
    tsne_base = TSNE(n_components=2, random_state=42)
    emb_base = tsne_base.fit_transform(feat_base)
    
    # TTT Model t-SNE
    tsne_ttt = TSNE(n_components=2, random_state=42)
    emb_ttt = tsne_ttt.fit_transform(feat_ttt)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Base Model t-SNE (Subj {args.subject_id})")
    plt.scatter(emb_base[:, 0], emb_base[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    
    plt.subplot(1, 2, 2)
    plt.title(f"TTT Model t-SNE (Subj {args.subject_id})")
    plt.scatter(emb_ttt[:, 0], emb_ttt[:, 1], c=labels, cmap='tab10', alpha=0.6, s=10)
    
    plt.savefig(os.path.join(args.out_dir, f"tsne_subj{args.subject_id}.png"))
    plt.close()
    
    # Entropy Histogram
    print("Plotting Entropy Histogram...")
    plt.figure(figsize=(8, 6))
    plt.hist(ent_base, bins=30, alpha=0.5, label='Base Model', color='blue', density=True)
    plt.hist(ent_ttt, bins=30, alpha=0.5, label='TTT Model', color='orange', density=True)
    plt.xlabel('Prediction Entropy')
    plt.ylabel('Density')
    plt.title(f"Entropy Distribution (Subj {args.subject_id})")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, f"entropy_hist_subj{args.subject_id}.png"))
    plt.close()
    
    print(f"Analysis complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    main()
