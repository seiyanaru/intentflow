import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import glob

# Set style for paper figures
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

def load_history(result_dir, subject_id, model_name):
    # Search recursively for the history file
    pattern = os.path.join(result_dir, "**", f"history_s{subject_id}_{model_name}.json")
    # Also handle the timestamped folder structure
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        # try case insensitive or partial match on model name
        pattern = os.path.join(result_dir, "**", f"history_s{subject_id}_*{model_name}*.json")
        matches = glob.glob(pattern, recursive=True)

    if matches:
        # Use the most recent file if multiple found
        path = sorted(matches, key=os.path.getmtime, reverse=True)[0]
        with open(path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: history file for S{subject_id} {model_name} not found in {result_dir}")
        return None

def load_features(result_dir, subject_id, model_name):
    pattern = os.path.join(result_dir, "**", f"features_s{subject_id}_{model_name}.npz")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        pattern = os.path.join(result_dir, "**", f"features_s{subject_id}_*{model_name}*.npz")
        matches = glob.glob(pattern, recursive=True)

    if matches:
        path = sorted(matches, key=os.path.getmtime, reverse=True)[0]
        data = np.load(path)
        return data['features'], data['labels']
    else:
        print(f"Warning: features file for S{subject_id} {model_name} not found in {result_dir}")
        return None, None

def load_logits(result_dir, subject_id, model_name):
    pattern = os.path.join(result_dir, "**", f"logits_s{subject_id}_{model_name}.npy")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        pattern = os.path.join(result_dir, "**", f"logits_s{subject_id}_*{model_name}*.npy")
        matches = glob.glob(pattern, recursive=True)

    if matches:
        path = sorted(matches, key=os.path.getmtime, reverse=True)[0]
        return np.load(path)
    else:
        print(f"Warning: logits file for S{subject_id} {model_name} not found in {result_dir}")
        return None

def load_final_acc(result_dir, model_name):
    pattern = os.path.join(result_dir, "**", f"final_acc_{model_name}.json")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        pattern = os.path.join(result_dir, "**", f"final_acc_*{model_name}*.json")
        matches = glob.glob(pattern, recursive=True)

    if matches:
        path = sorted(matches, key=os.path.getmtime, reverse=True)[0]
        with open(path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: final_acc file for {model_name} not found in {result_dir}")
        return None

def plot_tsne_for_subject(result_dir, save_dir, subject_id):
    models = ["TCFormer", "TCFormer_TTT", "TCFormer_Hybrid"]
    display_names = ["Base", "TTT", "Hybrid"]
    
    plt.figure(figsize=(18, 5))
    
    for i, (model_name, disp_name) in enumerate(zip(models, display_names)):
        features, labels = load_features(result_dir, subject_id, model_name)
        if features is None:
            continue
            
        tsne = TSNE(n_components=2, random_state=42)
        
        # Flatten features if they are 3D [N, F, T] -> [N, F*T] or similar
        # Or mean pool
        if features.ndim > 2:
             # Assuming [N, F, 1] or [N, F, T]
             # Flatten starting from dimension 1
             features = features.reshape(features.shape[0], -1)

        features_2d = tsne.fit_transform(features)
        
        plt.subplot(1, 3, i+1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        if i == 2: # Legend only on last one to save space
            plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title(f"{disp_name} (S{subject_id})")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tsne_subj{subject_id}_comparison.png"))
    plt.close()

def plot_entropy_for_subject(result_dir, save_dir, subject_id):
    models = ["TCFormer", "TCFormer_TTT", "TCFormer_Hybrid"]
    display_names = ["Base", "TTT", "Hybrid"]
    colors = ['skyblue', 'orange', 'salmon']
    
    def calc_entropy(logits):
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
        return entropy

    plt.figure(figsize=(8, 6))
    
    for model_name, disp_name, color in zip(models, display_names, colors):
        logits = load_logits(result_dir, subject_id, model_name)
        if logits is None:
            continue
        ent = calc_entropy(logits)
        sns.kdeplot(ent, color=color, label=disp_name, fill=True, alpha=0.3)
        
    plt.xlabel('Prediction Entropy')
    plt.ylabel('Density')
    plt.title(f"Entropy Distribution (S{subject_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"entropy_subj{subject_id}_comparison.png"))
    plt.close()

def plot_confmat_for_subject(result_dir, save_dir, subject_id):
    models = ["TCFormer", "TCFormer_TTT", "TCFormer_Hybrid"]
    display_names = ["Base", "TTT", "Hybrid"]
    
    plt.figure(figsize=(18, 5))
    
    from sklearn.metrics import confusion_matrix
    
    for i, (model_name, disp_name) in enumerate(zip(models, display_names)):
        logits = load_logits(result_dir, subject_id, model_name)
        _, labels = load_features(result_dir, subject_id, model_name)
        
        if logits is None or labels is None:
            continue
            
        preds = np.argmax(logits, axis=1)
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                    xticklabels=['LH', 'RH', 'Foot', 'Tongue'],
                    yticklabels=['LH', 'RH', 'Foot', 'Tongue'])
        plt.title(f"{disp_name} (S{subject_id})")
        if i == 0:
            plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confmat_subj{subject_id}_comparison.png"))
    plt.close()

def plot_learning_curve_for_subject(result_dir, save_dir, subject_id):
    models = ["TCFormer", "TCFormer_TTT", "TCFormer_Hybrid"]
    display_names = ["Base", "TTT", "Hybrid"]
    colors = ['skyblue', 'orange', 'salmon']
    
    plt.figure(figsize=(10, 5))
    
    for model_name, disp_name, color in zip(models, display_names, colors):
        hist = load_history(result_dir, subject_id, model_name)
        if hist is None:
            continue
            
        epochs = [e['epoch'] for e in hist]
        val_acc = [e['val_acc'] for e in hist]
        
        plt.plot(epochs, val_acc, label=disp_name, color=color)
        
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f"Learning Curve (S{subject_id})")
    plt.legend()
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"learning_curve_subj{subject_id}_comparison.png"))
    plt.close()

def plot_acc_comparison_bar(result_dir, save_dir):
    models = ["TCFormer", "TCFormer_TTT", "TCFormer_Hybrid"]
    display_names = ["Base", "TTT", "Hybrid"]
    colors = ['skyblue', 'orange', 'salmon']
    
    subjects = [f"S{i}" for i in range(1, 10)]
    all_accs = []
    
    for model_name in models:
        acc_data = load_final_acc(result_dir, model_name)
        if acc_data is None:
            all_accs.append([0]*9)
        else:
            all_accs.append([acc_data.get(f"Subject_{i}", 0) * 100 for i in range(1, 10)])
            
    x = np.arange(len(subjects))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    for i, (accs, name, color) in enumerate(zip(all_accs, display_names, colors)):
        plt.bar(x + (i-1)*width, accs, width, label=name, color=color)
        
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison per Subject')
    plt.xticks(x, subjects)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_comparison_bar.png"))
    plt.close()

def plot_latency(save_dir):
    latency_data = {'Base': 5.2, 'TTT': 6.8, 'Hybrid': 7.6} # Dummy
    
    models = list(latency_data.keys())
    times = list(latency_data.values())
    colors = ['skyblue', 'orange', 'salmon']
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, times, color=colors, width=0.5)
    
    plt.ylabel('Inference Latency per Sample (ms)')
    plt.title('Latency Comparison')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval} ms', ha='center', va='bottom')
        
    plt.ylim(0, max(times) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "latency_bar.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--save_dir", type=str, default="fig")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Generating Figures for All Subjects (1-9)...")
    for subj_id in range(1, 10):
        print(f"Processing Subject {subj_id}...")
        try:
            plot_tsne_for_subject(args.results_dir, args.save_dir, subj_id)
            plot_entropy_for_subject(args.results_dir, args.save_dir, subj_id)
            plot_confmat_for_subject(args.results_dir, args.save_dir, subj_id)
            plot_learning_curve_for_subject(args.results_dir, args.save_dir, subj_id)
        except Exception as e:
            print(f"Error processing S{subj_id}: {e}")

    print("Generating Summary Figures...")
    plot_acc_comparison_bar(args.results_dir, args.save_dir)
    plot_latency(args.save_dir)
    
    print("Done! Check the 'fig/' directory.")

if __name__ == "__main__":
    main()

