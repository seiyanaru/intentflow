import numpy as np
import glob
import os
import torch


class VirtualOnlineNormalizer:
    def __init__(self, momentum=0.1):
        self.momentum = momentum
        self.running_mean = 0.0
        self.running_var = 0.0001
        self.count = 0
        
    def forward(self, x_np):
        x = torch.from_numpy(x_np).float()
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False) if x.numel() > 1 else 0.0
        
        # Update (Always update)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.item()
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.item()
        self.count += 1
        
        # Calc Z
        std = np.sqrt(self.running_var + 1e-6)
        z = (x_np - self.running_mean) / std
        z = np.clip(z, -3.0, 3.0)
        return z

def simulate_conservative_neuro_gating():
    results_dir = "intentflow/offline/results/tcformer_otta_bcic2a_seed-0_aug-True_GPU0_20260129_2126" # Hardcoded latest
    files = sorted(glob.glob(os.path.join(results_dir, "otta_stats_s*_tcformer_otta.npz")))
    
    print("=== Virtual Neuro-Conservative Gating Simulation (Warmup=0) ===")
    
    total_samples = 0
    total_flips = 0
    
    for f in files:
        data = np.load(f)
        if 'neuro_score' not in data:
            continue
            
        sal = data['sal']
        neuro_score = data['neuro_score']
        
        # Split into "batches" of 144 (approx) to match online behavior
        # Or just feed sequentially if we want.
        # But wait, the npz has all data concatenated.
        # Let's process in chunks of 144 to mimic batch size.
        
        batch_size = 144
        n_samples = len(sal)
        subject_flips = 0
        
        normalizer = VirtualOnlineNormalizer()
        
        for i in range(0, n_samples, batch_size):
            batch_sal = sal[i:i+batch_size]
            batch_ns = neuro_score[i:i+batch_size]
            
            # 1. Normalize
            z_scores = normalizer.forward(batch_ns)
            
            # 2. Threshold (aligned with Conservative Neuro-Gating in pmax_sal_otta.py)
            base_sal_th = 0.5
            beta = 0.1
            
            negative_z = np.maximum(-z_scores, 0) # ReLU(-Z)
            modifier = beta * negative_z
            
            dynamic_th = base_sal_th + modifier
            
            # 3. Gating (SAL-only approximation; full model also checks Pmax and Energy)
            should_adapt = batch_sal > dynamic_th
            
            subject_flips += np.sum(should_adapt)
            
        flip_rate = (subject_flips / n_samples) * 100
        sid = os.path.basename(f).split('_')[2]
        print(f"{sid}: Virtual Flip Rate = {flip_rate:.2f}% (Mean Z: {normalizer.running_mean:.4f})")
        
        total_samples += n_samples
        total_flips += subject_flips

    print(f"Overall Virtual Flip Rate: {(total_flips/total_samples)*100:.2f}%")


if __name__ == "__main__":
    simulate_conservative_neuro_gating()
