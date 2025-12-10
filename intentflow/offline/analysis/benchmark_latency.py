"""
Benchmark inference latency of TCFormer vs TCFormer_TTT.
Measures execution time on GPU with torch.cuda.synchronize().
"""
import time
import torch
import numpy as np
from pathlib import Path
import sys

# Adjust path to allow imports from intentflow/offline
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import models
try:
    from intentflow.offline.models.tcformer.tcformer import TCFormer
    from intentflow.offline.models.tcformer_ttt.tcformer_ttt import TCFormerTTT
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from models.tcformer.tcformer import TCFormer
    from models.tcformer_ttt.tcformer_ttt import TCFormerTTT

def measure_latency_stats(model, input_shape, device, warmup=100, runs=1000):
    model.eval()
    model.to(device)
    
    x = torch.randn(input_shape).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    if "cuda" in str(device):
        torch.cuda.synchronize()
        
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            _ = model(x)
            if "cuda" in str(device):
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000) # ms
            
    return np.mean(times), np.std(times)

def main():
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Benchmarking on {device_str}")
    
    # Input config
    batch_size = 1
    channels = 22
    time_steps = 1000
    input_shape = (batch_size, channels, time_steps)
    
    # 1. Benchmark TCFormer (Base)
    print("-" * 40)
    print("TCFormer (Base)")
    model_base = TCFormer(n_channels=channels, n_classes=4)
    mean_base, std_base = measure_latency_stats(model_base, input_shape, device)
    print(f"Latency: {mean_base:.3f} ms ± {std_base:.3f} ms")
    
    # 2. Benchmark TCFormer_TTT (TTT)
    print("-" * 40)
    print("TCFormer_TTT (TTT)")
    # Default TTT config for benchmarking structure
    model_ttt = TCFormerTTT(n_channels=channels, n_classes=4, ttt_config={})
    mean_ttt, std_ttt = measure_latency_stats(model_ttt, input_shape, device)
    print(f"Latency: {mean_ttt:.3f} ms ± {std_ttt:.3f} ms")
    print("-" * 40)
    
    diff = mean_ttt - mean_base
    ratio = mean_ttt / mean_base
    print(f"Overhead: +{diff:.3f} ms (x{ratio:.2f})")

if __name__ == "__main__":
    main()


