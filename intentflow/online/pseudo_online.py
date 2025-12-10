import numpy as np
import time
import mne
import torch

class RingBuffer:
    def __init__(self, channels, window_size):
        self.channels = channels
        self.window_size = window_size
        self.buffer = np.zeros((channels, window_size))
        self.ptr = 0
        self.filled = False
        
    def add(self, chunk):
        # chunk: (channels, n_samples)
        n_samples = chunk.shape[1]
        
        if n_samples >= self.window_size:
            self.buffer = chunk[:, -self.window_size:]
            self.ptr = 0
            self.filled = True
            return

        remaining = self.window_size - self.ptr
        if n_samples < remaining:
            self.buffer[:, self.ptr:self.ptr+n_samples] = chunk
            self.ptr += n_samples
        else:
            # Wrap around not strictly needed if we just shift, 
            # but usually ring buffer overwrites oldest.
            # Here we implement a sliding window via shifting for simplicity in Python (slower but clear)
            # Or circular buffer.
            # Let's do simple shift:
            self.buffer = np.roll(self.buffer, -n_samples, axis=1)
            self.buffer[:, -n_samples:] = chunk
            self.filled = True
            self.ptr = 0 # Not strictly a pointer in shift mode, but indicates ready state

    def get_window(self):
        return self.buffer

class RealtimePredictor:
    def __init__(self, model_path, config_path=None, window_sec=4.0, sfreq=250, 
                 reset_state=True, device="cuda:0"):
        self.window_samples = int(window_sec * sfreq)
        self.sfreq = sfreq
        self.reset_state = reset_state
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load Model (Placeholder for actual loading logic)
        # Assuming we can load TCFormer/TTT here.
        # For simulation, we'll mock or assume the class is available.
        print(f"Loading model from {model_path}...")
        # self.model = ... (Load logic similar to analyze_behavior.py)
        # self.model.to(self.device)
        # self.model.eval()
        self.model = None # Placeholder
        
        self.buffer = RingBuffer(channels=22, window_size=self.window_samples)
        self.n_processed = 0

    def input_stream(self, chunk):
        """
        Receive a chunk of data (Channels x Time).
        If buffer is full, trigger inference.
        """
        # chunk shape expected: (22, n_samples)
        self.buffer.add(chunk)
        self.n_processed += chunk.shape[1]
        
        # Check if we should predict?
        # Typically we predict every chunk (sliding window) or every window?
        # Prompt says "Buffer reaches window size -> Predict". 
        # But if it's a stream, we usually predict periodically (e.g. every 0.1s update).
        # Assuming we predict on every update if buffer is full (filled at least once).
        
        if self.buffer.filled:
            self._predict()

    def _predict(self):
        window = self.buffer.get_window() # (22, 1000)
        
        # Preprocessing (Z-score etc) should happen here
        # window = (window - mean) / std ...
        
        # Tensor conversion
        x = torch.from_numpy(window).float().unsqueeze(0) # (1, 22, 1000)
        x = x.to(self.device)
        
        if self.model:
            with torch.no_grad():
                # TTT State handling
                # If TTT model, we might want to reset state before forward if reset_state is True
                # if self.reset_state and hasattr(self.model, 'reset_state'):
                #     self.model.reset_state()
                
                logits = self.model(x)
                pred = torch.argmax(logits, dim=1).item()
                # print(f"Pred: {pred}")

def simulate_from_gdf(gdf_path, predictor: RealtimePredictor, chunk_sec=0.1):
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    data = raw.get_data() # (Channels, Time)
    sfreq = raw.info['sfreq']
    
    chunk_samples = int(chunk_sec * sfreq)
    n_total = data.shape[1]
    
    print(f"Starting simulation: {n_total/sfreq:.1f} sec data, {chunk_sec}s chunks")
    
    start_time = time.time()
    current_idx = 0
    
    while current_idx < n_total:
        end_idx = min(current_idx + chunk_samples, n_total)
        chunk = data[:, current_idx:end_idx]
        
        # Feed to predictor
        predictor.input_stream(chunk)
        
        current_idx = end_idx
        
        # Simulate real-time wait
        # elapsed = time.time() - start_time
        # expected = current_idx / sfreq
        # if expected > elapsed:
        #     time.sleep(expected - elapsed)
            
    print("Simulation finished.")

if __name__ == "__main__":
    # Example usage
    # predictor = RealtimePredictor("path/to/model.ckpt")
    # simulate_from_gdf("path/to/data.gdf", predictor)
    pass


