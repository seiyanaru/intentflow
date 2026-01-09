#!/usr/bin/env python3
"""
Pseudo-online simulation for TCFormer Hybrid with TTT.

This script simulates real-time EEG processing using recorded BCIC IV 2a data.
It reads GDF files, streams data through a ring buffer, and performs inference
with the trained TCFormer Hybrid model.

Usage:
    python run_ttt_stream_sim.py --checkpoint /path/to/model.ckpt --subject 1

Example:
    python run_ttt_stream_sim.py \
        --checkpoint results/paper_experiments/bcic2a/20260104_171049/runs/blr0.05_ls0.05_reg0.05_thr0.85_am0.3_lrm0.3/data/checkpoints/subject_1_model.ckpt \
        --data_dir /workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf \
        --subject 1 \
        --gpu_id 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import mne

# Add parent directories to path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "offline"))

from intentflow.online.models.online_wrapper import OnlineTCFormerWrapper
from intentflow.online.dsp.stream_preprocessor import StreamNormalizer, WindowNormalizer


# -----------------------------------------------------------------------------
# Ring Buffer (from pseudo_online.py with minor improvements)
# -----------------------------------------------------------------------------

class RingBuffer:
    """
    Ring buffer for streaming EEG data.
    
    Maintains a sliding window of the most recent samples.
    """
    
    def __init__(self, n_channels: int, window_size: int):
        """
        Initialize the ring buffer.
        
        Args:
            n_channels: Number of EEG channels.
            window_size: Window size in samples.
        """
        self.n_channels = n_channels
        self.window_size = window_size
        self.buffer = np.zeros((n_channels, window_size), dtype=np.float32)
        self.n_received = 0
        self.filled = False
        
    def add(self, chunk: np.ndarray) -> None:
        """
        Add a chunk of data to the buffer.
        
        Args:
            chunk: Data chunk of shape [C, T].
        """
        n_samples = chunk.shape[1]
        self.n_received += n_samples
        
        if n_samples >= self.window_size:
            # Chunk is larger than window - just take the last window_size samples
            self.buffer = chunk[:, -self.window_size:].astype(np.float32)
            self.filled = True
            return
        
        # Shift and append
        self.buffer = np.roll(self.buffer, -n_samples, axis=1)
        self.buffer[:, -n_samples:] = chunk.astype(np.float32)
        
        if self.n_received >= self.window_size:
            self.filled = True
    
    def get_window(self) -> np.ndarray:
        """Get the current window."""
        return self.buffer.copy()
    
    def reset(self) -> None:
        """Reset the buffer."""
        self.buffer.fill(0)
        self.n_received = 0
        self.filled = False


# -----------------------------------------------------------------------------
# Trial Extraction from GDF
# -----------------------------------------------------------------------------

def extract_trials_from_gdf(
    gdf_path: str,
    sfreq_target: int = 250,
    window_sec: float = 4.0,
    event_codes: Dict[str, int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Extract motor imagery trials from a BCIC IV 2a GDF file.
    
    Args:
        gdf_path: Path to the GDF file.
        sfreq_target: Target sampling frequency.
        window_sec: Trial window duration in seconds.
        event_codes: Mapping from event description to class label.
        
    Returns:
        Tuple of (data, labels, onsets) where:
            - data: [N_trials, C, T] array
            - labels: [N_trials] array of class labels
            - onsets: List of onset times in seconds
    """
    if event_codes is None:
        # BCIC IV 2a event codes
        event_codes = {'769': 0, '770': 1, '771': 2, '772': 3}
    
    # Load raw data
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    
    # Pick first 22 channels (EEG only)
    raw.pick(range(22))
    
    # Scale to microvolts
    raw._data *= 1e6
    
    # Resample if needed
    if raw.info['sfreq'] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)
    
    sfreq = raw.info['sfreq']
    window_samples = int(window_sec * sfreq)
    
    # Extract trials based on annotations
    trials = []
    labels = []
    onsets = []
    
    for annot in raw.annotations:
        if annot['description'] in event_codes:
            onset_sec = annot['onset']
            onset_sample = int(onset_sec * sfreq)
            end_sample = onset_sample + window_samples
            
            # Check bounds
            if end_sample > raw.n_times:
                continue
            
            trial_data = raw.get_data(start=onset_sample, stop=end_sample)
            trials.append(trial_data)
            labels.append(event_codes[annot['description']])
            onsets.append(onset_sec)
    
    if len(trials) == 0:
        return np.array([]), np.array([]), []
    
    data = np.stack(trials, axis=0)  # [N, C, T]
    labels = np.array(labels)
    
    return data, labels, onsets


def stream_continuous_from_gdf(
    gdf_path: str,
    chunk_sec: float = 0.1,
    sfreq_target: int = 250,
) -> Tuple[np.ndarray, int, mne.io.BaseRaw]:
    """
    Load GDF and prepare for continuous streaming simulation.
    
    Args:
        gdf_path: Path to GDF file.
        chunk_sec: Chunk duration in seconds.
        sfreq_target: Target sampling frequency.
        
    Returns:
        Tuple of (data, chunk_samples, raw) where:
            - data: Continuous data [C, T_total]
            - chunk_samples: Number of samples per chunk
            - raw: The loaded raw object (for annotation access)
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    raw.pick(range(22))
    raw._data *= 1e6
    
    if raw.info['sfreq'] != sfreq_target:
        raw.resample(sfreq_target, verbose=False)
    
    sfreq = raw.info['sfreq']
    chunk_samples = int(chunk_sec * sfreq)
    data = raw.get_data()
    
    return data, chunk_samples, raw


# -----------------------------------------------------------------------------
# Simulation Runner
# -----------------------------------------------------------------------------

class StreamSimulator:
    """
    Simulates real-time EEG processing with TTT model.
    """
    
    CLASS_LABELS = ["left_hand", "right_hand", "feet", "tongue"]
    
    def __init__(
        self,
        model_wrapper: OnlineTCFormerWrapper,
        normalizer: StreamNormalizer | WindowNormalizer,
        n_channels: int = 22,
        window_sec: float = 4.0,
        sfreq: int = 250,
        hop_sec: float = 0.5,
    ):
        """
        Initialize the simulator.
        
        Args:
            model_wrapper: The online model wrapper.
            normalizer: The stream normalizer.
            n_channels: Number of EEG channels.
            window_sec: Window duration in seconds.
            sfreq: Sampling frequency.
            hop_sec: Prediction hop duration (how often to predict).
        """
        self.model = model_wrapper
        self.normalizer = normalizer
        self.sfreq = sfreq
        self.window_samples = int(window_sec * sfreq)
        self.hop_samples = int(hop_sec * sfreq)
        
        self.buffer = RingBuffer(n_channels, self.window_samples)
        
        # Metrics tracking
        self.predictions: List[Dict] = []
        self.ground_truth: List[int] = []
        self.latencies: List[float] = []
        
    def reset(self) -> None:
        """Reset the simulator state."""
        self.buffer.reset()
        self.model.reset_state()
        if hasattr(self.normalizer, 'reset'):
            self.normalizer.reset()
        self.predictions = []
        self.ground_truth = []
        self.latencies = []
    
    def run_trial_by_trial(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run simulation in trial-by-trial mode.
        
        Each trial is a complete window that gets normalized and fed to the model.
        
        Args:
            trials: Trial data [N, C, T].
            labels: Ground truth labels [N].
            verbose: Print progress.
            
        Returns:
            Results dictionary with accuracy and predictions.
        """
        self.reset()
        n_trials = len(trials)
        correct = 0
        
        for i, (trial, label) in enumerate(zip(trials, labels)):
            # Normalize
            t_start = time.perf_counter()
            normalized = self.normalizer(trial)  # [C, T]
            
            # Predict
            result = self.model.predict_step(normalized)
            latency = (time.perf_counter() - t_start) * 1000  # ms
            
            pred = result['pred_idx']
            conf = result['confidence']
            
            self.predictions.append(result)
            self.ground_truth.append(int(label))
            self.latencies.append(latency)
            
            if pred == label:
                correct += 1
            
            if verbose and (i + 1) % 10 == 0:
                running_acc = correct / (i + 1) * 100
                print(f"  Trial {i+1}/{n_trials}: "
                      f"pred={self.CLASS_LABELS[pred]}, "
                      f"true={self.CLASS_LABELS[label]}, "
                      f"conf={conf:.2f}, "
                      f"acc={running_acc:.1f}%")
        
        accuracy = correct / n_trials * 100
        avg_latency = np.mean(self.latencies)
        
        results = {
            'accuracy': accuracy,
            'n_trials': n_trials,
            'n_correct': correct,
            'avg_latency_ms': avg_latency,
            'predictions': [p['pred_idx'] for p in self.predictions],
            'ground_truth': self.ground_truth,
            'confidences': [p['confidence'] for p in self.predictions],
        }
        
        if verbose:
            print(f"\n=== Results ===")
            print(f"Accuracy: {accuracy:.2f}% ({correct}/{n_trials})")
            print(f"Avg Latency: {avg_latency:.2f} ms")
        
        return results
    
    def run_streaming(
        self,
        data: np.ndarray,
        raw: mne.io.BaseRaw,
        chunk_samples: int,
        event_codes: Dict[str, int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run continuous streaming simulation.
        
        Data is fed chunk by chunk, and predictions are made when the buffer
        is full and a hop interval has passed.
        
        Args:
            data: Continuous data [C, T_total].
            raw: Raw MNE object (for annotations).
            chunk_samples: Samples per chunk.
            event_codes: Event description to class mapping.
            verbose: Print progress.
            
        Returns:
            Results dictionary.
        """
        if event_codes is None:
            event_codes = {'769': 0, '770': 1, '771': 2, '772': 3}
        
        self.reset()
        
        n_total = data.shape[1]
        current_idx = 0
        samples_since_last_pred = 0
        pred_count = 0
        
        # Build event lookup by sample index
        event_windows = []
        for annot in raw.annotations:
            if annot['description'] in event_codes:
                onset_sample = int(annot['onset'] * self.sfreq)
                end_sample = onset_sample + self.window_samples
                label = event_codes[annot['description']]
                event_windows.append((onset_sample, end_sample, label))
        
        if verbose:
            print(f"Streaming {n_total / self.sfreq:.1f} sec data...")
            print(f"Found {len(event_windows)} labeled events.")
        
        while current_idx < n_total:
            end_idx = min(current_idx + chunk_samples, n_total)
            chunk = data[:, current_idx:end_idx]
            
            self.buffer.add(chunk)
            samples_since_last_pred += chunk.shape[1]
            
            # Predict when buffer is full and hop interval has passed
            if self.buffer.filled and samples_since_last_pred >= self.hop_samples:
                window = self.buffer.get_window()
                normalized = self.normalizer(window)
                
                t_start = time.perf_counter()
                result = self.model.predict_step(normalized)
                latency = (time.perf_counter() - t_start) * 1000
                
                self.predictions.append(result)
                self.latencies.append(latency)
                
                # Find ground truth for this window (if any event is contained)
                window_start = current_idx - self.window_samples
                window_end = current_idx
                gt_label = -1
                for ev_start, ev_end, label in event_windows:
                    # Check if event is mostly within this window
                    if ev_start >= window_start and ev_start < window_end:
                        gt_label = label
                        break
                
                self.ground_truth.append(gt_label)
                pred_count += 1
                samples_since_last_pred = 0
                
                if verbose and pred_count % 20 == 0:
                    print(f"  Predictions: {pred_count}, "
                          f"Latest: {self.CLASS_LABELS[result['pred_idx']]} "
                          f"(conf={result['confidence']:.2f})")
            
            current_idx = end_idx
        
        # Calculate accuracy only for windows with ground truth
        valid_mask = np.array(self.ground_truth) >= 0
        if np.any(valid_mask):
            preds = np.array([p['pred_idx'] for p in self.predictions])
            gt = np.array(self.ground_truth)
            correct = np.sum(preds[valid_mask] == gt[valid_mask])
            n_valid = np.sum(valid_mask)
            accuracy = correct / n_valid * 100
        else:
            accuracy = 0.0
            n_valid = 0
            correct = 0
        
        results = {
            'accuracy': accuracy,
            'n_predictions': pred_count,
            'n_valid': n_valid,
            'n_correct': correct,
            'avg_latency_ms': np.mean(self.latencies) if self.latencies else 0,
        }
        
        if verbose:
            print(f"\n=== Streaming Results ===")
            print(f"Total predictions: {pred_count}")
            print(f"Valid (with GT): {n_valid}")
            print(f"Accuracy (on valid): {accuracy:.2f}%")
            print(f"Avg Latency: {results['avg_latency_ms']:.2f} ms")
        
        return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pseudo-online simulation for TCFormer Hybrid with TTT"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (optional, will auto-detect)"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/workspace-cloud/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf",
        help="Path to BCIC IV 2a GDF files"
    )
    parser.add_argument(
        "--subject", type=int, default=1,
        help="Subject ID (1-9)"
    )
    parser.add_argument(
        "--session", type=str, default="E",
        choices=["T", "E"],
        help="Session: T (training) or E (evaluation)"
    )
    parser.add_argument(
        "--mode", type=str, default="trial",
        choices=["trial", "stream"],
        help="Simulation mode: trial-by-trial or continuous streaming"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU ID to use (-1 for CPU)"
    )
    parser.add_argument(
        "--window_sec", type=float, default=4.0,
        help="Window duration in seconds"
    )
    parser.add_argument(
        "--normalizer", type=str, default="window",
        choices=["stream", "window"],
        help="Normalizer type: stream (EMA) or window (per-window)"
    )
    parser.add_argument(
        "--reset_state", action="store_true",
        help="Reset TTT state before each trial"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device
    device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu"
    
    print("=" * 60)
    print("TTT Online Simulation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Subject: {args.subject}")
    print(f"Session: {args.session}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load model
    model = OnlineTCFormerWrapper(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        reset_state_each_trial=args.reset_state,
    )
    
    # Create normalizer
    if args.normalizer == "stream":
        normalizer = StreamNormalizer(n_channels=22, alpha=0.01, warmup_samples=250)
    else:
        normalizer = WindowNormalizer()
    
    print(f"Normalizer: {args.normalizer}")
    
    # Load data
    gdf_path = Path(args.data_dir) / f"A{args.subject:02d}{args.session}.gdf"
    if not gdf_path.exists():
        print(f"Error: GDF file not found: {gdf_path}")
        sys.exit(1)
    
    print(f"Loading data from: {gdf_path}")
    
    # Create simulator
    simulator = StreamSimulator(
        model_wrapper=model,
        normalizer=normalizer,
        n_channels=22,
        window_sec=args.window_sec,
        sfreq=250,
        hop_sec=0.5,
    )
    
    # Run simulation
    if args.mode == "trial":
        # Extract trials
        trials, labels, onsets = extract_trials_from_gdf(
            str(gdf_path),
            sfreq_target=250,
            window_sec=args.window_sec,
        )
        
        if len(trials) == 0:
            print("No valid trials found in the GDF file.")
            sys.exit(1)
        
        print(f"Extracted {len(trials)} trials")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print()
        
        results = simulator.run_trial_by_trial(trials, labels, verbose=True)
        
    else:  # stream mode
        data, chunk_samples, raw = stream_continuous_from_gdf(
            str(gdf_path),
            chunk_sec=0.1,
            sfreq_target=250,
        )
        
        print(f"Data shape: {data.shape}")
        print(f"Duration: {data.shape[1] / 250:.1f} seconds")
        print()
        
        results = simulator.run_streaming(
            data, raw, chunk_samples, verbose=True
        )
    
    print("\nSimulation complete!")
    return results


if __name__ == "__main__":
    main()


