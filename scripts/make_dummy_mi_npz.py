"""Generate synthetic motor imagery EEG data and save as NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


def synthesize_mi(
  n_per_class: int = 400,
  fs: int = 250,
  T: int = 500,
  C: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
  """Create synthetic motor imagery data for left and right hand classes."""
  rng = np.random.default_rng(42)
  times = np.arange(T) / fs

  def build_class(channel_indices: Tuple[int, int]) -> np.ndarray:
    data = np.zeros((n_per_class, C, T), dtype=np.float32)
    for i in range(n_per_class):
      noise = rng.normal(0.0, 0.5, size=(C, T)).astype(np.float32)
      drift_freq = rng.uniform(0.5, 1.0)
      drift_amp = rng.uniform(0.1, 0.3)
      drift = drift_amp * np.sin(2.0 * np.pi * drift_freq * times)
      drift = np.tile(drift, (C, 1))
      freq = rng.uniform(10.0, 12.0)
      phase = rng.uniform(0.0, 2.0 * np.pi)
      mi_wave = np.sin(2.0 * np.pi * freq * times + phase)
      mi_wave = np.tile(mi_wave, (C, 1))
      mask = np.zeros(C, dtype=np.float32)
      mask[channel_indices[0]] = rng.uniform(1.5, 2.0)
      mask[channel_indices[1]] = rng.uniform(1.5, 2.0)
      mask += rng.uniform(0.2, 0.4, size=C)
      patterned = mask[:, None] * mi_wave
      sample = noise + drift + patterned
      sample = gaussian_filter1d(sample, sigma=1.0, axis=1, mode="reflect")
      data[i] = sample.astype(np.float32)
    return data

  left = build_class((2, 3))
  right = build_class((4, 5))
  X = np.concatenate([left, right], axis=0)
  y = np.concatenate([
    np.zeros(n_per_class, dtype=np.int64),
    np.ones(n_per_class, dtype=np.int64),
  ])
  indices = np.arange(X.shape[0])
  rng.shuffle(indices)
  X = X[indices]
  y = y[indices]
  return X, y


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for dataset synthesis."""
  parser = argparse.ArgumentParser(description="Generate synthetic MI EEG dataset")
  parser.add_argument("--out", type=str, default="data/processed/mi_dummy.npz", help="Target NPZ file path")
  return parser.parse_args()


def main() -> None:
  """Generate the dataset and write it to disk."""
  args = parse_args()
  try:
    X, y = synthesize_mi()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X.astype(np.float32), y=y.astype(np.int64), fs=250)
    print(f"Saved dataset to {args.out} with shape {X.shape}")
  except Exception as exc:  # noqa: BLE001
    print(f"Dataset generation failed: {exc}")
    raise


if __name__ == "__main__":
  main()
