"""Data loading and preprocessing utilities for NPZ EEG datasets."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
  """Load an NPZ file containing EEG data and labels."""
  with np.load(path) as data:
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    fs = int(data.get("fs", 1))
  return X, y, fs


def train_val_split(
  X: np.ndarray,
  y: np.ndarray,
  val_ratio: float = 0.2,
  seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Split arrays into train and validation partitions."""
  rng = np.random.default_rng(seed)
  n_samples = X.shape[0]
  indices = np.arange(n_samples)
  rng.shuffle(indices)
  n_val = int(round(n_samples * val_ratio))
  n_val = max(1, min(n_samples - 1, n_val))
  val_idx = indices[:n_val]
  train_idx = indices[n_val:]
  return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def compute_zscore_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Compute per-channel mean and std from training data."""
  mean = X_train.mean(axis=(0, 2))
  std = X_train.std(axis=(0, 2))
  std = np.where(std < 1e-6, 1e-6, std)
  return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
  """Apply z-score normalisation using provided statistics."""
  mean = mean.reshape(1, -1, 1)
  std = std.reshape(1, -1, 1)
  return ((X - mean) / std).astype(np.float32)


class NPZDataset(Dataset):
  """Torch dataset wrapping NPZ arrays."""

  def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
    self.X = torch.from_numpy(X.astype(np.float32))
    self.y = torch.from_numpy(y.astype(np.int64))

  def __len__(self) -> int:
    """Return number of samples in dataset."""
    return self.X.size(0)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a single sample and label."""
    return self.X[index], self.y[index]
