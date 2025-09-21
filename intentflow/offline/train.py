"""Training utilities for intentflow offline models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from intentflow.offline.data_npz import (
  NPZDataset,
  apply_zscore,
  compute_zscore_stats,
  load_npz,
  train_val_split,
)
from intentflow.offline.models.eeg_transformer import EEGTransformerTiny
from intentflow.offline.models.eegnet_lawhern import EEGNet


def seed_everything(seed: int) -> None:
  """Seed numpy and torch for reproducible experiments."""
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, in_ch: int, time_steps: int, n_classes: int = 2) -> nn.Module:
  """Instantiate a model by name with the given input dimensions."""
  name = name.lower()
  if name == "eegnet":
    return EEGNet(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  if name == "transformer":
    return EEGTransformerTiny(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
  """Run a single training epoch and return the mean loss."""
  model.train()
  criterion = nn.CrossEntropyLoss()
  running_loss = 0.0
  total_batches = 0
  for inputs, targets in loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    opt.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()
    opt.step()
    running_loss += loss.item()
    total_batches += 1
  return running_loss / max(1, total_batches)


def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
  """Evaluate accuracy over a data loader."""
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, targets in loader:
      inputs = inputs.to(device)
      targets = targets.to(device)
      logits = model(inputs)
      preds = logits.argmax(dim=1)
      correct += (preds == targets).sum().item()
      total += targets.size(0)
  return correct / max(1, total)


def run_training(
  npz: str,
  model_name: str,
  epochs: int,
  batch_size: int,
  lr: float,
  out_dir: str,
  seed: int,
) -> Tuple[Path, float]:
  """Execute the full training loop and return the checkpoint path and best accuracy."""
  seed_everything(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  X, y, _ = load_npz(npz)
  X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=seed)
  mean, std = compute_zscore_stats(X_train)
  X_train_norm = apply_zscore(X_train, mean, std)
  X_val_norm = apply_zscore(X_val, mean, std)

  train_dataset = NPZDataset(X_train_norm, y_train)
  val_dataset = NPZDataset(X_val_norm, y_val)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  in_ch = X.shape[1]
  time_steps = X.shape[2]
  model = build_model(model_name, in_ch=in_ch, time_steps=time_steps).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  best_acc = 0.0
  os.makedirs(out_dir, exist_ok=True)
  best_path = Path(out_dir) / "best.pt"

  for epoch in range(epochs):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    acc = eval_acc(model, val_loader, device)
    print(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - val_acc: {acc:.4f}")
    if acc >= best_acc:
      best_acc = acc
      ckpt = {
        "state_dict": model.state_dict(),
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "meta": {"model": model_name, "C": in_ch, "T": time_steps},
      }
      torch.save(ckpt, best_path)
  return best_path, best_acc


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for the training script."""
  parser = argparse.ArgumentParser(description="Train intentflow offline models on NPZ data")
  parser.add_argument("--npz", type=str, default="data/processed/mi_dummy.npz", help="Path to NPZ dataset")
  parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet", "transformer"], help="Model name")
  parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
  parser.add_argument("--batch", type=int, default=64, help="Batch size")
  parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
  parser.add_argument("--out", type=str, default="runs/eegnet", help="Output directory for checkpoints")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  return parser.parse_args()


def main() -> None:
  """CLI entry point for training models."""
  args = parse_args()
  try:
    run_training(
      npz=args.npz,
      model_name=args.model,
      epochs=args.epochs,
      batch_size=args.batch,
      lr=args.lr,
      out_dir=args.out,
      seed=args.seed,
    )
  except Exception as exc:  # noqa: BLE001
    print(f"Training failed: {exc}")
    raise


if __name__ == "__main__":
  main()
