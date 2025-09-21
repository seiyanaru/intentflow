"""Inference utilities for intentflow offline models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from intentflow.offline.data_npz import NPZDataset, apply_zscore, load_npz
from intentflow.offline.models.eeg_transformer import EEGTransformerTiny
from intentflow.offline.models.eegnet_lawhern import EEGNet


LABELS = {0: "left", 1: "right"}


def build_model(name: str, in_ch: int, time_steps: int, n_classes: int = 2) -> nn.Module:
  """Instantiate a model for evaluation by name."""
  name = name.lower()
  if name == "eegnet":
    return EEGNet(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  if name == "transformer":
    return EEGTransformerTiny(in_ch=in_ch, n_classes=n_classes, T=time_steps)
  raise ValueError(f"Unknown model: {name}")


def run_inference(npz: str, ckpt: str, out_path: str, step: float) -> Path:
  """Run inference over the NPZ dataset and emit a CSV with predictions."""
  X, y, _ = load_npz(npz)
  checkpoint = torch.load(ckpt, map_location="cpu")
  mean = np.asarray(checkpoint["mean"], dtype=np.float32)
  std = np.asarray(checkpoint["std"], dtype=np.float32)
  X_norm = apply_zscore(X, mean, std)
  dataset = NPZDataset(X_norm, y)
  loader = DataLoader(dataset, batch_size=64, shuffle=False)
  meta = checkpoint.get("meta", {})
  model_name = meta.get("model", "eegnet")
  in_ch = meta.get("C", X.shape[1])
  time_steps = meta.get("T", X.shape[2])
  model = build_model(model_name, in_ch=in_ch, time_steps=time_steps)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  outputs: List[Tuple[float, str, float]] = []
  with torch.no_grad():
    idx = 0
    for inputs, _ in loader:
      inputs = inputs.to(device)
      logits = model(inputs)
      probs = torch.softmax(logits, dim=1)
      conf, pred = probs.max(dim=1)
      for c, p in zip(conf.cpu().numpy(), pred.cpu().numpy()):
        outputs.append((idx * step, LABELS.get(int(p), "left"), float(c)))
        idx += 1
  out_file = Path(out_path)
  out_file.parent.mkdir(parents=True, exist_ok=True)
  with out_file.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["ts", "intent", "conf"])
    for ts, intent, conf in outputs:
      writer.writerow([f"{ts:.6f}", intent, f"{conf:.6f}"])
  return out_file


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for evaluation."""
  parser = argparse.ArgumentParser(description="Evaluate checkpoints on NPZ datasets")
  parser.add_argument("--npz", type=str, default="data/processed/mi_dummy.npz", help="Path to NPZ dataset")
  parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file path")
  parser.add_argument("--out", type=str, default="runs/preds.csv", help="Output CSV path")
  parser.add_argument("--step", type=float, default=0.2, help="Timestamp step between samples")
  return parser.parse_args()


def main() -> None:
  """CLI entry point for running inference."""
  args = parse_args()
  try:
    run_inference(npz=args.npz, ckpt=args.ckpt, out_path=args.out, step=args.step)
  except Exception as exc:  # noqa: BLE001
    print(f"Evaluation failed: {exc}")
    raise


if __name__ == "__main__":
  main()
