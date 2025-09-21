"""End-to-end tests for offline training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from intentflow.offline import eval as eval_module
from intentflow.offline import train as train_module
from intentflow.offline.models.eeg_transformer import EEGTransformerTiny
from intentflow.offline.models.eegnet_lawhern import EEGNet
from scripts.make_dummy_mi_npz import synthesize_mi


def test_model_forward_shapes() -> None:
  """Both model variants should return logits shaped [B, 2]."""
  x = torch.randn(4, 8, 500)
  eegnet = EEGNet()
  transformer = EEGTransformerTiny()
  assert eegnet(x).shape == (4, 2)
  assert transformer(x).shape == (4, 2)


def test_training_and_eval_flow(tmp_path: Path) -> None:
  """Training for one epoch should produce checkpoints and prediction CSV."""
  X, y = synthesize_mi(n_per_class=4)
  npz_path = tmp_path / "dummy.npz"
  np.savez(npz_path, X=X.astype(np.float32), y=y.astype(np.int64), fs=250)
  out_dir = tmp_path / "run"
  ckpt_path, _ = train_module.run_training(
    npz=str(npz_path),
    model_name="eegnet",
    epochs=1,
    batch_size=4,
    lr=1e-3,
    out_dir=str(out_dir),
    seed=42,
  )
  assert ckpt_path.exists()
  csv_path = tmp_path / "preds.csv"
  eval_module.run_inference(npz=str(npz_path), ckpt=str(ckpt_path), out_path=str(csv_path), step=0.2)
  with csv_path.open("r", encoding="utf-8") as handle:
    header = handle.readline().strip()
  assert header == "ts,intent,conf"
