"""Shape and training smoke tests for EEGEncoder."""

from __future__ import annotations

import torch

from intentflow.offline.models.eegencoder import EEGEncoder


def test_eegencoder_forward_shape() -> None:
  """EEGEncoder should map [B,C,T] to [B,2]."""
  model = EEGEncoder()
  x = torch.randn(4, 8, 500)
  logits = model(x)
  assert logits.shape == (4, 2)


def test_eegencoder_training_step() -> None:
  """Single optimization step should run without errors."""
  model = EEGEncoder()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.CrossEntropyLoss()
  x = torch.randn(16, 8, 500)
  y = torch.randint(0, 2, (16,))
  model.train()
  optimizer.zero_grad()
  logits = model(x)
  loss = criterion(logits, y)
  loss.backward()
  optimizer.step()
