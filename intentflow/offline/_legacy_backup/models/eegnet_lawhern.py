"""Implementation of EEGNet following Lawhern et al. 2018."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
  """EEGNet architecture tailored for motor imagery classification."""

  def __init__(
    self,
    in_ch: int = 8,
    n_classes: int = 2,
    T: int = 500,
    F1: int = 8,
    D: int = 2,
    kernel_length: int = 64,
    dropout: float = 0.25,
  ) -> None:
    super().__init__()
    F2 = F1 * D
    self.reshape = nn.Identity()
    self.block1 = nn.Sequential(
      nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
      nn.BatchNorm2d(F1),
    )
    self.block2 = nn.Sequential(
      nn.Conv2d(F1, F2, (in_ch, 1), groups=F1, bias=False),
      nn.BatchNorm2d(F2),
      nn.ELU(),
      nn.AvgPool2d((1, 4)),
      nn.Dropout(dropout),
    )
    self.block3 = nn.Sequential(
      nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
      nn.Conv2d(F2, F2, (1, 1), bias=False),
      nn.BatchNorm2d(F2),
      nn.ELU(),
      nn.AvgPool2d((1, 8)),
      nn.Dropout(dropout),
      nn.AdaptiveAvgPool2d((1, 1)),
    )
    self.classifier = nn.Linear(F2, n_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass accepting input shaped [B, C, T]."""
    b, c, t = x.shape
    x = x.view(b, 1, c, t)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = x.view(b, -1)
    return self.classifier(x)

  def _get_name(self) -> str:
    """Return the class name for nicer repr."""
    return "EEGNet"
