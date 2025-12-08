"""Tiny Transformer baseline for EEG intent classification."""

from __future__ import annotations

import math

import torch
from torch import nn


class EEGTransformerTiny(nn.Module):
  """Transformer encoder operating on temporal EEG patches."""

  def __init__(
    self,
    in_ch: int = 8,
    n_classes: int = 2,
    T: int = 500,
    embed_dim: int = 64,
    depth: int = 2,
    num_heads: int = 4,
    patch_stride: int = 10,
    mlp_ratio: float = 2.0,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.patch_stride = patch_stride
    self.proj = nn.Conv1d(in_ch, embed_dim, kernel_size=patch_stride, stride=patch_stride, bias=False)
    self.activation = nn.ReLU()
    num_patches = math.floor((T - patch_stride) / patch_stride + 1)
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=embed_dim,
      nhead=num_heads,
      dim_feedforward=int(embed_dim * mlp_ratio),
      dropout=dropout,
      batch_first=True,
      activation="gelu",
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    self.dropout = nn.Dropout(dropout)
    self.head = nn.Linear(embed_dim, n_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass receiving input [B, C, T] and returning logits."""
    x = self.proj(x)
    x = self.activation(x)
    x = x.transpose(1, 2)
    if x.size(1) != self.pos_embed.size(1):
      pos = torch.nn.functional.interpolate(
        self.pos_embed.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False
      ).transpose(1, 2)
    else:
      pos = self.pos_embed
    x = x + pos
    x = self.encoder(x)
    x = self.dropout(x)
    pooled = x.mean(dim=1)
    return self.head(pooled)
