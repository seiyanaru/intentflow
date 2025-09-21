"""EEGEncoder with dual-stream temporal processing for MI classification."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
  """Root-mean-square normalization."""

  def __init__(self, dim: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize input tensor by its RMS magnitude."""
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return x * rms * self.weight


class SwiGLU(nn.Module):
  """SwiGLU feed-forward block."""

  def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
    super().__init__()
    self.fc1 = nn.Linear(dim, hidden_dim * 2)
    self.fc2 = nn.Linear(hidden_dim, dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU transformation."""
    gate, value = self.fc1(x).chunk(2, dim=-1)
    activated = F.silu(gate) * value
    return self.dropout(self.fc2(activated))


def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  """Create upper-triangular causal mask filled with negative infinity."""
  mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
  return mask


class StableTransformerBlock(nn.Module):
  """Pre-norm transformer block with causal attention and SwiGLU MLP."""

  def __init__(
    self,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    attn_dropout: float,
  ) -> None:
    super().__init__()
    if embed_dim % num_heads != 0:
      raise ValueError("embed_dim must be divisible by num_heads")
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.norm_attn = RMSNorm(embed_dim)
    self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
    self.attn_dropout = nn.Dropout(attn_dropout)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    self.out_dropout = nn.Dropout(dropout)

    hidden_dim = int(embed_dim * mlp_ratio)
    self.norm_ff = RMSNorm(embed_dim)
    self.ff = SwiGLU(embed_dim, hidden_dim, dropout)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply causal self-attention followed by SwiGLU MLP."""
    residual = x
    x_norm = self.norm_attn(x)
    qkv = self.qkv(x_norm)
    q, k, v = qkv.chunk(3, dim=-1)

    B, L, _ = q.shape
    q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) * self.scale
    k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    attn_scores = torch.matmul(q, k.transpose(-1, -2))
    mask = _causal_mask(L, attn_scores.device, attn_scores.dtype)
    attn_scores = attn_scores + mask
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = self.attn_dropout(attn_probs)
    context = torch.matmul(attn_probs, v).transpose(1, 2).reshape(B, L, self.embed_dim)

    x = residual + self.out_dropout(self.out_proj(context))

    residual = x
    x_norm = self.norm_ff(x)
    x = residual + self.dropout(self.ff(x_norm))
    return x


class TCNBlock(nn.Module):
  """Temporal convolutional block with dilated convolutions."""

  def __init__(
    self,
    embed_dim: int,
    tcn_channels: int,
    dilations: Iterable[int],
    dropout: float,
  ) -> None:
    super().__init__()
    self.in_proj = (
      nn.Conv1d(embed_dim, tcn_channels, kernel_size=1, bias=False)
      if embed_dim != tcn_channels
      else nn.Identity()
    )
    self.convs = nn.ModuleList(
      [
        nn.Conv1d(
          tcn_channels,
          tcn_channels,
          kernel_size=3,
          dilation=dilation,
          padding=dilation,
          bias=False,
        )
        for dilation in dilations
      ]
    )
    self.activation = nn.GELU()
    self.dropout = nn.Dropout(dropout)
    self.out_proj = (
      nn.Conv1d(tcn_channels, embed_dim, kernel_size=1, bias=False)
      if embed_dim != tcn_channels
      else nn.Identity()
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply dilated convolutions with residual connections."""
    y = self.in_proj(x)
    for conv in self.convs:
      residual = y
      out = self.activation(conv(y))
      out = self.dropout(out)
      y = residual + out
    return self.out_proj(y)


class DSTSLayer(nn.Module):
  """Dual-stream block combining TCN and transformer paths."""

  def __init__(
    self,
    embed_dim: int,
    tcn_channels: int,
    dilations: Tuple[int, ...],
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    attn_dropout: float,
  ) -> None:
    super().__init__()
    self.tcn = TCNBlock(embed_dim, tcn_channels, dilations, dropout)
    self.transformer = StableTransformerBlock(
      embed_dim=embed_dim,
      num_heads=num_heads,
      mlp_ratio=mlp_ratio,
      dropout=dropout,
      attn_dropout=attn_dropout,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Return fused representation from TCN and transformer outputs."""
    tcn_out = self.tcn(x)
    transformer_out = self.transformer(x.transpose(1, 2)).transpose(1, 2)
    return tcn_out + transformer_out


class DSTSBranch(nn.Module):
  """Branch composed of stacked DSTS layers with temporal pooling."""

  def __init__(
    self,
    depth: int,
    embed_dim: int,
    tcn_channels: int,
    dilations: Tuple[int, ...],
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    attn_dropout: float,
  ) -> None:
    super().__init__()
    self.layers = nn.ModuleList(
      [
        DSTSLayer(
          embed_dim=embed_dim,
          tcn_channels=tcn_channels,
          dilations=dilations,
          num_heads=num_heads,
          mlp_ratio=mlp_ratio,
          dropout=dropout,
          attn_dropout=attn_dropout,
        )
        for _ in range(depth)
      ]
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process branch and reduce over time via mean pooling."""
    y = x
    for layer in self.layers:
      y = layer(y)
    return y.mean(dim=2)


class EEGEncoder(nn.Module):
  """Transformer-enhanced encoder for motor imagery EEG."""

  def __init__(
    self,
    in_ch: int = 8,
    n_classes: int = 2,
    T: int = 500,
    embed_dim: int = 64,
    branches: int = 3,
    tcn_channels: int = 64,
    tcn_dilations: Tuple[int, ...] = (1, 2, 4),
    num_heads: int = 4,
    depth: int = 2,
    mlp_ratio: float = 2.0,
    dropout: float = 0.2,
    attn_dropout: float = 0.1,
  ) -> None:
    super().__init__()
    hidden1 = max(embed_dim // 2, 32)
    hidden2 = max(embed_dim, hidden1)
    self.projector = nn.Sequential(
      nn.Conv1d(in_ch, hidden1, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm1d(hidden1),
      nn.GELU(),
      nn.Conv1d(hidden1, hidden2, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm1d(hidden2),
      nn.GELU(),
      nn.Conv1d(hidden2, embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
      nn.BatchNorm1d(embed_dim),
      nn.GELU(),
      nn.AvgPool1d(kernel_size=2, stride=2),
    )
    self.branch_dropout = nn.Dropout(dropout)
    self.branches = nn.ModuleList(
      [
        DSTSBranch(
          depth=depth,
          embed_dim=embed_dim,
          tcn_channels=tcn_channels,
          dilations=tcn_dilations,
          num_heads=num_heads,
          mlp_ratio=mlp_ratio,
          dropout=dropout,
          attn_dropout=attn_dropout,
        )
        for _ in range(branches)
      ]
    )
    self.classifier = nn.Linear(embed_dim, n_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode EEG signals and output logits."""
    features = self.projector(x.float())
    branch_outputs = []
    for branch in self.branches:
      branch_outputs.append(branch(self.branch_dropout(features)))
    fused = torch.stack(branch_outputs, dim=0).mean(dim=0)
    return self.classifier(fused)
