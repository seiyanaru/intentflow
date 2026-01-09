"""Stream preprocessor for online EEG normalization.

This module provides real-time signal normalization using exponential moving
average (EMA) statistics, suitable for online BCI applications.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class StreamNormalizer:
    """
    Real-time Z-score normalizer using Exponential Moving Average (EMA).
    
    This class maintains running statistics (mean and variance) and applies
    Z-score normalization to incoming data chunks. Unlike sklearn's StandardScaler,
    this does not require fitting on the entire dataset beforehand.
    
    Attributes:
        n_channels: Number of EEG channels.
        alpha: EMA smoothing factor (higher = more weight to recent data).
        eps: Small constant for numerical stability in division.
        mean: Current EMA estimate of the mean per channel.
        var: Current EMA estimate of the variance per channel.
    """
    
    def __init__(
        self,
        n_channels: int,
        alpha: float = 0.01,
        eps: float = 1e-8,
        warmup_samples: int = 250,
    ):
        """
        Initialize the StreamNormalizer.
        
        Args:
            n_channels: Number of input channels (e.g., 22 for BCIC IV 2a).
            alpha: EMA smoothing factor. 0.01 means ~100 samples half-life.
                   Lower values = smoother/slower adaptation.
            eps: Small constant to prevent division by zero.
            warmup_samples: Number of samples to collect before enabling
                            normalization. During warmup, raw data is returned.
        """
        self.n_channels = n_channels
        self.alpha = alpha
        self.eps = eps
        self.warmup_samples = warmup_samples
        
        # Running statistics (initialized to None, updated on first call)
        self.mean: Optional[np.ndarray] = None  # [C]
        self.var: Optional[np.ndarray] = None   # [C]
        self.n_seen: int = 0
        
    def reset(self) -> None:
        """Reset all running statistics."""
        self.mean = None
        self.var = None
        self.n_seen = 0
        
    def _initialize_stats(self, chunk: np.ndarray) -> None:
        """Initialize statistics from the first chunk."""
        # chunk: [C, T]
        self.mean = chunk.mean(axis=1)  # [C]
        self.var = chunk.var(axis=1) + self.eps  # [C]
        
    def _update_stats(self, chunk: np.ndarray) -> None:
        """
        Update running mean and variance using EMA.
        
        Uses Welford-like online update for numerical stability.
        
        Args:
            chunk: Input data of shape [C, T].
        """
        # Compute batch statistics
        batch_mean = chunk.mean(axis=1)  # [C]
        batch_var = chunk.var(axis=1)    # [C]
        
        # EMA update
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * batch_var
        
    def transform(self, chunk: np.ndarray) -> np.ndarray:
        """
        Normalize a chunk of data using current running statistics.
        
        Args:
            chunk: Input data of shape [C, T] (channels x time samples).
            
        Returns:
            Normalized data of the same shape [C, T].
        """
        if chunk.ndim != 2:
            raise ValueError(f"Expected 2D input [C, T], got shape {chunk.shape}")
        if chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.n_channels}, got {chunk.shape[0]}"
            )
        
        n_samples = chunk.shape[1]
        self.n_seen += n_samples
        
        # Initialize on first call
        if self.mean is None:
            self._initialize_stats(chunk)
            # During first chunk, use its own stats
            if self.n_seen < self.warmup_samples:
                return chunk  # Return raw during warmup
        
        # Update statistics
        self._update_stats(chunk)
        
        # During warmup, return raw data (stats are being accumulated)
        if self.n_seen < self.warmup_samples:
            return chunk
        
        # Z-score normalization: (x - mean) / std
        std = np.sqrt(self.var + self.eps)  # [C]
        normalized = (chunk - self.mean[:, np.newaxis]) / std[:, np.newaxis]
        
        return normalized.astype(np.float32)
    
    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        """Alias for transform()."""
        return self.transform(chunk)
    
    def get_stats(self) -> dict:
        """Return current statistics for debugging."""
        return {
            "mean": self.mean.copy() if self.mean is not None else None,
            "var": self.var.copy() if self.var is not None else None,
            "std": np.sqrt(self.var + self.eps) if self.var is not None else None,
            "n_seen": self.n_seen,
            "is_warmed_up": self.n_seen >= self.warmup_samples,
        }


class WindowNormalizer:
    """
    Simple per-window Z-score normalizer.
    
    Unlike StreamNormalizer (EMA-based), this class normalizes each window
    independently using only the data within that window. This is useful
    for comparing with offline preprocessing that uses per-trial normalization.
    """
    
    def __init__(self, eps: float = 1e-8):
        """
        Initialize the WindowNormalizer.
        
        Args:
            eps: Small constant to prevent division by zero.
        """
        self.eps = eps
        
    def transform(self, window: np.ndarray) -> np.ndarray:
        """
        Normalize a window using its own statistics.
        
        Args:
            window: Input data of shape [C, T].
            
        Returns:
            Normalized data of the same shape [C, T].
        """
        if window.ndim != 2:
            raise ValueError(f"Expected 2D input [C, T], got shape {window.shape}")
        
        # Per-channel normalization
        mean = window.mean(axis=1, keepdims=True)  # [C, 1]
        std = window.std(axis=1, keepdims=True) + self.eps  # [C, 1]
        
        normalized = (window - mean) / std
        return normalized.astype(np.float32)
    
    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Alias for transform()."""
        return self.transform(window)


