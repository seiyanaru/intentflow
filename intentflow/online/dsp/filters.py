"""Streaming IIR filters for EEG preprocessing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


class StreamingFilter:
  """Sequential bandpass and notch filtering with stateful processing."""

  def __init__(
    self,
    fs: int,
    ch: int,
    bp: Tuple[float, float] = (8.0, 30.0),
    notch: float = 50.0,
    q: float = 30.0,
) -> None:
    """Initialise cascaded filters and per-channel internal state."""
    if fs <= 0:
        raise ValueError("fs must be positive")
    if ch <= 0:
        raise ValueError("ch must be positive")

    nyquist = 0.5 * fs
    low, high = bp
    if not (0.0 < low < high < nyquist):
        raise ValueError("bandpass must satisfy 0 < low < high < fs/2")
    notch_norm = notch / nyquist
    if not (0.0 < notch_norm < 1.0):
        raise ValueError("notch frequency must be < Nyquist")

    self.fs = fs
    self.ch = ch

    self.band_sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    b_notch, a_notch = signal.iirnotch(notch_norm, q)
    self.notch_sos = signal.tf2sos(b_notch, a_notch)

    zi_band = signal.sosfilt_zi(self.band_sos).astype(np.float32)
    zi_notch = signal.sosfilt_zi(self.notch_sos).astype(np.float32)
    self.band_state = np.tile(zi_band, (ch, 1, 1))
    self.notch_state = np.tile(zi_notch, (ch, 1, 1))

def reset(self) -> None:
    """Reset internal filter states."""
    self.band_state[:] = signal.sosfilt_zi(self.band_sos)
    self.notch_state[:] = signal.sosfilt_zi(self.notch_sos)

def apply(self, block: np.ndarray) -> np.ndarray:
    """Filter a block of samples while preserving the input layout."""
    if block.ndim != 2:
        raise ValueError("block must be 2D")

    data = np.asarray(block, dtype=np.float32, order="C")
    transposed = False

    if data.shape[0] != self.ch:
        if data.shape[1] == self.ch:
            data = data.T
            transposed = True
        else:
            raise ValueError(f"expected channel dimension={self.ch}, got {data.shape}")

    filtered = np.empty_like(data, dtype=np.float32)
    for idx in range(self.ch):
        notch_out, self.notch_state[idx] = signal.sosfilt(self.notch_sos, data[idx], zi=self.notch_state[idx])
        band_out, self.band_state[idx] = signal.sosfilt(self.band_sos, notch_out, zi=self.band_state[idx])
        filtered[idx] = band_out.astype(np.float32, copy=False)

    return filtered.T if transposed else filtered

