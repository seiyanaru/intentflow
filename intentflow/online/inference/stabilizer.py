"""Inference stabilizer utilities for smoothing intent predictions."""
# TODO(P1): Integrate actual model confidences sourced from the ONNX pipeline.

from __future__ import annotations

import time
from typing import Dict


class Stabilizer:
  """Stabilize streaming intent predictions via EMA and hysteresis."""

  def __init__(self, conf_th: float, ema_ms: int, require_consec: int, min_interval_ms: int) -> None:
    """Initialize the stabilizer with smoothing and gating parameters."""
    self.conf_th = conf_th
    self.ema_ms = ema_ms
    self.require_consec = max(1, require_consec)
    self.min_interval_ms = max(0, min_interval_ms)
    self._ema: float | None = None
    self._last_update: float | None = None
    self._last_emit: float | None = None
    self._current_candidate: str = "none"
    self._consecutive: int = 0

  def apply(self, intent: str, conf: float) -> Dict[str, float | str]:
    """Update stabilizer state and return the stabilized intent and EMA."""
    now = time.monotonic()
    if self._ema is None or self.ema_ms <= 0:
      self._ema = conf
    else:
      last_time = self._last_update if self._last_update is not None else now
      dt = max(now - last_time, 1e-6)
      window = max(self.ema_ms / 1000.0, dt)
      alpha = min(1.0, dt / (window + dt))
      self._ema = (1.0 - alpha) * self._ema + alpha * conf
    self._last_update = now

    stable = "none"
    if intent != "none" and self._ema >= self.conf_th:
      if intent == self._current_candidate:
        self._consecutive += 1
      else:
        self._current_candidate = intent
        self._consecutive = 1
      if self._consecutive >= self.require_consec:
        elapsed_ms = (now - self._last_emit) * 1000.0 if self._last_emit else None
        if elapsed_ms is None or elapsed_ms >= self.min_interval_ms:
          stable = intent
          self._last_emit = now
          self._consecutive = 0
    else:
      self._current_candidate = "none"
      self._consecutive = 0

    return {"intent": stable, "conf_ema": float(self._ema)}
