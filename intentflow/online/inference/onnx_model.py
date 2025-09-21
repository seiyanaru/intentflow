"""ONNX Runtime wrapper for streaming intent inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort


class OnnxModel:
  """Thin convenience layer around an ONNX Runtime session."""

  def __init__(self, onnx_path: str, stats_json: str) -> None:
    """Initialise ONNX session with statistics for z-score normalisation."""
    stats = json.loads(Path(stats_json).read_text(encoding="utf-8"))
    self.model = str(stats.get("model", "unknown"))
    self.C = int(stats["C"])
    self.T = int(stats["T"])
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, self.C, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, self.C, 1)
    self.mean = mean
    self.std = np.where(std < 1e-6, 1e-6, std)

    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
      providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    self.session = ort.InferenceSession(str(onnx_path), providers=providers)
    self.input_name = self.session.get_inputs()[0].name

  def preprocess(self, x: np.ndarray) -> np.ndarray:
    """Apply z-score normalisation and return a batched tensor."""
    data = np.asarray(x, dtype=np.float32)
    if data.shape == (self.T, self.C):
      data = data.T
    if data.shape != (self.C, self.T):
      raise ValueError(f"Expected shape {(self.C, self.T)}, got {data.shape}")
    return ((data.reshape(1, self.C, self.T) - self.mean) / self.std).astype(np.float32)

  def infer(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run the ONNX session and return logits and probabilities."""
    outputs = self.session.run(None, {self.input_name: x})
    logits = np.asarray(outputs[0], dtype=np.float32)
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_logits)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return logits.astype(np.float32), probs.astype(np.float32)
