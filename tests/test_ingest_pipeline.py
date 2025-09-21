"""Integration test for ingest-to-control pipeline."""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
import types
from typing import Any, Dict

import numpy as np
import pytest
from starlette.testclient import TestClient

from intentflow.online.server.app import INTENT_BUS, META, RAW_BUF, app


class DummyOnnxModel:
  """Mock ONNX model returning deterministic left intents."""

  def __init__(self, onnx_path: str, stats_json: str) -> None:
    self.C = 8
    self.T = 250

  def preprocess(self, x: np.ndarray) -> np.ndarray:
    """Return a dummy batch tensor."""
    return np.zeros((1, self.C, self.T), dtype=np.float32)

  def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return logits and probabilities favouring the left class."""
    logits = np.array([[5.0, 0.0]], dtype=np.float32)
    probs = np.array([[0.99, 0.01]], dtype=np.float32)
    return logits, probs


class DummyStabilizer:
  """Stabilizer that confirms every incoming intent immediately."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    pass

  def apply(self, intent: str, conf: float) -> Dict[str, float | str]:
    """Return the received intent without delay."""
    return {"intent": intent, "conf_ema": conf}


class DummyFilter:
  """Identity streaming filter used for testing."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    pass

  def apply(self, block: np.ndarray) -> np.ndarray:
    """Return the input block unchanged."""
    return np.asarray(block, dtype=np.float32)


@pytest.mark.timeout(10)
def test_ingest_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
  """Ensure ingest samples propagate to control WebSocket as intents."""
  dummy_filters = types.ModuleType("intentflow.online.dsp.filters")
  dummy_filters.StreamingFilter = DummyFilter
  monkeypatch.setitem(sys.modules, "intentflow.online.dsp.filters", dummy_filters)

  from intentflow.online.inference import worker as worker_module

  monkeypatch.setattr(worker_module, "OnnxModel", DummyOnnxModel, raising=False)
  monkeypatch.setattr(worker_module, "Stabilizer", DummyStabilizer, raising=False)
  monkeypatch.setenv("IF_ONNX", "dummy.onnx")
  monkeypatch.setenv("IF_STATS", "dummy.json")

  RAW_BUF.clear()
  META["fs"] = 0
  META["ch"] = 0

  with TestClient(app, raise_server_exceptions=False) as client:
    with client.websocket_connect("/control") as control_ws:
      with client.websocket_connect("/ingest") as ingest_ws:
        fs = 250
        ch = 8
        chunk = np.zeros((int(0.2 * fs), ch), dtype=np.float32)
        payload: Dict[str, Any] = {
          "fs": fs,
          "ch": ch,
          "samples": chunk.tolist(),
        }
        for _ in range(5):
          ingest_ws.send_text(json.dumps(payload))
          time.sleep(0.05)
        time.sleep(0.5)
      with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(control_ws.receive_text)
        message = future.result(timeout=5.0)
      data = json.loads(message)
      assert data["type"] == "intent"
      assert data["intent"] in {"left", "right"}
      assert 0.0 <= data["conf"] <= 1.0
